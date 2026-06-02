"""
Backbone factory: build pretrained feature extractors adapted for 1-channel medical images.

Supported backbones:
  - resnet18, resnet34   (output: 512 × H/32 × W/32)
  - efficientnet_b0      (output: 1280 × H/32 × W/32)

All backbones:
  1. Replace first conv from 3→1 channel (copy mean of pretrained weights)
  2. Remove final avgpool + fc (return spatial feature map)
"""

import torch
from torch import nn
import torchvision.models as models


# --- output channel sizes per backbone ---
BACKBONE_OUT_CHANNELS = {
    'resnet18': 512,
    'resnet34': 512,
    'efficientnet_b0': 1280,
}


def _adapt_first_conv(module, attr='conv1'):
    """Replace 3-channel conv1 with 1-channel, copying mean of pretrained weights."""
    old_conv = getattr(module, attr)
    new_conv = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    setattr(module, attr, new_conv)


def build_backbone(name='resnet18', pretrained=True):
    """
    Returns:
        backbone: nn.Module that maps (B, 1, H, W) -> (B, C, H/32, W/32)
        out_channels: int, the channel dimension C
    """
    if name in ('resnet18', 'resnet34'):
        weights = 'IMAGENET1K_V1' if pretrained else None
        if name == 'resnet18':
            base = models.resnet18(weights=weights)
        else:
            base = models.resnet34(weights=weights)
        _adapt_first_conv(base, 'conv1')
        backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        return backbone, BACKBONE_OUT_CHANNELS[name]

    elif name == 'efficientnet_b0':
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        first_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        base.features[0][0] = new_conv
        backbone = base.features
        return backbone, BACKBONE_OUT_CHANNELS[name]

    else:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(BACKBONE_OUT_CHANNELS.keys())}")


def build_backbone_with_film(name='resnet18', pretrained=True, num_domains=4, emb_dim=32):
    """
    Build a domain-conditioned backbone using FiLM.

    For ResNet: loads pretrained weights into conv/BN layers, then wraps BN with FiLM.
    The FiLM layers are zero-initialized so initial behavior = standard pretrained BN.

    Returns:
        backbone: nn.Module with forward(x, domain_ids) -> feature_map
        out_channels: int
    """
    from .feature_learning import ResNetFiLM

    if name in ('resnet18', 'resnet34'):
        layers_cfg = [2, 2, 2, 2] if name == 'resnet18' else [3, 4, 6, 3]
        film_net = ResNetFiLM(layers_cfg, z_dim=32, num_domains=num_domains, emb_dim=emb_dim)
        out_channels = BACKBONE_OUT_CHANNELS[name]

        if pretrained:
            weights = 'IMAGENET1K_V1'
            if name == 'resnet18':
                pretrained_model = models.resnet18(weights=weights)
            else:
                pretrained_model = models.resnet34(weights=weights)
            _load_pretrained_into_film(film_net, pretrained_model)

        return film_net, out_channels

    elif name == 'efficientnet_b0':
        # EfficientNet with FiLM: use standard backbone + external FiLM adapter
        backbone, out_channels = build_backbone(name, pretrained)
        film_backbone = _EfficientNetFiLMWrapper(backbone, out_channels, num_domains, emb_dim)
        return film_backbone, out_channels

    else:
        raise ValueError(f"Unknown backbone: {name}")


def _load_pretrained_into_film(film_net, pretrained_model):
    """Transfer pretrained ResNet weights into ResNetFiLM, skipping FiLM-specific params."""
    # conv1: 3ch -> 1ch
    with torch.no_grad():
        film_net.conv1.weight.copy_(
            pretrained_model.conv1.weight.mean(dim=1, keepdim=True))

    # film1.bn ← pretrained bn1
    film_net.film1.bn.load_state_dict(pretrained_model.bn1.state_dict())

    # layer1-4: match conv weights and BN weights
    for layer_idx in range(1, 5):
        film_layer = getattr(film_net, f'layer{layer_idx}')
        pretrained_layer = getattr(pretrained_model, f'layer{layer_idx}')

        for block_idx, (film_block, pre_block) in enumerate(
                zip(film_layer.blocks, pretrained_layer)):
            film_block.conv1.load_state_dict(pre_block.conv1.state_dict())
            film_block.film1.bn.load_state_dict(pre_block.bn1.state_dict())
            film_block.conv2.load_state_dict(pre_block.conv2.state_dict())
            film_block.film2.bn.load_state_dict(pre_block.bn2.state_dict())

            if pre_block.downsample is not None and film_block.downsample is not None:
                film_block.downsample.conv.load_state_dict(
                    pre_block.downsample[0].state_dict())
                film_block.downsample.film.bn.load_state_dict(
                    pre_block.downsample[1].state_dict())


class _EfficientNetFiLMWrapper(nn.Module):
    """Wraps a frozen EfficientNet backbone with a lightweight FiLM adapter at the output."""
    def __init__(self, backbone, out_channels, num_domains, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.domain_emb = nn.Embedding(num_domains, emb_dim)
        self.film = nn.Linear(emb_dim, out_channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, domain_ids):
        x = self.backbone(x)
        d_emb = self.domain_emb(domain_ids)
        gamma_beta = self.film(d_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        x = x * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        return x
