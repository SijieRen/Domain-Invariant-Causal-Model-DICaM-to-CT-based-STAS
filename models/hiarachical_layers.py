import torch
from torch import nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on domain embedding."""
    def __init__(self, num_channels, emb_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.film = nn.Linear(emb_dim, num_channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, d_emb):
        x = self.bn(x)
        gamma_beta = self.film(d_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class FiLMDownsample(nn.Module):
    """Downsample with FiLM-conditioned BatchNorm."""
    def __init__(self, inplanes, planes, expansion, stride, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes * expansion,
                              kernel_size=1, stride=stride, bias=False)
        self.film = FiLMLayer(planes * expansion, emb_dim)

    def forward(self, x, d_emb):
        return self.film(self.conv(x), d_emb)
