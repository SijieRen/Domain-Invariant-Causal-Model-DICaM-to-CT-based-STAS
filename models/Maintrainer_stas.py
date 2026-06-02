import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from typing import List, TypeVar
from torch import tensor as Tensor
from torch.nn import Parameter
from .ma_learning import GraphConvolution, gen_A, gen_adj
from .Basenet import Basenet
from .feature_learning import BasicBlock, ResNetbasic, ResNetFiLM
from .backbone_factory import build_backbone, build_backbone_with_film, BACKBONE_OUT_CHANNELS
from utils.opts import parse_opt
from utils.losses import image_gradient_loss, laplacian_detail_loss, multiscale_recon_loss, ssim_loss


class UnFlatten(nn.Module):
    def __init__(self, type='3d'):
        super(UnFlatten, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == '3d':
            return input.view(input.size(0), input.size(1), 1, 1, 1)
        else:
            return input.view(input.size(0), input.size(1), 1, 1)


Tensor = TypeVar('torch.tensor')
args = parse_opt()


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class UpDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpDecoderBlockWithSkip(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class InputSkipEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 32, 48, 64, 96, 96)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[0]),
        )
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[1]),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[2]),
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[3]),
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[4]),
        )
        self.down5 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[5]),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(channels[5]),
        )

    def forward(self, x):
        skip128 = self.stem(x)
        skip64 = self.down1(skip128)
        skip32 = self.down2(skip64)
        skip16 = self.down3(skip32)
        skip8 = self.down4(skip16)
        skip4 = self.down5(skip8)
        return {
            '4': skip4,
            '8': skip8,
            '16': skip16,
            '32': skip32,
            '64': skip64,
            '128': skip128,
        }


class FactorizedDecoder128(nn.Module):
    def __init__(self, chunk_dims, base_channels=96, out_channels=1, grid_size=4, component_skip_strength=0.18):
        super().__init__()
        self.base_channels = base_channels
        self.grid_size = grid_size
        self.component_skip_strength = component_skip_strength
        self.skip_channels = {'4': 96, '8': 96, '16': 64, '32': 48, '64': 32, '128': 32}
        self.skip_keys = ['4', '8', '16', '32', '64', '128']
        self.chunk_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(chunk_dim, base_channels * grid_size * grid_size),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for chunk_dim in chunk_dims
        ])
        self.component_skip_gaters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(chunk_dim, 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, len(self.skip_keys)),
            )
            for chunk_dim in chunk_dims
        ])
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(base_channels + self.skip_channels['4'], base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(base_channels),
        )
        self.up8 = UpDecoderBlockWithSkip(base_channels, self.skip_channels['8'], 128)
        self.up16 = UpDecoderBlockWithSkip(128, self.skip_channels['16'], 96)
        self.up32 = UpDecoderBlockWithSkip(96, self.skip_channels['32'], 64)
        self.up64 = UpDecoderBlockWithSkip(64, self.skip_channels['64'], 32)
        self.up128 = UpDecoderBlockWithSkip(32, self.skip_channels['128'], 32)
        self.head = nn.Sequential(
            ResidualConvBlock(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _decode_map(self, feature_map, spatial_skips):
        feature_map = self.skip_fuse(torch.cat([feature_map, spatial_skips['4']], dim=1))
        x = self.stem(feature_map)
        x = self.up8(x, spatial_skips['8'])
        x = self.up16(x, spatial_skips['16'])
        x = self.up32(x, spatial_skips['32'])
        x = self.up64(x, spatial_skips['64'])
        x = self.up128(x, spatial_skips['128'])
        return self.head(x)

    def _blur_skip(self, skip):
        return F.avg_pool2d(skip, kernel_size=3, stride=1, padding=1)

    def _make_component_skips(self, spatial_skips, skip_gate_logits):
        skip_gates = torch.sigmoid(skip_gate_logits)
        controlled_skips = {}
        for idx, key in enumerate(self.skip_keys):
            gate = skip_gates[:, idx].view(-1, 1, 1, 1)
            controlled_skips[key] = self.component_skip_strength * gate * self._blur_skip(spatial_skips[key])
        return controlled_skips

    def forward(self, z_chunks, spatial_skips, gates=None, return_components=False):
        component_maps = []
        component_skip_logits = []
        for projector, skip_gater, z in zip(self.chunk_projectors, self.component_skip_gaters, z_chunks):
            component_maps.append(
                projector(z).view(z.size(0), self.base_channels, self.grid_size, self.grid_size)
            )
            component_skip_logits.append(skip_gater(z))

        if gates is None:
            gates = component_maps[0].new_ones(len(component_maps))

        fused_map = 0
        for idx, feature_map in enumerate(component_maps):
            fused_map = fused_map + gates[idx] * feature_map

        recon = self._decode_map(fused_map, spatial_skips)

        if not return_components:
            return recon

        component_recons = [
            self._decode_map(
                gates[idx] * feature_map,
                self._make_component_skips(spatial_skips, component_skip_logits[idx]),
            )
            for idx, feature_map in enumerate(component_maps)
        ]
        return recon, component_recons


class InvariantEncoder(nn.Module):
    """Shared encoder for invariant factors q_psi1(v_dis, v_cli, v_res | x)."""
    def __init__(self, z_dim=32, backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.z_dim = z_dim
        self.backbone, out_ch = build_backbone(backbone_name, pretrained)
        self.feat_dim = out_ch * 4 * 4
        self.mlp_latent = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_latent(x)
        return x


class DomainEncoder(nn.Module):
    """Domain-conditioned encoder q_psi2^d(v_dom | x) with FiLM modulation."""
    def __init__(self, z_dim=32, num_domains=4, emb_dim=32,
                 backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.z_dim = z_dim
        self.backbone, out_ch = build_backbone_with_film(
            backbone_name, pretrained, num_domains, emb_dim)
        self.feat_dim = out_ch * 4 * 4
        self.mlp_latent = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x, domain_ids):
        """
        Args:
            x: (B, 1, H, W)
            domain_ids: (B,) integer tensor
        """
        x = self.backbone(x, domain_ids)
        x = x.view(x.size(0), -1)
        x = self.mlp_latent(x)
        return x


class DomainConditionalPrior(nn.Module):
    """Learnable domain-conditional prior p(v_dom | d) — each domain has its own Gaussian."""
    def __init__(self, num_domains, dom_dim):
        super().__init__()
        self.prior_mu = nn.Embedding(num_domains, dom_dim)
        self.prior_logvar = nn.Embedding(num_domains, dom_dim)
        nn.init.zeros_(self.prior_mu.weight)
        nn.init.zeros_(self.prior_logvar.weight)

    def forward(self, domain_ids):
        return self.prior_mu(domain_ids), self.prior_logvar(domain_ids)


class Model_STAS_2D(Basenet):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 gcndir: str,
                 args=None,
                 **kwargs) -> None:
        super(Model_STAS_2D, self).__init__()
        self.args = args

        self.latent_dim = latent_dim
        self.dis_dim = latent_dim // 2
        self.dom_dim = latent_dim // 4
        remain_dim = latent_dim - self.dis_dim - self.dom_dim
        self.cli_dim = remain_dim // 2
        self.res_dim = remain_dim - self.cli_dim
        self.inv_dim = self.dis_dim + self.cli_dim + self.res_dim
        self.chunk_dims = [self.dis_dim, self.dom_dim, self.cli_dim, self.res_dim]

        backbone_name = getattr(args, 'backbone', 'resnet18')
        use_pretrained = bool(getattr(args, 'pretrained', False))

        # Shared encoder for invariant factors (v_dis, v_cli, v_res)
        self.Resnetencoder = InvariantEncoder(
            backbone_name=backbone_name, pretrained=use_pretrained)
        # Domain-conditioned encoder with FiLM for v_dom
        self.Resnetencoder2 = DomainEncoder(
            num_domains=args.num_domains,
            backbone_name=backbone_name, pretrained=use_pretrained)

        self.dec_x = FactorizedDecoder128(
            self.chunk_dims,
            component_skip_strength=1.0,
        )
        self.visual_dec_x = FactorizedDecoder128(
            self.chunk_dims,
            component_skip_strength=self.args.component_skip_strength,
        )

        # Invariant latent projection heads
        self.fc_mu_inv = nn.Sequential(
            self.Fc_bn_ReLU(64, 128),
            nn.Linear(128, self.inv_dim),
        )
        self.fc_var_inv = nn.Sequential(
            self.Fc_bn_ReLU(64, 128),
            nn.Linear(128, self.inv_dim),
        )

        # Domain latent projection heads
        self.fc_mu_dom = nn.Sequential(
            self.Fc_bn_ReLU(64, 64),
            nn.Linear(64, self.dom_dim),
        )
        self.fc_var_dom = nn.Sequential(
            self.Fc_bn_ReLU(64, 64),
            nn.Linear(64, self.dom_dim),
        )

        # Domain-conditional prior p(v_dom | d)
        self.domain_prior = DomainConditionalPrior(self.args.num_domains, self.dom_dim)

        # GCN for clinical attribute relationships
        self.gc1 = GraphConvolution(14, 128)
        self.gc2 = GraphConvolution(128, 64)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(14, 0.4, str(gcndir))
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.enc_A1 = self._build_enc_A1()
        self.enc_A1_x = nn.Sequential(
            self.Fc_bn_ReLU(self.latent_dim * 2, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, self.latent_dim),
        )
        self.dec_y_by_vdis = self._build_dec_y_by_vdis()
        self.chunk_gates = nn.Parameter(torch.ones(4))
        self.input_skip_encoder = InputSkipEncoder(in_channels=in_channels)
        # p(A1 | v_cli, v_dis): A1 depends on both clinical and disease factors
        self.cli_head_a1 = nn.Sequential(
            self.Fc_bn_ReLU(self.cli_dim + self.dis_dim, 64),
            nn.Linear(64, 2),
        )

        self.get_y = nn.Sequential(
            self.Fc_bn_ReLU(32, 128),
            self.Fc_bn_ReLU(128, 64),
            nn.Linear(64, 2),
        )

    def _build_enc_A1(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(2, 32),
            self.Fc_bn_ReLU(32, 64),
            nn.Linear(64, self.latent_dim),
        )

    def _build_dec_y_by_vdis(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.dis_dim, 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, 32),
        )

    def encode_invariant(self, input: Tensor, A1: Tensor):
        """q_psi1(v_dis, v_cli, v_res | x, A1) — shared across all domains."""
        v = self.Resnetencoder(input)
        A1_enc = self.enc_A1(A1)
        v = self.enc_A1_x(torch.cat((v, A1_enc), dim=1))
        mu = self.fc_mu_inv(v)
        logvar = self.fc_var_inv(v)
        return mu, logvar

    def encode_domain(self, input: Tensor, domain_ids):
        """q_psi2^d(v_dom | x) — domain-conditioned encoder with FiLM modulation.
        Args:
            domain_ids: int (broadcast to batch) or (B,) integer tensor
        """
        if isinstance(domain_ids, int):
            domain_ids = torch.full((input.size(0),), domain_ids,
                                    dtype=torch.long, device=input.device)
        v = self.Resnetencoder2(input, domain_ids)
        mu = self.fc_mu_dom(v)
        logvar = self.fc_var_dom(v)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _split_invariant(self, z_inv: Tensor):
        """Split invariant latent into disease, clinical, residual components."""
        z_dis = z_inv[:, :self.dis_dim]
        z_cli = z_inv[:, self.dis_dim:self.dis_dim + self.cli_dim]
        z_res = z_inv[:, self.dis_dim + self.cli_dim:]
        return z_dis, z_cli, z_res

    def _latent_dropout(self, z_chunks, p_drop):
        if p_drop <= 0:
            return z_chunks
        out = []
        for z in z_chunks:
            keep = (torch.rand(z.size(0), 1, device=z.device) > p_drop).float()
            out.append(z * keep)
        return out

    def _decode_from_chunks(self, z_chunks, input_image, return_components=False):
        gates = torch.sigmoid(self.chunk_gates)
        spatial_skips = self.input_skip_encoder(input_image)
        rec = self.dec_x(z_chunks, spatial_skips=spatial_skips, gates=gates, return_components=False)
        if not return_components:
            return rec
        _, component_recons = self.visual_dec_x(
            z_chunks,
            spatial_skips=spatial_skips,
            gates=gates,
            return_components=True,
        )
        return rec, component_recons

    def forward(self, input: Tensor, m_id, h_id, A1: Tensor,
                gcn_target: Tensor, criterion_gcn, state: str, **kwargs) -> List[Tensor]:
        if 'train' in state:
            return self._forward_train(input, m_id, A1, gcn_target, criterion_gcn, **kwargs)
        elif 'intervene' in state:
            return self._forward_intervene(input, m_id, A1, **kwargs)
        elif 'val' in state:
            return self._forward_val(input, A1, gcn_target, criterion_gcn)

    def _forward_train(self, input, m_id, A1, gcn_target, criterion_gcn, **kwargs):
        # m_id is now a (B,) integer tensor — whole batch at once
        # Encode invariant factors (shared encoder)
        mu_inv, logvar_inv = self.encode_invariant(input, A1)
        # Encode domain factor (FiLM-conditioned encoder)
        mu_dom, logvar_dom = self.encode_domain(input, m_id)

        # Domain-conditional prior
        prior_mu_dom, prior_logvar_dom = self.domain_prior(m_id)

        noise_coeff = kwargs.get('latent_noise_coeff', 1.0)

        # Invariant reparameterization
        z_inv_sample = self.reparameterize(mu_inv, logvar_inv)
        z_inv_recon = (1.0 - noise_coeff) * mu_inv + noise_coeff * z_inv_sample
        z_inv_cls = mu_inv if int(self.args.cls_use_mu) else z_inv_sample
        z_inv_aux = mu_inv if int(self.args.aux_use_mu) else z_inv_sample

        # Domain reparameterization
        z_dom_sample = self.reparameterize(mu_dom, logvar_dom)
        z_dom_recon = (1.0 - noise_coeff) * mu_dom + noise_coeff * z_dom_sample

        # Split invariant for task heads
        z_dis_cls, z_cli_cls, _ = self._split_invariant(z_inv_cls)
        z_dis_recon, z_cli_recon, z_res_recon = self._split_invariant(z_inv_recon)

        # Latent dropout for classifiers
        z_dis_cls, z_cli_cls = self._latent_dropout(
            [z_dis_cls, z_cli_cls], self.args.classifier_latent_drop_prob
        )[:2]

        # p(y | v_dis): disease classification depends only on v_dis
        cls_latent = self.dec_y_by_vdis(z_dis_cls)
        cls = self.get_y(cls_latent)

        # p(A1 | v_cli, v_dis): clinical attributes depend on both factors
        a1_pred = self.cli_head_a1(torch.cat([z_cli_cls, z_dis_cls], dim=1))

        # p(A2 | v_dis, y): GCN predicts radiological features from (v_dis, cls_latent)
        inp = torch.eye(14, device=input.device)
        adj = gen_adj(self.A).detach()
        gcn = self.gc1(inp, adj)
        gcn = self.relu(gcn)
        gcn = self.gc2(gcn, adj)
        gcn = gcn.transpose(0, 1)
        gcn_x = torch.matmul(torch.cat((z_dis_cls, cls_latent), dim=1), gcn)

        # p(x | v): reconstruction from all factors
        rec, component_recons = self._decode_from_chunks(
            [z_dis_recon, z_dom_recon, z_cli_recon, z_res_recon],
            input,
            return_components=True,
        )

        return [
            rec, input,
            mu_inv, logvar_inv,
            mu_dom, logvar_dom,
            prior_mu_dom, prior_logvar_dom,
            a1_pred,
            gcn_x, cls,
            component_recons[0], component_recons[1],
            component_recons[2], component_recons[3],
        ]

    def _forward_intervene(self, input, m_id, A1, **kwargs):
        with torch.no_grad():
            mu_inv, _ = self.encode_invariant(input, A1)
            mu_dom, _ = self.encode_domain(input, m_id)
            z_dis, z_cli, z_res = self._split_invariant(mu_inv)
            z_dom = mu_dom

            intervention_chunk = kwargs.get('intervention_chunk', 'none')
            if intervention_chunk == 'dis':
                z_dis = torch.zeros_like(z_dis)
            elif intervention_chunk == 'dom':
                z_dom = torch.zeros_like(z_dom)
            elif intervention_chunk == 'cli':
                z_cli = torch.zeros_like(z_cli)
            elif intervention_chunk == 'res':
                z_res = torch.zeros_like(z_res)

            cls_latent = self.dec_y_by_vdis(z_dis)
            cls = self.get_y(cls_latent)
            return [self._decode_from_chunks([z_dis, z_dom, z_cli, z_res], input), cls]

    def _forward_val(self, input, A1, gcn_target, criterion_gcn):
        with torch.no_grad():
            mu_inv, logvar_inv = self.encode_invariant(input, A1)
            z_inv = mu_inv if int(self.args.cls_use_mu) else self.reparameterize(mu_inv, logvar_inv)
            z_dis, z_cli, z_res = self._split_invariant(z_inv)

            # Environment selection: pick domain with best reconstruction per sample
            B = input.size(0)
            best_z_dom = torch.zeros(B, self.dom_dim, device=input.device)
            best_rec_loss = torch.full((B,), float('inf'), device=input.device)
            for d in range(self.args.num_domains):
                mu_dom_d, _ = self.encode_domain(input, d)
                rec_d = self._decode_from_chunks([z_dis, mu_dom_d, z_cli, z_res], input)
                per_sample = F.smooth_l1_loss(
                    rec_d, input, reduction='none'
                ).view(B, -1).mean(dim=1)
                improved = per_sample < best_rec_loss
                best_rec_loss[improved] = per_sample[improved]
                best_z_dom[improved] = mu_dom_d[improved]
            z_dom = best_z_dom

            cls_latent = self.dec_y_by_vdis(z_dis)

            # Precompute spatial skips (independent of optimized variables)
            spatial_skips_val = {k: v.detach() for k, v in
                                self.input_skip_encoder(input).items()}

        # Eq.5: optimize (v_cli*, v_dis*) to maximize posterior
        cls_init = cls_latent.detach().clone()
        v_dis = z_dis.detach().clone().requires_grad_(True)
        v_cli = z_cli.detach().clone().requires_grad_(True)

        optimizer_cls = optim.Adam(
            params=[v_dis, v_cli], lr=self.args.lr2, weight_decay=self.args.wd2
        )

        para_a1_val = getattr(self.args, 'para_a1_val', self.args.para_cli)

        with torch.no_grad():
            inp = torch.eye(14, device=input.device)
            adj = gen_adj(self.A).detach()
            gcn = self.gc1(inp, adj)
            gcn = self.relu(gcn)
            gcn = self.gc2(gcn, adj)
            gcn = gcn.transpose(0, 1)

        for i in range(self.args.val_ep):
            optimizer_cls.zero_grad()

            # Recompute cls from optimized v_dis each step
            cls = self.dec_y_by_vdis(v_dis)

            # log p(A1 | v_cli, v_dis): clinical attribute constraint
            a1_pred = self.cli_head_a1(torch.cat([v_cli, v_dis], dim=1))
            loss_a1 = F.smooth_l1_loss(a1_pred, A1)

            # log Σ_y p(A2 | v_dis, y) p(y | v_dis): marginalized A2 via GCN
            gcn_x = torch.matmul(torch.cat((v_dis, cls), dim=1), gcn)
            loss_gcn = criterion_gcn(gcn_x, gcn_target)

            # log p(x | v): reconstruction constraint
            gates_val = torch.sigmoid(self.chunk_gates).detach()
            z_chunks_val = [v_dis, z_dom.detach(), v_cli, z_res.detach()]
            rec_opt = self.dec_x(
                z_chunks_val,
                spatial_skips=spatial_skips_val,
                gates=gates_val,
                return_components=False,
            )
            loss_rec = F.smooth_l1_loss(rec_opt, input)

            loss = (loss_gcn
                    + para_a1_val * loss_a1
                    + self.args.para_recon_val * loss_rec)

            # Only compute gradients for optimized latent variables,
            # not model parameters. create_graph=False avoids graph retention.
            grads = torch.autograd.grad(loss, [v_dis, v_cli], create_graph=False)
            v_dis.grad = grads[0]
            v_cli.grad = grads[1]
            optimizer_cls.step()

        with torch.no_grad():
            cls = self.dec_y_by_vdis(v_dis)
        cls = self.get_y(cls)

        rec, component_recons = self._decode_from_chunks(
            [v_dis.detach(), z_dom, v_cli.detach(), z_res], input, return_components=True
        )
        return [
            input, rec,
            component_recons[0], component_recons[1],
            component_recons[2], component_recons[3],
            gcn_x,
            self.get_y(cls_init),
            cls,
        ]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu_inv = args[2]
        logvar_inv = args[3]
        mu_dom = args[4]
        logvar_dom = args[5]
        prior_mu_dom = args[6]
        prior_logvar_dom = args[7]

        beta = kwargs.get('beta', 1.0)
        loss_l1 = torch.nn.SmoothL1Loss()

        if self.args.rec_loss_check == "bce":
            recons_loss = F.binary_cross_entropy(recons, input, reduction='mean')
        elif self.args.rec_loss_check == "l1":
            recons_loss = loss_l1(recons, input)
        elif self.args.rec_loss_check == "l2":
            recons_loss = F.mse_loss(recons, input)
        elif self.args.rec_loss_check == "l1_ssim":
            recons_loss = (
                loss_l1(recons, input)
                + self.args.ssim_weight * ssim_loss(recons, input, data_range=2.0)
                + self.args.grad_weight * image_gradient_loss(recons, input)
                + self.args.ms_recon_weight * multiscale_recon_loss(recons, input)
                + self.args.hf_weight * laplacian_detail_loss(recons, input)
            )

        # KL for invariant factors: KL( q(v_inv|x) || N(0,I) )
        kl_inv = torch.mean(
            -0.5 * torch.sum(1 + logvar_inv - mu_inv.pow(2) - logvar_inv.exp(), dim=1),
            dim=0,
        )

        # KL for domain factor: KL( q(v_dom|x) || p(v_dom|d) )
        kl_dom = torch.mean(
            -0.5 * torch.sum(
                1 + logvar_dom - prior_logvar_dom
                - (logvar_dom.exp() + (mu_dom - prior_mu_dom).pow(2)) / prior_logvar_dom.exp(),
                dim=1,
            ),
            dim=0,
        )

        para_kld_dom = getattr(self.args, 'para_kld_dom', self.args.para_kld)

        loss = (self.args.para_recon * recons_loss
                + self.args.para_kld * beta * kl_inv
                + para_kld_dom * beta * kl_dom)

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'KLD_inv': -kl_inv,
            'KLD_dom': -kl_dom,
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU())
        return layer
