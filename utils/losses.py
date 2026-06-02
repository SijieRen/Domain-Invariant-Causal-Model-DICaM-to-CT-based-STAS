import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    kernel_2d = g @ g.t()
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(x, y, data_range=2.0, window_size=11, sigma=1.5):
    # x, y: [B, C, H, W]
    c = x.size(1)
    kernel = _gaussian_kernel(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=c)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = ssim_n / (ssim_d + 1e-12)
    return 1.0 - ssim_map.mean()


def image_gradient_loss(x, y):
    device = x.device
    dtype = x.dtype
    channels = x.size(1)

    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.expand(channels, 1, 3, 3).contiguous()
    kernel_y = kernel_y.expand(channels, 1, 3, 3).contiguous()

    grad_x_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
    grad_x_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
    grad_y_x = F.conv2d(y, kernel_x, padding=1, groups=channels)
    grad_y_y = F.conv2d(y, kernel_y, padding=1, groups=channels)

    loss_x = F.l1_loss(grad_x_x, grad_y_x)
    loss_y = F.l1_loss(grad_x_y, grad_y_y)
    return loss_x + loss_y


def laplacian_detail_loss(x, y):
    channels = x.size(1)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3).contiguous()
    lap_x = F.conv2d(x, kernel, padding=1, groups=channels)
    lap_y = F.conv2d(y, kernel, padding=1, groups=channels)
    return F.l1_loss(lap_x, lap_y)


def multiscale_recon_loss(x, y, scales=(1, 2, 4)):
    loss = x.new_tensor(0.0)
    count = 0
    for scale in scales:
        if scale == 1:
            x_scaled = x
            y_scaled = y
        else:
            x_scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            y_scaled = F.avg_pool2d(y, kernel_size=scale, stride=scale)
        loss = loss + F.l1_loss(x_scaled, y_scaled)
        count += 1
    return loss / max(1, count)


def component_consistency_loss(recons, component_recons):
    if not component_recons:
        return recons.new_tensor(0.0)
    recon_sum = torch.stack(component_recons, dim=0).sum(dim=0)
    return F.l1_loss(recon_sum, recons)


def component_overlap_loss(component_recons, eps=1e-6):
    if len(component_recons) < 2:
        return component_recons[0].new_tensor(0.0) if component_recons else torch.tensor(0.0)

    maps = []
    for recon in component_recons:
        flat = recon.abs().flatten(1)
        maps.append(flat / (flat.sum(dim=1, keepdim=True) + eps))

    loss = maps[0].new_tensor(0.0)
    pair_count = 0
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            loss = loss + (maps[i] * maps[j]).sum(dim=1).mean()
            pair_count += 1
    return loss / max(1, pair_count)


def residual_sparsity_loss(residual_recon):
    return residual_recon.abs().mean()
