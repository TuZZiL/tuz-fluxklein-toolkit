from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

ReplaceMode = Literal["zeros", "gaussian_noise", "channel_mean", "lowpass_reference"]
FadeMode = Literal["none", "center_out", "edges_out", "top_down", "left_right"]
ChannelMode = Literal["all", "low", "high"]


def gaussian_blur_per_channel(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(max(0, radius))
    if radius <= 0 or tensor.numel() == 0:
        return tensor

    kernel_size = radius * 2 + 1
    sigma = max(radius / 3.0, 1e-6)
    axis = torch.arange(kernel_size, device=tensor.device, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = int(tensor.shape[1])
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)

    pad_mode_x = "reflect" if tensor.shape[-1] > radius else "replicate"
    pad_mode_y = "reflect" if tensor.shape[-2] > radius else "replicate"

    blurred = F.pad(tensor, (radius, radius, 0, 0), mode=pad_mode_x)
    blurred = F.conv2d(blurred, kernel_x, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode=pad_mode_y)
    blurred = F.conv2d(blurred, kernel_y, groups=channels)
    return blurred


def _resolve_channel_range(
    channels: int,
    channel_mode: ChannelMode = "all",
    channel_start: Optional[int] = None,
    channel_end: Optional[int] = None,
) -> Tuple[int, int]:
    if channel_start is not None and channel_end is not None and int(channel_end) > int(channel_start):
        ch_start = max(0, min(int(channel_start), channels))
        ch_end = max(ch_start, min(int(channel_end), channels))
        return ch_start, ch_end

    if channel_mode == "low":
        return 0, channels // 2
    if channel_mode == "high":
        return channels // 2, channels
    return 0, channels


def create_spatial_mask(h: int, w: int, mode: FadeMode, strength: float, *, device: torch.device) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if mode == "none":
        return torch.ones((1, 1, h, w), dtype=torch.float32, device=device)

    y = torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32).unsqueeze(1).expand(h, w)
    x = torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w)

    if mode == "center_out":
        dist = torch.sqrt((y - 0.5) ** 2 + (x - 0.5) ** 2)
        dist = dist / max(float(dist.max().item()), 1e-8)
        mask = 1.0 - dist * strength
    elif mode == "edges_out":
        dist = torch.sqrt((y - 0.5) ** 2 + (x - 0.5) ** 2)
        dist = dist / max(float(dist.max().item()), 1e-8)
        mask = (1.0 - strength) + dist * strength
    elif mode == "top_down":
        mask = 1.0 - y * strength
    elif mode == "left_right":
        mask = 1.0 - x * strength
    else:
        mask = torch.ones((h, w), dtype=torch.float32, device=device)

    return mask.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def build_replacement(selected: torch.Tensor, mode: ReplaceMode) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros_like(selected)
    if mode == "gaussian_noise":
        scale = selected.std().clamp_min(1e-8)
        return torch.randn_like(selected) * scale
    if mode == "channel_mean":
        return selected.mean(dim=(-2, -1), keepdim=True).expand_as(selected)
    if mode == "lowpass_reference":
        radius = max(1, min(selected.shape[-2], selected.shape[-1]) // 16)
        return gaussian_blur_per_channel(selected, radius)
    raise ValueError(f"Unsupported replacement mode: {mode}")


def mix_reference_latent(
    ref: torch.Tensor,
    *,
    reference_keep: float,
    replace_mode: ReplaceMode,
    channel_start: int,
    channel_end: int,
    spatial_fade: FadeMode,
    spatial_fade_strength: float,
) -> torch.Tensor:
    if ref.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] reference latent, got shape {tuple(ref.shape)}")

    keep = float(max(0.0, min(1.0, reference_keep)))
    if keep == 1.0 and spatial_fade == "none":
        return ref

    working = ref.to(dtype=torch.float32)
    _, channels, h, w = working.shape
    ch_start, ch_end = _resolve_channel_range(
        channels,
        channel_mode="all",
        channel_start=channel_start,
        channel_end=channel_end,
    )
    if ch_start == ch_end:
        return working.to(dtype=ref.dtype)

    mask = create_spatial_mask(h, w, spatial_fade, spatial_fade_strength, device=working.device)
    keep_mask = mask * keep + (1.0 - mask)

    selected = working[:, ch_start:ch_end, :, :]
    replacement = build_replacement(selected, replace_mode)
    mixed = selected * keep_mask.expand(-1, ch_end - ch_start, -1, -1) + replacement * (
        1.0 - keep_mask.expand(-1, ch_end - ch_start, -1, -1)
    )

    result = working.clone()
    result[:, ch_start:ch_end, :, :] = mixed
    return result.to(dtype=ref.dtype)


def apply_mask_to_reference_latent(
    ref: torch.Tensor,
    mask,
    *,
    strength: float,
    invert_mask: bool = False,
    feather: int = 0,
    channel_mode: ChannelMode = "all",
    channel_start: Optional[int] = None,
    channel_end: Optional[int] = None,
) -> torch.Tensor:
    if ref is None:
        return None

    ref = ref.float().clone()
    _, num_ch, lat_h, lat_w = ref.shape

    if isinstance(mask, torch.Tensor):
        spatial_mask = mask.detach().to(device=ref.device, dtype=torch.float32)
    else:
        spatial_mask = torch.as_tensor(mask, dtype=torch.float32, device=ref.device)
    if spatial_mask.ndim == 3 and spatial_mask.shape[0] == 1:
        spatial_mask = spatial_mask[0]
    if spatial_mask.ndim == 3 and spatial_mask.shape[-1] == 1:
        spatial_mask = spatial_mask[..., 0]
    if spatial_mask.ndim != 2:
        raise ValueError(f"Expected MASK tensor with shape [B,H,W] or [H,W], got {tuple(spatial_mask.shape)}")
    if spatial_mask.shape != (lat_h, lat_w):
        spatial_mask = F.interpolate(
            spatial_mask.unsqueeze(0).unsqueeze(0), size=(lat_h, lat_w), mode="bilinear", align_corners=False
        )[0, 0]
    if invert_mask:
        spatial_mask = 1.0 - spatial_mask
    if feather > 0:
        kernel_size = feather * 2 + 1
        sigma = feather / 3.0
        axis = torch.arange(kernel_size, dtype=torch.float32, device=spatial_mask.device) - feather
        gauss_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        spatial_mask = F.conv2d(spatial_mask.unsqueeze(0).unsqueeze(0), kernel, padding=feather)[0, 0].clamp(0.0, 1.0)

    ch_start, ch_end = _resolve_channel_range(
        num_ch,
        channel_mode=channel_mode,
        channel_start=channel_start,
        channel_end=channel_end,
    )
    multiplier = 1.0 - float(max(0.0, min(1.0, strength))) * (1.0 - spatial_mask)
    multiplier = multiplier.unsqueeze(0).unsqueeze(0).expand(-1, ch_end - ch_start, -1, -1).to(ref.device)

    modified = ref.clone()
    modified[:, ch_start:ch_end, :, :] = ref[:, ch_start:ch_end, :, :] * multiplier
    return modified


def apply_masked_reference_mix(
    ref: torch.Tensor,
    mask,
    *,
    strength: float,
    reference_keep: float,
    replace_mode: ReplaceMode,
    invert_mask: bool = False,
    feather: int = 0,
    channel_mode: ChannelMode = "all",
    channel_start: Optional[int] = None,
    channel_end: Optional[int] = None,
) -> torch.Tensor:
    if ref is None:
        return None

    ref = ref.float().clone()
    _, num_ch, lat_h, lat_w = ref.shape

    if isinstance(mask, torch.Tensor):
        spatial_mask = mask.detach().to(device=ref.device, dtype=torch.float32)
    else:
        spatial_mask = torch.as_tensor(mask, dtype=torch.float32, device=ref.device)
    if spatial_mask.ndim == 3 and spatial_mask.shape[0] == 1:
        spatial_mask = spatial_mask[0]
    if spatial_mask.ndim == 3 and spatial_mask.shape[-1] == 1:
        spatial_mask = spatial_mask[..., 0]
    if spatial_mask.ndim != 2:
        raise ValueError(f"Expected MASK tensor with shape [B,H,W] or [H,W], got {tuple(spatial_mask.shape)}")
    if spatial_mask.shape != (lat_h, lat_w):
        spatial_mask = F.interpolate(
            spatial_mask.unsqueeze(0).unsqueeze(0), size=(lat_h, lat_w), mode="bilinear", align_corners=False
        )[0, 0]
    if invert_mask:
        spatial_mask = 1.0 - spatial_mask
    if feather > 0:
        kernel_size = feather * 2 + 1
        sigma = feather / 3.0
        axis = torch.arange(kernel_size, dtype=torch.float32, device=spatial_mask.device) - feather
        gauss_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        spatial_mask = F.conv2d(spatial_mask.unsqueeze(0).unsqueeze(0), kernel, padding=feather)[0, 0].clamp(0.0, 1.0)

    ch_start, ch_end = _resolve_channel_range(
        num_ch,
        channel_mode=channel_mode,
        channel_start=channel_start,
        channel_end=channel_end,
    )
    selected = ref[:, ch_start:ch_end, :, :]
    replacement = build_replacement(selected, replace_mode)

    keep = 1.0 - float(max(0.0, min(1.0, strength))) * (1.0 - float(max(0.0, min(1.0, reference_keep)))) * spatial_mask
    keep = keep.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0).expand(-1, ch_end - ch_start, -1, -1).to(ref.device)
    mixed = selected * keep + replacement * (1.0 - keep)

    modified = ref.clone()
    modified[:, ch_start:ch_end, :, :] = mixed
    return modified


def rebalance_reference_appearance(
    ref: torch.Tensor,
    *,
    appearance_scale: float,
    detail_scale: float,
    blur_radius: int,
    channel_start: int,
    channel_end: int,
) -> torch.Tensor:
    if ref.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] reference latent, got shape {tuple(ref.shape)}")

    if appearance_scale == 1.0 and detail_scale == 1.0:
        return ref

    working = ref.to(dtype=torch.float32)
    _, channels, _, _ = working.shape
    ch_start = max(0, min(int(channel_start), channels))
    ch_end = max(ch_start, min(int(channel_end), channels))
    if ch_start == ch_end:
        return working.to(dtype=ref.dtype)

    result = working.clone()
    selected = working[:, ch_start:ch_end, :, :]
    lowpass = gaussian_blur_per_channel(selected, blur_radius)
    detail = selected - lowpass
    result[:, ch_start:ch_end, :, :] = lowpass * float(appearance_scale) + detail * float(detail_scale)
    return result.to(dtype=ref.dtype)

