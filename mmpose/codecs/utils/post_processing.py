# Copyright (c) OpenMMLab. All rights reserved.
import math
from itertools import product
from typing import Tuple, TypeVar

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_ArrayLike = TypeVar('_ArrayLike', np.ndarray, Tensor)


def _is_torch_tensor(value) -> bool:
    return isinstance(value, Tensor)


def _ensure_same_backend(x, y) -> None:
    if _is_torch_tensor(x) != _is_torch_tensor(y):
        raise TypeError('Expected both inputs to be the same type (numpy or torch).')


def _torch_gaussian_kernel1d(kernel: int, dtype: torch.dtype,
                             device: torch.device) -> Tensor:
    kernel_np = cv2.getGaussianKernel(kernel, 0).astype(np.float64)
    kernel_1d = torch.from_numpy(kernel_np.squeeze()).to(device=device)
    kernel_1d = kernel_1d.to(dtype=dtype)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d


def _torch_gaussian_kernel2d(kernel: int, dtype: torch.dtype,
                             device: torch.device) -> Tensor:
    kernel_1d = _torch_gaussian_kernel1d(kernel, dtype, device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def get_simcc_normalized(batch_pred_simcc: _ArrayLike,
                         sigma: float | None = None) -> _ArrayLike:
    """Normalize the predicted SimCC.

    Args:
        batch_pred_simcc (torch.Tensor): The predicted SimCC.
        sigma (float): The sigma of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized SimCC.
    """
    if _is_torch_tensor(batch_pred_simcc):
        B, K, _ = batch_pred_simcc.shape
        if sigma is not None:
            normalizer = batch_pred_simcc.new_tensor(sigma * math.sqrt(2 * math.pi))
            batch_pred_simcc = batch_pred_simcc / normalizer
        batch_pred_simcc = batch_pred_simcc.clamp(min=0)
        amax = batch_pred_simcc.amax(dim=-1, keepdim=True)
        mask = amax > 1
        norm = batch_pred_simcc / amax.clamp_min(torch.finfo(batch_pred_simcc.dtype).tiny)
        return torch.where(mask, norm, batch_pred_simcc)

    batch_pred_simcc = np.asarray(batch_pred_simcc)
    B, K, _ = batch_pred_simcc.shape
    if sigma is not None:
        batch_pred_simcc = batch_pred_simcc / (sigma * np.sqrt(np.pi * 2))
    batch_pred_simcc = np.clip(batch_pred_simcc, a_min=0, a_max=None)
    amax = batch_pred_simcc.max(axis=-1, keepdims=True)
    mask = amax > 1
    norm = batch_pred_simcc / np.clip(amax, a_min=np.finfo(batch_pred_simcc.dtype).tiny, a_max=None)
    return np.where(mask, norm, batch_pred_simcc)


def get_simcc_maximum(simcc_x: _ArrayLike,
                      simcc_y: _ArrayLike,
                      apply_softmax: bool = False
                      ) -> Tuple[_ArrayLike, _ArrayLike]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    _ensure_same_backend(simcc_x, simcc_y)

    if _is_torch_tensor(simcc_x):
        if simcc_x.ndim not in (2, 3):
            raise ValueError(f'Invalid shape {tuple(simcc_x.shape)}')
        if simcc_y.ndim != simcc_x.ndim:
            raise ValueError(f'{tuple(simcc_x.shape)} != {tuple(simcc_y.shape)}')

        original_shape = simcc_x.shape
        if simcc_x.ndim == 3:
            N, K, _ = original_shape
            flat_x = simcc_x.reshape(N * K, -1)
            flat_y = simcc_y.reshape(N * K, -1)
        else:
            K, _ = original_shape
            N = None
            flat_x = simcc_x.reshape(K, -1)
            flat_y = simcc_y.reshape(K, -1)

        if apply_softmax:
            flat_x = torch.softmax(flat_x, dim=1)
            flat_y = torch.softmax(flat_y, dim=1)

        max_val_x, x_idx = flat_x.max(dim=1)
        max_val_y, y_idx = flat_y.max(dim=1)

        locs = torch.stack((x_idx, y_idx), dim=-1).to(dtype=torch.float32)
        better_mask = max_val_x > max_val_y
        vals = torch.where(better_mask, max_val_y, max_val_x)
        invalid = vals <= 0
        if invalid.any():
            locs[invalid] = -1

        if N:
            locs = locs.reshape(N, K, 2)
            vals = vals.reshape(N, K)
        return locs.to(simcc_x.device), vals.to(simcc_x.device)

    simcc_x = np.asarray(simcc_x)
    simcc_y = np.asarray(simcc_y)
    if simcc_x.ndim not in (2, 3):
        raise ValueError(f'Invalid shape {simcc_x.shape}')
    if simcc_x.ndim != simcc_y.ndim:
        raise ValueError(f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, _ = simcc_x.shape
        flat_x = simcc_x.reshape(N * K, -1)
        flat_y = simcc_y.reshape(N * K, -1)
    else:
        N = None
        K, _ = simcc_x.shape
        flat_x = simcc_x.reshape(K, -1)
        flat_y = simcc_y.reshape(K, -1)

    if apply_softmax:
        flat_x = flat_x - np.max(flat_x, axis=1, keepdims=True)
        flat_y = flat_y - np.max(flat_y, axis=1, keepdims=True)
        ex, ey = np.exp(flat_x), np.exp(flat_y)
        flat_x = ex / np.sum(ex, axis=1, keepdims=True)
        flat_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(flat_x, axis=1)
    y_locs = np.argmax(flat_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(flat_x, axis=1)
    max_val_y = np.amax(flat_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals


def get_heatmap_3d_maximum(heatmaps: _ArrayLike
                           ) -> Tuple[_ArrayLike, _ArrayLike]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap dimension: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, D, H, W) or
            (B, K, D, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 3) or (B, K, 3)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    if _is_torch_tensor(heatmaps):
        if heatmaps.ndim not in (4, 5):
            raise ValueError(f'Invalid shape {tuple(heatmaps.shape)}')

        if heatmaps.ndim == 4:
            K, D, H, W = heatmaps.shape
            B = None
            flat = heatmaps.reshape(K, -1)
        else:
            B, K, D, H, W = heatmaps.shape
            flat = heatmaps.reshape(B * K, -1)

        vals, idx = flat.max(dim=1)
        hw = H * W
        z_locs = idx // hw
        hw_rem = idx % hw
        y_locs = hw_rem // W
        x_locs = hw_rem % W

        locs = torch.stack((x_locs, y_locs, z_locs), dim=-1).to(torch.float32)
        invalid = vals <= 0
        if invalid.any():
            locs[invalid] = -1

        if B:
            locs = locs.reshape(B, K, 3)
            vals = vals.reshape(B, K)
        return locs.to(heatmaps.device), vals.to(heatmaps.device)

    heatmaps = np.asarray(heatmaps)
    if heatmaps.ndim not in (4, 5):
        raise ValueError(f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 4:
        K, D, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, D, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    z_locs, y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(D, H, W))
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 3)
        vals = vals.reshape(B, K)

    return locs, vals


def get_heatmap_maximum(heatmaps: _ArrayLike
                        ) -> Tuple[_ArrayLike, _ArrayLike]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    if _is_torch_tensor(heatmaps):
        if heatmaps.ndim not in (3, 4):
            raise ValueError(f'Invalid shape {tuple(heatmaps.shape)}')

        if heatmaps.ndim == 3:
            K, H, W = heatmaps.shape
            B = None
            flat = heatmaps.reshape(K, -1)
        else:
            B, K, H, W = heatmaps.shape
            flat = heatmaps.reshape(B * K, -1)

        vals, idx = flat.max(dim=1)
        y_locs = idx // W
        x_locs = idx % W
        locs = torch.stack((x_locs, y_locs), dim=-1).to(torch.float32)
        invalid = vals <= 0
        if invalid.any():
            locs[invalid] = -1

        if B:
            locs = locs.reshape(B, K, 2)
            vals = vals.reshape(B, K)
        return locs.to(heatmaps.device), vals.to(heatmaps.device)

    heatmaps = np.asarray(heatmaps)
    if heatmaps.ndim not in (3, 4):
        raise ValueError(f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def gaussian_blur(heatmaps: _ArrayLike, kernel: int = 11) -> _ArrayLike:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1
    border = (kernel - 1) // 2

    if _is_torch_tensor(heatmaps):
        if heatmaps.ndim != 3:
            raise ValueError('Expected heatmaps with shape (K, H, W)')
        dtype = heatmaps.dtype
        device = heatmaps.device
        K, H, W = heatmaps.shape
        origin_max = heatmaps.amax(dim=(-2, -1), keepdim=True)
        tensor = heatmaps.unsqueeze(0)
        if border > 0:
            tensor = F.pad(tensor, (border, border, border, border))
        kernel_2d = _torch_gaussian_kernel2d(kernel, dtype, device).view(1, 1, kernel, kernel)
        weight = kernel_2d.repeat(K, 1, 1, 1)
        blurred = F.conv2d(tensor, weight, groups=K)
        blurred = blurred.squeeze(0)
        blurred_max = blurred.amax(dim=(-2, -1), keepdim=True)
        scale = torch.where(blurred_max > 0,
                            origin_max / blurred_max,
                            torch.zeros_like(origin_max))
        return blurred * scale

    heatmaps = np.asarray(heatmaps)
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        max_val = np.max(heatmaps[k])
        if max_val > 0:
            heatmaps[k] *= origin_max / max_val
        else:
            heatmaps[k] = 0
    return heatmaps


def gaussian_blur1d(simcc: _ArrayLike, kernel: int = 11) -> _ArrayLike:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1
    border = (kernel - 1) // 2

    if _is_torch_tensor(simcc):
        if simcc.ndim != 3:
            raise ValueError('Expected simcc with shape (N, K, L)')
        dtype = simcc.dtype
        device = simcc.device
        N, K, Wx = simcc.shape
        origin_max = simcc.amax(dim=-1, keepdim=True)
        tensor = simcc.reshape(-1, 1, Wx)
        if border > 0:
            tensor = F.pad(tensor, (border, border))
        kernel_1d = _torch_gaussian_kernel1d(kernel, dtype, device)
        weight = kernel_1d.view(1, 1, kernel)
        blurred = F.conv1d(tensor, weight)
        blurred = blurred.reshape(N, K, Wx)
        blurred_max = blurred.amax(dim=-1, keepdim=True)
        scale = torch.where(blurred_max > 0,
                            origin_max / blurred_max,
                            torch.zeros_like(origin_max))
        return blurred * scale

    simcc = np.asarray(simcc)
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        max_val = np.max(simcc[n, k])
        if max_val > 0:
            simcc[n, k] *= origin_max / max_val
        else:
            simcc[n, k] = 0
    return simcc


def batch_heatmap_nms(batch_heatmaps: Tensor, kernel_size: int = 5):
    """Apply NMS on a batch of heatmaps.

    Args:
        batch_heatmaps (Tensor): batch heatmaps in shape (B, K, H, W)
        kernel_size (int): The kernel size of the NMS which should be
            a odd integer. Defaults to 5

    Returns:
        Tensor: The batch heatmaps after NMS.
    """

    assert isinstance(kernel_size, int) and kernel_size % 2 == 1, \
        f'The kernel_size should be an odd integer, got {kernel_size}'

    padding = (kernel_size - 1) // 2

    maximum = F.max_pool2d(
        batch_heatmaps, kernel_size, stride=1, padding=padding)
    maximum_indicator = torch.eq(batch_heatmaps, maximum)
    batch_heatmaps = batch_heatmaps * maximum_indicator.float()

    return batch_heatmaps
