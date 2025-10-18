# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import cv2
import numpy as np

try:  # pragma: no cover
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _is_torch_tensor(obj) -> bool:
    return _HAS_TORCH and isinstance(obj, torch.Tensor)


def _infer_dtype_device(*values):
    if not _HAS_TORCH:
        return None, None
    for value in values:
        if _is_torch_tensor(value):
            return value.dtype, value.device
    return None, None


def _ensure_torch(value, dtype, device):
    if not _HAS_TORCH:
        raise RuntimeError("Torch support is not available")
    if _is_torch_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(dtype=dtype, device=device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def bbox_xyxy2xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    if _is_torch_tensor(bbox_xyxy):
        bbox_xywh = bbox_xyxy.clone()
        bbox_xywh[..., 2] = bbox_xyxy[..., 2] - bbox_xyxy[..., 0]
        bbox_xywh[..., 3] = bbox_xyxy[..., 3] - bbox_xyxy[..., 1]
        return bbox_xywh

    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    if _is_torch_tensor(bbox_xywh):
        bbox_xyxy = bbox_xywh.clone()
        bbox_xyxy[..., 2] = bbox_xyxy[..., 2] + bbox_xyxy[..., 0]
        bbox_xyxy[..., 3] = bbox_xyxy[..., 3] + bbox_xyxy[..., 1]
        return bbox_xyxy

    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    if _is_torch_tensor(bbox):
        dim = bbox.dim()
        bbox_in = bbox.unsqueeze(0) if dim == 1 else bbox
        half = bbox_in.new_tensor(0.5)
        padding_t = bbox_in.new_tensor(padding)
        scale = (bbox_in[..., 2:] - bbox_in[..., :2]) * padding_t
        center = (bbox_in[..., 2:] + bbox_in[..., :2]) * half
        if dim == 1:
            return center[0], scale[0]
        return center, scale

    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_xywh2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (x, y, h, w)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """

    if _is_torch_tensor(bbox):
        dim = bbox.dim()
        bbox_in = bbox.unsqueeze(0) if dim == 1 else bbox
        half = bbox_in.new_tensor(0.5)
        padding_t = bbox_in.new_tensor(padding)
        x = bbox_in[..., 0:1]
        y = bbox_in[..., 1:2]
        w = bbox_in[..., 2:3]
        h = bbox_in[..., 3:4]
        center = torch.cat([x + w * half, y + h * half], dim=-1)
        scale = torch.cat([w, h], dim=-1) * padding_t
        if dim == 1:
            return center[0], scale[0]
        return center, scale

    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x, y, w, h = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x + w * 0.5, y + h * 0.5])
    scale = np.hstack([w, h]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_cs2xyxy(center: np.ndarray,
                 scale: np.ndarray,
                 padding: float = 1.) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x1,y1,x2,y2).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x1, y1, x2, y2) in shape (4, ) or (n, 4)
    """

    if _is_torch_tensor(center) or _is_torch_tensor(scale):
        dtype, device = _infer_dtype_device(center, scale)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        center_t = _ensure_torch(center, dtype, device)
        scale_t = _ensure_torch(scale, dtype, device)
        dim = center_t.dim()
        if dim == 1:
            center_t = center_t.unsqueeze(0)
            scale_t = scale_t.unsqueeze(0)
        padding_t = center_t.new_tensor(padding)
        half = center_t.new_tensor(0.5)
        wh = scale_t / padding_t
        xy = center_t - half * wh
        bbox_t = torch.cat((xy, xy + wh), dim=-1)
        if dim == 1:
            bbox_t = bbox_t[0]
        return bbox_t

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, xy + wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_cs2xywh(center: np.ndarray,
                 scale: np.ndarray,
                 padding: float = 1.) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x,y,w,h).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x, y, w, h) in shape (4, ) or (n, 4)
    """

    if _is_torch_tensor(center) or _is_torch_tensor(scale):
        dtype, device = _infer_dtype_device(center, scale)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        center_t = _ensure_torch(center, dtype, device)
        scale_t = _ensure_torch(scale, dtype, device)
        dim = center_t.dim()
        if dim == 1:
            center_t = center_t.unsqueeze(0)
            scale_t = scale_t.unsqueeze(0)
        padding_t = center_t.new_tensor(padding)
        half = center_t.new_tensor(0.5)
        wh = scale_t / padding_t
        xy = center_t - half * wh
        bbox_t = torch.cat((xy, wh), dim=-1)
        if dim == 1:
            bbox_t = bbox_t[0]
        return bbox_t

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_xyxy2corner(bbox: np.ndarray):
    """Convert bounding boxes from xyxy format to corner format.

    Given a numpy array containing bounding boxes in the format
    (xmin, ymin, xmax, ymax), this function converts the bounding
    boxes to the corner format, where each box is represented by four
    corner points (top-left, top-right, bottom-right, bottom-left).

    Args:
        bbox (numpy.ndarray): Input array of shape (N, 4) representing
            N bounding boxes.

    Returns:
        numpy.ndarray: An array of shape (N, 4, 2) containing the corner
            points of the bounding boxes.

    Example:
        bbox = np.array([[0, 0, 100, 50], [10, 20, 200, 150]])
        corners = bbox_xyxy2corner(bbox)
    """
    if _is_torch_tensor(bbox):
        dim = bbox.dim()
        bbox_in = bbox.unsqueeze(0) if dim == 1 else bbox
        x1 = bbox_in[..., 0]
        y1 = bbox_in[..., 1]
        x2 = bbox_in[..., 2]
        y2 = bbox_in[..., 3]
        tl = torch.stack([x1, y1], dim=-1)
        bl = torch.stack([x1, y2], dim=-1)
        tr = torch.stack([x2, y1], dim=-1)
        br = torch.stack([x2, y2], dim=-1)
        corners = torch.stack([tl, bl, tr, br], dim=-2)
        if dim == 1:
            corners = corners[0]
        return corners

    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None]

    bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
    bbox[:, 1:3, 0] = bbox[:, 0:2, 0]

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_corner2xyxy(bbox: np.ndarray):
    """Convert bounding boxes from corner format to xyxy format.

    Given a numpy array containing bounding boxes in the corner
    format (four corner points for each box), this function converts
    the bounding boxes to the (xmin, ymin, xmax, ymax) format.

    Args:
        bbox (numpy.ndarray): Input array of shape (N, 4, 2) representing
            N bounding boxes.

    Returns:
        numpy.ndarray: An array of shape (N, 4) containing the bounding
            boxes in xyxy format.

    Example:
        corners = np.array([[[0, 0], [100, 0], [100, 50], [0, 50]],
            [[10, 20], [200, 20], [200, 150], [10, 150]]])
        bbox = bbox_corner2xyxy(corners)
    """
    if _is_torch_tensor(bbox):
        if bbox.shape[-1] == 8:
            bbox = bbox.view(*bbox.shape[:-1], 4, 2)

        dim = bbox.dim()
        bbox_in = bbox.unsqueeze(0) if dim == 2 else bbox
        mins = bbox_in.amin(dim=-2)
        maxs = bbox_in.amax(dim=-2)
        xyxy = torch.cat((mins, maxs), dim=-1)
        if dim == 2:
            xyxy = xyxy[0]
        return xyxy

    if bbox.shape[-1] == 8:
        bbox = bbox.reshape(*bbox.shape[:-1], 4, 2)

    dim = bbox.ndim
    if dim == 2:
        bbox = bbox[None]

    bbox = np.concatenate((bbox.min(axis=1), bbox.max(axis=1)), axis=1)

    if dim == 2:
        bbox = bbox[0]

    return bbox


def bbox_clip_border(bbox: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip bounding box coordinates to fit within a specified shape.

    Args:
        bbox (np.ndarray): Bounding box coordinates of shape (..., 4)
            or (..., 2).
        shape (Tuple[int, int]): Shape of the image to which bounding
            boxes are being clipped in the format of (w, h)

    Returns:
        np.ndarray: Clipped bounding box coordinates.

    Example:
        >>> bbox = np.array([[10, 20, 30, 40], [40, 50, 80, 90]])
        >>> shape = (50, 50)  # Example image shape
        >>> clipped_bbox = bbox_clip_border(bbox, shape)
    """
    if _is_torch_tensor(bbox):
        bbox_clipped = bbox.clone()
        width = bbox_clipped.new_tensor(shape[0])
        height = bbox_clipped.new_tensor(shape[1])
        zero = bbox_clipped.new_tensor(0)
        if bbox_clipped.shape[-1] == 2:
            bbox_clipped[..., 0] = bbox_clipped[..., 0].clamp(zero, width)
            bbox_clipped[..., 1] = bbox_clipped[..., 1].clamp(zero, height)
        else:
            bbox_clipped[..., ::2] = bbox_clipped[..., ::2].clamp(zero, width)
            bbox_clipped[..., 1::2] = bbox_clipped[..., 1::2].clamp(zero, height)
        return bbox_clipped

    width, height = shape[:2]

    if bbox.shape[-1] == 2:
        bbox[..., 0] = np.clip(bbox[..., 0], a_min=0, a_max=width)
        bbox[..., 1] = np.clip(bbox[..., 1], a_min=0, a_max=height)
    else:
        bbox[..., ::2] = np.clip(bbox[..., ::2], a_min=0, a_max=width)
        bbox[..., 1::2] = np.clip(bbox[..., 1::2], a_min=0, a_max=height)

    return bbox


def flip_bbox(bbox: np.ndarray,
              image_size: Tuple[int, int],
              bbox_format: str = 'xywh',
              direction: str = 'horizontal') -> np.ndarray:
    """Flip the bbox in the given direction.

    Args:
        bbox (np.ndarray): The bounding boxes. The shape should be (..., 4)
            if ``bbox_format`` is ``'xyxy'`` or ``'xywh'``, and (..., 2) if
            ``bbox_format`` is ``'center'``
        image_size (tuple): The image shape in [w, h]
        bbox_format (str): The bbox format. Options are ``'xywh'``, ``'xyxy'``
            and ``'center'``.
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        np.ndarray: The flipped bounding boxes.
    """
    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    format_options = {'xywh', 'xyxy', 'center'}
    assert bbox_format in format_options, (
        f'Invalid bbox format "{bbox_format}". '
        f'Options are {format_options}')

    if _is_torch_tensor(bbox):
        bbox_flipped = bbox.clone()
        w = bbox.new_tensor(image_size[0])
        h = bbox.new_tensor(image_size[1])
        one = bbox.new_tensor(1)

        if direction == "horizontal":
            if bbox_format in ("xywh", "center"):
                bbox_flipped[..., 0] = w - bbox[..., 0] - one
            elif bbox_format == "xyxy":
                x1 = bbox[..., 0]
                x2 = bbox[..., 2]
                bbox_flipped[..., 0] = w - x2 - one
                bbox_flipped[..., 2] = w - x1 - one
        elif direction == "vertical":
            if bbox_format in ("xywh", "center"):
                bbox_flipped[..., 1] = h - bbox[..., 1] - one
            elif bbox_format == "xyxy":
                y1 = bbox[..., 1]
                y2 = bbox[..., 3]
                bbox_flipped[..., 1] = h - y2 - one
                bbox_flipped[..., 3] = h - y1 - one
        elif direction == "diagonal":
            if bbox_format in ("xywh", "center"):
                bbox_flipped[..., 0] = w - bbox[..., 0] - one
                bbox_flipped[..., 1] = h - bbox[..., 1] - one
            elif bbox_format == "xyxy":
                base = torch.stack([w, h, w, h])
                bbox_flipped = base - bbox - one
                bbox_flipped = torch.cat(
                    (bbox_flipped[..., 2:], bbox_flipped[..., :2]), dim=-1
                )
        return bbox_flipped

    bbox_flipped = bbox.copy()
    w, h = image_size

    # TODO: consider using "integer corner" coordinate system
    if direction == 'horizontal':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., 0] = w - bbox[..., 0] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[..., ::2] = w - bbox[..., -2::-2] - 1
    elif direction == 'vertical':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., 1] = h - bbox[..., 1] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[..., 1::2] = h - bbox[..., ::-2] - 1
    elif direction == 'diagonal':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., :2] = [w, h] - bbox[..., :2] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[...] = [w, h, w, h] - bbox - 1
            bbox_flipped = np.concatenate(
                (bbox_flipped[..., 2:], bbox_flipped[..., :2]), axis=-1)

    return bbox_flipped


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(output_size) == 2

    if _is_torch_tensor(center) or _is_torch_tensor(scale):
        dtype, device = _infer_dtype_device(center, scale)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        center_t = _ensure_torch(center, dtype, device)
        scale_t = _ensure_torch(scale, dtype, device)
        assert center_t.numel() == 2
        assert scale_t.numel() == 2
        input_size = center_t * 2
        rot_rad = torch.deg2rad(center_t.new_tensor(rot))
        warp_mat = torch.zeros((2, 3), dtype=dtype, device=device)
        scale_x = center_t.new_tensor(output_size[0] - 1) / scale_t[0]
        scale_y = center_t.new_tensor(output_size[1] - 1) / scale_t[1]
        cos = torch.cos(rot_rad)
        sin = torch.sin(rot_rad)
        half = center_t.new_tensor(0.5)
        warp_mat[0, 0] = cos * scale_x
        warp_mat[0, 1] = -sin * scale_x
        warp_mat[0, 2] = scale_x * (
            -half * input_size[0] * cos + half * input_size[1] * sin + half * scale_t[0]
        )
        warp_mat[1, 0] = sin * scale_y
        warp_mat[1, 1] = cos * scale_y
        warp_mat[1, 2] = scale_y * (
            -half * input_size[0] * sin - half * input_size[1] * cos + half * scale_t[1]
        )
        return warp_mat

    assert len(center) == 2
    assert len(scale) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                                0.5 * input_size[1] * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                                0.5 * input_size[1] * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0., 0.),
    inv: bool = False,
    fix_aspect_ratio: bool = True,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
            Defaults to True.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    if _is_torch_tensor(center) or _is_torch_tensor(scale):
        dtype, device = _infer_dtype_device(center, scale)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        center_t = _ensure_torch(center, dtype, device)
        scale_t = _ensure_torch(scale, dtype, device)
        shift_t = _ensure_torch(shift, dtype, device)
        if shift_t.dim() == 0:
            shift_t = shift_t.repeat(2)
        src_w, src_h = scale_t[:2]
        dst_w = center_t.new_tensor(output_size[0])
        dst_h = center_t.new_tensor(output_size[1])

        rot_rad = torch.deg2rad(center_t.new_tensor(rot))
        src_dir = _rotate_point(
            torch.stack([src_w * center_t.new_tensor(-0.5), center_t.new_tensor(0.0)]),
            rot_rad,
        )
        dst_dir = torch.stack(
            [dst_w * center_t.new_tensor(-0.5), center_t.new_tensor(0.0)]
        )

        src = torch.zeros((3, 2), dtype=dtype, device=device)
        src[0, :] = center_t + scale_t * shift_t
        src[1, :] = center_t + src_dir + scale_t * shift_t

        dst = torch.zeros((3, 2), dtype=dtype, device=device)
        dst_center = torch.stack(
            [
                dst_w * center_t.new_tensor(0.5),
                dst_h * center_t.new_tensor(0.5),
            ]
        )
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_dir

        if fix_aspect_ratio:
            src[2, :] = _get_3rd_point(src[0, :], src[1, :])
            dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
        else:
            src_dir_2 = _rotate_point(
                torch.stack(
                    [center_t.new_tensor(0.0), src_h * center_t.new_tensor(-0.5)]
                ),
                rot_rad,
            )
            dst_dir_2 = torch.stack(
                [
                    center_t.new_tensor(0.0),
                    dst_h * center_t.new_tensor(-0.5),
                ]
            )
            src[2, :] = center_t + src_dir_2 + scale_t * shift_t
            dst[2, :] = dst_center + dst_dir_2

        if inv:
            src_pts, dst_pts = dst, src
        else:
            src_pts, dst_pts = src, dst

        ones = torch.ones((3, 1), dtype=dtype, device=device)
        src_aug = torch.cat([src_pts, ones], dim=1)
        warp = torch.linalg.solve(src_aug, dst_pts).transpose(0, 1)
        return warp

    shift = np.array(shift)
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    if fix_aspect_ratio:
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    else:
        src_dir_2 = _rotate_point(np.array([0., src_h * -0.5]), rot_rad)
        dst_dir_2 = np.array([0., dst_h * -0.5])
        src[2, :] = center + src_dir_2 + scale * shift
        dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def get_pers_warp_matrix(center: np.ndarray, translate: np.ndarray,
                         scale: float, rot: float,
                         shear: np.ndarray) -> np.ndarray:
    """Compute a perspective warp matrix based on specified transformations.

    Args:
        center (np.ndarray): Center of the transformation.
        translate (np.ndarray): Translation vector.
        scale (float): Scaling factor.
        rot (float): Rotation angle in degrees.
        shear (np.ndarray): Shearing angles in degrees along x and y axes.

    Returns:
        np.ndarray: Perspective warp matrix.

    Example:
        >>> center = np.array([0, 0])
        >>> translate = np.array([10, 20])
        >>> scale = 1.2
        >>> rot = 30.0
        >>> shear = np.array([15.0, 0.0])
        >>> warp_matrix = get_pers_warp_matrix(center, translate,
                                               scale, rot, shear)
    """
    if any(_is_torch_tensor(x) for x in (center, translate, shear)):
        dtype, device = _infer_dtype_device(center, translate, shear)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        center_t = _ensure_torch(center, dtype, device)
        translate_t = _ensure_torch(translate, dtype, device)
        shear_t = _ensure_torch(shear, dtype, device)
        scale_t = center_t.new_tensor(scale)
        rot_t = center_t.new_tensor(rot)

        translate_mat = torch.eye(3, dtype=dtype, device=device)
        translate_mat[0, 2] = translate_t[0] + center_t[0]
        translate_mat[1, 2] = translate_t[1] + center_t[1]

        shear_x = torch.deg2rad(shear_t[0])
        shear_y = torch.deg2rad(shear_t[1])
        shear_mat = torch.eye(3, dtype=dtype, device=device)
        shear_mat[0, 1] = torch.tan(shear_x)
        shear_mat[1, 0] = torch.tan(shear_y)

        rotate_angle = torch.deg2rad(rot_t)
        rotate_mat = torch.eye(3, dtype=dtype, device=device)
        rotate_mat[0, 0] = torch.cos(rotate_angle)
        rotate_mat[0, 1] = -torch.sin(rotate_angle)
        rotate_mat[1, 0] = torch.sin(rotate_angle)
        rotate_mat[1, 1] = torch.cos(rotate_angle)

        scale_mat = torch.eye(3, dtype=dtype, device=device)
        scale_mat[0, 0] = scale_t
        scale_mat[1, 1] = scale_t

        recover_center_mat = torch.eye(3, dtype=dtype, device=device)
        recover_center_mat[0, 2] = -center_t[0]
        recover_center_mat[1, 2] = -center_t[1]

        warp_matrix = (
            translate_mat @ shear_mat @ rotate_mat @ scale_mat @ recover_center_mat
        )
        return warp_matrix

    translate_mat = np.array([[1, 0, translate[0] + center[0]],
                              [0, 1, translate[1] + center[1]], [0, 0, 1]],
                             dtype=np.float32)

    shear_x = math.radians(shear[0])
    shear_y = math.radians(shear[1])
    shear_mat = np.array([[1, np.tan(shear_x), 0], [np.tan(shear_y), 1, 0],
                          [0, 0, 1]],
                         dtype=np.float32)

    rotate_angle = math.radians(rot)
    rotate_mat = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle), 0],
                           [np.sin(rotate_angle),
                            np.cos(rotate_angle), 0], [0, 0, 1]],
                          dtype=np.float32)

    scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                         dtype=np.float32)

    recover_center_mat = np.array([[1, 0, -center[0]], [0, 1, -center[1]],
                                   [0, 0, 1]],
                                  dtype=np.float32)

    warp_matrix = np.dot(
        np.dot(
            np.dot(np.dot(translate_mat, shear_mat), rotate_mat), scale_mat),
        recover_center_mat)

    return warp_matrix


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    if _is_torch_tensor(pt) or (_HAS_TORCH and _is_torch_tensor(angle_rad)):
        dtype, device = _infer_dtype_device(pt, angle_rad)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        pt_t = _ensure_torch(pt, dtype, device)
        angle_t = (
            angle_rad
            if _is_torch_tensor(angle_rad)
            else torch.as_tensor(angle_rad, dtype=dtype, device=device)
        )
        sn = torch.sin(angle_t)
        cs = torch.cos(angle_t)
        rot_mat = torch.stack([torch.stack([cs, -sn]), torch.stack([sn, cs])])
        return rot_mat @ pt_t

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        dtype, device = _infer_dtype_device(a, b)
        dtype = dtype or torch.float32
        device = device or torch.device("cpu")
        a_t = _ensure_torch(a, dtype, device)
        b_t = _ensure_torch(b, dtype, device)
        direction = a_t - b_t
        c = b_t + torch.stack([-direction[1], direction[0]])
        return c

    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c
