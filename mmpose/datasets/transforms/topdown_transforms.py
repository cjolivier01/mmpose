# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

# Optional torch/kornia support (added)
try:  # pragma: no cover
    import torch
    from kornia.geometry.transform import (
        warp_affine as kornia_warp_affine,
    )  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        # --- Torch/Kornia path support (reapplied) ---
        def _torch_warp_single(img_t, warp_np):
            if not _HAS_TORCH:
                raise RuntimeError(
                    "torch/kornia required for tensor warping (pip install kornia)."
                )

            # Normalize to BCHW
            layout = "CHW"
            if img_t.ndim == 2:  # HW
                img_bchw = img_t.unsqueeze(0).unsqueeze(0)
                layout = "HW"
            elif img_t.ndim == 3:  # CHW or HWC
                if img_t.shape[-1] <= 4:  # treat as HWC
                    img_bchw = img_t.permute(2, 0, 1).unsqueeze(0)
                    layout = "HWC"
                else:  # CHW
                    img_bchw = img_t.unsqueeze(0)
                    layout = "CHW"
            elif img_t.ndim == 4:  # BCHW or BHWC
                if img_t.shape[1] in (1, 3, 4):
                    img_bchw = img_t
                    layout = "BCHW"
                else:
                    img_bchw = img_t.permute(0, 3, 1, 2)
                    layout = "BHWC"
            else:
                raise ValueError(f"Unsupported tensor image ndim={img_t.ndim}")

            device = img_bchw.device
            orig_dtype = img_bchw.dtype
            needs_cast = not img_bchw.dtype.is_floating_point
            img_float = img_bchw.float() if needs_cast else img_bchw

            B = img_float.shape[0]
            dsize = (int(h), int(w))  # (H,W)
            M = torch.as_tensor(warp_np, dtype=img_float.dtype, device=device).view(
                1, 2, 3
            )
            if B > 1:
                M = M.expand(B, -1, -1)

            out = kornia_warp_affine(
                img_float,
                M,
                dsize=dsize,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            if needs_cast:
                try:
                    info = torch.iinfo(orig_dtype)
                    out = out.clamp(0, info.max).round().to(orig_dtype)
                except TypeError:
                    out = out.to(orig_dtype)

            # Restore layout
            if layout == "HW":
                out = out.squeeze(0).squeeze(0)
            elif layout == "HWC":
                out = out.squeeze(0).permute(1, 2, 0)
            elif layout == "CHW":
                out = out.squeeze(0)
            elif layout == "BHWC":
                out = out.permute(0, 2, 3, 1)
            return out

        imgs = results["img"]
        if isinstance(imgs, list):
            if _HAS_TORCH and any(hasattr(x, "ndim") and isinstance(x, type(getattr(torch, "Tensor", torch.tensor([])))) for x in imgs):  # type: ignore
                results["img"] = [_torch_warp_single(x, warp_mat) for x in imgs]
            else:
                results["img"] = [
                    cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                    for img in imgs
                ]
        else:
            if (
                _HAS_TORCH
                and hasattr(imgs, "ndim")
                and "torch" in imgs.__class__.__module__
            ):
                results["img"] = _torch_warp_single(imgs, warp_mat)
            else:
                results["img"] = cv2.warpAffine(
                    imgs, warp_mat, warp_size, flags=cv2.INTER_LINEAR
                )

        if results.get('keypoints', None) is not None:
            kps = results["keypoints"]
            existing = results.get("transformed_keypoints")
            if existing is None:
                if _HAS_TORCH and "torch" in kps.__class__.__module__:
                    transformed_keypoints = kps.clone()
                else:
                    transformed_keypoints = kps.copy()
            else:
                transformed_keypoints = (
                    existing.clone()
                    if (_HAS_TORCH and "torch" in existing.__class__.__module__)
                    else existing.copy()
                )

            if _HAS_TORCH and "torch" in transformed_keypoints.__class__.__module__:
                pts = transformed_keypoints[..., :2]
                shape = pts.shape
                pts2 = pts.reshape(-1, 2)
                ones = torch.ones(
                    (pts2.shape[0], 1), dtype=pts2.dtype, device=pts2.device
                )
                pts_h = torch.cat([pts2, ones], dim=-1)  # (N,3)
                M = torch.as_tensor(
                    warp_mat, dtype=pts2.dtype, device=pts2.device
                )  # (2,3)
                pts_warp = (M @ pts_h.t()).t().reshape(shape)
                transformed_keypoints[..., :2] = pts_warp
            else:
                transformed_keypoints[..., :2] = cv2.transform(kps[..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str
