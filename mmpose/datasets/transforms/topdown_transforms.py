# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

# Optional torch/kornia support (added)
try:  # pragma: no cover
    import torch
    import torch.nn.functional as F

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

        if _HAS_TORCH and isinstance(bbox_scale, torch.Tensor):
            aspect = bbox_scale.new_tensor(aspect_ratio)
            w = bbox_scale[..., 0]
            h = bbox_scale[..., 1]
            cond = w > h * aspect
            new_w = torch.where(cond, w, h * aspect)
            new_h = torch.where(cond, w / aspect, h)
            return torch.stack([new_w, new_h], dim=-1)

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

        def _is_torch_tensor(obj):
            return _HAS_TORCH and isinstance(obj, torch.Tensor)

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if _is_torch_tensor(warp_mat):
            warp_mat = warp_mat.to(dtype=torch.float32)
        else:
            warp_mat = warp_mat.astype(np.float32, copy=False)

        # --- Torch-specific path support ---
        def _torch_warp_single(img_t, warp_np):
            if not _HAS_TORCH:
                raise RuntimeError("torch is required for tensor warping support.")

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

            B, C, h_in, w_in = img_float.shape
            dsize = (int(h), int(w))  # (H,W)
            H_out, W_out = dsize

            w_in_f = float(max(w_in, 1))
            h_in_f = float(max(h_in, 1))
            W_out_f = float(max(W_out, 1))
            H_out_f = float(max(H_out, 1))

            A_out = torch.tensor(
                [
                    [W_out_f / 2.0, 0.0, (W_out_f - 1.0) / 2.0],
                    [0.0, H_out_f / 2.0, (H_out_f - 1.0) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=img_float.dtype,
                device=device,
            )

            A_in_inv = torch.tensor(
                [
                    [2.0 / w_in_f, 0.0, -(w_in_f - 1.0) / w_in_f],
                    [0.0, 2.0 / h_in_f, -(h_in_f - 1.0) / h_in_f],
                    [0.0, 0.0, 1.0],
                ],
                dtype=img_float.dtype,
                device=device,
            )

            M_cv2 = torch.eye(3, dtype=img_float.dtype, device=device)
            M_cv2[:2, :] = torch.as_tensor(
                warp_np, dtype=img_float.dtype, device=device
            )
            M_inv = torch.linalg.inv(M_cv2)

            theta = A_in_inv @ M_inv @ A_out
            theta = theta[:2, :]
            theta = theta.unsqueeze(0)
            if B > 1:
                theta = theta.expand(B, -1, -1)

            grid = F.affine_grid(
                theta,
                size=(B, C, H_out, W_out),
                align_corners=False,
            )

            out = F.grid_sample(
                img_float,
                grid,
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
            warped_imgs = []
            for img in imgs:
                if _is_torch_tensor(img):
                    warped_imgs.append(_torch_warp_single(img, warp_mat))
                elif isinstance(img, np.ndarray):
                    warped_imgs.append(
                        cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                    )
                else:
                    raise TypeError(
                        "Unsupported image type in list: "
                        f"{type(img).__name__}"  # pragma: no cover
                    )
            results["img"] = warped_imgs
        else:
            if _is_torch_tensor(imgs):
                results["img"] = _torch_warp_single(imgs, warp_mat)
            elif isinstance(imgs, np.ndarray):
                results["img"] = cv2.warpAffine(
                    imgs, warp_mat, warp_size, flags=cv2.INTER_LINEAR
                )
            else:
                raise TypeError(
                    "Unsupported image type: "
                    f"{type(imgs).__name__}"  # pragma: no cover
                )

        if results.get('keypoints', None) is not None:
            kps = results["keypoints"]
            existing = results.get("transformed_keypoints")
            if existing is None:
                if _is_torch_tensor(kps):
                    transformed_keypoints = kps.clone()
                else:
                    transformed_keypoints = kps.copy()
            else:
                transformed_keypoints = (
                    existing.clone() if _is_torch_tensor(existing) else existing.copy()
                )

            if _is_torch_tensor(transformed_keypoints):
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
                transformed_keypoints[..., :2] = cv2.transform(
                    kps[..., :2].astype(np.float32, copy=False), warp_mat
                ).astype(transformed_keypoints.dtype, copy=False)
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
