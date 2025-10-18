# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class RegressionLabel(BaseKeypointCodec):
    r"""Generate keypoint coordinates.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_labels (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]

    """

    label_mapping_table = dict(
        keypoint_labels='keypoint_labels',
        keypoint_weights='keypoint_weights',
    )

    def __init__(self, input_size: Tuple[int, int]) -> None:
        super().__init__()

        self.input_size = input_size

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        w, h = self.input_size
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        keypoint_labels = (keypoints / np.array([w, h])).astype(np.float32)
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        encoded = dict(
            keypoint_labels=keypoint_labels, keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        torch_input = _HAS_TORCH and isinstance(encoded, torch.Tensor)
        if torch_input:
            device = encoded.device
            dtype = encoded.dtype
            encoded_np = encoded.detach().cpu().numpy()
        else:
            encoded_np = np.asarray(encoded)

        if encoded_np.shape[-1] == 2:
            N, K, _ = encoded_np.shape
            normalized_coords = encoded_np.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif encoded_np.shape[-1] == 4:
            # split coords and sigma if outputs contain output_sigma
            normalized_coords = encoded_np[..., :2].copy()
            output_sigma = encoded_np[..., 2:4].copy()

            scores = (1 - output_sigma).mean(axis=-1)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {encoded_np.shape[-1]}')

        w, h = self.input_size
        keypoints = normalized_coords * np.array([w, h])

        if torch_input:
            keypoints = torch.from_numpy(keypoints).to(device=device, dtype=dtype)
            scores = torch.from_numpy(scores).to(device=device, dtype=dtype)

        return keypoints, scores
