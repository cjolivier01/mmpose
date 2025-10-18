# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmpose.codecs.utils.post_processing import (
    gaussian_blur,
    gaussian_blur1d,
    get_heatmap_3d_maximum,
    get_heatmap_maximum,
    get_simcc_maximum,
    get_simcc_normalized,
)


class TestPostProcessingUtils(TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.rng = np.random.default_rng(123)

    def test_get_simcc_maximum_torch_parity(self):
        simcc_x = self.rng.random((2, 5, 32), dtype=np.float32)
        simcc_y = self.rng.random((2, 5, 28), dtype=np.float32)

        loc_np, val_np = get_simcc_maximum(simcc_x.copy(), simcc_y.copy())
        loc_t, val_t = get_simcc_maximum(
            torch.from_numpy(simcc_x.copy()),
            torch.from_numpy(simcc_y.copy()))

        self.assertTrue(
            np.allclose(loc_np, loc_t.cpu().numpy(), atol=1e-4, rtol=1e-4))
        self.assertTrue(
            np.allclose(val_np, val_t.cpu().numpy(), atol=1e-5, rtol=1e-4))

        loc_np_soft, val_np_soft = get_simcc_maximum(
            simcc_x.copy(), simcc_y.copy(), apply_softmax=True)
        loc_t_soft, val_t_soft = get_simcc_maximum(
            torch.from_numpy(simcc_x.copy()),
            torch.from_numpy(simcc_y.copy()),
            apply_softmax=True)

        self.assertTrue(
            np.allclose(
                loc_np_soft, loc_t_soft.cpu().numpy(), atol=1e-4, rtol=1e-4))
        self.assertTrue(
            np.allclose(
                val_np_soft, val_t_soft.cpu().numpy(), atol=1e-5, rtol=1e-4))

    def test_get_heatmap_maximum_torch_parity(self):
        heatmaps = self.rng.random((3, 4, 6, 6), dtype=np.float32)

        loc_np, val_np = get_heatmap_maximum(heatmaps.copy())
        loc_t, val_t = get_heatmap_maximum(torch.from_numpy(heatmaps.copy()))

        self.assertTrue(
            np.allclose(loc_np, loc_t.cpu().numpy(), atol=1e-4, rtol=1e-4))
        self.assertTrue(
            np.allclose(val_np, val_t.cpu().numpy(), atol=1e-5, rtol=1e-4))

    def test_get_heatmap_3d_maximum_torch_parity(self):
        heatmaps = self.rng.random((2, 3, 4, 5, 5), dtype=np.float32)

        loc_np, val_np = get_heatmap_3d_maximum(heatmaps.copy())
        loc_t, val_t = get_heatmap_3d_maximum(torch.from_numpy(heatmaps.copy()))

        self.assertTrue(
            np.allclose(loc_np, loc_t.cpu().numpy(), atol=1e-4, rtol=1e-4))
        self.assertTrue(
            np.allclose(val_np, val_t.cpu().numpy(), atol=1e-5, rtol=1e-4))

    def test_gaussian_blur_torch_parity(self):
        heatmaps = self.rng.random((4, 6, 6), dtype=np.float32)

        blurred_np = gaussian_blur(heatmaps.copy(), kernel=5)
        blurred_t = gaussian_blur(torch.from_numpy(heatmaps.copy()), kernel=5)

        self.assertTrue(
            np.allclose(blurred_np, blurred_t.cpu().numpy(), atol=1e-4, rtol=1e-4))

    def test_gaussian_blur1d_torch_parity(self):
        simcc = self.rng.random((2, 5, 40), dtype=np.float32)

        blurred_np = gaussian_blur1d(simcc.copy(), kernel=7)
        blurred_t = gaussian_blur1d(torch.from_numpy(simcc.copy()), kernel=7)

        self.assertTrue(
            np.allclose(blurred_np, blurred_t.cpu().numpy(), atol=1e-4, rtol=1e-4))

    def test_get_simcc_normalized_torch_parity(self):
        simcc = torch.rand(2, 4, 10)
        sigma = 3.5

        norm_t = get_simcc_normalized(simcc.clone(), sigma)
        norm_np = get_simcc_normalized(simcc.cpu().numpy(), sigma)

        self.assertTrue(
            np.allclose(norm_np, norm_t.cpu().numpy(), atol=1e-5, rtol=1e-4))

        # Verify behavior when maximum value below threshold
        simcc_small = torch.full((1, 2, 6), 0.2)
        norm_t_small = get_simcc_normalized(simcc_small.clone(), sigma)
        norm_np_small = get_simcc_normalized(
            simcc_small.cpu().numpy(), sigma)
        self.assertTrue(
            np.allclose(
                norm_np_small, norm_t_small.cpu().numpy(), atol=1e-6, rtol=1e-6))
