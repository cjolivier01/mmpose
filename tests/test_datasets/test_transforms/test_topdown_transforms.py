# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase, skipUnless

import numpy as np

try:  # pragma: no cover
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from mmpose.datasets.transforms import TopdownAffine
from mmpose.testing import get_coco_sample


class TestTopdownAffine(TestCase):

    def setUp(self):
        # prepare dummy top-down data sample with COCO metainfo
        self.data_info = get_coco_sample(num_instances=1, with_bbox_cs=True)

    def test_transform(self):
        # without udp
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))
        self.assertIn('transformed_keypoints', results)

        # with udp
        transform = TopdownAffine(input_size=(192, 256), use_udp=True)
        results = transform(deepcopy(self.data_info))
        self.assertEqual(results['input_size'], (192, 256))
        self.assertEqual(results['img'].shape, (256, 192, 3))
        self.assertIn('transformed_keypoints', results)

    def test_repr(self):
        transform = TopdownAffine(input_size=(192, 256), use_udp=False)
        self.assertEqual(
            repr(transform),
            'TopdownAffine(input_size=(192, 256), use_udp=False)')

    @skipUnless(_HAS_TORCH, "torch is required for tensor tests")
    def test_torch_tensor_matches_numpy(self):
        transform_np = TopdownAffine(input_size=(192, 256), use_udp=False)
        np_results = transform_np(deepcopy(self.data_info))

        transform_torch = TopdownAffine(input_size=(192, 256), use_udp=False)
        tensor_sample = deepcopy(self.data_info)
        tensor_sample["img"] = torch.from_numpy(tensor_sample["img"])
        tensor_sample["keypoints"] = torch.from_numpy(tensor_sample["keypoints"]).to(
            dtype=torch.float32
        )

        torch_results = transform_torch(tensor_sample)

        self.assertTrue(isinstance(torch_results["img"], torch.Tensor))
        self.assertTrue(
            isinstance(torch_results["transformed_keypoints"], torch.Tensor)
        )

        np.testing.assert_allclose(
            torch_results["img"].cpu().numpy(),
            np_results["img"],
            rtol=0.0,
            atol=6.0,
        )
        np.testing.assert_allclose(
            torch_results["transformed_keypoints"].cpu().numpy(),
            np_results["transformed_keypoints"],
            rtol=0.0,
            atol=1e-4,
        )

        self.assertEqual(torch_results["input_size"], np_results["input_size"])
        np.testing.assert_allclose(
            np.array(torch_results["input_center"]),
            np_results["input_center"],
            rtol=0.0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(torch_results["input_scale"]),
            np_results["input_scale"],
            rtol=0.0,
            atol=1e-4,
        )

    @skipUnless(_HAS_TORCH, "torch is required for tensor tests")
    def test_fix_aspect_ratio_torch(self):
        aspect_ratio = 192 / 256
        bbox_scales_np = np.array(
            [[100.0, 120.0], [40.0, 30.0]], dtype=np.float32
        )
        expected = TopdownAffine._fix_aspect_ratio(
            bbox_scales_np.copy(), aspect_ratio
        )

        bbox_scales_t = torch.from_numpy(bbox_scales_np)
        result_t = TopdownAffine._fix_aspect_ratio(
            bbox_scales_t.clone(), aspect_ratio
        )

        self.assertIsInstance(result_t, torch.Tensor)
        self.assertEqual(result_t.dtype, bbox_scales_t.dtype)
        self.assertEqual(result_t.device, bbox_scales_t.device)

        np.testing.assert_allclose(
            result_t.cpu().numpy(), expected, rtol=0.0, atol=1e-5
        )
