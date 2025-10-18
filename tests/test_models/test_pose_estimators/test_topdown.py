# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from copy import deepcopy
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData
from parameterized import parameterized

from mmpose.structures import PoseDataSample
from mmpose.testing import get_packed_inputs, get_pose_estimator_cfg
from mmpose.utils import register_all_modules

configs = [
    'body_2d_keypoint/topdown_heatmap/coco/'
    'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    'configs/body_2d_keypoint/topdown_regression/coco/'
    'td-reg_res50_8xb64-210e_coco-256x192.py',
    'configs/body_2d_keypoint/simcc/coco/'
    'simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192.py',
]

configs_with_devices = [(config, ('cpu', 'cuda')) for config in configs]


class TestTopdownPoseEstimator(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    @parameterized.expand(configs)
    def test_init(self, config):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator
        model = build_pose_estimator(model_cfg)
        self.assertTrue(model.backbone)
        self.assertTrue(model.head)
        if model_cfg.get('neck', None):
            self.assertTrue(model.neck)

    @parameterized.expand(configs_with_devices)
    def test_forward_loss(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2)
            data = model.data_preprocessor(packed_inputs, training=True)
            losses = model.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand(configs_with_devices)
    def test_forward_predict(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            param_device = next(model.parameters()).device

            packed_inputs = get_packed_inputs(2)
            model.eval()
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, training=True)
                batch_results = model.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], PoseDataSample)
                pred_instances = batch_results[0].pred_instances
                if 'keypoints' in pred_instances:
                    self.assertIsInstance(pred_instances.keypoints,
                                          torch.Tensor)
                    self.assertEqual(pred_instances.keypoints.device,
                                     param_device)
                if 'keypoints_visible' in pred_instances:
                    self.assertIsInstance(pred_instances.keypoints_visible,
                                          torch.Tensor)

    @parameterized.expand(configs_with_devices)
    def test_forward_tensor(self, config, devices):
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        for device in devices:
            model = build_pose_estimator(model_cfg)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = get_packed_inputs(2)
            data = model.data_preprocessor(packed_inputs, training=True)
            batch_results = model.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, (tuple, torch.Tensor))

    def test_add_pred_to_datasample_tensor_parity(self):
        config = configs[0]
        model_cfg = get_pose_estimator_cfg(config)
        model_cfg.backbone.init_cfg = None

        from mmpose.models import build_pose_estimator

        model = build_pose_estimator(model_cfg)

        rng = np.random.RandomState(1)
        num_instances = 2
        num_keypoints = 4

        keypoints = rng.rand(num_instances, num_keypoints, 3).astype(np.float32)
        keypoint_scores = rng.rand(num_instances, num_keypoints).astype(
            np.float32)
        input_center = rng.rand(num_instances, 2).astype(np.float32)
        input_scale = rng.rand(num_instances, 2).astype(np.float32) + 0.5
        input_size = np.array([192., 256.], dtype=np.float32)
        bboxes = rng.rand(num_instances, 4).astype(np.float32)

        base_meta = dict(  # avoid in-place mutation
            input_center=input_center,
            input_scale=input_scale,
            input_size=input_size)

        gt_instances = InstanceData()
        gt_instances.bboxes = bboxes
        gt_instances.bbox_scores = np.ones((num_instances, ), dtype=np.float32)

        def _make_data_sample():
            data_sample = PoseDataSample(metainfo=deepcopy(base_meta))
            data_sample.gt_instances = deepcopy(gt_instances)
            return data_sample

        data_sample_np = _make_data_sample()
        data_sample_tensor = _make_data_sample()

        pred_instances_np = InstanceData()
        pred_instances_np.keypoints = keypoints.copy()
        pred_instances_np.keypoint_scores = keypoint_scores.copy()

        pred_instances_tensor = InstanceData()
        pred_instances_tensor.keypoints = torch.from_numpy(keypoints)
        pred_instances_tensor.keypoint_scores = torch.from_numpy(
            keypoint_scores)

        model.add_pred_to_datasample([pred_instances_np], None,
                                     [data_sample_np])
        model.add_pred_to_datasample([pred_instances_tensor], None,
                                     [data_sample_tensor])

        torch_keypoints = data_sample_tensor.pred_instances.keypoints
        np_keypoints = data_sample_np.pred_instances.keypoints
        self.assertIsInstance(torch_keypoints, torch.Tensor)
        self.assertIsInstance(np_keypoints, np.ndarray)
        np.testing.assert_allclose(torch_keypoints.detach().cpu().numpy(),
                                   np_keypoints,
                                   atol=1e-5)

        torch_visible = data_sample_tensor.pred_instances.keypoints_visible
        np_visible = data_sample_np.pred_instances.keypoints_visible
        self.assertIsInstance(torch_visible, torch.Tensor)
        self.assertIsInstance(np_visible, np.ndarray)
        np.testing.assert_allclose(torch_visible.detach().cpu().numpy(),
                                   np_visible,
                                   atol=1e-5)

        # bbox information should be preserved without type conversions
        self.assertIsInstance(data_sample_tensor.pred_instances.bboxes,
                              np.ndarray)
