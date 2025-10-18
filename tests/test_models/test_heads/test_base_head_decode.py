# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.heads.base_head import BaseHead
from mmpose.registry import KEYPOINT_CODECS


class DummyHead(BaseHead):

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, feats):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def predict(self, feats, batch_data_samples, test_cfg=None):  # pragma: no cover
        raise NotImplementedError

    def loss(self, feats, batch_data_samples, train_cfg=None):  # pragma: no cover
        raise NotImplementedError


class TestBaseHeadDecode(TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.decoder_cfg = dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=6.0,
            simcc_split_ratio=2.0)

    def test_decode_with_torch_tensors(self):
        decoder = KEYPOINT_CODECS.build(self.decoder_cfg)
        head = DummyHead(decoder)

        batch_pred_x = torch.rand(2, 17, int(192 * decoder.simcc_split_ratio))
        batch_pred_y = torch.rand(2, 17, int(256 * decoder.simcc_split_ratio))

        preds = head.decode((batch_pred_x, batch_pred_y))

        self.assertEqual(len(preds), 2)
        for pred in preds:
            self.assertIsInstance(pred.keypoints, torch.Tensor)
            self.assertIsInstance(pred.keypoint_scores, torch.Tensor)
            self.assertEqual(pred.keypoints.shape, (1, 17, 2))
            self.assertEqual(pred.keypoint_scores.shape, (1, 17))
            self.assertFalse(hasattr(pred, 'keypoints_visible'))

    def test_decode_with_visibility(self):
        decoder_cfg = dict(self.decoder_cfg)
        decoder_cfg['decode_visibility'] = True
        decoder = KEYPOINT_CODECS.build(decoder_cfg)
        head = DummyHead(decoder)

        batch_pred_x = torch.rand(1, 17, int(192 * decoder.simcc_split_ratio))
        batch_pred_y = torch.rand(1, 17, int(256 * decoder.simcc_split_ratio))

        preds = head.decode((batch_pred_x, batch_pred_y))

        self.assertEqual(len(preds), 1)
        pred = preds[0]
        self.assertTrue(hasattr(pred, 'keypoints_visible'))
        self.assertEqual(pred.keypoints_visible.shape, (1, 17))
        self.assertIsInstance(pred.keypoints_visible, torch.Tensor)
