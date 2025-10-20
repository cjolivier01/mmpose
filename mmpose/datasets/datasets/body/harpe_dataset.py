import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_cs2xyxy
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class HarpeDataset(BaseCocoStyleDataset):
    """HARPE(HARPET) Dataset in HDF5 format for top-down 2D keypoint.

    Expects H5 files with datasets 'imgname' (N, L) and 'part' (N, K, 2), and
    image directories named per split. The annotation format contains no
    visibility; all keypoints are considered visible.

    Args (following BaseCocoStyleDataset):
        ann_file (str): Path to H5 annotation file.
        data_mode (str): 'topdown' or 'bottomup'. Only 'topdown' supported.
        metainfo (dict, optional): Dataset meta info; defaults to
            configs/_base_/datasets/harpe18.py
        data_root (str, optional): Root directory for data.
        data_prefix (dict): dict(img='path/to/images_split').
        ... others follow BaseCocoStyleDataset.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/harpe18.py')

    def __init__(
        self,
        ann_file: str = "",
        data_mode: str = "topdown",
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img=""),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = True,
        max_refetch: int = 1000,
    ):
        if data_mode != 'topdown':
            raise ValueError(
                f'{self.__class__.__name__} only supports topdown data_mode.')

        super().__init__(
            ann_file=ann_file,
            bbox_file=None,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        assert osp.exists(self.ann_file), (
            f'Annotation file `{self.ann_file}` does not exist')

        with h5py.File(self.ann_file, 'r') as f:
            imgname = f['imgname'][:]
            parts = f['part'][:].astype(np.float32)  # (N, K, 2)

        def _decode(row: np.ndarray) -> str:
            return ''.join(chr(int(x)) for x in row if int(x) != 0)

        instance_list: List[dict] = []
        image_list: List[dict] = []
        used_img_ids = set()

        # As in MPII, bbox scales are normalized with factor 200.
        pixel_std = 200.0

        for idx in range(parts.shape[0]):
            kpts = parts[idx]
            name = _decode(imgname[idx])
            img_path = osp.join(self.data_prefix['img'], name)

            # Compute center/scale from keypoints bbox
            xs = kpts[:, 0]
            ys = kpts[:, 1]
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            # Make square bbox by using the larger side
            side = max(max_x - min_x, max_y - min_y)
            scale = np.array([[side, side]], dtype=np.float32)
            center = np.array([[cx, cy]], dtype=np.float32)
            bbox = bbox_cs2xyxy(center, scale)

            # unify shapes
            keypoints = kpts.reshape(1, -1, 2).astype(np.float32)
            # no visibility flag in H5; use ones
            keypoints_visible = np.ones((1, kpts.shape[0]), dtype=np.float32)

            x1, y1, x2, y2 = np.split(bbox, axis=1, indices_or_sections=4)
            area = np.clip((x2 - x1) * (y2 - y1), a_min=1.0, a_max=None)
            area = area[..., 0].astype(np.float32)

            instance_info = {
                'id': idx,
                'img_id': idx,  # use running index as image id
                'img_path': img_path,
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'area': area,
                'category_id': [1],
            }

            if instance_info['img_id'] not in used_img_ids:
                used_img_ids.add(instance_info['img_id'])
                image_list.append({
                    'img_id': instance_info['img_id'],
                    'img_path': instance_info['img_path'],
                })

            instance_list.append(instance_info)

        return instance_list, image_list

