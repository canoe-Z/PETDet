import mmcv
from mmcv import print_log

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
import numpy as np
from collections import OrderedDict

from mmrotate.core.evaluation import eval_rbbox_recalls
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class ShipRSImageNet(CocoDataset):
    CLASSES_level = []
    CLASSES_level.append(('Ship', 'Dock',))
    CLASSES_level.append(('Other Ship', 'Warship', 'Merchant', 'Dock',))
    CLASSES_level.append(('Other Ship', 'Other Warship', 'Submarine', 'Aircraft Carrier', 'Cruiser', 'Destroyer',
                          'Frigate', 'Patrol', 'Landing', 'Commander', 'Auxiliary Ships', 'Other Merchant',
                          'Container Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht',
                          'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft', 'Motorboat', 'Dock',))
    CLASSES_level.append(('Other Ship', 'Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway',
                          'Ticonderoga',
                          'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate',
                          'Perry FF',
                          'Patrol', 'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 'Osumi LL',
                          'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 'Medical Ship', 'Test Ship',
                          'Training Ship',
                          'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 'Other Merchant', 'Container Ship', 'RoRo', 'Cargo',
                          'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft',
                          'Motorboat', 'Dock',))

    def __init__(self,
                 level=0,
                 version='oc',
                 **kwargs):
        self.level = level
        self.version = version
        classes = self.CLASSES_level[level]
        super(ShipRSImageNet, self).__init__(
            classes=classes,
            **kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            polygon = np.array(ann['segmentation'], dtype=np.float32)[0]
            bbox = np.array(poly2obb_np(
                polygon, self.version), dtype=np.float32)
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_polygons.append(polygon)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_polygons = np.array(gt_polygons, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_polygons = np.zeros((0, 8), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            polygons=gt_polygons)

        return ann

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 500, 1000, 2000),
            iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            assert mmcv.is_list_of(results, np.ndarray)
            gt_bboxes = []
            for i in range(len(self)):
                ann = self.get_ann_info(i)
                bboxes = ann['bboxes']
                gt_bboxes.append(bboxes)
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_rbbox_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        else:
            raise NotImplementedError

        return eval_results
