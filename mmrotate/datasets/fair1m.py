# Copyright (c) OpenMMLab. All rights reserved.
import glob
import mmcv
import numpy as np
import os
import os.path as osp
import zipfile
from collections import OrderedDict
from mmcv import print_log

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_recalls
from mmrotate.datasets.dota import DOTADataset
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class FAIR1MDataset(DOTADataset):
    """FAIR1M dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('A220', 'A321', 'A330', 'A350', 'ARJ21', 'Baseball Field',
               'Basketball Court', 'Boeing737', 'Boeing747', 'Boeing777',
               'Boeing787', 'Bridge', 'Bus', 'C919', 'Cargo Truck',
               'Dry Cargo Ship', 'Dump Truck', 'Engineering Ship',
               'Excavator', 'Fishing Boat', 'Football Field', 'Intersection',
               'Liquid Cargo Ship', 'Motorboat', 'Passenger Ship', 'Roundabout',
               'Small Car', 'Tennis Court', 'Tractor', 'Trailer', 'Truck Tractor',
               'Tugboat', 'Van', 'Warship', 'other-airplane', 'other-ship',
               'other-vehicle')

    COURSE_TABLE = [0, 0, 0, 0, 0, 3,
                    3, 0, 0, 0,
                    0, 4, 2, 0, 2,
                    1, 2, 1,
                    2, 1, 3, 4,
                    1, 1, 1, 4,
                    2, 3, 2, 2, 2,
                    1, 2, 1, 0, 1, 2]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 course_label=False,
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.course_label = course_label
        self.difficulty = difficulty

        super(FAIR1MDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        if self.course_label:
            cls_map = {c: self.COURSE_TABLE[i]
                       for i, c in enumerate(self.CLASSES)
                       }  # in mmdet v2.0 label is 0-based
        else:
            cls_map = {c: i
                       for i, c in enumerate(self.CLASSES)
                       }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = ' '.join(bbox_info[8:-1])
                        difficulty = int(bbox_info[-1])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 500, 1000, 2000),
                 iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7,
                          0.75, 0.8, 0.85, 0.9, 0.95],
                 scale_ranges=None,
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
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
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
            for info in self.data_infos:
                bboxes = info['ann']['bboxes']
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

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Params:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, img_id+'.xml')
            for img_id in id_list
        ]
        file_objs = [open(f, 'w') for f in files]
        for f, dets_per_cls in zip(file_objs, dets_list):
            self._write_head(f)
            for cls, dets in zip(self.CLASSES, dets_per_cls):
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    self._write_obj(f, cls, bbox[:-1], str(bbox[-1]))
            self._write_tail(f)

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        zip_folder = osp.join(out_folder, 'submission_zip')
        os.makedirs(zip_folder)
        with zipfile.ZipFile(
                osp.join(zip_folder, 'test.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, os.path.join('test', osp.split(f)[-1]))

        return files

    def _write_head(self, f):
        head = """<?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <source>
            <filename>placeholder_filename</filename>
            <origin>GF2/GF3</origin>
            </source>
            <research>
                <version>4.0</version>
                <provider>placeholder_affiliation</provider>
                <author>placeholder_authorname</author>
                <!--参赛课题 -->
                <pluginname>placeholder_direction</pluginname>
                <pluginclass>placeholder_suject</pluginclass>
                <time>2020-07-2020-11</time>
            </research>
            <size>
                <width>placeholder_width</width>
                <height>placeholder_height</height>
                <depth>placeholder_depth</depth>
            </size>
            <!--存放目标检测信息-->
            <objects>
        """
        f.write(head)

    def _write_obj(self, f, cls: str,  bbox, conf: str):
        obj_str = """        <object>
                    <coordinate>pixel</coordinate>
                    <type>rectangle</type>
                    <description>None</description>
                    <possibleresult>
                        <name>palceholder_cls</name>                
                        <probability>palceholder_prob</probability>
                    </possibleresult>
                    <!--检测框坐标，首尾闭合的矩形，起始点无要求-->
                    <points>  
                        <point>palceholder_coord0</point>
                        <point>palceholder_coord1</point>
                        <point>palceholder_coord2</point>
                        <point>palceholder_coord3</point>
                        <point>palceholder_coord0</point>
                    </points>
                </object>
        """
        obj_xml = obj_str.replace("palceholder_cls", cls)
        obj_xml = obj_xml.replace("palceholder_prob", conf)
        obj_xml = obj_xml.replace(
            "palceholder_coord0", f'{bbox[0]:.2f}'+", "+f'{bbox[1]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord1", f'{bbox[2]:.2f}'+", "+f'{bbox[3]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord2", f'{bbox[4]:.2f}'+", "+f'{bbox[5]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord3", f'{bbox[6]:.2f}'+", "+f'{bbox[7]:.2f}')
        f.write(obj_xml)

    def _write_tail(self, f):
        tail = """    </objects>
        </annotation>
        """
        f.write(tail)


@ROTATED_DATASETS.register_module()
class FAIR1MCourseDataset(FAIR1MDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 course_label=True,
                 difficulty=100,
                 **kwargs):

        super(FAIR1MCourseDataset, self).__init__(
            ann_file,
            pipeline,
            version,
            course_label,
            difficulty,
            **kwargs)

        self.CLASSES = ('Airplane', 'Ship', 'Vehicle', 'Court', 'Road')
