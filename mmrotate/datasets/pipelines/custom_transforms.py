import numpy as np
from numpy import random

from ..builder import ROTATED_PIPELINES
from icecream import ic


@ROTATED_PIPELINES.register_module()
class OBBMixUp(object):
    def __init__(self, p=0.3, lambd=0.5, to_float32=False):
        self.lambd = lambd
        self.p = p
        self.to_float32 = to_float32
        self.img2 = None
        self.boxes2 = None
        self.masks2 = None
        self.labels2 = None

        ic("USE OBBMixUp")
        ic(p, lambd)

    def __call__(self, results):
        img1, boxes1, labels1 = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]

        if random.random() < self.p and self.img2 is not None and img1.shape[1] == self.img2.shape[1]:
            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.shape[1],
                        :] = img1.astype('float32') * self.lambd
            mixup_image[:self.img2.shape[0], :self.img2.shape[1],
                        :] += self.img2.astype('float32') * (1. - self.lambd)
            if not self.to_float32:
                mixup_image = mixup_image.astype('uint8')
            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_label = np.hstack((labels1, self.labels2))
            results['img'] = mixup_image
            results['gt_bboxes'] = mixup_boxes
            results['gt_labels'] = mixup_label
        else:
            pass

        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 = labels1
        return results


@ROTATED_PIPELINES.register_module()
class OBBoxJitter(object):
    def __init__(self, min=0.95, max=1.05, min_t=0.95, max_t=1.05):
        self.min_scale = min
        self.max_scale = max
        self.min_t_scale = min_t
        self.max_t_scale = max_t
        self.count = 0

        ic("USE OBBOX_JITTER")
        ic(min, max)

    def __call__(self, results):
        if len(results['gt_bboxes']) == 0:
            pass
        else:
            obboxes = results['gt_bboxes']

            for obbox in obboxes:
                box_scale = np.random.uniform(self.min_scale, self.max_scale)
                theta_scale = np.random.uniform(
                    self.min_t_scale, self.max_t_scale)
                obbox[2:4] = obbox[2:4] * box_scale
                obbox[4] = obbox[4] * theta_scale

            results['gt_bboxes'] = obboxes
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_jitter={}-{})'.format(
            self.min_scale, self.max_scale, self.min_t_scale, self.max_t_scale)
