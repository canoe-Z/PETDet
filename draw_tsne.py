from argparse import ArgumentParser

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
import torch.nn.functional as F
from mmdet.datasets.builder import build_dataset
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import cv2
from tqdm import tqdm
import os
from mmdet.datasets import build_dataloader, replace_ImageToTensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def draw_tsne(X, y):
    colormap = plt.cm.viridis

    colors = [colormap(i) for i in np.linspace(0, 1, 20)]
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    for i, c in zip(range(len(np.unique(y))), colors):
        plt.scatter(X_tsne[y[:, 0] == i, 0],
                    X_tsne[y[:, 0] == i, 1], c=c, label=i, s=4)
    plt.legend()
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('./maskformer_r50_ctl/tsne.png', dpi=300)


def main(args):
    cfg = Config.fromfile(args.config)
    cfg.data.train.test_mode = True
    cfg.data.train.pipeline = replace_ImageToTensor(
        cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.train)
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # np_list = []
    # mean = []
    # hue = []

    # c = [f'A{i + 1}' for i in range(20)]
    # classes = tuple(c)
    # label2cls = {i + 1: c for i, c in enumerate(classes)}

    features = []

    # cls_convs = model.bbox_head.cls_convs

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # feats_per_cat = {i: [] for i in range(20)}

    features = []

    def hook(module, input, output):
        # print(output[0])
        features.append(output[1].clone().detach())
    model.neck.register_forward_hook(hook)
    model = MMDataParallel(model, device_ids=range(1))
    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            print(len(features))
            print(features[0].shape)
            assert (1 == 2)
    # for item in dataset:
    #     item['img_metas'] = [item['img_metas'].data]
    #     print(item['img_metas'])
    #     assert (1 == 2)
    #     item['img'] = [item['img'].data]
    #     # data['img_metas'] = [img_metas.data[0] for img_metas in item['img_metas']]
    #     # data['img'] = [img.data[0] for img in item['img']]
    #     model(return_loss=False, rescale=True, **item)
    #     # inference_detector(model, item['img'].data.numpy())
        print(len(features))
    #     feat = features[0]
    #     with torch.no_grad():
    #         for cls_layer in cls_convs:
    #             feat = cls_layer(feat)
    #         # print(feat.size())

    #     feat_h, feat_w = feat.shape[2:]
    #     scale_h = feat_h / H
    #     scale_w = feat_w / W
    #     for ann in anns:
    #         # print(feat.size())
    #         cat_id = ann['category_id']
    #         x, y, w, h = ann['bbox']
    #         w_f = int(w * scale_w)
    #         h_f = int(h * scale_h)

    #         x1 = max(int(x * scale_w), 1)
    #         y1 = max(int(y * scale_h), 1)

    #         x2 = min(x1 + w_f, feat_w)
    #         y2 = min(y1 + h_f, feat_h)
    #         if x2 == x1:
    #             if x2 == W:
    #                 x1 = x1 - 1
    #             else:
    #                 x2 = x2 + 1
    #         if y2 == y1:
    #             if y2 == H:
    #                 y1 = y1 - 1
    #             else:
    #                 y2 = y2 + 1

    #         obj = feat[:, :, y1:y2, x1:x2]

    #         obj_v = F.adaptive_max_pool2d(obj, (1, 1))
    #         obj_v = obj_v.squeeze().cpu().numpy()
    #         # print(np.min(obj_v))
    #         # np_list.append(
    #         # hue.append(label2cls[cat_id+1])

    #         feats_per_cat[cat_id].append(obj_v)
    #     features.clear()
    #     # handle.remove()

    # for k, v in feats_per_cat.items():
    #     for np_v in v:
    #         np_list.append(np_v)
    #         mean.append(np_v)
    #         hue.append(label2cls[k + 1])

    # t = np.stack(np_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
