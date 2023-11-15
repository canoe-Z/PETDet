import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from argparse import ArgumentParser
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.apis import init_detector
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.datasets.builder import build_dataset
from sklearn.manifold import TSNE

import mmrotate  # noqa: F401

AIRPLANE_CLASSES = {7: 'B737', 9: 'B777', 8: 'B747', 10: 'B787', 1: 'A321',
                    0: 'A220', 2: 'A330', 3: 'A350', 13: 'C919', 4: 'ARJ21'}
# , 34: 'OP'}
SHIP_CLASSES = {24: 'PS', 23: 'MB', 19: 'FB', 31: 'TB', 17: 'ES',
                22: 'LCS', 15: 'DCS', 33: 'WS'}
# , 35: 'OS'}
VEHICLE_CLASSES = {26: 'SC', 12: 'BUS', 14: 'CT', 16: 'DT', 32: 'VAN',
                   29: 'TRI', 28: 'TRC', 30: 'TT', 18: 'EX'}
# , 36: 'OV'}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--course-class', default='airplane')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def draw_tsne(X: np.ndarray,
              y: np.ndarray,
              classes: dict,
              out_file: str):
    colormap = plt.cm.tab10

    colors = [colormap(i) for i in np.linspace(0, 1, len(classes))]
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10))
    for i, c in zip(np.unique(y), colors):
        if i in classes:
            plt.scatter(X_tsne[y == i, 0],
                        X_tsne[y == i, 1], c=c, label=classes[i], s=4)
    plt.legend()
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(out_file, dpi=300)


def main(args):
    if args.course_class == 'airplane':
        MASK_CLASSES = AIRPLANE_CLASSES
    elif args.course_class == 'ship':
        MASK_CLASSES = SHIP_CLASSES
    elif args.course_class == 'vehicle':
        MASK_CLASSES = VEHICLE_CLASSES
    else:
        raise NotImplementedError

    cfg = Config.fromfile(args.config)
    cfg.data.train.pipeline = replace_ImageToTensor(
        cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.train)
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = init_detector(args.config, args.checkpoint, device=args.device)

    features = []
    labels = []

    def hook(module, input, output):
        features.append(input[0])

    model.roi_head.bbox_head.fc_cls.register_forward_hook(hook)

    model = MMDataParallel(model, device_ids=range(1))
    model.eval()

    iter = 5000
    prog_bar = mmcv.ProgressBar(iter)
    for i, data in enumerate(data_loader):
        data['img'] = [data['img']]
        data['img_metas'] = [data['img_metas']]

        gt_bboxes = data['gt_bboxes'].data[0][0]
        gt_labels = data['gt_labels'].data[0][0]
        mask = torch.tensor(list(MASK_CLASSES.keys()), dtype=int)
        index = torch.where(torch.isin(gt_labels, mask))[0]

        if index.numel() != 0:
            data.pop('gt_bboxes', None)
            data.pop('gt_labels', None)
            data['proposals'] = [[torch.index_select(gt_bboxes, 0, index)]]
            labels.append(torch.index_select(gt_labels, 0, index))

            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)

        prog_bar.update()
        if i == iter-1:
            break

    features: np.ndarray = torch.cat(features, dim=0).cpu().numpy()
    labels: np.ndarray = torch.cat(labels, dim=0).cpu().numpy()
    print(features.shape)
    print(labels.shape)
    draw_tsne(features, labels, MASK_CLASSES, args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
