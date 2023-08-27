# PETDet
Official implement for the paper: PETDet: Proposal Enhancement for Two-Stage Fine-Grained Object Detection (under review).


## Introduction
Fine-grained object detection (FGOD) extends object detection with the capability of fine-grained recognition. In recent two-stage FGOD methods, the region proposal serves as a crucial link between detection and fine-grained recognition. However, current methods overlook that some proposal-related procedures inherited from general detection are not equally suitable for FGOD, limiting the multi-task learning from generation, representation, to utilization. In this paper, we present PETDet (Proposal Enhancement for Two-stage fine-grained object detection) to properly handle the sub-tasks in two-stage FGOD methods. Firstly, an anchor-free Quality Oriented Proposal Network (QOPN) is proposed with dynamic label assignment and attention-based decomposition to generate high-quality oriented proposals. Additionally, we present a Bilinear Channel Fusion Network (BCFN) to extract independent and discriminative features from the proposals. Furthermore, we design a novel Adaptive Recognition Loss (ARL) which offers guidance for the R-CNN head to focus on high-quality proposals. Extensive experiments validate the effectiveness of PETDet. Quantitative analysis reveals that PETDet with ResNet50 reaches state-of-the-art performance on various FGOD datasets, including FAIR1M-v1.0 (42.96 AP), FAIR1M-v2.0 (48.81 AP), MAR20 (85.91 AP) and ShipRSImageNet (74.90 AP). The proposed method also achieves superior compatibility between accuracy and inference speed. Our code and models will be released at https://github.com/canoe-Z/PETDet.

## Results and Models
comming soon.

## Installation
This repo is based on [mmrotate 0.x](https://github.com/open-mmlab/mmrotate) and [OBBDetection](https://github.com/jbwang1997/OBBDetection).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name petdet python=3.10 -y
conda activate petdet
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3.** Install [MMCV 1.x](https://github.com/open-mmlab/mmcv) and [MMDetection 2.x](https://github.com/open-mmlab/mmdetection) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
```

**Step 4.** Install PETDet from source.
```shell
git clone https://github.com/canoe-Z/PETDet.git
cd mmrotate
pip install -v -e .
```
## Data Preparation
For FAIR1M, Please crop the original images into 1024×1024 patches with an overlap of 200 by run the [split tool](tools/data/fair1m/README.md).

The data structure is as follows:

```none
PETDet
├── mmrotate
├── tools
├── configs
├── data
|   ├── FAIR1M1_0
│   │   ├── train
│   │   ├── test
│   ├── FAIR1M2_0
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   ├── MAR20
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   ├── ShipRSImageNet
│   │   ├── COCO_Format
│   │   ├── VOC_Format
```

## Inference

Assuming you have put the splited FAIR1M dataset into `data/split_ss_fair1m2_0/` and have downloaded the models into the `weights/`, you can now evaluate the models on the FAIR1M_V2.0 test split:

```
./dist_test.sh configs/petdet/ \
  petdet_r50_fpn_1x_fair1m_le90.py \
  weights/petdet_r50_fpn_1x_fair1m_le90.pth 4 --format-only \
  --eval-options submission_dir=work_dirs/FAIR1M_2.0_results
```

Then, you can upload `work_dirs/FAIR1M_2.0_results/submission_zip/test.zip` to [ISPRS Benchmark](https://www.gaofen-challenge.com/benchmark).

## Training

The following command line will train `petdet_r50_fpn_1x_fair1m_le90` on 4 GPUs:

```
./dist_train.sh configs/petdet/petdet_r50_fpn_1x_fair1m_le90.py 4
```

**Notes:**
- The models will be saved into `work_dirs/petdet_r50_fpn_1x_fair1m_le90`.
- If you use a different mini-batch size, please change the learning rate according to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).
- We use 4 RTX3090 GPUs for the training of these models with a mini-batch size of 8 images (2 images per GPU). However, we found that training with a smaller batchsize may yield slightly better results on the FGOD tasks.
