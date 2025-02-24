# PETDet: Proposal Enhancement for Two-Stage Fine-Grained Object Detection (TGRS 2023)

[[📖 Paper](https://arxiv.org/pdf/2312.10515.pdf)]Official implement for PETDet.

The second place winning solution (2/220) in the track of Fine-grained Object Recognition in High-Resolution Optical Images, 2021 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation.

<img src="docs/AIRVICLAB.png" width="200px">

## Updates
* ***[08/25/2024]*** We have fixed the bug in BCFN ([#7](https://github.com/canoe-Z/PETDet/issues/7)) and released new model weights. The performance does not change significantly on the three datasets due to this bug.
* ***[11/23/2023]*** PETDet is accepted to TGRS 2023.

## Introduction
![petdet_arch](docs/petdet.png) 

Fine-grained object detection (FGOD) extends object detection with the capability of fine-grained recognition. In recent two-stage FGOD methods, the region proposal serves as a crucial link between detection and fine-grained recognition. However, current methods overlook that some proposal-related procedures inherited from general detection are not equally suitable for FGOD, limiting the multi-task learning from generation, representation, to utilization. In this paper, we present PETDet (Proposal Enhancement for Two-stage fine-grained object detection) to properly handle the sub-tasks in two-stage FGOD methods. Firstly, an anchor-free Quality Oriented Proposal Network (QOPN) is proposed with dynamic label assignment and attention-based decomposition to generate high-quality oriented proposals. Additionally, we present a Bilinear Channel Fusion Network (BCFN) to extract independent and discriminative features from the proposals. Furthermore, we design a novel Adaptive Recognition Loss (ARL) which offers guidance for the R-CNN head to focus on high-quality proposals. Extensive experiments validate the effectiveness of PETDet. Quantitative analysis reveals that PETDet with ResNet50 reaches state-of-the-art performance on various FGOD datasets, including FAIR1M-v1.0 (42.96 AP), FAIR1M-v2.0 (48.81 AP), MAR20 (85.91 AP) and ShipRSImageNet (74.90 AP). The proposed method also achieves superior compatibility between accuracy and inference speed. Our code and models will be released at https://github.com/canoe-Z/PETDet.

## Results and Models
### FAIR1M-v2.0
|                       Method                        |           Backbone            | Angle | lr<br>schd |  Aug  | Batch<br>Size | AP<sub>50</sub> |                                                                                                                                               Download                                                                                                                                                |
| :-------------------------------------------------: | :---------------------------: | :---: | :--------: | :---: | :-----------: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [Faster R-CNN](https://arxiv.org/abs/1506.01497)   |  ResNet50<br>(1024,1024,200)  | le90  |     1x     |   -   |     2\*4      |      41.64      | [model](https://files.airvic.top/d/PETDet/FAIR1M2_0/faster/rotated_faster_rcnn_r50_fpn_1x_fair1m_le90.pth) \| [log](https://files.airvic.top/d/PETDet/FAIR1M2_0/faster/20230509_063048.log.json) \| [submission](https://files.airvic.top/d/PETDet/FAIR1M2_0/faster/rotated_faster_rcnn_r50_fpn_1x_fair1m_le90.zip) |
| [RoI Transformer](https://arxiv.org/abs/1812.00155) |  ResNet50<br>(1024,1024,200)  | le90  |     1x     |   -   |     2\*4      |      44.03      | [model](https://files.airvic.top/d/PETDet/FAIR1M2_0/roitrans/roi_trans_r50_fpn_1x_fair1m_le90.pth) \| [log](https://files.airvic.top/d/PETDet/FAIR1M2_0/roitrans/20230505_093458.log.json) \| [submission](https://files.airvic.top/d/PETDet/FAIR1M2_0/roitrans/roi_trans_r50_fpn_1x_fair1m_le90.zip) |
| [Oriented R-CNN](https://arxiv.org/abs/2108.05699)  |  ResNet50<br>(1024,1024,200)  | le90  |     1x     |   -   |     2\*4      |      43.90      | [model](https://files.airvic.top/d/PETDet/FAIR1M2_0/orcnn/oriented_rcnn_r50_fpn_1x_fair1m_le90.pth) \| [log](https://files.airvic.top/d/PETDet/FAIR1M2_0/orcnn/20230508_174344.log.json) \| [submission](https://files.airvic.top/d/PETDet/FAIR1M2_0/orcnn/oriented_rcnn_r50_fpn_1x_fair1m_le90.zip) |
|      [ReDet](https://arxiv.org/abs/2103.07733)      | ReResNet50<br>(1024,1024,200) | le90  |     1x     |   -   |     2\*4      |      46.03      | [model](https://files.airvic.top/d/PETDet/FAIR1M2_0/redet/redet_re50_refpn_1x_fair1m_le90.pth) \| [log](https://files.airvic.top/d/PETDet/FAIR1M2_0/redet/20230629_222320.log) \| [submission](https://files.airvic.top/d/PETDet/FAIR1M2_0/redet/redet_re50_refpn_1x_fair1m_le90.zip) |
|                       PETDet                        |  ResNet50<br>(1024,1024,200)  | le90  |     1x     |   -   |     2\*4      |      48.73      | [model](https://files.airvic.top/d/PETDet/FAIR1M2_0/petdet/petdet_r50_fpn_1x_fair1m_le90.pth) \| [log](https://files.airvic.top/d/PETDet/FAIR1M2_0/petdet/20240822_161836.log.json) \| [submission](https://files.airvic.top/d/PETDet/FAIR1M2_0/petdet/petdet_r50_fpn_1x_fair1m_le90.zip) |

### MAR20
|                       Method                        |       Backbone        | Angle | lr<br>schd |  Aug  | Batch<br>Size | AP<sub>50</sub> |  mAP  |                                                                                            Download                                                                                            |
| :-------------------------------------------------: | :-------------------: | :---: | :--------: | :---: | :-----------: | :-------------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [Faster R-CNN](https://arxiv.org/abs/1506.01497)   | ResNet50<br>(800,800) | le90  |     3x     |   -   |     2\*4      |      75.01      | 47.57 | [model](https://files.airvic.top/d/PETDet/MAR20/faster/rotated_faster_rcnn_r50_fpn_3x_mar20_le90.pth) \| [log](https://files.airvic.top/d/PETDet/MAR20/faster/20230618_172950.log.json) |
| [RoI Transformer](https://arxiv.org/abs/1812.00155) | ResNet50<br>(800,800) | le90  |     3x     |   -   |     2\*4      |      82.46      | 56.43 | [model](https://files.airvic.top/d/PETDet/MAR20/roitrans/roi_trans_r50_fpn_3x_mar20_le90.pth) \| [log](https://files.airvic.top/d/PETDet/MAR20/roitrans/20230615_194723.log) |
| [Oriented R-CNN](https://arxiv.org/abs/2108.05699)  | ResNet50<br>(800,800) | le90  |     3x     |   -   |     2\*4      |      82.71      | 58.14 | [model](https://files.airvic.top/d/PETDet/MAR20/orcnn/oriented_rcnn_r50_fpn_3x_mar20_le90.pth) \| [log](https://files.airvic.top/d/PETDet/MAR20/orcnn/20230601_153811.log) |
|                       PETDet                        | ResNet50<br>(800,800) | le90  |     3x     |   -   |     2\*4      |      85.89      | 61.46 | [model](https://files.airvic.top/d/PETDet/MAR20/petdet/petdet_r50_fpn_3x_mar20_le90.pth) \| [log](https://files.airvic.top/d/PETDet/MAR20/petdet/20240823_233637.log.json) |

### ShipRSImageNet
|                       Method                        |        Backbone         | Angle | lr<br>schd |  Aug  | Batch<br>Size | AP<sub>50</sub> |  mAP  |                                                                                            Download                                                                                            |
| :-------------------------------------------------: | :---------------------: | :---: | :--------: | :---: | :-----------: | :-------------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [Faster R-CNN](https://arxiv.org/abs/1506.01497)   | ResNet50<br>(1024,1024) | le90  |     3x     |   -   |     2\*4      |      54.75      | 27.60 | [model](https://files.airvic.top/d/PETDet/ShipRSImageNet/faster/rotated_faster_rcnn_r50_fpn_3x_shiprs3_le90.pth) \| [log](https://files.airvic.top/d/PETDet/ShipRSImageNet/faster/20230618_165046.log.json) |
| [RoI Transformer](https://arxiv.org/abs/1812.00155) | ResNet50<br>(1024,1024) | le90  |     3x     |   -   |     2\*4      |      60.98      | 33.56 | [model](https://files.airvic.top/d/PETDet/ShipRSImageNet/roitrans/roi_trans_r50_fpn_3x_shiprs3_le90.pth) \| [log](https://files.airvic.top/d/PETDet/ShipRSImageNet/roitrans/20230619_183640.log.json) |
| [Oriented R-CNN](https://arxiv.org/abs/2108.05699)  | ResNet50<br>(1024,1024) | le90  |     3x     |   -   |     2\*4      |      71.76      | 51.90 | [model](https://files.airvic.top/d/PETDet/ShipRSImageNet/orcnn/oriented_rcnn_r50_fpn_3x_shiprs3_le90.pth) \| [log](https://files.airvic.top/d/PETDet/ShipRSImageNet/orcnn/20230421_171234.log.json) |
|                       PETDet                        | ResNet50<br>(1024,1024) | le90  |     3x     |   -   |     2\*4      |      75.39      | 55.65 | [model](https://files.airvic.top/d/PETDet/ShipRSImageNet/petdet/petdet_r50_fpn_3x_shiprs3_le90.pth) \| [log](https://files.airvic.top/d/PETDet/ShipRSImageNet/petdet/20240822_201409.log.json) |

## Installation
This repo is based on [mmrotate 0.x](https://github.com/open-mmlab/mmrotate) and [OBBDetection](https://github.com/jbwang1997/OBBDetection).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name petdet python=3.10 -y
conda activate petdet
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/). Pytorch 1.13.1 is recommend.

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
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
cd PETDet
pip install -v -e .
```
## Data Preparation
Download datassts:
- [FAIR1M Dataset](https://www.gaofen-challenge.com/benchmark)
- [MAR20 Dataset](https://gcheng-nwpu.github.io/)
- [ShipRSImageNet Dataset](https://github.com/zzndream/ShipRSImageNet)  
  
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
./tools/dist_test.sh configs/petdet/ \
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

## Citations

Please cite our paper in your publications if you find this repository useful.

```latex
@article{li2023petdet,
  title={PETDet: Proposal enhancement for two-stage fine-grained object detection},
  author={Li, Wentao and Zhao, Danpei and Yuan, Bo and Gao, Yue and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
