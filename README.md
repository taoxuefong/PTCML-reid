# Patch-based Tendency Camera Multi-Constraint Learning (PTCML)

Official PyTorch implementation of **Patch-based Tendency Camera Multi-Constraint Learning for Unsupervised Person Re-identification**.

## Abstract

Unsupervised person re-identification (ReID) is a task that aims to retrieve pedestrians across different cameras from unlabeled data. Existing methods rely on clustering to generate pseudo-labels, but they are inevitably noisy. Although pseudo-label refinement approaches have been presented, the essentiality of patch contours is ignored. The tendency analysis of retrieval between global and patch features has not been well investigated.

In this paper, we propose a **Patch-based Tendency Camera Multi-Constraint Learning (PTCML)** model for unsupervised person ReID:

1. **RTS (Ranking Tendency Similarity)** score — to explore the tendentious retrieval of global and patch features by gauging the distribution discrepancy of distance changes
2. **TMC (Tendency-based Mutual Complementation)** loss — to improve the quality of global and patch pseudo-labels based on RTS score
3. **ACM (Adaptive Camera Multi-Constraint)** loss — to resist camera variations with camera distribution constraint and instance constraint simultaneously

Numerous experiments on Market-1501 and MSMT17 demonstrate that our method can significantly surpass the state-of-the-art performance.

## Overview

- **RTS**: Captures the complementary tendency of increased distance between global and patch features
- **TMC**: Refines global and patch pseudo-labels based on RTS score, calibrating patch features to smooth label distributions
- **ACM**: Applies both camera distribution and instance-level constraints for cross-camera matching

## Getting Started

### Installation

```shell
cd PTCML-main
python setup.py develop
```

### Preparing Datasets

```shell
cd examples && mkdir data
```

Download [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), and [VeRi-776](https://github.com/JDAI-CV/VeRidataset) to `PTCML-main/examples/data`.

The directory should look like:

```
PTCML-main/examples/data
├── Market-1501-v15.09.15
├── MSMT17_V1
└── VeRi
```

## Training

### Training without camera labels

For Market-1501:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml.py \
-d market1501 --logs-dir $PATH_FOR_LOGS
```

For MSMT17:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml.py \
-d msmt17 --logs-dir $PATH_FOR_LOGS
```

For VeRi-776:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml.py \
-d veri -n 8 --height 256 --width 256 --eps 0.7 --logs-dir $PATH_FOR_LOGS
```

### Training with camera labels (ACM)

For Market-1501:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml_cam.py \
-d market1501 --eps 0.4 --logs-dir $PATH_FOR_LOGS
```

For MSMT17:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml_cam.py \
-d msmt17 --eps 0.6 --lam-cam 1.0 --logs-dir $PATH_FOR_LOGS
```

For VeRi-776:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_ptcml_cam.py \
-d veri -n 8 --height 256 --width 256 --eps 0.7 --logs-dir $PATH_FOR_LOGS
```

## Testing

For Market-1501:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
-d market1501 --resume $PATH_FOR_MODEL
```

For MSMT17:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
-d msmt17 --resume $PATH_FOR_MODEL
```

For VeRi-776:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
-d veri --height 256 --width 256 --resume $PATH_FOR_MODEL
```

## Acknowledgement

Some parts of the code are borrowed from [SpCL](https://github.com/yxgeee/SpCL) and [PPLR](https://github.com/yoonkicho/PPLR).

## Citation

If you find this code useful for your research, please consider citing:

```bibtex
@article{ptcml2024,
  title={Patch-based Tendency Camera Multi-Constraint Learning for Unsupervised Person Re-identification},
  author={},
  journal={},
  year={2024}
}
```
