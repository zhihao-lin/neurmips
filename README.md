# NeurMips: Neural Mixture of Planar Experts for View Synthesis
This is the official repo for PyTorch implementation of paper "NeurMips: Neural Mixture of Planar Experts for View Synthesis", CVPR 2022. 
### [Paper]() | [Data]()
![Overview](figures/overview.png)

## Prerequisites
- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA TITAN RTX
- Python package manager `conda`
## Setup
### Datasets
Download and put datasets under folder `data` by running:
```
bash run/dataset.sh
```
For more details of file structure and camera convention, please refer to [Dataset](dataset.md). 
### Environment
```
conda env create -f environment.yml
conda activate neurmips
```
- CUDA extension
- pretrained models
## Usage 
- Training 
- Distillation
- CUDA acceleration

BibTex