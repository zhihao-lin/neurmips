# NeurMips: Neural Mixture of Planar Experts for View Synthesis
This is the official repo for PyTorch implementation of paper "NeurMips: Neural Mixture of Planar Experts for View Synthesis", CVPR 2022. 
### [Paper]() | [Data]()
![Overview](doc/overview.png)

## Prerequisites
- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA TITAN RTX
- Python package manager `conda`
## Setup
### Datasets 
Download and put datasets under folder `data/` by running:
```
bash run/dataset.sh
```
For more details of file structure and camera convention, please refer to [Dataset](doc/dataset.md). 
### Environment
Install all python packages for training and evaluation with conda environment setup file: 
```
conda env create -f environment.yml
conda activate neurmips
```
### CUDA extension
@Hao-Yu please complete this part.

### Pretrained models (optional)
Download pretrained model weights for evaluation without training from scratch:
```
bash run/checkpoints.sh
```
## Usage 
We provide hyperparameters for each experiment in config file `configs/*.yaml`, which is used for training and evaluation. For example, `replica-kitchen.yaml` corresponds to *Replica* dataset *Kitchen* scene, and `tat-barn.yaml` corresponds to *Tanks&Temple* dataset *Barn* scene.

### Training 
Train the teacher and experts model by running:
```
bash run/train.sh [config]
# example
bash run/train.sh replica-kitchen
```
### Evaluation
### CUDA Acceleration

BibTex