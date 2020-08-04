# Series-Photo-Selection

  Implementation of Paper：LEARNING MULTI-SCALE ATTENTIVE FEATURES FOR SERIES PHOTO SELECTION //ICASSP2020   
  
## Introduction
  In this paper, we develop a novel deep CNN architecture that aggregates multi scale features from different network layers, in order to capture the subtle differences between series photos. To reduce the risk of redundant or even interfering features, we introduce the spatial-channel self-attention mechanism to adaptively recalibrate the features at each layer, so that informative features can be selectively emphasized and less useful ones suppressed.   
  
## Structure
![image](https://github.com/zhenshen-mla/Series-Photo-Selection/blob/master/examples/structure.png)  
   Overview of the proposed deep network architecture for series photo selection. The backbone of our learning algorithm is an end-to-end deep CNN architecture. In our study, features from different network layers are jointly leveraged to help capture the subtle differences between series photos.  
  
## Models
  * `/models/PAUnit.py`: implementation of Parallel Attention Unit;  
  * `/models/ResNet18.py`: baseline with resnet18 backbone;  
  * `/models/ResNet50.py`: baseline with resnet50 backbone;  
  * `/models/ResNet101.py`: baseline with resnet101 backbone;  
  * `/models/Inceptionv3.py`: baseline with Inception network;  
  * `/models/PAUnit.py`: resnet50 network with PAUnit;   
  
## Requirements  

  Python >= 3.6  
  numpy  
  PyTorch >= 1.0  
  torchvision  
  tensorboardX  
  sklearn  
  

## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/zhenshen-mla/Series-Photo-Selection.git   
    ```   
    ```
    cd Series-Photo-Selection  
    ```
  2. For custom dependencies:   
    ```
    pip install matplotlib tensorboardX sklearn   
    ```
## Usage   
  1. Download the dataset([Automatic triage for a photo series](https://phototriage.cs.princeton.edu/dataset.html)) and configure the data path.   
  2. Train the baseline with ResNet backbone:  
  ``` python train_resnet.py ```  
  3. Train the network with PAUnit:  
  ``` python train_pau.py ```  
