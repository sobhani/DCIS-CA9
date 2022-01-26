# Spatial interplay of tissue hypoxia and T-cell regulation in ductal carcinoma in situ

<p align="center">
<img src="https://github.com/sobhani/DCIS-CA9/blob/main/Images/DCIS.jpg"
width=50% height=50%>
</p>

<p align="center">
<img src="https://github.com/sobhani/DCIS-CA9/blob/main/Images/pipline.png"
width=50% height=50%>
</p>

The deep learning framework used to analyze pure DCIS and IDC/DCIS samples in this study consists of four parts: 
* Tiling: to convert a raw microscopy image into 2000x2000 JPEG tiles. 
* Tissue segmentation: to segment viable tissue area from a dual staining CA9 FOXP3 slide.
* Cell detection: identifying cell nucleus
* Cell classification: predicting the class of an identified cell (Stroma, FOXP3+ lymphocyte, FOXP3- lymphocyte, CA9+ epithelial cells and CA9- epithelial cells)
* Item DCIS segmentation: to detect and segment DCIS

* For the detailed Python-TensorFlow virtual envs (Linux) for the SC-CNN detection and classification you can refer to   https://github.com/qalid7/compath 

We describe a novel deep-learning model for the simultaneous detection and segmentation of DCIS ducts from IHC images. 
An improved Generative Adversarial Networks (GANs) architecture was used to train a deep learning model capable of delineating DCIS duct regions from surrounding tissue.

To use this model, follow the steps bellow.
# Installation
* Install PyTorch and dependencies from http://pytorch.org
* Install python libraries .
* Clone this repo:

```bash
git clone (https://github.com/sobhani/DCIS-CA9) 
cd DCIS_segmentation
```

# Training
Train a model at 1024 x 1024 resolution:

```bash
python train.py --name [NAME_OF_PROJECT] --dataroot [PATH_TO_DATA] --no_instance
```

To view training results, please checkout intermediate results in
```bash
./checkpoints/[NAME_OF_PROJECT]/web/index.html.
```

# Testing
* Test the model:

```bash
python test.py --name [NAME_OF_PROJECT] --dataroot [PATH_TO_DATA] --results_dir [PATH_TO_SAVE] --no_instance
```

* the trained model is not uploaded due to the size. You can contact the authours to have access to the final generator model after training. Better way to use the trained model is to pull the docker image.

# Docker
We provide the pre-built Docker image and Dockerfile that can run this code repo. See Dockerfile and get the image by: docker pull 
