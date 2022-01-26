# Spatial interplay of tissue hypoxia and T-cell regulation in ductal carcinoma in situ
The deep-learning pipeline for digital pathology image analysis is available for noncommercial research purposes.

The deep learning framework used to analyze pure DCIS and IDC/DCIS samples in this study consists of four parts: 
* Tiling: to convert a raw microscopy image into 2000x2000 JPEG tiles. 
* Tissue segmentation: to segment viable tissue area from a dual staining CA9 FOXP3 slide.
* Cell detection: identifying cell nucleus
* Cell classification: predicting the class of an identified cell (Stroma, FOXP3+ lymphocyte, FOXP3- lymphocyte, CA9+ epithelial cells and CA9- epithelial cells)
* Item DCIS segmentation: to detect and segment DCIS
* 
* for the detailed Python-TensorFlow virtual envs (Linux) for the SC-CNN detection and classification you can refer to   https://github.com/qalid7/compath 

We describe a novel deep-learning model for the simultaneous detection and segmentation of DCIS ducts from IHC images. 
An improved Generative Adversarial Networks (GANs) architecture was used to train a deep learning model capable of delineating DCIS duct regions from surrounding tissue.

To use this model, follow the steps bellow.
# Installation
*item Install PyTorch and dependencies from http://pytorch.org
*item Install python libraries .
*item Clone this repo:

'''bash
copy.each(git clone https://github.com/sobhani/DCIS-CA9 //
cd DCIS_segmentation)
'''

# Testing
* Test the model:
python test.py --name [NAME_OF_PROJECT] --dataroot [PATH_TO_DATA] --results_dir [PATH_TO_SAVE] --no_instance

*item the trained model is not uploaded due to the size. You can contact the authours to have access to the final generator model after training. Better way to use the trained model is to pull the docker image.

# Docker
We provide the pre-built Docker image and Dockerfile that can run this code repo. See Dockerfile and get the image by: docker pull 
