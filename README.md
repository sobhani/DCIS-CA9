# Spatial interplay of tissue hypoxia and T-cell regulation in ductal carcinoma in situ
The deep-learning pipeline for digital pathology image analysis is available for noncommercial research purposes.

The deep learning framework used to analyze pure DCIS and IDC/DCIS samples in this study consists of four parts: 
* Item 	Tiling: to convert a raw microscopy image into 2000x2000 JPEG tiles.
* Item Tissue segmentation: to segment viable tissue area from a dual staining CA9 FOXP3 slide.
* Item 	Cell detection: identifying cell nucleus
* Item 	Cell classification: predicting the class of an identified cell (Stroma, FOXP3+ lymphocyte, FOXP3- lymphocyte, CA9+ epithelial cells and CA9- epithelial cells)
* Item 	DCIS segmentation: to detect and segment DCIS
We describe a novel deep-learning model for the simultaneous detection and segmentation of DCIS ducts from IHC images. 
An improved Generative Adversarial Networks (GANs) architecture was used to train a deep learning model capable of delineating DCIS duct regions from surrounding tissue.


