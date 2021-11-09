# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:45:29 2021

@author: fsobhani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:18:46 2021

@author: fsobhani
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:06:29 2021

@author: fsobhani
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:02:42 2021

@author: fsobhani
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:55:09 2021

@author: fsobhani
"""

import os

import numpy as np
import cv2
import math
from skimage import io
import pandas as pd
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.viewer
from skimage.io import imsave
import itertools
from PIL import Image

from skimage import io


def calculate_magnitude(od):
	channel = cv2.split(od)
	square_b = cv2.pow(channel[0], 2)
	square_g = cv2.pow(channel[1], 2)
	square_r = cv2.pow(channel[2], 2)
	square_bgr = square_b + square_g + square_r
	magnitude = cv2.sqrt(square_bgr)

	return magnitude;


def normaliseOD(od, magnitude):
	channels = cv2.split(od)  # , (channels[0],channels[1],channels[2]))

	od_norm_b = cv2.divide(channels[0], magnitude);
	od_norm_g = cv2.divide(channels[1], magnitude);
	od_norm_r = cv2.divide(channels[2], magnitude);

	od_norm = cv2.merge((od_norm_b, od_norm_g, od_norm_r))

	return od_norm;


def clean_artifact(cws_img, image_path_full):
	I = cws_img.transpose()
	k, width, height = I.shape
	I = I.reshape(k, width * height);
	I = np.float32(I)

	od = cv2.max(I, 1)

	grey_angle = 0.2;

	magnitude_threshold = 0.0001;
	# 0.05
	channels = cv2.split(od);
	#
	magnitude = np.zeros(od.shape)
	#
	background = 245
	#
	channels[0] /= background

	od = cv2.merge(channels)

	od = cv2.log(od)

	od *= (1 / cv2.log(10)[0])

	od = -od
	od = od.reshape(3, width, height).transpose()
	magnitude = calculate_magnitude(od)

	tissue_and_artefact_mask = (magnitude > magnitude_threshold);

	od_norm = normaliseOD(od, magnitude);

	chan = cv2.split(od_norm)

	grey_mask = (chan[0] + chan[1] + chan[2]) >= (math.cos(grey_angle) * cv2.sqrt(3)[0])

	other_colour_mask = (chan[2] > chan[1]) | (chan[0] > chan[1])

	mask = grey_mask | other_colour_mask

	mask = (255 - mask) & tissue_and_artefact_mask
	mask1 = mask.astype(np.int8)

	clean = cv2.bitwise_and(cws_img, cws_img, mask=mask1)
	clean = cv2.bitwise_not(clean)

	clean = cv2.bitwise_and(clean, clean, mask=mask1)
	clean = cv2.bitwise_not(clean)

	write_mask1 = mask.astype(np.uint8) * 255

	return (clean, write_mask1)


#
# cws_path= r'Z:\Rob\DCIS_Scoring\cws\MDA-011_D8_1_1565490.svs'
# results_dir = r'Z:\Rob\ALMA-test_result\MDA-011_D8_TBCRC_ALL_nc\Postprocess'
# DCIS_path = r'Z:\Rob\ALMA-test_result\MDA-011_D8_TBCRC_ALL_nc\resizemask\MDA-011_D8_1_1565490.svs'


cws_path = r'Z:\Rob\DCIS_Scoring\V2_Datafor19oct_meeting\cws'
DCIS_path = r'Z:\Rob\ALMA-test_result\04102021-40WSI\ORG\mask_2000r - Copy'
resizedir = r'Z:\Rob\ALMA-test_result\04102021-40WSI\test\resize'
results_dir = r'Z:\Rob\ALMA-test_result\04102021-40WSI\test\result'

for slide in os.listdir(cws_path):
	os.makedirs(os.path.join(resizedir, slide), exist_ok=True)
	os.makedirs(os.path.join(results_dir, slide), exist_ok=True)

	print(slide)
	for cws_name in os.listdir(os.path.join(cws_path, slide)):
		print('cws_name', cws_name)
		if cws_name.startswith('Da') is True:
			image_path_full = os.path.join(cws_path, slide, cws_name)
			if cws_name in os.listdir(os.path.join(DCIS_path, slide)):
				mask_path_full = os.path.join(DCIS_path, slide, cws_name)
				DCIS_mask_o = io.imread(mask_path_full)
				cws_img = io.imread(image_path_full)
				cws_img_s = cws_img.shape
				DCIS_mask_res = skimage.transform.resize(DCIS_mask_o, (cws_img_s[0], cws_img_s[1]))
				imsave(os.path.join(resizedir, slide, cws_name), DCIS_mask_res)

				# DCIS_mask = io.imread(os.path.join(resizedir,slide, cws_name))[:,:,0]
				DCIS_mask2 = io.imread(os.path.join(resizedir, slide, cws_name))[:, :, 0]

				img, mask = clean_artifact(cws_img, image_path_full)
				# imsave(os.path.join(results_dir,slide, cws_name[:-4] +'_m.jpg'), mask)
				img, post_img_mask = clean_artifact(cws_img, mask_path_full)
				# imsave(os.path.join(results_dir,slide, cws_name[:-4]+'_postmask.jpg'), post_img_mask)
				post_img_mask2 = cv2.bitwise_and(mask, DCIS_mask2)
				imsave(os.path.join(results_dir, slide, cws_name), post_img_mask2)