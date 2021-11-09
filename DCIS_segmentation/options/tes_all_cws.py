import os

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:47:25 2021

@author: fsobhani
"""

import os
from glob import glob
import shutil

rootdir = r'Z:\Rob\2021\DCIS-Segemenatiom\cws\Unprocessed_S2_cws\1'

for folder in glob(rootdir+"/*/"):
    print(folder)
    if not os.path.isdir(folder +'test_A'):
        os.mkdir(folder + 'test_A')
    source = folder
    dest =  folder + r'test_A/'
    for Da in os.listdir(source):
        if Da.endswith('.jpg') and Da.find("Slide")==-1 and Da.find("Ss1")==-1:
            shutil.move(source + Da, dest)
            print(Da)




# for d in $root/*
# do
# 	mkdir $d/$t
# 	cd $d
# 	mv 'Da'* $t/
# 	cd ..
# done


#dataroot:r'T:\COPAINGE\TIER2\TBCRC\BK\cws-TBCRC\SET1_cws'
# for d in  $dataroot/*
# do:
# 	cd $d
#   python test.py --name 070221_512_HEDCIS --dataroot $d  --netG global --no_instance --label_nc 0
#   cd ..
#done



