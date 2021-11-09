dataroot:r'T:\COPAINGE\TIER2\TBCRC\BK\cws-TBCRC\SET1_cws'
for d in  $dataroot/*; do
	cd $d
	python test.py --name 070221_512_HEDCIS --dataroot $d  --netG global --no_instance --label_nc 0
    cd ..
done