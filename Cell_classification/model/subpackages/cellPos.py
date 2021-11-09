import numpy as np
import pickle
import os
import math
import pandas as pd

def merge_csv_files(wsi_path, results_dir, output_csv):
    if not os.path.isdir(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
    param = pickle.load(open(os.path.join(wsi_path, 'param.p'), 'rb'))
    print(param)

    slide_dimension = np.array(param['slide_dimension']) / param['rescale']
    slide_h = slide_dimension[1]
    slide_w = slide_dimension[0]
    cws_read_size = param['cws_read_size']
    cws_h = cws_read_size[0]
    cws_w = cws_read_size[1]
    divisor = np.float64(16)

    # Initialize Pandas Data Frame
    cellPos = pd.DataFrame(columns=['class', 'x', 'y'])
    iter_tot_tiles = 0
    for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
        for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
            # print('Processing Da_' + str(iter_tot_tiles))
            start_h = h * cws_h
            start_w = w * cws_w

            if os.path.isfile(os.path.join(results_dir, 'Da' + str(iter_tot_tiles) + '.csv')):
                csv = pd.read_csv(os.path.join(results_dir, 'Da' + str(iter_tot_tiles) + '.csv'))
                csv.columns = ['class', 'x', 'y']
                csv.x = csv.x + start_w
                csv.y = csv.y + start_h
                # detection = np.divide(np.float64(detection), divisor)
                cellPos = cellPos.append(csv)

            iter_tot_tiles += 1

    cellPos.x = np.round(np.divide(np.float64(cellPos.x), divisor)).astype('int')
    cellPos.y = np.round(np.divide(np.float64(cellPos.y), divisor)).astype('int')
    cellPos.loc[cellPos.x == 0, 'x'] = 1
    cellPos.loc[cellPos.y == 0, 'y'] = 1
    cellPos.to_csv(output_csv, index=False)
