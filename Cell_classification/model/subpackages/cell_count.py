import os
import pandas as pd
from collections import OrderedDict

def cell_count(cell_csv_path, output_path, file_name=None, classes=('f', 'l', 't', 'o')):
    if file_name is None:
        file_name = os.path.basename(cell_csv_path)

    cellPos = pd.read_csv(cell_csv_path)
    
    cell_counts = [sum(cellPos.loc[:, 'class']==c) for c in classes]
    total = sum(cell_counts)
    cell_percentages = [100*count/total if total != 0 else 0 for count in cell_counts]
    
    column_names = ['FileName']+['#'+c for c in classes]+['%'+c for c in classes]
    row = [file_name]+cell_counts+cell_percentages
    
    M_f = pd.DataFrame(data=OrderedDict(zip(column_names, row)), index=[0])
    
    M_f.to_csv(output_path, index=False)
