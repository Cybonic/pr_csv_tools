import os,sys
import pandas as pd
import numpy as np


def load_results(dir,model_key='L2',seq_key='eval'):
    # all csv files in the root and its subdirectories
    assert os.path.isdir(dir), "ERROR - {} is not a valid directory".format(dir)
    print("INFO - Loading results from {}".format(dir))
    
    #files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.startswith(dir) and file.endswith("results_recall.csv")]
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if  file.endswith("results_recall.csv")]
    print("INFO - Found {} files".format(len(files)))
    
    matches = {}
    for file in files:
        file_Struct = file.split("/")
        model_index = np.array([i for i,field in enumerate(file_Struct) if model_key in field])[0]
        seq_index = np.array([i for i,field in enumerate(file_Struct) if seq_key in field])[0]
        filter_model_name = file_Struct[model_index].split("-")[0]
        
        if 'eval' in file_Struct[seq_index]:
            file_Struct[seq_index] = file_Struct[seq_index].split("-")
            file_Struct[seq_index][0] = 'kitti'
            file_Struct[seq_index] = "-".join(file_Struct[seq_index][:-1])
        
        filter_name = []
        filter_name.append(file_Struct[seq_index])
        seq_name = '-'.join(filter_name)
        # load csv
        df = pd.read_csv(file)

        if seq_name not in matches:
            matches[seq_name] = {filter_model_name:df}
        else:
            matches[seq_name][filter_model_name] = df
    return matches