import os,sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def find_file(data_struct:list,file:str):
    for i,score in enumerate(data_struct):
        if score['path'].endswith(file):
            return i
    return -1


def load_results(dir,model_key='L2',seq_key='eval-',score_key = "@"):
    # all csv files in the root and its subdirectories
    """_summary_

    Args:
        dir (_type_): _description_
        model_key (str, optional): _description_. Defaults to '#'.
        seq_key (str, optional): _description_. Defaults to 'eval-'.
        score_key (str, optional): _description_. Defaults to "@".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith(".csv")]
    
    sequence_pool = []
    model_pool = []
    matches = {}
    
    
    for file in tqdm(files,total=len(files)):
        file_Struct = file.split("/")
        model_index = np.array([i for i,field in enumerate(file_Struct) if field.startswith(model_key)])
        
        # No match continue
        if len(model_index) == 0:
            continue 
        
        model_index = model_index[0]
        seq_index = np.array([i for i,field in enumerate(file_Struct) if field.startswith(seq_key)])
        # No match continue
        if len(seq_index) == 0:
            continue
        
        seq_index = seq_index[0]
        score_index = np.array([i for i,field in enumerate(file_Struct) if score_key in field ])
        # No match continue
        if len(score_index) == 0:
            continue
        
        # Clear name
        # remove set of strings from the name
        #file_Struct[seq_index] = file_Struct[seq_index].split()[-1]
            
        score_index = score_index[0]
        
        model_name = file_Struct[model_index]
        seq_name = file_Struct[seq_index]
        score_name = file_Struct[score_index]
        
        # Remove keys from the names
        model_name = model_name.replace(model_key,"")
        seq_name = seq_name.replace(seq_key,"")
        score_name = score_name.replace(score_key,"")
        
        sequence_pool.append(seq_name)
        model_pool.append(model_name)
        # load csv
        df = pd.read_csv(file)

        if seq_name not in matches:
            matches[seq_name] = {model_name:{score_name:[{'df':df,'path':file}]}}
        elif model_name not in matches[seq_name]:
            matches[seq_name][model_name] = {score_name:[{'df':df,'path':file}]}
        elif score_name in matches[seq_name][model_name]:
            matches[seq_name][model_name][score_name].append({'df':df,'path':file})
        else:
            raise ValueError(f"Error: {seq_name} {model_name} {score_name}")
    
    sequences = np.unique(sequence_pool)
    models = np.unique(model_pool)
    return matches,sequences,models