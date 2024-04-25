import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from utils import load_results, find_file







def generate_table(results,models,sequences,files_to_show,topk,range=10,res=3):
    
    table_recall = pd.DataFrame(columns=models, index=sequences)
    
    table =[]
    meta_table = []
    for model in models:
        model_row = []
        columns = []
        for seq in sequences:
            for tag,scores in  results[seq][model].items():
                for file in files_to_show:
                    index = find_file(scores,file)
                    if index == -1:
                        print(f"File {file} not found in {model} {seq}")
                        #model_row.append('-')
                        #columns.append(f"{seq}_{file}")
                        continue
                    
                    
                    dataframe_ = scores[index]['df']
                    path = scores[index]['path']
                    
                    
                    target_column = np.asarray(dataframe_[str(range)])
                    target_cell = target_column[topk-1]
                    model_row.append(target_cell)
                    columns.append(f"{seq}_{file}")
        meta_table.append(columns)
        table.append(model_row)
    
    #table = np.asarray(table)
    unqiue_columns = np.unique(np.array(meta_table),axis=0).tolist()      
    return pd.DataFrame(table, columns=unqiue_columns, index=models)   






def run_table(root,files_to_show,seq_order,model_order, topk, range, tag, save_dir,save_latex=True):
    
    results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    sequences = sequences.tolist()
    # print all models and sequences
    print(models)
    print(sequences)
    
    seq_bool = [True for seq in seq_order if seq in sequences]
    assert sum(seq_bool) == len(seq_order), "Sequence not found in the dataset"
    
    if model_order !=  None:
        model_bool = [True for item in model_order if item in models]
        assert sum(model_bool) == len(model_order), "Sequence not found in the dataset"
    else:
        model_order = models

    table_r = generate_table(results,model_order,seq_order,files_to_show,topk,range=range,res=3)

    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    # save dataframe to csv
    table_r.to_csv(os.path.join(save_dir,f"{tag}_recall_{range}m@{topk}.csv"))

    if save_latex:
        latex_table = tabulate(table_r, tablefmt="latex", headers="keys",floatfmt=".3f")
        latex_table = latex_table.replace(" ", "")
        file = os.path.join(save_dir,f"{tag}_recall_{range}m@{topk}.tex")
        os.makedirs(save_dir, exist_ok=True)
        f = open(file, "w")
        f.write(latex_table)
        f.close()

    print("\n")
    print(sequences)
    print(latex_table)
    
    return table_r
    

if __name__ == "__main__":
    root = "/home/tiago/workspace/pointnetgap-RAL/RALv2/predictions_RALv1"
    save_dir = "RALv2"
    sequences = ['SJ23','ON22','OJ23','OJ22']
    topk = 1
    target_range = 5
    
    files_to_show = ["recall.csv"]
    run_table(root,files_to_show,sequences,model_order=None, topk=topk, range = target_range, tag = 'global', save_dir = save_dir)

    
    
    files_to_show = ["recall_0.csv","recall_1.csv","recall_2.csv","recall_3.csv","recall_4.csv","recall_5.csv"]
    run_table(root,files_to_show,sequences,model_order=None, topk=topk, range = target_range, tag = 'segments', save_dir = save_dir)

