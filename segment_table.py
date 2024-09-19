import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from utils import load_results, find_file


def generate_table(results,models,sequences,files_to_show,topk,range=10,res=3):
    
    table_recall = pd.DataFrame(columns=models, index=sequences)
    
    table =[]
    meta_table = []
    segment_ids = []
    for model in models:
        model_row = []
        columns = []
        for seq in sequences:
            for tag,scores in  results[seq][model].items():
                for tags in files_to_show:
                    
                    if not isinstance(tags,list):
                        tags = [tags]
        
                    index = find_file(scores,tags)
                    if index == -1 or len(index) == 0:
                        print(f"File {tags} not found in {model} {seq}")
                        continue
                    
                    
                    dataframe_ = scores[index[0]]['df']
                    path = scores[index[0]]['path']
                    
                    file_name = path.split('/')[-1]
                    segment_ids.append([f"{model}_{seq}_{file_name}"])
                    
                    target_column = np.asarray(dataframe_[str(range)])
                    target_cell = target_column[topk-1]
                    model_row.append(target_cell)
                    columns.append(f"{seq}_{file_name}")
        meta_table.append(columns)
        table.append(model_row)
    
    #table = np.asarray(table).T
    column_names = np.unique(np.array(meta_table),axis=0)[0]
    panda_frame = []
    #str_array = np.array([np.unique(table) for table in meta_table]).flatten()
    try:
        #unqiue_columns = np.unique(str_array,axis=0).tolist()  
        panda_frame = [pd.DataFrame(table, columns=column_names, index=models)]   
    except:
        remap_colm = []
        remap_row  = []
        for values, segment in zip(table,segment_ids):
            for seq in sequences:
                line_value = []
                line_label = []
                for i, seg in enumerate(segment):
                    if seq in seg:
                        line_value.append(values[i])
                        line_label.append(segment[i])
                        #remap_row[seq] = values
                        #remap_row[seq] = seg
        unqiue_columns = np.unique(segment_ids).tolist()
        seq_remap = {}
        #for seq in sequences:
        #    seq_bundle = [  for model in meta_table]
        #    seq_remap[seq] = 
        #array = [  for ]
        
        panda_frame = [pd.DataFrame(table, columns=unqiue_columns, index=models)]
        
        
         
    return panda_frame  






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
        assert sum(model_bool) == len(model_order), "model not found in the dataset"
    else:
        model_order = models

    table_r = generate_table(results,model_order,seq_order,files_to_show,topk,range=range,res=3)
    table_r = table_r[0] # Quick fixe
    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    # save dataframe to csv
    os.makedirs(save_dir,exist_ok=True)
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
    #root = "/home/tiago/workspace/pointnetgap-RAL/RALv2/predictions_RALv1"
    root = "/home/tiago/workspace/pointnetgap-RAL/thesis/horto_predictions"
    
    save_dir = "thesis"
    sequences = ['SJ23','ON22','OJ23','OJ22']
    sequences = ['ON23','OJ22','OJ23','ON22','SJ23','GTJ23']
     
    topk = 1
    target_range = 10
    
    files_to_show = ["recall.csv"]
    run_table(root,files_to_show,sequences,model_order=None, topk=topk, range = target_range, tag = 'global', save_dir = save_dir)

    
    
    #files_to_show = ["recall_0.csv","recall_1.csv","recall_2.csv","recall_3.csv","recall_4.csv","recall_5.csv"]
    #run_table(root,files_to_show,sequences,model_order=None, topk=topk, range = target_range, tag = 'segments', save_dir = save_dir)

