import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from graphs import load_results

def generate_table(table,rows,columns,ranges,k_cand,res=3):
    table_recall = pd.DataFrame(columns=columns, index=rows)
  
    n_sub_grupes = len(ranges)
   
    rows_array = []
    for model in rows:
        n_seq = 0 # len(columns[:-2])
        tuple_list = []
        rows_local = []
        for ktop in k_cand:
            seq_array = []
            for seq in columns[:-2]:
                
                for range in ranges:
                    ktop_str = str(ktop)
                    ktop_idx = ktop-1
                    if ktop == -1:
                        ktop_str = "1%" 
                        
                    sq_name = seq
            
                    label_proxy = str(range)+"m"
                    if range == ranges[-1] and len(ranges) > 1:
                        label_proxy = "row"
                    
                    #if label not in  table[model][seq]:
                    #    continue
                    
                    recall_array = []
                    
                    # check if the model is in the table
                    if not model in table[seq]:
                        continue
                    
                    tuple_list.append(label_proxy)
                    
                    # count the number of sequences
                    n_seq+=1
                    
                    for key, value in table[seq][model].items():
                
                        recall_table = table[seq][model][key]['df'][str(range)]
                        recall_value = np.array(recall_table)
                        print(recall_value)
                        if ktop == -1:
                            recall_array.append(recall_value[-2])
                        else:
                            recall_array.append(recall_value[ktop_idx])
                    
                    max_value = np.array(recall_array).max()
                    rows_local.append(round(max_value,res))
            
            seq_array = np.array(rows_local).reshape(n_seq,n_sub_grupes)
            seq_mean = np.round(np.mean(seq_array,axis=0),4)
            seq_std = np.std(seq_array,axis=0)
            
            rows_local.extend(seq_mean)
            for label in ranges:
                label_proxy = str(label)+"m"
                if label == "100":
                    label_proxy = "row"
                tuple_list.append(label_proxy)
        rows_array.append(rows_local)

    #header = pd.MultiIndex.from_tuples(tuple_list)
    df = pd.DataFrame(rows_array, columns=tuple_list, index=rows)

    return df




if __name__ == "__main__":
    root = "/home/tbarros/workspace/pointnetgap-RAL/thesis/horto_predictions"
    
    save_dir = "thesis_v1"
    
    sequences = ['00','02','05','06','08']   
    
    sequences = ['ON23','OJ22','OJ23','ON22','SJ23','GTJ23']
    
    model_order = [ 'PointNetPGAP',
                    'PointNetPGAPLoss',
                    'PointNetVLAD', # -segment_loss-m0.5',
                    'LOGG3D', # -segment_lossM0.1-descriptors',
                    'overlap_transformer', #-segment_loss-m0.5',
                   ]
    
    
    new_model = [#'PointNetGAP',
                 'PointNetPGAP',
                 'PointNetPGAP_SCL',
                 #'PointNetGeM','PointNetMAC',
                 'PointNetVLAD','LOGG3D','OverlapTransformer']
    
    ROWS = [#'PointNetGAP',
            'PointNetPGAP',
            'PointNetPGAP_SCL',
            #'PointNetGeM','PointNetMAC',
            'PointNetVLAD','LOGG3D','OverlapTransformer']
    
    results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    #models =  list(results.keys())
    print(models)
    sequences = sequences.tolist()


    #sequences.append("Mean")
    #sequences.append("Std")
    
    topk = 1


    table_r = generate_table(results,models,sequences,[1,5,10,20],[topk],res=3)

    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    table_r.to_csv(f"recall@{topk}_paper.csv")

    latex_table = tabulate(table_r, tablefmt="latex", headers="keys",floatfmt=".3f")
    latex_table = latex_table.replace(" ", "")
    
    file = os.path.join(save_dir,f"recall@{topk}.tex")
    os.makedirs(save_dir, exist_ok=True)
    f = open(file, "w")
    f.write(latex_table)
    f.close()

    print("\n")
    print(sequences)
    print(latex_table)


