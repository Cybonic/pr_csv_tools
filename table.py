import os
import pandas as pd
from tabulate import tabulate
import numpy as np

def generate_table(table,columns,rows,ranges,k_cand,res=3):
    table_recall = pd.DataFrame(columns=columns, index=rows)
  
    n_sub_grupes = len(ranges)
    n_seq = len(seqs[:-2])
    rows_array = []
    for model in models:
        tuple_list = []
        rows_local = []
        for ktop in k_cand:
            seq_array = []
            for seq in seqs[:-2]:
                
                for label in ranges:
                    ktop_str = str(ktop)
                    ktop_idx = ktop-1
                    if ktop == -1:
                        ktop_str = "1%"
                        ktop_idx = ktop
                    sq = seq.split("-")
                    sq_name = sq[0][0]+sq[1][0]+sq[1][-2:]

                    label_proxy = label+"m"
                    if label == ranges[-1]:
                        label_proxy = "row"
                    
                    #if label not in  table[model][seq]:
                    #    continue
                    tuple_list.append(label_proxy)
                    value = table[model][seq][label].values[ktop_idx]
                    rows_local.append(round(value,res))
            
            seq_array = np.array(rows_local).reshape(n_seq,n_sub_grupes)
            seq_mean = np.mean(seq_array,axis=0)
            seq_std = np.std(seq_array,axis=0)
            
            rows_local.extend(seq_mean)
            for label in ranges:
                label_proxy = label+"m"
                if label == "100":
                    label_proxy = "row"
                tuple_list.append(label_proxy)

            #rows_local.extend(seq_std)
            #for label in ranges:
            #    label_proxy = label+"m"
            #    if label == "100":
            #        label_proxy = "row"
            #    tuple_list.append(label_proxy)
        rows_array.append(rows_local)

    #header = pd.MultiIndex.from_tuples(tuple_list)
    df = pd.DataFrame(rows_array, columns=tuple_list, index=models)

    return df


def compile_results(table,columns,rows,label,top_k,res=3):
    table_recall = pd.DataFrame(columns=columns, index=rows)
    row_idx = top_k-1
    for model in models:
        for seq in seqs[:-2]:
            value = table[model][seq][label]
            table_recall.loc[model][seq] = round(value.values[row_idx],res)
            # table_recall_1percent.loc[model][seq] = round(value.values[-1],2)
    
        table_recall.loc[model]["Mean"] = round(table_recall.loc[model].mean(),res)
        table_recall.loc[model]["Std"] = round(table_recall.loc[model].std(),res)

    return table_recall


def load_results(dir,model_key='L2',seq_key='eval'):
    # all csv files in the root and its subdirectories
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.startswith(dir) and file.endswith("results_recall.csv")]
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


if __name__ == "__main__":
    root = "/home/deep/Dropbox/SHARE/orchards-uk/v2/aa0.5/my_model"

    results = load_results(root)
    models =  list(results.keys())
    print(models)

    seqs = ['kitti-strawberry-june23','kitti-orchards-aut22','kitti-orchards-june23','kitti-orchards-sum22']
    #seqs = list(results[models[0]].keys())
    seqs.append("Mean")
    seqs.append("Std")
    
    topk = -1
    models = np.sort(models)
    table_r = generate_table(results,seqs,models,['1','10','20','500'],[topk],res=3)

    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    table_r.to_csv(f"recall@{topk}_paper.csv")

    latex_table = tabulate(table_r, tablefmt="latex", headers="keys",floatfmt=".3f")
    latex_table = latex_table.replace(" ", "")
    
    file = os.path.join("my_model",f"recall@{topk}_paper.tex")
    os.makedirs(os.path.dirname(file), exist_ok=True)
    f = open(file, "w")
    f.write(latex_table)
    f.close()

    print("\n")
    print(seqs)
    print(latex_table)

    exit()
    
    k_cand = -1
    for range in ['1','5','10','20','100']:
        table = compile_results(results,seqs,models,range,k_cand,res=3)
        
        k_cand_str = str(k_cand)
        if k_cand == -1:
            k_cand_str = "1%"

        table.to_csv(f"recall{range}m@{k_cand_str}.csv")
        latex_table = tabulate(table, tablefmt="latex", headers="keys")
 
        print(latex_table)
