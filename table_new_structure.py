import os
import pandas as pd
from tabulate import tabulate
import numpy as np
def load_results(dir,row_field_idx=9,column_field_idx=8):
    # all csv files in the root and its subdirectories
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith("results_recall.csv")]

    matches = {}
    for file in files:
        file_Struct = file.split("/")
        filter_model_name = file_Struct[row_field_idx].split("-")[0]
        
        column_filed = file_Struct[column_field_idx]

        # load csv
        df = pd.read_csv(file)


        # filter column name 
        if 'eval' in column_filed:
            column_filed = column_filed.split("-")
            column_filed[0] = 'kitti'
            column_filed = "-".join(column_filed[:-1])
        if filter_model_name not in matches:

            matches[filter_model_name] = {column_filed:[df]}
        else:
            if column_filed not in matches[filter_model_name]:
                matches[filter_model_name][column_filed] = [df]
            else:
                matches[filter_model_name][column_filed].append(df)
    return matches

def print_results(dir,query,row_field_idx=9,column_field_idx=8,depth_field_idx = 10, distance=0, res=3, knn=0):
    # all csv files in the root and its subdirectories
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith("results_recall.csv")]

    matches = {}
    for file in files:
        file_Struct = file.split("/")
        
        if query not in file_Struct:
            continue
        
        filter_model_name = file_Struct[row_field_idx].split("-")[0]
        
        column_filed = file_Struct[column_field_idx]
        
        depth = file_Struct[depth_field_idx]
        
        
        # load csv
        df = pd.read_csv(file)

        print(f"{filter_model_name} {column_filed} {depth}: {round(df[distance].values[knn],res)}")
    return matches

def compile_results(table,rows,columns,distance,res,knn):
    
    table_recall1 = pd.DataFrame(columns=columns, index=rows)
    decimal_res = res
    for row in rows:
        for column in columns[:-2]:
            value = table[row][column]
            selected = [round(array[distance].values[knn],decimal_res) for array in value]
            table_recall1.loc[row][column] = np.mean(selected)
    
        table_recall1.loc[row]["Mean"] = round(table_recall1.loc[row].mean(),decimal_res)
        table_recall1.loc[row]["Std"] = round(table_recall1.loc[row].std(),decimal_res)
    
    return table_recall1

def print_multi_dim_table(table,rows,columns,distance,res,knn):
    for row in rows:
        for column in columns:
            value = [value[distance].values[knn] for value in table[row][column]]
            print(f"{row} {column} {value}")
        print("\n")

if __name__ == "__main__":
    root = "/home/deep/Dropbox/SHARE/orchards-uk/v2/aa0.5"

    label = '10'
    print_results(root,"kitti-orchards-sum22",row_field_idx=9,column_field_idx=8, depth_field_idx = 10, distance=label, res=3, knn=0)
    
    
    
    
    
    results = load_results(root,row_field_idx=9,column_field_idx=10)
    
   
    rows =  list(results.keys())
    columns = list(results[rows[0]].keys())
    columns.append("Mean")
    columns.append("Std")
    
    
    resolution =3
    
    
    #for k in [1,5,10,20]:
    #    print_multi_dim_table(results,rows,columns,label,resolution,k-1)
        
    for k in [1,5,10,20]:
        table = compile_results(results,rows,columns,label,resolution,k-1)
        table.to_csv(f"recall{label}m@{k}.csv")
        latex_table = tabulate(table, tablefmt="latex", headers="keys")
        print(f"=====================================================\n")
        print(latex_table)
   