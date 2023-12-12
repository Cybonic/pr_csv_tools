import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def load_results(dir,model_key='L2',seq_key='eval',score_key = "@"):
    # all csv files in the root and its subdirectories
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith("results_recall.csv")]
    matches = {}
    for file in files:
        file_Struct = file.split("/")
        model_index = np.array([i for i,field in enumerate(file_Struct) if model_key in field])[0]
        seq_index = np.array([i for i,field in enumerate(file_Struct) if seq_key in field])[0]
        score_index = np.array([i for i,field in enumerate(file_Struct) if score_key in field])[0]
        filter_model_name = file_Struct[model_index].split("-")[:-1]
        
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


def gen_range_fig(seqs,models,top,results,save_dir,size_param=15,linewidth=5):
    
    graph_dir = os.path.join(save_dir,f"top")
    os.makedirs(graph_dir, exist_ok=True)
    index = [1,5,10,15,20,30,40,50]
    for seq in seqs:
        array = {}
        for model in models:
            dist_array = []
            for dist in index:
                print(model,seq)
                value = results[seq][model][str(dist)][top-1]
                dist_array.append(value)
            array[model] = dist_array
    
        df = pd.DataFrame(array, index=index)
        plt.figure(figsize=(10,10))
        sns.lineplot( data=df,linewidth=linewidth)
        #plt.title(f'{dist}m {seqs[0]}')
        
        file = os.path.join(graph_dir,f'{seq}Top{top}.pdf')
        plt.xlabel('Range[m]',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{top}',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        # turn on grid  
        plt.grid()
        # change background color to light blue
        #plt.gca().set_facecolor('#EFF7FC')
        # resize legend size
        plt.legend(fontsize=size_param)
        plt.savefig(file, transparent=True)
        
        

def gen_top25_fig(seqs,models,dist,results,save_dir,size_param=15,linewidth=5):
    
    graph_dir = os.path.join(save_dir,f"range")
    os.makedirs(graph_dir, exist_ok=True)
    
    top = np.arange(1,25,1)
    for seq in seqs:
        array = {}
        for model in models:
            print(model,seq)
            value = results[seq][model][dist]
            array[model] = value.values[top-1]
    
        df = pd.DataFrame(array, index=top)
        plt.figure(figsize=(10,10))
        sns.lineplot( data=df,linewidth=linewidth)
        #plt.title(f'{dist}m {seqs[0]}')
        
        file = os.path.join(graph_dir,f'{seq}_range{dist}m.pdf')
        plt.xlabel('Top k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel('Recall@k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        plt.legend(fontsize=size_param)
        plt.savefig(file,transparent=True)
    
if __name__ == "__main__":
    root = "/home/deep/Dropbox/SHARE/orchards-uk/v2/aa0.5/crop_evaluation"

    size_param = 15
    
    save_dir = "saved_graphs_my_model/crop_evaluation"

    results = load_results(root)
    models =  list(results.keys())
    
    #models = ['LOGG3DNet','PointNetVLAD','OverlapTransformer','ORCHNet','PointNetGeM']
    models = ['pointnetORCHNetMultiHead']
    seqs   = ['kitti-orchards-sum22'] # 'kitti-orchards-aut22' 'kitti-strawberry-june23','kitti-orchards-june23',
    
    dist = '500'
    
    # Create directory
    
    for i in [1,5,10]:
        gen_range_fig(seqs,models,i,results,save_dir,size_param=25,linewidth=4)
    
    for i in [10,20,500]:
        gen_top25_fig(seqs,models,str(i),results,save_dir,size_param=25)
    
    
    plt.show()

  