import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def load_results(dir,model_key='L2',seq_key='eval-',score_key = "@"):
    # all csv files in the root and its subdirectories
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith("results_recall.csv")]
    
    sequence_pool = []
    model_pool = []
    matches = {}
    for file in files:
        file_Struct = file.split("/")
        model_index = np.array([i for i,field in enumerate(file_Struct) if model_key in field])[0]
        seq_index = np.array([i for i,field in enumerate(file_Struct) if seq_key in field])[0]
        score_index = np.array([i for i,field in enumerate(file_Struct) if score_key in field])[0]
        filter_model_name = ''.join(file_Struct[model_index].split("-")[:-1])
        
        if 'eval' in file_Struct[seq_index]:
            file_Struct[seq_index] = file_Struct[seq_index].split("-")
            file_Struct[seq_index][0] = 'kitti'
            file_Struct[seq_index] = "-".join(file_Struct[seq_index][:-1])
        
        filter_name = []
        filter_name.append(file_Struct[seq_index])
        seq_name = '-'.join(filter_name)
        sequence_pool.append(seq_name)
        model_pool.append(filter_model_name)
        # load csv
        df = pd.read_csv(file)

        if seq_name not in matches:
            matches[seq_name] = {filter_model_name:{file_Struct[score_index]:{'df':df,'path':file}}}
        elif filter_model_name not in matches[seq_name]:
            matches[seq_name][filter_model_name] = {file_Struct[score_index]:{'df':df,'path':file}}
        else:
            matches[seq_name][filter_model_name][file_Struct[score_index]] = {'df':df,'path':file}
    
    sequences = np.unique(sequence_pool)
    models = np.unique(model_pool)
    return matches,sequences,models






def gen_crop_fig(seqs,models,top,results,save_dir,crop_range=['10m','20m','30m','40m','50m','60m','100m','150m','200m'],size_param=15,linewidth=5,**argv):
    
    range_key = '10'
    graph_dir = save_dir # os.path.join(save_dir,f"crop")
    os.makedirs(graph_dir, exist_ok=True)
    
    
    for model in models:
    
        print("=====================================")
        model_array = {}
        for i,seq in enumerate(seqs):
            array = []
            crop_xx_axis = []
            for dist in crop_range:
                recall_array = []
                for key, value in results[seq][model].items():
                    path_structures = value['path'].split("/")
                    if dist in path_structures:
                        recall_table = results[seq][model][key]['df'][range_key]
                        recall_value = recall_table.loc[top-1]
                        recall_array.append(recall_value)
                max_value = np.array(recall_array).max()
                array.append(max_value)
                crop_value = int(dist[:-1])
                crop_xx_axis.append(crop_value)

            
            if "new_name" in argv:
                seq = argv["new_name"][i]
                
            model_array[seq] =  array#,'x':crop_xx_axis}
            
        
        df = pd.DataFrame(model_array,index=crop_xx_axis) 
        plt.figure(figsize=(10,10)) 
        sns.lineplot(data = df ,linewidth=linewidth)
        file = os.path.join(graph_dir,f'{model}Top{top}.pdf')
        print(file)
        plt.xlabel('m',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{top}',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)

        plt.ylim(0, 1)
        # turn on grid  
        plt.grid()

        plt.legend(fontsize=size_param)
        plt.savefig(file, transparent=True)   



    
if __name__ == "__main__":
    root = "/home/deep/Dropbox/SHARE/orchards-uk/v2/aa0.5/crop_evaluation"

    size_param = 15
    
    save_dir = "saved_graphs_my_model/crop_evaluation"

    results,sequences,models = load_results(root)
 
    # sota_models = ['LOGG3DNet','PointNetVLAD','OverlapTransformer','PointORCHNet']
    models = ['pointnetORCHNetMultiHead']
    
    seqs   =['kitti-orchards-aut22','kitti-orchards-sum22'] #  ['kitti-orchards-sum22',  'kitti-strawberry-june23','kitti-orchards-june23']
    
    name_seqs   =['ON22','OJ22']
    dist = '500'
    
    
    colors = ["red", "green","orange","blue","purple","brown","pink","gray","olive","cyan"]
    linestyles = [ "-.", "--","-","-", "-.", "--","-", "-.", "--","-", "-.", "--"]

    # Create directory
    
    for i in [1,5,10]:
        gen_crop_fig(seqs,models,i,results,save_dir,crop_range=['10m','50m','100m','150m','200m'],size_param=25,linewidth=4,new_name=name_seqs)
    
    
    plt.show()

  