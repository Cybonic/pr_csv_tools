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


def gen_range_fig(seqs,models,top,results,save_dir,size_param=15,linewidth=5,**args):
    
    
    index = [1,5,10,15,20,30,40,50] # ranges
    
    graph_dir = os.path.join(save_dir,f"top")
    os.makedirs(graph_dir, exist_ok=True)
    
    for seq in seqs:
        model_array = {}
        
        for model in models:
            array = []
            
            print(model,seq)
            for dist in index:
                recall_array = []
                for key, value in results[seq][model].items():
                
                    recall_table = results[seq][model][key]['df'][str(dist)]
                    recall_value = recall_table.loc[top-1]
                    recall_array.append(recall_value)
                max_value = np.array(recall_array).max()
                array.append(max_value)

            model_array[model] =  array#,'x':crop_xx_axis}
    
        df = pd.DataFrame(model_array, index=index)
        plt.figure(figsize=(10,10))
        
        if 'colors' in args:
            colors = args['colors']
            linestyles = args['linestyles']
            for i,model in enumerate(models):
                sns.lineplot(data=df[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i],label=model)
        else:
            sns.lineplot( data=df,linewidth=linewidth)

        file = os.path.join(graph_dir,f'{seq}Top{top}.pdf')
        plt.xlabel('Range[m]',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{top}',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        # turn on grid  
        plt.grid()
        plt.legend(fontsize=size_param)
        plt.savefig(file, transparent=True)




def gen_crop_fig(seqs,models,top,results,save_dir,crop_range=['10m','20m','30m','40m','50m','60m','100m','150m','200m'],size_param=15,linewidth=5):
    
    range_key = '10'
    graph_dir = os.path.join(save_dir,f"crop")
    os.makedirs(graph_dir, exist_ok=True)
    
    
    for seq in seqs:
        model_array = {}
        dataframe_collection = {} 
        crop_xx_axis = []
        array = []
        for dist in crop_range:
            for model in models:
                recall_array = []
                print(model,seq)
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

            model_array[model] =  array#,'x':crop_xx_axis}
            
        
        df = pd.DataFrame(model_array,index=crop_xx_axis) 
        
        plt.figure(figsize=(10,10))
         
        sns.lineplot(data = df ,linewidth=linewidth)

 
        file = os.path.join(graph_dir,f'{seq}Top{top}.pdf')
        plt.xlabel('m',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{top}',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)

        plt.ylim(0, 1)
        # turn on grid  
        plt.grid()

        plt.legend(fontsize=size_param)
        plt.savefig(file, transparent=True)   




def gen_top25_fig(seqs,models,dist,results,save_dir,size_param=15,linewidth=5,**args):
    
    graph_dir = os.path.join(save_dir,f"range")
    os.makedirs(graph_dir, exist_ok=True)
    
    top = np.arange(1,25,1)
    for seq in seqs:
        model_array = {}        
        for model in models:
            array = []
            recall_array = []
            key_array = []
            for key, value in results[seq][model].items():
                key_array.append(key)
                recall_table = results[seq][model][key]['df'][str(dist)]
                recall_value = recall_table.loc[top-1]
                recall_array.append(recall_value)
            
            max_key_idx = np.array(key_array).argmax()
            max_value = np.array(recall_array)[max_key_idx]
            model_array[model] =  max_value#,'x':crop_xx_axis}
    
        df = pd.DataFrame(model_array, index=top)
        plt.figure(figsize=(10,10))
        if 'colors' in args:
            colors = args['colors']
            linestyles = args['linestyles']
            for i,model in enumerate(models):
                sns.lineplot(data=df[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i],label=model)
        else:
            sns.lineplot( data=df,linewidth=linewidth)
        
        file = os.path.join(graph_dir,f'{seq}_range{dist}m.pdf')
        plt.xlabel('Top k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel('Recall@k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        plt.legend(fontsize=size_param)
        plt.savefig(file,transparent=True)
    
if __name__ == "__main__":
    root = "/home/deep/Dropbox/SHARE/orchards-uk/v2/aa0.5/baselines"

    size_param = 15
    
    save_dir = "saved_graphs_my_model/baselines"

    results,sequences,models = load_results(root)
 
    
    sota_models = ['LOGG3DNet','PointNetVLAD','OverlapTransformer','PointORCHNet']
    baseline_models = ['PointNetSPoC','PointNetMAC','PointNetGeM','PointORCHNet']
    
    #seqs   = ['kitti-orchards-sum22', 'kitti-orchards-aut22' 'kitti-strawberry-june23','kitti-orchards-june23']
    
    dist = '500'
    
    
    colors = ["red", "green","orange","blue","purple","brown","pink","gray","olive","cyan"]
    linestyles = [ "-.", "--","-","-", "-.", "--","-", "-.", "--","-", "-.", "--"]

    # Create directory
    
    #for i in [1,5,10]:
    #    gen_crop_fig(seqs,models,i,results,save_dir,crop_range=['10m','50m','100m','150m','200m'],size_param=25,linewidth=4)
    
    for i in [1,5,10]:
        gen_range_fig(sequences,baseline_models,i,results,save_dir,size_param=25,linewidth=4,colors=colors,linestyles=linestyles)
    
    for i in [10,20,500]:
   
        gen_top25_fig(sequences,baseline_models,str(i),results,save_dir,size_param=25,colors=colors,linestyles=linestyles)
    
    
    plt.show()

  