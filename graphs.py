import os
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

COLORS     = ["blue","gray","red", "green","orange","brown","pink","gray","olive","purple"]
LINESTYLES = ["-","-.","--", "-.","-", "-", "-","-", "-.", "--","-", "-.", "--"]
SIZE_PARAM = 25
LINEWIDTH  = 5
   
    
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





def gen_range_fig(seqs,models,top,seq_ranges,results,save_dir,size_param=15,linewidth=5,**args):
    
    show_legend = True
    if 'show_legend' in args:
        show_legend = args['show_legend']
        
    graph_dir = os.path.join(save_dir,f"top")
    os.makedirs(graph_dir, exist_ok=True)
    
    
    models = models[::-1]   
    n_lines = len(models)
    
    colors = None
    linestyles = None
    if 'colors' in args:
        # TODO: 
        #  [] Add default color and linestyle
        #  [] Inversion of oder is required. plotting approach overlap the line 
        
        # Line colors
        colors = args['colors'][:n_lines]
        # invert order
        colors = colors[::-1]
        
        # Line styles
        linestyles = args['linestyles'][:n_lines]
        # invert order
        linestyles = linestyles[::-1]
        
        
        
    for seq,ranges in zip(seqs,seq_ranges):
        model_array = {}
        
        for model in models:
            array = []
  
            for dist in ranges:
                recall_array = []
                for key, value in results[seq][model].items():
                    recall_table = np.array(results[seq][model][key]['df'][str(dist)])

                    recall_value = recall_table[top-1]
                    recall_array.append(recall_value)
                    
                max_value = np.array(recall_array).max() # When There are more than one prediction, get the max
                array.append(max_value)

            model_array[model] =  array#,'x':crop_xx_axis}
    
        df = pd.DataFrame(model_array, index=np.array(ranges))
        plt.figure(figsize=(10,12))

        model_names = models
        if 'new_model_name' in args:
            model_names = args['new_model_name']
        
        model_names = model_names[::-1]
        
        if 'new_seq_name' in args:
            seq = args['new_seq_name']
            
        if colors != None and linestyles != None:

            for i,model in enumerate(models):
                values = np.array(df[model].values)
                index = np.array(df[model].index)
                
                if show_legend:
                    sns.lineplot(x=index, y=values,linewidth=linewidth,color=colors[i],linestyle=linestyles[i],label=model_names[i])
                else:
                    sns.lineplot(data=df[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i])
        else:
            sns.lineplot(data=df,linewidth=linewidth)

      
        str_top_graph = str(top)
        str_top_pdf   = str(top)
            
        if top == -1:
            str_top_graph = '1%'
            str_top_pdf = '1p'
                   
        file = os.path.join(graph_dir,f'{seq}-Top{str_top_pdf}.pdf')
        plt.xlabel('Range[m]',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{str_top_graph}',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        # turn on grid  
        plt.ylim(0, 1)
        plt.grid()
        if show_legend:
            plt.legend(fontsize=size_param)
        #plt.legend(fontsize=size_param)
        plt.savefig(file, transparent=True)
        print(f"Saved {file}")






def gen_top25_fig(seqs,models,dist,results,save_dir,size_param=15,linewidth=5,**args):
    
    
    
    show_legend = True
    if "show_legend" in args:
        show_legend = args["show_legend"]
    
    if show_legend:
        graph_dir = os.path.join(save_dir,"top25","w_label")
    else:
        graph_dir = os.path.join(save_dir,f"top25","no_label")
             
    os.makedirs(graph_dir, exist_ok=True)
    
    
    n_lines = len(models)
    
    colors = None
    linestyles = None
    if 'colors' in args:
        # TODO: 
        #  [] Add default color and linestyle
        #  [] Inversion of oder is required. plotting approach overlap the line 
        
        # Line colors
        colors = args['colors'][:n_lines]
        # invert order
        colors = colors[::-1]
        
        # Line styles
        linestyles = args['linestyles'][:n_lines]
        # invert order
        linestyles = linestyles[::-1]
    
            
    ## Invert order 
    models = models[::-1]
    
    top = np.arange(1,25,1)
    for seq in seqs:
        model_array = {}        
        for model in models:
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

        
        # Plot the graph
        df = pd.DataFrame(model_array, index=top)
        plt.figure(figsize=(10,12))
        
        # Original model names
        model_names = models
        
        # New model names
        if 'new_model_name' in args:
            model_names = args['new_model_name']
        
        # Invert order
        model_names = model_names[::-1]
        
        # New sequence name
        if 'new_seq_name' in args:
            seq = args['new_seq_name']
            
        if colors != None and linestyles != None:
            # Plot results for each model
            for i,model in enumerate(models):
                if show_legend: 
                    sns.lineplot(data=df[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i],label=model_names[i])
                else:
                    sns.lineplot(data=df[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i])
            
        else:
            sns.lineplot( data=df,linewidth=linewidth)
        
        file = os.path.join(graph_dir,f'{seq}_range{dist}m.pdf')
        plt.xlabel('Top k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel('Recall@k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.ylim(0, 1)
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        
        if show_legend:
            plt.legend(fontsize=size_param)
        plt.savefig(file,transparent=True)
        plt.close()



def abstract_range_fig(sequences,models,seq_ranges,results,save_dir,new_names,show_legend_flag):
    
    if show_legend_flag:
        save_dir = save_dir + "_legend"
        
    for i in [1,-1]:
        gen_range_fig(sequences,models,i,seq_ranges,results,save_dir,
                      size_param     = SIZE_PARAM,
                      linewidth      = LINEWIDTH,
                      colors         = COLORS,
                      linestyles     = LINESTYLES, 
                      new_model_name = new_names,
                      show_legend    = show_legend_flag)
        
    
if __name__ == "__main__":
    root = "/home/deep/workspace/SPCoV/predictions/iros24/"

    
    save_dir = "saved_graphs_paper_iros24v2"

    results,sequences,models = load_results(root)
    
    print(models)
    
    print('\n'.join(sequences))
    
    sota_models = ['SPCov3DCOVfcpeLazyTripletLoss_L2pcl_binary_lossM0.5','LOGG3D','PointNetVLAD','overlap_transformer']
    #sequences   = ['kitti-GEORGIA-FR-husky-orchards-10nov23-00','kitti-strawberry-june23','kitti-orchards-sum22', 'kitti-orchards-june23','kitti-orchards-aut22']# 'kitti-strawberry-june23']
    
    new_names = ['SPCov3D','LOGG3D-Net','PointNetVLAD','OverlapTransformer']
    # Create directory
    
    range50m = list(range(1,50,1))
    range100m = list(range(1,100,1))
    
    seq_ranges = [range100m,range100m,range100m,range100m,range100m,range100m]
    show_legend_flag = False
    
    #abstract_range_fig(sequences,sota_models,seq_ranges,results,save_dir,new_names,True)
    #abstract_range_fig(sequences,sota_models,seq_ranges,results,save_dir,new_names,False)
    
    
    for i in [1,-1]:
        gen_range_fig(sequences,sota_models,i,seq_ranges,results,save_dir,
                      size_param     = 25,
                      linewidth      = 5,
                      colors         = COLORS,
                      linestyles     = LINESTYLES, 
                      new_model_name = new_names,
                      show_legend    = show_legend_flag)
    
    for i in [1,5,10,20,100]:
        gen_top25_fig(sequences,sota_models,str(i),results,save_dir,
                      size_param     = 25,
                      linewidth      = 5,
                      colors         = COLORS,
                      linestyles     = LINESTYLES,
                      new_model_name = new_names,
                      show_legend    = show_legend_flag)
        #gen_top25_fig(sequences,baseline_models,str(i),results,baselines_dir,size_param=25,colors=colors,linestyles=linestyles, new_model_name=new_baseline_name)
        
    
    

plt.close('all')