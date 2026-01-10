import os
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_table import run_table
from utils import  find_file_old as find_file, parse_result_path
from utils import  load_results

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('Greens')(np.linspace(0, 1, 10)))

sns.set_palette('colorblind')

COLORS     = ["blue","gray","red", "green","orange","brown","pink","gray","olive","purple"]
LINESTYLES = ["-","-.","--", ":", "-", "-","-", "-.", "--","-", "-.", "--"]
LINESTYLES = ['*', '--', '-.', ':',  "-", "-"]
# Define line styles and markers
line_styles = ['-', '--', '-.']  # Line styles
#markers = ['o', 's', '^']        # Markers
markers = ['s', '^', 'v', 'D', 'p','o']

SIZE_PARAM = 25
LINEWIDTH  = 5
   
    
    

def generate_top25(results,models,sequences,files_to_show,range=10,res=3,**args):
    """ generate top 25 data structure
    

    Args:
        results (_type_): _description_
        models (_type_): _description_
        sequences (_type_): _description_
        files_to_show (_type_): _description_
        range (int, optional): _description_. Defaults to 10.
        res (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
   
    
    table ={}
    topk = np.arange(1,25+1,1) # reset topk
    for seq in sequences:
        model_row = []
        columns = []
        
        for i,model in enumerate(models):
            
            for tag,scores in  results[seq][model].items():
                for file in files_to_show:
                    
                    index = find_file(scores,file)
                    if index == -1:
                        print(f"File {file} not found in {model} {seq}")
                        continue
                    
                    dataframe_ = scores[index]['df']
                    path = scores[index]['path']
                    
                    range_lookup = np.array(dataframe_.columns) # -> column direction (str)
                    topk_lookup  = np.array(dataframe_.index) # -> row direction
                
                    range_idx = np.array([i for i,value in enumerate(range_lookup) if not ' ' in value and int(value) == range])[0].item()
                    topk_idx = np.array([i for i,value in enumerate(topk_lookup) for t in topk if int(value) == t-1])#[0].item()
                
                    recall_value = np.asarray(dataframe_)[topk_idx,range_idx]
                
                    #target_column = np.asarray(dataframe_[str(range)])
                    
                    #print(target_column.shape)
                    #topk =  np.arange(target_column.shape[0]) if len(target_column)< max(topk) else topk
                    #target_cell   = target_column[topk-1].astype(np.float32)
                    #print(topk)
                    #print(target_cell)
                    if 'new_model_names' in args:
                        model_ = args['new_model_names'][i] # Must be in the same order as the model
                        model = model_
                    
                    model_row.append(recall_value)
                    columns.append(f"{model}")
                    
                    topk = np.arange(1,25+1,1,np.int32) # reset topk
        table[seq] = pd.DataFrame(model_row, columns=topk, index=columns)       
    return  table



def generate_range(results,models,sequences,files_to_show,seq_ranges,topk=10,res=3,**args):
    """ generate top 25 data structure
    

    Args:
        results (_type_): _description_
        models (_type_): _description_
        sequences (_type_): _description_
        files_to_show (_type_): _description_
        range (int, optional): _description_. Defaults to 10.
        res (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    table ={}
    #topk = np.arange(1,25+1,1) # reset topk
    for seq in sequences:
        ranges = np.arange(1,seq_ranges[seq],5)
        model_row = []
        columns = []
        
        for i,model in enumerate(models):
            
            for tag,scores in  results[seq][model].items():
                for file in files_to_show:
                    
                    index = find_file(scores,file)
                    if index == -1:
                        print(f"File {file} not found in {model} {seq}")
                        continue
                    
                    dataframe_ = scores[index]['df']
                    path = scores[index]['path']
                    
                    #target_column = np.asarray(dataframe_)[topk-1,1:]#[str(range)]) [0] -> top k value
                    
                    range_lookup = np.array(dataframe_.columns) # -> column direction (str)
                    topk_lookup  = np.array(dataframe_.index) # -> row direction
                
                    range_idx = np.array([i for i,value in enumerate(range_lookup) for r in ranges if not ' ' in value and int(value) == r])#[0].item()
                    topk_idx = np.array([i for i,value in enumerate(topk_lookup)  if int(value) == topk-1])[0].item()
                
                    recall_value = np.asarray(dataframe_)[topk_idx,range_idx]
                    
                    #print(target_column.shape)
                    #topk =  np.arange(target_column.shape[0]) if len(target_column)< max(topk) else topk
                    #target_cell   = target_column[ranges].astype(np.float32)
                    #print(topk)
                    #print(target_cell)
                    if 'new_model_names' in args:
                        model_ = args['new_model_names'][i] # Must be in the same order as the model
                        model = model_
                    
                    model_row.append(recall_value)
                    columns.append(f"{model}")
                    
                    #topk = np.arange(1,25+1,1,np.int32) # reset topk
        table[seq] = pd.DataFrame(model_row, columns=ranges, index=columns)       
    return  table




def run_range_graphs(root,seq_order,model_order,**args):
    
    seq_ranges = {'OJ22':120,'GTJ23':120,'ON23':120,'OJ23':120,'ON22':120,'SJ23':120}
    
    graph_path = args['save_dir']
    
    results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    sequences = sequences.tolist()
    
    
    # print all models and sequences
    print(models)
    print(sequences)
    
    
    seq_bool = [True for seq in seq_order if seq in sequences]
    assert sum(seq_bool) == len(seq_order), "Sequence not found in the dataset"
    
    if model_order !=  None:
        model_bool = [True for item in model_order if item in models]
        assert sum(model_bool) == len(model_order), "Model not found in the dataset"
    else:
        model_order = models
    
    show_legend = False
    if 'show_legend' in args:
        show_legend = args['show_legend']
        
    for top_k in [1,5,10]:
        
        curr_graph_path = os.path.join(graph_path,f'top{top_k}_range')
        os.makedirs(curr_graph_path, exist_ok=True)
        
        range_table = generate_range(results,model_order,
                                 seq_order,
                                 ["recall.csv"],
                                 seq_ranges,
                                 topk=top_k,
                                 **args)
        
        gen_range_fig(range_table,
                    curr_graph_path,
                    size_param     = 30,
                    linewidth      = 3,
                    marker_size    = 15,
                    colors         = COLORS,
                    linestyles     = LINESTYLES,
                    show_legend    = show_legend
        )
                    
         
def gen_range_fig(results,save_dir,size_param=15,linewidth=5,**args):
    
    
    marker_size = 15
    if "marker_size" in args:
        marker_size = args["marker_size"]
    
    show_legend = True
    if "show_legend" in args:
        show_legend = args["show_legend"]
    
    if show_legend:
        graph_dir = os.path.join(save_dir,"w_label")
    else:
        graph_dir = os.path.join(save_dir,"no_label")
             
    os.makedirs(graph_dir, exist_ok=True)
    
    seqs = list(results.keys())
    
    
  
    colors = None
    linestyles = None
    
    for seq in seqs:
        
        table = results[seq]
        
        models = table.index.tolist()
        
        models = models[::-1]
    

        plt.figure(figsize=(10,12))    
        # Plot results for each model
        for i,model in enumerate(models):
            if show_legend: 
                plt.plot(table.loc[model],linewidth=linewidth, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)],markersize=marker_size,label=model)
            else:
                plt.plot(table.loc[model],linewidth=linewidth, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)],markersize=marker_size)
        
        file = os.path.join(graph_dir,f'{seq}.pdf')
        plt.xlabel('Range [m]',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel('Recall@Range',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.ylim(0, 1)
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        
        if show_legend:
            plt.legend(fontsize=size_param)
            
        plt.savefig(file,transparent=True)
        plt.close()
        
# ==========================================================================================================
#  TOP 25 Graphs
# ==========================================================================================================

def gen_top25_fig(results,save_dir,size_param=15,linewidth=5,**args):
    
    
    marker_size = 15
    if "marker_size" in args:
        marker_size = args["marker_size"]
    
    show_legend = True
    if "show_legend" in args:
        show_legend = args["show_legend"]
    
    if show_legend:
        graph_dir = os.path.join(save_dir,"top25","w_label")
    else:
        graph_dir = os.path.join(save_dir,f"top25","no_label")
             
    os.makedirs(graph_dir, exist_ok=True)
    
    seqs = list(results.keys())
    
    
  
    colors = None
    linestyles = None
    
    for seq in seqs:
        
        table = results[seq]
        
        models = table.index.tolist()
        
        models = models[::-1]
        if 'colors' in args:
            # TODO: 
            #  [] Add default color and linestyle
            #  [] Inversion of oder is required. plotting approach overlap the line 
            
            # Line colors
            n_lines = len(models)
            colors = args['colors'][:n_lines]
            # invert order
            colors = colors[::-1]
            
            # Line styles
            linestyles = args['linestyles'][:n_lines]
            # invert order
            linestyles = linestyles[::-1]

        plt.figure(figsize=(10,12))    
        if colors != None and linestyles != None:
            # Plot results for each model
            for i,model in enumerate(models):
                if show_legend: 
                    plt.plot(table.loc[model],linewidth=linewidth, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)],markersize=marker_size,label=model)
                    #sns.lineplot(data=table.loc[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i],label=model)
                else:
                    plt.plot(table.loc[model],linewidth=linewidth, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)],markersize=marker_size)
                    #sns.lineplot(data=table.loc[model],linewidth=linewidth,color=colors[i],linestyle=linestyles[i])
            
        else:
            sns.lineplot( data=table,linewidth=linewidth)
        
        file = os.path.join(graph_dir,f'{seq}.pdf')

        
        plt.xlabel('Top k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel('Recall@k',fontsize=size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.ylim(0, 1)
        plt.tick_params(axis='y', labelsize=size_param) 
        plt.tick_params(axis='x', labelsize=size_param)
        
        if show_legend:
            plt.legend(fontsize=size_param)
            
        plt.savefig(file,transparent=True)
        
        print("\nFile saved to:", file)
        plt.close()
        

def run_top25_graphs(root,seq_order,model_order,show_legend,**args):
    
    
    graph_path = args['save_dir']
    
    
    matches = {}
    model_name = []
    seq_name = []
    score_name = []
    files_selected = []
    df = []

    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(root) for file in files if file.endswith(".csv")]

    for file in files:
        csv_file = os.path.basename(file)
        
        if 'recall.csv' not in csv_file:
            continue
        # get which dataset and model
        seq_key = [seq for seq in seq_order if seq in file]
        model_key = [model for model in model_order if model in file]
        if model_key == [] or seq_key == []:
            continue
        seq_key = seq_key[0]
        model_key = model_key[0]
        match = parse_result_path(file,model_key=model_key,seq_key=seq_key,score_key = "@")
        
        
        # Unpack the dictionary 
        model_name.append(match['model'])  
        seq_name.append(match['seq'])
        score_name.append(match['score'])
        df.append(match['df'])
        files_selected.append(file)


    for m_name, s_name, sc_name, d, file in zip(model_name, seq_name, score_name, df, files_selected):
        if s_name not in matches:
            matches[s_name] = {} 
        if m_name not in matches[s_name]:
            matches[s_name][m_name] = {}
        if sc_name not in matches[s_name][m_name]:
            matches[s_name][m_name][sc_name] = []
            
        matches[s_name][m_name][sc_name].append({'df':d,'path':file})


    #results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    models = list(set([m for s in matches for m in matches[s]]))
    sequences = list(matches.keys())
    
    # print all models and sequences
    print(models)
    print(sequences)
    
    
    seq_bool = [True for seq in seq_order if seq in sequences]
    #assert sum(seq_bool) == len(seq_order), "Sequence not found in the dataset"
    
    if model_order !=  None:
        pass
    else: 
        model_order = models

    # Note: The following loop attempting to merge 'results' into 'matches' references an undefined 'results' variable
    # if load_results is commented out. Assuming the intention is just to use the data collected in 'matches'
    # we can remove the merge loop or adapt it if 'results' was meant to be something else.
    # Since load_results is commented out, 'results' is undefined here. 
    # The previous code had:
    # for seq in sequences: ... if model in results[seq]: matches[seq][model].update(results[seq][model])
    
    # I will assign matches to results to proceed with the rest of the function
    results = matches
    
    
    results = generate_top25(results,model_order,seq_order,["recall.csv"],**args)
    
    print(f"***** SAVING TO {graph_path}***********")
    
    gen_top25_fig(  results,
                    graph_path,
                    size_param     = 30,
                    linewidth      = 3,
                    marker_size    = 15,
                    colors         = COLORS,
                    linestyles     = LINESTYLES,
                    show_legend    = show_legend
    )
        
    
        
def main_fig(root,sequences,org_model,save_dir,new_model,ROWS,**args):
    size_param = args['size_param']
    topk = args['topk']
    target_range = args['target_range']
    
    show_legend = False
    if 'show_legend' in args:
        show_legend = args['show_legend']
        
    #idx_y = [i for i,r in enumerate(new_model) if r in ROWS]
    
    # Global
    # ========================================
    files_to_show = ["recall.csv"]
    pd_array = run_top25_graphs(root,sequences,org_model, 
                          range = target_range, 
                          res = 3,
                          tag = 'global', 
                          save_dir = save_dir,
                          new_model_names = new_model,
                          show_legend = show_legend)

  


if __name__ == "__main__":
 
    root = "/home/tiago/workspace/place_uk/PointNetGAP/saved"
    
    save_dir = "hortov2_uk"
    
    # sequences = ['00','02','05','06','08']  
    
    sequences = ['PCD_Easy_DARK'] #,'OJ22','OJ23','ON22','SJ23','GTJ23']
    
    model_order = [ #'PointNetGeM',
                    #'PointNetMAC',
                    'SPVSoAP3D',
                    'PointNetPGAP',
                    'PointNetVLAD',
                    'LOGG3D',
                    'overlap_transformer'
                    #'ResNet50GeM',
                    #'ResNet50MAC',
                    #'ResNet50VLAD',
                    #'SPVGeM',
                    #'SPVMAC',
                    #'SPVVLAD',
                    
                   ]
    
    new_model = [ #'PointNetGeM',
                    #'PointNetMAC',
                    'SPVSoAP3D',
                    'PointNetPGAP',
                    'PointNetVLAD',
                    'LOGG3D',
                    'OverlapTransformer'
                    #'ResNet50GeM',
                    #'ResNet50MAC',
                    #'ResNet50VLAD',
                    #'SPVGeM',
                    #'SPVMAC',
                    #'SPVVLAD',
                    
                   ]
    
    ROWS = [        #'PointNetGeM',
                    #'PointNetMAC',
                    'SPVSoAP3D',
                    'PointNetPGAP',
                    'PointNetVLAD',
                    'LOGG3D',
                    'overlap_transformer',
                    #'ResNet50GeM',
                    #'ResNet50MAC',
                    #'ResNet50VLAD',
                    #'SPVGeM',
                    #'SPVMAC',
                    #'SPVVLAD',
                    
                   ]

    idx_y = [i for i,r in enumerate(new_model) if r in ROWS]
    
    size_param = 20
    topk = 1
    target_range = 10 
    
    
    top25 = True
    if top25:
        
        graph_path = os.path.join(save_dir,'graphs_top25',str(target_range)+'m')
        os.makedirs(graph_path, exist_ok=True)
        main_fig(root,sequences,model_order,graph_path,
                new_model,
                ROWS,
                size_param = size_param, 
                topk = topk, 
                target_range = target_range,
                show_legend = True)
    
    
    range_flag = False
    if range_flag:
        
        graph_path = os.path.join(save_dir,'graphs_range')
        os.makedirs(graph_path, exist_ok=True)
        
        run_range_graphs(root,sequences,model_order,
                save_dir    = graph_path,
                size_param  = size_param, 
                show_legend = False)
    

plt.close('all')