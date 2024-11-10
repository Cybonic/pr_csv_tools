import os
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_table import run_table
from utils import  find_file_old as find_file
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
MARKERS = ['s', '^', 'v', 'D', 'p','o']

SIZE_PARAM = 25
LINEWIDTH  = 5
   

# model_key='L2',seq_key='eval-',score_key = "@"

def recursive_build(keys,date_structure):
    #key,value  = key_dict.item()   
    if len(keys) == 1:
        value  = keys[0]
        return {str(value):date_structure}
    
    out = recursive_build(keys[:-1],date_structure)
    value  = keys[-1]
    return {str(value):out}    


def recursive_update(key_dict,matches,date_structure):
    #key,value  = key_dict.item()  
    
    if len(key_dict)==1:
            return  recursive_build(key_dict,date_structure)
    
    #k = key_dict[-1]
    # matches[k]
    main_keys = list(matches.keys())
    main_values = np.array(list(matches.values()))[0]
    if key_dict[-1] in main_keys:
        main_values = matches[key_dict[-1]]
        updated = recursive_update(key_dict[:-1],main_values,date_structure)
    else: 
        nex_keys = key_dict[:-1]
        updated = recursive_build(nex_keys,date_structure)

    matches[key_dict[-1]] = updated
            
    return  matches

       
def parse_results(dir,foi, **input_keys):
    # all csv files in the root and its subdirectories
    """_summary_

    Args:
        dir (_type_): _description_
        model_key (str, optional): _description_. Defaults to '#'.
        seq_key (str, optional): _description_. Defaults to 'eval-'.
        score_key (str, optional): _description_. Defaults to "@".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(dir) for file in files if file.endswith(foi)]
    
    matches = {}
    
    # Create a storing object for all values found 
    key_array = {}
    for key,value in input_keys.items():
        key_array[key]=[]
    
    
    for file in tqdm(files,total=len(files)):
        file_Struct = file.split("/")
        
        # Get key positions 
        key_index={}
        key_values = {}
        for key,value in input_keys.items():
            if not isinstance(value,str):
                # VALUE is the position in string
                print(f"{key}: {file_Struct[value]}")
                key_values[key]=file_Struct[value]
            else: 
                key_index = np.array([i for i,field in enumerate(file_Struct) if field.startswith(value) or value in field ]) # no repeated keys
                if len(key_index) == 0: # If some key is not found ignore this iteration
                    continue 
                value_idx = key_index[0].item()
                key_values[key] = file_Struct[value_idx].replace(value,'')
            
            # Store all values found 
            key_array[key].append(key_values[key])

        values = np.array(list(key_values.values()))[::-1] # get all values, but inverse the order
        df = pd.read_csv(file)

        date_structure = [{'df':df,'path':file}]
        
        if len(matches)==0:
            matches = recursive_build(values,date_structure)
            continue 
        else:
            matches = recursive_update(values,matches,date_structure)
            continue
    
    for key, value in key_array.items():
        key_array[key]=  np.unique(value)
        
    return matches, key_array







def generate_density_seq_mean(results,models,sequences,topk=10,range=10,res=3,**args):
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
    
    
    key_structure = ['model','density','seq','score']
    
    if 'key_structure' in args:
        key_structure = args['key_structure']
   
   
    new_model_names = models
    if 'new_model_names' in args:
        new_model_names = args['new_model_names']
    
    # Create a plot instance 
    graph = plot_graph(**args)
    
    table ={}
    
    
    
    # ***************************
    #
    # The issue is here ! The for cycle assumes a determined structure, it does not allow
    # for a new data structure
    #
    # 1) Need to split it on levels not on specific keys
    # 
    #*****************************
    
    
    # Here it is assumed to have only 3 levels 
    
    #for level1_key, level1_values in results.items():
    #    pass
        
    #    for level2_key, level2_values in level1_values.items():
    #        pass
        
    #        for level3_key, level3_values in level2_values.items():
    #            pass
                
                
    xxplot = []
    yyplot = {'mean':[],'std':[]}
        
    model_set = []
    for model,next_value in  results.items():#  [seq][model].items():
        
        xx = []
        yy = {'mean':[],'std':[]}
        
        
        idx = np.array([i for i,m in enumerate(models) if model.startswith(m)])
        if len(idx) == 0:
            continue 
        # density 
        for density, dens_value in next_value.items():
            
            x = []
            y = []    
            for seq, seq_value in dens_value.items():
                for d_tag, score_value in seq_value.items():
                    
                    v =  np.asarray(list(score_value))[0]['df']
                    
                    #v.in
                    range_lookup = np.array(v.columns) # -> column direction (str)
                    topk_lookup  = np.array(v.index) # -> row direction
                    
                    
                    range_idx = np.array([i for i,value in enumerate(range_lookup) if not ' ' in value and int(value) == range])[0].item()
                    topk_idx = np.array([i for i,value in enumerate(topk_lookup) if int(value) == topk-1])[0].item()
                    
                    recall_value = np.asarray(v)[topk_idx,range_idx]
                    
                    # = v[:,0]
                    x.append(int(density))
                    y.append(recall_value)
            
            xx.append(int(density))
            yy['mean'].append(np.mean(y))
            yy['std'].append(np.std(y))


        x_idx = np.argsort(xx)
        
        xxplot.append(np.array(xx)[x_idx])
        yy['mean']=np.array(yy['mean'])[x_idx]
        yy['std']==np.array(yy['std'])[x_idx]
        
        yyplot['mean'].append(yy['mean'])
        yyplot['std'].append(yy['std'])
                 
        model_set.append(new_model_names[idx[0]])
                
                # Find idx to reorder model_set to match models_new_names
    model_set = np.array(model_set)
    new_xx = []
    new_yy_mean = []
    new_yy_std = []


    new_model_names = new_model_names[::-1] # invert the order, so the first in the list is plot above all 
    for i,modela in enumerate(new_model_names):
        for j,modelb in enumerate(model_set):
            if modela == modelb:
                new_xx.append(xxplot[j])
                new_yy_mean.append(yyplot['mean'][j])
                new_yy_std.append(yyplot['std'][j])
        
    
    for x,m,s,name in zip(new_xx,new_yy_mean,new_yy_std,new_model_names):
        graph.plot_graph(x,m,s,name)
        graph.save_graph(f'{name}_recall@{topk}_range@{range}',topk)
    
    graph.plot_full_graph(new_xx,new_yy_mean,new_yy_std,new_model_names)
    graph.save_graph(f'mean_recall@{topk}_range@{range}',topk)

    return  None


def generate_density(results,models,sequences,topk=10,range=10,res=3,**args):
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
    
   
    key_structure = ['model','density','seq','score']
    
    if 'key_structure' in args:
        key_structure = args['key_structure']
   
   
    new_model_names = models
    if 'new_model_names' in args:
        new_model_names = args['new_model_names']
    
    # Create a plot instance 
    graph = plot_graph(**args)
    
    table ={}
    
        
    
    for seq,next_value in  results.items():#  [seq][model].items():
        model_set = []
        if not seq in sequences:
            print(f"Seq: {seq} is not included in {' '.join(sequences)}")
            continue 
        
        xx = []
        yy = []
        
        for model, value in next_value.items():
            
            idx = np.array([i for i,m in enumerate(models) if model.startswith(m)])
            if len(idx) == 0:
                continue 
            
            density = np.array(list(value.keys()))
            scores = list(value.values())
            
            x = []
            y = []
            for d_tag, scores in value.items():
                v =  np.asarray(list(scores.values()))[0][0]['df']
                
                #v.in
                range_lookup = np.array(v.columns) # -> column direction (str)
                topk_lookup  = np.array(v.index) # -> row direction
                
                
                range_idx = np.array([i for i,value in enumerate(range_lookup) if not ' ' in value and int(value) == range])[0].item()
                topk_idx = np.array([i for i,value in enumerate(topk_lookup) if int(value) == topk-1])[0].item()
                
                recall_value = np.asarray(v)[topk_idx,range_idx]
                
                # = v[:,0]
                y.append(int(d_tag))
                x.append(recall_value)
            
            y_idx = np.argsort(y)
            y = np.array(y)[y_idx]
            x = np.array(x)[y_idx]
            
            xx.append(x)
            yy.append(y)
            
            model_set.append(new_model_names[idx[0]])
        
        # Find idx to reorder model_set to match models_new_names
        model_set = np.array(model_set)
        new_xx = []
        new_yy = []
       
       
        new_model_names = new_model_names[::-1] # invert the order, so the first in the list is plot above all 
        for i,modela in enumerate(new_model_names):
            for j,modelb in enumerate(model_set):
                if modela == modelb:
                    new_xx.append(xx[j])
                    new_yy.append(yy[j])
        
        
        
        graph.plot_full_graph(new_xx,new_yy,new_model_names)
        graph.save_graph(f'{seq}_recall@{topk}_range@{range}',topk)

    return  None




class plot_graph():
 def __init__(self,save_dir,size_param=15,linewidth=5,**args):
    
    
    self.marker_size = 15
    if "marker_size" in args:
        self.marker_size = args["marker_size"]
    
    show_legend = True
    if "show_legend" in args:
        show_legend = args["show_legend"]
    
    if show_legend:
        self.graph_dir = os.path.join(save_dir,"w_label")
    else:
        self.graph_dir = os.path.join(save_dir,"no_label")
    
    self.linewidth = linewidth
    self.size_param = size_param
    self.colors = None
    if 'colors' in args:
        # TODO: 
        #  [] Add default color and linestyle
        #  [] Inversion of oder is required. plotting approach overlap the line 
        self.colors = args['colors']

        
    # Line styles
    self.linestyles = None
    if 'linestyles' in args:
        self.linestyles = args['linestyles']#[:n_lines]
        # invert order
        #linestyles = linestyles[::-1]
    
             
    os.makedirs(self.graph_dir, exist_ok=True)
    self.show_legend = show_legend
    
    #seqs = list(results.keys())

 
 def plot_graph(self,x,mean,std,labels):
    
    if isinstance(x,list):
        # it is possible arrays have inconsistencies, having the same size. numpy does not allow
        if len(x)>1:
            # check if dim match
            assert len(x)==len(labels)
            assert len(mean)==len(labels)
    

    plt.figure(figsize=(10,12))    
    
    
    if self.show_legend: 
        plt.plot(x,mean,linewidth=self.linewidth, linestyle=line_styles[1 % len(line_styles)], marker=MARKERS[1 % len(MARKERS)],markersize=self.marker_size,label=labels)
    else:
        plt.plot(x,mean,linewidth=self.linewidth, linestyle=line_styles[1 % len(line_styles)], marker=MARKERS[1 % len(MARKERS)],markersize=self.marker_size)
    
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
         
         
           
 def plot_full_graph(self,x,mean,std,labels):
    
    if isinstance(x,list):
        # it is possible arrays have inconsistencies, having the same size. numpy does not allow
        if len(x)>1:
            # check if dim match
            assert len(x)==len(labels)
            assert len(mean)==len(labels)
    

    plt.figure(figsize=(10,12))    
    
    #labels = labels[::-1]
   
    # Plot results for each model
    for i,model in enumerate(labels):
        if self.show_legend: 
            plt.plot(x[i],mean[i],linewidth=self.linewidth, linestyle=line_styles[i % len(line_styles)], marker=MARKERS[i % len(MARKERS)],markersize=self.marker_size,label=model)
        else:
            plt.plot(x[i],mean[i],linewidth=self.linewidth, linestyle=line_styles[i % len(line_styles)], marker=MARKERS[i % len(MARKERS)],markersize=self.marker_size)
        
        plt.fill_between(x[i], mean[i] - std[i], mean[i] + std[i], alpha=0.2)


 def save_graph(self,file,topk):
        #if not os.path.exists(path)
        
        file = os.path.join(self.graph_dir,f'{file}.pdf')
        
        plt.xlabel('Scan Size',fontsize=self.size_param, labelpad=5)  # Set x-axis label here
        plt.ylabel(f'Recall@{topk}',fontsize=self.size_param, labelpad=5)  # Set x-axis label here
        plt.grid()
        plt.ylim(0, 1)
        plt.tick_params(axis='y', labelsize=self.size_param) 
        plt.tick_params(axis='x', labelsize=self.size_param)
        
        if self.show_legend:
            plt.legend(fontsize=self.size_param)
            
        plt.savefig(file,transparent=True)
        print("Saved at "+ file)
        plt.close()

        
        
        
def run_graphs_density(root,seq_order,model_order, 
                        topk,
                        target_range,  
                       show_legend,**args):
    
    
    #graph_path = args['save_dir']
    # The order defines the structure
    keys = {'model':-5,'density':-6,'seq':'eval-','score':'@'}
    results,elem_array = parse_results(root,'recall.csv',**keys)
    
    
    #results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    sequences = elem_array['seq'].tolist()
    models =  elem_array['model'].tolist()
    
    #models = [model.split('-')[0] for model in models]
    # print all models and sequences
    print(models)
    print(sequences)
    
    
    seq_bool = [True for seq in seq_order if seq in sequences]
    assert sum(seq_bool) == len(seq_order), "Sequence not found in the dataset"
    
    if model_order !=  None:
        model_bool = []
        for item in model_order:
            for item_found in models:
                if item_found.startswith(item):
                    model_bool.append(True) 
        assert sum(model_bool) == len(model_order), "Model not found in the dataset"
    else:
        model_order = models
    
    
    method = 1
    
    if method == 1:
    
        results = generate_density_seq_mean(results,model_order,seq_order,
                                topk,
                                target_range,
                                #save_dir = graph_path,
                                    size_param     = 30,
                                    linewidth      = 3,
                                    marker_size    = 15,
                                    colors         = COLORS,
                                    linestyles     = LINESTYLES,
                                    show_legend    = show_legend,
                                    key_structure  = list(keys.keys()),
                                    **args)
    
    
    elif method ==2 :
        # create  a figure for each sequence
        results = generate_density(results,model_order,seq_order,
                                topk,
                                target_range,
                                #save_dir = graph_path,
                                    size_param     = 30,
                                    linewidth      = 3,
                                    marker_size    = 15,
                                    colors         = COLORS,
                                    linestyles     = LINESTYLES,
                                    show_legend    = show_legend,
                                    key_structure  = list(keys.keys()),
                                    **args)
        
    else:
        raise ValueError
    



def main_fig(root,sequences,org_model,save_dir,new_model,**args):
    size_param = args['size_param']
    topk = args['topk']
    target_range = args['target_range']
    
    show_legend = False
    if 'show_legend' in args:
        show_legend = args['show_legend']
    
    # Global
    # ========================================
    files_to_show = ["recall.csv"]
    pd_array = run_graphs_density(root,sequences,org_model, 
                                    topk = topk,
                                    target_range = target_range, 
                                    res = 3,
                                    tag = 'global', 
                                    save_dir = save_dir,
                                    new_model_names = new_model,
                                    show_legend = show_legend)

  

    



if __name__ == "__main__":
 
    root = "/home/tiago/workspace/pointnetgap-RAL/thesis/Thesis_density"
    
    save_dir = "thesis_density"
    
    # sequences = ['00','02','05','06','08']  
    
    sequences = ['OJ23']#'ON23','OJ23','ON22','SJ23','GTJ23']
    
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
    
    topk = 10
    target_range = 1 
    
    
    graph_path = os.path.join(save_dir,'graphs',str(target_range)+'m')
    os.makedirs(graph_path, exist_ok=True)
    
    
    
    main_fig(root,sequences,model_order,graph_path,new_model,
             size_param = 20, 
             topk = topk, 
             target_range = target_range,
             show_legend = False)
     
    
    
    

plt.close('all')