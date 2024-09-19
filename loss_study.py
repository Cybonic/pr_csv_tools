import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
from segment_table import run_table

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
    
    dataset_mean = []
    for seq in seqs:
        
        table = results[seq]
        models = table.index.tolist()        
        models = models[::-1]

        plt.figure(figsize=(10,12))    
        
        data = []
        for model in models:
            value = float(model.split("-")[-1][1:])
            dataset= table.loc[model]
            
            data.append([value,dataset])
      
            # plot a line
        data = np.array(data)
        
        
        plt.plot(data[:,0],data[:,1],linestyle='--',color='black',linewidth=linewidth)
        
        dataset_mean.append(data.transpose()[1])
        
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
        plt.close()
    
    
    plt.figure(figsize=(10,12))  
    
    mean_values = np.mean(np.array(dataset_mean),axis=0)
    
    # plot mean and std 
    
    plt.plot(data[:,0],mean_values,linestyle='--',color='black',linewidth=linewidth,label='Mean')
    plt.fill_between(data[:,0], mean_values - np.std(np.array(dataset_mean),axis=0), mean_values + np.std(np.array(dataset_mean),axis=0), alpha=0.2)
    
    #plt.plot(data[:,0],mean_values,linestyle='--',color='black',linewidth=linewidth)
        
    
    file = os.path.join(graph_dir,f'mean.pdf')
    plt.xlabel('loss margin',fontsize=size_param, labelpad=5)  # Set x-axis label here
    plt.ylabel('Recall@1',fontsize=size_param, labelpad=5)  # Set x-axis label here
    plt.grid()
    plt.ylim(0, 1)
    plt.tick_params(axis='y', labelsize=size_param) 
    plt.tick_params(axis='x', labelsize=size_param)
    
    if show_legend:
        plt.legend(fontsize=size_param)
        
    plt.savefig(file,transparent=True)
    plt.close()
        
        
def loss_study(root,sequences,model_order,heatmap_path,new_model,ROWS,**args):
    
    size_param = args['size_param']
    topk = args['topk']
    target_range = args['target_range']
    
    idx_y = [i for i,r in enumerate(new_model) if r in ROWS]
    
    # Global
    # ========================================
    files_to_show = ["recall.csv"]
    pd_array = run_table(root,files_to_show,sequences,model_order=model_order, topk=topk, range = target_range, tag = 'global', save_dir = heatmap_path)
    
    scores = pd_array.to_numpy()
    
    save_dir = os.path.join(heatmap_path,f'global_{target_range}_{topk}.pdf')
    
    gen_top25_fig(pd_array,save_dir,size_param=15,linewidth=5)
    


if __name__ == "__main__":
    root = "/home/tiago/workspace/pointnetgap-RAL/RALv2/supplementary_material/loss_study"
    
    save_dir = "RALv6/loss_study"
    
    sequences = ['OJ22','OJ23','ON22','SJ23']   
    
    model_order = [#'PointNetGAP-LazyTripletLoss_L2-segment_loss-m0.5',
                   'PointNormalNet_PointNet_16_cov_avg-LazyTripletLoss_L2-segment_loss-m0.1',
                   'PointNormalNet_PointNet_16_cov_avg-LazyTripletLoss_L2-segment_loss-m0.3',
                   'PointNormalNet_PointNet_16_cov_avg-LazyTripletLoss_L2-segment_loss-m0.5',
                   'PointNormalNet_PointNet_16_cov_avg-LazyTripletLoss_L2-segment_loss-m0.7',
                   'PointNormalNet_PointNet_16_cov_avg-LazyTripletLoss_L2-segment_loss-m0.9',
                   ]
    
    
    new_model = [
                 '0.1',
                 '0.3',
                 '0.5',
                 '0.7',
                 '0.9']
    
    ROWS = [
                 '0.1',
                 '0.3',
                 '0.5',
                 '0.7',
                 '0.9']
        

    heatmap_path = save_dir # os.path.join(save_dir,'heatmaps')
    os.makedirs(heatmap_path, exist_ok=True)
    
    size_param = 20
    topk = 1
    target_range = 10 
    
    
    
    
    loss_study(root,
                 sequences,
                 model_order,
                 heatmap_path,
                 new_model,
                 ROWS,
                 target_range = target_range,
                 topk = topk,
                 size_param=size_param)
    
    
    
    
    