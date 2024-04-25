import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
from segment_table import run_table

def main_heatmap(root,sequences,model_order,heatmap_path,new_model,ROWS,**args):
    
    size_param = args['size_param']
    topk = args['topk']
    target_range = args['target_range']
    
    idx_y = [i for i,r in enumerate(new_model) if r in ROWS]
    
    # Global
    # ========================================
    files_to_show = ["recall.csv"]
    pd_array = run_table(root,files_to_show,sequences,model_order=model_order, topk=topk, range = target_range, tag = 'global', save_dir = save_dir)
    
    scores = pd_array.to_numpy()
    
    values = scores[idx_y,:]
    
    # Create a heatmap
    plt.figure(figsize=(20, 5)) # 
    sns.heatmap(values, annot=True, cmap='RdYlGn', linewidths=.5, fmt='.3f', annot_kws={"size": size_param})
    
    # clean xticks
    xticks = []
    for i in range(len(pd_array.columns)):
        # get the sequence from tuple
        for seq in sequences:
            if seq in pd_array.columns[i][0]:
                xticks.append(seq)
                break
    
    yticks = []
    for i in range(len(ROWS)):
        # get the sequence from tuple
        yticks.append(new_model[i])
             
    
    
    plt.xticks(np.arange(len(pd_array.columns)) + 0.5,xticks, rotation=45, fontsize=size_param)
    plt.yticks(np.arange(len(ROWS)) + 0.5, ROWS, rotation=0, fontsize=size_param)
    plt.xlabel('Sequences',fontsize=size_param)
    plt.ylabel('Models',fontsize=size_param)
    #plt.title('Heatmap of Scores')
    plt.savefig(os.path.join(heatmap_path,f'global_{target_range}_{topk}.pdf'),transparent=True)
    #plt.show()
        
    
    ############################################# 
    # Segments
    files_to_show = ["recall_0.csv","recall_1.csv","recall_2.csv","recall_3.csv","recall_4.csv","recall_5.csv"]
    pd_seg_array = run_table(root,files_to_show,sequences,model_order=model_order, topk=topk, range = target_range, tag = 'segments', save_dir = save_dir)

    
    scores = pd_seg_array.to_numpy()
    
    columns = np.array([pd_seg_array.columns[i][0] for i in range(len(pd_seg_array.columns))])
    print(columns)
    
    
    for i in range(len(sequences)):
        seq = sequences[i]
        
        idx_x = [i for i,c in enumerate(columns) if seq in c]
        
        plt.figure(figsize=(6, 6)) # 
        
        values = scores[:,idx_x]
        values = values[idx_y,:]
        sns.heatmap(values, annot=True, cmap='RdYlGn', linewidths=0.01, fmt='.2f', annot_kws={"size": size_param}, cbar=False, square=True,xticklabels=False, yticklabels=False)
        plt.xticks(np.arange(values.shape[1]) + 0.5,np.arange(values.shape[1]), rotation=0, fontsize=size_param)
        file = os.path.join(heatmap_path,f'segment_{sequences[i]}_{target_range}_{topk}.png')
        plt.savefig(file, transparent=True)
        

        plt.yticks(np.arange(len(ROWS)) + 0.5, ROWS, rotation=0, fontsize=size_param)
        plt.xticks(np.arange(values.shape[1]) + 0.5,np.arange(values.shape[1]), rotation=0, fontsize=size_param)
        #plt.yticks(np.arange(len(ROWS)) + 0.5, ROWS, rotation=45, fontsize=size_param)
        plt.xlabel('Segments',fontsize=size_param)
        plt.ylabel('Models',fontsize=size_param)
        
        
        file = os.path.join(heatmap_path,f'segment_{sequences[i]}_{target_range}_{topk}_labels.png')
        plt.savefig(file, transparent=True)
        
    




if __name__ == "__main__":
    root = "/home/tiago/workspace/pointnetgap-RAL/RALv2/on_paper"
    save_dir = "RALv3"
    
    sequences = ['SJ23','ON22','OJ23','OJ22']   
    
    model_order = ['PointNetGAP-LazyTripletLoss_L2_segmentlossM0.5',
                   'PointNetGeM-LazyTripletLoss_L2-segment_loss-m0.5',
                   'PointNetMAC-LazyTripletLoss_L2-segment_loss-m0.5',
                   'PointNetVLAD-LazyTripletLoss_L2-segment_loss-m0.5',
                   'LOGG3D-LazyTripletLoss_L2-segment_lossM0.1-descriptors',
                   'overlap_transformer-LazyTripletLoss_L2-segment_loss-m0.5',
                   ]
    
    new_model = ['PointNetGAP','PointNetGeM','PointNetMAC','PointNetVLAD','LOGG3D','OverlapTransformer']
    
    ROWS = ['PointNetGAP','PointNetGeM','PointNetMAC','PointNetVLAD','LOGG3D','OverlapTransformer']
    
    

    heatmap_path = os.path.join(save_dir,'heatmaps')
    os.makedirs(heatmap_path, exist_ok=True)
    
    size_param = 20
    topk = 1
    target_range = 10 
    
    
    
    
    main_heatmap(root,
                 sequences,
                 model_order,
                 heatmap_path,
                 new_model,
                 ROWS,
                 target_range = target_range,
                 topk = topk,
                 size_param=size_param)
    
    
    
    
    