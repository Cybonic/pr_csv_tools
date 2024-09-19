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
    # plot only overall PL performance 
    files_to_show = ["recall.csv"] 
    pd_array = run_table(root,files_to_show,sequences,model_order=model_order, topk=topk, range = target_range, tag = 'global', save_dir = save_dir)
    
    scores = pd_array.to_numpy()
    values = scores[idx_y,:]
    
    # Create a heatmap
    plt.figure( figsize=(30, 15)) # figsize=(30, 10)
    sns.heatmap(values, annot=True, cmap='RdYlGn', linewidths=.5, fmt='.3f', annot_kws={"size": size_param})
    
    # clean xticks
    xticks = []
    for i in range(len(pd_array.columns)):
        # get the sequence from tuple
        column =  pd_array.columns[i].split('_')[0] # Get only the segment number 
        xticks.append(column)
    
    # clean yticks
    yticks = []
    for i in range(len(ROWS)):
        # get the sequence from tuple
        yticks.append(new_model[i])
        
    plt.xticks(np.arange(len(pd_array.columns)) + 0.5,xticks, rotation=45, fontsize=size_param)
    plt.yticks(np.arange(len(ROWS)) + 0.5, ROWS, rotation=0, fontsize=size_param)
    plt.xlabel('Sequences',fontsize=size_param)
    plt.ylabel('Models',fontsize=size_param)
    #plt.title('Heatmap of Scores')
    plt.savefig(os.path.join(heatmap_path,f'global_r{target_range}m_top{topk}.pdf'),transparent=True)
    #plt.show()
        
    
    ############################################# 
    # Segments
    # plot the performance for each segment
    
    for seq in sequences:
        files_to_show = ["recall_1.csv","recall_2.csv","recall_3.csv","recall_4.csv","recall_5.csv"]
        #sequences = [seq]
        seq = [seq]
        pd_seg_array = run_table(root,files_to_show,seq,model_order=model_order, topk=topk, range = target_range, tag = 'segments', save_dir = save_dir)

    
        scores = pd_seg_array.to_numpy()
        values = scores[idx_y,:]

        
        plt.figure(figsize=(15, 10)) # 
        
        sns.heatmap(values, annot=True, cmap='RdYlGn', linewidths=0.01, fmt='.2f', annot_kws={"size": size_param}, cbar=False, square=True,xticklabels=False, yticklabels=False)
        
        # clean xticks
        xticks = []
        for i in range(len(pd_seg_array.columns)):
            # get the sequence from tuple
            column =  pd_seg_array.columns[i].split('_')[-1].split('.')[0] # Get only the first part of  e.g. ON23_recall
            xticks.append(column)
        yticks = []
        for i in range(len(ROWS)):
            yticks.append(new_model[i])
            
        plt.xticks(np.arange(len(xticks)) + 0.5,xticks, rotation=45, fontsize=size_param)
        plt.yticks(np.arange(len(ROWS)) + 0.5, ROWS, rotation=0, fontsize=size_param)
        plt.xlabel('Segments',fontsize=size_param)
        plt.ylabel('Models',fontsize=size_param)
        #plt.title('Heatmap of Scores')
        plt.savefig(os.path.join(heatmap_path,f'segments_{seq[0]}_r{target_range}m_top{topk}.pdf'),transparent=True)
        plt.close()
    
        
    




if __name__ == "__main__":
    root = "/home/tiago/workspace/pointnetgap-RAL/thesis/horto_predictions"
    save_dir = "thesis"
    
    sequences = ['ON23','OJ22','OJ23','ON22','SJ23','GTJ23']
    
    model_order = [ 'SPVSoAP3DLoss',
                    'SPVSoAP3D',
                    'PointNetPGAPLoss',
                    'PointNetPGAP',
                    'PointNetVLAD',
                    'LOGG3D',
                    'overlap_transformer',
                   ]
    
    new_model = ['SPVSoAP3D_SLC','SPVSoAP3D','PointNetGAP_SLC','PointNetGAP','PointNetVLAD','LOGG3D','OverlapTransformer']
    
    ROWS = ['SPVSoAP3D_SLC','SPVSoAP3D','PointNetGAP_SLC','PointNetGAP','PointNetVLAD','LOGG3D','OverlapTransformer']
    
    

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
    
    
    
    
    