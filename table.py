import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from graphs import load_results
from graphs import run_top25_graphs



def generate_table(table,rows,columns,ranges,k_cand,res=3):
    table_recall = pd.DataFrame(columns=columns, index=rows)
  
    n_sub_grupes = len(ranges)
   
    rows_array = []
    for model in rows:
        n_seq = 0 # len(columns[:-2])
        tuple_list = []
        rows_local = []
        for ktop in k_cand:
            seq_array = []
            #for seq in columns[:-2]:
            for seq in columns:
                
                # Check if model exists in this sequence's table
                if not model in table[seq]:
                    continue
                    
                n_seq += 1
                
                for range in ranges:
                    ktop_str = str(ktop)
                    ktop_idx = ktop-1
                    if ktop == -1:
                        ktop_str = "1%" 
                    
                    # Columns should be (Sequence, Range, KTop)
                    label_proxy = (seq, range, ktop_str)
                    
                    
                    recall_array = []
                    
                    # check if the model is in the table
                    #if not model in table[seq]:
                    #    continue
                    
                    # Uncommenting this, but only if it's the first model?
                    # Or just overwrite it? 
                    # If I append for every model, tuple_list grows indefinitely?
                    # No, tuple_list = [] at start of model loop.
                    
                    tuple_list.append(label_proxy)
                    
                    for key, value in table[seq][model].items():
                
                        #recall_table = table[seq][model][key]['df'][str(range)]
                        # The error happens because table[seq][model][key] is a list of dicts, but accessed like a dict
                        # In graphs.py run_top25_graphs(), it constructs: matches[s_name][m_name][sc_name] = [{'df':d,'path':file}, ...]
                        
                        # Assuming we just need the first element if multiple exist, or iterate
                        # If table structure is list of dicts:
                        if isinstance(table[seq][model][key], list):
                             # Take the first entry, or adapt logic if multiple entries need handling
                             element = table[seq][model][key][0]
                             df = element['df']
                        else:
                             # Fallback to old behavior if it's somehow a dict
                             df = table[seq][model][key]['df']#[range]
                        
                        recall_value = np.array(df)
                        print(recall_value)
                        if ktop == -1:
                            recall_array.append(recall_value[-2])
                        else:
                            recall_array.append(recall_value[ktop_idx])
                    
                    max_value = np.array(recall_array).max()
                    rows_local.append(round(max_value,res))
            
            seq_array = np.array(rows_local).reshape(n_seq,n_sub_grupes)
            seq_mean = np.round(np.mean(seq_array,axis=0),4)
            seq_std = np.std(seq_array,axis=0)
            
            # Here we are appending 'seq_mean' (length 4) to 'rows_local' (length 4)
            # So rows_local becomes length 8.
            rows_local.extend(seq_mean)
            
            # The column loop only adds 'ranges' (length 4) once?
            # Wait, tuple_list is being populated AFTER the per-sequence loop now?
            
            # In validation logic:
            # We iterate ktop (1).
            # We iterate seqs (1).
            # We iterate ranges (4).
            
            # Original code seemed to be building columns for EACH sequence + the mean?
            # Or just the mean?
            
            # Looking at previous structure:
            # tuple_list was appended inside the range loop: `tuple_list.append(label_proxy)`
            # But I commented it out: `#tuple_list.append(label_proxy)`
            
            # Now tuple_list is populated only with the mean labels:
            # for label in ranges: tuple_list.append(...)
            
            # So tuple_list has 4 items.
            # But rows_local has 8 items (4 for the sequence + 4 for the mean).
            # Hence: "ValueError: 4 columns passed, passed data had 8 columns"
            
            # I need to restore tuple_list population for the individual sequence columns too.
            
            # BUT: If I restore it inside the range loop, I need to make sure I don't duplicate logic.
            # Actually, `tuple_list` should probably just correspond to what we want in the final table.
            # If we want columns for each sequence AND the mean, we need headers for all.
            pass

            # FIX:
            # 1. Clear rows_local ONLY at start of ktop loop? No, it's accumulating across sequences?
            # No, `rows_local = []` is inside `for model in rows`.
            # So `rows_local` accumulates ALL data for that model row.
            
            # `rows_array` collects `rows_local` for each model.
            
            # `tuple_list` defines the DataFrame columns.
            # It must match `rows_local` length.
            
            # Currently: 
            # rows_local has: [seq1_range1, seq1_range2, seq1_range3, seq1_range4, mean_range1, mean_range2, mean_range3, mean_range4]
            # tuple_list has: [range1, range2, range3, range4] (from the mean loop at end)
            
            # I must populate `tuple_list` for the sequences too.
            # However, `tuple_list` is reset inside `for model in rows`.
            # This is weird because columns should be same for all rows?
            # Ah, `tuple_list` is defined inside `for model in rows`.
            # If models have same sequences, columns are same.
            
            # I should uncomment `tuple_list.append` in the range loop?
            # Yes, but check logic. 
            # `label_proxy` was `(model,seq,range, ktop_str)`.
            # This makes a MultiIndex if we used `pd.MultiIndex`.
            # But now `columns=tuple_list` expects flat index if just list of strings?
            # Or list of tuples?
            
            # The error says "4 columns passed".
            # The code `df = pd.DataFrame(..., columns=tuple_list, ...)`
            # So `tuple_list` length is 4.
            # `rows_array` (data) has length 8.
            
            # I will uncomment the `tuple_list.append` inside the range loop
            # AND handle the fact that we might process multiple models.
            # Wait, `tuple_list` is rebuilt for every model?
            # Yes. But the DataFrame is created at the very end using the LAST `tuple_list`?
            # `df = pd.DataFrame(rows_array, columns=tuple_list, index=rows)`
            # This implies all rows must have same structure.
            
            # So, for the FIRST model I process, I should build the `tuple_list`.
            # Or just build it once.
            
            # Let's just uncomment the line and see. But `label_proxy` uses `model` name?
            # `label_proxy = (model,seq,range, ktop_str)`
            # If `columns` includes `model`, that's wrong for a table where rows are models.
            # The columns should represent (Sequence, Range).
            
            # Let's change `label_proxy` to not include model, and append to `tuple_list`.
            for label in ranges:
                label_proxy = str(label)+"m"
                if label == "100":
                    label_proxy = "row"
                tuple_list.append(label_proxy)
        rows_array.append(rows_local)

    #header = pd.MultiIndex.from_tuples(tuple_list)
    df = pd.DataFrame(rows_array, columns=tuple_list, index=rows)

    return df




if __name__ == "__main__":
    root = "/home/tiago/workspace/place_uk/PointNetGAP/saved"
    
    save_dir = "hortov2_uk"
    
    sequences = ['00','02','05','06','08']   
    
    sequences = ['ON23','OJ22','OJ23','ON22','SJ23','GTJ23']
    
    sequences = ['PCD_MED',"PCD_Easy_DARK"]
    
    model_order = [ 
                    'SPVSoAP3D',
                    'PointNetPGAP',
                    'PointNetVLAD',
                    'LOGG3D',
                    'overlap_transformer'
                   ]
    
    
    new_model = [
                'SPVSoAP3D',
                'PointNetPGAP',
                'PointNetVLAD',
                'LOGG3D',
                'overlap_transformer'
                 ]
    
    ROWS = [
            'SPVSoAP3D',
            'PointNetPGAP',
            'PointNetVLAD',
            'LOGG3D',
            'overlap_transformer'
            ]
    
    target_range = 10
    results,sequences,models = run_top25_graphs(root,sequences,new_model, 
                          range = target_range, 
                          res = 3,
                          tag = 'global', 
                          save_dir = save_dir,
                          new_model_names = new_model,
                          show_legend = True
                          )
    
    #results,sequences,models  = load_results(root,model_key='#',seq_key='eval-',score_key = "@")
    #models =  list(results.keys())
    print(models)
    #sequences = sequences.tolist()


    #sequences.append("Mean")
    #sequences.append("Std")
    
    topk = 1


    table_r = generate_table(results,models,sequences,[1,5,10,20],[topk],res=3)

    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    table_r.to_csv(f"recall@{topk}_paper.csv")

    latex_table = tabulate(table_r, tablefmt="latex", headers="keys",floatfmt=".3f")
    latex_table = latex_table.replace(" ", "")
    
    file = os.path.join(save_dir,f"recall@{topk}.tex")
    os.makedirs(save_dir, exist_ok=True)
    f = open(file, "w")
    f.write(latex_table)
    f.close()

    print("\n")
    print(sequences)
    print(latex_table)


