import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from utils import find_file


def generate_table(results, models, sequences, files_to_show, topk, range=10, res=3):
    table_recall = pd.DataFrame(columns=models, index=sequences)

    table = []
    meta_table = []
    segment_ids = []
    for model in models:
        model_row = []
        columns = []
        for seq in sequences:
            for tag, scores in results[seq][model].items():
                for tags in files_to_show:
                    if not isinstance(tags, list):
                        tags = [tags]

                    index = find_file(scores, tags)
                    if index == -1 or len(index) == 0:
                        print(f"File {tags} not found in {model} {seq}")
                        continue

                    dataframe_ = scores[index[0]]['df']
                    path = scores[index[0]]['path']

                    file_name = path.split('/')[-1]
                    segment_ids.append([f"{model}_{seq}_{file_name}"])

                    target_column = np.asarray(dataframe_[str(range)])
                    target_cell = target_column[topk-1]
                    model_row.append(target_cell)
                    columns.append(f"{seq}_{file_name}")
        meta_table.append(columns)
        table.append(model_row)

    # table = np.asarray(table).T
    column_names = np.unique(np.array(meta_table), axis=0)[0]
    panda_frame = []
    # str_array = np.array([np.unique(table) for table in meta_table]).flatten()
    try:
        # unqiue_columns = np.unique(str_array,axis=0).tolist()
        panda_frame = [pd.DataFrame(table, columns=column_names, index=models)]
    except:
        remap_colm = []
        remap_row = []
        for values, segment in zip(table, segment_ids):
            for seq in sequences:
                line_value = []
                line_label = []
                for i, seg in enumerate(segment):
                    if seq in seg:
                        line_value.append(values[i])
                        line_label.append(segment[i])
                        # remap_row[seq] = values
                        # remap_row[seq] = seg
        unqiue_columns = np.unique(segment_ids).tolist()
        seq_remap = {}
        # for seq in sequences:
        #    seq_bundle = [  for model in meta_table]
        #    seq_remap[seq] = 
        # array = [  for ]

        panda_frame = [pd.DataFrame(table, columns=unqiue_columns, index=models)]

    return panda_frame


def run_table(root, files_to_show, seq_order, model_order, topk, range, tag, save_dir, save_latex=True):
    # Custom loading logic replacing load_results
    matches = {}
    from utils import parse_result_path

    files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(root) for file in files if file.endswith(".csv")]

    # Logic to auto-discover models if not provided is tricky without rules, 
    # but let's assume if model_order is None, we might fail or need a fallback list.
    # We will rely on model_order being passed correctly now.
    known_models = model_order if model_order else []
    
    for file in files:
        # get which dataset and model
        seq_matches = [seq for seq in seq_order if seq in file]
        if not seq_matches:
            continue
        seq_key = seq_matches[0]
        
        model_key = None
        if known_models:
            model_matches = [model for model in known_models if model in file]
            if not model_matches:
                continue
            model_key = model_matches[0]
        else:
             # Fallback: try to guess or skip? 
             # For now, if no model_order, we skip logic requiring it, 
             # but populate using whatever info we can? 
             # Unlikely to work well without model keys.
             continue

        match = parse_result_path(file,model_key=model_key,seq_key=seq_key,score_key = "@")
        
        # Unpack the dictionary 
        model_name = match['model']
        seq_name = match['seq']
        score_name = match['score']
        df = match['df']

        
        if seq_name not in matches:
            matches[seq_name] = {} 
        if model_name not in matches[seq_name]:
            matches[seq_name][model_name] = {}
        if score_name not in matches[seq_name][model_name]:
            matches[seq_name][model_name][score_name] = []
            
        matches[seq_name][model_name][score_name].append({'df':df,'path':file})


    results = matches
    models = list(set([m for s in matches for m in matches[s]]))
    sequences = list(matches.keys())
    
    sequences =  sequences # sequences.tolist() not needed as it is list
    # print all models and sequences
    
    print("\n")
    print("*"*10)
    
    print(models)
    print(sequences)

    print("*"*10)
    print("\n")
    
    seq_bool = [True for seq in seq_order if seq in sequences]
    assert sum(seq_bool) == len(seq_order), "Sequence not found in the dataset"

    if model_order is not None:
        pass # We already filtered by model_order during loading
        #model_bool = [True for item in model_order if item in models]
        #assert sum(model_bool) == len(model_order), "model not found in the dataset"
    else:
        model_order = models

    table_r = generate_table(results,model_order,seq_order,files_to_show,topk,range=range,res=3)
    table_r = table_r[0]  # Quick fixe
    topk = str(topk)
    if topk == '-1':
        topk = "1%"

    # save dataframe to csv
    os.makedirs(save_dir, exist_ok=True)
    table_r.to_csv(os.path.join(save_dir, f"{tag}_recall_{range}m@{topk}.csv"))

    if save_latex:
        latex_table = tabulate(table_r, tablefmt="latex", headers="keys", floatfmt=".3f")
        latex_table = latex_table.replace(" ", "")
        file = os.path.join(save_dir, f"{tag}_recall_{range}m@{topk}.tex")
        os.makedirs(save_dir, exist_ok=True)
        f = open(file, "w")
        f.write(latex_table)
        f.close()

    print("\n")
    print(sequences)
    print(latex_table)

    return table_r


if __name__ == "__main__":
    root = "/home/tiago/workspace/place_uk/PointNetGAP/saved"

    save_dir = "hortov2_uk"

    sequences = ['00', '02', '05', '06', '08']

    sequences = ['ON23', 'OJ22', 'OJ23', 'ON22', 'SJ23', 'GTJ23']

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

    topk = 1
    target_range = 10
    
    files_to_show = ["recall.csv"]
    run_table(root,files_to_show,sequences,model_order=model_order, topk=topk, range = target_range, tag = 'global', save_dir = save_dir)

    
    
    #files_to_show = ["recall_0.csv","recall_1.csv","recall_2.csv","recall_3.csv","recall_4.csv","recall_5.csv"]
    #run_table(root,files_to_show,sequences,model_order=None, topk=topk, range = target_range, tag = 'segments', save_dir = save_dir)

