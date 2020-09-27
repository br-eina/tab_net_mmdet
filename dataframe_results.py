from os import listdir
from os.path import isfile, join
import ast
import pandas as pd
import json
import numpy as np

pretrained_model = 'icdar19'
dataset_type = 'type_all'

log_path = f'results/{pretrained_model}_{dataset_type}.log.json'
# model_results_path = f'results/{pretrained_model}_{dataset_type}_results.json'

# Read log:
total_log = []
with open(log_path, 'r') as f:
    for ind, line in enumerate(f):
        if ind != 0:
            log_line = ast.literal_eval(line)
            total_log.append(log_line)
total_log = sorted(total_log, key=lambda k: (k['epoch'], k['iter']))

data = []
epoches = [36, 37, 38, 39, 40, 41, 42]
eval_types = ['train', 'test', 'unseen']
# Get object detection metrics for each epoch and train/test/unseen split:
for epoch in epoches:
    for eval_type in eval_types:
        model_results_path = f'results/metrics/{pretrained_model}/{dataset_type}/results_{epoch}_{eval_type}.txt'
        # Parse precision and recall:
        with open(model_results_path, 'r') as f:
            for line in f:
                if line.startswith('Precision'):
                    prec_line = line.rstrip()
                if line.startswith('Recall'):
                    recall_line = line.rstrip()
                if line.startswith('AP:'):
                    ap_line = line.rstrip()
        for line in (prec_line, recall_line):
            arr_str = line[line.find('[') :]
            if line == prec_line:
                precision = np.array([float(val) for val in ast.literal_eval(arr_str)])
            elif line == recall_line:
                recall = np.array([float(val) for val in ast.literal_eval(arr_str)])

        ap_percent = ap_line.split(' ')[1]
        ap = float(ap_percent[:len(ap_percent)-1])
        ap = ap / 100

        data_line = dict()
        data_line['pretrained_model'] = 'icdar19'
        data_line['dataset_type'] = 'type_all'
        data_line['epoch'] = epoch

        # Handle log:
        data_line['log'] = []
        if epoch != 36:
            for log_line in total_log:
                data_line['log'].append(log_line)
                if log_line['epoch'] == epoch and log_line['iter'] == 100:
                    break
        data_line['evaluate'] = eval_type
        data_line['precision'] = precision
        data_line['recall'] = recall
        data_line['ap'] = ap
        data.append(data_line)

df = pd.DataFrame(data)
path_df = 'results/df.pkl'
df.to_pickle(path_df)

# Construct df_doctypes dataframe:
vgg_annotations_path = 'annotations/vgg_json.json'
with open(vgg_annotations_path, 'r+') as f:
    ann_json = json.load(f)
df_data = []
for key in ann_json:
    ann_image = ann_json[key]
    df_series = dict()
    df_series['filename'] = ann_image['filename']
    df_series['img_type'] = ann_image['file_attributes']['img_type']
    df_data.append(df_series)
df_doctypes = pd.DataFrame(df_data)

path_df_doctypes = 'results/df_doctypes.pkl'
df_doctypes.to_pickle(path_df_doctypes)
