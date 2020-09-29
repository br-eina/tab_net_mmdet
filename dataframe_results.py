from os import listdir
from os.path import isfile, join
import ast
import pandas as pd
import json
import numpy as np
from argparse import ArgumentParser

_DEF_PRETRAINED_MODEL = 'icdar19'
_DEF_DATASET_TYPE = 'type_all'
_DEF_MODE = 'create'
_DEF_FILENAME = 'df'
_DEF_TRAINING_EPOCHS = 6

_PRETR_MODEL_EPOCHS = {
    'icdar19': 36,
    'icdar13': 1
}

_DTYPE_ITERATIONS_PER_EPOCH = {
    'type_all': 101,
    'type_opl_fact': 84,
    'type_opl': 58
}

_ITER_STEP = 10

def create_df_doctypes():
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
    return pd.DataFrame(df_data)

def create_df_metrics(pretrained_model, dataset_type, train_epochs):
    # Construct df with metrics and log:
    log_path = f'results/{pretrained_model}_{dataset_type}.log.json'
    # Read log:
    total_log = []
    with open(log_path, 'r') as f:
        for ind, line in enumerate(f):
            if ind != 0:
                log_line = ast.literal_eval(line)
                total_log.append(log_line)
    total_log = sorted(total_log, key=lambda k: (k['epoch'], k['iter']))

    data = []

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + train_epochs
    epochs = [epoch for epoch in range(pretr_epochs, total_epochs + 1)]

    iter_per_epoch = _DTYPE_ITERATIONS_PER_EPOCH[dataset_type]
    final_iter_in_log = (iter_per_epoch // _ITER_STEP) * _ITER_STEP

    eval_types = ['train', 'test', 'unseen']
    # Get object detection metrics for each epoch and train/test/unseen split:
    for epoch in epochs:
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
            data_line['pretrained_model'] = pretrained_model
            data_line['dataset_type'] = dataset_type
            data_line['epoch'] = epoch

            # Handle log:
            data_line['log'] = []
            if epoch != epochs[0]:
                for log_line in total_log:
                    data_line['log'].append(log_line)
                    if log_line['epoch'] == epoch and log_line['iter'] == final_iter_in_log:
                        data_line['loss'] = log_line['loss']
                        data_line['accuracy'] = log_line['s0.acc'] / 100
                        break
            data_line['evaluate'] = eval_type
            data_line['precision'] = precision
            data_line['recall'] = recall
            data_line['ap'] = ap
            data.append(data_line)

    return pd.DataFrame(data)

def parsing():
    parser = ArgumentParser(description='Create or update the resulting dataframe.')
    parser.add_argument(
        '--pretr', metavar='PRETRAINED_MODEL', default=_DEF_PRETRAINED_MODEL,
        type=str, help='Select pretrained model: icdar19/icdar13/etc. (default: icdar19)'
    )
    parser.add_argument(
        '--dtype', metavar='DATASET_TYPE', default=_DEF_DATASET_TYPE,
        type=str, help='Select dataset type: type_all/type_opl_fact/type_opl (default: type_all)'
    )
    parser.add_argument(
        '--mode', metavar='MODE', default=_DEF_MODE, type=str,
        help='Create dataframe or update the existing one "results/df.pkl": create/update (default: create)'
    )
    parser.add_argument(
        '--name', metavar='DF_FILENAME', default=_DEF_FILENAME, type=str,
        help='Define the output dataframe (with metrics and logs) filename (default: df)'
    )
    parser.add_argument(
        '--train_epochs', metavar='TRAINING_EPOCHS', default=_DEF_TRAINING_EPOCHS, type=int,
        help='Define the number of training epochs (default: 6)'
    )
    args = parser.parse_args()
    return args.pretr, args.dtype, args.mode, args.name, args.train_epochs


def main():
    pretrained_model, dataset_type, mode, res_df_name, train_epochs = parsing()

    df_doctypes = create_df_doctypes()
    df_metrics = create_df_metrics(pretrained_model, dataset_type, train_epochs)

    if mode == 'update':
        df_exist_path = 'results/df.pkl'
        df_exist = pd.read_pickle(df_exist_path)
        df_metrics = pd.concat([df_exist, df_metrics], ignore_index=True)

    path_df_metrics = f'results/{res_df_name}.pkl'
    path_df_doctypes = 'results/df_doctypes.pkl'
    df_metrics.to_pickle(path_df_metrics)
    df_doctypes.to_pickle(path_df_doctypes)
    df_metrics.to_csv(f'results/{res_df_name}.csv')

main()
