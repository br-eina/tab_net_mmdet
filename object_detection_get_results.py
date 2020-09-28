from os import listdir, makedirs
from os.path import isfile, join, exists
import shutil
import os
from argparse import ArgumentParser

_DEF_PRETRAINED_MODEL = 'icdar19'
_DEF_DATASET_TYPE = 'type_all'
_DEF_TRAINING_EPOCHS = 6

_PRETR_MODEL_EPOCHS = {
    'icdar19': 36,
    'icdar13': 1
}

def create_folders(*folders):
    """Create folders if they don't exist

        Args:
            folders (unnamed args): folders to create

        Returns:
            None

    """
    for folder in folders:
        if not exists(folder):
            makedirs(folder)

def parsing():
    parser = ArgumentParser(description='Calculate object detection metrics for train/test/unseen split of dataset.')
    parser.add_argument(
        '--pretr', metavar='PRETRAINED_MODEL', default=_DEF_PRETRAINED_MODEL,
        type=str, help='Select pretrained model: icdar19/icdar13/etc. (default: icdar19)'
    )
    parser.add_argument(
        '--dtype', metavar='DATASET_TYPE', default=_DEF_DATASET_TYPE,
        type=str, help='Select dataset type: type_all/type_opl_fact/type_opl (default: type_all)'
    )
    parser.add_argument(
        '--train_epochs', metavar='TRAINING_EPOCHS', default=_DEF_TRAINING_EPOCHS,
        type=int, help='Define the number of training epochs (default: 6)'
    )
    args = parser.parse_args()
    return args.pretr, args.dtype, args.train_epochs

def main():
    pretrained_model, dataset_type, training_epochs = parsing()

    pretr_epochs = _PRETR_MODEL_EPOCHS[pretrained_model]
    total_epochs = pretr_epochs + training_epochs

    # epoches = [36, 37, 38, 39, 40, 41, 42]
    epochs = [epoch for epoch in range(pretr_epochs, total_epochs + 1)]
    list_types = ['test', 'train', 'unseen']
    temp_res_folder = f'{pretrained_model}/{dataset_type}/temp_results/'
    res_folder = f'results/metrics/{pretrained_model}/{dataset_type}/'

    create_folders('object_detection_metrics/' + temp_res_folder, res_folder)

    for epoch in epochs:
        for list_type in list_types:
            groundtruths_folder = f'{pretrained_model}/{dataset_type}/groundtruths_{list_type}/'
            detections_folder = f'{pretrained_model}/{dataset_type}/detections/{epoch}/{list_type}'
            # cmd = f'python pascalvoc.py -gt icdar19/type_all/groundtruths_{list_type}/ -det icdar19/type_all/detections/{epoch}/{list_type}/ -t 0.75 -gtformat xyrb -detformat xyrb -sp {temp_res_folder} -res_suf {epoch}_{list_type}'
            cmd = f'python object_detection_metrics/pascalvoc.py -gt {groundtruths_folder} -det {detections_folder} -t 0.75 -gtformat xyrb -detformat xyrb -sp {temp_res_folder} -res_suf {epoch}_{list_type} -np'
            os.system(cmd)
            plot_name = f'bordered_{epoch}_{list_type}.png'
            text_name = f'results_{epoch}_{list_type}.txt'
            shutil.move('object_detection_metrics/' + temp_res_folder + plot_name, res_folder + plot_name)
            shutil.move('object_detection_metrics/' + temp_res_folder + text_name, res_folder + text_name)

main()
