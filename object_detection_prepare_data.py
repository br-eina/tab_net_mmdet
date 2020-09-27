from os import listdir, makedirs
from os.path import isfile, join, exists
import xml.etree.ElementTree as ET
import xmltodict
import json
from argparse import ArgumentParser

_DEF_PRETRAINED_MODEL = 'icdar19'
_DEF_DATASET_TYPE = 'type_all'

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

def create_groundtruth(pretrained_model, dataset_type):
    xml_folder = 'annotations/PASCAL_VOC_annotations/xml_all/'
    list_types = ['test', 'train', 'unseen']

    # Create groundtruth files (for train/test/unseen split of dataset):
    for list_type in list_types:

        gth_folder = f'object_detection_metrics/{pretrained_model}/{dataset_type}/groundtruths_{list_type}/'
        create_folders(gth_folder)
        list_path = f'annotations/COCO_annotations/{dataset_type}/{list_type}_list.json'

        # Open train/test/unseen list for corresponding {dataset_type}:
        with open(list_path, 'r+') as f:
            img_paths_list = json.load(f)

        image_names = [file_name.split('.')[0] for file_name in img_paths_list]

        xml_files = sorted([f for f in listdir(xml_folder) if isfile(join(xml_folder, f))])

        for xml_file in xml_files:
            if xml_file.split('.')[0] not in image_names:
                continue
            ann_list = []

            xml_file_path = xml_folder + xml_file
            doc = xmltodict.parse(open(xml_file_path).read())

            file_name = doc['annotation']['filename'].split('.')[0]
            if type(doc['annotation']['object']) == type([]):
                for table in doc['annotation']['object']:
                    class_name = table['name']
                    left = table['bndbox']['xmin']
                    top = table['bndbox']['ymin']
                    right = table['bndbox']['xmax']
                    bottom = table['bndbox']['ymax']
                    ann = f'{class_name} {left} {top} {right} {bottom}'
                    ann_list.append(ann)
            else:
                table = doc['annotation']['object']
                class_name = table['name']
                left = table['bndbox']['xmin']
                top = table['bndbox']['ymin']
                right = table['bndbox']['xmax']
                bottom = table['bndbox']['ymax']
                ann = f'{class_name} {left} {top} {right} {bottom}'
                ann_list.append(ann)

            gth_path = gth_folder + file_name + '.txt'
            with open(gth_path, 'w+') as f:
                for ann in ann_list:
                    f.writelines(ann + '\n')

def create_detections(pretrained_model, dataset_type):
	# For corresponding model_results (specific pretrained_model and dataset_type)
	# Create detection files (for train/test/unseen split of dataset and for each epoch):
	results_json_path = f'results/{pretrained_model}_{dataset_type}_results.json'
	with open(results_json_path, 'r+') as f:
		detections = json.load(f)

	thresh_add_borderless = 0.8
	list_types = ['test', 'train', 'unseen']

	for epoch in detections['results']:
		epoch_num = epoch['epoch']
		epoch_folder = f'object_detection_metrics/{pretrained_model}/{dataset_type}/detections/{epoch_num}/'

		for list_type in list_types:
			detections_folder = f'{epoch_folder}{list_type}/'
			create_folders(detections_folder)

			for image in epoch[f'results_{list_type}']:
				file_name = image['image_name'].split('.')[0]
				ann_list = []

				# Bordered:
				for table in image['tables_bordered']:
					class_name = 'bordered'
					conf = table['conf']
					left = int(table['bbox'][0])
					top = int(table['bbox'][1])
					right = int(table['bbox'][2])
					bottom = int(table['bbox'][3])
					ann = f'{class_name} {conf} {left} {top} {right} {bottom}'
					ann_list.append(ann)

				# Borderless:
				for table in image['tables_borderless']:
					if table['conf'] < thresh_add_borderless:
						continue
					class_name = 'bordered'
					conf = table['conf']
					left = int(table['bbox'][0])
					top = int(table['bbox'][1])
					right = int(table['bbox'][2])
					bottom = int(table['bbox'][3])
					ann = f'{class_name} {conf} {left} {top} {right} {bottom}'
					ann_list.append(ann)

				detection_path = detections_folder + file_name + '.txt'
				with open(detection_path, 'w+') as f:
					for ann in ann_list:
						f.writelines(ann + '\n')

def parsing():
	parser = ArgumentParser(description='Prepare groundtruths and detection results to a special format for metrics calculation.')
	parser.add_argument(
		'--pretr', metavar='PRETRAINED_MODEL', default=_DEF_PRETRAINED_MODEL,
		type=str, help='Select pretrained model: icdar19/icdar13/etc. (default: icdar19)'
	)
	parser.add_argument(
		'--dtype', metavar='DATASET_TYPE', default=_DEF_DATASET_TYPE,
		type=str, help='Select dataset type: type_all/type_opl_fact/type_opl (default: type_all)'
	)
	args = parser.parse_args()
	return args.pretr, args.dtype

def main():
    pretrained_model, dataset_type = parsing()
    create_groundtruth(pretrained_model, dataset_type)
    create_detections(pretrained_model, dataset_type)

main()
