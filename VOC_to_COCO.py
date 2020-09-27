import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import xmltodict
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from argparse import ArgumentParser

def create_folders(*folders):
    for folder in folders:
        if not exists(folder):
            makedirs(folder)

def convert_VOC_to_COCO(rootDir,xmlFiles):
  attrDict = dict()
  # Add categories according to you Pascal VOC annotations
  attrDict["categories"]=[{"supercategory":"none","id":1,"name":"bordered"},
                          {"supercategory":"none","id":2,"name":"cell"},
                          {"supercategory":"none","id":3,"name":"borderless"}]
  images = list()
  annotations = list()
  id1 = 1

  # Some random variables
  cnt_bor = 0
  cnt_cell = 0
  cnt_bless = 0

  # Main execution loop
  for root, dirs, files in os.walk(rootDir):
    image_id = 0
    for file in xmlFiles:
      image_id = image_id + 1
      if file in files:
        annotation_path = os.path.abspath(os.path.join(root, file))
        image = dict()
        doc = xmltodict.parse(open(annotation_path).read())
        image['file_name'] = str(doc['annotation']['filename'])
        image['height'] = int(doc['annotation']['size']['height'])
        image['width'] = int(doc['annotation']['size']['width'])
        image['id'] = image_id

        print("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)
        if 'object' in doc['annotation']:
          for key,vals in doc['annotation'].items():
            if(key=='object'):
              for value in attrDict["categories"]:
                if(not isinstance(vals, list)):
                  vals = [vals]
                for val in vals:
                  if str(val['name']) == value["name"]:
                    annotation = dict()
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image_id
                    x1 = int(val["bndbox"]["xmin"])  - 1
                    y1 = int(val["bndbox"]["ymin"]) - 1
                    x2 = int(val["bndbox"]["xmax"]) - x1
                    y2 = int(val["bndbox"]["ymax"]) - y1
                    annotation["bbox"] = [x1, y1, x2, y2]
                    annotation["area"] = float(x2 * y2)
                    annotation["category_id"] = value["id"]

                    # Tracking the count
                    if(value["id"] == 1):
                      cnt_bor += 1
                    if(value["id"] == 2):
                      cnt_cell += 1
                    if(value["id"] == 3):
                      cnt_bless += 1

                    annotation["ignore"] = 0
                    annotation["id"] = id1
                    annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                    id1 +=1
                    annotations.append(annotation)
        else:
          print("File: {} doesn't have any object".format(file))
      else:
        print("File: {} not found".format(file))

  attrDict["images"] = images
  attrDict["annotations"] = annotations
  attrDict["type"] = "instances"

  # Printing out some statistics
  print(len(images))
  print("Bordered : ",cnt_bor," Cell : ",cnt_cell," Bless : ",cnt_bless)
  print(len(annotations))

  jsonString = json.dumps(attrDict, indent = 4, sort_keys=True)
  return jsonString

_DEF_DATASET_TYPE = 'type_all'

def parsing():
	parser = ArgumentParser(description='Make test/train/unseen split of df and convert VOC to COCO (train/test).')
	parser.add_argument(
		'--dtype', metavar='DATASET_TYPE', default=_DEF_DATASET_TYPE,
		type=str, help='Select subDF: type_all/type_opl_fact/type_opl (default: type_all)'
	)
	args = parser.parse_args()
	return args.dtype

def main():
	_DATASET_TYPE = parsing()
	# Construct dataframe with filenames and image_types:
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
	df = pd.DataFrame(df_data)

	if _DATASET_TYPE == 'type_all':
		pass
	elif _DATASET_TYPE == 'type_opl_fact':
		df = df[(df['img_type'] == 'opl') | (df['img_type'] == 'fact')]
	elif _DATASET_TYPE == 'type_opl':
		df = df[df['img_type'] == 'opl']

	df_learn, df_unseen = train_test_split(df, train_size=0.8, stratify=df['img_type'], random_state=42)
	df_train, df_test = train_test_split(df_learn, train_size=0.8, stratify=df_learn['img_type'], random_state=42)

	img_list_train = sorted(list(df_train['filename']))
	img_list_test = sorted(list(df_test['filename']))
	img_list_unseen = sorted(list(df_unseen['filename']))

	xml_list_train = []
	xml_list_test = []
	xml_list_unseen = []

	for image_name in img_list_train:
		xml_list_train.append(image_name.replace('jpg', 'xml'))
	for image_name in img_list_test:
		xml_list_test.append(image_name.replace('jpg', 'xml'))
	for image_name in img_list_unseen:
		xml_list_unseen.append(image_name.replace('jpg', 'xml'))

	# Sort lists:
	xml_list_train = sorted(xml_list_train)
	xml_list_test = sorted(xml_list_test)
	xml_list_unseen = sorted(xml_list_unseen)

	xml_dir_all = 'annotations/PASCAL_VOC_annotations/xml_all/'

	xml_dir_train = f'annotations/PASCAL_VOC_annotations/{_DATASET_TYPE}/train/'
	xml_dir_test = f'annotations/PASCAL_VOC_annotations/{_DATASET_TYPE}/test/'
	xml_dir_unseen = f'annotations/PASCAL_VOC_annotations/{_DATASET_TYPE}/unseen/'

	create_folders(xml_dir_train, xml_dir_test, xml_dir_unseen)

	# Copy train/test/unseen xml from xml_all/ to corresponding folder:
	for xml in xml_list_train:
		shutil.copy2(f'{xml_dir_all}{xml}', f'{xml_dir_train}{xml}')
	for xml in xml_list_test:
		shutil.copy2(f'{xml_dir_all}{xml}', f'{xml_dir_test}{xml}')
	for xml in xml_list_unseen:
		shutil.copy2(f'{xml_dir_all}{xml}', f'{xml_dir_unseen}{xml}')

	# Generate COCO annotations for train and test splits:
	coco_train = convert_VOC_to_COCO(xml_dir_train, xml_list_train)
	coco_test = convert_VOC_to_COCO(xml_dir_test, xml_list_test)

	coco_dir = f'annotations/COCO_annotations/{_DATASET_TYPE}/'
	create_folders(coco_dir)
	coco_train_path = f'{coco_dir}coco_train.json'
	coco_test_path = f'{coco_dir}coco_test.json'

  	# Save COCO annotations for train and test splits:
	with open(coco_train_path, 'w') as f:
  		f.write(coco_train)
	with open(coco_test_path, 'w') as f:
		f.write(coco_test)

	img_list_train_path = f'{coco_dir}train_list.json'
	img_list_test_path = f'{coco_dir}test_list.json'
	img_list_unseen_path = f'{coco_dir}unseen_list.json'
	# Save lists with image_names for train/test/unseen splits:
	with open(img_list_train_path, 'w') as f:
		json.dump(img_list_train, f)
	with open(img_list_test_path, 'w') as f:
		json.dump(img_list_test, f)
	with open(img_list_unseen_path, 'w') as f:
		json.dump(img_list_unseen, f)

main()
