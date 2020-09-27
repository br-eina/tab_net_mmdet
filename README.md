You must have a **tab_net/** directory in Google Drive.

**tab_net/** folder structure:
- pretrained_models/
- images/
- training/{pretrained_model}/{dataset_type}/
- libs/

## 1. Prepare a dataset:
Dataset consists from 3 types of documents: payment orders (opl), invoices (fact) and other documents (misc).

Place document images to the **images/** project folder.

Pattern of document images filenames: **inv-XXXX.jpg** (**inv-0000.jpg**, etc.).

## 2. Annotate the dataset:
1. Annotate tables on document images:
- Open Label_IMG Annotator from **annotations/annotators/** project folder;
- **Open Dir** -> **images/** project folder;
- **Change Save Dir** -> **annotations/PASCAL_VOC_annotations/xml_all/** project folder;
- Use default label **bordered**;
- Draw rectangles around each table on your dataset.
2. Annotate type of document images:
- Open VGG_Image Annotator from **annotations/annotators/** project folder;
- **File attributes: "img_type": ["opl", "fact", "misc"]**
- Annotate only file_attributes according to the document type;
- **Annotations -> Export Annotations (as json)** to the **annotations/** project folder.

## 3. Convert VOC annotations to COCO format:
MMDetection config models work with annotations in COCO format, so you have to convert annotations from сlause 2.1.

**VOC_to_COCO** script apart from convertation provides *train/test/unseen* split of your dataset. Model will validate only on *test* split, so *unseen* images can be used as evaluation on unkown data.

You can choose **dataset_type** to use only a part of you dataset: all document types, or only specifc document types. Given the provided **dataset_type**, model will train only on these document types. Providen **dataset_type** values:
- **type_all** - all document types (the whole dataset);
- **type_opl_fact** - only payment orders (opl) and invoices (fact);
- **type_opl** - only payment orders (opl).
1. Choose **dataset_type**;
2. Launch **VOC_to_COCO** script from Terminal: `python VOC_TO_COCO.py --dtype dataset_type`
3. Output files locations:
    - **train/test/unseen** splits of XML annotations (PASCAL VOC) for each **dataset_type** will be located in **annotations/PASCAL_VOC_Annotations/{dataset_type}/{test/train/unseen}** project folder;
    - **train/test** COCO annotations for each **dataset_type** will be located in **annotations/COCO_Annotations/{dataset_type}** project folder.

## 4. Upload necessary files to Google Drive:
All training process is done with Colab service. Colab is strongly integrated with Google Drive, and processes with files on Google Dirve are handled quite effectively.

1. Place large-sized libs (*pytorch*, etc.) to the **libs/** Google Drive folder;
1. Place pretrained models to the **pretrained_models/** Google Drive folder;
2. Place document images dataset into **images/** Google Drive folder;
3. Place corresponding (for *dataset_type*) **COCO Annotations** files *(coco_train.json / coco_test.json)* to the **training/{pretrained_model}/{dataset_type}/** Google Drive folder.

## 5. Start the training:
Training is done through the Colab notebook with Google GPUs.
1. Launch **tab_net_finetuning.ipynb** as Colab notebook;
2. Define 