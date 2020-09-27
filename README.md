You must have a **tab_net/** directory in Google Drive.

**tab_net/** folder structure:
- pretrained_models/
- images/
- training/{pretrained_model}/{dataset_type}/
- libs/
- results/

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
- **Annotations -> Export Annotations (as json)** to the **annotations/** project folder as *vgg_json.json*.

## 3. Convert VOC annotations to COCO format:
MMDetection config models work with annotations in COCO format, so you have to convert annotations from сlause 2.1.

**VOC_to_COCO** script is based on D. Prasad script for convertation of Pascal VOC XML annotation files to a single COCO Json file. Refer to section **8. Training** of [CascadeTabNet project](https://github.com/DevashishPrasad/CascadeTabNet).

**VOC_to_COCO** script was modified to perform *train/test/unseen* split of your dataset apart from convertation. Model will validate only on *test* split, so *unseen* images can be used as evaluation on unkown data.

You can choose **dataset_type** to use only a part of you dataset: all document types, or only specifc document types. Given the provided **dataset_type**, model will train only on these document types. Provided **dataset_type** values:
- **type_all** - all document types (the whole dataset);
- **type_opl_fact** - only payment orders (opl) and invoices (fact);
- **type_opl** - only payment orders (opl).
1. Choose **dataset_type**;
2. Launch **VOC_to_COCO.py** script from Terminal: `python VOC_TO_COCO.py --dtype dataset_type`
3. Output files locations:
    - **train/test/unseen** splits of XML annotations (PASCAL VOC) for each **dataset_type** will be located in **annotations/PASCAL_VOC_Annotations/{dataset_type}/{test/train/unseen}** project folder;
    - **train/test** COCO annotations for each **dataset_type** will be located in **annotations/COCO_Annotations/{dataset_type}** project folder;
    - **train/test/unseen** JSON lists of image_names will be located in the same folder as COCO annotations.

## 4. Upload necessary files to Google Drive:
All training process is done with Colab service. Colab is strongly integrated with Google Drive, and processes with files on Google Dirve are handled quite effectively.

1. Place large-sized libs (*pytorch*, etc.) to the **libs/** Google Drive folder;
1. Place pretrained models to the **pretrained_models/** Google Drive folder;
2. Place document images dataset into **images/** Google Drive folder;
3. Place corresponding (for *dataset_type*) **COCO Annotations** files *(coco_train.json / coco_test.json)* to the **training/{pretrained_model}/{dataset_type}/** Google Drive folder;
4. Place corresponding (for *dataset_type*) **train/test/unseen** JSON lists of image_names (*test_list.json*, etc.) to the **training/{pretrained_model}/{dataset_type}/** Google Drive folder.

## 5. Start the training:
Training is done through the Colab notebook with Google GPUs.
1. Launch **tab_net_finetuning.ipynb** as Colab notebook;
2. First cell installs required environment. Restart the runtime after this step;
3. Then launch **Define desired parameters** cell. Change **pretrained_model**, **dataset_type** or **total_epoches** if necessary;
4. Launch **Train a model** cell. Checkpoints for each epoch will be stored in **training/{pretrained_model}/{dataset_type}/workdir/** Google Drive folder.
5. Launch cells from **Save predictions to JSON** section;
6. Model results and training log will be stored in **results/** Google Drive folder as JSON file;
7. Save model results and training log to **results/** project folder.

## 6. Calculate object detection metrics to measure performance:
This section is based on R. Padilla open-source project [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics).

1. Object detections script processes only prepared data with special structure. Launch **object_detection_prepare_data.py** from Terminal: `python object_detection_prepare_data.py --pretr pretrained_model --dtype dataset_type` to prepare ground truths and detection results (from model results). For more detailed information about the required structure refer to the aforementioned work.
    - Prepared groundtruth files will be stored in **object_detection_metrics/{pretrained_model}/groundtruths_test(train/unseen)/** project folder.
    - Prepared detection files will be stored in **object_detection_metrics/{pretrained_model}/detections/** project folder. They are divided into epoches folders and train/test/unseen splits.
2. Calculate object detection metrics of model results for each epoch and split (train/test/unseen) and store it in **results/metrics/{pretrained_model}/{dataset_type}/** project folder. Launch **object_detection_get_results.py** from Terminal: `python object_detection_get_results.py --pretr pretrained_model --dtype dataset_type`.
    - Precision x Recall curve is displayed as **bordered(class_name)\_{epoch}_{split}(train/test/unseen).png**;
    - Metric values are stored as **results\_{epoch}_{split}(train/test/unseen).txt**.

## 7. Construct/update results dataframe:
Collect all metrics, logs and results to one single dataframe. It will be used in visualization of model metrics and logs for every **pretrained_model**, **dataset_type**, **evaluation_split** and **epoch**.

Launch **dataframe_results.py**. Resulting dataframe is stored in **results/** project folder as **df.pkl**. Also the script constructs **df_doctypes.pkl** dataframe to store the distribution of document types across the dataset.

## 8. Visualize results:
An interactive web-app in the dashboard format is constructed to visualize model results. It provides plots of AP Precision x Recall metrics, loss and accuracy and document_types distribution. To visualize results for each **pretrained_model**, **dataset_type**, **evaluation_split** and **epoch**, simply select desired parameters on the dashboard.

Launch **dash_app.py** to interact with dashboard.
