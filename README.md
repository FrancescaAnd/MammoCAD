# *Mammography CAD system*
AI-powered mammography image processing pipeline for mass detection, segmentation,
classification, and enhancement using deep learning.

## Table of Contents
1. [About the project](#1-about-the-project)
2. [Experiment results](#2-experiment-results)
3. [How to use](#3-how-to-use)


## 1. About the project
This is the repository for the implementation of a complete Computer-Aided Detection (CAD) system for mammogram analysis, integrating: 
- A mass detection stage, performed through YoloV8 model
- An instance segmentation stage
- A ResNEt-based classification 
- GAN-based super-resolution and synthetic image generation over the dataset

### Abstract
...


### Dataset
The dataset we are using is *INBreast*.
The INBreast dataset is a high-quality mammographic database designed for breast cancer research. 
It consists of 115 cases from 90 different patients, with a total of 410 full-field digital mammograms (FFDMs) in DICOM format.
Each image is expert-annotated with regions of interest (ROIs), which are labeled as masses, calcifications, distortions
or spiculated regions.
The dataset provides both:
- annotations related to ROIs of each image (including lesion properties such as size, intensity statistics, and contour
points in pixel and millimeter spaces), avaiable in XML format.
- metadata (e.g., BIRADs class) in CSV format.

The high-quality annotations and comprehensive lesion diversity made 
INbreast dataset particularly suitable for our system tasks.

## 2. Experiment results
... 
TO PUT image of the table with results compared between different versions of model
- yolov8n
- yolov8m


## 3. How to use
It is needed a working Python environment to run the code.
We use conda virtual environments to manage the project dependencies 
in isolation. You can install our dependencies without causing 
conflicts with your setup.

### Requirements
As first step, please install the dependencies needed for running the system.
Recomended: use a virtual environment conda

```shell
pip install -r requirements.txt
```
or sh with also conda????????????

### Data collection 
INbreast dataset is publicly avaiable at this link: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset .
Once you have downloaded and unzipped the data, you have to place inside data/raw/ repository the following folders/files:
- AllDICOMs
- AllXML
- INbreast.csv 

The *data* directory should have the following structure:
```graphql
    mammography_CAD
    │── data/                  # Datasets and preprocessing scripts
    │   │── raw/               
    │   │   │── AllDICOMs/            # DICOMs folder
    │   │   │   │── 20586908_6c613....dcm
    │   │   │   │── 20586934_6c613....dcm
    │   │   │    ...
    │   │   │── AllXML/               # XML folder
    │   │   │   │── 20586908.dcm
    │   │   │   │── 20586934.dcm
    │   │   │   ...
    │   │   └── INbreast.csv/         # CSV file
    │   │
    │   │── processed/         # Preprocessed data 
     ...

```

### Data preprocessing
1. To preprocess the dataset, run the script *dataset_preparation* which provides the conversion
of DICOM images into PNG images, which are then augmented and enhanced by applying **CLAHE** *(Contrast Limited Adaptive Histogram Equalization)*. 
It creates also the JSON file containing all the information related to the images.
Specifically, *json_preparation.py* script extracts and saves the information from both XML files and CSV annotations in a structured JSON format.

    ```shell
    python data/dataset_preparation.py
    ```
   
    After running the script, the directory should have the following structure:
    After running the script, the directory should have the following structure:
   In particular for detection, Point_px data will be used. 
   It is a list of points in pixel space, representing contour points of the masses.

2. To generate the labels for detection and segmentation stage, run the following two scripts:
   ```shell

   ```

3. To split the dataset into train, val, and test sets for detection and segmentation models training, 
   run the following script

   ```shell
       python data/split_dataset.py
   ```
4. To generate the binary masks from the original dataset, for testing classification directly without using results from
segmentation, run the following command
5. 
   ```shell
      python utils/extract_masks.py
   ```

### Running the CAD system

#### 1. Detection and instance segmentation stages (YOLO v8)
Once you have everything ready, we can start to train the system.
- For training the detection system, use the following command:
    ```shell
    python main_det.py --model yolov8n.pt --epochs 80 --batch 8 
    ```
    For testing the system, you can choose to use either one of the different pre-trained versions of the YOLOv8 model: 
    
    *yolov8n.pt*, *yolov8s.pt*, yolov8m.pt


- For training the instance segmentation system, use the following command:
   ```shell
      python main_seg.py --model yolov8n_seg.pt --epochs 80 --batch 8 
   ```

    For testing the system, you can choose to use either one of the different pre-trained versions of the YOLOv8 model for segmentation: 
    
    *yolov8n_seg.pt*, *yolov8s_seg.pt*, yolov8m_seg.pt
    
    

The results will be saved in `runs/` directory

#### 2. Classification stage (ResNet)

- First, we have to split the dataset in *train, val* and *test* sets, running the following command
    ```shell
    python main_class.py --mode split --json_path data/json/augmented.json --out_dir data/json/class
    ```
- Then, for training the classification model, run the command below
    ```shell
    python main_class.py --mode train --json_path data/json/class/train.json --img_dir data/processed/clahePNG --mask_dir data/mass_masks --epochs 80
    ```
- For evaluating the classification model, run the command below 
    ```shell
    python main_class.py --mode eval --json_path data/json/class/val.json --img_dir data/processed/clahePNG --mask_dir data/mass_masks --epochs 80
    ```



### Running the Super Resolution GAN system
- For creating high-resolution and low-resolution pairs, run the following command
 ```shell
    python utils/esrgan_data.py
 ```

- For training the ESRGAN, run the following command
 ```shell
    python main_esrgan.py --task train
 ```

- For evaluating the ESRGAN, run the following command
 ```shell
    python main_esrgan.py --task eval
 ```

