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
This project presents a Computer-Aided Diagnosis (CAD) system for mammogram analysis that integrates deep learning techniques for image enhancement, mass detection, segmentation, classification. The system is trained and evaluated on the INBreast dataset, that due to its high-resolution images and detailed ground truth annotations, is a particular suitable mammographic database for this purpose. Our proposed pipeline starts from some pre-processing that includes data augmentation techniques, such as contrast enhancement and noise addition, and the utilization of an ESRGAN for image enhancement and synthetic data generation, in order to improve model robustness. Successively it exploits a YOLOv8-based mass detection and instance segmentation model and a ResNet-based classifier to differentiate between benign and malignant masses. 
RESULTS


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
This project is implemented on a *Linux-based operating system* (Ubuntu 22.04.5 LTS, 64-bit).
A working Python environment is needed to run the code. 

Thus, It is recommended to create a Conda environment for this purpose.


### Requirements
As first step, please install the dependencies needed for running the system.
```shell
pip install -r requirements.txt
```

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
To preprocess the dataset, go to *data/* directory and run dataset_preparation.py script. 
This step can be done running consequently the following commands
```shell
    cd data
    python dataset_preparation.py
```
   And then come back to the main directory

```shell 
    cd -
```
After the execution of this preprocessing step, the PNG processed images results to be located in `data/processed/` directory, while the JSON files are located in `data/json/` directory.

Specifically, it generates:
- the converted version of INbreast images from DICOM to PNG format (*`AllPNG`*)
- the augmented version of the dataset (augmentations: contrast adjustment, noise addition) in PNG format (*`augmentedPNG`*)
- the previous augmented version of the dataset enhanced using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) (*`clahePNG`*)
- the file JSON containing a dictionary with information from INbreast.csv and from the annotations in XML format (*`dataset.json`*)
- the file JSON containing a dictionary with information from INbreast.csv and from the annotations in XML format for the augmented dataset (*`augmented.json`*)


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

