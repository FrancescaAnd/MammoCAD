# **Progect title: Models **

The two main tasks:
- **Super-Resolution GAN (SR-GAN)** → Enhance the resolution of mammograms while preserving calcification details.
- **Synthetic Image Generation GAN** → Create realistic mammogram images to augment datasets, particularly those showing calcifications.
## Table of Contents
1. [SR-GAN](#sr-gan)
   - [Datasets](#datasets)
   - [Preprocessing](#preprocessing) 

## SR-GAN
### Datasets
Two input data needed:
- paired low-resolution (LR) and high-resolution (HR) images for the Super-Resolution task
- high-quality images to train the GAN for synthetic data generation.

The datasets used are the following:
- DDSM (Digital Database for Screening Mammography)
- INbreast 

### Preprocessing
For Super-Resolution task 
- Convert HR images to LR using downsampling (bicubic interpolation or Gaussian blurring).
- Normalize pixel values ([0,1] or [-1,1] for GAN stability).
- Resize images to a consistent dimension (e.g., 256x256 or 512x512).

For Synthetic Data Generation task:
- Segment images to focus on areas with calcifications.
- Apply data augmentation (rotation, flipping) to diversify input.



# **IMPLEMENTATION**

## Table of Contents
1. [Ojectives](#Objectives)
2. [Project repository](#project-repository)
2. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
3. [Build Super-Resolution GAN (SRGAN)](#build-super-resolution-GAN-SRGAN)

## Project repository

```graphql
    mammography-GAN
    │── data/                  # Datasets and preprocessing scripts
    │   │── raw/               # Original dataset
    │   │── processed/         # Preprocessed data (after downsampling, etc.)
    │   └── dataset_preparation.py  # Preprocessing script
    │
    │── models/                # GAN model architectures
    │   │── srgan.py           # Super-Resolution GAN model
    │   │── gan_generator.py   # Synthetic Image GAN model
    │   └── discriminator.py   # Shared discriminator for GANs
    │
    │── training/              # Training scripts
    │   │── train_srgan.py     # Training script for SRGAN
    │   │── train_gan.py       # Training script for synthetic data GAN
    │   └── utils.py           # Utility functions (loss functions, metrics)
    │
    │── evaluation/            # Evaluation scripts
    │   ├── evaluate_srgan.py  # PSNR/SSIM for super-resolution
    │   ├── evaluate_gan.py    # FID/Inception Score for synthetic images
    │   └── visualization.py   # Image visualization tools
    │
    │── results/               # Store generated images & model checkpoints
    │   ├── srgan/             # Super-Resolution results
    │   ├── synthetic/         # Synthetic images
    │   └── checkpoints/       # Saved model weights
    │
    │── notebooks/             # Jupyter notebooks for exploration & testing
    │   ├── SRGAN_Training.ipynb
    │   ├── GAN_Training.ipynb
    │   └── Data_Exploration.ipynb
    │
    │── requirements.txt       # Dependencies
    │── README.md              # Project documentation
    │── .gitignore             # Files to ignore in Git
    }
}
```
.

## Data Collection and Preprocessing
We need:
+ Paired low-resolution (LR) and high-resolution (HR) images for SRGAN
+ Dataset for syntetic image generation

### 1.1 Download Datasets
+ CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography): avaiable at [CBIS-DDSM website] (https://www.cancerimagingarchive.net/collection/cbis-ddsm/?utm_source=chatgpt.com) 
+ INbreast: avaiable at [INbreast dataset] (https://www.kaggle.com/datasets/martholi/inbreast?resource=download)

Once downloaded, place the images of the two datasets (original mammography images) inside data/raw repository, as shown below.

```graphql
data/raw/
│── CBIS-DDSM/
│   │── case001.dcm
│   │── case002.dcm
│   └── ...
│
│── INbreast/
│   │── 20586902.dcm
│   │── 41045127.dcm
│   └── ...
```
### 1.2 Dataset preparation: XML Parsing and ROI Extraction from Medical Imaging Data

#### 1.2.1 dataxml.py
**Objective**: The goals of dataxml.py script are the following:
- Parse XML files containing annotation data for medical images, 
- Extract specific Regions of Interest (ROIs) and related details,
- Save the extracted information in a structured JSON format. 

Code Breakdown and Intentions:

Importing Required Libraries:
- The code uses the xml.etree.ElementTree library to parse and navigate through XML files (reading and manipulating XML data in a hierarchical structure)
- The json library is used to output the final results into a readable and structured JSON format
- The os library is used for extracting file names and for using them appropriately in the data structure.

The **parse_xml_file** function is used for:
- reading each given XML file
- extracting relevant data of each ROI
- storing it in a dictionary. 
It processes the XML file line by line, iterating over each ROI entry found within the XML structure.
The script processes multiple XML files, storing each file's parsed data under the name of the XML file, for easy identification of the source file.

The **get_sibling_value()** function is used to extract specific information associated with each ROI. 
It searches for specific keys in the XML file and retrieves their corresponding values. 
It handles different data types (real, integer, string, and array) to ensure that the extracted data is correctly processed and stored.

NumberofROIs: the number of annotations present in the image

Information Extracted:
- Area: The area of the ROI, likely a measure of its size or extent.
- Center: The coordinates of the center of the ROI, typically a tuple indicating the X, Y, Z position in the image space.
- Dev: Represents the deviation value (potentially a measure of intensity variation or heterogeneity within the ROI).
- IndexInImage: A unique index identifying the position of the ROI within the image.
- Max, Min, Mean: These represent the maximum, minimum, and average intensity values within the ROI, respectively.
- Name: ytpe of finding (mass, calcification, distortion, spiculated region)
- NumberOfPoints: The number of data points (likely pixels or sample points) that constitute the ROI.
- Point_mm: A list of points in millimeter space, representing contour points 
- Point_px: A list of points in pixel space, representing contour points
- Total: This could represent the total intensity value within the ROI.??????????
- Type: A numeric identifier for the type of ROI.

Iterating Over Multiple XML Files:
The main loop in the code processes multiple XML files stored in a specified directory. 
- For each XML file, it extracts the filename (without the extension) and uses it as the key for storing the parsed data in the resulting dictionary.
- For each XML file, the script checks whether it contains the necessary XML structure (e.g., the <array> and <dict> tags containing the relevant data). 
- It ensures that only well-formed XML files are processed.

Output:
- Once all XML files are parsed and the data is organized in the dictionary, the code serializes this dictionary into a JSON file (annotations.json). 
- The resulting JSON file contains all the extracted ROI data for each XML file, with the file names as keys and the ROI details as values.
- The JSON structure is organized to reflect the hierarchical nature of the original XML data, making it easy to access specific pieces of information about each image and its associated ROIs.

Error Handling:
- The code includes error handling mechanisms that ensure the script continues to process other data, even if some entries are missing or corrupted.
- This is achieved through try-except blocks, which allow the script to skip problematic data without crashing.

Detailed Structure of the Output JSON:
Each entry in the output JSON corresponds to one XML file, identified by its filename (without extension).
Each file entry contains:
- The number of ROIs present in the file.
- A list of ROIs, where each ROI contains the following attributes:
            Area: The area of the ROI.
            Center: The center of the ROI, typically as a tuple.
            Dev: The deviation value.
            IndexInImage: The index of the ROI in the image.
            Max, Min, Mean: Intensity metrics within the ROI.
            Name: The type of the ROI (e.g., "Calcification").
            NumberOfPoints: The number of points in the ROI.
            Point_mm: Physical coordinates of key points in the ROI.
            Point_px: Pixel coordinates of key points in the ROI.
            Total: The total intensity within the ROI.
            Type: The type of ROI.

The standard deviation (Dev) is a measure of the variation or dispersion of the pixel values within that region. A higher standard deviation means there is more variability in the intensity values of the pixels in the ROI, while a lower standard deviation indicates that the pixel values are more consistent.
For example:
- A low standard deviation (Dev) would indicate a region with relatively uniform intensity (such as a uniform tissue area in an image).
- A high standard deviation (Dev) would indicate a region with more variation in intensity (such as an area with heterogeneity, like a mass or calcification).
In medical imaging, like mammography or other imaging modalities, the standard deviation can help characterize the texture of the tissue in a specific region,
which can be useful for detecting abnormalities or differences in tissue types.

#### 1.2.2 merged_data.py
The goal of this file is to load both the xml annotations and csv data saved in JSON files, extract the required data, and then combine them
Loading JSON Files: 
- The load_json function loads JSON data from a file into a Python dictionary.
Merging Data: 
- For each file_name in inbreast_data, the corresponding ROI information is fetched from annotations_data.
- The relevant data is extracted and merged into a new dictionary, ensuring the final structure is consistent with the desired format (i.e., including the NumberOfROIs and ROIs list).
Processing ROIs: 
- Each ROI in annotations_data[file_name] is processed individually, with its relevant fields (like Area, Center, Dev, Name, Type, etc.) extracted and added to the ROIs list.
- Missing or null values are given default values.
Saving the Merged Data:
- The merged_data dictionary, which now contains all the combined data, is written to a new JSON file (merged_data.json) with proper indentation for readability.

#### 1.2.3 matchPNG.py

### 1.3 Preprocess Data
At this stage, the task is to:
+ Convert HR images to LR images (for SRGAN training)
+ Normalize images ([0,1] range)
+ Resize images to a fixed size (in this case, 256x256)
+ Save them inside `data/processed/`

Since the images in the two dataset are in DICOM (`.dcm`) format, we should convert them into `.png` format before processing.
Run the file `dataset_preparation.py`, which:
+ **Loads DICOM files** (.dcm), extracts the mammography images, and saves them as PNG.
+ **Prepares low-resolution (LR) and high-resolution (HR) images** for training the Super-Resolution GAN.

## Build Super-Resolution GAN (SRGAN)
SRGAN improves image resolution (enhance low-resolution images), while preserving details which are crucials for medical diagnosis





