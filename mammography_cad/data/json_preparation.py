import xml.etree.ElementTree as ET
import json
import os
import pandas as pd
import numpy as np
import copy
import collections

''' Modules/section for extracting data from CSV file and save them in json dictionary'''
# Load the CSV data (INBreast.csv)
def load_csv(file_path):
    return pd.read_csv(file_path, delimiter=';')

# Convert all numerical values from int64 (pandas default) to int for compatibility with JSON
def convert_to_int(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_to_int(value)
        elif isinstance(value, np.int64):
            d[key] = int(value)  # Convert numpy int64 to native Python int
    return d

'''Modules/section for extracting data from XML files and save them into json dictionary'''
def get_sibling_value(element, key_name, value_type):
    ''' Find the value of the sibling element that follows a specific key tag. '''
    found_key = False  # Flag to track when we find the key
    for child in element:
        if found_key:
            if value_type == 'real':
                return float(child.text) if child.text else None
            elif value_type == 'integer':
                return int(child.text) if child.text else None
            elif value_type == 'string':
                return child.text
            elif value_type == 'array':
                return child  # Return the array element
            return None  # Unknown type, return None
        if child.tag == 'key' and child.text == key_name:
            found_key = True  # Found the key, next element should be its value
    return None  # Return None if the key or its sibling is not found

def parse_xml_file(file_path):
    ''' Parses a single XML file and extracts the annotation data. '''
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    image_data = {}
    images = root.find('.//array')

    if images is None:
        print(f"Skipping {file_path}: No 'array' element found in the XML structure.")
        return {}

    for image in images.findall('./dict'):
        image_index = get_sibling_value(image, 'ImageIndex', 'integer')
        num_rois = get_sibling_value(image, 'NumberOfROIs', 'integer')

        if image_index is None or num_rois is None:
            print(f"Skipping entry in {file_path}: Missing ImageIndex or NumberOfROIs.")
            continue

        rois = []
        rois_array = get_sibling_value(image, 'ROIs', 'array')  # Get the array of ROIs

        if rois_array is None:
            print(f"No ROI array found for image {image_index} in {file_path}")
            continue

        for roi in rois_array.findall('./dict'):
            roi_info = {}

            try:
                roi_info['Area'] = get_sibling_value(roi, 'Area', 'real')
                roi_info['Center'] = get_sibling_value(roi, 'Center', 'string')
                roi_info['Dev'] = get_sibling_value(roi, 'Dev', 'real')
                roi_info['IndexInImage'] = get_sibling_value(roi, 'IndexInImage', 'integer')
                roi_info['Max'] = get_sibling_value(roi, 'Max', 'real')
                roi_info['Mean'] = get_sibling_value(roi, 'Mean', 'real')
                roi_info['Min'] = get_sibling_value(roi, 'Min', 'real')
                roi_info['Name'] = get_sibling_value(roi, 'Name', 'string')
                roi_info['NumberOfPoints'] = get_sibling_value(roi, 'NumberOfPoints', 'integer')

                # Extract 'Point_px' array
                point_px_array = get_sibling_value(roi, 'Point_px', 'array')
                roi_info['Point_px'] = [point.text for point in point_px_array.findall('string')] if point_px_array else []

                roi_info['Total'] = get_sibling_value(roi, 'Total', 'real')
                roi_info['Type'] = get_sibling_value(roi, 'Type', 'integer')

            except Exception as e:
                print(f"Error extracting ROI data in {file_path}: {e}")
                continue  # Skip this ROI if there is an error

            rois.append(roi_info)

        image_data[filename] = {
            'NumberOfROIs': num_rois,
            'ROIs': rois
        }

    return image_data

def parse_all_xml_files(xml_folder):
    '''Processes all XML files in the given folder and saves annotations in JSON format. '''
    all_annotations = {}
    total_xml_files = 0  # Count of XML files processed

    # Loop through all XML files in the folder
    for file_name in os.listdir(xml_folder):
        if file_name.endswith('.xml'):
            file_path = os.path.join(xml_folder, file_name)
            xml_data = parse_xml_file(file_path)

            if xml_data:
                all_annotations.update(xml_data)  # Merge data from each file
                total_xml_files += 1

    # Add total XML files count to the final dictionary
    all_annotations['total_xml_files'] = total_xml_files

    # Save as JSON
    with open("json/xml.json", "w") as json_file:
        json.dump(all_annotations, json_file, indent=4)

    print(f"Processed {total_xml_files} XML files. Saved as 'xml.json'.")


'''Modules for creating an augmented version of the json dictionary (noise and contrast)'''
def augment_dataset(dataset):
    '''Create augmented copies without modifying the original dataset'''
    augmented_dataset = {}

    for key, value in dataset.items():
        original_instance = copy.deepcopy(value)  # Ensure original data is not modified
        original_instance['FileName'] = str(key)  # Set correct filename
        augmented_dataset[key] = original_instance  # Add original instance to new dataset

        # Create copies with different suffixes
        for suffix in ['_noise', '_contrast']:
            new_key = f"{key}{suffix}"
            augmented_instance = copy.deepcopy(value)  # Copy original entry
            augmented_instance['FileName'] = str(new_key)  # Update FileName
            augmented_dataset[new_key] = augmented_instance  # Store augmented instance

    return augmented_dataset

#-------------------------------------------------------------------------------------------------#
def main():

    csv_file_path = 'raw/INbreast.csv'
    data = load_csv(csv_file_path)

    ''' Extracting data from CSV file and save them in json dictionary '''
    # Extract necessary columns from CSV
    biradsclass = data['Bi-Rads'].values  # BIRADS class (8th column)
    acrclass = data['ACR'].values  # ACR class (7th column)
    biradsclassFilename = data['File Name'].values  # File Name (6th column)
    date = data['Acquisition date'].values  # Acquisition Date (5th column)
    view_position = data['View'].values  # View (4th column)

    # Scan the DICOM files in the AllDICOMs directory
    dicom_dir = 'raw/AllDICOMs'
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

    # Initialize INbreast structure as a dictionary
    INbreast = {}

    # Process each DICOM file and extract necessary information
    for k, dicom_file in enumerate(dicom_files):
        print(f"Processing file {k + 1}/{len(dicom_files)}: {dicom_file}")

        # Extract Patient ID and File Name from the DICOM filename
        parts = dicom_file.split('_')
        file_name = parts[0]
        patient_id = parts[1]

        # Print debug information for filename matching
        print(f"  DICOM File: {dicom_file}")
        print(f"  Extracted File Name: {file_name}")

        # Create a dictionary to store DICOM info
        dicom_entry = {
            'Index': k + 1,  # Index of the file
            'FileName': file_name,  # Extracted File Name from filename
            'PatientID': patient_id,  # Extracted Patient ID from filename
            'ViewPosition': '',  # (from CSV)
            'Date': '',
            'BIRADS': '',
            'ACR': '',
        }

        # Find matching row in CSV by File Name
        file_name_numeric = ''.join(filter(str.isdigit, file_name))  # Extract only digits from DICOM filename
        matching_idx = np.where(np.char.strip(biradsclassFilename.astype(str)) == file_name_numeric)[0]  # Matching by numeric File Name

        # Print debug information for CSV match
        print(f"  Matching indices in CSV for numeric File Name {file_name_numeric}: {matching_idx}")

        if matching_idx.size > 0:
            # If a match is found, assign the ViewPosition, Date, and BIRADS from CSV
            dicom_entry['ViewPosition'] = view_position[matching_idx[0]]
            dicom_entry['Date'] = date[matching_idx[0]]
            dicom_entry['BIRADS'] = biradsclass[matching_idx[0]]
            dicom_entry['ACR'] = acrclass[matching_idx[0]]

        else:
            print(f"  No match found for File Name {file_name_numeric} in the CSV")

        # file_name as a key for the dictionary
        INbreast[file_name] = dicom_entry

        # Convert all dictionary entries to native Python types before saving to JSON
        INbreast = {file_name: convert_to_int(entry) for file_name, entry in INbreast.items()}

        # Save the data as a JSON file
        json_file_path = 'json/csv.json'

        # Saving the sorted INbreast data to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(INbreast, json_file, indent=4)

        print(f"\nINbreast data has been saved as 'csv.json'")


    '''Extracting data from XML files and save them into json dictionary'''
    # from XML to json
    xml_folder = "raw/AllXML"
    parse_all_xml_files(xml_folder)

    '''Creating a json file containing a dictionary with information from both csv.json and xml.json'''
    # Load CSV-based image data (contains all images, including those without ROIs)
    with open("json/csv.json", "r") as f:
        inbreast_data = json.load(f)

    # Load XML-based ROI annotations (only contains images with ROIs)
    with open("json/xml.json", "r") as f:
        annotations_data = json.load(f)

    # Create a dictionary to store the merged data
    merged_data = {}

    # Iterate through the INbreast data (csv.json contains all images)
    for file_name, inbreast_entry in inbreast_data.items():
        if file_name in annotations_data:
            # Image has ROI data
            merged_entry = inbreast_entry.copy()
            merged_entry["NumberOfROIs"] = annotations_data[file_name]["NumberOfROIs"]
            merged_entry["ROIs"] = annotations_data[file_name]["ROIs"]
        else:
            # Image has no ROIs, add empty ROI info
            merged_entry = inbreast_entry.copy()
            merged_entry["NumberOfROIs"] = 0
            merged_entry["ROIs"] = []

        # Add to the final merged dataset
        merged_data[file_name] = merged_entry

    # Save the merged data to dataset.json
    with open("json/dataset.json", "w") as merged_file:
        json.dump(merged_data, merged_file, indent=4)

    '''Creating an augmented version of the json dictionary (noise and contrast)'''
    # Load the dataset.json file
    with open('json/dataset.json', 'r') as file:
        dataset = json.load(file)


    # Augment the dataset
    augmented_data = augment_dataset(dataset)

    # Check the number of original and augmented entries
    print(f"Original dataset size: {len(dataset)}")
    print(f"Augmented dataset size: {len(augmented_data) - len(dataset)}")  # Should be 820
    print(f"Total dataset size after augmentation: {len(augmented_data)}")  # Should be 1230

    # Check for duplicates
    duplicates = [key for key, count in collections.Counter(augmented_data.keys()).items() if count > 1]
    if duplicates:
        print(f"Found duplicates: {duplicates}")

    # Save the updated dataset back to a new file
    with open('json/augmented.json', 'w') as file:
        json.dump(augmented_data, file, indent=4)

    print("Dataset has been augmented and saved as 'augmented.json'.")

if __name__ == "__main__":
    main()