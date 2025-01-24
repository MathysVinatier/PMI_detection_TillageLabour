import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import radiomics.featureextractor

from skimage.feature import graycomatrix, graycoprops
from skimage import exposure


params = {
    'featureClass': {

        'glcm': [
            'Contrast', 'Correlation', 'JointEnergy', 'JointEntropy'
        ],
        'glrlm': [
            'ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity', 'RunLengthNonUniformity', 'RunPercentage',
        ],
        'glszm': [
            'SmallAreaEmphasis', 'LargeAreaEmphasis', 'GrayLevelNonUniformity', 'ZoneVariance', 'ZonePercentage',
        ],
        'ngtdm': [
            'Coarseness', 'Contrast', 'Complexity', 'Strength'
        ]
    },
    'setting': {
        'binWidth': 1
    }
}


# =====================================================================================================================================
def calculate_features_skimage(image_gray):
    """
    ex => image_gray = cv2.imread("Sentinel2_Images_Beauvais/03_06_23_gray.png", cv2.IMREAD_GRAYSCALE)
    """
    # Calcul de la GLCM
    glcm = graycomatrix(image_gray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    glcm_normalized = glcm / glcm.sum()

    # Propriétés texturales
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))  # Entropie manuelle

    print(f"Contraste : {contrast}, Entropie : {entropy}")

def calculate_laplacian(image):
    """ 
    Calculate if the image given is clear 
    
    Parameters:
    ----------
    image: the image (.png)

    Returns:
    -------
    None

    This will print if the resolution of the image is enough or not, 
    based on the variance of the laplacian
    """

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    print(f"Variance du Laplacien : {variance}")

    if variance < 100: 
        print("L'image est probablement floue.")
    else:
        print("L'image semble nette.")
# =====================================================================================================================================

def preprocess_image(image_path):
    """
    Preprocess a RGB image : conversion in grey levels and normalisation
    of the light

    Parameters:
    ----------
    image_path: the path of the image 

    Returns:
    -------
    None

    This will create a grayscale image and add it in the same file as the image
    """
    # Image of reference for the normalisation of the light
    image_reference_gray = cv2.imread("Sentinel2_Images_Beauvais/03_06_23_gray.png", cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.imread(str(image_path))
    image_path = Path(image_path)

    if image_rgb is None:
        raise FileNotFoundError(f"L'image {image_path} n'a pas pu être chargée.")

    # Conversion RGB => Grey Levels
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY) 
    # Normalisation of the light, based on the reference image
    matched_image = exposure.match_histograms(image_gray, image_reference_gray) 
    
    mask_name = image_path.stem[:-4] + "_gray" + image_path.suffix 
    cv2.imwrite(str(image_path.parent / "grays" / mask_name), matched_image)


def create_mask(image_path):    
    """
    Create a mask for the image given

    Parameters:
    ----------
    image_path: the path of the image 

    Returns:
    -------
    None

    This will create a mask and add it in the same file as the image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_path = Path(image_path)

    # Creation of the mask
    binary_mask = np.where(image > 0, 1, 0).astype(np.uint8)
    mask_name = image_path.stem[:-4] + "_mask" + image_path.suffix
    cv2.imwrite(str(image_path.parent / "masks" / mask_name), binary_mask * 255)


def create_pathfiles_csv(folder_masks, folder_grays):
    """
    Create a CSV containig the paths of the images/masks

    Parameters:
    ----------
    folder: the path of the folder containing all the images

    Returns:
    -------
    None

    This will create a CSV, linking each image to its mask
    """
    image_paths = []
    mask_paths = []
    data = []

    for filename in os.listdir(folder_grays):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_grays, filename)
            image_paths.append(file_path)

    for filename in os.listdir(folder_masks):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_masks, filename)
            mask_paths.append(file_path)

    # for filename in os.listdir(folder):
    #     if filename.endswith(".png"):
    #         file_path = os.path.join(folder, filename)
    #         if filename.endswith("_mask.png"):
    #             mask_paths.append(file_path)
    #         elif filename.endswith("_gray.png"):
    #             image_paths.append(file_path)

    # Association image/mask based on their names
    for image_path in image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = next((mask for mask in mask_paths if os.path.splitext(os.path.basename(mask))[0] == image_id[:-5] + "_mask"), None)
        
        if mask_path:
            data.append({'Id': image_id[:-5], 'path_image': image_path, 'path_mask': mask_path})

    df = pd.DataFrame(data)
    df.to_csv('pathfiles_Catillon.csv', index=False)


def process_mask_and_image(image_filepath, mask_filepath, mask_name):
        # Initialize the feature extractor
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)
        extractor.settings['enableDiagnostics'] = False
        # Determine label and label_channel
        label = 255  # Example label, change as needed
        label_channel = None  # For single-channel masks
        # Execute extraction
        result_extraction = pd.Series(extractor.execute(image_filepath, mask_filepath, label=label, label_channel=label_channel))
        # Convert the result to a DataFrame
        feature_vector = pd.DataFrame([result_extraction])
        feature_vector.insert(0, 'mask_name', mask_name)
        # Print or further process the feature vector
        #print(feature_vector)
        return feature_vector


def process_csv(path_csv, path_output):
    results = []
    not_working = []
    df = pd.read_csv(path_csv)
    print("\n==============================================================================")
    print(df)
    print("==============================================================================\n")
    
    for index, rows in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            Id = rows[0]
            image_filepath = rows[1]
            mask_filepath = rows[2]
            result_line = process_mask_and_image(image_filepath, mask_filepath, Id)
            results.append(result_line)
        except:
            not_working.append(rows[0])
    
    combined_df = pd.concat(results, axis=0, ignore_index=True)
    combined_df.to_csv(path_output, index=False)
    if len(not_working) != 0:
        print("\n==================================")
        print(not_working)
        print("==================================\n")
    print("\n => Done !\n")
    return combined_df


"""
#================================================
#         TO CREATE PREPROCESSED IMAGES
#================================================

folder = "Analysis/Sentinel2_Images_Catillon"
image_paths = []
gray_paths = []

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        file_path = os.path.join(folder, filename)
        if "_mask" not in filename and "_gray" not in filename:
            image_paths.append(file_path)
        
for path in image_paths:
    path = Path(path)
    if path.stem + "_gray.png" not in gray_paths:
        preprocess_image(str(path))
"""

"""
#================================================
#         TO CREATE MASKS
#================================================

folder = "Analysis/Sentinel2_Images_Catillon"
image_paths = []
mask_paths = []

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        file_path = os.path.join(folder, filename)
        if "_mask" not in filename and "_gray" not in filename:
            image_paths.append(file_path)
        if "_mask" in filename:
            mask_paths.append(filename)

for path in image_paths:
    path = Path(path)
    if path.stem + "_mask.png" not in mask_paths:
        create_mask(str(path))
"""

"""
#================================================
#         TO CREATE CSV PATHFILES
#================================================

folder_masks = "Analysis/Sentinel2_Images_Catillon/masks"
folder_grays = "Analysis/Sentinel2_Images_Catillon/grays"
create_pathfiles_csv(folder_masks, folder_grays)
"""

"""
#================================================
#         TO EXTRACT FEATURES
#================================================

csv_file =  "pathfiles_Catillon.csv"
output_path = "extraction_results_Catillon.csv"
process_csv(csv_file, output_path)
"""



