"""
To configure the environment run:
pip install napari?
"""

import napari
import zarr
import pathlib as pl
import tqdm
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import regionprops

# Function to explore the Zarr file
def explore_zarr(z):
    """
    Explore the contents of a Zarr file.
    Args:
        z (zarr.Group): The Zarr group to explore.
    """
    # Explore the contents of the Zarr file
    treatments = list(z.keys())
    groups = list(z[treatments[0]].keys())
    for treatment in treatments:
        print(f"Treatment: {treatment}")
        for group in groups:
            print(f"  Group: {group}")
            print(f"    Shape: {z[treatment][group].shape}")
    return treatments, groups

# Clear border and pad images
def clear_and_pad(mask, raw_image):
    cleared_border = clear_border(mask)
    padded_mask = np.pad(cleared_border, pad_width=64, mode='constant', constant_values=0)
    padded_raw = np.pad(raw_image, pad_width=((0, 0),(0, 0),(64, 64), (64, 64)), mode='symmetric')
    return padded_mask, padded_raw

def crop_and_extract_single_nuclei(padded_mask, padded_raw):
    list_of_cropped_raw_per_image = []
    list_of_cropped_mask_per_image = []
    list_of_cropped_mask_maskout_per_image = []
    list_of_cropped_raw_maskout_per_image = []

    for i,props in enumerate(regionprops(padded_mask)):
        print(f"Region properties {props}")
        centroid = props.centroid
        print(f"Centroid for label {props.label}: {centroid}")
        if np.isnan(centroid[0]):
            print(f"Centroid is NaN for label {props.label}, skipping.")
            continue
        crop_box = [int(centroid[0])-64, int(centroid[0])+64, int(centroid[1])-64, int(centroid[1])+64]
        raw_crop = padded_raw[:,:,crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
        mask_crop = padded_mask[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
        mask_bool = mask_crop == props.label
        raw_maskout=raw_crop.copy()
        raw_maskout[:, :, ~mask_bool] = 0
        mask_maskout= mask_crop.copy()
        mask_maskout[~mask_bool] = 0
        cropped_raw_np=raw_crop
        cropped_raw_maskout_np= raw_maskout
        cropped_mask_np = mask_crop
        cropped_mask_maskout_np = mask_maskout
        list_of_cropped_raw_per_image.append(cropped_raw_np)
        list_of_cropped_mask_per_image.append(cropped_mask_np)
        list_of_cropped_mask_maskout_per_image.append(cropped_mask_maskout_np)
        list_of_cropped_raw_maskout_per_image.append(cropped_raw_maskout_np)
    return list_of_cropped_mask_per_image, list_of_cropped_raw_per_image, list_of_cropped_mask_maskout_per_image, list_of_cropped_raw_maskout_per_image
   

# Main execution
if __name__ == "__main__":
    path_to_zarr = '/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr'
    #load zarr file
    z = zarr.open(path_to_zarr, mode='r+')

    treatments, groups = explore_zarr(z)
    print(treatments, groups)
    for treatment in tqdm.tqdm(treatments):
        list_of_cmasks_per_treatment =[]
        list_of_cimages_per_treatment = []
        list_of_cmasks_maskout_per_treatment = []
        list_of_cimages_maskout_per_treatment = []
        for i in range(z[treatment]['masks'].shape[0]):
            raw_image = z[treatment]['raw'][i]
            mask = z[treatment]['masks'][i]
            padded_mask, padded_raw = clear_and_pad(mask, raw_image)
            props_num = len(regionprops(padded_mask))

            if props_num > 0:
                list_of_cropped_mask_per_image, list_of_cropped_raw_per_image, list_of_cropped_mask_maskout_per_image, list_of_cropped_raw_maskout_per_image = crop_and_extract_single_nuclei(padded_mask, padded_raw)
                print(f"Shape of cropped images lists created: cropped raw per image: {list_of_cropped_raw_per_image}, cropped mask per image: {list_of_cropped_mask_per_image}, cropped raw maskout per image: {list_of_cropped_raw_maskout_per_image}, cropped mask maskout per image: {list_of_cropped_mask_maskout_per_image}")
                # Append per_image lists to per_treatment lists
                list_of_cmasks_per_treatment.append(list_of_cropped_mask_per_image)
                list_of_cimages_per_treatment.append(list_of_cropped_raw_per_image)
                list_of_cimages_maskout_per_treatment.append(list_of_cropped_raw_maskout_per_image)
                list_of_cmasks_maskout_per_treatment.append(list_of_cropped_mask_maskout_per_image)
            else:
                print(f"No nuclei detected in image {i} of treatment {treatment}. Skipping cropping.")
        # Convert lists to numpy arrays
        list_of_cmasks_per_treatment_np = np.concatenate(list_of_cmasks_per_treatment, axis=0)
        list_of_cimages_per_treatment_np = np.concatenate(list_of_cimages_per_treatment, axis=0)
        list_of_cmasks_maskout_per_treatment_np = np.concatenate(list_of_cmasks_maskout_per_treatment, axis=0)
        list_of_cimages_maskout_per_treatment_np = np.concatenate(list_of_cimages_maskout_per_treatment, axis=0)
        zarr.create_array(store="/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr", name=f"{treatment}/cmasks", data=list_of_cmasks_per_treatment_np)
        zarr.create_array(store="/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr", name=f"{treatment}/cimages", data=list_of_cimages_per_treatment_np)
        zarr.create_array(store="/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr", name=f"{treatment}/cmasks_maskout", data=list_of_cmasks_maskout_per_treatment_np)
        zarr.create_array(store="/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr", name=f"{treatment}/cimages_maskout", data=list_of_cimages_maskout_per_treatment_np)
