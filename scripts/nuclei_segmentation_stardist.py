"""
To configure the environment run:
pip install StarDist "tensorflow>2" tqdm zarr "numpy<2"
"""

from tqdm import tqdm
import zarr as zarr
import imageio
import numpy as np
import pathlib as pl
from imageio import imread
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

## Defining functions

# Function to segment nuclei
def segment_nuclei(treatment, path_to_zarr, model):
    """Segment nuclei in 2D images using a StarDist2D model.
    Args:
        treatment (str): The treatment group to process.
        path_to_zarr (str): Path to the Zarr file containing images.
        model (StarDist2D): Pretrained StarDist2D model for segmentation."""
    # Determine the number of images in the treatment
    num_images = z[treatment]["max"].shape[0]
    sample_image = z[treatment]["max"][0][0]
    print(f"Sample image shape: {sample_image.shape}")
    # create numpy array to contain the labels (masks)
    treatment_mask_array = np.zeros((num_images, sample_image.shape[0], sample_image.shape[1]), dtype=np.int32)
    print(f"Segmenting nuclei for Treatment: {treatment}, Group: max")
     # iterate through the images to segment
    for i in tqdm(range(num_images)):
        # Obtain DAPI image
        image = z[treatment]["max"][i][0]
        # Normalize the image
        image = normalize(image)
        # Run the model
        labels, _ = model.predict_instances(image)
        # Store the labels in the treatment mask array
        treatment_mask_array[i] = labels
        # Increment the counter
        i += 1
    return treatment_mask_array

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

if __name__ == "__main__":
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    path_to_zarr = '/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr'

    z = zarr.open(path_to_zarr, mode='a')

    treatments, groups = explore_zarr(z)
    print(treatments, groups)

    for treatment in treatments:
        print(f"Segmenting nuclei for Treatment: {treatment}, Group: max")
        treatment_mask_array = segment_nuclei(treatment, z, model)
        # Save or process the treatment_mask_array as needed
        z[treatment]['masks'] = treatment_mask_array
        print(f"Finished segmenting nuclei for Treatment: {treatment}, Group: max. Array shape was {treatment_mask_array.shape}")


