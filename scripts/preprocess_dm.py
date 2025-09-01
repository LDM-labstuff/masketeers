import zarr 
import numpy as np
import imageio
import os

#Getting List of Image File Names
path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/raw_1_channel_images/"
tiff_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tiff') or f.endswith('.tif')])


# Sorting files into conditions
conditions = ["AD", "CBD", "PiD", "PSP"]
condition_files = {}
for condition in conditions:
    condition_files[condition] = [f for f in tiff_files if condition in f]

# Getting files for segmentation channels
segmentation_path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/classes/"
seg_folders = sorted(os.listdir (segmentation_path))
n_channels = len (seg_folders)

# Creating .zarr object

#Initiating zarr
zarr_path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/zarr_storage/"
test_zarr = zarr_path +"tauopathies.zarr"

root = zarr.open(test_zarr, mode='w')
for condition, file_list in condition_files.items():
    # Create a group for each condition
    group = root.create_group(condition)
    # Loop through each image file in the list
    for fname in file_list:
        # Load the image as a NumPy array
        img = imageio.imread(fname)
        #Getting image name
        image_name = fname.split ("/")[-1]
        #Getting image shape
        image_shape = img.shape
        #Defining the shape of instant segmentation array
        segment_shape = ((n_channels,) +image_shape)

        

        # Create an image group inside the condition group
        group_fov = group.create_group(image_name)
        #Create source and target array for image
        group_fov.create_array(name = "x", data = img)
        group_fov.create_array (name = "y", data = np.zeros(segment_shape))
        # Getting instance segmentation images when available
        for i in range (0, n_channels):
            channel_path = os.path.join (segmentation_path, seg_folders[i], image_name)
            print (channel_path)
            if os.path.exists (channel_path):
                group_fov["y"][i, :, :] = imageio.imread (channel_path)



