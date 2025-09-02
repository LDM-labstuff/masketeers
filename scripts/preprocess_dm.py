import zarr 
import numpy as np
import imageio
import os

# Helper Functions for Data Preprocessing
def crop_tiles(image, crop_size):
    #Function for convertion of large images into series of tiles stacked along the 0 axis
    h, w = image.shape
    h_tiles = (h + crop_size - 1) // crop_size  # ceil division
    v_tiles = (w + crop_size - 1) // crop_size

    # Padding
    pad_h = h_tiles * crop_size - h
    pad_w = v_tiles * crop_size - w
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')

    # Preallocate output array
    tiles = np.empty((h_tiles * v_tiles, crop_size, crop_size), dtype=image.dtype)

    idx = 0
    for i in range(h_tiles):
        for j in range(v_tiles):
            y_start = i * crop_size
            x_start = j * crop_size
            tiles[idx] = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
            idx += 1

    return tiles

def relabel_and_compress (segmentation, start_index = 1):
    # This function sequentially relabelles instances of segmented objects and
    # compresses all class segmentation channels to 1
    for i in range (segmentation.shape[0]):
        if segmentation [i, :, :].max() > 0:
            segmentation [i, :, :] = relabel_sequential (segmentation [i, :, :], offset = start_index)[0]
            start_index = segmentation [i, :, :].max()
        else:
            segmentation [i, :, :] = segmentation [i, :, :]
    segmentation = segmentation.sum (axis = 0)
    return segmentation

def min_max_normalize (image):
    image = np.array (image)
    min = image.min()
    max = image.max()
    image = (image - min)/(max - min)
    return image

#Getting List of Image File Names
path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/raw_1_channel_images/"
tiff_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tiff') or f.endswith('.tif')])


# Sorting files into conditions
conditions = ["AD", "CBD", "PiD", "PSP"]
condition_files = {}
for condition in conditions:
    condition_files[condition] = [f for f in tiff_files if condition in f]

# Getting files for segmentation channels
segmentation_path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/classes_2/"
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

# Creating additional arrays with cropped images and compressed segmentation channels
for conditions in list (root.keys()):
    images = root [conditions].keys()
    for fov in images:
        fov_group = root [conditions][fov]
        x = root[conditions][fov] ["x"][:]
        y = root[conditions][fov]["y"][:].astype ("int16")

        y_compressed = relabel_and_compress (y)
        x_normalized = min_max_normalize (x)
        x_cropped = crop_tiles (x_normalized, crop_size = 512)
        y_cropped = crop_tiles (y_compressed, crop_size = 512)
        fov_group.create_array (name = "x_cropped", data = x_cropped)
        fov_group.create_array (name = "y_cropped", data = y_cropped)



