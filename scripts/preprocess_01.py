import zarr 
import numpy as np
import imageio
import glob
import os
raw_files = sorted(glob.glob("/mnt/efs/aimbl_2025/student_data/S-LS/raw_bacteria/*.tif"))
mask_files = sorted(glob.glob("/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria/*.tif"))
raw_images = [imageio.imread(f) for f in raw_files]
mask_images= [imageio.imread(f) for f in mask_files]
print (len(raw_images))



import os
from pathlib import Path
import glob
import numpy as np
import imageio
import zarr

# Inputs
mask_dir = "/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria"
raw_dir  = "/mnt/efs/aimbl_2025/student_data/S-LS/raw_bacteria"

# Output Zarr store
store_path = "/mnt/efs/aimbl_2025/student_data/S-LS/my_data2.zarr"
zarr.open(store_path, zarr_version=2)

# Open/create root and subgroups
root = zarr.group(store_path)
masks_grp = root.create_group("mask_bacteria") 
raw_grp   = root.create_group("bacteria")    

def stack_and_write(img_folder, group, dataset_name, chunks=(1,552,688)):
    files = sorted(glob.glob(os.path.join(img_folder, "*.tif")))
    N = len(files)
    # Load all images, stack to (N, H, W)
    stack = np.stack([imageio.imread(f) for f in files], axis=0)
    N, H, W = stack.shape
    

    ds = group.create_array(name=dataset_name, shape=(N, H, W), chunks=chunks, dtype="uint16")
    ds[...] = stack

    ds.attrs["filenames"] = [Path(f).name for f in files]

    print(f" Wrote '{group.path}/{dataset_name}' | shape={(N,H,W)} | chunks={chunks} | dtype=uint16")

# Build the two stacks
stack_and_write(mask_dir, masks_grp, "masks_bacteria_stack", chunks=(1,552,688))
stack_and_write(raw_dir,  raw_grp,   "raw_bacteria_stack",   chunks=(1,552,688))


print(root.tree())
print("hola")
