import imageio
import numpy as np
import os

# Input and output paths
in_path  = "/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria/mask_012_bacteria.tif"
out_path = "/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria/mask_012_bacteria_uint16.tif"

# Load the mask
arr = imageio.imread(in_path)
print("Original dtype:", arr.dtype, "max value:", arr.max())

# Check before converting
if arr.max() > 65535:
    raise ValueError(f" {in_path} has values >65535; cannot safely convert to uint16")
else:
    arr16 = arr.astype(np.uint16)
    imageio.imwrite(out_path, arr16)
    print(f" Saved {out_path} as uint16")
