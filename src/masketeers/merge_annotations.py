import zarr 
import numpy as np
import imageio
import glob
import os
raw_files = sorted(glob.glob("/mnt/efs/aimbl_2025/student_data/S-LS/raw_bacteria/*.tif"))
mask_files = sorted(glob.glob("/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria/*.tif"))
raw_images = [imageio.imread(f) for f in raw_files]
mask_images= [imageio.imread(f) for f in mask_files]
def merge_instance_masks(arr1: np.ndarray,
                         arr2: np.ndarray,
                         prefer: str = "arr1"):
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shapes must match, got {arr1.shape} vs {arr2.shape}")
    a1 = np.asarray(arr1).astype(np.uint32, copy=False)
    a2 = np.asarray(arr2).astype(np.uint32, copy=False)
    max1 = int(a1.max())  # max label in first mask (0 is background)
    offset = max1
    a2_off = a2.copy()
    mask2_fg = a2_off > 0
    a2_off[mask2_fg] = a2_off[mask2_fg] + offset
    if prefer == "arr1":
        # Keep arr1 where it already has labels; fill only where arr1 is 0
        merged = a1.copy()
        fill = (merged == 0) & (a2_off > 0)

        merged[fill]=a2_off[fill]
        
    else:
        raise ValueError("prefer must be 'arr1', 'arr2', or 'error'.")
    return merged, offset

def save_merged_mask(merged, base_path, extra_path):
    folder = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    root, ext = os.path.splitext(base_name)
    new_name = f"{root}_merged{ext}"
    save_path = os.path.join(folder, new_name)
    imageio.imwrite(save_path, merged.astype("uint16"))
    print(f"Saved merged mask to: {save_path}")
    return save_path

mask_folder="/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria"

arr1 = imageio.imread(os.path.join(mask_folder, "mask_009_bacteria.tif"))
arr2 = imageio.imread(os.path.join(mask_folder, "mask_009_bacteria-2.tif"))

merged, _ = merge_instance_masks(arr1, arr2, prefer="arr1")

save_merged_mask(merged, "mask_009_bacteria.tif", "mask_009_bacteria-2.tif")