import torch
import numpy as np
import random
import zarr
from scipy.ndimage import distance_transform_edt, map_coordinates
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2

def load_data (zarr_path):
    # Helper function for uploading stack of tiles from zarr 
    x_arrays = []
    y_arrays = []
    root = zarr.open (zarr_path)
    for conditions in list (root.keys()):
        images = root [conditions].keys()
        for fov in images:
            x = root[conditions][fov]["x_cropped"][:]
            y = root[conditions][fov]["y_cropped"][:].astype ("int16")
            #y1 = root[conditions][fov]["y_cropped"][:].astype ("int64")
            #assert (y == y1).all()
            # print (x.dtype, y.dtype)
            x_arrays.append (x)
            y_arrays.append (y)
    x_array = np.concatenate (x_arrays)
    y_array = np.concatenate (y_arrays)
    return x_array, y_array

def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""
    dims = len(labels.shape)
    # Create a placeholder array of infinite distances
    distances = np.ones(labels.shape, dtype=np.float32) * np.inf
    for axis in range(dims):
        # Here we compute the boundaries by shifting the labels and comparing to the original labels
        # This can be visualized in 1D as:
        # a a a b b c c c
        #   a a a b b c c c
        #   1 1 0 1 0 1 1
        # Applying a half pixel shift makes the result more obvious:
        # a a a b b c c c
        #  1 1 0 1 0 1 1
        bounds = (
            labels[*[slice(None) if a != axis else slice(1, None) for a in range(dims)]]
            == labels[
                *[slice(None) if a != axis else slice(None, -1) for a in range(dims)]
            ]
        )
        # pad to account for the lost pixel
        bounds = np.pad(
            bounds,
            [(1, 1) if a == axis else (0, 0) for a in range(dims)],
            mode="constant",
            constant_values=1,
        )
        # compute distances on the boundary mask
        axis_distances = distance_transform_edt(bounds)

        # compute the coordinates of each original pixel relative to the boundary mask and distance transform.
        # Its just a half pixel shift in the axis we computed boundaries for.
        coordinates = np.meshgrid(
            *[
                (
                    range(axis_distances.shape[a])
                    if a != axis
                    else np.linspace(
                        0.5, axis_distances.shape[a] - 1.5, labels.shape[a]
                    )
                )
                for a in range(dims)
            ],
            indexing="ij",
        )
        coordinates = np.stack(coordinates)

        # Interpolate the distances to the original pixel coordinates
        sampled = map_coordinates(
            axis_distances,
            coordinates=coordinates,
            order=3,
        )

        # Update the distances with the minimum distance to a boundary in this axis
        distances = np.minimum(distances, sampled)

    # Normalize the distances to be between -1 and 1
    distances = np.tanh(distances / scale)

    # Invert the distances for pixels in the background
    distances[labels == 0] *= -1
    return distances

class CropDataset(Dataset):
    def __init__(self, zarr_path, transform = None, img_transform = None):
        
        
        self.zarr_path = zarr_path

        self.x, self.y = load_data(self.zarr_path)
        self.transform = transform
        self.img_transform = img_transform
        
       

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]  
        seg = self.y[idx] 

        img = torch.tensor(img).unsqueeze(0)
        seg = torch.tensor(seg).unsqueeze(0)
        
        if self.transform:
            img = self.transform (img)
            seg = self.transform(seg)
        
        if self.img_transform:
            img = self.img_transform(img)

        sdt = compute_sdt (seg) 

    

        return torch.tensor(img, dtype=torch.float32), torch.tensor(sdt, dtype=torch.float32)

