import numpy as np
import imageio
from skimage.segmentation import relabel_sequential
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker


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

def load_data (zarr_path):
    # This function takes individual stacks of cropped images and concatelates them into 
    # single image stack
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

def plot_two(img: np.ndarray, intermediate: np.ndarray, label: str):
    """
    Helper function to plot an image and the auxiliary (intermediate)
    representation of the target.
    """
    if img.shape[0] == 2 and len(img.shape) == 3:
        img = np.array([img[0], img[1], img[0] * 0]).transpose((1, 2, 0))
    if intermediate.shape[0] == 4 and len(intermediate.shape) == 3:
        intermediate = np.array(
            [
                (intermediate[0] + intermediate[2]) / 2,
                (intermediate[1] + intermediate[3]) / 2,
                intermediate.sum(axis=0) > 0,  # any affinity is 1
            ]
        ).transpose((1, 2, 0))
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_xlabel("Image", fontsize=20)
    plt.imshow(img)
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_xlabel(label, fontsize=20)
    if len(intermediate.shape) == 2:
        t = plt.imshow(intermediate, cmap="coolwarm")
        cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        _ = [ax.set_xticks([]) for ax in [ax1, ax2]]
        _ = [ax.set_yticks([]) for ax in [ax1, ax2]]
    elif len(intermediate.shape) == 3:
        plt.imshow(intermediate)

    plt.tight_layout()
    plt.show()

def plot_three(
    img: np.ndarray,
    intermediate: np.ndarray,
    pred: np.ndarray,
    label: str = "Target",
    label_cmap=None,
):
    """
    Helper function to plot an image, the auxiliary (intermediate)
    representation of the target and the model prediction.
    """
    if img.shape[0] == 2 and len(img.shape) == 3:
        img = np.array([img[0], img[1], img[0] * 0]).transpose((1, 2, 0))
    if intermediate.shape[0] == 4 and len(intermediate.shape) == 3:
        intermediate = np.array(
            [
                (intermediate[0] + intermediate[2]) / 2,
                (intermediate[1] + intermediate[3]) / 2,
                intermediate.sum(axis=0) > 0,  # any affinity is 1
            ]
        ).transpose((1, 2, 0))
    if pred.shape[0] == 4 and len(pred.shape) == 3:
        pred = np.array(
            [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2, pred.mean(axis=0)]
        ).transpose((1, 2, 0))
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_xlabel("Image", fontsize=20)
    plt.imshow(img)
    ax2 = fig.add_subplot(spec[0, 1])
    if label_cmap is not None:
        ax2.set_xlabel("Labels", fontsize=20)
    else:
        ax2.set_xlabel(label, fontsize=20)

    if len(intermediate.shape) == 2:
        if label_cmap is None:
            plt.imshow(intermediate, cmap="coolwarm")
        else:
            plt.imshow(intermediate, cmap=label_cmap, interpolation="none")
    else:
        plt.imshow(intermediate)
    ax3 = fig.add_subplot(spec[0, 2])
    if label_cmap is not None:
        ax3.set_xlabel(label, fontsize=20)
    else:
        ax3.set_xlabel("Prediction", fontsize=20)

    if len(pred.shape) == 2:
        t = plt.imshow(pred, cmap="coolwarm")
        cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3]]  # remove the xticks
        _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3]]  # remove the yticks
    else:
        plt.imshow(pred)
    plt.tight_layout()
    plt.show()


