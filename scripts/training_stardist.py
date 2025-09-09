from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import shutil
# import matplotlib
# matplotlib.rcParams["image.interpolation"] = 'none'
# import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

X = sorted(glob('/mnt/efs/aimbl_2025/student_data/S-LS/raw_bacteria/*.tif'))
Y = sorted(glob('/mnt/efs/aimbl_2025/student_data/S-LS/mask_bacteria/*.tif'))

X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

X_annotated=X[0:12]

rng = np.random.RandomState(42)
ind = rng.permutation(12)
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X_annotated[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X_annotated[i] for i in ind_train], [Y[i] for i in ind_train] 

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# # Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# # Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)

from stardist.models import StarDist2D

# creates a pretrained model
# model = StarDist2D.from_pretrained('2D_versatile_fluo')
# shutil.copytree(model.logdir, 'Stardist_training_model_02', dirs_exist_ok=True)
# model=StarDist2D(None, 'Stardist_training_model') #Made a mistake here

model = StarDist2D.from_pretrained('2D_versatile_fluo')
shutil.copytree(model.logdir, 'Stardist_training_model_03', dirs_exist_ok=True)
model=StarDist2D(None, 'Stardist_training_model_03') 


def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=100)