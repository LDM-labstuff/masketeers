import torch
import numpy as np
import random
import zarr
from skimage.segmentation import relabel_sequential
from scipy.ndimage import distance_transform_edt, map_coordinates
from matplotlib import gridspec, ticker
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from dlmbl_unet import UNet
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torch.utils.tensorboard import SummaryWriter
import subprocess

zarr_path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/zarr_storage/test.zarr"
root = zarr.open (zarr_path)

def load_data (zarr_path):
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

def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y, *w) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        if len(w) > 0:
            w = w[0]
            w = w.to(device)
        else:
            w = None
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        assert prediction.shape == y.shape, (prediction.shape, y.shape)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)
        if w is not None:
            weighted_loss = loss * w
            loss = torch.mean(weighted_loss)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        



        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break

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
        sdt = torch.tensor (compute_sdt (seg))
        
        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transform (img)
            torch.manual_seed(seed)
            seg = self.transform(seg)
            torch.manual_seed(seed)
            sdt = self.transform(sdt)
            # img, sdt, seg = self.transform (img, sdt, seg)
        
        if self.img_transform:
            img = self.img_transform(img)

        

    

        return torch.tensor(img, dtype=torch.float32), torch.tensor(sdt, dtype=torch.float32)
    
dataset = CropDataset (zarr_path=zarr_path)
print ("Dataset has been generated")

torch.manual_seed (41)
np.random.seed(42)
training, validation = random_split(dataset, lengths = (0.2, 0.8))
print (f"Training dataset contains {len (training)} images")
train_dataloader = DataLoader (training, shuffle=True, batch_size=128)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = UNet(
    depth = 4,
    in_channels=1,
    out_channels=1,
    final_activation=nn.Tanh(),
    num_fmaps=16
)

learning_rate = 1e-4
loss = nn.MSELoss()
optimizer = torch.optim.Adam (model.parameters(), lr = learning_rate)

logger = SummaryWriter("runs/Unet")


# Function to find an available port and launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process


launch_tensorboard("runs")

for epoch in range(3):
    train (model= model,
           loader = train_dataloader,
            optimizer = optimizer,
            loss_function = loss, epoch = epoch, device=device, tb_logger=logger)
    if epoch % 2 ==0:
        torch.save(
            {
            "unet": model.state_dict(),
            "epoch": epoch,
                # "losses": losses,
                },
                f"/mnt/efs/aimbl_2025/student_data/S-DM/Data/chackpoints/unet_{epoch}.pth",)