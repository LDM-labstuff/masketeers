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
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision import models



#Helper functions
def load_data (zarr_path):
    x_arrays = []
    y_array = []
    root = zarr.open (zarr_path)
    for conditions in list (root.keys()):
        images = root [conditions].keys()
        for i in range (len(root[conditions])):
            x = root[conditions][str (i)][:].astype ("float32")
            x = np.expand_dims (x, axis = 0)
            y = dict[conditions]
            x_arrays.append (x)
            y_array.append (y)
    x_array = np.concatenate (x_arrays)
    

    return x_array, y_array

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
    for batch_id, (x, y) in enumerate(loader):
 
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        # if y.dtype != prediction.dtype:
        #     y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

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
            # # check if we log images in this iteration
            # if step % log_image_interval == 0:
            #     tb_logger.add_images(
            #         tag="input", img_tensor=x.to("cpu"), global_step=step
            #     )
            #     tb_logger.add_images(
            #         tag="target", img_tensor=y.to("cpu"), global_step=step
            #     )
            #     tb_logger.add_images(
            #         tag="prediction",
            #         img_tensor=prediction.to("cpu").detach(),
            #         global_step=step,
            #     )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break

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

class ClassificationDataset(Dataset):
    def __init__(self, zarr_path, transform = None):
        
        
        self.zarr_path = zarr_path

        self.x, self.y = load_data(self.zarr_path)
        self.transform = transform
        self.transform = transform
        
       

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]  
        label = self.y[idx] 

        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(label, dtype = torch.long)
        
        
        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transform (img)
            
        return img, label
    
dict = {"tangles": 0, "Pick bodies": 1}

zarr_path = "/mnt/efs/aimbl_2025/student_data/S-DM/Data/zarr_storage/nft_pb_classification.zarr"
root = zarr.open (zarr_path)
print (f"Opening zarr file at {zarr_path}")

print ("Loading files from zarr to dataset")
dataset = ClassificationDataset (zarr_path=zarr_path)
print ("File upload completed")

lesion_classifier = models.resnet18(weights = None)
lesion_classifier.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3)
lesion_classifier.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=2)
)

torch.manual_seed (41)

training, validation = random_split(dataset, lengths = (0.8, 0.2))
print (len(training))
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (lesion_classifier.parameters(), lr = learning_rate)
train_dataloader = DataLoader (training, shuffle=True, batch_size=16)
val_dataloader = DataLoader (validation)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logger = SummaryWriter("runs/classifier_test_CE_loss")

launch_tensorboard("runs/classifier_test_CE_loss")

print ("Starting model training")
for epoch in range(10000):
    train (model= lesion_classifier,
           loader = train_dataloader,
           
            optimizer = optimizer,
            loss_function = loss, epoch = epoch, device=device, tb_logger=logger)
    if epoch % 500 ==0:
        torch.save(
            {
            "unet": lesion_classifier.state_dict(),
            "epoch": epoch,
                # "losses": losses,
                },
                f"/mnt/efs/aimbl_2025/student_data/S-DM/Data/checkpoints_classifier_CE/resnet_{epoch}.pth",)
print ("Training completed!")