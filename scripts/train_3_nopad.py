import numpy as np
import torch
import subprocess
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
import zarr
import subprocess
import imageio
import zarr
import matplotlib.pyplot as plt
from masketeers.dataProcessing import crop_tiles
import pandas as pd
from torchvision.io import decode_image
from scipy.ndimage import distance_transform_edt, map_coordinates
from torch.utils.tensorboard import SummaryWriter
from dlmbl_unet import UNet
import torchvision.transforms.v2 as transforms_v2
import os


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
    loader_idx=None,
    ckpt_freq=50
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
        if loader_idx==None:
            x, y = x.to(device), y.to(device)
        elif loader_idx==1:
            x, y = x.to(device), y.to(device)
        else:
            counter=loader_idx-2
            y=torch.cat((y,w[0],w[1]), dim=1)
            x, y = x.to(device), y.to(device)

        w=None

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

class Crypto_fungal_eating_datset(Dataset):
    def __init__(self, zarr_path, transform=None, img_transform=None):
        self.zarr_path = zarr_path
        self.zarr_file = zarr.open(zarr_path)
        self.transform = transform
        self.img_transform = img_transform
        self.image = self.zarr_file["image_array"][:]
        not_padded = np.median(self.image, axis=(1,2)) != 0
        self.image = self.image[not_padded][:,np.newaxis]
        self.crypto_mask=self.zarr_file["crypto_mask_array"][:][not_padded,np.newaxis]
        self.spores_mask=self.zarr_file["spores_mask_array"][:][not_padded,np.newaxis]
        self.hyphae_mask=self.zarr_file["hyphae_mask_array"][:][not_padded,np.newaxis]
        print("input image shape"+str(self.image.shape))
        self.from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
        self.inp_transforms = transforms_v2.Compose(
            [
                self.from_np,
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        img_path = self.zarr_path
        image = self.image[idx]
        crypto_mask=self.crypto_mask[idx]
        crypto_mask[crypto_mask > 0] = 1
        crypto_mask[crypto_mask < 0] = 0
        crypto_mask.astype(int)
        spores_mask=self.spores_mask[idx]
        spores_mask[spores_mask > 0] = 1
        spores_mask[spores_mask < 0] = 0
        spores_mask.astype(int)
        hyphae_mask=self.hyphae_mask[idx]
        hyphae_mask[hyphae_mask > 0] = 1
        hyphae_mask[hyphae_mask < 0] = 0
        hyphae_mask.astype(int)
        crypto_mask=torch.from_numpy(crypto_mask)
        hyphae_mask=torch.from_numpy(hyphae_mask)
        spores_mask=torch.from_numpy(spores_mask)
        image=self.inp_transforms(image)
        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            crypto_mask = self.transform(crypto_mask)
            torch.manual_seed(seed)
            spores_mask = self.transform(spores_mask)
            torch.manual_seed(seed)
            hyphae_mask = self.transform(hyphae_mask)
        if self.img_transform:
            image = self.img_transform(image)

        return image, crypto_mask, spores_mask, hyphae_mask
    
    def create_sdt_target(self, mask):
        sdt_target_array = compute_sdt(mask)
        sdt_target = self.from_np(sdt_target_array)
        return sdt_target.float()

if __name__ == "__main__":
    path_to_zarrs = "/mnt/efs/aimbl_2025/student_data/S-GS/training_data_608_normalized.zarr" 

    train_dataset=Crypto_fungal_eating_datset(path_to_zarrs,
        transform=transforms_v2.Compose([transforms_v2.RandomVerticalFlip(),
        transforms_v2.RandomRotation(36)]))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)


    NUM_THREADS = 8
    NUM_EPOCHS = 1000
    np.random.seed(1)

    training_output_path="/mnt/efs/aimbl_2025/student_data/S-GS/unet_training/"
    modelname="3_nopad"

    if not os.path.exists(training_output_path):
        os.mkdir(training_output_path)
    logger = SummaryWriter(f"{training_output_path}runs/{modelname}")

    unet = UNet(
        depth=3,
        in_channels=1,
        out_channels=3,
        final_activation=torch.nn.Sigmoid(),
        num_fmaps=16,
        fmap_inc_factor=3,
        downsample_factor=2,
        padding="same",
    )

    #load weights
    #weights = torch.load("/mnt/efs/aimbl_2025/student_data/S-GS/unet_training/unet_3_BCE_990.pth")
    #load weights
    #unet.load_state_dict(weights["unet"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    learning_rate = 1e-4
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        train(
            model=unet,
            loader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss,
            epoch=epoch,
            log_interval=2,
            device=device,
            loader_idx=3,
            tb_logger=logger
        )

        if epoch % 10 ==0:
            torch.save(
                {
                    "unet": unet.state_dict(),
                    "epoch": epoch,
                    # "losses": losses,
                },
                f"{training_output_path}unet_{modelname}_{epoch}.pth",)