""" 
Nucleoli classifier model based on ResNet18

To run in remote:
tmux att
After training, to load weights to run predictions use:

weights = torch.load("/mnt/efs/aimbl_2025/student_data/S-DD/nucleoli_restnet_trained.pth")
model.load_state_dict(weights["nucleoli_resnet"])
"""

#Imports
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm

#Load images from zarr to torch dataset

class ZarrImageDataset(Dataset):
    def __init__(self, zarr_path, transform=None):
        self.root = zarr.open(zarr_path, mode='r')
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {name: idx for idx, name in enumerate(self.root.group_keys())}

        # Load images and labels
        # map treatments to integer labels
        for treatment_name in self.root.group_keys():
            treatment_group = self.root[treatment_name]['cimages_max']
            print(f"the treatment group shape for treatment {treatment_name} is {treatment_group.shape}")
            for i in range(treatment_group.shape[0]):
                img = treatment_group[i]
                self.images.append(img)
                self.labels.append(self.label_map[treatment_name])
        # self.labels = np.array(self.labels).reshape(-1, 1)
        # enc = OneHotEncoder(categories=[['8nMActD', 'DMSO', '1uMdoxo', 'CX5461', '5uMflavo', '800nMActD', '10uMmg132', '10uMwort']]).fit(self.labels)
        # self.labels = enc.transform(self.labels).toarray()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # Convert to torch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor

# Setup model using ResNet18 
def split_dataset(dataset):
    #Split dataset
    # Total size of the dataset
    total_size = len(dataset)

    # Calculate split sizes
    train_size = int(0.7 * total_size)
    eval_size = int(0.2 * total_size)
    test_size = total_size - train_size - eval_size  # Ensures full coverage

    # Random split
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset,
        [train_size, eval_size, test_size],
        generator=torch.Generator().manual_seed(42))  # For reproducibility
    print(f"The size of the train dataset is {len(train_dataset)}")
    print(f"The size of the eval dataset is {len(eval_dataset)}")
    print(f"The size of the test dataset is {len(test_dataset)}")
    return train_dataset, eval_dataset, test_dataset

def train_nucleoli_resnet(model, train_loader, batch_size, criterion, optimizer):
    model.train()
    pbar = tqdm.tqdm(total=len(train_dataset) // batch_size)
    history = []
    for batch_idx, (raw, target) in enumerate(train_loader):
        optimizer.zero_grad()
        raw = raw.to(device)
        target = target.to(device)
        output = model(raw)
        # output_probabilities = F.softmax(output, dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        pbar.update(1)
    return history

# Create the dataset
zarr_path = '/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr'
dataset = ZarrImageDataset(zarr_path)


#Provide model parameters
num_epochs = 2
learning_rate = 0.001
batch_size = 32
num_classes = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Split dataset
train_dataset, eval_dataset, test_dataset = split_dataset(dataset)


# Data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Load pretrained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Modify the final fully connected layer and establish model parameters
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


# Training loop:
loss_history = []
for epoch in range(num_epochs):
    his = train_nucleoli_resnet(model, train_loader, batch_size, criterion, optimizer)
    loss_history.extend(his)

fig, ax = plt.subplots()
ax.plot(loss_history)
fig.savefig('/mnt/efs/aimbl_2025/student_data/S-DD/nucleoli_restnet_trained_loss_plot.png')

# Save the model
torch.save(
        {"nucleoli_resnet": model.state_dict()}, '/mnt/efs/aimbl_2025/student_data/S-DD/nucleoli_restnet_trained.pth')