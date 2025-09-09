import numpy as np
import zarr
import datetime
from cellpose import models, io
from cellpose import train
import matplotlib.pyplot as plt
io.logger_setup() # run this to get printing of progress

model = models.CellposeModel(gpu=True) 

#loading zarrs of raw data and true masks
data_raw = '/mnt/efs/aimbl_2025/student_data/S-LS/my_data2.zarr'
data_raw= zarr.open(data_raw, mode='r')
gt=data_raw['mask_bacteria']['masks_bacteria_stack'][:]
raw_bacteria = data_raw['bacteria']['raw_bacteria_stack'][:12]

#to shuffle  data
np.random.seed(3)
shuffled_indices=np.random.permutation(12)
#setting a consistent random index

shuffled_raw_data=raw_bacteria[shuffled_indices] #shuffled raw data 
shuffled_gt=gt[shuffled_indices] #shuffled true masks 

#to split data
split_idx = 9
train_raw = shuffled_raw_data[:split_idx] #raw data for training
train_gt  = shuffled_gt[:split_idx] #true masks for training
test_raw = shuffled_raw_data[split_idx:] #raw data for validation/testing
test_gt  = shuffled_gt[split_idx:] #true masks for validation/testing


#to train new model ang generate a timestamp with YYYYMMDD-HHMMSS
tstamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#model_dir = '/mnt/efs/aimbl_2025/student_data/S-LS/models'
model_dir='/mnt/efs/aimbl_2025/student_data/S-LS/models/models Sept 9'
model_name = f'{model_dir}/Cellpose-{tstamp}'


# defining training params
n_epochs = 100
#learning_rate = 1e-3 #increasing
learning_rate = 1e-4
weight_decay = 0.1
batch_size = 4
save_every = 25




#training new model
new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=train_raw, #[:, np.newaxis],
                                                            test_data=test_raw, #[:, np.newaxis],
                                                            test_labels=test_gt,
                                                            #channel_axis=1,
                                                            train_labels=train_gt,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=max(2, len(train_raw)), # can change this
                                                            model_name=model_name,
                                                            save_every=save_every,
                                                            save_each=True)

plt.plot(train_losses) #1d array
plt.title("Train losses")
plt.savefig("/mnt/efs/aimbl_2025/student_data/S-LS/models-train-losses/cp_train-losses.png")


plt.plot(test_losses) #1d array
plt.title("Test losses")
plt.savefig("/mnt/efs/aimbl_2025/student_data/S-LS/models-test-losses/cp-test-losses.png")