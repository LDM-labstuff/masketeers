### Set up NoMachine with port forwarding
# 1. From VSCode connected to your remote machine, forward a port (e.g. `4000`) to your local machine.
#     - Open you command palette in VSCode (usually CMD-Shift-P) and type "forward a port"
#     - Then type in the desired port number `4000` and hit enter
#     - From the "PORTS" tab, you should see port 4000 listed as a forwarded port
# 2. Download and install [NoMachine](https://www.nomachine.com/download) on your local machine if it is not already installed.
# 3. Enter the server address in host, set the port to match the port you forwarded in step 1 and protocol as NX. Feel free to enter any name you would like.
# 4. Click on the configuration tab on the left.
# 5. Choose "Use key-based authentication with a key you provide" and hit the "Modify" button.
# 6. Provide the path to your ssh key .pem file.
# 7. Finally hit connect (or Add).
# 8. If you are asked to create a desktop, click yes.
# 9. You should then see a time and date, hitting enter should let you enter your username and access the desktop. The first login may be slow.
# 10. Still in NoMachine, open a shell window. Hit the application button in the bottom left corner and launch "Konsole"
# 11. From the shell, run `echo $DISPLAY`. Copy the output. It should be something like `:1005`
# 12. Return to your notebook in VSCode, and proceed with the exercise.
# 13. Modify the cell below to input the DISPLAY port you retrieved in step 11

import os
os.environ["DISPLAY"] = ':1001'
import napari
import zarr

data_omni ='/mnt/efs/aimbl_2025/student_data/S-LS/omni_seg.zarr'
data_raw = '/mnt/efs/aimbl_2025/student_data/S-LS/my_data2.zarr'
data_omni = zarr.open(data_omni, mode='r')
data_raw= zarr.open(data_raw, mode='r')
gt=data_raw['mask_bacteria']['masks_bacteria_stack'][:]
raw_bacteria = data_raw['bacteria']['raw_bacteria_stack'][:]
omni = data_omni[:]

viewer = napari.Viewer()

viewer.add_image(raw_bacteria[0:12], name="raw_image")
viewer.add_labels(omni, name='omnipose')
viewer.add_labels(gt, name='groundtruth')