import napari
import zarr
#open zarr file in read mode
z = zarr.open("/mnt/efs/aimbl_2025/student_data/S-DD/LDM_treatments.zarr", mode="r")
# define treatment
treatment="800nMActD"
image=z[treatment]["cimages"]
image_maskout = z[treatment]["cimages_maskout"]
mask = z[treatment]["cmasks"]

viewer = napari.Viewer()
image_layer = viewer.add_image(image, name=f"{treatment} raw", colormap="gray")
image_maskout_layer = viewer.add_image(image_maskout, name=f"{treatment} raw maskout", colormap="gray")
#label_layer = viewer.add_labels(mask, name='cmasks')
napari.run()