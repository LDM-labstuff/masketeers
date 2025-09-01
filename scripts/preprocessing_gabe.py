import zarr
import imageio
import numpy as np
from pathlib import Path
import glob
import re
from collections import defaultdict
import os



image_name_list=glob.glob("/mnt/efs/aimbl_2025/student_data/S-GS/Ex*/*ORG.tif")

root=zarr.group("/mnt/efs/aimbl_2025/student_data/S-GS/fungal_eating.zarr", overwrite=True)

conditions=defaultdict(dict)
tif_ID=[]
for index,tif in enumerate(image_name_list):
    tif=Path(tif)
    image_name=re.findall(r'\d+',tif.name.split(".")[0].split("_")[1])
    image_name[1]=int(image_name[1])
    if tif.name.split(".")[0].split("_")[0][-2:] == str(43):
        conditions["sample_"+image_name[0]]["timepoint_"+str(image_name[1])]= []
        ID=(tif,str(index))
        tif_ID.append(ID)
    elif tif.name.split(".")[0].split("_")[0][-2:] == str(53):
        conditions["sample_"+image_name[0]]["timepoint_"+str(image_name[1]+2)]= []
        ID=(tif,str(index))
        tif_ID.append(ID)
    elif tif.name.split(".")[0].split("_")[0][-2:] == str(73):
        conditions["sample_"+image_name[0]]["timepoint_"+str(image_name[1]+13)]= []
        ID=(tif,str(index))
        tif_ID.append(ID)
       

for index,tif in enumerate(image_name_list):
    image_name=[tif.split("/")[-1].split("_")[0][-2:],re.findall(r'\d+',tif.split("/")[-1].split("_")[1])]
    #if image_name[0] == str(43):
    #    tif_image=imageio.imread(tif)
        #print(tif_image)
        #print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
    #elif image_name[0] == str(53):
        #print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
    #elif image_name[0] == str(73):
        #print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
#tif_image=np.array()

for sample_key, time_point in conditions.items():
    sample_group = root.create_group(name=str(sample_key))
    print(time_point.keys())
    for t in time_point.keys():
        time_point_zarr = sample_group.create_group(name=str(t))
        total_tif_list=[]
        for index,tif in enumerate(image_name_list):
            image_name=[tif.split("/")[-1].split("_")[0][-2:],re.findall(r'\d+',tif.split("/")[-1].split("_")[1])]
            if image_name[0] == str(43):
                if image_name[1][0]==sample_key[-2:]:
                    
                    if 'timepoint_'+image_name[1][1]==t:

        
                        total_tif_list.append(imageio.imread(tif))
            elif image_name[0] == str(53):
                if image_name[1][0]==sample_key[-2:]:
                    
                    if 'timepoint_'+str(int(image_name[1][1])+2)==t:

        
                        total_tif_list.append(imageio.imread(tif))
            elif image_name[0] == str(73):
                if image_name[1][0]==sample_key[-2:]:
                    
                    if 'timepoint_'+str(int(image_name[1][1])+13)==t:
        
                        total_tif_list.append(imageio.imread(tif))
            
        if total_tif_list == []:
            print("look at this"+tif+str(image_name)+sample_key+t)
        zarr_array =np.stack(total_tif_list)  
        print(zarr_array.dtype) 
        tiles=time_point_zarr.create_array(name="tiles", shape=(100, 1216,1920), chunks=(1,1216,1920), dtype="uint8")
        tiles[:,:,:] = zarr_array 


                
                
                #tif_image_stack=tif_image.stack()

                #bar = root.create_array(name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4")
        '''        tif_image=imageio.imread(tif)
                    print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
                elif image_name[0] == str(53):
                    print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
                elif image_name[0] == str(73):
                    print("sample_"+image_name[1][0]+"timepoint_"+image_name[1][1]+"tile_"+image_name[1][2]+image_name_list[index])
            '''
            
        

        #tiles_shape=[len()]
        #time_group= sample_group.create_group(name=str(t),shape=(tiles_shape,1216,1920))
        #time_group.create_array


print(os.system("ls -R "+"/mnt/efs/aimbl_2025/student_data/S-GS/fungal_eating.zarr"))
#        if image_name[0]==str(43):
           
    #print(tif_tup.name.split(".")[0].split("_")[0][-2:])# == str(43):
        
     #   conditions["sample_"+image_name[0]]["timepoint_"+str(image_name[1])].

#for tif in tif_ID:

#print(conditions["sample_01"]["timepoint_1"])
#    sample_group = root.create_group(name=str(sample))
#        for time in conditions[sample].keys():
#            time_group = sample_group.create_group(name=str(time))
#counter=0
#for sample_key, time_point in conditions.items():
 #   for time_point_key, index in time_point.items():
  #      if index in tif_ID[1]:
            #print("found",index)
   #         counter= counter+1    
#print("number of images",len(image_name_list),"counter",counter)
            #else:
            #    print("problem",index)


#print(tif_ID)


#for index,tif in enumerate(image_name_list):
    #tif_image=imageio.imread(tif) 
#    image_name=re.findall(r'\d+',tif.name.split(".")[0].split("_")[1])
#    conditions[

#    tif_array=imageio.imread(tif)        





#tif_to_zarr=zarr.create_array(store="data/fungal_eating.zarr", shape=(len(image_name_list),1216,1920), chunks=(1,1216,1920), dtype="uint8")
