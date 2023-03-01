from cv2 import GC_INIT_WITH_MASK
import numpy as np
from imutils import paths
import os
import create_dataset_funnel.create_dataset_funnel as cdf
import matplotlib.pyplot as plt


a='../../'
path_funnel=f'{a}data/class0'


paths=[
        {'path_insect':f'{a}data/datasets_Helicoverpa_armigera/Helicoverpa_armigera_insects_new',
        'path_dirtes':f'{a}data/dirtes',
        'path_dataset':f'{a}data/datasets_Helicoverpa_armigera_10k',
        'groups_insect_per_image':[],#[10,11,12,13,14,15,16,17,18,19,20]
        'train_image_per_group':0,#100
        'val_image_per_group':0,#20
        'train_num_images_rand':[0,60,10000],#LOW,HIGH,SIZE
        'val_num_images_rand':[0,60,500],#LOW,HIGH,SIZE
        'overlap':False,
        'remove_old_dataset':True,
        'enable':True,
        'dest_image_width':1664,
        'dest_image_height':1232},
       
        {'path_insect':f'{a}data/datasets_Plodia_interpunctella/Plodia_interpunctella_insects_new',
         'path_dirtes':f'{a}data/dirtes',
         'path_dataset':f'{a}data/datasets_Plodia_interpunctella_10k',
         'groups_insect_per_image':[],#[20,30,40,55,75,95,110,120,130]
         'train_image_per_group':0,#100
         'val_image_per_group':0,#20
         'train_num_images_rand':[0,110,10000],#LOW,HIGH,SIZE
         'val_num_images_rand':[0,110,500],#LOW,HIGH,SIZE
         'overlap':False,
         'remove_old_dataset':True,
         'enable':True,
         'dest_image_width':1664,
         'dest_image_height':1232},

        {'path_insect':f'{a}data/datasets_Helicoverpa_armigera/Helicoverpa_armigera_insects_new',
        'path_dirtes':f'{a}data/dirtes',
        'path_dataset':f'{a}data/datasets_Helicoverpa_armigera_25_10k',
        'groups_insect_per_image':[],#[10,11,12,13,14,15,16,17,18,19,20]
        'train_image_per_group':0,#100
        'val_image_per_group':0,#20
        'train_num_images_rand':[0,25,10000],#LOW,HIGH,SIZE
        'val_num_images_rand':[0,25,500],#LOW,HIGH,SIZE
        'overlap':False,
        'remove_old_dataset':True,
        'enable':False,
        'dest_image_width':1664,
        'dest_image_height':1232},

         {'path_insect':f'{a}data/datasets_Plodia_interpunctella/Plodia_interpunctella_insects_new',
         'path_dirtes':f'{a}data/dirtes',
         'path_dataset':f'{a}data/datasets_Plodia_interpunctella_25_10k',
         'groups_insect_per_image':[],#[20,30,40,55,75,95,110,120,130]
         'train_image_per_group':0,#100
         'val_image_per_group':0,#20
         'train_num_images_rand':[0,25,10000],#LOW,HIGH,SIZE
         'val_num_images_rand':[0,25,500],#LOW,HIGH,SIZE
         'overlap':False,
         'remove_old_dataset':True,
         'enable':False,
         'dest_image_width':1664,
         'dest_image_height':1232}
        ]

def main():
    for p in paths:
        if p['enable']:
            overlap=p['overlap']
            obj_cdf1=cdf.CreateDatasetFunnel(path_funnel,p['path_dirtes'],p['path_insect'],p['path_dataset'],'train',p['remove_old_dataset'],p['groups_insect_per_image'],p['train_image_per_group'],p['dest_image_width'],p['dest_image_height'])
            #obj_cdf1.create_dataset_yolov5(overlap)
            obj_cdf1.create_dataset_yolo_norm_rand(p['train_num_images_rand'][0],p['train_num_images_rand'][1],p['train_num_images_rand'][2],overlap)
            print('OK TrainSet',p['path_dataset'])
            obj_cdf2=cdf.CreateDatasetFunnel(path_funnel,p['path_dirtes'],p['path_insect'],p['path_dataset'],'val',p['remove_old_dataset'],p['groups_insect_per_image'],p['val_image_per_group'],p['dest_image_width'],p['dest_image_height'])
            #obj_cdf2.create_dataset_yolov5(overlap)
            obj_cdf2.create_dataset_yolo_norm_rand(p['val_num_images_rand'][0],p['val_num_images_rand'][1],p['val_num_images_rand'][2],overlap)
            print('OK ValSet',p['path_dataset'])
    #!shutdown -h now

        

if __name__ == '__main__':
    main()
    
    

   
   