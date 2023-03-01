import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import shutil
import math
import torch.nn as nn


import sys
sys.path.append('../')
from gsutils import init_logger,a,load_image,saved_image_path,mape_fun,create_head_image
sys.path.append('../train_insect_CSRNet/')
from loader import create_loader
import numpy as np
tmp_folder="../../"
dataset_paths = [
        
         {'name':'Helicoverpa_armigera_color_medium_HVGA_10k_2',
            'model_name':'CSRNET_Helicoverpa_color_medium_HVGA_10k_2',
            'model_class':'CSRNet_medium_color',  
            'model_class_attribute':(3,False), 
            'train_image_path':f'{tmp_folder}data/datasets_Helicoverpa_armigera_10k/images/train',
            'train_density_path':f'{tmp_folder}data/datasets_Helicoverpa_armigera_10k/labels/train',
            'val_image_path':f'{tmp_folder}data/datasets_Helicoverpa_armigera_10k/images/val',
            'val_density_path':f'{tmp_folder}data/datasets_Helicoverpa_armigera_10k/labels/val',
            'kernel_size':21,
            'grey':False,
            'Input_Image_resize':(480,320),#height, width
            'scale_image':1,
            'scale_density':8 ,
            'start_epoch':0,
            'epochs':300,#100
            'enable':False,
            'preload_model':''},    

        {'name':'Plodia_interpunctella_color_medium_HVGA_10k_2',
            'model_name':'CSRNET_color_medium_HVGA_10k_2',
            'model_class':'CSRNet_medium_color',  
            'model_class_attribute':(3,False), 
            'train_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/train',
            'train_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/train',
            'val_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/val',
            'val_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/val',
            'kernel_size':15,
            'grey':False,
            'Input_Image_resize':(1024,768),#height, width
            'scale_image':1,
            'scale_density':8 ,
            'start_epoch':0,
            'epochs':300,#100
            'enable':True,
            'preload_model':''},
 ]

def print_array(ar) :
    print('\n',30*'*','\n')
    sum=0    
    for g in ar:
        for j in g:
            if j==0:
                print(f'{j:.0f}', end=" ")    
            else:
                sum=sum+j    
                print(f'{j:.3f}', end=" ")
        print()
    print('\n',sum,'\n',30*'*','\n')

def main():
    LOGS='aa'
    try:
        shutil.rmtree(LOGS)
    except:
        pass    
    os.makedirs(LOGS)
    loaders=create_loader(dataset_paths)
    for loader in loaders:
        for i,(img, density,name)in enumerate(loader['train_loader'].dataloader):
            #print_array(density.cpu().squeeze().detach().numpy())
            label=density.sum().cpu().squeeze().detach().numpy().item()
            density=density.cpu().squeeze().detach().numpy()
          

            #density=density.unsqueeze(0)
            #density=nn.functional.interpolate(density, scale_factor=(8,8),mode='area')
            #im_density=density.cpu().squeeze().detach().numpy()
            
            #im_density=cv2.resize(im_density,(im_density.shape[1]*8,im_density.shape[0]*8),interpolation = cv2.INTER_AREA)
            #print_array(density.cpu().squeeze().detach().numpy())
            #label=density.sum().cpu().squeeze().detach().numpy().item()
            density=cv2.resize(density,(density.shape[1]*8,density.shape[0]*8),interpolation = cv2.INTER_CUBIC)
            #label2=np.sum(density)
            img=cv2.imread(name[0])
            img = cv2.resize(img, (density.shape[1],density.shape[0]), interpolation=cv2.INTER_AREA)
            #cmap = cm.get_cmap('jet')
            #plt.imshow(im_density,cmap=cmap)
            a=name[0].split('_')[-1].split('.')[0]
            image=create_head_image(img,density,"CSRNet",label ,0.6,float(a))
            cv2.imwrite(saved_image_path(LOGS,name[0],"aa"),image)
            if i >50 :
                break
    

if __name__ == "__main__":
    main()