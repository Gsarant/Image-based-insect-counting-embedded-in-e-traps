
import os
import torch
from loader import create_loader
import sys
sys.path.append('../')
from gsutils  import display_loader
from datetime import datetime
import numpy as np
from train_insect_CSRNet.train_val import run_train

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Is cuda available: {torch.cuda.is_available()}") 
    print(f" Num of GPU {torch.cuda.device_count()}")
    print(f" Num of Cur GPU {torch.cuda.current_device()}")
    print('Using device:', device)
    
    torch.cuda.empty_cache()

    log_dir = "logs_10k_5"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    today = datetime.today().strftime("%Y%m%d")


    ckpt_dir = "ckpt_10k_5"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

   
    tmp_folder='../../'

    dataset_paths = [
        
         {'name':'Helicoverpa_armigera_color_medium_HVGA_10k',
            'model_name':'CSRNET_Helicoverpa_color_medium_HVGA_10k',
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

        {'name':'Plodia_interpunctella_color_medium_HVGA_10k',
            'model_name':'CSRNET_color_medium_HVGA_10k',
            'model_class':'CSRNet_medium_color',  
            'model_class_attribute':(3,False), 
            'train_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/train',
            'train_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/train',
            'val_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/val',
            'val_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/val',
            'kernel_size':15,
            'grey':False,
            'Input_Image_resize':(480,320),#height, width
            'scale_image':1,
            'scale_density':8 ,
            'start_epoch':0,
            'epochs':300,#100
            'enable':True,
            'preload_model':''},

        

        {'name':'Helicoverpa_armigera_color_HVGA_10k',
            'model_name':'CSRNET_Helicoverpa_color_HVGA_10k',
            'model_class':'CSRNet',  
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
            'epochs':300,
            'enable':True,
            'preload_model':''},

        {'name':'Plodia_interpunctella_color_HVGA_10k',
            'model_name':'CSRNET_Plodia_color_HVGA_10k',
            'model_class':'CSRNet',  
            'model_class_attribute':(3,False), 
            'train_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/train',
            'train_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/train',
            'val_image_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/images/val',
            'val_density_path':f'{tmp_folder}data/datasets_Plodia_interpunctella_10k/labels/val',
            'kernel_size':15,
            'grey':False,
            'Input_Image_resize':(480,320),#height, width
            'scale_image':1,
            'scale_density':8 ,
            'start_epoch':0,
            'epochs':300,
            'enable':True,
            'preload_model':''},
       
        ]
    loaders=create_loader(dataset_paths)
    #for loader in loaders:
    #    print(loader['name'])
    #    display_loader(loader['train_loader'])
   

    run_train(loaders,device,log_dir,ckpt_dir)
if __name__ == "__main__":
    main()