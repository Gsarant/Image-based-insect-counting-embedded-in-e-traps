import os
import torch
from datetime import datetime
import numpy as np
from loader import create_loader
from trainregression import run_train

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Is cuda available: {torch.cuda.is_available()}") 
    print(f" Num of GPU {torch.cuda.device_count()}")
    print(f" Num of Cur GPU {torch.cuda.current_device()}")
    print('Using device:', device)
    
    torch.cuda.empty_cache()


    log_dir = "logs_10k_2"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ckpt_dir = "ckpt_10k_2"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    tmp="../"
    
    dataset_paths = [
         #ResNet Plodia vgg16
        {'name':'Plodia_interpunctella_vgg16_HVGA',
            'model_name':'CountRegr_Plodia_interpunctella_vgg16_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'vgg16'),
            'train_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':True,
            'preload_model':''},
        #ResNet Helicoverpa vgg16
        {'name':'Helicoverpa_armigera_vgg16_HVGA',
            'model_name':'CountRegr_Helicoverpa_vgg16_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'vgg16'),
            'train_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':True,
            'preload_model':''},

        #ResNet Helicoverpa18
        {'name':'Helicoverpa_armigera_resnet_HVGA',
            'model_name':'CountRegr_Helicoverpa_resnet18_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'resnet18'),
            'train_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':False,
            'preload_model':''},
        #ResNet Plodia18
        {'name':'Plodia_interpunctella_resnet_HVGA',
            'model_name':'CountRegr_Plodia_interpunctella_resnet18_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'resnet18'),
            'train_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':False,
            'preload_model':''},
        

        #ResNet Helicoverpa50
        {'name':'Helicoverpa_armigera_resnet_HVGA',
            'model_name':'CountRegr_Helicoverpa_resnet50_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'resnet50'),
            'train_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':False,
            'preload_model':''},
        #ResNet Plodia50
        {'name':'Plodia_interpunctella_resnet_HVGA',
            'model_name':'CountRegr_Plodia_interpunctella_resnet50_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(3,'resnet50'),
            'train_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_10k/images/val',
            'grey':False,
            'Input_Image_resize':(224,224),#height, width
            'start_epoch':0,
            'epochs':300,
            'enable':False,
            'preload_model':''},
        
        #Custom Large Model Helicoverpa
        {'name':'Helicoverpa_armigera_grey_large_HVGA',
            'model_name':'CountRegr_Helicoverpa_grey_large_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(1,'large'),
            'train_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/val',
            'grey':True,
            'Input_Image_resize':(480,320),#height, width
            'start_epoch':0,
            'epochs':150,
            'enable':False,
            'preload_model':''},
        #Custom small Model Helicoverpa  
        {'name':'Helicoverpa_armigera_grey_small_HVGA',
            'model_name':'CountRegr_Helicoverpa_grey_small_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(1,'small'),
            'train_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Helicoverpa_armigera_10k/images/val',
            'grey':True,
            'Input_Image_resize':(480,320),#height, width
            'start_epoch':0,
            'epochs':150,
            'enable':False,
            'preload_model':''},
        #Custom Large Model Plodia
        {'name':'Plodia_interpunctella_grey_large_HVGA',
            'model_name':'CountRegr_Plodia_interpunctella_large_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(1,'large'),
            'train_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_25_10k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_25_10k/images/val',
            'grey':True,
            'Input_Image_resize':(480,320),#height, width
            'start_epoch':0,
            'epochs':150,
            'enable':False,
            'preload_model':''},
        #Custom Small Model Plodia
        {'name':'Plodia_interpunctella_grey_small_HVGA',
            'model_name':'CountRegr_Plodia_interpunctella_small_HVGA',
            'model_class':'Count_Regression_Model',
            'model_class_attribute':(1,'small'),
            'train_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_25_20k/images/train',
            'val_image_path':f'{tmp}../data/datasets_Plodia_interpunctella_25_20k/images/val',
            'grey':True,
            'Input_Image_resize':(480,320),#height, width
            'start_epoch':0,
            'epochs':50,
            'enable':False,
            'preload_model':''},

    ]

    loaders=create_loader(dataset_paths)
    run_train(loaders,device,log_dir,ckpt_dir)

if __name__ == "__main__":
    main()