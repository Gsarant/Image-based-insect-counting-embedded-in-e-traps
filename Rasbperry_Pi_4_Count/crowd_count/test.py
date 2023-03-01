import os
import sys
import copy
from io import BytesIO

from PIL import Image,ImageOps
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib import cm
import matplotlib.pyplot as plt

import cv2
import logging
from tqdm import tqdm

import numpy as np

import matplotlib.cm as mpl_color_map
from torchvision import  transforms
from torchvision.transforms.functional import to_pil_image


from csrnet import CSRNet,CSRNet_small,CSRNet_medium,CSRNet_medium_color
import torch
from imutils import paths
from datetime import datetime
import time
import shutil
import math
import torch.nn as nn


sys.path.append('../')
from gsutils import init_logger,a,load_image,saved_image_path,mape_fun,create_head_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transforms= transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                        ])


test_params=[

             #overlap full_color 
             {'test_image_path':'../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
              'model':CSRNet(3,False),
              'model_name':'CSRNet_Helicoverpa_color_HVGA_10k_overlap',
              'transform':test_transforms ,
              'size':(480,320),
              'round':True,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0.00045534-CSRNET_Helicoverpa_color_HVGA_10k_ep_297.pt',
              'test_save_images_path':'tests/CSRNet2/Helicoverpa_armigera_color_HVGA_10k_overlap'},
            
            #overlap medium_color 
             {'test_image_path':'../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
              'model':CSRNet_medium_color(3,False),
              'model_name':'CSRNet_Helicoverpa_color_HVGA_medium_10k_overlap',
              'transform':test_transforms ,
              'size':(480,320),
              'round':True,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Helicoverpa_armigera_color_medium_HVGA_10k-mae 1.0292-CSRNET_Helicoverpa_color_medium_HVGA_10k_ep_59.pt',
              'test_save_images_path':'tests/CSRNet2/Helicoverpa_armigera_color_HVGA_medium_10k_overlap'},
             
            #HVGA
            #Helicoverpa color full model
            {'test_image_path':'../data/Helicoverpa_armigera',
              'model':CSRNet(3,False),
              'model_name':'CSRNet_Helicoverpa_color_HVGA_10k',
              'transform':test_transforms ,
              'size':(480,320),
              'round':False,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0.00045534-CSRNET_Helicoverpa_color_HVGA_10k_ep_297.pt',
              'test_save_images_path':'tests/CSRNet2/Helicoverpa_armigera_color_HVGA_10k'},
            #Helicoverpa color medium model
            {'test_image_path':'../data/Helicoverpa_armigera',
              'model':CSRNet_medium_color(3,False),
              'model_name':'CSRNet_Helicoverpa_color_HVGA_medium_10k',
              'transform':test_transforms ,
              'size':(480,320),
              'round':False,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Helicoverpa_armigera_color_medium_HVGA_10k-mae 1.0292-CSRNET_Helicoverpa_color_medium_HVGA_10k_ep_59.pt',
              'test_save_images_path':'tests/CSRNet2/Helicoverpa_armigera_color_HVGA_medium_10k'},
           
            #Plodia color  full model
            {'test_image_path':'../data/Plodia_interpunctella',
              'model':CSRNet(3,False),
              'model_name':'CSRNet_plodia_color_HVGA_10k',
              'transform':test_transforms ,
              'size':(480,320),
              'round':False,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Plodia_interpunctella_color_HVGA_10k-valloss 0.00082795-CSRNET_Plodia_color_HVGA_10k_ep_296.pt',
              'test_save_images_path':'tests/CSRNet2/Plodia_interpunctella_color_HVGA_10k'},
           
            #Plodia color  medium  model
            {'test_image_path':'../data/Plodia_interpunctella',
              'model':CSRNet_medium_color(3,False),
              'model_name':'CSRNet_plodia_color_HVGA_medium_10k',
              'transform':test_transforms ,
              'size':(480,320),
              'round':False,
              'grey':False,
              'create_image':True,
              'enable':True,
              'load_saved_parameters':'models/Plodia_interpunctella_color_medium_HVGA_10k-mae 1.5006-CSRNET_color_medium_HVGA_10k_ep_69.pt',
              'test_save_images_path':'tests/CSRNet2/Plodia_interpunctella_color_HVGA_10k'},
            ]
LOGS='tests/logs_10k_2'
try:
    shutil.rmtree(LOGS)
except:
    pass    
os.makedirs(LOGS)

gen_res_logger=init_logger(LOGS,'crowd_counting_genResults')
for test_param in test_params:

    if test_param['enable']==True:
        try:
          shutil.rmtree(test_param['test_save_images_path'])
        except:
          pass    
        os.makedirs(test_param['test_save_images_path'])
        
        #Load Logger
        test_logger=init_logger(LOGS,test_param['model_name'])
        
        #checkpoint = torch.load(test_param['load_saved_parameters'])
        #model = test_param['model']
        #model.load_state_dict(checkpoint['model_state_dict'])
        
        #Load model
        if device.type=='cpu':
          model = torch.load(test_param['load_saved_parameters'],map_location=torch.device('cpu'))
        else:
          model = torch.load(test_param['load_saved_parameters'])
        model.to(device)
        model.eval()
        
      
        #Init Images
        full_image_path=list(paths.list_images(test_param['test_image_path']))
        #init bar
        test_pbar = tqdm(range(len(full_image_path)), f"Test in progress : {test_param['test_save_images_path']} Model :{test_param['model_name']} Checkpoint : {test_param['load_saved_parameters']}" )
        # Head Log
        test_logger.info(f"Test in progress : {test_param['test_save_images_path']} Model :{test_param['model_name']} Checkpoint : {test_param['load_saved_parameters']}")
        test_logger.info(f"Image Name , Prediction , Label , Predict time")
        
               
        #Init metrics
        sum_mae_1_20=0.0
        sum_a_1_20=0
        sum_mape_1_20=0
        sum_mse_1_20=0
        sum_time_1_20=0
        count_1_20=0


        sum_mae_50_100=0.0
        sum_a_50_100=0
        sum_mape_50_100=0
        sum_mse_50_100=0
        sum_time_50_100=0
        count_50_100=0


        for img in full_image_path:
            #Init timer
            start_im_time = time.time()
            #Load image
            image,image1=load_image(img,test_param['size'],test_param['transform'],test_param['grey']  )
           
            #Read Label from image filename
            label=int(img.split(".")[-2].split("_")[-1:][0])
            
            #Prepair image and inference from model
            image = image.to(device)
            output = model(image)
            #Proccessing output and predict label 
            pred_label=output.data.abs().sum().cpu().squeeze().detach()
            if test_param['round']==True:
              pred_label=round(pred_label.item())
            #
            # Create image
            if test_param['create_image']==True:
                output=output.cpu().squeeze().detach().numpy()
                #output=nn.functional.interpolate(output, scale_factor=(8,8),mode='area')
                output=cv2.resize(output,(output.shape[1]*8,output.shape[0]*8),interpolation=cv2.INTER_AREA)  
                image=create_head_image(image1,output,"CSRNet",label,0.7,pred_label)
                cv2.imwrite(saved_image_path(test_param['test_save_images_path'],img,'bled'),image)
            
            #Update metrics
            if label>20:
              sum_a_50_100 +=a(pred_label , label)     
              sum_mae_50_100 += abs(label-pred_label)
              sum_mse_50_100 += math.pow((label-pred_label),2)
              sum_time_50_100 += (time.time()-start_im_time)*1000
              sum_mape_50_100 += mape_fun(pred_label , label)
              count_50_100 +=1
            else:
              sum_a_1_20 += a(pred_label , label)     
              sum_mae_1_20 += abs(label-pred_label)
              sum_mse_1_20 += math.pow((label-pred_label),2)
              sum_time_1_20 += (time.time()-start_im_time)*1000
              sum_mape_1_20 += mape_fun(pred_label , label)
              count_1_20 +=1

            #Update bar
            test_pbar.update()
            #Update log
            #test_logger.info(f"{img},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")  
            
            test_logger.info(f"{img} , {pred_label:.2f} , {label:.0f} , {(time.time()-start_im_time)*1000:.1f}")  
           
         #Final metrics
        count=count_1_20 + count_50_100
        mae = (sum_mae_1_20 + sum_mae_50_100)/count
        avg_a = (sum_a_1_20 + sum_a_50_100)/count
        mape=(sum_mape_1_20+sum_mape_50_100)/count
        mse=(sum_mse_1_20+sum_mse_50_100)/count
        rmse=math.sqrt(mse)
        sum_time=sum_time_1_20 + sum_time_50_100
        test_logger.info(f"{test_param['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
        gen_res_logger.info(f" {test_param['model_name']} \t \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.2f}% \t RMSE:{rmse:.4f} \t AVG_TIME {(sum_time/(count)):.1f}")
        if count_1_20>0:
          gen_res_logger.info(f" {test_param['model_name']}  1-20   \t A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.2f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.2f} \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
        if count_50_100>0:
          gen_res_logger.info(f" {test_param['model_name']}  50-100 \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.2f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.2f} \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")

