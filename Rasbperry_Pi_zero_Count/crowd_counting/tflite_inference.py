#import tensorflow as tf
import tflite_runtime.interpreter as tflite

from imutils import paths
from tqdm import tqdm
import logging
import time
from datetime import datetime
import cv2
import math
import numpy as np
import os
import shutil
import sys
sys.path.append('..')
from gsutils import init_logger, a, load_images_inference, saved_image_path, mape_fun


def predict_num(output_data):
    pred_img=np.squeeze(np.squeeze(output_data))
    return np.sum(pred_img)
    
tmp_folder=''
tflite_parameters=[
                #Helicoverpa full_color  overlap
                {'model_name':'CSRNet_Helicoverpa_color_HVGA_10k_overlap',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False},   
                #Helicoverpa medium color  overlap
                {'model_name':'CSRNet_Helicoverpa_color_HVGA_medium_10k_overlap',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_medium_HVGA_10k-0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False}, 
                #Helicoverpa full_color  overlap quant
                {'model_name':'CSRNet_Helicoverpa_color_HVGA_10k_overlap_quant',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0_quant.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':True,
                    'create_image':False},   

                 #Helicoverpa full_color   
                {'model_name':'CSRNet_Helicoverpa_color_HVGA_10k',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False},   
                #Helicoverpa medium color  
                {'model_name':'CSRNet_Helicoverpa_color_medium_HVGA_10k',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_medium_HVGA_10k-0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False}, 
                 #Helicoverpa full_color quant
                {'model_name':'CSRNet_Helicoverpa_color_HVGA_10k_quant',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'models/Helicoverpa_armigera_color_HVGA_10k-valloss 0_quant.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':True,
                    'create_image':False},   



                #Plodia full_color   
                {'model_name':'CSRNet_Plodia_color_HVGA_10k',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'models/Plodia_interpunctella_color_HVGA_10k-valloss 0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False},
                #Plodia medium color   
                {'model_name':'CSRNet_Plodia_color_medium_HVGA_10k',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'models/Plodia_interpunctella_color_medium_HVGA_10k-0.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':False,
                    'create_image':False},

                 #Plodia full_color  quant
                {'model_name':'CSRNet_Plodia_color_HVGA_10k_quant',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'models/Plodia_interpunctella_color_HVGA_10k-valloss 0_quant.tflite',
                    'size':(480,320),
                    'grey':False,
                    'enable':True,
                    'create_image':False},
               
                
]
LOGS='logs_10k_quant'
try:
    shutil.rmtree(LOGS)
except:
    pass    
os.makedirs(LOGS)
#general results
gen_res_logger=init_logger(LOGS,'crowd_counting_genResults')

for tflite_parameter in tflite_parameters:
    if tflite_parameter['enable']==True:
        save_path=os.path.join(LOGS,tflite_parameter['model_name'])    
        try:
            shutil.rmtree(save_path)
        except:
            pass    
        os.makedirs(save_path)

        #Load Logger
        logger=init_logger(LOGS,tflite_parameter['model_name'])
        
        #Load model
        interpreter_qt =tflite.Interpreter(
        model_path=tflite_parameter['path_saved_model_tflite'], num_threads=None)
        #Init Input Output            
        interpreter_qt.allocate_tensors()
        input_details = interpreter_qt.get_input_details()
        output_details = interpreter_qt.get_output_details()
       
        #Init Images
        full_image_path=list(paths.list_images(tflite_parameter['test_image_path']))
        #init bar
        pbar_epoch = tqdm(range(len(full_image_path)), f"Inference image in progress")
        # Head Log
        logger.info(f"Test in progress : Model :{tflite_parameter['model_name']} Checkpoint : {tflite_parameter['path_saved_model_tflite']}")
        logger.info(f"Image Name , Prediction , Label , Predict time")

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

        for image_name in full_image_path:
            #Init timer
            start_im_time = time.time()
            #Load image
            image,image_before_tranform=load_images_inference(image_name,tflite_parameter['size'],tflite_parameter['grey'])
            #Read Label from image filename
            label=int(image_name.split(".")[-2].split("_")[-1:][0])
            
            #Quantaization
            scale, zero_point = input_details[0]['quantization']
            if scale!=0 or zero_point!=0:
                input_data= image / (scale*1.0) + (zero_point*1.0)
            else:
                input_data=image
            #Prepair image and inference from model
            #input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])
            input_data=input_data.astype(input_details[0]["dtype"])
            interpreter_qt.set_tensor(input_details[0]['index'], input_data)
            interpreter_qt.invoke()

            #Proccessing output and predict label     
            output_data = interpreter_qt.get_tensor(output_details[0]['index'])
            scale, zero_point = output_details[0]['quantization']
            if scale!=0 or zero_point!=0:
                output_data = ((scale*1.0) * (output_data - zero_point*1.0))
            pred_label=predict_num(output_data)
            
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
            #pbar_epoch.set_description(f'A:{sum_a/(count+1):.6f}  MAE:{ mae/(count+1):.2f} Total time {(time.time()-start_im_time)*1000:.3f}')
            pbar_epoch.update()
            #Update log
            #logger.info(f"{image_name},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")    
            logger.info(f"{image_name} , {pred_label:.2f} , {label:.0f} , {(time.time()-start_im_time)*1000:.1f}")  
            
         #Final metrics
        count=count_1_20 + count_50_100
        mae = (sum_mae_1_20 + sum_mae_50_100)/count
        avg_a = (sum_a_1_20 + sum_a_50_100)/count
        mape=(sum_mape_1_20+sum_mape_50_100)/count
        mse=(sum_mse_1_20+sum_mse_50_100)/count
        rmse=math.sqrt(mse)
        sum_time=sum_time_1_20 + sum_time_50_100
        logger.info(f"{tflite_parameter['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
        gen_res_logger.info(f" {tflite_parameter['model_name']} \t \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.2f}% \t RMSE:{rmse:.4f} \t AVG_TIME {(sum_time/(count)):.1f}")
        if count_1_20>0:
          gen_res_logger.info(f" {tflite_parameter['model_name']}  1-20   \t A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.2f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.2f} \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
        if count_50_100>0:
          gen_res_logger.info(f" {tflite_parameter['model_name']}  50-100 \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.2f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.2f} \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")



