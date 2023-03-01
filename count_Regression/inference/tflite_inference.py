import tensorflow as tf
#import tflite_runtime.interpreter as tflite
from imutils import paths
from tqdm import tqdm
import logging
import time
from datetime import datetime
import cv2
import numpy as np
import os
import shutil
import math

import sys
sys.path.append('..')
from gsutils import init_logger, a, load_images_inference, saved_image_path, mape_fun

def predict_num(output_data):
    pred_img=np.squeeze(np.squeeze(output_data))
    return pred_img.item()

tflite_parameters=[
                #Helicoverpa ResNet50 Overlap
                {'model_name':'Count_Regression_Helicoverpa_resnet50_overlap',
                    'test_image_path':'../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Helicoverpa_armigera_resnet50_tf/Helicoverpa_armigera_resnet_HVGA-0.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':True},
                #Helicoverpa ResNet18 Overlap
                {'model_name':'Count_Regression_Helicoverpa_resnet18_overlap',
                    'test_image_path':'../../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Helicoverpa_armigera_resnet18_tf/Helicoverpa_armigera_resnet_HVGA-0.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':True},
                #Helicoverpa ResNet50     
                {'model_name':'Count_Regression_Helicoverpa_resnet50',
                    'test_image_path':'../../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Helicoverpa_armigera_resnet50_tf/Helicoverpa_armigera_resnet_HVGA-0.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':True},
                #Helicoverpa ResNet18     
                {'model_name':'Count_Regression_Helicoverpa_resnet18',
                    'test_image_path':'../../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Helicoverpa_armigera_resnet18_tf/Helicoverpa_armigera_resnet_HVGA-0.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':True},
                #Helicoverpa ResNet grey     
                {'model_name':'Count_Regression_Helicoverpa_grey_last_HVGA',
                    'test_image_path':'../../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'../convert_model/export_20k/Count_Regrassion/Helicoverpa_grey_last_HVGA_tf/Helicoverpa_armigera_grey_large_HVGA-2.tflite',
                    'size':(480,320),
                    'grey':True,
                    'enable':False},
                
                #Plodia ResNet50
                {'model_name':'Count_Regression_plodia_resnet50',
                    'test_image_path':'../../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Plodia_interpunctella_resnet_tf/Plodia_interpunctella_resnet_HVGA-2.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':False},
                #Plodia ResNet18
                {'model_name':'Count_Regression_plodia_resnet18',
                    'test_image_path':'../../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'../convert_model/export_10k/Count_Regression/Plodia_interpunctella_resnet_tf/Plodia_interpunctella_resnet_HVGA-2.tflite',
                    'size':(224,224),
                    'grey':False,
                    'enable':False},
                
                #Plodia ResNet grey
                {'model_name':'CSRNet_plodia_grey_last_HVGA',
                    'test_image_path':'../../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'../convert_model/export_20k/Count_Regrassion/Plodia_grey_last_HVGA_tf/Plodia_interpunctella_grey_large_HVGA-1.tflite',
                    'size':(480,320),
                    'grey':True,
                    'enable':False,},
]
LOG='log_10k'
try:
    shutil.rmtree(LOG)
except:
    pass    
os.makedirs(LOG)
#general results
gen_res_logger=init_logger(LOG,'count_regression_genResults')

for tflite_parameter in tflite_parameters:
    if tflite_parameter['enable']==True:
        save_path=os.path.join(LOG,tflite_parameter['model_name'])    
        try:
            shutil.rmtree(save_path)
        except:
            pass    
        os.makedirs(save_path)

        #Load Logger
        logger=init_logger(LOG,tflite_parameter['model_name'])
        
        #Load model
        interpreter_qt = tf.lite.Interpreter(
        model_path=tflite_parameter['path_saved_model_tflite'], num_threads=None)
        #Init Input Output            
        interpreter_qt.allocate_tensors()
        input_details = interpreter_qt.get_input_details()
        output_details = interpreter_qt.get_output_details()
        # Init metrics
        percent_acceptable_error = 0.2
        acceptable_errors_1_20 = 0
        sum_mae_1_20 = 0.0
        sum_a_1_20 = 0
        sum_mape_1_20 = 0
        sum_mse_1_20 = 0
        sum_time_1_20 = 0
        count_1_20 = 0

        acceptable_errors_50_100 = 0
        sum_mae_50_100 = 0.0
        sum_a_50_100 = 0
        sum_mape_50_100 = 0
        sum_mse_50_100 = 0
        sum_time_50_100 = 0
        count_50_100 = 0
        #Init Images
        full_image_path=list(paths.list_images(tflite_parameter['test_image_path']))
        #init bar
        pbar_epoch = tqdm(range(len(full_image_path)), f"Inference image in progress")
        # Head Log
        logger.info(f"Test in progress : Inference Model :{tflite_parameter['model_name']} Checkpoint : {tflite_parameter['path_saved_model_tflite']}")
        logger.info(f" Image Name , Prediction ,Label , Predict time")
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
            interpreter_qt.set_tensor(input_details[0]['index'], input_data)
            interpreter_qt.invoke()

            #Proccessing output and predict label     
            output_data = interpreter_qt.get_tensor(output_details[0]['index'])
            scale, zero_point = output_details[0]['quantization']
            if scale!=0 or zero_point!=0:
                output_data = ((scale*1.0) * (output_data - zero_point*1.0))
            pred_label=predict_num(output_data)
            
              # Update metrics
            if label > 20:
                sum_a_50_100 += a(pred_label, label)
                sum_mae_50_100 += abs(label-pred_label)
                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_50_100 += 1
                sum_mse_50_100 = math.pow((label-pred_label), 2)
                sum_time_50_100 += (time.time()-start_im_time)*1000
                sum_mape_50_100 = mape_fun(pred_label, label)
                count_50_100 += 1
            else:
                sum_a_1_20 += a(pred_label, label)
                sum_mae_1_20 += abs(label-pred_label)

                zz = math.pow((label-pred_label), 2)
                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_1_20 += 1

                sum_mse_1_20 += zz
                sum_time_1_20 += (time.time()-start_im_time)*1000
                sum_mape_1_20 = mape_fun(pred_label, label)
                count_1_20 += 1

            #Update bar
            pbar_epoch.set_description(f'A:{(sum_a_50_100+sum_a_1_20) / (count_50_100 + count_1_20) :.6f}  MAE:{ (sum_mae_50_100+sum_mae_1_20)/(count_50_100+count_1_20):.2f} Total time {(time.time()-start_im_time)*1000:.3f}')
            pbar_epoch.update()
            #Update log
            #logger.info(f"{image_name},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")    
            logger.info(f"{image_name} , {pred_label:.2f} , {label:.0f} , {(time.time()-start_im_time)*1000:.1f}")    
        
         # Final metrics
        count = count_1_20 + count_50_100
        mae = (sum_mae_1_20 + sum_mae_50_100)/count
        avg_a = (sum_a_1_20 + sum_a_50_100)/count
        mape = (sum_mape_1_20+sum_mape_50_100)/count
        mse = (sum_mse_1_20+sum_mse_50_100)/count
        rmse = math.sqrt(mse)
        sum_time = sum_time_1_20 + sum_time_50_100
        avg_acceptable_errors = (
            acceptable_errors_1_20+acceptable_errors_50_100)/count
        logger.info(
            f"{tflite_parameter['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
        gen_res_logger.info(
            f" {tflite_parameter['model_name']} \t \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.3f}% \t RMSE:{rmse:.4f} \t AE {avg_acceptable_errors:.2f}% \t AVG_TIME {(sum_time/(count)):.1f}")
        if count_1_20 > 0:
            gen_res_logger.info(f" {tflite_parameter['model_name']} 1-20 \t A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.3f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.4f} \t AE {acceptable_errors_1_20/count_1_20:.2f}% \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
        if count_50_100 > 0:
            gen_res_logger.info(f" {tflite_parameter['model_name']} 50-100 \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.3f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.4f} \t AE {acceptable_errors_50_100/count_50_100:.2f}% \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")
