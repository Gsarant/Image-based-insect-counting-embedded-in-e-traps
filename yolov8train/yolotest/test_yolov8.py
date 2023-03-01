#! pip install shutils
#!pip install imutils
import sys
sys.path.append('../../ultralytics')
#%cd "/notebooks/ultralytics/ultralytics"


from ultralytics import YOLO

sys.path.append('..')
from gsutils  import init_logger,a,saved_image_path,draw_win,mape_fun
import cv2
import torch
import os
import math
from imutils import paths
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import time 
import shutil
today = datetime.today().strftime("%Y%m%d")
tmp_folder='../'

test_params=[
            #Overlap Color
            {'model_name':'Helicoverpa_armigera_Yolov8_color_10k_overlap',
                'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                'load_saved_parameters':'/home/giannis/paper2/yolov8train/ultralytics2/runs/detect/Yolov8runs_10k/Helicoverpa-armigera2/weights/best.pt',
                'test_save_images_path':'Yolov8_Test_10k_2/Helicoverpa_armigera_color_overlap',
                'size':(480,480), 
                'conf':0.3,
                'iou':0.8,
                'grey':False,
                'enable':True,
                'create_image':True,
                },
            #Helicoverpa Color
            {'model_name':'Helicoverpa_armigera_Yolov8_color_10k',
                'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                'load_saved_parameters':'/home/giannis/paper2/yolov8train/ultralytics/runs/detect/Yolov8runs_10k/Helicoverpa-armigera2/weights/best.pt',
                'test_save_images_path':'Yolov8_Test_10k/Helicoverpa_armigera_color',
                'size':(480,480), 
                'conf':0.3,
                'iou':0.4,
                'grey':False,
                'enable':False,
                'create_image':True,
                },
            #Plodia Color
            {'model_name':'Plodia_interpunctella_Yolov8_color_10k',
                'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                'load_saved_parameters':'/home/giannis/paper2/yolov8train/ultralytics/runs/detect/Yolov8runs_10k/Plodia/weights/best.pt',
                'test_save_images_path':'Yolov8_Test_10k/Plodia_color',
                'size':(480,480),
                'conf':0.3,
                'iou':0.70,
                'grey':False,
                'enable':False,
                'create_image':True,
                }, 
            ]

#general results
path_general_results='Yolov8_Test_10k_2/General_Results'
try:
    shutil.rmtree(path_general_results)
except:
    pass    
os.makedirs(path_general_results)
gen_res_logger=init_logger(path_general_results,'Yolov8_Test_10k_2')

for test_param in test_params:
    if test_param['enable']:
        try:
            shutil.rmtree(test_param['test_save_images_path'])
        except:
            pass    
        os.makedirs(test_param['test_save_images_path'])
        test_logger=init_logger(test_param['test_save_images_path'],test_param['model_name'])
        
        full_image_path=list(paths.list_images(test_param['test_image_path']))

        model = YOLO(model=test_param['load_saved_parameters']) 
    
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

        # NMS Parameters
        #model.predict(iou=test_param['iou'])
        #model.predict()
        
        test_pbar = tqdm(range(len(full_image_path)), f"Test in progress : {test_param['test_save_images_path']} Checkpoint : {test_param['load_saved_parameters']}" )
        #test_logger.info(f"Test in progress : {test_param['test_save_images_path']}  Checkpoint : {test_param['load_saved_parameters']} conf {test_param['conf']} iou {test_param['iou']} ")
        test_logger.info(f'Filename, Prediction , Label, Predict time')
        
        for img in full_image_path:
            #Init timer
            start_im_time = time.time()
            #label Value
            label=int(img.split(".")[-2].split("_")[-1:][0])
           
            image=cv2.imread(img)
            image=cv2.resize(image,test_param['size'],interpolation = cv2.INTER_AREA)
            if test_param['grey']:
                image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        
            results=model.predict(source=image,imgsz=test_param['size'][0],conf=test_param['conf'],iou=test_param['iou'])
            #results=model.predict(source=image,imgsz=test_param['size'][0])
            pred_label=0
            for result  in results:
                result = result.cpu()
                result = result.numpy()
                for box in result.boxes:
                    x0,y0,x1,y1=box.xyxy[0]
                    conf=box.conf[0]
                    obj=box.cls[0]
                    if float(obj)>0:
                        pred_label +=1
                        image=draw_win(image,int(x0),int(y0),int(x1),int(y1),None)
                
            
            # Update metrics
            if label > 20:
                sum_a_50_100 += a(pred_label, label)
                sum_mae_50_100 += abs(label-pred_label)
                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_50_100 += 1
                sum_mse_50_100 += math.pow((label-pred_label), 2)
                sum_time_50_100 += (time.time()-start_im_time)*1000
                sum_mape_50_100 += mape_fun(pred_label, label)
                count_50_100 += 1
            else:
                sum_a_1_20 += a(pred_label, label)
                sum_mae_1_20 += abs(label-pred_label)

                
                if abs(label-pred_label) <= label*percent_acceptable_error:
                    acceptable_errors_1_20 += 1

                sum_mse_1_20 += math.pow((label-pred_label), 2)
                sum_time_1_20 += (time.time()-start_im_time)*1000
                sum_mape_1_20 += mape_fun(pred_label, label)
                count_1_20 += 1

            if test_param['create_image']:
                image=cv2.putText(image,f"Yolo7: {pred_label}/{label}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
                image=cv2.putText(image,f"conf: {test_param['conf']} ",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
                image=cv2.putText(image,f"iou: {test_param['iou']}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
                cv2.imwrite(saved_image_path(test_param['test_save_images_path'],img,'dest'),image)
            #test_logger.info(f'{count} filename={img} label={label} pred={pred_label_n}')
            test_logger.info(f'{img}, {pred_label:.2f}, {label:.0f}, {(time.time()-start_im_time)*1000:.1f}')
            test_pbar.update()
        
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
        test_logger.info(
            f"{test_param['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
        gen_res_logger.info(
            f" {test_param['model_name']} \t Conf {test_param['conf']} \t IOU {test_param['iou']} \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.3f}% \t RMSE:{rmse:.4f} \t AE {avg_acceptable_errors:.2f}% \t AVG_TIME {(sum_time/(count)):.1f}")
        if count_1_20 > 0:
            gen_res_logger.info(f" {test_param['model_name']} 1-20 \t Conf {test_param['conf']} \t IOU {test_param['iou']} \t A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.3f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.4f} \t AE {acceptable_errors_1_20/count_1_20:.2f}% \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
        if count_50_100 > 0:
            gen_res_logger.info(f" {test_param['model_name']} 50-100 \t Conf {test_param['conf']} \t IOU {test_param['iou']} \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.3f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.4f} \t AE {acceptable_errors_50_100/count_50_100:.2f}% \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")

       
