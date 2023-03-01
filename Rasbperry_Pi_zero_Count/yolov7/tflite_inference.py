import os
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
from imutils import paths
import numpy as np
from tqdm import tqdm

import shutil
import time
import cv2
import math

from nms import nms_python
import sys
sys.path.append('..')
from gsutils import init_logger, a, load_images_inference, saved_image_path, mape_fun
tmp_folder=''
tflite_parameters=[
                 {'model_name':'Yolov7_Helicoverpal',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'models/Helicoverpa_yolov7.tflite',
                    'size':(480,320),
                    'grey':False,
                    'score_thres':0.3,
                    'iou_thresh':0.4,
                    'obj':1,
                    'create_image':False},
                     
                    
                {'model_name':'Yolov7_plodia',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'models/Plodia_yolov7.tflite',
                    'size':(480,320),
                    'grey':False,
                    'score_thres':0.3,
                    'iou_thresh':0.8,
                    'obj':1,
                    'create_image':False},
                     
]
LOG='log_10k'
try:
    shutil.rmtree(LOG)
except:
    pass    
os.makedirs(LOG)
#general results
gen_res_logger=init_logger(LOG,'yolov7_genResults')

for tflite_parameter in tflite_parameters:

    logger=init_logger(LOG,tflite_parameter['model_name'])
    save_path=os.path.join(LOG,tflite_parameter['model_name'])    
    try:
        shutil.rmtree(save_path)
    except:
        pass    
    os.makedirs(save_path)
    interpreter_qt = tflite.Interpreter(
      model_path=tflite_parameter['path_saved_model_tflite'], num_threads=None)
    
    interpreter_qt.allocate_tensors()

    input_details = interpreter_qt.get_input_details()

    output_details = interpreter_qt.get_output_details()

   # height = input_details[0]['shape'][1]
   # width = input_details[0]['shape'][2]
    
   
    true_count=0
    full_image_path=list(paths.list_images(tflite_parameter['test_image_path']))
    #init bar
    pbar_epoch = tqdm(range(len(full_image_path)), f"Inference image in progress")
    
    logger.info(f"Test in progress : Model :{tflite_parameter['model_name']} Checkpoint : {tflite_parameter['path_saved_model_tflite']}")
    
    logger.info(f"Filename,  Prediction, Label,  Predict time")    
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
        image,image_before_tranform=load_images_inference(image_name,tflite_parameter['size'],tflite_parameter['grey'],tflite_parameter['grey'])
        
        label=int(image_name.split(".")[-2].split("_")[-1:][0])
  
        scale, zero_point = input_details[0]['quantization']
        if scale>0 or zero_point>0: 
            input_data= image / (scale*1.0) + (zero_point*1.0)
        
        input_data= image
  
       # input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])
        input_data = input_data.astype(input_details[0]["dtype"])
        interpreter_qt.set_tensor(input_details[0]['index'], input_data)
        interpreter_qt.invoke()
  
    
        output_data = interpreter_qt.get_tensor(output_details[0]['index'])
  
        scale, zero_point = output_details[0]['quantization']
        if scale>0 or zero_point>0: 
            output_data = ((scale*1.0) * (output_data - zero_point*1.0))

        class_ids = []
        scores = []
        boxes = []
        for row in np.squeeze(np.squeeze(output_data)):
            confidence = row[4]
            score = row[5:]*confidence

            class_id = np.argmax(score)
            #confidence = score[class_id]
            if confidence>tflite_parameter['score_thres'] and class_id==tflite_parameter['obj']:
               # print(confidence)
                #cx, cy, w, h = row[0], row[1], row[2], row[3]
                cy, cx, h, w = row[0], row[1], row[2], row[3]
                
                #use this with tf.image.non_max_suppression
                #boxes.append([cy, cx, cy+h, cx+w])
                
                #use this with nms_python
                boxes.append([cx, cy, cx+w, cy+h ])
                scores.append(confidence)
                class_ids.append(class_id)
        
        selected_indices=[]
        if len(boxes)>0:        
            # selected_indices=tf.image.non_max_suppression(boxes,
            #                                          scores,
            #                                          200,
            #                                          tflite_parameter['iou_thresh'],
            #                                          tflite_parameter['score_thres'])
       
            selected_indices=nms_python(boxes,
                                   scores,
                                   tflite_parameter['score_thres'],
                                   tflite_parameter['iou_thresh'])
        
        pred_label=len(selected_indices)       
        if pred_label==label:
            true_count+=1
        
        if tflite_parameter['create_image']:
            for box in selected_indices:
                x0,y0,x1,y1=boxes[box]
                #x,y,w,h=denorm(x0,y0,x0+x1,y0+y1,image_before_tranform.shape[1],image_before_tranform.shape[0])
                #x0,y0,x1,y1=find_edge(x,y,w,h,image_before_tranform.shape[1],image_before_tranform.shape[0])
                image_before_tranform=draw_win(image_before_tranform,int(x0),int(y0),int(x1),int(y1))
            image_before_tranform=cv2.putText(image_before_tranform,f"{pred_label}/{label}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            image_before_tranform=cv2.putText(image_before_tranform,f"conf {tflite_parameter['score_thres']} iou {tflite_parameter['iou_thresh']}",(20,70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.imwrite(saved_image_path(save_path,image_name,'dest'),image_before_tranform)
        
       
        
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
        pbar_epoch.set_description(f'A:{(sum_a_50_100+sum_a_1_20) / (count_50_100 + count_1_20) :.6f}  MAE:{ (sum_mae_50_100+sum_mae_1_20)/(count_50_100+count_1_20):.2f} Total time {(time.time()-start_im_time)*1000:.3f}')
        pbar_epoch.update()
        #Update log
        logger.info(f"{image_name}, {pred_label:.2f}, {label:.0f},  {(time.time()-start_im_time)*1000:.1f}")    
        #logger.info(f"{image_name},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")    
    
    #Final metrics
    count=count_1_20 + count_50_100
    mae = (sum_mae_1_20 + sum_mae_50_100)/count
    avg_a = (sum_a_1_20 + sum_a_50_100)/count
    mape=(sum_mape_1_20+sum_mape_50_100)/count
    mse=(sum_mse_1_20+sum_mse_50_100)/count
    rmse=math.sqrt(mse)
    sum_time=sum_time_1_20 + sum_time_50_100
    logger.info(f"{tflite_parameter['model_name']} Mae:{mae:.4f} a:{avg_a:.4f}")
    gen_res_logger.info(f" {tflite_parameter['model_name']} \t CONF {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh'] }  \t A:{avg_a:.4f} \t MAE:{mae:.4f} \t MSE:{mse:.4f} \t MAPE:{mape:.2f}% \t RMSE:{rmse:.4f} \t AVG_TIME {(sum_time/(count)):.1f}")
    if count_1_20>0:
        gen_res_logger.info(f" {tflite_parameter['model_name']}  1-20   \t CONF {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh'] } \t  A:{sum_a_1_20/count_1_20:.4f} \t MAE:{sum_mae_1_20/count_1_20:.4f} \t MSE:{sum_mse_1_20/count_1_20:.4f} \t MAPE:{sum_mape_1_20/count_1_20:.2f}% \t RMSE:{math.sqrt(sum_mse_1_20/count_1_20):.2f} \t AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
    if count_50_100>0:
        gen_res_logger.info(f" {tflite_parameter['model_name']}  50-100 \t CONF {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh'] }  \t A:{sum_a_50_100/count_50_100:.4f} \t MAE:{sum_mae_50_100/count_50_100:.4f} \t MSE:{sum_mse_50_100/count_50_100:.4f} \t MAPE:{sum_mape_50_100/count_50_100:.2f}% \t RMSE:{math.sqrt(sum_mse_50_100/count_50_100):.2f} \t AVG_TIME {(sum_time_50_100/count_50_100):.1f}")
