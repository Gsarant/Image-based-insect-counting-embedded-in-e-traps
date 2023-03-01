import os
import tensorflow as tf
from imutils import paths
import numpy as np

import shutil
import time
import cv2
from nms import nms_python
import sys
sys.path.append('..')
from gsutils  import init_logger,a,saved_image_path,draw_win,load_images_inference,denorm,find_edge
tmp_folder='../'
tflite_parameters=[
                {'model_name':'Yolov7_Helicoverpal_color_overlap',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_tf/best.tflite',
                    'size':(480,320),
                    'grey':False,
                    'score_thres':0.4,
                    'iou_thresh':0.4,
                    'create_image':True},
                 {'model_name':'Yolov7_Helicoverpal_grey_overlap',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera_crowd_overlap/all_overlap',
                    'path_saved_model_tflite':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_tf/best.tflite',
                    'size':(480,320),
                    'grey':True,
                    'score_thres':0.4,
                    'iou_thresh':0.4,
                    'create_image':True},

                {'model_name':'Yolov7_Helicoverpal_color',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_tf/best.tflite',
                    'size':(480,320),
                    'grey':False,
                    'score_thres':0.4,
                    'iou_thresh':0.4,
                    'create_image':True},
                {'model_name':'Yolov7_Helicoverpal_grey',
                    'test_image_path':f'{tmp_folder}../data/Helicoverpa_armigera',
                    'path_saved_model_tflite':'../export/yolov7_10k/Helicoverpa_armigera_HVGA_tf/best.tflite',
                    'size':(480,320),
                    'grey':True,
                    'score_thres':0.4,
                    'iou_thresh':0.4,
                    'create_image':True},
                    
                {'model_name':'Yolov7_plodia_color',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'../export/yolov7_10k/Plodia_interpunctella_HVGA__tf/best.tflite',
                    'size':(480,320),
                    'grey':False,
                    'score_thres':0.3,
                    'iou_thresh':0.7,
                    'create_image':True},
                {'model_name':'Yolov7_plodia_grey',
                    'test_image_path':f'{tmp_folder}../data/Plodia_interpunctella',
                    'path_saved_model_tflite':'../export/yolov7_10k/Plodia_interpunctella_HVGA__tf/best.tflite',
                    'size':(480,320),
                    'grey':True,
                    'score_thres':0.3,
                    'iou_thresh':0.7,
                    'create_image':True},
                     
]
LOG='log_10k'
try:
    shutil.rmtree(LOG)
except:
    pass    
os.makedirs(LOG)
#general results
gen_res_logger=init_logger(LOG,'yolov7_10k_genResults')

for tflite_parameter in tflite_parameters:

    logger=init_logger(LOG,tflite_parameter['model_name'])
    save_path=os.path.join(LOG,tflite_parameter['model_name'])    
    try:
        shutil.rmtree(save_path)
    except:
        pass    
    os.makedirs(save_path)

    interpreter_qt = tf.lite.Interpreter(
      model_path=tflite_parameter['path_saved_model_tflite'], num_threads=None)
    
    interpreter_qt.allocate_tensors()

    input_details = interpreter_qt.get_input_details()

    output_details = interpreter_qt.get_output_details()

   # height = input_details[0]['shape'][1]
   # width = input_details[0]['shape'][2]
    
    #Init metrics
    sum_mae_1_20=0.0
    sum_a_1_20=0
    sum_time_1_20=0
    count_1_20=0

    sum_mae_50_100=0.0
    sum_a_50_100=0
    sum_time_50_100=0
    count_50_100=0
    true_count=0
    full_image_path=list(paths.list_images(tflite_parameter['test_image_path']))
    
    logger.info(f"Test in progress : Model :{tflite_parameter['model_name']} Checkpoint : {tflite_parameter['path_saved_model_tflite']}")
    
    logger.info(f"Filename,  Prediction, Label,  Predict time")    
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
            score = row[5:]
            class_id = np.argmax(score)
            if class_id>0:
                #confidence = score[class_id]
                if confidence>tflite_parameter['score_thres'] :
                # print(confidence)
                    #cx, cy, w, h = row[0], row[1], row[2], row[3]
                    cy, cx, h, w = row[0], row[1], row[2], row[3]
                    
                    #use this with tf.image.non_max_suppression
                    #boxes.append([cy, cx, cy+h, cx+w])
                    
                    #use this with nms_python
                    boxes.append([cx-(w/2), cy-(h/2), cx+(w/2), cy+(h/2) ])
                    scores.append(confidence)
                    class_ids.append(class_id)
        
        selected_indices=[]
        if len(boxes)>0:        
            #selected_indices=tf.image.non_max_suppression(boxes,
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
        
       
        
         #Update metrixs
        if label>20:
            sum_a_50_100 +=a(pred_label , label)     
            sum_mae_50_100 += abs(pred_label-label)
            sum_time_50_100 += (time.time()-start_im_time)*1000
            count_50_100 +=1
        else:
            sum_a_1_20 +=a(pred_label , label)     
            sum_mae_1_20 += abs(pred_label-label)
            sum_time_1_20 += (time.time()-start_im_time)*1000
            count_1_20 +=1
        #Update log
        logger.info(f"{image_name}, {pred_label:.2f}, {label:.0f},  {(time.time()-start_im_time)*1000:.1f}")    
        #logger.info(f"{image_name},  Prediction {pred_label} Label {label} Predict time {(time.time()-start_im_time)*1000:.3f}")    
        
    #Final metrixs    
   #Final metrixs
    count=count_1_20 + count_50_100
    avg_mae = (sum_mae_1_20 + sum_mae_50_100)/count
    avg_a = (sum_a_1_20 + sum_a_50_100)/count
    sum_time=sum_time_1_20 + sum_time_50_100
    logger.info(f" {tflite_parameter['model_name']}  A:{avg_a:.4f}  Mae:{avg_mae:.4f} {true_count}/{count}")
    gen_res_logger.info(f" {tflite_parameter['model_name']} CONF {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh'] }  A:{avg_a:.4f}  Mae:{avg_mae:.4f} {true_count}/{count} AVG_TIME {(sum_time/count):.1f}")

    if count_1_20>0:
        gen_res_logger.info(f" {tflite_parameter['model_name']} Conf {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh']} 1-20  A:{sum_a_1_20/count_1_20:.4f}  Mae:{sum_mae_1_20/count_1_20:.4f}  AVG_TIME {(sum_time_1_20/count_1_20):.1f}")
    if count_50_100>0:
        gen_res_logger.info(f" {tflite_parameter['model_name']} Conf {tflite_parameter['score_thres']} IOU {tflite_parameter['iou_thresh']} 50-100  A:{sum_a_50_100/count_50_100:.4f}  Mae:{sum_mae_50_100/count_50_100:.4f}  AVG_TIME {(sum_time_50_100/count_50_100):.1f}")


