#!pip install imutils
import cv2
import torch
import os
from datetime import datetime
from tqdm import tqdm
import shutil
import sys
sys.path.append('..')
from gsutils  import find_edge,denorm,draw_circle,draw_win,saved_image_path


today = datetime.today().strftime("%Y%m%d")



tmp_folder='../'

dataset_paths = [
        {'name':'Helicoverpa_armigera',
        'train_image_path':f'{tmp_folder}../data/datasets_Helicoverpa_armigera_10k/images/val',
        'train_labels_path':f'{tmp_folder}../data/datasets_Helicoverpa_armigera_10k/labels/val',
        'train_save_path':'../Trainboxes/Helicoverpa_armigera',
        'size':(480,320)
        },
        {'name':'Plodia_interpunctella',
        'train_image_path':f'{tmp_folder}../data//datasets_Plodia_interpunctella_10k/images/val',
        'train_labels_path':f'{tmp_folder}../data//datasets_Plodia_interpunctella_10k/labels/val',
        'train_save_path':'../Trainboxes/Plodia_interpunctella',
        'size':(480,320)
        }
    ]
                    
for test_param in dataset_paths:
    try:
        shutil.rmtree(test_param['train_save_path'])
    except:
        pass    
    os.makedirs(test_param['train_save_path'])
    
    images=[]
    boxes=[]
    for s in os.listdir(test_param['train_image_path']):
        image_name=s.split(os.sep)[-1]
        ext_imgename=image_name.split(".")[-1]

        if ext_imgename=='jpg':
            images.append(image_name)
            boxes.append(f'{image_name.split(".")[-2]}.txt')
    test_pbar = tqdm(range(len(images)), f"Test in progress : {test_param['train_image_path']} " )
    for image_name,box_name in zip(images,boxes):
        image_full_path=os.path.join(test_param['train_image_path'],image_name)
        boxes_full_path=os.path.join(test_param['train_labels_path'],box_name)
        img=cv2.imread(image_full_path)
        if not img is None:
            img=cv2.resize(img,test_param['size'],cv2.INTER_AREA)
            with open(boxes_full_path,'r') as f:
                count=0
                for l in f.readlines():
                    l_array=l.split('  ')
                    if l_array[0] != '0':
                        x=l_array[1]
                        y=l_array[2]
                        w=l_array[3]
                        h=l_array[4]
                        x,y,w,h=denorm(x,y,w,h,img.shape[1],img.shape[0])
                        x0,y0,x1,y1=find_edge(x,y,w,h,img.shape[1],img.shape[0])
                        img=draw_win(img,x0,y0,x1,y1)
                        x=x0+int((x1-x0)/2)
                        y=y0+int((y1-y0)/2)
                        img=draw_circle(img,x,y,3)
                        count +=1
        else:
            pass
        img=cv2.putText(img,f"{count}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imwrite(os.path.join(test_param['train_save_path'],image_name),img)
        test_pbar.update()
