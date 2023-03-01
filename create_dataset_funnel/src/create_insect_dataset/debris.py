import numpy as np
import cv2
import os

from rembg import remove #https://github.com/danielgatis/rembg
from PIL import Image
from imutils import paths
a='/home/giannis/paper2/'
path_dirtes=f'{a}data/dirtes_'
path_dirtes_new=f'{a}data/dirtes'
# Load and resize image
def load_image(path_image,size_w=0,size_h=0):
            try:
                image=cv2.imread(path_image)
                if size_w!=0 and size_h!=0:
                    image=cv2.resize(image,(int(size_w),int(size_h)), cv2.INTER_AREA)
                return image
            except:
                return None

for indx,image_name in enumerate(paths.list_images(path_dirtes)):
    image=load_image(image_name)
    if image is not None:
        image_insect=remove(image)
        image_insect=cv2.cvtColor(image_insect, cv2.COLOR_RGBA2RGB)
        image_insect =np.where(image_insect==0 ,255,image_insect)
    
        h,w=image_insect.shape[:2]
        mask=np.zeros((h+2,w+2),np.uint8)
    
        image_insect_grey=cv2.cvtColor(image_insect,cv2.COLOR_RGB2GRAY)

        image_insect_grey = cv2.dilate(image_insect_grey, None, iterations=2)
        #image_insect = cv2.erode(image_insect, None, iterations=2)
        #cv2.floodFill(im_flood_fill,mask,(0,0),255)
        image_insect[:,:,0] = np.where( image_insect_grey < 250, image_insect[:,:,0], 255)
        image_insect[:,:,1] = np.where( image_insect_grey < 250, image_insect[:,:,1], 255)
        image_insect[:,:,2] = np.where( image_insect_grey < 250, image_insect[:,:,2], 255)

    
        filename=image_name.split(os.sep)[-1:][0]
        splitfilename=filename.split('.')
        newfilename=f"{splitfilename[-2:][0]}_{indx}.{splitfilename[-1:][0]}"
        path_save_insect=os.path.join(path_dirtes_new,newfilename)
        cv2.imwrite(path_save_insect,image_insect)
        indx += 1
        print(f' saved image {path_save_insect}')

