from re import A
import numpy as np
import cv2
import os

from rembg import remove #https://github.com/danielgatis/rembg
from PIL import Image
from imutils import paths


# Load and resize image
def load_image(path_image,size_w=0,size_h=0):
            try:
                image=cv2.imread(path_image)
                if size_w!=0 and size_h!=0:
                    image=cv2.resize(image,(int(size_w),int(size_h)), cv2.INTER_AREA)
                return image
            except:
                return None

# Remove files from folder
def remove_files(folder):
        if os.path.isdir(folder):
            for n in os.listdir(folder):
                path=os.path.join(folder,n)
                if os.path.isfile(path):
                    os.remove(path)

# Create folders of path
def create_folders(folders):
    folders_array=folders.split(os.sep)
    for f in  range(len(folders_array)):
        cur_folder=os.path.sep.join(folders_array[:(f+1)])
        try:
            os.mkdir(cur_folder)
        except:
            pass

# Create insect images from insects in specific position in Funnel image
def main():
    a='/home/giannis/paper2/'

    path_images=[
                {'paths_original_insect':f'{a}data/class_2_fixed',
                    'path_dataset':f'{a}data/datasets_Plodia_interpunctella/Plodia_interpunctella_insects_new'},
                {'paths_original_insect':f'{a}data/class_1_fixed',
                    'path_dataset':f'{a}data/datasets_Helicoverpa_armigera/Helicoverpa_armigera_insects_new'}
                #{'paths_original_insect':f'{a}data/class_2_fixed',
                #    'path_dataset':f'{a}data/potamitis/Plodia_interpunctella_insects'},
                #{'paths_original_insect':f'{a}data/class_1_fixed',
                #    'path_dataset':f'{a}data/potamitis/Helicoverpa_armigera_insects'}
                ]

    for path_image in path_images:
        create_folders(path_image['path_dataset'])
        remove_files(path_image['path_dataset'])
        indx=0
        for index,full_path_insect in enumerate(paths.list_images(path_image['paths_original_insect'])):
            image=load_image(full_path_insect)
            if image is not None:
                image_corp=image[120:350,820:1100]
                
                #image_insect=Image_Insect(image)
                image_insect=remove(image_corp)
                
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

            
                filename=full_path_insect.split(os.sep)[-1:][0]
                splitfilename=filename.split('.')
                #newfilename=f"{splitfilename[-2:][0]}_{indx}_2.{splitfilename[-1:][0]}"
                newfilename=f"{splitfilename[-2:][0]}_{indx}.{splitfilename[-1:][0]}"
                path_save_insect=os.path.join(path_image['path_dataset'],newfilename)
                cv2.imwrite(path_save_insect,image_insect)
                #newfilename=f"{splitfilename[-2:][0]}_{indx}_1.{splitfilename[-1:][0]}"
                #path_save_insect=os.path.join(path_image['path_dataset'],newfilename)
                #cv2.imwrite(path_save_insect,image_corp)
                #newfilename=f"{splitfilename[-2:][0]}_{indx}_0.{splitfilename[-1:][0]}"
                #path_save_insect=os.path.join(path_image['path_dataset'],newfilename)
                #cv2.imwrite(path_save_insect,image)
                indx += 1
                print(f' saved image {path_save_insect}')
                
    print('End')

        
if __name__ == '__main__':
    main()