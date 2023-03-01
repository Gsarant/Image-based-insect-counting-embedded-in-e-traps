import os

from torchvision import transforms
from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
 

class CCDataset(Dataset):
    def __init__(self, image_dirname,
                 grey=False,
                 Input_Image_resize=None,
                 transform= None
                 
                ):
        self.image_dirname=image_dirname
        self.images_list_name=np.array(glob.glob(f"{self.image_dirname}/*.jpg"))
        self.images_list_name=[]
        list_names=np.array(glob.glob(f"{self.image_dirname}/*.jpg"))
        for image_name in list_names:
            if int(image_name.split(".")[-2].split("_")[-1:][0]) <=25:
                self.images_list_name.append(image_name)
                
        self.grey=grey
        self.Input_Image_resize=Input_Image_resize
        self.transform=transform
        
    
    def __len__(self):
        return len(self.images_list_name)

    
    def __getitem__(self, idx):
        image_name = self.images_list_name[idx]
        image=cv2.imread(image_name)
        
        if self.grey:
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            
        if not self.Input_Image_resize is None:
            image = cv2.resize(image, self.Input_Image_resize, interpolation=cv2.INTER_AREA)

        label=int(image_name.split(".")[-2].split("_")[-1:][0])
        image =np.float32(image/255.0)
        if not self.transform is None:
            image=self.transform(image)
        return image,label,image_name