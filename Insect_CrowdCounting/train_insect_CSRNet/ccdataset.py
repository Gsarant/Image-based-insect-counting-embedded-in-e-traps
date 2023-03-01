import os
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
import torchvision.transforms.functional as F


class CCDataset(Dataset):
    def __init__(self, image_dirname,
                 density_dirname,
                 image_transforms=None,
                 density_transforms=None,
                 grey=False,
                 Input_Image_resize=None,
                 kernel_size=15,
                 create_growndtruth_images_train=None):
        self.image_dirname=image_dirname
        self.density_dirname=density_dirname
        self.image_transforms = image_transforms
        self.density_transforms=density_transforms
        self.images_list_name=glob.glob(f"{self.image_dirname}/*.jpg")
        self.create_growndtruth_images_train=create_growndtruth_images_train
        self.grey=grey
        self.kernel_size=kernel_size
        self.Input_Image_resize=Input_Image_resize
        
    
    def __len__(self):
        return len(self.images_list_name)

    
    def __getitem__(self, idx):
        image_name = self.images_list_name[idx]
        #image = Image.open(image_name)
        image=cv2.imread(image_name)
        
        if self.grey:
            #image=ImageOps.grayscale(image)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            
        if not self.Input_Image_resize is None:
            #image=image.resize(self.Input_Image_resize)
            image = cv2.resize(image, self.Input_Image_resize, interpolation=cv2.INTER_AREA)

        if  not self.create_growndtruth_images_train is None:
            den= self.create_growndtruth_images_train.create_growndtruth_from_image(
                image_name,
                self.density_dirname,
                self.kernel_size,#3
                4,#2
            )
        else:
            den=None

        if self.image_transforms is not None:
            if self.grey:
                grey_transform = transforms.Compose([])
                for obj in  self.image_transforms.transforms:
                    if type(obj) is transforms.Normalize:
                        grey_mean=[obj.mean[:1]]
                        grey_std=[obj.std[:1]]
                        grey_normalize=transforms.Normalize(mean=grey_mean,std=grey_std)
                        #grey_transform.transforms.append(grey_normalize)
                    else:
                        grey_transform.transforms.append(obj)
                pixels = np.asarray(image)
                # convert from integers to floats
                pixels = pixels.astype('float32')
                # normalize to the range 0-1
                pixels /= 255.0
                image = grey_transform(pixels)
            else:
                image = self.image_transforms(image)
        #den2=cv2.resize(den,(den.shape[1]//8,den.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
        #den=cv2.resize(den,(den.shape[1]//8,den.shape[0]//8),interpolation = cv2.INTER_AREA)*64
        #den=den/255.0
        #_den1=np.sum(den)
        #_den2=np.sum(den2)
        den=cv2.resize(den,(den.shape[1]//8,den.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
        if self.density_transforms is not None:
            den = self.density_transforms(den) 
        
        return image,den,image_name