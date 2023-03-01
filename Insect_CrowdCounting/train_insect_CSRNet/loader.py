from torchvision import transforms
from create_growndTruth_image import Create_GrowndTruth_Image

from ccdataloader import CCDataLoader

import torch.nn as nn


image_transform= transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

class InterpolateDesnityImage(object):
    def __call__(self,dmap,scale=8,upscale=False):
        dmap=dmap.unsqueeze(0) 
        if upscale:
            dmap=nn.functional.interpolate(dmap, scale_factor=(scale,scale),mode='area')
        else:
            dmap=nn.functional.interpolate(dmap,scale_factor=(1/scale,1/scale),mode='area')
        return dmap.squeeze()

#density_transform= transforms.Compose([transforms.ToTensor(),InterpolateDesnityImage()])
density_transform= transforms.Compose([transforms.ToTensor()])

def create_loader(dataset_paths):
    
    loaders=[]
    
    # Create Data Loaders

    for p in dataset_paths:
        if p['enable']==True:
            create_growndtruth_images_train=Create_GrowndTruth_Image(p['train_image_path'],
                                p['train_density_path'],
                                p['Input_Image_resize'],
                                )
            train_loader=CCDataLoader(p['train_image_path'],
                                p['train_density_path'],
                                image_transform,
                                density_transform,
                                p['grey'],
                                True,
                                p['Input_Image_resize'],
                                p['kernel_size'],
                                1,
                                4,
                                0,
                                create_growndtruth_images_train)
            val_loader=CCDataLoader(p['val_image_path'],
                                p['val_density_path'],
                                image_transform,
                                density_transform,
                                p['grey'],
                                True,
                                p['Input_Image_resize'],
                                p['kernel_size'],
                                1,
                                4,
                                0,
                                create_growndtruth_images_train)
            loaders.append({'name':p['name'],
                            'model_name':p['model_name'],
                            'model_class':p['model_class'],
                            'model_class_attribute':p['model_class_attribute'],
                            'grey':p['grey'],
                            'train_loader':train_loader,
                            'val_loader':val_loader,
                            'start_epoch':p['start_epoch'],
                            'epochs':p['epochs'],
                            'enable':p['enable'],
                            'preload_model':p['preload_model']
                        })
            print(f"name:{p['name']} train dataset={len(train_loader)}  val dataset={len(val_loader)}")  
    return loaders           
    #logger.info(f"name:{p['name']} train dataset={len(train_loader)}  val dataset={len(val_loader)}")             

