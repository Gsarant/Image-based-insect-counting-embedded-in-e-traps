from ccdataloader import CCDataLoader
import numpy as np
from torchvision import transforms

transform=transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation((-45,45)),
                                transforms.RandomSolarize(0.3),
                                transforms.RandomAutocontrast()
                                ])

def create_loader(dataset_paths):
    
    loaders=[]
    
    # Create Data Loaders

    for p in dataset_paths:
        if p['enable']==True:
            train_loader=CCDataLoader(p['train_image_path'],
                                transform,
                                p['grey'],
                                True,
                                p['Input_Image_resize'],
                                16,
                                4,
                                0)
            val_loader=CCDataLoader(p['val_image_path'],
                                transform,
                                p['grey'],
                                True,
                                p['Input_Image_resize'],
                                16,
                                4,
                                0)
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

