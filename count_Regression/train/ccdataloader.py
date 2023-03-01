from ccdataset import CCDataset
import torch
from torch.utils.data import  DataLoader
import numpy as np


class CCDataLoader(object):
    
    def __init__(self, image_dirname,
                 transform=None,
                 grey=False,
                 shuffle=True,
                 Input_Image_resize=None,
                 batch_size=1,
                 num_workers=1,
                 subset=0,
                 ):
        self.ds=CCDataset(image_dirname,
                          grey,
                          Input_Image_resize,
                          transform
                        )
        if subset>0:
            self.ds= torch.utils.data.Subset(self.ds,np.arange(subset))
        if not self.ds is None:
            self.dataloader = DataLoader(dataset=self.ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def __len__(self):
        return len(self.ds)
    

