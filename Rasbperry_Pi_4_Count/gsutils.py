import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging
from datetime import datetime
from torchvision import  transforms
from matplotlib import cm


def create_head_image(image,image_predict,str_model,label,alpha=0.7,pred_label=None):
    
    #cmap = cm.get_cmap('jet')
    #image_predict = np.uint8(255 * image_predict)
    #jet_colors = cmap(np.arange(256))[:, :3]
    #jet_heatmap = jet_colors[image_predict]
    #jet_heatmap = np.uint8(255 * jet_heatmap)
    #superimposed_img = jet_heatmap * alpha + image
    
    
    #cmap = cm.get_cmap('jet')
    #overlay = (255 * cmap(np.asarray(image_predict) ** 2)[:, :, :3]).astype(np.uint8)
    #superimposed_img = (alpha * np.asarray(image) + (1 - alpha) * overlay).astype(np.uint8)
    image_predict = np.maximum(image_predict, 0) / np.amax(image_predict)
    image_predict = np.uint8(255 * image_predict)
    overlay=cv2.applyColorMap(image_predict,cv2.COLORMAP_JET)
    superimposed_img=cv2.addWeighted(image,alpha,overlay,1-alpha,0.0)
    if pred_label is None:
        return cv2.putText(superimposed_img,f"{str_model}: {label:.0f}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    else:
        return cv2.putText(superimposed_img,f"{str_model}: {pred_label:.1f}/{label:.0f}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


def init_logger(log_dir,log_name,time_per_line=False):
     #Init Logger
    today = datetime.today().strftime("%Y%m%d")
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    if time_per_line:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        formatter = logging.Formatter("%(message)s")
    fh = logging.FileHandler(f"{log_dir}/{log_name}_{today}.log")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def draw_win(img,x0,y0,x1,y1,confidence=None,color=(255,255,255)):
    img=cv2.rectangle(img,(x0,y0),(x1,y1),color,1)
    if not confidence is None:
        img=cv2.putText(img,f'{confidence:.2f}',(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    #img=cv2.circle(img,(int(x0+((x1-x0)/2)),int(y0+((y1-y0)/2))),2,(255,0,0),2)
    return img

def find_edge(x,y,w,h,width=240,height=240):
    x0=int(float(x))
    y0=int(float(y))
    x1=int(x0+float(w))
    y1=int(y0+float(h))
    return x0,y0,x1,y1

def denorm(x,y,w,h,width=240,height=240):
    w=int(float(w)*width)
    h=int(float(h)*height)
    x=int(float(x)*width)-int(w/2)
    y=int(float(y)*height)-int(h/2)
    return x,y,w,h

def draw_circle(img,x,y,r=10,color=(255,0,0)):
    return cv2.circle(img,(x,y),r,color)

def saved_image_path(test_save_images_path,image_full_path,type_im):
    image_name=image_full_path.split(os.sep)[-1:]
    
    image_name=f'{type_im}_{image_name[0]}'
    im_orig_path=os.path.join(test_save_images_path,image_name)
    return im_orig_path

def load_images_inference(image_name,size,grey,d3_grey=False):
    image=cv2.imread(image_name)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image_before_tranform=image.copy()
    
    if grey:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
        image_before_tranform=image.copy()
        if d3_grey:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)    
            image=np.transpose(image,[2,1,0])     
            image=np.expand_dims(image, axis=0)   
        else:
            image=np.expand_dims(image, axis=0)   
            if len(image.shape)==4:
                image=np.transpose(image,[0,3,2,1])     
            if len(image.shape)==3:
                image=np.transpose(image,[0,2,1])     
                image=np.expand_dims(image, axis=0)   
    else:
        image=np.expand_dims(image, axis=0)   
        image=np.transpose(image,[0,3,2,1])     
    image = (np.float32(image) / 255.0)
    return image.astype(np.float32),image_before_tranform

def load_image(image_name,size,transform,grey=False):
    image=cv2.imread(image_name)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image_before_tranform=image.copy()
    if grey:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)   
        image1=image.copy() 
        grey_transform = transforms.Compose([])
        for obj in  transform.transforms:
            if not type(obj) is transforms.Normalize:
                grey_transform.transforms.append(obj)
            #else:    
            #    grey_mean=[obj.mean[:1]]
            #    grey_std=[obj.std[:1]]
            #    grey_normalize=transforms.Normalize(mean=grey_mean,std=grey_std)
            #    #grey_transform.transforms.append(grey_normalize)
        #image=np.array(image)
        image=np.float32(image) / 255.0
        image = grey_transform(image)
    else:
        image = transform(image).float()
    image=image[None,:]
    return image ,image_before_tranform

def a(predict,label):
    #α=1-|Μc-Αc|/Mc
    #print(f'1-|{label}-{predict}|/{label}')
    if float(label)!=0.0:
        return float(1-(abs(float(label)-float(predict))/float(label)))
    else:
        return 0

def mape_fun(predict,label):
    if float(label)!=0.0:
        return (abs((float(label)-float(predict))/float(label)))*100
    else:
        return 0

def display_loader(loader):
    img,label,name = next(iter(loader.dataloader))
    print(name[0])
    img = img.cpu().data[0].permute(1, 2, 0)
    img = img.numpy()
    image = Image.open(name[0])
    image = image.convert('RGB')
    #image=np.array(image)
    #image_scale= cv2.resize(image, (image.shape[1] //4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)    
    #image_scale= cv2.resize(image, (int(image.shape[1] /8), int(image.shape[0] / 8)), interpolation=cv2.INTER_AREA)    
    image_scale=image.resize((int(image.size[0] /8), int(image.size[1] / 8))) 
    den = label.cpu().data[0]
    den = den.squeeze().cpu().numpy()
    print(f" name: {name}  img: {img.shape}, den: {den.shape}, count: {np.sum(den)}")
    #plt.imshow(img)
    #plt.show()
    
    #plt.imshow(image_scale)
    #plt.imshow(den, alpha=0.3)
    #plt.show()
    #plt.imshow(den, alpha=0.3)
    #plt.show()
    plt.imshow(image)
    plt.show()

    
class ScaleDown(object):
    """Scale-down the density map"""
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, density):
        den=np.array(density)
        (w, h) = density.shape
        density = cv2.resize(density, (w / self.factor, h / self.factor), interpolation=cv2.INTER_CUBIC)*64
        return density

class Resize(object):
    def __init__(self, width,heigth):
        self.width = width
        self.heigth=heigth

    def __call__(self, image):
        image=np.array(image)
        #image=np.moveaxis(image,0,-1)
        #image=image.permute(1,2,0)
        print(self.width,self.heigth,image.shape)
        image = cv2.resize(image,( self.width,self.heigth), interpolation=cv2.INTER_CUBIC)*64
        return image

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
