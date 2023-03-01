import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import reduce
from operator import __add__
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
        
CNN_layers_small=[
    {"filters":16, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":16, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":32, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":32, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
]

CNN_layers_large=[
    {"filters":16, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":False},
    {"filters":16, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":32, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":False},
    {"filters":32, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":64, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
    {"filters":64, "kernel":3,"kernel_regulazire":False,"batchnormalization":False,"poolmax":True},
]

Linear_backend_small=[
    {'size':64,'activation':'relu','dropout':0.2},
    {'size':32,'activation':'relu','dropout':0.2},
    {'size':16,'activation':'relu','dropout':0}
]

Linear_backend_large=[
    {'size':128,'activation':'relu','dropout':0.2},
    {'size':64,'activation':'relu','dropout':0.2},
    {'size':32,'activation':'relu','dropout':0},
    #{'size':16,'activation':'relu','dropout':0}
]

Linear_backend_resnet=[
    {'size':1024,'activation':'relu','dropout':0.2},
    {'size':512,'activation':'relu','dropout':0.2},
]

def make_layers_frontend(in_channels,frontend):
    layers=[]
    for i,cnn_layer in enumerate(frontend):
        conv2d = Conv2dSamePadding(in_channels,
                            cnn_layer["filters"],
                            kernel_size=cnn_layer['kernel'], 
                            #padding='same'
                            )
        if cnn_layer['batchnormalization']:
            layers += [conv2d, nn.BatchNorm2d(cnn_layer["filter"]), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        if cnn_layer['poolmax']:
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        in_channels=cnn_layer["filters"]
        #layers +=[nn.AdaptiveMaxPool2d(60)]

    return nn.Sequential(*layers) 

def make_layers_backend(input_layer_size,backend):
    layers=[]
    layers += [nn.Flatten()]
    previouse_size=input_layer_size
    for i,linear_layer in enumerate(backend):
        linear=nn.Linear(in_features=previouse_size,out_features=linear_layer['size'])
        if linear_layer['activation']=='relu':
            layers += [linear, nn.ReLU(inplace=True)]
        if linear_layer['dropout']>0:
            layers += [nn.Dropout(float(linear_layer['dropout']))]
        previouse_size=linear_layer['size']
    layers +=[nn.Linear(previouse_size,1)]
    return nn.Sequential(*layers)

class Count_Regression_Model(nn.Module):
    def __init__(self, input_shape=1,type='small'):
        super(Count_Regression_Model, self).__init__()
        self.input_shape=input_shape
        self.type=type
        if self.type=='small':
            self.frontend = make_layers_frontend(self.input_shape,CNN_layers_small)
            self.backend = make_layers_backend(32*30*20,Linear_backend_small)
        if self.type=='large':
            self.frontend = make_layers_frontend(self.input_shape,CNN_layers_large)
            self.backend = make_layers_backend(64*30*20,Linear_backend_large)
        if self.type=='resnet18':
            net= models.resnet18(pretrained=True)
            #self.set_parameter_requires_grad(net,False)
            net.fc=nn.Sequential(
                nn.Linear(net.fc.in_features,2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048,1024),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(1024,1),
            )
            self.frontend =net
            self.backend = None
            
            #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        if self.type=='resnet50':
            net= models.resnet50(pretrained=True)        
            #self.set_parameter_requires_grad(net,False)
            net.fc=nn.Sequential(
                nn.Linear(net.fc.in_features,2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048,1024),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(1024,1),
            )
            self.frontend =net
            self.backend = None
        
        if self.type=='vgg16':
            net= models.vgg16(pretrained=True)        
            #self.set_parameter_requires_grad(net,False)
            net.classifier[3] = nn.Linear(in_features=4096, out_features=2048)
            net.classifier[5] =nn.Dropout(0.2)
            net.classifier[6] = nn.Linear(in_features=2048, out_features=1)
            self.frontend =net
            self.backend = None

    def forward(self,x):
        
        x = self.frontend(x)
        if not self.backend is None:
            x = self.backend(x)
        return x
    
    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    