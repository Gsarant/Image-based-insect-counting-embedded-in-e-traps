import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
import os
import importlib
import sys
sys.path.append('..')
from gsutils import init_logger,EarlyStopping



def train(train_loader,model,criterion,optimizer,epoch,device,logger):
    train_loss=0.0
    train_items=0
    mae=0.0
    model.train()
    start_epoch_time = time.time()
    pbar_epoch = tqdm(range(len(train_loader.dataloader)), f"Training in progress Epoch:{str(epoch)}")
    for i,(img, label,name) in enumerate(train_loader.dataloader):
        start_im_time = time.time()
        img = img.to(device)
        #img = Variable(img)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):   
            output = model(img)
            output=output.to(device)
            label=label.to(device)
            loss = criterion(output.squeeze().float(),label.squeeze().float())
            loss.backward()
            optimizer.step()
            output=output.data.cpu().detach().float().mean().item()
            label=label.data.cpu().detach().float().mean().item()
            train_loss  += loss.item() * img.size(0)
            train_items +=img.size(0)
            mae += abs(output-label) * img.size(0)
            pbar_epoch.set_description(f'Loss:{train_loss/train_items:.6f}  MAE:{mae/train_items:.2f} Total time {(time.time()-start_im_time)*1000:.3f}')
            pbar_epoch.update()
            start_im_time = time.time()
    epoch_loss = train_loss / train_items
    mae = mae/train_items
    epoch_time=time.time()-start_epoch_time
    pbar_epoch.set_description(f'Epoch Time  {epoch_time//60:.0f}m {epoch_time%60:.0f}s MAE:{mae:.4f} Loss:{epoch_loss:.6f}')
    
    return epoch_loss

def validate(test_loader,model,criterion,device):
    val_loss=0.0
    test_items=0
    mae=0.0
    model.eval()
    #pbar_epoch_val = tqdm(range(len(test_loader.dataloader)), f"Val in progress Epoch:{str(epoch)}")
    for i,(img, label,name)in enumerate(test_loader.dataloader):
        img = img.to(device)
        with torch.set_grad_enabled(False):
            output = model(img)
            output=output.to(device)
            label=label.to(device)
            loss = criterion(output.squeeze().float(),label.squeeze().float())
            output=output.data.cpu().detach().float().mean().item()
            label=label.data.cpu().detach().float().mean().item()
            val_loss  += loss.item() * img.size(0)
            test_items +=img.size(0)
            mae += abs(output-label)* img.size(0)
            #pbar_epoch_val.update()
    mae = mae/test_items    
    val_epoch_loss=val_loss / test_items
    #print(f'test_items {test_items}')
    return val_epoch_loss,mae


def run_train(loaders,device,log_dir,ckpt_dir):
    for loader in loaders:
        #Init Logger
        logger=init_logger(log_dir,loader['name'])
        
        early_stopping = EarlyStopping(30,0.00001)
        
        if len(loader['preload_model'])>0 :
            model = torch.load(loader['preload_model'])
        else:
           #model =CSRNet(1,True)
           model_class = importlib.import_module('count_regression_model')
           #model=getattr(model_class, loader['model_class'])(1 if loader['grey'] else 3,'small')
           model=getattr(model_class, loader['model_class'])(int(loader['model_class_attribute'][0]),str(loader['model_class_attribute'][1])) 
        
        model=model.to(device)
        
        #criterion = nn.MSELoss(size_average=False).to(device)
        criterion = nn.MSELoss().to(device)
        val_loss=0
        best_mae = np.inf
        best_val_loss= np.inf
        previouse_best_saved_mae=None
        previouse_best_saved_loss=None
        pbar = tqdm(range(loader['epochs']), f"Training in progress : {loader['name']} Model :{loader['model_name']}" )
        logger.info(f"Name,Model,Epoch,Training Loss,Val Loss,Mae,Best Mae")    
        for epoch in range(loader['epochs']):
            #logger.info(f"Training in progress : {loader['name']} Model :{loader['model_name']}" )
            #adjust_learning_rate(optimizer, epoch)
          
            lr = 1e-3
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-08, amsgrad=False)

            train_loss=train(loader['train_loader'],model,criterion,optimizer,epoch,device,logger)

            val_loss,mae=validate(loader['val_loader'],model,criterion,device)
       
            logger.info(f"{loader['name']},{loader['model_name']},{loader['start_epoch']+epoch+1},{train_loss:.6f},{val_loss:.6f},{mae:.4f},{best_mae:.4f}")    
            pbar.set_description(f"Name :{loader['name']} Model :{loader['model_name']} Epoch {loader['start_epoch']+epoch+1}/{loader['start_epoch']+loader['epochs']} Training Loss:{train_loss:.4f} Val Loss:{val_loss:.4f} Mae:{mae:.4f} Best Mae :{best_mae:.4f}")    
        
            pbar.update()
            if mae<best_mae:
                best_mae=mae
                if previouse_best_saved_mae is not None:
                    os.remove(previouse_best_saved_mae)
                saved_path_model= f"{ckpt_dir}/{loader['name']}-{best_val_loss:.8f}-mae {best_mae}-({loader['model_name']}_ep_{loader['start_epoch']+epoch+1}.pt"
                torch.save(model,saved_path_model)
                previouse_best_saved_mae=saved_path_model

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if previouse_best_saved_loss is not None:
                    os.remove(previouse_best_saved_loss)
                saved_path_model= f"{ckpt_dir}/{loader['name']}-{best_val_loss:.8f}-valloss {best_val_loss}- {loader['model_name']}_ep_{loader['start_epoch']+epoch+1}.pt"
                torch.save(model,saved_path_model)
                previouse_best_saved_loss=saved_path_model
                #logger.info(f'Saved pt file {saved_path_model} Epoch {epoch}')
            #Early Stop
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break     
        if loader['epochs']>0:
            last_saved_path_model=f"{ckpt_dir}/{loader['name']}-{val_loss:.8f}-last-{loader['name']}_ep_{loader['start_epoch']+epoch+1}.pt"
            torch.save(model,last_saved_path_model)
            #logger.info(f'Saved pt file {last_saved_path_model}')

