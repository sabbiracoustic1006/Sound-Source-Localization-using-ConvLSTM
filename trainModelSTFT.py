# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:18:03 2019

@author: papon
"""

import torch
from functions import config, perform_validation,TQDM,saveFig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import save_checkpoint
from utils import csvLogger, resumeTrain, plot
from models import model,dataLoader,valLoader,optimizer,lossFunc
#%%


resumeArgs = {'key':config.key,
              'model':model,
              'optimizer':optimizer,
              'epoch':100,
              'new_lr':1e-4,
              'resumeTrain':True,
              'model_initial_weight':None}
              
#%% 

model,optimizer,epochStrt,epochEnd,best_val_loss = resumeTrain(**resumeArgs)
  
scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,min_lr=1e-7)

lossFunction = lossFunc().cuda()

best_results = [best_val_loss]

#amp_handle = amp.init(enabled=False)

with TQDM() as pbar:
    for epoch in range(epochStrt,epochEnd):
        pbar.on_train_begin({'num_batches':len(dataLoader),'num_epoch':epochEnd+1})
        pbar.on_epoch_begin(epoch)
        losses = []
        
        model.train()
        for idx,(img,labels,db) in enumerate(dataLoader):
            img,labels=img.cuda().squeeze(0).permute(0,1,3,2).float(),labels.cuda().squeeze(0).float()
            
            pbar.on_batch_begin()
            
            predicted = model(img)
            
            loss = lossFunction(predicted,labels)
            
#            with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
#                scaled_loss.backward()
            loss.backward()
            optimizer.step()    
            optimizer.zero_grad()
            
            losses.append(loss.item())
            
            pbar.on_batch_end(logs={'loss':loss})
            
            with torch.no_grad():
                if idx % 100 == 0: saveFig(predicted,labels,db,epoch,idx)
                                                                                                                                                                                 
        val_metrics = perform_validation(valLoader,model,lossFunction,epoch)
        
        scheduler.step(val_metrics)
        
        is_best_val_loss = val_metrics < best_results[0]
    
        if is_best_val_loss:state = 'only val loss improved from %.4f to %.4f'%(best_results[0],val_metrics)
        else: state = 'No Improvement'
        
        best_results[0] = min(val_metrics,best_results[0])
        
        save_checkpoint({  "epoch":epoch + 1,
                           "model_name":config.key,
                           "state_dict":model.state_dict(),
                           "best_loss":best_results[0],
                           "optimizer":optimizer.state_dict(),
                },is_best_val_loss)
    
       
    
        csvArgs = {'key':config.key,
                   'epoch':epoch+1,
                   'learning rate':optimizer.state_dict()['param_groups'][0]['lr'],
                   'train_loss':sum(losses)/len(losses),
                   'val_loss':val_metrics,
                   'state':state
                   }
        
        csvLogger(**csvArgs)
        
        if epoch == epochEnd:
            plot()
            
            
        pbar.on_epoch_end({'loss':sum(losses)/len(losses),'val_loss':val_metrics,'parameters':sum([param.sum() for param in model.parameters()])})
    
