#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 01:54:21 2019

@author: ratul
"""

import os,torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np


#%%

class DefaultConfigs(object):
    def __init__(self,key,epochs,lr):
        self.key = key
        self.epochs = epochs
        self.lr = lr
        self.batch_size = 1
        self.cores = 6
        os.makedirs('saved_models/%s'%key,exist_ok=True)

        
config = DefaultConfigs('justSpeech_10db_0db',60,0.0001)

def saveFig(predicted,labels,db,epoch,idx):
    
    for idxStep,(pred,lab) in enumerate(zip(predicted,labels[:,0])):
        plt.subplot(2,1,1)
        plt.imshow(pred.cpu().numpy())
        plt.subplot(2,1,2)
        plt.imshow(lab.cpu().numpy())
        if len(labels) == 15: type_ = 'flight'
        else: type_ = 'static'
        saveDir = 'plots/%s/train/epoch_%d/%s_%d'%(config.key,epoch+1,type_,idx+1)
        os.makedirs(saveDir,exist_ok=True)
        plt.savefig('%s/step%d_dB%d.png'%(saveDir,idxStep,db))
        plt.close('all')

    

def perform_validation(dataloader,model,lossFunc,epoch):
    validation_loss = []
    model.eval()
    count = 0
    with torch.no_grad():
        for idx,(img,labels,db) in enumerate(dataloader):
            img,labels = img.cuda().squeeze(0).permute(0,1,3,2).float(),labels.cuda().squeeze(0).float()
            predicted = model(img)
            loss = lossFunc(predicted,labels)
            if count < 10:
                if np.random.randint(2):
#                i = np.random.randint(0,len(labels))
                    for idxStep,(pred,lab) in enumerate(zip(predicted,labels[:,0])):
                        plt.subplot(2,1,1)
                        plt.imshow(pred.cpu().numpy())
                        plt.subplot(2,1,2)
                        plt.imshow(lab.cpu().numpy())
                        if len(labels) == 15: type_ = 'flight'
                        else: type_ = 'static'
                        saveDir = 'plots/%s/validation/epoch_%d/%s_%d'%(config.key,epoch+1,type_,idx+1)
                        os.makedirs(saveDir,exist_ok=True)
#            plt.subplot(2,2,3)
#            plt.imshow(labels[i][0].cpu().numpy())
#            plt.subplot(2,2,4)
#            plt.imshow(labels[i][1].cpu().numpy())
#            
                        plt.savefig('%s/step%d_dB%d.png'%(saveDir,idxStep,db))
                        plt.close('all')
                    count += 1

            validation_loss.append(loss.item())
    return sum(validation_loss)/len(validation_loss)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class TQDM(Callback):

    def __init__(self):
        """
        TQDM Progress Bar callback

        This callback is automatically applied to 
        every SuperModule if verbose > 0
        """
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs

    def on_epoch_begin(self, epoch):
        try:
            self.progbar = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar.set_description('Epoch %i/%i' % 
                            (epoch+1, self.train_logs['num_epoch']))
        except:
            pass

    def on_epoch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self):
        self.progbar.update(1)

    def on_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar.set_postfix(log_data)
        


calcAcc = lambda p,l:(p.max(1)[1]==l).sum().float()/len(l)