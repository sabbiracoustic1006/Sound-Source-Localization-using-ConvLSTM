#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 04:52:15 2019

@author: ratul
"""


import torch, os
import pandas as pd
import shutil
from functions import config
from matplotlib import pyplot as plt

# save best model
def save_checkpoint(state, is_best_val_loss):
    filename = "saved_models/%s/checkpoint.pth.tar"%config.key
    torch.save(state, filename)
    shutil.copyfile(filename,"saved_models/%s/current_epoch.pth.tar"%(config.key))
    if is_best_val_loss:
        shutil.copyfile(filename,"saved_models/%s/%s_best_val_loss.pth.tar"%(config.key,config.key))
    

def csvLogger(**args):
    dic = {}
    if os.path.exists('logs/%s_log.csv'%args['key']):
        df = pd.read_csv('logs/%s_log.csv'%args['key'])
        for key in df.keys():
            tmp = df[key].values
            dic[key] = list(tmp) + [args[key]]
        pd.DataFrame(dic).to_csv('logs/%s_log.csv'%args['key'],index=False)
    else:
        for key in args.keys():
            dic[key] = [args[key]]
        pd.DataFrame(dic).to_csv('logs/%s_log.csv'%args['key'],index=False)    
        
def resumeTrain(**args):
    os.makedirs('logs',exist_ok=True)
    if os.path.exists('logs/%s_log.csv'%args['key']) and args['resumeTrain']:
        logs = pd.read_csv('logs/%s_log.csv'%args['key'])
        state = torch.load('saved_models/%s/current_epoch.pth.tar'%config.key)
        args['model'].load_state_dict(state['state_dict'])
        args['optimizer'].load_state_dict(state['optimizer'])
        if not args['new_lr']: args['optimizer'].state_dict()['param_groups'][0]['lr'] = logs['learning rate'].values[-1];print('last learning rate loaded')
        else: args['optimizer'].state_dict()['param_groups'][0]['lr'] = args['new_lr'];print('New learning rate loaded')
        epoch_start = logs['epoch'].values[-1]
        bL = logs['val_loss'].values.min()
    else:
        if args['model_initial_weight']: args['model'].load_state_dict(torch.load(args['model_initial_weight']))
        epoch_start = list(range(args['epoch']))[0]
        bL = 999999
    epoch_end = list(range(args['epoch']))[-1]
    
    return args['model'],args['optimizer'],epoch_start,epoch_end,bL


def plot():
    df = pd.read_csv('logs/%s_log.csv'%config.key)
    epochs = df['epoch'].values
    train_losses = df['train_loss'].values
    val_losses = df['val_loss'].values
    train_iou = df['train_iou'].values
    val_iou = df['val_iou'].values
    
    plt.figure(figsize=(7,7))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs,train_losses,'r--',epochs,val_losses,'k')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    

    plt.subplot(2, 1, 2)
    plt.plot(epochs,train_iou,'r--',epochs,val_iou,'k')
    plt.title('IOU graph')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    
    plt.show()
    