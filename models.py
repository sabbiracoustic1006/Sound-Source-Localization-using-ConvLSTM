#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:18:27 2019

@author: ratul
"""

import torch, numpy as np
import os
from glob import glob
from scipy import io
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from itertools import combinations
import soundfile as sf
from time import time
from convLSTM import ConvLSTM


#fRoot = r'C:\ratul\thesis\spcup\dev_flight'+'\\'
gtSpeech = r'C:\ratul\thesis\spcup\speech_gt'+'\\'
gtNoise = r'C:\ratul\thesis\spcup\noise_gt'+'\\'

noiseRoot = r'C:\ratul\thesis\spcup\dev_static'+'\\'
speechRoot=r'C:\ratul\thesis\spcup\dev_static_speech'+'\\'

numSlices=8
idxSpeech=22050*numSlices

nRoot = r'C:\ratul\thesis\spcup\individual_motors_recordings'+'\\'

read_wav = lambda x:sf.read(x)[0]
db2Var = lambda s,rdb:s/10**(rdb/10)

torchStftModified = lambda audio:torch.stft(torch.tensor(audio),n_fft=4096,hop_length=2048,win_length=4096,
                           window=torch.tensor(np.sin(np.arange(.5,4096+.5)/4096*np.pi)))[1:,1:-1,-1].unsqueeze(0)


def timeit(func):
    def wrapper(*args):
        start = time()
        result = func(*args)
        print('%s took %.4f s'%(func.__name__,time()-start))
        return result
    return wrapper



def noiseGen(paths,nPower,flag):
    if flag == 'speech': end = idxSpeech
    else: end = 97020
    noise = np.zeros((end,8))
    for path in paths:
        start = np.random.randint(88200,617400-end)
        index = (slice(start,start+end),slice(0,8))
        noise += read_wav(path)[index] 
    
    pFactor = nPower/(noise**2).sum()
    
    return noise*(pFactor**0.5)
    


class spcupDataset(Dataset):
    def __init__(self,iterations,validation=False):

        if validation:
            self.paths = [glob(speechRoot+'*.wav')[-1]]#+ [glob(noiseRoot+'*.wav')[-1]]
        else:
            self.paths = glob(speechRoot+'*.wav')[:-1]#+ glob(noiseRoot+'*.wav')[:-1]
            
        self.noise_angles = [np.load(gtNoise+'%d.npy'%i) for i in range(3)]
        self.speech_angles = [np.load(gtSpeech+'%d.npy'%i) for i in range(3)]
#        self.flight_angles = [np.load(Gt+'flight\\%d.npy'%i) for i in range(15)]
        self.noisePaths = [path for path in glob(nRoot+'*.wav') if 'allMotors' not in path]
        self.allComb = list(combinations(self.noisePaths,4))
        self.allMotor = [path for path in glob(nRoot+'*.wav') if 'allMotors' in path]
        self.sIdx = [slice(i,i+22050) for i in range(0,int(44100*2.2),22050)][:-1]
        self.fIdx = [slice(i,i+22050) for i in range(0,idxSpeech,22050)]
        self.iterations = iterations
        self.dbs = np.arange(10,-2,-2)
        
    def __len__(self):
        return self.iterations
    

    def __getitem__(self,idx):
        path = self.paths[np.random.randint(len(self.paths))]
        idxAngle = int(path.split(os.sep)[-1].replace('.wav',''))-1
        
        if 'speech' in path:
            idxs = self.fIdx
            flag = 'speech'
            audio = read_wav(path)[:idxSpeech,:]
            angles = torch.tensor(self.speech_angles[idxAngle]).unsqueeze(0).repeat(numSlices,1,1,1)
        else: 
            audio = read_wav(path)
            idxs = self.sIdx
            flag = 'static'
            angles = torch.tensor(self.noise_angles[idxAngle]).unsqueeze(0).repeat(4,1,1,1)
            
        chosenDb = self.dbs[np.random.randint(len(self.dbs))]
         
        audio += noiseGen(self.allComb[np.random.randint(4845)],db2Var((audio**2).sum(),chosenDb),flag)
        fAudio = torch.cat([torch.cat(list(map(torchStftModified,audio[idx_].transpose(1,0))),0).unsqueeze(0) for idx_ in idxs],0)
        
        return fAudio,angles,chosenDb


dataLoader = DataLoader(spcupDataset(600),batch_size=1,shuffle=True,pin_memory=True)        
valLoader = DataLoader(spcupDataset(100,True),batch_size=1,shuffle=True,pin_memory=True)    


class Model(nn.Module):
    def __init__(self,p=48):
        super(Model,self).__init__()
        self.initBlk = nn.Sequential(nn.Conv2d(8,128,kernel_size=(1,7),stride=(1,3)),
                                     nn.BatchNorm2d(128),nn.ReLU(),
                                     nn.Conv2d(128,p,kernel_size=(1,5),stride=(1,2)),
                                     nn.BatchNorm2d(p),nn.ReLU(),nn.Dropout()
                                    )
        
        self.identityBlk1 = self.residualBlock(p)
        self.identityBlk2 = self.residualBlock(p)
        self.identityBlk3 = self.residualBlock(p)
        self.identityBlk4 = self.residualBlock(p)
        self.identityBlk5 = self.residualBlock(p)
        
        self.lastBlk1 = nn.Sequential(nn.Conv2d(p,360,1),nn.BatchNorm2d(360),nn.ReLU(),nn.Dropout())
#        self.lstmBlk = self.convLstm()
        self.finalBlk1 = nn.Sequential(nn.Conv2d(339,128,1),
                                      nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout())
        self.finalConv = nn.Conv2d(128,180,kernel_size=(9,1))
        
#                                      nn.Conv2d(128,180,kernel_size=(9,1)))

#        self.lastBlk2 = nn.Sequential(nn.Conv2d(p,360,1),nn.BatchNorm2d(360),nn.ReLU())
#        self.finalBlk2 = nn.Sequential(nn.Conv2d(339,128,1),
#                                      nn.BatchNorm2d(128),nn.ReLU(),
#                                      nn.Conv2d(128,180,kernel_size=(9,1)))

        
        
    @staticmethod
    def residualBlock(p):
        return nn.Sequential(nn.Conv2d(p,p,1,1,0),
                             nn.BatchNorm2d(p),nn.ReLU(),
                             nn.Conv2d(p,p,3,1,1),
                             nn.BatchNorm2d(p),nn.ReLU(),
                             nn.Conv2d(p,p,1,1,0),
                             nn.BatchNorm2d(p),nn.ReLU(),
                             nn.Dropout())
    @staticmethod   
    def convLstm(height=9,width=360,channels=128):
        return ConvLSTM(input_size=(height, width),
                        input_dim=channels,
                        hidden_dim=[64, 128],#, 180],
                        kernel_size=(3, 3),
                        num_layers=2,
                        batch_first=True,
                        bias=False,
                        return_all_layers=False)
        

    def forward(self,x):
        out = self.initBlk(x)
        out = self.identityBlk1(out)+out
        out = self.identityBlk2(out)+out
        out = self.identityBlk3(out)+out
        out = self.identityBlk4(out)+out
        out = self.identityBlk5(out)+out
        out = self.finalBlk1(self.lastBlk1(out).permute(0,3,2,1))
        out = self.finalConv(out).squeeze(2)
#        out = self.lstmBlk(self.finalBlk1(self.lastBlk1(out).permute(0,3,2,1)).unsqueeze(0))
#        out = self.finalConv(out.squeeze(0)).squeeze(2)
#        out2 = self.finalBlk2(self.lastBlk2(finalIdOut).permute(0,3,2,1)).squeeze(2)
        return out#,out2
    

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)



class lossFunc(nn.Module):
    def __init__(self,factor=0):
        super(lossFunc,self).__init__()
        self.mse = nn.MSELoss(size_average=False)
        self.mae = nn.L1Loss(size_average=False)
        self.factor = factor
        
    def forward(self,pred,label):
        idxLoss = 0
        for p,l in zip(pred,label[:,0]):
            idxLoss += self.mse((p == p.max()).nonzero().float(),(l == l.max()).nonzero().float())
        
        mseLoss = self.mse(pred,label[:,0])
        alpha = (mseLoss.item()/idxLoss.item() if idxLoss.item() != 0 else 1.0)
        loss = (mseLoss + self.factor*alpha*idxLoss)/len(label)
        #+ self.mse(pred[1],label[:,1])#10*torch.mul((pred[1] - label[:,1])**2,label[:,0]).sum()
        return loss
   

if __name__ == '__main__':
#    with torch.no_grad():
    output = model(torch.rand(2,8,9,2048).cuda())
    print(output.shape)
