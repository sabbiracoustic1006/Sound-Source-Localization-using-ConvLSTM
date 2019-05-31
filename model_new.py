#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:04:52 2019

@author: ratul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:18:27 2019

@author: ratul
"""

import torch, numpy as np
from glob import glob
from scipy import io
from torch import nn
#from trainModelSTFT import timeit
from matplotlib import pyplot as plt

#%%
#fRoot = r'C:\ratul\thesis\spcup\dev_flight\\'
#sRoot = 'F:/code_new/sp_cup_2/dev_static/'
sRoot = r'C:\ratul\thesis\spcup\dev_static'+'\\'
std = 20


class Model(nn.Module):
    def __init__(self,numIdentityBlk=3):
        super(Model,self).__init__()
        self.initBlk = nn.Sequential(nn.Conv2d(16,32,kernel_size=(1,7),stride=(1,3)),
                                     nn.BatchNorm2d(32),nn.ReLU(),
                                     nn.Conv2d(32,128,kernel_size=(1,5),stride=(1,2)),
                                     nn.BatchNorm2d(128),nn.ReLU()
                                    )
        
        self.identityBlk = self.residualBlock()
        
        self.lastBlk1 = nn.Sequential(nn.Conv2d(128,360,1),nn.BatchNorm2d(360),nn.ReLU())
        self.finalBlk1 = nn.Sequential(nn.Conv2d(339,128,1),
                                      nn.BatchNorm2d(128),nn.ReLU(),
                                      nn.Conv2d(128,1,kernel_size=(9,1)))

        self.lastBlk2 = nn.Sequential(nn.Conv2d(128,360,1),nn.BatchNorm2d(360),nn.ReLU())
        self.finalBlk2 = nn.Sequential(nn.Conv2d(339,128,1),
                                      nn.BatchNorm2d(128),nn.ReLU(),
                                      nn.Conv2d(128,1,kernel_size=(9,1)))

        
        
    @staticmethod
    def residualBlock():
        return nn.Sequential(nn.Conv2d(128,128,1,1,0),
                             nn.BatchNorm2d(128),nn.ReLU(),
                             nn.Conv2d(128,128,3,1,1),
                             nn.BatchNorm2d(128),nn.ReLU(),
                             nn.Conv2d(128,128,1,1,0),
                             nn.BatchNorm2d(128),nn.ReLU())
        
#    @timeit
    def forward(self,x):
        out = self.initBlk(x)
        out = self.identityBlk(out)+out
        out = self.identityBlk(out)+out
        finalIdOut = self.identityBlk(out)+out
        out1 = self.finalBlk1(self.lastBlk1(finalIdOut).permute(0,3,2,1)).squeeze(1).squeeze(1)  
        out2 = self.finalBlk2(self.lastBlk2(finalIdOut).permute(0,3,2,1)).squeeze(1).squeeze(1)  
        return out1,out2
    

th =[2.3215,
    -2.3215,
    -0.7530,
     0.7530]


ph = [0.6848,
      0.6848,
      0.6522,
      0.6522]

thMotor = list(map(lambda x:int(x*180/np.pi)+180,th))
phMotor = list(map(lambda x:int(x*180/np.pi)+90,ph))


azimuthStatic,elevationStatic = io.loadmat(glob(sRoot+'*.mat')[0])['static_azimuth'],io.loadmat(glob(sRoot+'*.mat')[0])['static_elevation']
#azimuthFlight,elevationFlight = io.loadmat(glob(fRoot+'*new.mat')[0])['az'],io.loadmat(glob(fRoot+'*new.mat')[0])['el']
#%%
import pickle



all_azi_angle = np.arange(1,361).repeat(180).reshape(360,180).transpose()
all_elev_angle = np.arange(1,181).repeat(360).reshape(180,360)

def angSpec(az,el,idx,std=20):
    tot_azi = list(az[idx].astype('float32')+180.0)# + thMotor
    tot_el = list(el[idx].astype('float32')+90.0) #+ phMotor
        
    allAngleScores = []
 
    for azimuth,elevation in zip(tot_azi,tot_el):
        allAngleScores.append(np.exp(-(np.array([(all_azi_angle - azimuth)**2,(all_elev_angle - elevation)**2])/std**2).sum(0)))

#    angle_score = np.max(np.asarray(allAngleScores),0)
    angle_score = np.asarray(allAngleScores)
#    aziScore2 = np.asarray(np.argmin(np.array(list(map(lambda x:abs(all_azi_angle-x),tot_azi))),0)==0, dtype = int)
#    eleScore2 = np.asarray(np.argmin(np.array(list(map(lambda x:abs(all_elev_angle-x),tot_el))),0)==0, dtype = int)
#    zz = np.asarray(np.logical_and(aziScore2, eleScore2),dtype=int)
    
    return angle_score#, zz
   
def angSpecFlight(tot_azi,tot_el):        
    allAngleScores = []
 
    for azimuth,elevation in zip(tot_azi,tot_el):
        allAngleScores.append(np.exp(-(np.array([(all_azi_angle - azimuth)**2,(all_elev_angle - elevation)**2])/std**2).sum(0)))

#    angle_score = np.max(np.asarray(allAngleScores),0)
    angle_score = np.asarray(allAngleScores)
#    aziScore2 = np.asarray(np.argmin(np.array(list(map(lambda x:abs(all_azi_angle-x),tot_azi))),0)==0, dtype = int)
#    eleScore2 = np.asarray(np.argmin(np.array(list(map(lambda x:abs(all_elev_angle-x),tot_el))),0)==0, dtype = int)
#    zz = np.asarray(np.logical_and(aziScore2, eleScore2),dtype=int)
#    
    return angle_score#, zz
   

def get_dkh(az,el,path, idx, mode = 'static/',std = 20):
    
    if mode=='static/':
        angleScores = angSpec(az,el,idx)                
    else:
        tot_azi = list(map(lambda x:[int(x)]+thMotor,az[idx]+180))
        tot_el = list(map(lambda x:[int(x)]+phMotor,el[idx]+90))
        
        angleScores = []
        for azimuth,elevation in zip(tot_azi,tot_el):
            angleScores.append(angSpecFlight(azimuth,elevation))
        
    np.save(path + str(idx) + '.npy', angleScores)
#    np.save(path + mode + str(idx) + '.npy', [azi_score,ele_score,aziScore2,eleScore2])
    
    
from tqdm import tqdm
        
path = sRoot.replace('dev_static','noise_gt')
#path = r'C:\ratul\thesis\spcup\flight'+'\\'

for i in tqdm(range(len(elevationStatic))):
    get_dkh(azimuthStatic,elevationStatic,path,i,mode = 'static/')
    
#
#for i in tqdm(range(len(elevationFlight))):
#    get_dkh(azimuthFlight,elevationFlight,path,i,mode = 'flight/')

#%%

#class lossFunc(nn.Module):
#    def __init__(self):
#        super(lossFunc,self).__init__()



#    
#if __name__ == '__main__':
#    model = Model().cuda()
#    output = model(torch.rand(2,16,9,2048).cuda())
#    print(output.shape)