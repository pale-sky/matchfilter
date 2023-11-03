from RadarModel.Signal import Chirp
from RadarModel.parameters import getParameters,getRandPara
from tqdm import   tqdm
import os
import numpy as np
import torch
from utils import comx2real
import matplotlib.pyplot as plt
from RadarModel.functionLib import db

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Sample_Num = int(1e4)

def updatePara(par):
    # par['PulseWidth'] = getRandPara(par,'PulseWidth')
    par['Delay'] = getRandPara(par,'Delay')
    # par['SNR'] = getRandPara(par,'Power')
    par['SNR'] = -5
    # par['CarrierFrequency'] = getRandPara(par,'CenterFrequency')


def radarDataProcess(sig):
    sigData = sig.SignalGenerate()

    if 1 == len(sigData.shape):
        sigData = sigData.reshape(1,-1)

    profile = sig.pulseCompress(sigData)
    labelData = np.zeros([profile.shape[1],2,profile.shape[0]]).astype('float32')
    labelData[:,:,abs(profile).argmax(0)] = 1

    # plt.figure()
    # plt.plot(range(2001),db(profile)/db(profile).max(),marker = 'o',linestyle = 'dashed')
    # plt.plot(range(2001),labelData[:,0,:].squeeze(),marker = '*',linestyle = 'dashed')
    # plt.show()

    return sigData[:,:-1].T,labelData[:,:,:-1]


def main():
    para = getParameters('Chirp')
    para['SimulationNum'] = 1
    para['EnableNoise'] = 1
    para['SampleFrequency'] = 2e6
    para['BandWidth'] = 1e6
    para['SNR'] = -5

    path = './train'
    if not os.path.exists(path): os.makedirs(path)
    
    for i in tqdm(range(Sample_Num),desc='Generating'):
        sig = Chirp(para)
        sigData,labelData =  radarDataProcess(sig)

        data = {
            'sigData': comx2real(sigData),
            # 'refSig':comx2real(sig.RefSignal[:int(sig.PulseWidth*sig.RefSignal.shape[0]/sig.SampleTime)]),
            'refSig':comx2real(sig.RefSignal[:-1]),
            'label':torch.Tensor(labelData),
        }

        updatePara(para)
        np.save(path+f'/data_{i}.npy',data)
        

if __name__ == '__main__':
    main()