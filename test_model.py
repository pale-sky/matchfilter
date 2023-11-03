from RadarModel.Signal import Chirp
from model import Base
from utils import comx2real
import torch
#   数据格式：B*C*2*L
#   不能每个通道发不同波形

def main():
    from RadarModel.parameters import getParameters
    from RadarModel.functionLib import db
    import matplotlib.pyplot as plt
    
    para = getParameters('Chirp')
    para['SimulationNum'] = 1
    para['EnableNoise'] = 1
    para['SampleFrequency'] = 4e6
    para['BandWidth'] = 2e6

    sig = Chirp(para)
    sigData = sig.SignalGenerate()
    ref = sig.RefSignal[:int(sig.PulseWidth*sig.RefSignal.shape[0]/sig.SampleTime)]

    ref = comx2real(ref).unsqueeze(0)
    sigData = comx2real(sigData).unsqueeze(0)
    

    model = Base(1)
    output = model(sigData,ref)
    




if __name__ == '__main__':
    main()