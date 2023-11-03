import numpy as np
from RadarModel.functionLib import *


def getParameters(Type):
    return eval('get%sPara' % Type)()

def getSignalPara():
    ParaDic = {
        'CarrierFrequency': 300e6,
        'SampleFrequency' : 2e6,
        'SampleTime' : 0.1,
        'PulseWidth' : 0.01,
        'EnableNoise' : 0,
        'NoisePower' : 0, #0dB
        'SNR' : 5, #dB
        'Delay':0.03,
        'SimulationNum' : 1,
        'Type' : 'Signal'
    }
    return ParaDic

def getChirpPara():
    ParaDic = {
        'CarrierFrequency': 300e6,
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3,
        'PulseWidth' : 1e-4,
        'EnableNoise' : 1,
        'NoisePower' : 0, #0dB
        'SNR' : -20, #dB
        'Delay':0.1e-3,
        'BandWidth': 20e6,
        'SimulationNum' : 1,
        'Type' : 'Chirp'
    }
    return ParaDic

def getAMJammerPara():
    ParaDic = {
        'CenterFrequency': 300e6,
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3,
        # 'PulseWidth' : 0.01,
        'EnableNoise' : 1,
        'NoisePower' : 0, #0dB
        'INR' : 5, #dB
        'BandWidth': 10e6,
        'SimulationNum' : 1,
        'Type' : 'AMJam'
    }
    return ParaDic

def getNarrowPulseJammerPara():
    ParaDic = {
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3,
        'PRT' : 5e-5,
        'PulseWidth' : 1e-5,
        'RepeatNum' : 10,
        'EnableNoise' : 1,
        'NoisePower' : 0, #9
        'INR' : 50, #dB
        'Delay':0.03e-3,
        'SimulationNum' : 1,
        'Type' : 'NarrowPulseJam'
    }
    return ParaDic

def getSliceJammerPara():
    ParaDic = {
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3, 
        'PushTime' : 0, #每个切片转发之间的间隔 
        'SliceTime' : 3e-5,
        'RepeatNum' : 10,
        'EnableNoise' : 1,
        'NoisePower' : 0, #0dB
        'INR' : 30, #dB
        'Delay':1e-4, #延迟时间
        'SimulationNum' : 1,
        'Type' : 'SliceJam'
    }
    return ParaDic

def getDFTJammerPara():
    ParaDic = {
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3, 
        
        'PushTime' : 1e-5, #每个切片转发之间的间隔 
        'RepeatNum' : 10,
        'EnableNoise' : 1,
        'NoisePower' : 0, #0dB
        'INR' : 30, #dB
        'Delay':1e-4, #延迟时间
        'SimulationNum' : 1,
        'Type' : 'DFTJam'
    }
    return ParaDic

def getSmartNoiseJammerPara():
    ParaDic = {
        'SampleFrequency' : 40e6,
        'SampleTime' : 1e-3, 
        'PushTime' : 0, #每个切片转发之间的间隔 
        'SliceTime' : 3e-5,
        'RepeatNum' : 0,
        'EnableNoise' : 1,
        'NoisePower' : 0, #0dB
        'INR' : 10, #dB
        'Delay':1e-4, #延迟时间
        'SimulationNum' : 1,
        'Type' : 'SmartNoiseJam'
    }
    return ParaDic

def getRandPara(dic,par,num = 1):
    #配合打标签使用
    return eval('get%s' % par)(dic,num)
    

def getDelay(dic,num = 1):
    #延迟，需要先获取脉宽,重复次数，间隔时间
    if(dic['Type'] == 'Chirp'):
        return (dic['SampleTime']-dic['PulseWidth'])*np.random.rand(num)
    elif(dic['Type'] == 'NarrowPulseJam'):
        return (dic['SampleTime']-dic['PRT']*dic['RepeatNum'])*np.random.rand(num)
    elif(dic['Type'] == 'SliceJam'):
        return (dic['SampleTime']-dic['SliceTime']-(dic['SliceTime']+dic['PushTime'])*dic['RepeatNum'])*np.random.rand(num)
    elif(dic['Type'] == 'AMJam'):
        return 0
    elif(dic['Type'] == 'DFTJam'):
        return (dic['SampleTime']-(dic['SampleTime']/10+dic['PushTime'])*dic['RepeatNum'])*np.random.rand(num)
    elif(dic['Type'] == 'SmartNoiseJam'):
        a = (dic['SampleTime']-dic['RepeatNum']*(2*dic['SliceTime']+dic['PushTime']))/20
        return a*np.random.randint(20)
        

def getPulseWidth(dic,num = 1):
    #脉宽不大于PRT的1/5，不小于其1/10
    #需要先获取重复次数
    PRT = dic['SampleTime']
    result = PRT/10+1/10*PRT*np.random.rand(num)

    if(dic['Type'] == 'Chirp'):
        return result
    elif(dic['Type'] == 'AMJam'):
        return PRT
    elif(dic['Type'] == 'SliceJam'):
        #获取slicetime
        # a = PRT/(dic['RepeatNum'])
        # return a/10+a/4*np.random.rand(num)
        return PRT/40+1*PRT*np.random.rand(num)/40
    else:
        a = PRT/20
        return a/2+a/2*np.random.rand(num)

def getPower(dic,num = 1):
        #信号的信噪比,最小为脉压后13dB，最大脉压前18dB
    if(dic['Type'] == 'Chirp'):
        Gain = dic['SampleFrequency']*dic['PulseWidth']
        # result = 25-db(Gain)/2+5*np.random.rand(num)
        # result = 10 #
        result = 0 #6dB
        return result
    else:
        #干扰
        #0~10dB
        # if dic['Type'] == 'AMJam':
        #     return 20*np.ones(num)
        return 10*np.random.rand(num)
        # return 10*np.ones(num)

def getCenterFrequency(dic,num = 1):
    a = dic['SampleFrequency']-dic['BandWidth']
    return -a/2+a*np.random.rand(num)

def getBandWidth(dic,num = 1):
    a = dic['SampleFrequency']/2+dic['SampleFrequency']/2*np.random.rand(num)
    return a

def getRepeatNum(dic,num = 1):
    # N = int((dic['SampleTime']-dic['Delay'])/(dic['PushTime']+dic['SliceTime']))
    N = 10
    return 1+np.random.randint(N-1)#1-10个

def getNJPulseWidth(dic,num = 1):
    return 1/4*dic['PRT']+1/2*dic['PRT']*np.random.rand(num)

def getPushTime(dic,num = 1):
    if (dic['Type'] == 'SliceJam'):
        a = dic['SampleTime']/(dic['RepeatNum']+1)
        b = a/100
        return b*np.random.randint(5)
    elif(dic['Type'] == 'DFTJam'):
        a = dic['SampleTime']/(dic['RepeatNum'])
        b = (a-dic['SampleTime']/10)/20
        return b*np.random.randint(5)
        



if __name__ == '__main__':
    par = getParameters('Chirp')
    a = getRandPara(par,'PulseWidth',10)

    print(a.shape)
    