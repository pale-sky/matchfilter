import numpy as np
import matplotlib.pyplot as plt
from Signal import *


SliceNum = 100
NoiseMean,NoiseVar = 0,1
x = (np.random.normal(NoiseMean,NoiseVar,[10*SliceNum,SliceNum])+1j*np.random.normal(NoiseMean,NoiseVar,[10*SliceNum,SliceNum]))
# a = (np.random.randn(SliceNum)+1j*np.random.randn(SliceNum))/2**0.5
# x[:50,:] += 100
a = Signal.get_skewnessORkurtosis(x.real,kind = 'kurtosis')
b = Signal.get_maxAmplitudeORstd(x)
c = Signal.get_phiStdORbandORnormVar(x,kind = 'normVar')
d = Signal.get_envelopORcorRatio(x,kind = 'corRatio')


print(d)

pillar = 16
real = plt.hist(a.real,bins=pillar,density = True,range = [-0.01,0.01])
# imag = plt.hist(a.imag,bins=pillar,density = True,range = [-0.01,0.01])

pltData = real[0]
# pltData = imag[0]
xlab = real[1][:pillar]


plt.figure()
plt.plot(xlab,real[0])
plt.show()


fs = 1000
t = np.arange(0,1,1/fs)
f0 = 100
x = np.exp(2j*np.pi*f0*t)
f = np.arange(len(t))*fs/len(t)

plt.figure()
plt.plot(f,abs(np.fft.fft(x)))
plt.show()
