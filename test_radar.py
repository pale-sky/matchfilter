from RadarModel.Signal import Chirp
import matplotlib.pyplot as plt
from RadarModel.functionLib import db


if __name__ == '__main__':

    from RadarModel.parameters import getParameters
    import matplotlib.pyplot as plt
    
    para = getParameters('Chirp')
    para['SimulationNum'] = 1
    para['EnableNoise'] = 0

    sig = Chirp(para)
    sigData = sig.SignalGenerate()
    profile = sig.pulseCompress(sigData)

    plt.figure()
    plt.plot(sig.TimeAxis,db(profile))
    plt.show()

