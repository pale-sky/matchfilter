import torch.utils.data as tud
import numpy as np
import os
import torch


def comx2real(data):
    # data:[slowTime,FastTime]
    if 1 == len(data.shape):
        data = data.reshape(-1,1)
    data = data.T[:,np.newaxis,:]
    result = torch.zeros([data.shape[0],2,data.shape[2]])
    result[:,0,:] = torch.Tensor(data.real).squeeze()
    result[:,1,:] = torch.Tensor(data.imag).squeeze()
    return result

class RadarDataSet(tud.Dataset):
    data = []

    def __init__(self,path = './data'):
        super().__init__()
        for i in os.listdir(path):
            self.data.append(np.load(path+'/'+i,allow_pickle=True).item())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

if __name__ == '__main__':
    path = './data'
    dataset = RadarDataSet(path)
    dataloader = tud.DataLoader(dataset,batch_size=1,shuffle=False)
    
    data = next(iter(dataset))
    print(data['sigData'])
    print(data['refSig'])
    print(data['profile'])
    


