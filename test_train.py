from model import Base
from utils import RadarDataSet
import torch.utils.data as tud
import torch
from tqdm import tqdm
import torch.nn as nn
from loss import CELoss
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train(datapath = './data/train',savepath = './weights'):

    BATCH_SIZE = 8
    LEARN_RATE = 1e-4
    EPOCHS = 500

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(savepath): os.makedirs(savepath)

    dataset = RadarDataSet(datapath)
    dataloader = tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    total_step = len(dataloader)

    model = Base(channel_num=1).to(device)

    # loss_fn = nn.MSELoss()
    loss_fn = CELoss()
    
    optimal = torch.optim.Adam(model.parameters(),lr=LEARN_RATE)

    loss_record = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for i,data in tqdm(enumerate(dataloader),total=total_step,desc=f'Epoch {epoch}/{EPOCHS}'):
            inputdata = data['sigData'].to(device)
            ref = data['refSig'].to(device)
            labelData = data['label'].to(device)

            output = model(inputdata,ref)

            loss = loss_fn(output,labelData)

            total_loss += loss.item()
            optimal.zero_grad()
            loss.backward()
            optimal.step()
        
        avg_loss = total_loss/total_step
        
        if epoch>0:
            if avg_loss<loss_record[-1]:
                torch.save(model.state_dict(),savepath+f'/model_{model.Block.ModelType}_{avg_loss:.3f}.pth')
        else:
            torch.save(model.state_dict(),savepath+f'/model_{model.Block.ModelType}_{avg_loss:.3f}.pth')

        loss_record.append(avg_loss)
        tqdm.write(f'Average Loss: {avg_loss:.4f}')
        
    f = open('loss.txt','w')
    f.write(str(loss_record))
    f.close()

def predict(datapath = './data/test/data_5.npy',weightspath = './weights/model_Base_0.026.pth'):
    import numpy as np
    import matplotlib.pyplot as plt
    from RadarModel.functionLib import db
    data = np.load(datapath, allow_pickle=True)
    inputdata = data.item()['sigData']
    ref = data.item()['refSig']
    label = data.item()['label']

    model = Base(channel_num=1)
    model.load_state_dict(torch.load(weightspath))

    inputdata[:,:,900:1150] += inputdata[:,:,1000:1250]
    result = model(inputdata,ref)

    sig = result[:,:,0,:]+1j*result[:,:,1,:]
    ref = ref[:,0,:]+1j*ref[:,1,:] 
    inputdata = inputdata[:,0,:]+1j*inputdata[:,1,:]

    

    ref = torch.conj(ref).flip(dims=[-1])
    pc = torch.fft.ifft(torch.fft.fft(inputdata,dim = -1)*torch.fft.fft(ref,dim = -1),dim = -1)
    
    x = range(2000)

    plt.figure()
    plt.plot(x,inputdata.squeeze().real)
    plt.plot(x,db(sig.detach().numpy().squeeze()))
    plt.show()

    plt.figure()
    plt.plot(x,label[:,0,:].squeeze())
    plt.plot(x,db(pc.squeeze()))
    plt.plot(x,inputdata.squeeze().real)
    plt.show()

    plt.figure()    
    plt.plot(x,db(pc.squeeze()))
    plt.plot(x,db(sig.detach().numpy().squeeze()))
    plt.legend(['pc','model_output'])
    plt.show()



if __name__ == '__main__':
    train()
    # predict()