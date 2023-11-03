from model import Base
from utils import RadarDataSet
import torch.utils.data as tud
import torch
from tqdm import tqdm
import torch.nn as nn
from models.TransformerModel.model.transformer import Transformer
from loss import CELoss
from conf import *
import os

def train(datapath = './data/train',savepath = './weights'):

    dataset = RadarDataSet(datapath)
    dataloader = tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    total_step = len(dataloader)

    if not os.path.exists(savepath): os.makedirs(savepath)

    FeatureBlock = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

    model = Base(channel_num=1,FeatureBlock=FeatureBlock).to(device)

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

            output = model(inputdata.squeeze().transpose(1,2),ref.squeeze().transpose(1,2),labelData.squeeze().transpose(1,2)) 
            # output = model(inputdata.squeeze(),ref.squeeze(),labelData.squeeze()) 
            
            # loss = loss_fn(labelData.squeeze(),output.squeeze())
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
        tqdm.write(f'Average Loss: {avg_loss:.4f}')

        loss_record.append(avg_loss)

    f = open('loss.txt','w')
    f.write(str(loss_record))
    f.close()


if __name__ == '__main__':
    train()