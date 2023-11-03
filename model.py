import torch
import torch.nn as nn
from RadarModel.functionLib import db
import torch.fft

class Base(nn.Module):
    def __init__(self,channel_num = 1,FeatureBlock = None):
        super().__init__()
        self.channel_num = channel_num
        
        if FeatureBlock is None:
            self.Block = BaseBlock(self.channel_num)
        else:
            self.Block = FeatureBlock


    def forward(self,x,ref,y = None):
        matchfilterResult = self.matchfilter(x,ref)
        if self.Block.ModelType == 'Transformer':
            if self.Block.option:
                x,y = self.emb(x,y)
                f = self.Block(x,y).transpose(1,2).unsqueeze(1)
            else:
                f = self.Block(x,y).unsqueeze(1)
        else:
            f = self.Block(x,ref)
        result = matchfilterResult-f
        return result
        

    def matchfilter_sim(self,x,refsig):
        ref = refsig[:,:,0,:]+1j*refsig[:,:,1,:]
        ref = torch.conj(ref)
        data = x[:,:,0,:]+1j*x[:,:,1,:]
        result = torch.zeros_like(data)
        data = torch.cat((data,(1+1j)*torch.zeros([data.shape[0],data.shape[1],ref.shape[-1]]).to(data.device)),dim = 2)
        
        for i in range(result.shape[2]):
            result[:,:,i] = (ref*data[:,:,i:i+ref.shape[2]]).sum()
        
        profile = torch.zeros([result.shape[0],result.shape[1],2,result.shape[2]]).to(data.device)
        profile[:,:,0,:] = result.real
        profile[:,:,1,:] = result.imag
        return profile


    def matchfilter(self,x,refsig):
        if 3 == x.dim():
            if 2 == x.shape[1]:
                x = x.unsqueeze(1)
                refsig = refsig.unsqueeze(1)
            else:
                x = x.transpose(1,2).unsqueeze(1)
                refsig = refsig.transpose(1,2).unsqueeze(1)

        ref = refsig[:,:,0,:]+1j*refsig[:,:,1,:]
        ref = torch.conj(ref).flip(dims=[-1])
        data = x[:,:,0,:]+1j*x[:,:,1,:]
        profile = torch.fft.ifft(torch.fft.fft(ref,dim = -1)*torch.fft.fft(data,dim = -1),dim = -1)
        result = torch.zeros([profile.shape[0],profile.shape[1],2,profile.shape[2]]).to(data.device)
        result[:,:,0,:] = profile.real
        result[:,:,1,:] = profile.imag
        return result

    def emb(self,*args):
        result = []
        for i in args:
            result.append(torch.matmul(args[0],self.EmbKernel().to(args[0].device)))
        return tuple(result)

    def EmbKernel(self):
        kernel = nn.Parameter(torch.randn(2,8))
        return kernel


class BaseBlock(nn.Module):
    def __init__(self,channel_num = 1):
        super().__init__()
        self.channel_num = channel_num
        self.encoderbloc,self.decoderblock = self.getBlock()
        self.ModelType = 'Base'

    def getBlock(self):
        encoderblock = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_num,out_channels=10,kernel_size=(2,16),stride=1,padding=1),
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(2,8),stride=1,padding=1),
            nn.Conv2d(in_channels=20,out_channels=40,kernel_size=(2,4),stride=1,padding=1),
            nn.Conv2d(in_channels=40,out_channels=80,kernel_size=(2,4),stride=1,padding=1),
            nn.ReLU(),
        )
        decoderblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels=80,out_channels=40,kernel_size=(2,4),stride=1,padding=1),
            nn.ConvTranspose2d(in_channels=40,out_channels=20,kernel_size=(2,4),stride=1,padding=1),
            nn.ConvTranspose2d(in_channels=20,out_channels=10,kernel_size=(2,8),stride=1,padding=1),
            nn.ConvTranspose2d(in_channels=10,out_channels=self.channel_num,kernel_size=(2,16),stride=1,padding=1),
            nn.ReLU(),
        )
        return encoderblock,decoderblock

    def forward(self,x,y=None):
        return self.decoderblock(self.encoderbloc(x))