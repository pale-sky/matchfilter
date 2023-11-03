import torch
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,out,label):
        label = label[:,:,0,:].contiguous().view(-1)
        out = out.squeeze().transpose(-1,-2)
        out = out.contiguous().view(-1, out.shape[-1])
        return self.loss_fn(out,label.long())
