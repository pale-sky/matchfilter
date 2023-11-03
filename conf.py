import torch

BATCH_SIZE = 3
LEARN_RATE = 1e-4
EPOCHS = 500

max_len = 256
d_model = 8
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

src_pad_idx = None
trg_pad_idx = None
trg_sos_idx = None

enc_voc_size = None
dec_voc_size = 2