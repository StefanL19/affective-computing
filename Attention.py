import numpy as np
import torch
from torch import nn
from allennlp.common import Registrable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def masked_softmax(attn_odds, masks) :
    attn_odds.masked_fill_(masks, -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class TanhAttention(nn.Module) :
    def __init__(self, hidden_size) :
        print("Using Softmax Attention")
        super().__init__()
        print("The hidden size is: ", hidden_size)
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False).to(device)
        
    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = masked_softmax(attn2, masks)
        
        return attn
