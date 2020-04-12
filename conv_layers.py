import torch
from torch import nn
import numpy as np
import torch.functional as F

class Conv1dSamePad(nn.Module):
    def __init__(self, in_channels, out_channels, filter_len, stride=1, **kwargs):
        super(Conv1dSamePad, self).__init__()
        self.filter_len = filter_len
        self.conv = nn.Conv1d(in_channels, out_channels, filter_len, padding=(self.filter_len // 2), stride=stride,
                              **kwargs)
        nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.constant_(self.conv.bias, 1 / out_channels)

    def forward(self, x):
        if self.filter_len % 2 == 1:
            return self.conv(x)
        else:
            return self.conv(x)[:, :, :-1]