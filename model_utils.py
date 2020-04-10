def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)

import numpy as np
import torch

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchHolder() : 
    def __init__(self, data, data_lexicon, data_emoji) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []

        for _, d in enumerate(data) :
            rem = maxlen - len(d)
            expanded.append(d + [0]*rem)
            lengths.append(len(d))
            masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))

        self.lengths = torch.LongTensor(np.array(lengths)).to(device)
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
        self.lexicon_feats = torch.FloatTensor(np.array(data_lexicon, dtype='float64')).to(device)
        self.emoji_feats = torch.FloatTensor(np.array(data_emoji, dtype='float64')).to(device)

        self.masks = torch.BoolTensor(np.array(masks)).to(device)

        self.hidden = None
        self.predict = None

        self.inv_masks = ~self.masks

    def generate_frozen_uniform_attn(self):
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
        attn += inv_l[:, None]
        attn = torch.Tensor(attn).to(device)
        attn.masked_fill_(self.masks, 0) 
        return attn