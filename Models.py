import torch
import torch.nn as nn
from sklearn.utils import shuffle
from tqdm import tqdm
import time
import numpy as np
from Encoders import EncoderRNN
from Decoders import AttnDecoder
from model_utils import BatchHolder, get_sorting_index_with_noise_from_lengths

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model() :
    def __init__(self, vocab_size, embed_size, hidden_size, bsize, bidirectional=True, 
    pre_embed=None, use_attention=True, use_lexicons=False, lex_feat_length=None, use_emojis=False,
    lexicon_feat_target_dims=None, emoji_feat_target_dims=None, dropout_prob=0.5) :
        
        self.encoder = EncoderRNN(vocab_size, embed_size, hidden_size, bidirectional, pre_embed=pre_embed)
        self.encoder_params = list(self.encoder.parameters())

        self.decoder = AttnDecoder(2*hidden_size, 1, use_attention=use_attention, use_lexicons=use_lexicons,
         use_emojis=use_emojis, lexicon_feat_length=lex_feat_length, lexicon_feat_target_dims=lexicon_feat_target_dims,
         emoji_feat_target_dims=emoji_feat_target_dims, dropout_prob=dropout_prob)
         
        self.decoder_params = list(self.decoder.parameters())

        self.bsize = bsize

        self.weight_decay = 1e-5

        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr= 0.0001, weight_decay=self.weight_decay, amsgrad=True)

        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr= 0.0001, weight_decay=self.weight_decay, amsgrad=True)

        self.criterion = nn.MSELoss(reduction="mean").to(device)

    def train(self, data_in, target_in, data_lexicon, data_emoji, train=True):
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)

        data = [data_in[i] for i in sorting_idx]

        target = [target_in[i] for i in sorting_idx]

        lexicon_feat = [data_lexicon[i] for i in sorting_idx]

        emoji_feat = [data_emoji[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()

        bsize = self.bsize

        N = len(data)

        loss_total = 0

        # Deterine the number of batches that will be used to traing the model
        batches = list(range(0, N, bsize))

        # Shuffle them, so they are not the same all the time
        batches = shuffle(batches)

        for n in tqdm(batches):
            batch_doc = data[n:n+bsize]
            batch_lexicon = lexicon_feat[n:n+bsize]
            batch_emoji = emoji_feat[n:n+bsize]

            batch_data = BatchHolder(batch_doc, batch_lexicon, batch_emoji)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1 : #(B, )
                batch_target = batch_target.unsqueeze(-1) #(B, 1)
            
            loss = self.criterion(batch_data.predict, batch_target)

            if train :
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()

                loss.backward()

                # Take a training step for the optimizer
                self.encoder_optim.step()
                self.decoder_optim.step()

            loss_total += float(loss.data.cpu().item())

        return loss_total*bsize/N, loss_total

    def evaluate(self, data, data_lexicon, data_emoji) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        
        N = len(data)

        outputs = []

        for n in tqdm(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_lexicon = data_lexicon[n:n+bsize]
            batch_emoji = data_emoji[n:n+bsize]

            batch_data = BatchHolder(batch_doc, batch_lexicon, batch_emoji)

            self.encoder(batch_data)
            self.decoder(batch_data)

            predict = batch_data.predict.cpu().data.numpy()
            
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]

        return outputs
