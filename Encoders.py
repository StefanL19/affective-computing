import torch
import torch.nn as nn
from allennlp.common import Registrable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
        
class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, bidirectional, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0).to(device)
            # freeze_layer(self.embedding)
            
        else :
            print("Not setting Embedding")
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional).to(device)

        self.output_size = self.hidden_size * 2

    def forward(self, data) :
        seq = data.seq
        seq = seq.to(device)
        lengths = data.lengths
        
        embedding = self.embedding(seq) #(B, L, E)

        data.embeddings = embedding

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False).to(device)

        output, (h, c) = self.rnn(packseq)

        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)

        # if isTrue(data, 'keep_grads') :
        #     data.embedding = embedding
        #     data.embedding.retain_grad()
        #     data.hidden.retain_grad()
