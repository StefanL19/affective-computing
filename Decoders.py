import torch
from torch import nn
from Attention import TanhAttention
from model_utils import BatchHolder
from conv_layers import Conv1dSamePad

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size:int,  
                       output_size:int = 1, 
                       use_attention:bool = True,
                       use_lexicons:bool = False,
                       use_emojis:bool = False,
                       lexicon_feat_length:int = None,
                       lexicon_feat_target_dims:int = None,
                       emoji_feat_target_dims:int = None,
                       dropout_prob:float = 0.3) :
    
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(p=dropout_prob)

        self.attention = TanhAttention(hidden_size=self.hidden_size)

        # If this is set to False, then the classification will be performed on the output of the last hidden cell
        self.use_attention = use_attention
        
        # If this is set to True, then the average emoji vector extracted from emoji2vec will be added as a features
        self.use_emojis = use_emojis

        # If this is set to True, then the features coming from the affective lexicons will be utilized
        self.use_lexicons = use_lexicons

        self.output_activation = torch.nn.Sigmoid()

        # self.conv_1 = Conv1dSamePad(in_channels=128, out_channels=1, filter_len=1)
        # self.conv_2 = Conv1dSamePad(in_channels=128, out_channels=1, filter_len=2)
        # self.conv_3 = Conv1dSamePad(in_channels=128, out_channels=1, filter_len=3)

        if self.use_lexicons and self.use_emojis:
            print('Using lexicons and emojis')
            self.linear_1 = nn.Linear(hidden_size+300+53, output_size).to(device)
        
        elif self.use_lexicons and not self.use_emojis:
            print('Using lexicons')
            self.linear_1 = nn.Linear(hidden_size+53, output_size).to(device)
        
        elif self.use_emojis and not self.use_lexicons:
            print('Using emojis')
            self.linear_1 = nn.Linear(hidden_size+300, output_size).to(device)

        else:
            print('Using basic RNN features')
            self.linear_1 = nn.Linear(hidden_size, output_size).to(device)

    def decode(self, predict) :
        # predict = self.dropout(predict)
        predict = self.linear_1(predict)
        predict = self.output_activation(predict)
        return predict
    
    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            mask = data.masks
            attn = self.attention(data.seq.to(device), output, mask)

            context = (attn.unsqueeze(-1) * output).sum(1)
            

            data.attn = attn
        else :
            context = data.last_hidden

        data.context = context

        context = self.dropout(context)

        if self.use_lexicons and self.use_emojis:
            emoji_features = self.emoji_layer(data.emoji_feats.to(device))
            emoji_features  = nn.functional.relu(emoji_features)

            lexicon_features = data.lexicon_feats.to(device)

            context = torch.cat((context, lexicon_features, emoji_features), dim=1)

        elif self.use_emojis and not self.use_lexicons:
            emoji_features = data.emoji_feats.to(device)

            context = torch.cat((context, emoji_features), dim=1)
        
        elif self.use_lexicons and not self.use_emojis:
            lexicon_features = data.lexicon_feats.to(device)

            context = torch.cat((context, lexicon_features), dim=1)
        
        context = torch.nn.ReLU()(context)
        
        predict = self.decode(context)
        data.predict = predict
