import torch
from torch import nn
from Attention import TanhAttention
from model_utils import BatchHolder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size:int,  
                       output_size:int = 1, 
                       use_attention:bool = True,
                       use_lexicons:bool = False,
                       use_emojis:bool = False,
                       lexicon_feat_length:int = None,
                       lexicon_feat_target_dims:int = None) :
    
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
    
        self.dropout = nn.Dropout(p=0.3)

        self.attention = TanhAttention(hidden_size=self.hidden_size)

        # If this is set to False, then the classification will be performed on the output of the last hidden cell
        self.use_attention = use_attention
        
        # If this is set to True, then the average emoji vector extracted from emoji2vec will be added as a features
        self.use_emojis = use_emojis

        # If this is set to True, then the features coming from the affective lexicons will be utilized
        self.use_lexicons = use_lexicons

        if self.use_lexicons:
            self.linear_1 = nn.Linear(hidden_size+lexicon_feat_target_dims, output_size)
            self.lexicon_layer = nn.Linear(lexicon_feat_length, lexicon_feat_target_dims)
        else:
            self.linear_1 = nn.Linear(hidden_size, output_size)

    def decode(self, predict) :
        # predict = self.dropout(predict)
        predict = self.linear_1(predict)
        return predict
    
    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            mask = data.masks
            attn = self.attention(data.seq, output, mask)

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden

        context = self.dropout(context)

        if self.use_lexicons:
            lexicon_features = self.lexicon_layer(data.lexicon_feats)
            lexicon_features = nn.functional.relu(lexicon_features)
            context = torch.cat((context, lexicon_features), dim=1)
        
        predict = self.decode(context)
        data.predict = predict
