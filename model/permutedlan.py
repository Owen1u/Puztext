'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-24 21:02:26
'''
import math
import torch
import torch.nn as nn
from module.transformer import TransformerDecoderLayer,TransformerDecoder
from module.attn_decoder import AttentionRecognitionHead

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PermutedLanguage(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        d_model = config['lan_model']['d_model']
        nhead = config['lan_model']['nhead']
        d_inner = config['lan_model']['d_inner']
        dropout = config['lan_model']['dropout']
        activation = config['lan_model']['activation']
        num_layers = config['lan_model']['num_layers']
        self.use_self_attn = config['lan_model']['use_self_attn']

        self.d_model = d_model
        self.detach= config['lan_model']['detach']
        self.max_length = config['token_max_length']

        self.proj = nn.Linear(config['n_classes'],d_model,False) # input TBD
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model,nhead,dim_feedforward = d_inner,dropout=dropout,activation=activation)
        self.model = TransformerDecoder(decoder_layer, num_layers)
        self.cls = nn.Linear(d_model, config['n_classes'])

    def _get_padding_mask(self,length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    def _get_location_mask(self,sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        # mask[0][0]=mask[1][1]=0.
        return mask

    def forward(self,tokens,lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N,T,E)
        embed = embed.permute(1,0,2)    # (T,N,E)
        embed = self.token_encoder(embed)  # (T, N, E)
        
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        # location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                # memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)       # (T, N, E)
        # decoder feature
        output = output.permute(1, 0, 2)    # (N, T, E)
        # classification result
        logits = self.cls(output)           # (N, T, C)

        return output,logits



class PretrainLan(nn.Module):
    def __init__(self,config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = PermutedLanguage(config)
        # self.decoder = AttentionRecognitionHead(
        #               num_classes=config['n_classes'],
        #               in_planes=config['num_classes'],
        #               sDim=512,
        #               attDim=512,
        #               max_len_labels=config['token_max_length']) 
    
    def first_nonzero(self,x):
        non_zero_mask = x != 0
        mask_max_values, mask_max_indices = torch.max(non_zero_mask.int(), dim=-1)
        mask_max_indices[mask_max_values == 0] = -1
        return mask_max_indices
    
    
    def forward(self,logits,tgt,lengths,train):
        tokens = torch.softmax(logits, dim=-1)
        # print(tokens.size())
        # feat_lan,logits_lan = self.decoder(tokens, lengths)
        

        logits_lan, _ = self.decoder((tokens, tgt, lengths),train)
        
        return logits_lan