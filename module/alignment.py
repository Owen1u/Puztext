'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: https://github.com/FangShancheng/ABINet/blob/789743c56eacbf114c2d63dfaeada533098dd19a/modules/model_alignment.py#L8
LastEditTime: 2023-07-24 20:56:39
'''
import torch
import torch.nn as nn 

class Alignment(nn.Module):
    def __init__(self,d_model,n_class,null_label):
        super().__init__()

        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model,n_class)
        self.null_label = null_label

    def first_nonzero(self,x):
        non_zero_mask = x != 0
        mask_max_values, mask_max_indices = torch.max(non_zero_mask.int(), dim=-1)
        mask_max_indices[mask_max_values == 0] = -1
        return mask_max_indices

    def _get_length(self, logit):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.null_label)
        out = self.first_nonzero(out.int()) + 1
        return out

    def forward(self,l_feature,v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature,v_feature),dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)
        lengths = self._get_length(logits)
        
        return logits,lengths