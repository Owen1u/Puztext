'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: https://github.com/FangShancheng/ABINet/blob/main/modules/attention.py
LastEditTime: 2023-07-24 20:57:16
'''
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        # enc_output = enc_output.flatten(1)
        reading_order = torch.arange(self.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256
        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output, attn.reshape(*attn.shape[:2], -1)

