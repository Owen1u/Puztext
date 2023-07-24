'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-24 23:50:08
'''
import torch
import torch.nn as nn
# from module.vit_torch import ViT
from module.attn_USTC import Attention
# from module.bcn import BCNLanguage
from module.alignment import Alignment
from module.attn_decoder import AttentionRecognitionHead
from module.focalnet import FocalNet
from model.permutedlan import PermutedLanguage
from module.parseq import PARSeq


class Puztext(nn.Module):
    def __init__(self, config, dim=256, mlp_dim=4096):
        super().__init__()
        
        self.null_label = config['null_label']
        self.iter_size = config['lan_model']['iter_size']
        self.token_max_length = config['token_max_length']
        
        # self.base_encoder = ViT(
        #         image_size=config['img_size'][-1],
        #         patch_size=config['patch_size'][-1],
        #         num_classes=config['num_classes'],
        #         dim=dim,
        #         depth=config['depth'],
        #         heads=config['heads'],
        #         mlp_dim=mlp_dim
        #     )
        self.base_encoder = FocalNet(
                img_size=config['img_size'][1:], 
                patch_size=config['patch_size'], 
                in_chans=config['img_size'][0], 
                num_classes=config['num_classes'],
                embed_dim = 128, 
                depths=[2,2,18,2], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                patch_norm=True,
                use_checkpoint=False,                 
                focal_levels=[2, 2, 2, 2], 
                focal_windows=[3, 3, 3, 3], 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_postln=False, 
                use_postln_in_modulation=False, 
                normalize_modulator=False, 
        )
        self.patchnum = [config['img_size'][1]//config['patch_size'][0],config['img_size'][2]//config['patch_size'][1]]
        self.attn = Attention(config['num_classes'],config['token_max_length'],self.patchnum[0]*self.patchnum[1])
        # self.decoder = BCNLanguage(config)
        # self.decoder = PermutedLanguage(config)
        self.decoder = AttentionRecognitionHead(
                      num_classes=config['n_classes'],
                      in_planes=config['num_classes'],
                      sDim=512,
                      attDim=512,
                      max_len_labels=config['token_max_length']+1) 
        # self.decoder = PARSeq(
        #     token_max_length=config['token_max_length'],
        #     token_class=config['n_classes'],
        #     embed_dim = config['num_classes'],
        #     dec_num_heads=8,
        #     dec_mlp_ratio=4,
        #     dec_depth=1,
        #     autoregression=True,
        #     refine_iters=1,
        #     dropout=0.1,
        # )
        self.token_cls = nn.Linear(config['num_classes'], config['n_classes'])
        self.alignment = Alignment(config['lan_model']['d_model'],config['n_classes'],config['null_label'])
        
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
    
    def forward(self,x,tgt,tgt_lens,train,device):
        x = self.base_encoder(x)   # B,196+1,H

        # feat_vision, attn_scores = self.attn(x)         # b,25,512
        # logits_vision = self.token_cls(feat_vision)  # [b,w,c]

        # lengths_vision = self._get_length(logits_vision)
        # lengths_vision.clamp_(2, self.token_max_length) 
        # feat_lan,logits_lan = self.decoder(logits_vision, lengths_vision)
        # logits_align,lengths_align = self.alignment(feat_lan, feat_vision)
        
        logits_align, _ = self.decoder((x, tgt, tgt_lens),train)

        # logits_align = self.decoder(x,None,device)
        return logits_align,None
        