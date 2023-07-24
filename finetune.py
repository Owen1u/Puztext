import sys
import argparse
import builtins
import math
import os
import re
import random
import shutil
import time
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import cv2 as cv
cv.setNumThreads(1)

from config.cfg import Config
from dataset.str_dataset import SceneTextData
# from dataset.tokenizer import Tokenizer
from dataset.parseq.utils import Tokenizer

# from dataset.tokenizer import Tokenizer
from utils.logger import Log
from utils.meter import AverageMeter

from model.puztext import Puztext
from model.weights_init import init_weights

cudnn.benchmark = True
cudnn.deterministic = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
local_rank = int(os.getenv('LOCAL_RANK',-1))

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, help='path to config')
args = parser.parse_args()
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(local_rank)
'''
Config & Logging
'''
cfg = Config(args.cfg)
config = cfg()
if local_rank!=0:
    def print_pass(*args):
        pass
    builtins.print = print_pass
else:
    if not os.path.isdir(config['model_save_dir']):
        os.mkdir(config['model_save_dir'])
    if not os.path.isdir(os.path.join(config['model_save_dir'],config['model_name'])):
        os.mkdir(os.path.join(config['model_save_dir'],config['model_name']))
    logger = Log(os.path.join(config['model_save_dir'],config['model_name'],'logging.log'),['epoch','learning rate','train loss','val loss','accuracy','best_acc'],dec=6)
    logger.head(config)

if config['seed'] is not None:
    set_seed(config['seed'])

'''
Model
'''
# model = mocov3.builder.MoCo_ViT(
#             partial(mocov3.vits.__dict__[config['arch']], stop_grad_conv1=config['stop_grad_conv1']),
#             config['moco_dim'], config['moco_mlp_dim'], config['moco_t'])
model = Puztext(config,config['moco_dim'], config['moco_mlp_dim'])
# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
device = torch.device('cuda',local_rank)
model = model.to(device)
model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
model.apply(init_weights)
print('初始化完成')
if config['model_pretrained']is not None:
    for pretrain_model in config['model_pretrained']:
        new_state_dict = OrderedDict()
        state_dict = torch.load(pretrain_model,map_location = 'cpu')
        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                print(k)
                new_state_dict[k] = v
            elif k[7:] in model.state_dict().keys():
                new_state_dict[k[7:]] = v
            else:
                continue

        model.load_state_dict(new_state_dict,strict=False)
        # model.load_state_dict(torch.load(config['pretrained']))
        print('loading pre-trained model from {0}'.format(pretrain_model))

'''
Optimizer
'''
if config['optimizer'] == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), config['lr_f'],
                            weight_decay=config['weight_decay'])
else:
    assert False,'You must have one optimizer at least.'
    

'''
Dataset
'''
with open('/server19/lmj/github/puztext/dataset/charset/36.txt',encoding='utf-8') as file:
    tokens = []
    for k in file.readlines():
        k = re.sub('[\r\n\t]','',k)
        tokens.append(k)
    charset_train = ''.join(tokens)
train_data = []
for root, dirs, files in os.walk('/server19/lmj/dataset/textimage/train/real'):
    for dir in dirs:
        path = os.path.join(root, dir)
        if os.path.isfile(os.path.join(path,'data.mdb')):
            data = SceneTextData(path=path,
                                 img_size=config['img_size'][1:],
                                 max_label_length=config['token_max_length'],
                                 charset=charset_train,
                                 augment=True) 
            train_data.append(data)
train_data = ConcatDataset(train_data)

with open('/server19/lmj/github/puztext/dataset/charset/36.txt',encoding='utf-8') as file:
    tokens = []
    for k in file.readlines():
        k = re.sub('[\r\n\t]','',k)
        tokens.append(k)
    charset_test = ''.join(tokens)
test_data = []
for root, dirs, files in os.walk('/server19/lmj/dataset/textimage/test'):
    for dir in dirs:
        path = os.path.join(root, dir)
        if os.path.isfile(os.path.join(path,'data.mdb')):
            data = SceneTextData(path=path,
                                 img_size=config['img_size'][1:],
                                 max_label_length=config['token_max_length'],
                                 charset=charset_test,
                                 augment=False) 
            test_data.append(data)
test_data = ConcatDataset(test_data)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=config['batchsize'], shuffle=(train_sampler is None),
    num_workers=config['num_worker'], pin_memory=True, sampler=train_sampler, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=config['batchsize'], shuffle=(test_sampler is None),
    num_workers=config['num_worker'], pin_memory=True, sampler=test_sampler,)

'''
Loss
'''
if 'ce' in config['loss']:
    lossCE = torch.nn.CrossEntropyLoss()
if 'mse' in config['loss']:
    lossMSE = nn.MSELoss(reduce=True,size_average=True)

learning_rates = AverageMeter('LR', ':.4e')
losses = AverageMeter('Loss', ':.4e')
best_acc = 0

def val(model,lossCE):
    tokenizer = Tokenizer(charset_test)
    val_loss = AverageMeter('Loss', ':.4e')
    n_sample=0
    global best_acc
    # val_step=0
    n_correct_pic=0
    n_correct_patch=0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader),desc='val',ncols=80) as valbar:
            for batch_idx,data in enumerate(test_loader):
                img,label= data
                idx = tokenizer.encode(label)[:,1:]
                idx = idx.cuda(non_blocking=False)
                length = torch.Tensor([torch.count_nonzero(i) for i in idx]).int()
                batchsize = img.size(0)
                n_sample += batchsize
                
                preds,preds_vision = model(img,None,None,train = False,device = device)
                preds = preds.log_softmax(2).permute(0,2,1)
                
                cost = lossCE(preds,idx)
                val_loss.update(cost.item())

                _,preds = preds.max(1)
                preds = tokenizer.decode(preds.data)
                label_dec = tokenizer.decode(idx.data)
                for pred,gt in zip(preds,label_dec):
                    if pred==gt:
                        n_correct_pic+=1
            
                valbar.update(1)
                dist.barrier()
        accuracy = n_correct_pic/float(n_sample)
        patch_acc = n_correct_patch/float(n_sample*config['img_size'][1]*config['img_size'][2]/config['patch_size'][0]/config['patch_size'][1])
        if accuracy>best_acc:
            best_acc=accuracy
            torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'best.pth'))
        print('eval Loss:{0},accuracy:{1},patch_acc:{2}'.format(val_loss,accuracy,patch_acc))
        return val_loss,accuracy

print('running val() for debug...')
val_loss,acc=val(model,lossCE)
model.train()
print('start training...')

def adjust_learning_rate(optimizer, epoch, config):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = config['lr_f'] * epoch / config['warmup_epochs']
    else:
        lr = config['lr_f'] * 0.5 * (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

tokenizer = Tokenizer(charset_train)
for epoch in range(config['epochs']):
    with tqdm(total=len(train_loader),desc='epoch:{0}/{1}'.format(epoch+1,config['epochs']),ncols=80) as trainbar:
        for i,data in enumerate(train_loader):
            # 原图；增强；乱序；文本标签；乱序标签
            img,label= data
            idx = tokenizer.encode(label)[:,1:]
            idx = idx.cuda(non_blocking=False)
            length = torch.Tensor([torch.count_nonzero(i) for i in idx]).int()
            batchsize = img.size(0)
            
            iters_per_epoch = len(train_loader)
            lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, config)
            learning_rates.update(lr)
            
            preds,preds_vision = model(img,idx,length,train=True,device = device)
            preds = preds.log_softmax(2).permute(0,2,1)
            if preds_vision is not None:
                preds_vision = preds_vision.log_softmax(2).permute(0,2,1)
                loss = 0.5*lossCE(preds,idx)+0.5*lossCE(preds_vision,idx)
            else:
                loss = lossCE(preds,idx)
            losses.update(loss.item(), batchsize)
            # lossce = lossCE(z2,idx)
            optimizer.zero_grad()
            # lossce.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            trainbar.update(1)
            dist.barrier()

    if config['eval_epoch'] is not None and epoch%config['eval_epoch']==0:
        print('[epoch:{0}/{1}] Loss:{2}'.format(epoch+1,config['epochs'],losses))
        val_loss,acc=val(model,lossCE)
        model.train()
        if local_rank==0:
            logger.print([[str(epoch+1),learning_rates,losses,val_loss,str(acc),str(best_acc)]])
    if config['save_epoch'] is not None and epoch%config['save_epoch']==0:
        torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'lastest.pth'))
    losses.reset()
    dist.barrier()



