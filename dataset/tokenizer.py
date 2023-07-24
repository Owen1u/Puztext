import json
import re
from traceback import print_tb
import torch
from collections.abc import Iterable
import sys

class Tokenizer():
    PAD = '[PAD]'
    GO = '[GO]'
    UNK = '[UNK]'
    def __init__(self,config) -> None:
        self.dict_path = config['dict_path']
        self.token_max_length = config['token_max_length']+1
        if self.dict_path.endswith('.txt'):
            with open(self.dict_path,encoding='utf-8') as file:
                self.tokens = ["[PAD]","[GO]","[UNK]"]
                for k in file.readlines():
                    k = re.sub('[\r\n\t]','',k)
                    self.tokens.append(k)
            
                self.token2idx = {t:i for i,t in enumerate(self.tokens)}
                self.idx2token = {i:t for i,t in enumerate(self.tokens)}
                self.length = len(self.tokens)
        if self.length<=40:
            self.sensitive = False
        else:
            self.sensitive = True
                
    def __len__(self):
        return self.length
    
    @property
    def token(self):
        return self.tokens
    
    def prob_smooth_label(self, one_hot, smooth_factor=0.2):
        one_hot = one_hot.float()
        delta = torch.rand([]) * smooth_factor
        num_classes = len(one_hot)
        noise = torch.rand(num_classes)
        noise = noise / noise.sum() * delta
        one_hot = one_hot * (1 - delta) + noise
        return one_hot
    
    def onehot(self,text):
        label,length = self.encode(text)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        onehot = torch.zeros(label.size() + torch.Size([self.length]))
        onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)
        onehot = torch.stack([self.prob_smooth_label(l) for l in onehot])
        # onehot = self.prob_smooth_label(onehot)
        return onehot,length

        
    
    def encode(self,text):
        if not self.sensitive:
            text = text.lower()
        length = min(len(text)+1,self.token_max_length)
        text = text[:length-1] 
        ids= [0]*self.token_max_length
        ids[0] = self.token2idx['[GO]']
        for i,c in enumerate(text):
            ids[i+1] = self.token2idx[c] if c in self.token2idx else self.token2idx['[UNK]']
        # print(ids)
        return ids,length
        
    def decode(self,batch):
        res=[]
        for word in batch:
            r=[]
            for c in word:
                c = int(c)
                if self.idx2token[c]=='[PAD]' or self.idx2token[c]=='[GO]' or self.idx2token[c]=='[UNK]':
                    continue
                else:
                    r.append(self.idx2token[c]) 
            r = ''.join(r)
            res.append(r)
        return res
    
    # def encode(self,texts):
    #     # texts = [re.sub('[^0-9a-zA-Z]+', '', t) for t in texts]
    #     length = [min(len(s),self.token_max_length) for s in texts]
    #     batch_text = torch.LongTensor(len(texts), self.token_max_length).fill_(0)
    #     for i, t in enumerate(texts):
    #         if self.eng and self.sensitive:
    #             text=list(t)
    #         else:
    #             text = list(t.lower())
    #         # # temp
    #         # text = text if len(text)<=self.token_max_length else text[:self.token_max_length-1]
    #         # print(self.token2idx.keys())
    #         text = [self.token2idx[char] if char in self.token2idx else self.token2idx['[UNK]'] for char in text]
    #         batch_text[i][:length[i]] = torch.LongTensor(text[:length[i]])
    #     return (batch_text, torch.IntTensor(length))

    # def attnencode(self,texts):
    #     length = [min(len(s)+2,self.token_max_length) for s in texts]
    #     batch_text = torch.LongTensor(len(texts), self.token_max_length).fill_(0)
    #     for i, t in enumerate(texts):
    #         if self.eng and self.sensitive:
    #             text=list(t)
    #         else:
    #             text = list(t.lower())
    #         text_=[self.token2idx['[GO]']]
    #         for char in text:
    #             if char in self.token2idx:
    #                 text_.append(self.token2idx[char])
    #             else:
    #                 text_.append(self.token2idx['[UNK]'])
    #         try:
    #             if len(text)<self.token_max_length-1:
    #                 text_.append(self.token2idx['[EOS]'])
    #             else:
    #                 text_[self.token_max_length-1]=self.token2idx['[EOS]']
    #         except:
    #             print(text,text_)
    #             sys.exit()
            
    #         batch_text[i][:length[i]] = torch.LongTensor(text_[:length[i]])
        
    #     return (batch_text, torch.IntTensor(length))    
    
    # def attndecode(self,texts_encoded,lengths):
    #     """ convert text-index into text-label. """
    #     texts = []
    #     for index, l in enumerate(lengths):
    #         t = texts_encoded[index, :]

    #         char_list = []
    #         for i in range(l):
    #             if t[i]==self.token2idx['[UNK]'] or t[i]==self.token2idx['[GO]'] or t[i]==self.token2idx['[EOS]']:
    #                 continue
    #             if self.is_ctc and t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
    #                 char_list.append(self.idx2token[int(t[i])])
    #             elif not self.is_ctc and t[i] != 0 :
    #                 char_list.append(self.idx2token[int(t[i])])

    #         text = ''.join(char_list)
    #         if self.eng and self.sensitive==False:
    #             text = text.lower()
    #         texts.append(text)
    #     return texts

    # def decode(self,texts_encoded,lengths):
    #     """ convert text-index into text-label. """
    #     texts = []
    #     for index, l in enumerate(lengths):
    #         t = texts_encoded[index, :]

    #         char_list = []
    #         for i in range(l):
    #             if self.is_ctc and t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]!=self.token2idx['[UNK]']:  # removing repeated characters and blank.
    #                 char_list.append(self.idx2token[int(t[i])])
    #             elif not self.is_ctc and t[i] != 0 :
    #                 char_list.append(self.idx2token[int(t[i])])

    #         text = ''.join(char_list)
    #         if self.eng and self.sensitive==False:
    #             text = text.lower()
    #         texts.append(text)
    #     return texts
    
    # def encodeEOS(self,texts):
    #     length = [min(len(s)+1,self.token_max_length) for s in texts]
    #     batch_text = torch.LongTensor(len(texts), self.token_max_length).fill_(0)
    #     for i, t in enumerate(texts):
    #         if self.eng and self.sensitive:
    #             text=list(t)
    #         else:
    #             text = list(t.lower())
    #         text_=[]
    #         for char in text:
    #             if char in self.token2idx:
    #                 text_.append(self.token2idx[char])
    #             else:
    #                 text_.append(self.token2idx['[UNK]'])
    #                 pass
    #         if len(text)<self.token_max_length:
    #             text_.append(self.token2idx['[EOS]'])
    #         else:
    #             text_[self.token_max_length-1]=self.token2idx['[EOS]']
            
    #         batch_text[i][:length[i]] = torch.LongTensor(text_[:length[i]])
        
    #     return (batch_text, torch.IntTensor(length))    
    
    # def decodeEOS(self,texts_encoded,lengths):
    #     """ convert text-index into text-label. """
    #     texts = []
    #     for index, l in enumerate(lengths):
    #         t = texts_encoded[index, :]

    #         char_list = []
    #         for i in range(l):
    #             if t[i]==self.token2idx['[UNK]']:
    #                 continue 
    #             if  t[i]==self.token2idx['[EOS]']:
    #                 continue
    #             if self.idx2token[int(t[i])].lower() not in 'abcdefghijklmnopqrstuvwxyz0123456789':
    #                 continue
    #             if self.is_ctc and t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
    #                 char_list.append(self.idx2token[int(t[i])])
    #             elif not self.is_ctc and t[i] != 0 :
    #                 char_list.append(self.idx2token[int(t[i])])

    #         text = ''.join(char_list)
    #         text = text.lower()
    #         texts.append(text)
    #     return texts
    
    def create_mask(self,fine_path):
        self.mask = torch.ones([self.length])
        with open(fine_path,encoding='utf-8') as file:
                data = []
                for k in file.readlines():
                    k = re.sub('[\r\n\t]','',k)
                    self.mask[self.token2idx[k]]=0
                    data.append(k)

        self.mask[0]=0.
        self.mask = torch.repeat_interleave(self.mask.unsqueeze(0),repeats=self.token_max_length,dim=0)
        self.mask = self.mask*-100

    def get_mask(self,batchsize):
        mask = torch.repeat_interleave(self.mask.unsqueeze(0),repeats=batchsize,dim=0)
        return mask

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, config):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.dict_path = config['dict_path']
        self.token_max_length = config['token_max_length']
        if self.dict_path.endswith('.txt'):
            with open(self.dict_path,encoding='utf-8') as file:
                data = file.read()
                data=re.sub('[\r\n\t]','',data)

        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(data)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

if __name__ == '__main__':
    config={'dict_path':'/server19/lmj/puztext/dataset/charset/36_lowercase.txt','token_max_length': 25,'loss':['ce'],'sensitive':False,'model_name':'eng'}
    tokenizer = Tokenizer(config)
    print(tokenizer.length)
    print(tokenizer.encode('goodgoodgoodgoodgoodgoodgood'))
    onehot = tokenizer.onehot('goodgoodgoodgoodgoodgood')
    print(onehot)

    # print(t.size(),l.size())
    # print(t,l)
    # print(tokenizer.decodeEOS(t,l))