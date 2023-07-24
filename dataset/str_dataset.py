'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-24 15:54:41
'''
import sys
sys.path.append('/server19/lmj/github/puztext')
import re
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataset.parseq.augment import rand_augment_transform
from dataset.parseq.dataset import LmdbDataset

class SceneTextData(Dataset):
    def __init__(self,path,img_size,max_label_length,
                 charset_path,augment:bool,rotation:int=0,
                 remove_whitespace: bool = True, 
                 normalize_unicode: bool = True,
                 min_image_dim: int = 0) -> None:
        super().__init__()
        
        with open(charset_path,encoding='utf-8') as file:
            self.tokens = []
            for k in file.readlines():
                k = re.sub('[\r\n\t]','',k)
                self.tokens.append(k)
            charset = ''.join(self.tokens)

        transforms = []
        if augment:
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img:img.rotate(rotation,expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        transform = T.Compose(transforms)
        
        self.dataset = LmdbDataset(root=path,
                                   charset=charset,
                                   max_label_len=max_label_length,
                                   min_image_dim=min_image_dim,
                                   remove_whitespace=remove_whitespace,
                                   normalize_unicode=normalize_unicode,
                                   unlabelled=False,
                                   transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

if __name__=='__main__':
    d = SceneTextData(path = '/server19/lmj/dataset/textimage/test/ArT',
                      img_size=[32,128],
                      max_label_length=25,
                      charset_path='/server19/lmj/github/puztext/dataset/charset/94.txt',
                      augment = True
                      )
    print(d[3000])