'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-24 14:51:15
'''
import yaml

class Config():
    def __init__(self,path='config/config.yml') -> None:
        file = open(path,'r',encoding="utf-8")
        self.file_data = file.read()
        file.close()
        self.config = yaml.load(self.file_data,Loader=yaml.FullLoader)
    def __call__(self):
        return self.config


if __name__ == '__main__':
    cfg =  Config('/nvme0n1/lmj/disorder_selfsup/config/config2.yml')
    print(cfg.config['train_dir'][0])