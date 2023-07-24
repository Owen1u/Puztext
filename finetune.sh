###
 # @Descripttion: 
 # @version: 
 # @Contributor: Minjun Lu
 # @Source: Original
 # @LastEditTime: 2023-07-25 00:06:18
### 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29551 --use_env finetune.py --cfg /server19/lmj/github/puztext/config/finetune.yaml