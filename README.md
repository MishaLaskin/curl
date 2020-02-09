# SAC+CPC implementaiton in PyTorch

#

## Instructions
To train an SAC+CPC agent on the `cheetah run` task from image-based observations  run:
```
CUDA_VISIBLE_DEVICES=4 python train_cpc.py \
    --dmc2gym \
    --domain_name reacher \
    --task_name easy \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --frame_stack 1 \
    --save_tb --save_video --num_train_steps 1000000 \
    --work_dir ./tmp/dmc/reacher_easy_cpc \
    --agent sac_cpc \
    --seed 2 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000

CUDA_VISIBLE_DEVICES=7 python train.py \
    --domain_name walker \
    --task_name walk --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_tb --pre_transform_image_size 84 --image_size 84 \
    --work_dir ./tmp/icml/vanilla_sac/ML0107walker_vanilla_sac \
    --agent sac_ae --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```
Try - reducing log std actor max from 2->1 or increase 2->3 


This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).
