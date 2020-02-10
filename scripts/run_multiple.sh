#!/bin/bash

gpuList=(1 2 3 4 5 6 7)
(
for i in ${gpuList[@]}; do 
   CUDA_VISIBLE_DEVICES=$i python train.py \
    --domain_name reacher \
    --task_name easy \
    --encoder_type pixel \
    --action_repeat 4 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp/curl/reacher_easy \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 128 --num_train_steps 1000000 &
done
)
echo "Launched CURL script"