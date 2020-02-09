#!/bin/bash
# 256 
# 2e4 lr 2x
# 5e4 lr 2x
# 1e3 lr 2x

CUDA_VISIBLE_DEVICES=1 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr2e4_a \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --encoder_lr 2e-4 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=2 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr2e4_b \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed -1 --encoder_lr 2e-4 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=3 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr5e4_a \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --encoder_lr 5e-4 --critic_lr 5e-4 --actor_lr 5e-4 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=4 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr5e4_b \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed -1 --encoder_lr 5e-4 --critic_lr 5e-4 --actor_lr 5e-4 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=5 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr1e3_a \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --encoder_lr 1e-3 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=6 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb3cheetah/curl_cheetah_b256_84_lr1e3_b \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed -1 --encoder_lr 1e-3 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 