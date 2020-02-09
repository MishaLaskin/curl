
# curl cheetah, crop 76 > 64, grayscale + random crop, deep stack
# batch size = 256 instead of 128. maybe 256 makes 64x64 work. And, try using 512 with 64x64.
# then try adam LR (smaller) - 3e-4... You can try 2e-4 and 5e-4. 
# try using the stochastic policy for eval. (you can do later.. for now the important thing is to run ablations.)
# try bigger frame stack, maybe 8.
#     parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
# run 1: batch 256, first try with 84 
# run 2: batch 512, first try with 84, then try 64, then try their encoder
# run 3: batch 256, first try with 84 
# run 4: batch 512, first try with 84, then try 64, then try their encoder
# run 5: 2e-4 lr for all
# run 6: 5e-4 for all
# run 7: critic higher tau, 0.05

# try stochastic critic eval
CUDA_VISIBLE_DEVICES=1 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb2cheetah/curl_cheetah_b256_84 \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 3000000 &

CUDA_VISIBLE_DEVICES=2 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_tb --work_dir ./tmp/icml/feb2cheetah/curl_cheetah_b512_84 \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 512 --num_train_steps 3000000 & 

CUDA_VISIBLE_DEVICES=3 python train.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 --batch_size 256 \
    --save_tb --work_dir ./tmp/icml/feb2cheetah/rad_cheetah_b256_84 \
    --agent sac_ae --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 128 --num_train_steps 3000000 & 

CUDA_VISIBLE_DEVICES=4 python train.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_tb --work_dir ./tmp/icml/feb2cheetah/rad_cheetah_b512_84 \
    --agent sac_ae --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 20000 --batch_size 512 --num_train_steps 3000000 & 

CUDA_VISIBLE_DEVICES=7 python train_cpc.py \
    --domain_name cheetah \
    --task_name run --dmc2gym \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_tb --work_dir ./tmp/icml/feb2cheetah/curl_cheetah_b256_84_lr3e4 \
    --agent sac_cpc --frame_stack 3 --pre_transform_image_size 100 --image_size 84 \
    --seed 23 --encoder_lr 3e-4 --critic_lr 3e-4 --actor_lr 3e-4 \
    --eval_freq 20000 --batch_size 256 --num_train_steps 3000000 