CACHE_DIR="cache_dir"
TRAIN_ANNOTATION="data/phoenix/phoenix_train.json"
EVAL_ANNOTATION="data/phoenix/phoenix_train.json"
# RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
# this script is for 1024 total batch_size (n(64) GPUs * batch_size(16) * accum_freq(1))

TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node 1 \
    -m train_translation  \
    --train-data ${TRAIN_ANNOTATION} \
    --val-data ${EVAL_ANNOTATION} \
	--train-num-samples 20 \
    --clip-type "vl_new" --add-time-attn \
    --use_batched_dataset \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --lr 0.005 --coef-lr 1 \
    --convert_to_lora --lora_r 16 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.5 \
    --epochs 100 --batch-size 1 --accum-freq 1 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 5 --log-every-n-steps 20 --report-to "wandb" \
    --wandb-project-name "sign-bind-translate-mdgs" \
    --name "bobsl-phoenix-debug-delete" \
    --pretrained "/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/mdgs_pretrain_contrastive_gloss_cont/checkpoints/epoch_22.pt" \
    --do_train \
    --resume-translation "/home/ef0036/Projects/LanguageBind/logs/bobsl-phoenix-debug-delete/checkpoints/checkpoint_epoch_14.pt" \
    # --do_eval \

    # --val_vl_ret_data "bsl_dict" 
