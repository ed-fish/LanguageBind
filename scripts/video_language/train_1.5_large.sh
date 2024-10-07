CACHE_DIR="cache_dir"
ANNOTATION="data/all_data/annotations_train.json"
# RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
# this script is for 1024 total batch_size (n(64) GPUs * batch_size(16) * accum_freq(1))

TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP \
    -m main  \
    --train-data ${ANNOTATION} \
	--train-num-samples 49016 \
    --clip-type "vl_new" --add-time-attn \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 2 \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.5 \
    --epochs 10 --batch-size 12 --accum-freq 4 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "wandb" \
    --val_vl_ret_data "signbank" \
    --wandb-project-name "sign-bind" \
    --name "bsl_train_attention" \
    --do_eval \
    --val_vl_ret_data "signbank" \
    --resume "latest" \
    --do_train \
