CACHE_DIR="cache_dir"
ANNOTATION="data/all_data/annotations_train.json"
# RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
# this script is for 1024 total batch_size (n(64) GPUs * batch_s ze(16) * accum_freq(1))

TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node 1 \
    -m main  \
    --train-data ${ANNOTATION} \
	--train-num-samples 49016 \
    --clip-type "vl_new" --add-time-attn --semantic_loss \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 2 \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.5 \
    --epochs 25 --batch-size 12 --accum-freq 10 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "wandb" \
    --wandb-project-name "sign-bind" \
    --name "local_run_bs12_delete" \
    --resume "latest" \
    --do_train \
