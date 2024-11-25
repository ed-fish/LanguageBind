CACHE_DIR="cache_dir"
ANNOTATION="data/bobsl_captions/bobs_manual_captions.json"
# RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
# this script is for 1024 total batch_size (n(64) GPUs * batch_size(16) * accum_freq(1))

TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node 1 \
    -m train_translation  \
    --train-data ${ANNOTATION} \
	--train-num-samples 33307 \
    --clip-type "vl_new" --add-time-attn \
    --use_batched_dataset \
    --lock-text --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.0 \
    --epochs 100 --batch-size 2 --accum-freq 1 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "wandb" \
    --wandb-project-name "sign-bind-translate" \
    --name "bobsl-caption-mbart-1e-4" \
    --resume "/mnt/fast/nobackup/scratch4weeks/ef0036/langbind_weights/train_image/bs128_a100_acc_10/checkpoints/epoch_65.pt" \
    --do_train
    # --do_eval \
    # --val_vl_ret_data "bsl_dict" 
