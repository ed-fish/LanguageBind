
CACHE_DIR="cache_dir"
ANNOTATION="data/all_data/annotations_train.json"
# RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
# this script is for 1024 total batch_size (n(64) GPUs * batch_size(16) * accum_freq(1))

TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node 1 \
    -m inference_custom  \
    --train-data ${ANNOTATION} \
	--train-num-samples 49016 \
    --clip-type "vl_new" --add-time-attn \
    --lock-text --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --tube-size 1 --force-patch-dropout 0.0 \
    --epochs 25 --batch-size 12 --accum-freq 10 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "wandb" \
    --wandb-project-name "sign-bind" \
    --name "a100_bsl_dict_semantic_pop_bobsl_capt_8_2_polish" \
    --resume "/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/bs50_ac_50_mdgs_gloss/checkpoints/epoch_32.pt" \
    --do_eval \
    --val_vl_ret_data "bsl_dict" 
