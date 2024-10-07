RESUME="/home/ef0036/Projects/LanguageBind/logs/bsl_train_attention_vlnew/checkpoints/epoch_9.pt"
CACHE_DIR="/home/ef0036/Projects/LanguageBind/cache_dir/"
# ANNOTATION="path/to/data"
# cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nproc_per_node 1 \
    -m main  \
    --train-data "data/signbank/annotation.json" \
    --train-num-samples 3465 \
    --clip-type "vl_new" --max-depth 10 \
    --add-time-attn \
    --lock-text --lock-image --text-type "mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir "${CACHE_DIR}" \
    --convert_to_lora --lora_r 2 \
    --lr 5e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --force-patch-dropout 0.5 \
    --epochs 1 --batch-size 10 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "decord" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "wandb" --resume ${RESUME} \
    --do_eval \
    --val_vl_ret_data "signbank" \
    --wandb-project-name "sign-bind-ep1" \
    --name "testing_debug_vl_new" \

    