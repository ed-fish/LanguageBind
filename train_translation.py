# train.py

import sys
import time
import math
import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import MBartTokenizer, get_linear_schedule_with_warmup
from training.distributed import init_distributed_device, is_master
from training.params import parse_args
from training.precision import get_autocast
from training.logger import setup_logging
# from training.utils import AverageMeter, unwrap_model, backward
from open_clip import get_input_dtype
from model.translation_model import VideoTranslationModel
from model.build_model import create_vat_model
# from data import get_data  # Assuming you have a get_data function
from data.build_datasets import get_data
from datetime import datetime
import os

# Import wandb if needed
try:
    import wandb
except ImportError:
    wandb = None


MODEL_DICT = {
    "ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
}



LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
CHECKPOINT_DICT = {"ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
                   "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin",
                   "VIT-VL-14": "models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66/pytorch_model.bin"}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)



def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def load_model(args):
    # Your existing load_model function
    from model.build_model import create_vat_model
    from training.distributed import init_distributed_device
    from open_clip import get_model_config

    device = init_distributed_device(args)
    args.device = device

    # Map model names to model configurations

    # Helper function to load model
    # model_name = MODEL_DICT.get(args.model)
    # if not model_name:
    #     raise ValueError(f"Model {args.model} not found in MODEL_DICT.")
    # args.model = model_name

    # Create CLIP model
    model = create_vat_model(args)

    # Load pretrained weights if specified
    # if args.pretrained:
    #     checkpoint_path = args.pretrained_checkpoint_path  # Define this in args
    #     clip_weights = torch.load(checkpoint_path, map_location='cpu')
    #     model.load_state_dict(clip_weights['state_dict'], strict=True)
    #     print("Loaded CLIP pre-trained weights.")

    model.to(device)
    model.eval()
    return model, device

def train_one_epoch(model, data, loss_fn, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    # if args.distill:
    #     dist_model.eval()

    # Access the data loader from the data dictionary
    data_loader_key = f'{args.clip_type}_pt'
    data[data_loader_key].set_epoch(epoch)
    dataloader = data[data_loader_key].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler.step()

        images, input_ids, attention_mask = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                outputs = model(
                    pixel_values=images,
                    labels=input_ids,
                    # decoder_attention_mask=attention_mask,
                    # encoder_attention_mask=None,  # Adjust if needed
                )
                loss = outputs.loss

            backward(loss, scaler)
        else:
            # Gradient accumulation logic
            with torch.no_grad():
                with autocast():
                    outputs = model(
                        pixel_values=images,
                        labels=input_ids,
                        # decoder_attention_mask=attention_mask,
                        # encoder_attention_mask=None,
                    )
                    encoder_hidden_states = outputs.encoder_last_hidden_state
                    if 'encoder_hidden_states' in accum_features:
                        accum_features['encoder_hidden_states'].append(encoder_hidden_states)
                    else:
                        accum_features['encoder_hidden_states'] = [encoder_hidden_states]

                accum_images.append(images)
                accum_input_ids.append(input_ids)
                accum_attention_mask.append(attention_mask)

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                input_ids = accum_input_ids[j]
                attention_mask = accum_attention_mask[j]
                with autocast():
                    outputs = model(
                        pixel_values=images,
                        labels=input_ids,
                        # decoder_attention_mask=attention_mask,
                        # encoder_attention_mask=None,
                    )
                    loss = outputs.loss
                backward(loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        if args.accum_freq > 1:
            accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = images.size(0)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Update loss meter
            if 'loss' not in losses_m:
                losses_m['loss'] = AverageMeter()
            losses_m['loss'].update(loss.item(), batch_size)

            samples_per_second = args.accum_freq * batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * batch_size / batch_time_m.val

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
            )

            # Logging to TensorBoard or WandB
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": losses_m['loss'].val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            batch_time_m.reset()
            data_time_m.reset()
    # end for

def main(args):
    args = parse_args(args)
    device = init_distributed_device(args)
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"pt_{args.clip_type}",
            f"text_{args.text_type}",
            f"bs_{args.batch_size}",
            f"ep_{args.epochs}",
            f"mask_{args.force_patch_dropout}",
            f"lorar_{args.lora_r}" if args.convert_to_lora else "",
            f"lr_{args.lr}",
            f"coeflr_{args.coef_lr}",
            f"warm_{args.warmup}",
            f"accum_{args.accum_freq}",
            f"tattn_{args.add_time_attn}" if args.clip_type == 'vl' else "",
            f"model_{model_name_safe}",
            f"frm_{args.num_frames}",
            f"vdb_{args.video_decode_backend}",
        ])
    args.pretrained = CHECKPOINT_DICT[args.model]
    args.model = MODEL_DICT[args.model]

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)




    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        # args.train_sz = data["train"].dataloader.num_samples
        # if args.val_data is not None:
        #     args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        logging.debug('Finished loading wandb.')

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    args.device = device
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

    # Load the video encoder
    clip_model, device = load_model(args)

    # Create the translation model
    translation_model = VideoTranslationModel(
        clip_model=clip_model,
        mbart_model_name_or_path='facebook/mbart-large-cc25'  # or your desired MBart model
    )
    translation_model.to(device)

    # Prepare tokenizer
    tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    tokenizer.src_lang = 'en_XX'  # Adjust source language code
    tokenizer.tgt_lang = 'en_XX'  # Adjust target language code

    # Set decoder start token ID
    translation_model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    # Prepare data
    data = get_data(args, 0)  # Load data using your existing data loading function

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(translation_model.parameters(), lr=args.lr)
    print(data.keys())
    num_batches_per_epoch = data[f'{args.clip_type}_pt'].dataloader.num_batches // args.accum_freq

    num_training_steps = args.epochs * num_batches_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'amp' else None

    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(
            model=translation_model,
            data=data,
            loss_fn=loss_fn,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            dist_model=None,
            args=args,
            tb_writer=None  # Replace with TensorBoard writer if needed
        )

        # Save checkpoint (implement save_checkpoint function as needed)
        # save_checkpoint(translation_model, optimizer, epoch, args)

        # Optionally, perform validation
        # evaluate_model(translation_model, val_dataloader, device)

if __name__ == '__main__':
    main(sys.argv[1:])
