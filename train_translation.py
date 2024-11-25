import sys
import time
import math
import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from transformers import MBartTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from training.distributed import init_distributed_device, is_master
from training.params import parse_args
from training.precision import get_autocast
from training.logger import setup_logging
from open_clip import get_input_dtype
from model.translation_model import VideoTranslationModel
from model.build_model import create_vat_model
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

CHECKPOINT_DICT = {
    "ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
    "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin",
    "VIT-VL-14": "models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66/pytorch_model.bin"
}

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

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def load_model(args):
    from model.build_model import create_vat_model
    device = args.device
    # Create CLIP model
    model = create_vat_model(args)
    model.to(device)
    model.eval()
    return model

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('.pt', '_best.pt')
        torch.save(state, best_filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {start_epoch})")
    return start_epoch

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    data_loader_key = f'{args.clip_type}_pt'
    data[data_loader_key].set_epoch(epoch)
    dataloader = data[data_loader_key].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler.step()

        chunks = batch['chunks']  # [batch_size, max_num_chunks, channels, frames, H, W]
        chunk_attention_masks = batch['chunk_attention_masks']  # [batch_size, max_num_chunks]
        input_ids = batch['input_ids']  # [batch_size, seq_length]
        attention_mask = batch['attention_mask']  # [batch_size, seq_length]

        chunks = chunks.to(device=device, dtype=input_dtype, non_blocking=True)
        chunk_attention_masks = chunk_attention_masks.to(device=device, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        batch_size, max_num_chunks, channels, frames_per_chunk, height, width = chunks.shape
        chunks = chunks.view(-1, channels, frames_per_chunk, height, width)

        # Process chunks through the video encoder
        with autocast():
            decoder_input_ids = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].clone()
            labels[labels == model.mbart_model.config.pad_token_id] = -100

            outputs = model(
                chunks,
                labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask[:, :-1],
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

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            num_samples = batch_count * args.batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Update loss meter
            if 'loss' not in losses_m:
                losses_m['loss'] = AverageMeter()
            losses_m['loss'].update(loss.item(), args.batch_size)

            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:.5g} ({loss_m.avg:.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:.2f}/s, {samples_per_second_per_gpu:.2f}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:.5f} " + loss_log
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

def evaluate_model(model, data_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            chunks = batch['chunks']
            chunk_attention_masks = batch['chunk_attention_masks']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            chunks = chunks.to(device=device, non_blocking=True)
            chunk_attention_masks = chunk_attention_masks.to(device=device, non_blocking=True)
            input_ids = input_ids.to(device=device, non_blocking=True)
            attention_mask = attention_mask.to(device=device, non_blocking=True)

            batch_size, max_num_chunks, channels, frames_per_chunk, height, width = chunks.shape
            chunks = chunks.view(-1, channels, frames_per_chunk, height, width)

            # Process chunks through the video encoder
            chunk_embeddings = model.clip_model.encode_image(chunks)
            embedding_dim = chunk_embeddings.shape[-1]
            chunk_embeddings = chunk_embeddings.view(batch_size, max_num_chunks, embedding_dim)

            # Prepare decoder inputs
            decoder_input_ids = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].clone()
            labels[labels == model.mbart_model.config.pad_token_id] = -100

            outputs = model(
                encoder_outputs=chunk_embeddings,
                encoder_attention_mask=chunk_attention_masks,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask[:, :-1],
                labels=labels
            )
            loss = outputs.loss
            batch_size = chunks.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    args = parse_args(args)
    device = init_distributed_device(args)
    args.device = device
    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            date_str,
            f"pt_{args.clip_type}",
            f"text_{args.text_type}",
            f"bs_{args.batch_size}",
            f"ep_{args.epochs}",
            f"lr_{args.lr}",
            f"accum_{args.accum_freq}",
            f"model_{model_name_safe}",
        ])
    args.pretrained = CHECKPOINT_DICT[args.model]
    args.model = MODEL_DICT[args.model]

    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    args.log_path = None
    if is_master(args):
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, 'out.log')
        if os.path.exists(args.log_path) and not args.resume:
            print(
                f"Error. Experiment already exists. Use --name {args.name} to specify a new experiment."
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
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        logging.debug('Finished loading wandb.')

    random_seed(args.seed, 0)
    args.device = device

    # Load the video encoder
    clip_model = load_model(args)

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
    translation_model.mbart_model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    # Prepare data
    data = get_data(args, 0)  # Load data using your existing data loading function
    train_loader = data[f'{args.clip_type}_pt'].dataloader

    # Optionally prepare validation data
    val_loader = None
    if 'val' in data:
        val_loader = data['val'].dataloader

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(translation_model.parameters(), lr=args.lr)
    num_batches_per_epoch = train_loader.num_batches // args.accum_freq
    num_training_steps = args.epochs * num_batches_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'amp' else None

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        start_epoch = load_checkpoint(checkpoint_path, translation_model, optimizer, scheduler)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model=translation_model,
            data=data,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            tb_writer=None  # Replace with TensorBoard writer if needed
        )

        # Save checkpoint
        if is_master(args):
            checkpoint_filename = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            state = {
                'epoch': epoch + 1,
                'state_dict': translation_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state, is_best=False, filename=checkpoint_filename)

        # Evaluate the model
        if val_loader is not None:
            avg_val_loss = evaluate_model(translation_model, val_loader, tokenizer, device)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if is_master(args):
                    best_checkpoint_filename = os.path.join(args.checkpoint_path, "best_model.pt")
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': translation_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    save_checkpoint(state, is_best=True, filename=best_checkpoint_filename)
            logging.info(f"Validation Loss after epoch {epoch}: {avg_val_loss:.4f}")

    # Optionally, perform final evaluation
    if val_loader is not None:
        evaluate_model(translation_model, val_loader, tokenizer, device)

if __name__ == '__main__':
    main(sys.argv[1:])
