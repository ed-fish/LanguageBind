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
from training.distributed import init_distributed_device, is_master
from training.params import parse_args
from training.precision import get_autocast
from training.logger import setup_logging
from open_clip import get_input_dtype
from model.translation_model import gloss_free_model
from model.build_model import create_vat_model
from data.build_datasets import get_data
from datetime import datetime
import os

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
    model = create_vat_model(args)
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

def pad_attention_masks(attention_mask, target_len):
    return nn.functional.pad(attention_mask, (0, target_len - attention_mask.size(1)), value=0)

def train_one_epoch(clip_model, translation_model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    clip_model.train()
    translation_model.train().to(device)

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
        if batch is None:
            continue
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler.step()

        chunks = batch['chunks']
        chunk_attention_masks = batch['chunk_attention_masks']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        chunks = chunks.to(device=device, dtype=input_dtype, non_blocking=True)
        chunk_attention_masks = chunk_attention_masks.to(device=device, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        batch_size, max_num_chunks, channels, frames_per_chunk, height, width = chunks.shape
        chunks = chunks.view(-1, channels, frames_per_chunk, height, width)
        with torch.no_grad():
            model_out = clip_model.encode_image(chunks, normalize=True)
            model_out = model_out.view(batch_size, -1, 768)
        chunk_attention_masks = pad_attention_masks(chunk_attention_masks, model_out.size(1))

        labels = input_ids.clone()
        labels[labels == translation_model.mbart.config.pad_token_id] = -100

        with autocast():
            # DEBUG PRINTS:
            # Print shapes before calling the forward
            # print("DEBUG TRAIN EPOCH:", epoch)
            # print("model_out.shape:", model_out.shape)
            # print("chunk_attention_masks.shape:", chunk_attention_masks.shape)
            # print("labels.shape:", labels.shape)
            # print("labels min/max:", labels.min().item(), labels.max().item())
            # print("tokenizer pad_token_id:", translation_model.mbart.config.pad_token_id)

            loss, logits = translation_model(
                input_embeds=model_out,
                attention_mask=chunk_attention_masks,
                tgt_input={
                    "input_ids": labels,
                    "attention_mask": (labels != -100).long()
                }
            )

        backward(loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(translation_model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(translation_model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        if is_master(args) and (i_accum % args.log_every_n_steps == 0):
            num_samples = (i_accum + 1) * args.batch_size * args.accum_freq
            percent_complete = 100.0 * num_samples / dataloader.num_samples
            losses_m.setdefault("loss", AverageMeter()).update(loss.item(), args.batch_size)
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            logging.info(
                f"Epoch {epoch}, Step {i_accum}/{num_batches_per_epoch} - Loss: {loss.item():.4f}, "
                f"Samples/sec: {samples_per_second:.2f}"
            )

def evaluate_model(model, data_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            chunks = batch['chunks'].to(device=device, non_blocking=True)
            chunk_attention_masks = batch['chunk_attention_masks'].to(device=device, non_blocking=True)
            input_ids = batch['input_ids'].to(device=device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device=device, non_blocking=True)

            batch_size, max_num_chunks, channels, frames_per_chunk, height, width = chunks.shape
            chunks = chunks.view(-1, channels, frames_per_chunk, height, width)

            model_out = model.clip_model.encode_image(chunks)
            model_out = model_out.view(batch_size, -1, model_out.size(-1))
            chunk_attention_masks = pad_attention_masks(chunk_attention_masks, model_out.size(1))

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # DEBUG PRINTS in evaluation
            # print("DEBUG EVAL")
            # print("model_out.shape:", model_out.shape)
            # print("chunk_attention_masks.shape:", chunk_attention_masks.shape)
            # print("labels.shape:", labels.shape)
            # print("labels min/max:", labels.min().item(), labels.max().item())
            # print("tokenizer pad_token_id:", tokenizer.pad_token_id)

            outputs = model(
                input_embeds=model_out,
                attention_mask=chunk_attention_masks,
                tgt_input={
                    "input_ids": labels,
                    "attention_mask": (labels != -100).long()
                },
            )
            loss = outputs.loss
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples

def main(args):
    args = parse_args(args)
    device = init_distributed_device(args)
    args.device = device
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

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

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

    random_seed(args.seed, 0)
    args.device = device
    clip_model = load_model(args)

    # Debug print after loading tokenizer and model
    tokenizer = MBartTokenizer.from_pretrained("pretrain_models/MBart_trimmed/", src_lang='de_DE', tgt_lang='de_DE')
    # print("DEBUG MAIN: tokenizer.vocab_size:", tokenizer.vocab_size)
    translation_model = gloss_free_model(embed_layer=True)
    # print("DEBUG MAIN: model vocab_size:", translation_model.mbart.config.vocab_size)
    # print("DEBUG MAIN: model config:", translation_model.mbart.config)

    if args.wandb and is_master(args):
        assert wandb is not None
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
            wandb.watch(translation_model, log='all')
        logging.debug('Finished loading wandb.')

    data = get_data(args, 0, tokenizer)
    train_loader = data[f'{args.clip_type}_pt'].dataloader
    val_loader = data['val'].dataloader if 'val' in data else None

    optimizer = optim.AdamW(translation_model.parameters(), lr=args.lr)
    num_batches_per_epoch = train_loader.num_batches // args.accum_freq
    num_training_steps = args.epochs * num_batches_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    scaler = torch.amp.GradScaler("cuda") if args.precision == 'amp' else None

    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        start_epoch = load_checkpoint(checkpoint_path, translation_model, optimizer, scheduler)

    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            clip_model=clip_model,
            translation_model=translation_model,
            data=data,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            tb_writer=None
        )

        if is_master(args):
            checkpoint_filename = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            state = {
                'epoch': epoch + 1,
                'state_dict': translation_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state, is_best=False, filename=checkpoint_filename)

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

    if val_loader is not None:
        evaluate_model(translation_model, val_loader, tokenizer, device)

if __name__ == '__main__':
    main(sys.argv[1:])
