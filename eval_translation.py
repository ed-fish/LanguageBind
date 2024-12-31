import sys
import logging
import os
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration
from datetime import datetime
from tqdm import tqdm

# Custom modules (ensure these are available in your project)
from training.params import parse_args
from training.distributed import init_distributed_device, is_master
from training.logger import setup_logging
from model.translation_model import VideoTranslationModel
from model.build_model import create_vat_model
from data.build_datasets import get_data

# Dictionary mapping for models and checkpoints


MODEL_DICT = {
    "ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
}

CHECKPOINT_DICT = {
    "ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
    "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin",
    "VIT-VL-14": "models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66/pytorch_model.bin"
}

def load_model(args):
    device = args.device
    model = create_vat_model(args)
    model.to(device)
    model.eval()
    return model

def load_checkpoint(checkpoint_path, model):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

def generate_translations(model, tokenizer, data_loader, device, output_file, trg_lang_id):
    model.eval()
    generated_texts = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Translations"):
            chunks = batch['chunks'].to(device)
            # Reshape and process chunks
            batch_size, num_chunks, channels, frames, height, width = chunks.shape
            chunks = chunks.view(-1, channels, frames, height, width)

            # Encode video frames
            embeddings = model.clip_model.encode_image(chunks)
            embedding_dim = embeddings.shape[-1]
            embeddings = embeddings.view(batch_size, num_chunks, embedding_dim)
            embeddings = model.encoder_projection(embeddings)
            embeddings = model.embed_scale * embeddings

            # Generate translations
            generated_ids = model.generate(
                encoder_hidden_states=embeddings,
                max_length=tokenizer.model_max_length,
                num_beams=4,
                trg_lang_id=trg_lang_id
            )
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(decoded_texts)
            generated_texts.extend(decoded_texts)

    # Save translations
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(text + '\n')

    print(f"Generated translations saved to {output_file}")

def main(argv):
    args = parse_args(argv)
    device = init_distributed_device(args)
    args.device = device

    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = f"translation_{date_str}"

    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    if is_master(args):
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, 'out.log')

    setup_logging(args.log_path, logging.INFO)

    # Get model key before overwriting args.model
    model_key = args.model

    # Set model paths
    args.pretrained = CHECKPOINT_DICT.get(model_key)
    args.model = MODEL_DICT.get(model_key)

    if args.model is None or args.pretrained is None:
        raise ValueError(f"Model or checkpoint path for '{model_key}' not found.")

    # Load the video encoder
    clip_model = load_model(args)

    # Create the translation model
    translation_model = VideoTranslationModel(
        clip_model=clip_model,
        mbart_model_name_or_path='facebook/mbart-large-cc25'
    )
    translation_model.to(device)

    # Prepare tokenizer
    tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    tokenizer.src_lang = 'en_XX'  # Source language
    tokenizer.tgt_lang = 'de_DE'  # Target language

    # Set decoder start token ID
    translation_model.mbart_model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    trg_lang_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    # Load checkpoint if provided
    if args.resume:
        load_checkpoint(args.resume, translation_model)

    # Prepare validation data
    data = get_data(args, 0)  # Ensure this function returns the correct data loader
    
    val_loader = data[f'{args.clip_type}_pt'].dataloader
    # val_loader = data['val_loader']  # Adjust according to your data structure

    # Generate translations
    output_file = os.path.join(log_base_path, 'translations.txt')
    generate_translations(translation_model, tokenizer, val_loader, device, output_file, trg_lang_id)

if __name__ == '__main__':
    main(sys.argv[1:])
