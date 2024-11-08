import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from model.build_model import create_vat_model
from training.params import parse_args
from data.process_text import load_and_transform_text
from data.process_video import get_video_transform
from training.file_utils import pt_load
from training.distributed import init_distributed_device, is_master, broadcast_object
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch.nn as nn

# Model Dictionaries
MODEL_DICT = {
    "ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
}
CHECKPOINT_DICT = {
    "ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
    "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin"
}

def load_model(args):
    device = init_distributed_device(args)
    args.device = device
    model_name = MODEL_DICT.get(args.model)
    checkpoint_path = CHECKPOINT_DICT.get(args.model)
    if not model_name or not checkpoint_path:
        raise ValueError(f"Model or checkpoint path for {args.model} not found.")
    
    args.model = model_name
    model = create_vat_model(args)
    
    if args.pretrained:
        pretrained_path = os.path.join(args.cache_dir, checkpoint_path)
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"CLIP weights not found at {pretrained_path}")
        
        clip_weights = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(clip_weights['state_dict'], strict=True)
        print("Loaded CLIP pre-trained weights.")
    
    resume_from = None
    if args.resume:
        if args.resume == "latest":
            checkpoint_dir = "/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/bs128_a100_text_freeze_semantic_pop_bobsl_capt_8/checkpoints/"
            if is_master(args):
                resume_from = get_latest_checkpoint(checkpoint_dir)
            if args.distributed:
                resume_from = broadcast_object(args, resume_from)
        else:
            resume_from = args.resume

    if resume_from:
        checkpoint = pt_load(resume_from, map_location=device)
        checkpoint_state = checkpoint["state_dict"]
        checkpoint_state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint_state.items()}
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        print(f"Loaded custom checkpoint with missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
    else:
        print("No checkpoint loaded for resuming.")
    
    model.to(device)
    model.eval()
    return model, device

class InferenceProcessor:
    def __init__(self, args):
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)
    
    def load_video_segment(self, video_path, start_frame, segment_size):
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = list(range(start_frame, start_frame + segment_size))
        video_data = vr.get_batch(frame_indices).permute(3, 0, 1, 2)
        return self.video_transform(video_data)
    
    def tokenize_text(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return tokens['input_ids'], tokens['attention_mask']

class TransformerDecoder(nn.Module):
    def __init__(self, encoder_output_dim, decoder_model="facebook/mbart-large-cc25"):
        super().__init__()
        self.decoder = MBartForConditionalGeneration.from_pretrained(decoder_model)
        self.embed_projection = nn.Linear(encoder_output_dim, self.decoder.config.d_model)

    def forward(self, encoder_outputs, decoder_input_ids, attention_mask=None, labels=None):
        batch_size, seq_len, feature_dim = encoder_outputs.size()
        projected_encoder_outputs = self.embed_projection(encoder_outputs.view(-1, feature_dim)).view(batch_size, seq_len, -1)
        outputs = self.decoder(
            inputs_embeds=projected_encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

class SignLanguageTranslationModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def forward(self, video_segments, decoder_input_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            encoder_outputs = self.encoder.encode_image(video_segments.to(self.device))
        
        decoder_outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return decoder_outputs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    device = init_distributed_device(args)
    model, device = load_model(args)
    processor = InferenceProcessor(args)

    # Video paths for testing
    video_paths = [
        '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/d_010_079_000_do-you-use-bsl.mp4'
    ]

    decoder = TransformerDecoder(encoder_output_dim=768)
    translation_model = SignLanguageTranslationModel(model, decoder, device)

    # Prepare input for inference
    video_path = video_paths[0]
    segment_size = 8
    overlap = 1
    vr = VideoReader(video_path, ctx=cpu(0))
    video_segments = []
    for start in range(0, len(vr) - segment_size + 1, segment_size - overlap):
        segment = processor.load_video_segment(video_path, start, segment_size).unsqueeze(0).to(device)
        video_segments.append(segment)
        
    video_segments = torch.cat(video_segments, dim=0)

    # Pass through the encoder to generate embeddings
    with torch.no_grad():
        encoder_outputs = model.encode_image(video_segments.to(device)).unsqueeze(0)

    # Adjust attention mask based on sequence length
    attention_mask = torch.ones((1, encoder_outputs.shape[1]), device=device)

    # Target text for testing
    target_text = "Expected translation"
    input_ids, _ = processor.tokenize_text(target_text)
    input_ids = input_ids.to(device)

    # Perform inference for translation
    translation_model.eval()
    translation_model.decoder.decoder.resize_token_embeddings(len(processor.tokenizer))  # Resizes the token embeddings
    generated_tokens = translation_model.decoder.decoder.generate(
        inputs_embeds=translation_model.decoder.embed_projection(encoder_outputs),
        attention_mask=attention_mask,
        max_length=50,
        num_beams=3,
        decoder_start_token_id=translation_model.decoder.decoder.config.decoder_start_token_id
    )
    generated_text = processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("Generated translation:", generated_text)
