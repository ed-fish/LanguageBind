import torch
from model.build_model import create_vat_model
from training.params import parse_args
from training.distributed import init_distributed_device, is_master, broadcast_object
from training.file_utils import pt_load
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from decord import VideoReader, cpu
from data.process_text import load_and_transform_text
from data.process_video import get_video_transform
import os
import glob
import re
import sys
import numpy as np
import io
from PIL import Image
from torchvision.transforms import ToTensor

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
MODEL_DICT = {
    "ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
}

CHECKPOINT_DICT = {
    "ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
    "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin",
    "VIT-VL-14": "models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66/pytorch_model.bin"
}

def get_latest_checkpoint(path: str):
    """Get the latest checkpoint from a directory."""
    checkpoints = glob.glob(os.path.join(path, '**/*.pt'), recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
        return checkpoints[-1]
    return None

def load_model(args):
    """Load model with pre-trained weights and optionally resume from checkpoint."""
    device = init_distributed_device(args)
    args.device = device

    # Load CLIP model name and path
    model_name = MODEL_DICT.get(args.model)
    checkpoint_path = CHECKPOINT_DICT.get(args.model)
    if not model_name or not checkpoint_path:
        raise ValueError(f"Model or checkpoint path for {args.model} not found.")
    
    # Assign model name and create the model
    args.model = model_name
    model = create_vat_model(args)
    
    # Load pre-trained CLIP weights
    if args.pretrained:
        pretrained_path = os.path.join(args.cache_dir, checkpoint_path)
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"CLIP weights not found at {pretrained_path}")
        
        clip_weights = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(clip_weights['state_dict'], strict=True)
        print("Loaded CLIP pre-trained weights.")
    
    # Resolve resume path for custom checkpoint
    resume_from = None
    if args.resume:
        if args.resume == "latest":
            checkpoint_dir = "/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/bs50_ac_50_mdgs_gloss/checkpoints"
            if is_master(args):
                resume_from = get_latest_checkpoint(checkpoint_dir)
            if args.distributed:
                resume_from = broadcast_object(args, resume_from)
        else:
            resume_from = args.resume  # If a specific path is provided

    # Load the custom checkpoint if available
    if resume_from:
        checkpoint = pt_load(resume_from, map_location=device)
        checkpoint_state = checkpoint["state_dict"]

        # Strip any "module." prefixes if necessary
        checkpoint_state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint_state.items()}
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        print(f"Loaded custom checkpoint with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
    else:
        print("No checkpoint loaded for resuming.")
    
    model.to(device)
    model.eval()

    return model, device

# Use Decode class to handle video and text transformation
class InferenceProcessor:
    def __init__(self, args):
        # Initialize the tokenizer and transformations from loadVAT
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)
    
    def load_video(self, video_path, num_frames=8):
        """Load and preprocess a video using Decord and video transform pipeline."""
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_id_list = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        video_data = vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # Rearrange to (C, T, H, W)
        return self.video_transform(video_data)

    def tokenize_text(self, text):
        """Tokenize and preprocess text as in loadVAT."""
        tokens = load_and_transform_text(text, self.tokenizer)
        return tokens['input_ids'], tokens['attention_mask']

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    model, device = load_model(args)
    
    # Initialize InferenceProcessor for consistent preprocessing
    processor = InferenceProcessor(args)

    # Example inputs
    # video_paths = ['/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/d_010_079_000_do-you-use-bsl.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/c_010_033_000_clown.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/s_013_017_007_son.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/b_010_045_005_brother.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/r_001_036_002_railroad-train.mp4']
    # language_texts = ['do you use bsl?', 'clown', 'son', 'brother', 'train']
    video_paths = ['/mnt/fast/nobackup/scratch4weeks/ef0036/mdgs_gloss/1583882A-2_gloss_1_BUCHSTABE1.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/mdgs_gloss/1583882A-3_gloss_1_DAZU1.mp4', '/mnt/fast/nobackup/scratch4weeks/ef0036/mdgs_gloss/1583882A-4_gloss_5_NUM-EINER1A:2d.mp4']
    language_texts = ['buchtabe', 'dazu', 'einer', 'one']

    # Load and transform videos
    video_inputs = [processor.load_video(vp).unsqueeze(0) for vp in video_paths]
    video_inputs = torch.cat(video_inputs, dim=0).to(device)  # Stack video inputs

    # Tokenize language inputs
    input_ids_list, attention_masks_list = zip(*[processor.tokenize_text(text) for text in language_texts])
    input_ids = torch.stack(input_ids_list).to(device)
    attention_masks = torch.stack(attention_masks_list).to(device)

    # Prepare inputs for model inference
    inputs = {
        'video': video_inputs,
        'language': input_ids,
        'attention_mask': attention_masks
    }
    
    # Run inference
    with torch.no_grad():
        embeddings = model(video_inputs, input_ids, attention_masks)
         
        
    # Calculate similarities between video and text embeddings
    
    video_text_similarity = embeddings['image_features'] @ embeddings['text_features'].T
    
    # video_text_similarity = torch.softmax(embeddings['image_features'] @ embeddings['text_features'].T, dim=-1)
    print("Video x Text Similarity: \n", video_text_similarity.cpu().numpy())
