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
from tqdm import tqdm

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
            checkpoint_dir = "/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/bs128_a100_text_freeze_semantic_pop_bobsl_capt_8/checkpoints/"
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

class InferenceProcessor:
    def __init__(self, args):
        # Initialize the tokenizer and transformations from loadVAT
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)
    
    def load_video_segment(self, video_path, start_frame, segment_size):
        """Load a segment of a video given a start frame and segment size."""
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = list(range(start_frame, start_frame + segment_size))
        video_data = vr.get_batch(frame_indices).permute(3, 0, 1, 2)  # (C, T, H, W)
        return self.video_transform(video_data)

    def tokenize_text(self, text):
        """Tokenize and preprocess text."""
        tokens = load_and_transform_text(text, self.tokenizer)
        return tokens['input_ids'], tokens['attention_mask']

class FeatureExtractor:
    def __init__(self, model, device, processor, save_dir="features"):
        self.model = model.to(device)
        self.device = device
        self.processor = processor
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def extract_video_features(self, video_path, segment_size=8, overlap=1):
        """Extract overlapping embeddings from a video and save them as a tensor."""
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        embeddings = []
        
        # Process video in sliding windows
        for start in tqdm(range(0, num_frames - segment_size + 1, segment_size - overlap)):
            end = start + segment_size
            frame_indices = range(start, end)
            video_segment = vr.get_batch(frame_indices).permute(3, 0, 1, 2)  # (C, T, H, W)
            video_segment = self.processor.video_transform(video_segment).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(video_segment)
                embeddings.append(embedding.cpu())
        
        # Concatenate all embeddings and save to disk
        embeddings_tensor = torch.cat(embeddings, dim=0)
        save_path = os.path.join(self.save_dir, os.path.basename(video_path) + "_features.pt")
        torch.save(embeddings_tensor, save_path)
        print(f"Saved features to {save_path}")

    def process_video_list(self, video_paths, segment_size=8, overlap=1):
        """Process a list of videos and save embeddings for each."""
        for video_path in video_paths:
            self.extract_video_features(video_path, segment_size, overlap)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    device = init_distributed_device(args)

    # Load the model
    model, device = load_model(args)
    processor = InferenceProcessor(args)
    extractor = FeatureExtractor(model, device, processor, save_dir="extracted_features")

    # Example videos
    video_paths = [
        '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/d_010_079_000_do-you-use-bsl.mp4',
        '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/c_010_033_000_clown.mp4',
        '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/s_013_017_007_son.mp4',
        '/mnt/fast/nobackup/scratch4weeks/ef0036/bsldict/videos_original/b_010_045_005_brother.mp4'
    ]

    # Process each video and save features
    extractor.process_video_list(video_paths, segment_size=8, overlap=1)
