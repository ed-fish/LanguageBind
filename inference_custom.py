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
import torch.nn.functional as F  # Corrected import
import glob
import re
import sys
import numpy as np

# Model and checkpoint configurations
MODEL_DICT = {
    "ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "VIT-VL-14": "models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66/pytorch_model.bin"
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
            # Specify your checkpoint directory here
            checkpoint_dir = "/path/to/your/checkpoints/"
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
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)

    def load_video(self, video_path, num_frames=8):
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_id_list = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        video_data = vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return self.video_transform(video_data)

    def tokenize_text(self, text):
        tokens = load_and_transform_text(text, self.tokenizer)
        return tokens['input_ids'], tokens['attention_mask']

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    model, device = load_model(args)
    processor = InferenceProcessor(args)

    # Example video paths and corresponding texts
    video_paths = ['data/rachel/BROTHER.mp4', 'data/rachel/PAPER.mp4', 'data/rachel/SISTER.mp4']
    language_texts = ['brother', 'paper', 'sister']

    # Load and preprocess videos
    video_inputs = [processor.load_video(vp).unsqueeze(0) for vp in video_paths]
    video_inputs = torch.cat(video_inputs, dim=0).to(device)

    # Tokenize text inputs
    input_ids_list, attention_masks_list = zip(*[processor.tokenize_text(text) for text in language_texts])
    input_ids = torch.stack(input_ids_list).to(device)
    attention_masks = torch.stack(attention_masks_list).to(device)

    # Run inference
    with torch.no_grad():
        embeddings = model(video_inputs, input_ids, attention_masks)

        # Retrieve embeddings and normalize
        image_features = embeddings['image_features']
        text_features = embeddings['text_features']

        image_features_norm = F.normalize(image_features, dim=1)
        text_features_norm = F.normalize(text_features, dim=1)

        # Compute similarities using normalized features
        video_text_similarity = image_features_norm @ text_features_norm.T
        video_video_similarity = image_features_norm @ image_features_norm.T

        # Print similarities between each video and text
        for i, img_feat in enumerate(image_features_norm):
            for j, txt_feat in enumerate(text_features_norm):
                similarity = (img_feat @ txt_feat.T).item()
                print(f"{video_paths[i]} x {language_texts[j]}: {similarity:.4f}")

        # Optionally, you can print the similarity matrices
        print("\nVideo x Text Similarity Matrix:")
        print(video_text_similarity.cpu().numpy())

        print("\nVideo x Video Similarity Matrix:")
        print(video_video_similarity.cpu().numpy())
