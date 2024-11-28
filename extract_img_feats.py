import torch
from model.build_model import create_vat_model
from training.params import parse_args
from training.distributed import init_distributed_device, is_master, broadcast_object
from training.file_utils import pt_load
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
import os
import glob
import re
import sys
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

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

    # Ensure the model can run a test input
    test_input = torch.randn(1, 3, 8, 224, 224).to(device)  # Simulate a (C, T, H, W) input for batch size 1
    with torch.no_grad():
        try:
            _ = model.encode_image(test_input)
            print("Model loaded successfully and passed a test input check.")
        except Exception as e:
            raise RuntimeError(f"Model failed to process a test input. Error: {e}")

    return model, device

class InferenceProcessor:
    def __init__(self, args):
        # Initialize the tokenizer and transformations for images
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

    def load_image_segment(self, folder_path, start_frame, segment_size):
        """Load a segment of images from a folder given a start frame and segment size.
        If there aren't enough images, repeats the last image to fill the segment."""
        # Collect and sort all image files in the folder
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")

        # Load each image, apply transformation, and add to list
        segment_images = []
        for i in range(start_frame, start_frame + segment_size):
            if i < len(image_files):
                image_path = image_files[i]
            else:
                # Use the last image if not enough images
                image_path = image_files[-1]
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)  # Apply transformation
            segment_images.append(image_tensor)

        # Stack images to create (C, T, H, W) tensor
        video_segment = torch.stack(segment_images, dim=1)  # (C, T, H, W)
        
        # Check tensor shape
        if video_segment.shape != (3, segment_size, 224, 224):
            raise ValueError(f"Unexpected video segment shape: {video_segment.shape}")
        
        return video_segment
    
    
class FeatureExtractor:
    def __init__(self, model, device, processor, save_dir="features"):
        self.model = model.to(device)
        self.device = device
        self.processor = processor
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def extract_folder_features(self, folder_path, segment_size=8, overlap=1):
        """Extract overlapping embeddings from a folder of images and save them as a tensor."""
        output_dir = os.path.join(self.save_dir, os.path.basename(os.path.dirname(folder_path)))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(folder_path) + "_features.pt")
        
        # Skip extraction if features file already exists
        if os.path.exists(save_path):
            print(f"Features already exist for {folder_path}. Skipping extraction.")
            return

        image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        num_frames = len(image_files)
        embeddings = []

        # Process images in sliding windows
        for start in tqdm(range(0, num_frames - segment_size + 1, segment_size - overlap)):
            video_segment = self.processor.load_image_segment(folder_path, start, segment_size)
            video_segment = video_segment.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(video_segment)
                embeddings.append(embedding.cpu())

        # Concatenate all embeddings and save to disk
        embeddings_tensor = torch.cat(embeddings, dim=0)

        # Check embeddings tensor shape
        if embeddings_tensor.shape[0] != len(range(0, num_frames - segment_size + 1, segment_size - overlap)):
            raise ValueError("Mismatch in the number of embeddings and processed segments.")
        
        print(f"Final embeddings shape: {embeddings_tensor.shape}")
        torch.save(embeddings_tensor, save_path)
        print(f"Saved features to {save_path}")

    def process_dataset(self, base_dir, dataset_type, segment_size=8, overlap=1):
        """Process train, val, or test dataset and save embeddings for each folder."""
        dataset_dir = os.path.join(base_dir, dataset_type)
        for folder in sorted(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder_path}")
                self.extract_folder_features(folder_path, segment_size, overlap)

INPUT_BASE_DIR = "/vol/vssp/SF_datasets/singlevideo/phoenix-2014T/features/fullFrame-210x260px/"
OUTPUT_BASE_DIR = "/mnt/fast/nobackup/scratch4weeks/ef0036/phoenix/"

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    device = init_distributed_device(args)

    # Load the model
    model, device = load_model(args)
    processor = InferenceProcessor(args)
    extractor = FeatureExtractor(model, device, processor, save_dir=OUTPUT_BASE_DIR)

    # Process train, val, and test folders from the input directory, saving to output directory
    for dataset_type in ["train", "dev", "test"]:
        print(f"Processing {dataset_type} dataset...")
        extractor.process_dataset(INPUT_BASE_DIR, dataset_type, segment_size=8, overlap=1)
