import os
import random
import logging
import torch
from torch.utils.data import Dataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .process_video import load_and_transform_video, get_video_transform
from .process_text import load_and_transform_text
import json
import decord

class VAT_dataset(Dataset):
    def __init__(self, args, translation_tokenizer=None):
        super().__init__()
        self.video_decode_backend = args.video_decode_backend
        self.num_frames = args.num_frames
        self.text_type = args.text_type
        self.train_data = args.train_data
        self.val_data = args.val_data
        self.train_num_samples = args.train_num_samples
        self.model = args.model
        self.cache_dir = args.cache_dir
        self.total_text = ['raw', 'mplug', 'polish_mplug', 'sound_mplug'] + [f'ofa{i}' for i in range(8)]
        self.weight = [0.2, 0.2, 0.2, 0.2] + [0.2 / 8] * 8
        self.title = self.text_type == 'raw'
        self.data_root = ''
        if translation_tokenizer:
            print("type of translation_tokenizer:", type(translation_tokenizer))
            self.translate = True
            self.tokenizer = translation_tokenizer
        else:
            self.tokenizer = get_tokenizer(HF_HUB_PREFIX + self.model, cache_dir=self.cache_dir)
            
        if args.do_train:
            with open(self.train_data, 'r') as f:
                self.id2title_folder_caps = json.load(f)
        if args.do_eval:
            with open(self.val_data, 'r') as f:
                self.id2title_folder_caps = json.load(f)
            
        self.ids = list(self.id2title_folder_caps.keys())[:self.train_num_samples]

        self.clip_type = args.clip_type

        self.video_transform = get_video_transform(args)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            id = self.ids[idx]
            folder = self.id2title_folder_caps[id]['folder']
            text_output, ofa_number = self.get_text(id)
            input_ids, attention_mask = text_output['input_ids'], text_output['attention_mask']
            video_data = self.get_video(id, folder)
            return video_data['pixel_values'], input_ids, attention_mask
        except Exception as error_msg:
            logging.info(f"Failed at {idx} with \"{error_msg}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def get_video(self, id, folder, start_frame=None, num_frames=None):
        video_path = id
        video = load_and_transform_video(
            video_path, 
            self.video_transform,
            video_decode_backend=self.video_decode_backend,
            num_frames=num_frames if num_frames is not None else self.num_frames,
            start_frame=start_frame
        )
        return video

    def get_text(self, id):
        if self.text_type != 'mix':
            text = self.id2title_folder_caps[id][self.text_type]
            text_output = load_and_transform_text(text, self.tokenizer, title=self.title)
            return text_output, None
        else:
            text_type = random.choices(self.total_text, self.weight)[0]
            ofa_number = None
            if text_type.startswith('ofa'):
                ofa_number = int(text_type[-1])
                text = self.id2title_folder_caps[id]['ofa'][ofa_number]
            else:
                text = self.id2title_folder_caps[id][text_type]

            text_output = load_and_transform_text(text, self.tokenizer, title=text_type=='raw')
            return text_output, ofa_number

    def get_num_frames(self, idx):
        id = self.ids[idx]
        video_path = id
        total_frames = get_video_frame_count(video_path)
        return total_frames

    def get_video_frames(self, idx, start_frame, chunk_size):
        video_data = self.get_video(
            id=self.ids[idx],
            folder=self.id2title_folder_caps[self.ids[idx]]['folder'],
            start_frame=start_frame,
            num_frames=chunk_size
        )
        return video_data

def get_video_frame_count(video_path):
    vr = decord.VideoReader(video_path)
    return len(vr)

class VATBatchedDataset(Dataset):
    """
    A dataset that for each video returns all its chunks of frames.
    Each chunk consists of `chunk_size` frames with a sliding window of `stride` frames.
    """

    def __init__(self, base_dataset, chunk_size=8, stride=4):
        """
        Args:
            base_dataset (Dataset): The original VAT_dataset instance.
            chunk_size (int): Number of frames per chunk.
            stride (int): Number of frames to slide the window for the next chunk.
        """
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.stride = stride
        self.ids = base_dataset.ids
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        num_frames = self.base_dataset.get_num_frames(idx)
        if num_frames < self.chunk_size:
            # Skip videos with fewer frames
            return None

        num_chunks = 1 + (num_frames - self.chunk_size) // self.stride
        chunks = []
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * self.stride
            video_data = self.base_dataset.get_video_frames(idx, start_frame, self.chunk_size)
            chunks.append(video_data['pixel_values'])
        # Stack chunks along a new dimension
        chunks = torch.stack(chunks, dim=0)  # Shape: [num_chunks, channels, frames, height, width]

        # Get text data
        id = self.ids[idx]
        text_output, _ = self.base_dataset.get_text(id)
        input_ids, attention_mask = text_output['input_ids'], text_output['attention_mask']

        return chunks, input_ids, attention_mask
