import os
import time
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.base_datasets import VAT_dataset, VATBatchedDataset
from data.new_loadvat import get_wds_dataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_VAT_dataset(args):
    dataset = VAT_dataset(args)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # prefetch_factor=2,
        # persistent_workers=True,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_VAT_batched_dataset(args, chunk_size=8, stride=4):
    base_dataset = VAT_dataset(args)
    dataset = VATBatchedDataset(base_dataset, chunk_size=chunk_size, stride=stride)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
# data/base_datasets.py

import torch
from torch.utils.data import Dataset
import math

class VATChunksDataset(Dataset):
    """
    A dataset wrapper that splits each video into multiple overlapping chunks.
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

        # Precompute the total number of chunks across all videos
        self.chunk_offsets = []  # List of tuples: (video_idx, start_frame_idx)
        for video_idx in range(len(self.base_dataset)):
            num_frames = self.base_dataset.get_num_frames(video_idx)
            if num_frames < self.chunk_size:
                # Optionally, handle videos with fewer frames
                # For simplicity, we'll skip them
                continue
            num_chunks = 1 + (num_frames - self.chunk_size) // self.stride
            for chunk_idx in range(num_chunks):
                start_frame = chunk_idx * self.stride
                self.chunk_offsets.append((video_idx, start_frame))

    def __len__(self):
        return len(self.chunk_offsets)

    def __getitem__(self, idx):
        video_idx, start_frame = self.chunk_offsets[idx]
        frames, label = self.base_dataset.get_video_frames(video_idx, start_frame, self.chunk_size)
        return frames, label
    
def collate_fn(batch):
    # Filter out None values
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    # Sort batch by number of chunks (descending) to minimize padding
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    # Get the maximum number of chunks in the batch
    max_num_chunks = batch[0][0].shape[0]

    # Initialize lists
    batch_chunks = []
    chunk_attention_masks = []
    input_ids_list = []
    attention_mask_list = []

    for chunks, input_ids, attention_mask in batch:
        num_chunks = chunks.shape[0]
        # Pad chunks if necessary
        if num_chunks < max_num_chunks:
            pad_size = (max_num_chunks - num_chunks, ) + chunks.shape[1:]
            pad_tensor = torch.zeros(pad_size, dtype=chunks.dtype)
            chunks = torch.cat([chunks, pad_tensor], dim=0)
        batch_chunks.append(chunks)

        # Create attention mask for chunks
        chunk_attention_mask = torch.zeros(max_num_chunks, dtype=torch.long)
        chunk_attention_mask[:num_chunks] = 1
        chunk_attention_masks.append(chunk_attention_mask)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    # Stack tensors
    batch_chunks = torch.stack(batch_chunks, dim=0)  # [batch_size, max_num_chunks, channels, frames, H, W]
    chunk_attention_masks = torch.stack(chunk_attention_masks, dim=0)  # [batch_size, max_num_chunks]
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attention_mask_list, dim=0)

    return {
        'chunks': batch_chunks,
        'chunk_attention_masks': chunk_attention_masks,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def get_data(args, epoch=0):
    data = {}
    if args.do_train:
        print(args.train_data)
        if args.train_data.endswith(".json"):
            if args.use_batched_dataset:
                data[f"{args.clip_type}_pt"] = get_VAT_batched_dataset(args)
            else:
                data[f"{args.clip_type}_pt"] = get_VAT_dataset(args)
        elif args.train_data.endswith(".tar"):
            data[f"{args.clip_type}_pt"] = get_wds_dataset(args, is_train=True, epoch=epoch)
        else:
            raise NameError

    if args.do_eval:
        temp_batch_size = args.batch_size
        args.batch_size = 8 if args.val_vl_ret_data else 16
        data_root = "/mnt/fast/nobackup/scratch4weeks/ef0036/"
        if args.val_vl_ret_data:
            data["vl_ret"] = []
            for val_vl_ret_data in args.val_vl_ret_data:
                # if val_vl_ret_data == "msrvtt":
                #     args.train_csv = os.path.join(f'{data_root}/MSRVTT/MSRVTT_train.9k.csv')
                #     args.val_csv = os.path.join(f'{data_root}/MSRVTT/MSRVTT_JSFUSION_test.csv')
                #     args.data_path = os.path.join(f'{data_root}/MSRVTT/MSRVTT_data.json')
                #     args.features_path = os.path.join(f'{data_root}/MSRVTT/MSRVTT_Videos')
                # elif val_vl_ret_data == "msvd":
                #     args.data_path = os.path.join(f'{data_root}/MSVD')
                #     args.features_path = os.path.join(f'{data_root}/MSVD/MSVD_Videos')
                # elif val_vl_ret_data == "activity":
                #     args.data_path = os.path.join(f'{data_root}/ActivityNet')
                #     args.features_path = os.path.join(f'{data_root}/ActivityNet/Videos/Activity_Videos')
                # elif val_vl_ret_data == "didemo":
                #     args.data_path = os.path.join(f'{data_root}/Didemo')
                #     args.features_path = os.path.join(f'{data_root}/Didemo/videos')
                if val_vl_ret_data == "signbank":
                    args.data_path = os.path.join(f'{data_root}/signbank')
                    args.features_path = os.path.join(f'{data_root}/signbank/videos')

                elif val_vl_ret_data == "bsl_dict":
                    args.data_path = os.path.join(data_root, 'bsldict')
                    args.features_path = os.path.join(data_root, 'bsldict/videos_original')
                    
                elif val_vl_ret_data == "extract_feats":
                    args.data_path = os.path.join(data_root, 'phoenix')
                    args.features_path = os.path.join(data_root, 'phoenix/videos_original')
                else:
                    raise NameError

                args.batch_size_val = args.batch_size if args.batch_size_val == 0 else args.batch_size_val
                args.max_frames = args.num_frames
                args.num_thread_reader = args.workers
                args.slice_framepos = 2   # "0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly."

                from vl_ret.data_dataloaders import DATALOADER_DICT
                

                tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
                test_dataloader, test_length = None, 0
                if DATALOADER_DICT[val_vl_ret_data]["test"] is not None:
                    test_dataloader, test_length = DATALOADER_DICT[val_vl_ret_data]["test"](args, tokenizer)

                if DATALOADER_DICT[val_vl_ret_data]["val"] is not None:
                    val_dataloader, val_length = DATALOADER_DICT[val_vl_ret_data]["val"](args, tokenizer)
                else:
                    val_dataloader, val_length = test_dataloader, test_length
                ## report validation results if the ["test"] is None
                if test_dataloader is None:
                    test_dataloader, test_length = val_dataloader, val_length

                data["vl_ret"].append({val_vl_ret_data: test_dataloader})

        if args.val_v_cls_data:
            data["v_cls"] = []
            temp_val_v_cls_data = args.val_v_cls_data
            for val_v_cls_data in temp_val_v_cls_data:
                from v_cls import get_video_cls_dataloader
                args.val_v_cls_data = val_v_cls_data
                if args.val_v_cls_data == 'Kinetics-400':
                    args.video_data_path = "/apdcephfs_cq3/share_1311970/downstream_datasets/VideoCls/new_k400/Kinetics-400/raw/Kinetics-400"
                    args.nb_classes = 400
                elif args.val_v_cls_data == 'Kinetics-600':
                    args.video_data_path = "/apdcephfs_cq3/share_1311970/downstream_datasets/VideoCls/new_k600/Kinetics600/raw/Kinetics600"
                    args.nb_classes = 600
                args.data_root = args.video_data_path
                args.data_set = val_v_cls_data
                args.dist_eval = True
                args.sampling_rate = 8
                args.num_sample = 1
                args.test_num_segment = 5
                args.test_num_crop = 3
                args.num_workers = args.workers
                data['v_cls'].append({val_v_cls_data: get_video_cls_dataloader(args)})
            args.val_v_cls_data = temp_val_v_cls_data

        if args.val_a_cls_data:
            temp_audio_mean, temp_audio_std = args.audio_mean, args.audio_std
            args.audio_mean, args.audio_std = -4.2677393, 4.5689974
            data["a_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Audio"
            temp_val_a_cls_data = args.val_a_cls_data
            for val_a_cls_data in temp_val_a_cls_data:
                from a_cls.datasets import get_audio_dataset
                args.val_a_cls_data = val_a_cls_data
                args.audio_data_path = os.path.join(data_root, f'{val_a_cls_data.lower()}/test')
                data['a_cls'].append({val_a_cls_data: get_audio_dataset(args)})
            args.val_a_cls_data = temp_val_a_cls_data
            args.audio_mean, args.audio_mean = temp_audio_mean, temp_audio_std

        if args.val_al_ret_data:
            temp_audio_mean, temp_audio_std = args.audio_mean, args.audio_std
            args.audio_mean, args.audio_std = -4.2677393, 4.5689974

            data["al_ret"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Audio"
            temp_val_al_ret_data = args.val_al_ret_data
            for val_al_ret_data in temp_val_al_ret_data:
                from al_ret.datasets import get_audio_dataset
                args.val_al_ret_data = val_al_ret_data
                if val_al_ret_data.lower() != 'msrvtt':
                    args.audio_data_path = os.path.join(data_root, val_al_ret_data.lower())
                    data['al_ret'].append({val_al_ret_data: get_audio_dataset(args)})
                elif val_al_ret_data.lower() == 'msrvtt':
                    args.train_csv = os.path.join(f'/apdcephfs_cq3/share_1311970/downstream_datasets/VideoTextRetrieval/vtRetdata/MSRVTT/MSRVTT_train.9k.csv')
                    args.val_csv = os.path.join(f'/apdcephfs_cq3/share_1311970/downstream_datasets/VideoTextRetrieval/Audio/MSRVTT/MSRVTT_AUDIO_test.csv')
                    args.data_path = os.path.join(f'/apdcephfs_cq3/share_1311970/downstream_datasets/VideoTextRetrieval/vtRetdata/MSRVTT/MSRVTT_data.json')
                    args.features_path = os.path.join(f'/apdcephfs_cq3/share_1311970/downstream_datasets/VideoTextRetrieval/Audio/MSRVTT/videos/all')

                    args.num_thread_reader = args.workers
                    from al_ret.data_dataloaders import DATALOADER_DICT
                    args.batch_size_val = args.batch_size if args.batch_size_val == 0 else args.batch_size_val

                    tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
                    test_dataloader, test_length = None, 0
                    if DATALOADER_DICT[val_al_ret_data.lower()]["test"] is not None:
                        test_dataloader, test_length = DATALOADER_DICT[val_al_ret_data.lower()]["test"](args, tokenizer)

                    if DATALOADER_DICT[val_al_ret_data.lower()]["val"] is not None:
                        val_dataloader, val_length = DATALOADER_DICT[val_al_ret_data.lower()]["val"](args, tokenizer, subset="val")
                    else:
                        val_dataloader, val_length = test_dataloader, test_length
                    ## report validation results if the ["test"] is None
                    if test_dataloader is None:
                        test_dataloader, test_length = val_dataloader, val_length
                    data['al_ret'].append({val_al_ret_data: test_dataloader})

            args.val_al_ret_data = temp_val_al_ret_data
            args.audio_mean, args.audio_mean = temp_audio_mean, temp_audio_std

        if args.val_a_cls_data:
            temp_audio_mean, temp_audio_std = args.audio_mean, args.audio_std
            args.audio_mean, args.audio_std = -4.2677393, 4.5689974
            data["a_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Audio"
            temp_val_a_cls_data = args.val_a_cls_data
            for val_a_cls_data in temp_val_a_cls_data:
                from a_cls.datasets import get_audio_dataset
                args.val_a_cls_data = val_a_cls_data
                args.audio_data_path = os.path.join(data_root, f'{val_a_cls_data.lower()}/test')
                data['a_cls'].append({val_a_cls_data: get_audio_dataset(args)})
            args.val_a_cls_data = temp_val_a_cls_data
            args.audio_mean, args.audio_mean = temp_audio_mean, temp_audio_std

        if args.imagenet_val is not None:
            from i_cls.datasets import get_imagenet
            data['i_cls'] = {}
            data['i_cls']["imagenet-val"] = get_imagenet(args, "val")
        if args.imagenet_v2 is not None:
            from i_cls.datasets import get_imagenet
            if data.get('i_cls', None) is None:
                data['i_cls'] = {}
            data['i_cls']["imagenet-v2"] = get_imagenet(args, "v2")

        if args.val_d_cls_data:
            data["d_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Depth"
            temp_val_d_cls_data = args.val_d_cls_data
            for val_d_cls_data in temp_val_d_cls_data:
                from d_cls.datasets import get_depth_dataset
                args.val_d_cls_data = val_d_cls_data
                args.depth_data_path = os.path.join(data_root, f'{val_d_cls_data.lower()}/data/val')
                data['d_cls'].append({val_d_cls_data: get_depth_dataset(args)})
            args.val_d_cls_data = temp_val_d_cls_data


        if args.val_t_cls_data:
            data["t_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Thermal"
            temp_val_t_cls_data = args.val_t_cls_data
            for val_t_cls_data in temp_val_t_cls_data:
                from t_cls.datasets import get_thermal_dataset
                args.val_t_cls_data = val_t_cls_data
                args.thermal_data_path = os.path.join(data_root, f'{val_t_cls_data.lower()}/val')
                data['t_cls'].append({val_t_cls_data: get_thermal_dataset(args)})
            args.val_t_cls_data = temp_val_t_cls_data

        args.batch_size = temp_batch_size

    return data



