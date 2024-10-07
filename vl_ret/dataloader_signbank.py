from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
from .rawvideo_util import RawVideoExtractor

import os
import json
import numpy as np
from torch.utils.data import Dataset
from .rawvideo_util import RawVideoExtractor

class Signbank_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            annotation_path,
            tokenizer,
            max_words=1,
            feature_framerate=1.0,
            max_frames=8,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            data_path=None,
            features_path=None,
    ):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2], "Invalid frame_order value"
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2], "Invalid slice_framepos value"

        self.subset = subset
        assert self.subset in ["train", "val"], "subset must be 'train' or 'val'"

        # Load and filter the annotation JSON data based on the subset
        self.annotation_data = self._load_annotation_data(self.annotation_path)

        # Get list of video IDs based on the subset
        self.video_id_list = [
            vid for vid, data in self.annotation_data.items() if data['split'] == self.subset
        ][:100]

        # Build video paths and prepare the video dictionary
        # MAYBE THIS NEEDS TO BE FEATURE PATH
        self.video_dict = {
            vid: os.path.join(self.features_path, f"{vid}.mp4")
            for vid, data in self.annotation_data.items() if vid in self.video_id_list
        }

        # Create a mapping from index to video ID
        self.idx2video_id = {idx: vid for idx, vid in enumerate(self.video_id_list)}

        # Initialize RawVideoExtractor
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

    def _load_annotation_data(self, annotation_path):
        with open(annotation_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.video_id_list)


    def _get_text(self, video_id):
        # Retrieve the caption associated with the video_id
        caption = self.annotation_data[video_id]['mplug']

        # Use the LanguageBindImageTokenizer to tokenize the caption
        output = self.tokenizer(
            caption,
        )

        # Extract the relevant fields from the tokenized output
        
        input_ids = output[0].squeeze()
        input_mask = output[1].squeeze()
        segment_ids = [0] * len(input_ids)
        # input_ids = tokenized['input_ids']  # Shape: (1, max_words)
        # attention_mask = tokenized['attention_mask']  # Shape: (1, max_words)

        # # Since the model might not require token type ids (as it's not a BERT-style model),
        # # initialize it to zeros if necessary.
        # token_type_ids = tokenized.get('token_type_ids', torch.zeros_like(input_ids))

        # Return the tensors. Ensure they are in the shape the model expects.
        return input_ids, input_mask, segment_ids
    
    def _get_rawvideo(self, video_id):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)

        video = np.zeros(
            (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32
        )
        video_path = self.video_dict[video_id]
        try:
            # Read the entire video
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)

                # Select frames according to max_frames and slice_framepos
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indices = np.linspace(
                            0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int
                        )
                        video_slice = raw_video_slice[sample_indices, ...]
                else:
                    video_slice = raw_video_slice

                # Adjust frame order if necessary
                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                slice_len = video_slice.shape[0]
                video[0][:slice_len, ...] = video_slice
                video_mask[0][:slice_len] = 1
            else:
                print(f"Warning: Video data for {video_id} has invalid shape.")
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            raise e

        return video, video_mask

    def __getitem__(self, index):
        video_id = self.idx2video_id[index]

        pairs_text, pairs_mask, pairs_segment = self._get_text(video_id)
        video, video_mask = self._get_rawvideo(video_id)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask


# class Signbank_DataLoader(Dataset):
#     def __init__(
#             self,
#             subset,
#             data_path,
#             features_path,
#             tokenizer,
#             max_words=30,
#             feature_framerate=1.0,
#             max_frames=100,
#             image_resolution=224,
#             frame_order=0,
#             slice_framepos=0,
#     ):
#         self.data_path = data_path
#         self.features_path = features_path
#         self.feature_framerate = feature_framerate
#         self.max_words = max_words
#         self.max_frames = max_frames
#         self.tokenizer = tokenizer
#         self.frame_order = frame_order
#         assert self.frame_order in [0, 1, 2]
#         self.slice_framepos = slice_framepos
#         assert self.slice_framepos in [0, 1, 2]

#         self.subset = subset
#         assert self.subset in ["train", "val"]

#         annotation_path = "/home/ef0036/Projects/LanguageBind/data/signbank/annotation.json"

#         # Load and filter the annotation JSON data based on the subset
#         self.pseudo_video_id_list, self.video_id_list, self.pseudo_caption_dict = self._process_annotation_file(annotation_path)

#         # Populate the video dictionary
#         video_dict = {}
#         for root, _, video_files in os.walk(self.features_path):
#             for video_file in video_files:
#                 video_id_ = ".".join(video_file.split(".")[:-1])[2:]
#                 print(video_id)
#                 if video_id_ not in self.video_id_list:
#                     continue
#                 file_path_ = os.path.join(root, video_file)
#                 video_dict[video_id_] = file_path_
#         self.video_dict = video_dict
#         print("video dict: {}".format(len(video_dict)))

#         # Get iterator video ids
#         self.video_id2idx_dict = {pseudo_video_id: id for id, pseudo_video_id in enumerate(self.pseudo_video_id_list)}

#         # Get all captions
#         self.iter2video_pairs_dict = {}
#         for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
#             if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
#                 continue
#             caption = self.pseudo_caption_dict[pseudo_video_id]
#             n_caption = len(caption['start'])
#             for sub_id in range(n_caption):
#                 self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (pseudo_video_id, sub_id)

#         self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
#         self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
#                               "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

#     def _process_annotation_file(self, annotation_path):
#         with open(annotation_path, 'r') as f:
#             json_data = json.load(f)

#         pseudo_video_id_list = []
#         video_id_list = []
#         pseudo_caption_dict = {}

#         for pseudo_video_id, annotation in json_data.items():
#             if annotation['split'] == self.subset:
#                 video_id = pseudo_video_id
#                 pseudo_video_id_list.append(pseudo_video_id)
#                 video_id_list.append(video_id)
#                 pseudo_caption_dict[pseudo_video_id] = {
#                     "start": np.array([0], dtype=object),
#                     "end": np.array([int(math.ceil(float(annotation.get('duration', 1))))], dtype=object),
#                     "text": np.array([annotation.get('mplug', '')], dtype=object)
#                 }

#         return pseudo_video_id_list, video_id_list, pseudo_caption_dict

#     def __len__(self):
#         return len(self.iter2video_pairs_dict)

#     def _get_text(self, pseudo_video_id, sub_id):
#         caption = self.pseudo_caption_dict[pseudo_video_id]
#         k = 1
#         r_ind = [sub_id]

#         starts = np.zeros(k, dtype=np.int64)
#         ends = np.zeros(k, dtype=np.int64)
#         pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
#         pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
#         pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

#         for i in range(k):
#             ind = r_ind[i]
#             start_, end_ = caption['start'][ind], caption['end'][ind]
#             output = self.tokenizer(caption['text'][ind])
#             starts[i], ends[i] = start_, end_

#             input_ids = output[0].squeeze().tolist()
#             input_mask = output[1].squeeze().tolist()
#             segment_ids = [0] * len(input_ids)

#             while len(input_ids) < self.max_words:
#                 input_ids.append(0)
#                 input_mask.append(0)
#                 segment_ids.append(0)
#             assert len(input_ids) == self.max_words
#             assert len(input_mask) == self.max_words
#             assert len(segment_ids) == self.max_words

#             pairs_text[i] = np.array(input_ids)
#             pairs_mask[i] = np.array(input_mask)
#             pairs_segment[i] = np.array(segment_ids)

#         return pairs_text, pairs_mask, pairs_segment, starts, ends

#     def _get_rawvideo(self, idx, s, e):
#         video_mask = np.zeros((len(s), self.max_frames), dtype=np.int64)
#         max_video_length = [0] * len(s)

#         # Pair x L x T x 3 x H x W
#         video = np.zeros((len(s), self.max_frames, 1, 3,
#                           self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)
#         video_path = self.video_dict[idx]
#         try:
#             for i in range(len(s)):
#                 start_time = int(s[i])
#                 end_time = int(e[i])
#                 start_time = start_time if start_time >= 0. else 0.
#                 end_time = end_time if end_time >= 0. else 0.
#                 if start_time > end_time:
#                     start_time, end_time = end_time, start_time
#                 elif start_time == end_time:
#                     end_time = end_time + 1

#                 raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
#                 raw_video_data = raw_video_data['video']

#                 if len(raw_video_data.shape) > 3:
#                     raw_video_data_clip = raw_video_data
#                     raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
#                     if self.max_frames < raw_video_slice.shape[0]:
#                         if self.slice_framepos == 0:
#                             video_slice = raw_video_slice[:self.max_frames, ...]
#                         elif self.slice_framepos == 1:
#                             video_slice = raw_video_slice[-self.max_frames:, ...]
#                         else:
#                             sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
#                             video_slice = raw_video_slice[sample_indx, ...]
#                     else:
#                         video_slice = raw_video_slice

#                     video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

#                     slice_len = video_slice.shape[0]
#                     max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
#                     if slice_len < 1:
#                         pass
#                     else:
#                         video[i][:slice_len, ...] = video_slice
#                 else:
#                     print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
#         except Exception as excep:
#             print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
#             raise excep

#         for i, v_length in enumerate(max_video_length):
#             video_mask[i][:v_length] = [1] * v_length

#         return video, video_mask

#     def __getitem__(self, feature_idx):
#         pseudo_video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
#         idx = self.video_id2idx_dict[pseudo_video_id]

#         pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(pseudo_video_id, sub_id)
#         video, video_mask = self._get_rawvideo(self.video_id_list[idx], starts, ends)
#         return pairs_text, pairs_mask, pairs_segment, video, video_mask
