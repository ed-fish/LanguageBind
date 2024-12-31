from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import numpy as np
from pathlib import Path



class V_encoder(nn.Module):
    def __init__(self, emb_size, feature_size):
        super(V_encoder, self).__init__()
        self.src_emb = nn.Linear(feature_size, emb_size)
        self.bn_ac = nn.Sequential(
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, src: Tensor):
        # src: [B, L, 768]
        src = self.src_emb(src)  # [B, L, emb_size]
        return src

def config_decoder():
    from transformers import AutoConfig
    return MBartForConditionalGeneration.from_pretrained(
        "pretrain_models/mytran/",
        ignore_mismatched_sizes=True,
        config=AutoConfig.from_pretrained(Path('pretrain_models/mytran/config.json'))
    )

class gloss_free_model(nn.Module):
    def __init__(self, embed_dim=1024, pretrain=None, embed_layer=True):
        super(gloss_free_model, self).__init__()
        self.mbart = config_decoder()
        self.embed_layer = embed_layer
        if embed_layer:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=768)
            self.embed_scale = math.sqrt(embed_dim)
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0
        
        # Add dilated convolutions to process temporal embeddings after projection
        # Example: two-layer stack with increasing dilation
        # input_embeds shape: [B, L, embed_dim]
        # Conv1d expects [B, embed_dim, L]
        self.temporal_convs = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=2, padding=2), 
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=4, padding=4),
            nn.ReLU()
        )

    def forward(self, input_embeds, attention_mask, tgt_input):
        # Apply embedding projection if needed
        input_embeds = self.sign_emb(input_embeds) # [B, L, emb_size]
        input_embeds = self.embed_scale * input_embeds

        # Apply dilated 1D convolutions
        # permute to [B, C, L] for convolution
        input_embeds = input_embeds.permute(0, 2, 1)
        input_embeds = self.temporal_convs(input_embeds)  # [B, emb_dim, L]
        input_embeds = input_embeds.permute(0, 2, 1)      # [B, L, emb_dim]

        decoder_input_ids = shift_tokens_right(
            tgt_input["input_ids"], pad_token_id=self.mbart.config.pad_token_id
        )
        decoder_attention_mask = (decoder_input_ids != self.mbart.config.pad_token_id).long()
        
        out = self.mbart(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=tgt_input["input_ids"],
            return_dict=True,
        )
        
        return out["loss"], out["logits"]
    
    def generate(self, input_embeds, attention_mask, max_new_tokens, num_beams, decoder_start_token_id):
        # Apply embedding projection if needed
        input_embeds = self.sign_emb(input_embeds) # [B, L, emb_size]
        input_embeds = self.embed_scale * input_embeds

        # Apply dilated 1D convolutions
        input_embeds = input_embeds.permute(0, 2, 1)
        input_embeds = self.temporal_convs(input_embeds)
        input_embeds = input_embeds.permute(0, 2, 1)

        # Generate sequences
        
               # Generate sequences
               
        out = self.mbart.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            num_beams=4,
            pad_token_id=1,
            max_length=1024,
            eos_token_id=2,
            bos_token_id=0,
            forced_eos_token_id=2,
            # Consider sampling if beam search fails:
            # do_sample=True,
            # top_k=50,
            # top_p=0.95,
            # temperature=1.0,
            # decoder_start_token_id=decoder_start_token_id
        )
        # out = self.mbart.generate(
        #     inputs_embeds=input_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     num_beams=num_beams,
        #     decoder_start_token_id=decoder_start_token_id,
        #     no_repeat_ngram_size=3,
        #     length_penalty=1.0,
        #     min_length=50,
        #     repetition_penalty=1.2
        # )
        return out
