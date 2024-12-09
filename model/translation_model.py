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

def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('Unsupported resnet model.')
    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', 'P2']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0))
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = make_resnet(name='resnet18')
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)
        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self, src: Tensor, src_length_batch):
        src = self.conv_2d(src)
        x_batch = []
        start = 0
        for length in src_length_batch:
            end = start + length
            x_batch.append(src[start:end])
            start = end
        x = nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        x = self.conv_1d(x)
        return x

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
        src = self.src_emb(src)
        return src

def config_decoder():
    from transformers import AutoConfig
    return MBartForConditionalGeneration.from_pretrained(
        "pretrain_models/MBart_trimmed/",
        ignore_mismatched_sizes=True,
        config=AutoConfig.from_pretrained(Path('pretrain_models/MBart_trimmed/config.json'))
    )

class gloss_free_model(nn.Module):
    def __init__(self, embed_dim=1024, pretrain=None, embed_layer=True):
        super(gloss_free_model, self).__init__()
        self.mbart = config_decoder()
        if embed_layer:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=768)
            self.embed_scale = math.sqrt(embed_dim)
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

    def forward(self, input_embeds, attention_mask, tgt_input):
        # DEBUG PRINTS:
        # print("DEBUG FORWARD in gloss_free_model")
        # print("input_embeds.shape:", input_embeds.shape)
        # print("attention_mask.shape:", attention_mask.shape)
        # print("tgt_input['input_ids'].shape:", tgt_input["input_ids"].shape)
        # print("tgt_input['input_ids'] min/max:", tgt_input["input_ids"].min().item(), tgt_input["input_ids"].max().item())

        input_embeds = self.embed_scale * self.sign_emb(input_embeds)

        decoder_input_ids = shift_tokens_right(
            tgt_input["input_ids"], pad_token_id=self.mbart.config.pad_token_id
        )
        decoder_attention_mask = (decoder_input_ids != self.mbart.config.pad_token_id).long()

        # print("decoder_input_ids.shape:", decoder_input_ids.shape)
        # print("decoder_input_ids min/max:", decoder_input_ids.min().item(), decoder_input_ids.max().item())
        # print("decoder_attention_mask.shape:", decoder_attention_mask.shape)
        
        out = self.mbart(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=tgt_input["input_ids"],
            return_dict=True,
        )
        return out["loss"], out["logits"]

    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id):
        inputs_embeds, attention_mask = self.share_forward(src_input)
        out = self.mbart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id
        )
        return out
