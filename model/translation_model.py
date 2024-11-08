import torch
import math
from torch import nn
from transformers import MBartForConditionalGeneration, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

class VideoTranslationModel(PreTrainedModel):
    def __init__(self, clip_model, mbart_model_name_or_path):
        # Initialize with the MBart configuration
        config = AutoConfig.from_pretrained(mbart_model_name_or_path)
        super().__init__(config)

        self.clip_model = clip_model  # Your CLIP video encoder
        self.mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name_or_path)
        
        # If encoder and MBart hidden sizes differ, add a projection layer
        if self.clip_model.config.projection_dim != self.mbart_model.config.d_model:
            self.encoder_projection = nn.Linear(
                self.clip_model.config.projection_dim, self.mbart_model.config.d_model
            )
        else:
            self.encoder_projection = nn.Identity()

        # Scaling factor for embeddings (optional)
        self.embed_scale = math.sqrt(self.mbart_model.config.d_model)

    def forward(self, pixel_values, labels=None, decoder_input_ids=None, decoder_attention_mask=None):
        # Obtain encoder outputs from the CLIP model
        encoder_hidden_states = self.clip_model.encode_image(pixel_values)
        # encoder_hidden_states shape: (batch_size, seq_len, hidden_size)

        # Project encoder outputs to match MBart's hidden size
        encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
        # Apply scaling
        encoder_hidden_states = self.embed_scale * encoder_hidden_states

        # Create an attention mask for the encoder (assuming no padding)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long, device=encoder_hidden_states.device
        )

        # Prepare encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # If decoder_input_ids are not provided, generate them from labels
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.mbart_model.prepare_decoder_input_ids_from_labels(labels)

        # Pass encoder outputs and masks to the MBart model
        outputs = self.mbart_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs

    def generate(self, pixel_values, max_length=50, num_beams=5):
        encoder_hidden_states = self.clip_model.encode_image(pixel_values)
        encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
        encoder_hidden_states = self.embed_scale * encoder_hidden_states
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long, device=encoder_hidden_states.device
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        generated_tokens = self.mbart_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
        )
        return generated_tokens
