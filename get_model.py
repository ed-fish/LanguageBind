
import json
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer

# Load your JSON file
with open('data/phoenix/phoenix_train.json', 'r') as f:
    raw_data = json.load(f)

# Extract "polish_mplug" values
data = []

for key, value in raw_data.items():
    polish_mplug = value.get('polish_mplug', '').strip()
    if polish_mplug:
        data.append(polish_mplug)

# Initialize tokenizer and model
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="de_DE", tgt_lang="de_DE")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
configuration = model.config

# Trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

# Trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

new_tokenizer = tt.trimmed_tokenizer
new_model = mt.trimmed_model

# Save the trimmed tokenizer and model
new_tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
new_model.save_pretrained('pretrain_models/MBart_trimmed')

# Configure and save the MyTran model
configuration = MBartConfig.from_pretrained('pretrain_models/MBart_trimmed/config.json')
configuration.vocab_size = new_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = new_model.model.shared

mytran_model.save_pretrained('pretrain_models/mytran/')
