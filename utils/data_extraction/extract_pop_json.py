import os
import json
import re
import inflect
from nltk.corpus import wordnet

# Initialize inflect engine for number-to-word conversion
inflect_engine = inflect.engine()

# Clean mplug value
def clean_mplug(mplug_value):
    mplug_value = re.sub(r'\d+', '', mplug_value)
    mplug_value = mplug_value.replace('-', ' ')
    return mplug_value.strip()

# Check if string is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Convert number to word
def convert_number_to_word(number):
    return inflect_engine.number_to_words(int(number))

# Get synonyms for polish_mplug
def get_similar_words(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word

    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return ', '.join(synonyms) if synonyms else word

# Get part of speech
def get_part_of_speech(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None

    pos_map = {
        'n': 'noun',
        'v': 'verb',
        'a': 'adjective',
        'r': 'adverb'
    }
    pos_tag = synsets[0].pos()
    return pos_map.get(pos_tag, "unknown")

# Create JSON from folder structure
def create_json_from_folder_structure(base_path, output_file):
    data = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mp4"):
                folder_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                mplug_value = folder_name

                # Clean and process mplug value
                if is_number(mplug_value):
                    mplug_value = convert_number_to_word(mplug_value)
                    pos_tag = "number"
                else:
                    clean_mplug_value = clean_mplug(mplug_value)
                    mplug_value = clean_mplug_value
                    pos_tag = get_part_of_speech(clean_mplug_value) or "word"

                polish_mplug_value = get_similar_words(mplug_value)
                ofa_value = f"a video of an ASL interpreter signing the {pos_tag} '{mplug_value}'"

                data[file_path] = {
                    "folder": root,
                    "mplug": mplug_value,
                    "polish_mplug": polish_mplug_value,
                    "ofa": ofa_value,
                    "sound_mplug": "",
                    "raw": "",
                    "split": "train"
                }

    # Save JSON to output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to {output_file}")

# Main function
if __name__ == "__main__":
    base_path = "/mnt/fast/nobackup/scratch4weeks/ef0036/popsign_v1_0/game/train"  # Change to your base folder path
    output_file = "popsign_train.json"
    create_json_from_folder_structure(base_path, output_file)

