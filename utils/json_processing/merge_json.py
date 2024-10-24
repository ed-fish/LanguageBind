import json
import os
import sys
import re
import inflect
from nltk.corpus import wordnet

# Initialize inflect engine for number-to-word conversion
inflect_engine = inflect.engine()

def clean_mplug(mplug_value):
    """
    Cleans the mplug value by removing numbers and replacing hyphens with spaces.
    """
    # Remove numbers
    mplug_value = re.sub(r'\d+', '', mplug_value)
    # Replace hyphens with spaces
    mplug_value = mplug_value.replace('-', ' ')
    return mplug_value.strip()

def is_number(s):
    """
    Check if the string is a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def convert_number_to_word(number):
    """
    Convert a number to its word form.
    """
    return inflect_engine.number_to_words(int(number))

def get_similar_words(word):
    """
    Gets a set of similar words using WordNet for the polish_mplug field.
    """
    synsets = wordnet.synsets(word)
    if not synsets:
        return word  # No similar words found, return original word

    # Get a list of synonyms from WordNet
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    
    # Return the first synonym as the polish_mplug value (or a combination)
    return ', '.join(synonyms) if synonyms else word

def get_part_of_speech(word):
    """
    Gets the part of speech of a word using WordNet.
    """
    synsets = wordnet.synsets(word)
    if not synsets:
        return None

    # Map WordNet POS to a readable form
    pos_map = {
        'n': 'noun',
        'v': 'verb',
        'a': 'adjective',
        'r': 'adverb'
    }

    # Get the part of speech of the first synset
    pos_tag = synsets[0].pos()

    # Return a human-readable part of speech
    return pos_map.get(pos_tag, "unknown")

def merge_json_files(json_file_paths, output_train_file, output_val_file, notes_file):
    """
    Merges multiple JSON files into separate train and val JSON files,
    modifies the mplug field, adds polish_mplug, and updates the ofa field.
    Stores the number of train and val samples in a notes.txt file.

    Args:
        json_file_paths (list): List of paths to the JSON files to merge.
        output_train_file (str): Path to the output JSON file for the train split.
        output_val_file (str): Path to the output JSON file for the val split.
        notes_file (str): Path to the notes.txt file to store counts.
    """
    merged_train_data = {}
    merged_val_data = {}

    for json_file_path in json_file_paths:
        print(f"Processing file: {json_file_path}")
        with open(json_file_path, 'r') as f:
            data = json.load(f)

            for key, value in data.items():
                split = value.get('split', None)
                folder = value.get('folder', None)
                video_id = key  # Original key, assuming it relates to the video ID
                video_filename = f"{video_id}.mp4"  # Adjust extension if necessary

                # Construct the full video path
                if folder is not None:
                    video_path = os.path.join(folder, video_filename)
                else:
                    print(f"Warning: Entry '{key}' missing 'folder' information. Skipping.")
                    continue

                # Clean and modify mplug
                mplug_value = value.get('mplug', "").strip()

                # Check if the mplug value is a number
                if is_number(mplug_value):
                    # Convert the number to a word form
                    mplug_value = convert_number_to_word(mplug_value)
                    pos_tag = "number"
                else:
                    # If it's not a number, clean the mplug value
                    clean_mplug_value = clean_mplug(mplug_value)
                    value['mplug'] = clean_mplug_value

                    # Get part of speech for the mplug value
                    pos_tag = get_part_of_speech(clean_mplug_value) or "word"

                # Add polish_mplug using WordNet similar words
                polish_mplug_value = get_similar_words(mplug_value)
                value['polish_mplug'] = polish_mplug_value

                # Add ofa field with the dynamic phrase
                value['ofa'] = f"a video of a BSL interpreter signing the {pos_tag} '{mplug_value}'"

                # Use the full video path as the key
                unique_key = video_path

                # Depending on the split, add to the corresponding dictionary
                if split == 'train':
                    if unique_key in merged_train_data:
                        print(f"Duplicate key '{unique_key}' found in train split. Skipping duplicate.")
                        continue
                    merged_train_data[unique_key] = value
                elif split == 'val':
                    if unique_key in merged_val_data:
                        print(f"Duplicate key '{unique_key}' found in val split. Skipping duplicate.")
                        continue
                    merged_val_data[unique_key] = value
                else:
                    print(f"Warning: Entry '{key}' has an unrecognized split '{split}'. Skipping.")
                    continue

    # Save merged train data
    train_sample_count = len(merged_train_data)
    if merged_train_data:
        os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
        with open(output_train_file, 'w') as f:
            json.dump(merged_train_data, f, indent=4)
        print(f"Merged train data saved to {output_train_file}")
    else:
        print("No train data found to merge.")

    # Save merged val data
    val_sample_count = len(merged_val_data)
    if merged_val_data:
        os.makedirs(os.path.dirname(output_val_file), exist_ok=True)
        with open(output_val_file, 'w') as f:
            json.dump(merged_val_data, f, indent=4)
        print(f"Merged val data saved to {output_val_file}")
    else:
        print("No val data found to merge.")

    # Write counts to notes.txt
    with open(notes_file, 'w') as f:
        f.write(f"Total number of training samples: {train_sample_count}\n")
        f.write(f"Total number of validation samples: {val_sample_count}\n")

    print(f"Sample counts saved to {notes_file}")

if __name__ == "__main__":
    # Example usage:
    # python merge_json.py output_train.json output_val.json notes.txt input1.json input2.json input3.json

    if len(sys.argv) < 5:
        print("Usage: python merge_json.py <output_train.json> <output_val.json> <notes.txt> <input1.json> [<input2.json> ...]")
        sys.exit(1)

    output_train_file = sys.argv[1]
    output_val_file = sys.argv[2]
    notes_file = sys.argv[3]
    json_file_paths = sys.argv[4:]

    merge_json_files(json_file_paths, output_train_file, output_val_file, notes_file)
