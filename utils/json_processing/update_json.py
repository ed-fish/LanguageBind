import json
import os
import re
import inflect
from nltk.corpus import wordnet
import sys

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

def process_json_file(input_file, output_file):
    """
    Processes a single JSON file, modifies the mplug field, adds polish_mplug, and updates the ofa field.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file to save the processed data.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

        for key, value in data.items():
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

    # Save the processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Processed JSON saved to {output_file}")

if __name__ == "__main__":
    # Example usage:
    # python process_json.py input.json output.json

    if len(sys.argv) != 3:
        print("Usage: python process_json.py <input.json> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_json_file(input_file, output_file)
