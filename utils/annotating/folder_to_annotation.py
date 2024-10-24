import os
import json
import random

# Define the directory containing your videos
video_directory = "/mnt/4tb/data/rachel_cropped_mp4"

# Initialize the dictionary to hold the JSON structure
annotation = {}

# Collect all video files
video_files = [video_file for video_file in os.listdir(video_directory) if video_file.endswith(".mp4")]

# Shuffle the list to randomize the data
random.shuffle(video_files)

# Calculate the split index for 80/20
split_index = int(0.8 * len(video_files))

# Split the videos into train and val sets
train_videos = video_files[:split_index]
val_videos = video_files[split_index:]

# Populate the annotation dictionary
for video_file in train_videos:
    video_id = video_file.replace(".mp4", "")
    annotation[video_id] = {
        "folder": "/mnt/4tb/data/rachel_cropped_mp4",
        "mplug": video_id,
        "polish_mplug": "",
        "ofa": "",
        "sound_mplug": "",
        "raw": "",
        "split": "train"
    }

for video_file in val_videos:
    video_id = video_file.replace(".mp4", "")
    annotation[video_id] = {
        "folder": "/mnt/4tb/data/rachel_cropped_mp4",
        "mplug": video_id,
        "polish_mplug": "",
        "ofa": "",
        "sound_mplug": "",
        "raw": "",
        "split": "val"
    }

# Save the annotation dictionary to a JSON file
output_file = "data/rachel/annotation.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
with open(output_file, "w") as json_file:
    json.dump(annotation, json_file, indent=4)

# Count the number of samples in each set
total_samples = len(video_files)
train_samples = len(train_videos)
val_samples = len(val_videos)

# Create notes.txt file with counts
notes_file = "data/rachel/notes.txt"
with open(notes_file, "w") as notes:
    notes.write(f"Total number of samples: {total_samples}\n")
    notes.write(f"Number of training samples: {train_samples}\n")
    notes.write(f"Number of validation samples: {val_samples}\n")

print(f"JSON file saved to {output_file}")
print(f"Notes file saved to {notes_file}")
