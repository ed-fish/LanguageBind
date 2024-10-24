import os
import re
import json
import moviepy.editor as mp

# Input paths
subtitles_folder = "/vol/research/LF_datasets/mixedmode/BOBSL/bobsl-v1-release/subtitles/manually-aligned"
video_folder = "/vol/vssp/datasets/mixedmode/BOBSL/bobsl-v1-release/MASKED_VIDEOS_BIGGEST"
output_base_folder = "/mnt/fast/nobackup/scratch4weeks/ef0036/bobsl_captions"
global_json_path = "bobs_manual_captions.json"

# Make sure the output base folder exists
os.makedirs(output_base_folder, exist_ok=True)

# Load or initialize global JSON data
if os.path.exists(global_json_path):
    with open(global_json_path, "r") as json_file:
        global_json_data = json.load(json_file)
else:
    global_json_data = {}

# Function to parse WebVTT subtitle file
def parse_vtt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    segments = []
    pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})")
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            start_time = convert_to_seconds(match.groups()[:4])
            end_time = convert_to_seconds(match.groups()[4:])
            text = lines[i + 1].strip()
            segments.append((start_time, end_time, text))
    return segments

# Function to convert timestamp to seconds
def convert_to_seconds(groups):
    hours, minutes, seconds, milliseconds = map(int, groups)
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

# Loop through all subtitle files in the folder
for subtitle_filename in os.listdir(subtitles_folder):
    if subtitle_filename.endswith(".vtt"):
        base_filename = os.path.splitext(subtitle_filename)[0]
        video_filename = os.path.join(video_folder, f"{base_filename}.masked.mp4")
        subtitle_path = os.path.join(subtitles_folder, subtitle_filename)
        output_folder = os.path.join(output_base_folder, base_filename)

        # Make sure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Parse the VTT file
        segments = parse_vtt(subtitle_path)

        # Load the video
        video = mp.VideoFileClip(video_filename)

        # Extract video segments
        for idx, (start, end, text) in enumerate(segments):
            if "[NOT-SIGNED]" not in text:  # Skip segments marked as [NOT-SIGNED]
                segment = video.subclip(start, end)
                output_path = os.path.join(output_folder, f"segment_{idx + 1}.mp4")
                segment.write_videofile(output_path, codec="libx264")
                print(f"Extracted: {output_path}")

                # Add entry to global JSON data
                segment_key = output_path
                global_json_data[segment_key] = {
                    "folder": os.path.abspath(output_folder),
                    "mplug": text,
                    "polish_mplug": text,
                    "ofa": "",
                    "sound_mplug": "",
                    "raw": "",
                    "split": "train"
                }

                # Write the updated entry to the global JSON file
                with open(global_json_path, "w") as json_file:
                    json.dump(global_json_data, json_file, indent=4)

print("Extraction complete for all files!")
