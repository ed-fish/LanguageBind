import os
import csv
import subprocess
import json
import cv2
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base path to the directory containing the videos
video_base_dir = '/vol/vssp/LF_datasets/mixedmode/mein-dgs-korpus/MASKED_VIDEOS/'  # Adjust this if needed
# Output directory where the extracted segments will be saved
output_dir = '/mnt/fast/nobackup/scratch4weeks/ef0036/mdgs_gloss/'
# Output path for the JSON file
json_output_file = 'dataset.json'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Path to the CSV file
csv_file = 'mdgs.csv'  # Update this with the actual path

# Initialize JSON data storage, load existing data if available
if os.path.exists(json_output_file):
    with open(json_output_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
else:
    json_data = {}

# Function to get the frame rate using OpenCV
def get_frame_rate(video_file):
    try:
        video_capture = cv2.VideoCapture(video_file)
        if not video_capture.isOpened():
            print(f"Error: Cannot open video file {video_file}")
            return None
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        return frame_rate
    except Exception as e:
        print(f"Error retrieving frame rate for {video_file}: {e}")
        return None

# Function to extract and re-encode a video segment using ffmpeg
def extract_and_encode_segment(video_file, start_frame, stop_frame, output_file, frame_rate):
    start_time = start_frame / frame_rate  # Convert frames to seconds
    duration = (stop_frame - start_frame) / frame_rate  # Duration in seconds

    # Re-encode the segment using ffmpeg
    ffmpeg_command = [
        'ffmpeg', '-i', video_file, '-ss', str(start_time), '-t', str(duration),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
        output_file, '-y'
    ]
    subprocess.run(ffmpeg_command, check=True)

    return os.path.exists(output_file) and os.path.getsize(output_file) > 0

# Function to clean and extract the gloss text for polish_mplug
def process_gloss(gloss_text):
    # Remove anything after a colon
    cleaned_text = re.split(r':', gloss_text)[0]
    # Remove the "num-" prefix if present
    cleaned_text = re.sub(r'^num-', '', cleaned_text, flags=re.IGNORECASE)
    # Remove numbers and any letters that follow them
    cleaned_text = re.sub(r'\d+\w*', '', cleaned_text)
    # Preserve umlauts, hyphens, and remove other special characters
    cleaned_text = re.sub(r'[^\wäöüßÄÖÜ-]', '', cleaned_text).lower().strip()
    return cleaned_text

# Function to update JSON data incrementally, appending new entries without overwriting
def update_json_file(json_data, json_output_file):
    with open(json_output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

# Worker function to process each video segment
def process_gloss_segment(row, gloss_segment, idx, video_path, sentence_start_frame, frame_rate):
    gloss_word, start_offset, end_offset = gloss_segment.split('/')
    start_frame = sentence_start_frame + int(start_offset)
    end_frame = sentence_start_frame + int(end_offset)

    # Create output file path for the gloss
    output_file = os.path.join(output_dir, f"{row['filename']}_gloss_{idx + 1}_{gloss_word}.mp4")

    # Check if the file already exists
    if os.path.exists(output_file):
        return f"File already exists, skipping extraction: {output_file}"

    # Extract and encode the segment if the file doesn't already exist
    if not extract_and_encode_segment(video_path, start_frame, end_frame, output_file, frame_rate):
        return f"Failed to create playable video for: {output_file}"

    # Clean gloss for the polish_mplug
    clean_gloss = process_gloss(gloss_word)

    # Prepare JSON entry for each gloss
    json_data[output_file] = {
        "folder": output_dir,
        "mplug": row['ger_text'],
        "gloss": gloss_word,
        "polish_mplug": clean_gloss,
        "ofa": f"A video of a DGS interpreter signing: {gloss_word}",
        "sound_mplug": "",
        "raw": "",
        "split": "val"
    }

    # Update the JSON file
    update_json_file(json_data, json_output_file)

    return f"Processed gloss: {output_file}"

# Main function to process CSV file and run tasks in parallel
def process_csv_in_parallel():
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        tasks = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            for row in reader:
                speaker = row['camera'].lower()  # 'a' or 'b' based on the camera field in lowercase
                filename = row['filename'].split('-')[0][:-1]  # The long number
                directory = filename  # Get the base directory name (everything before '-')
                
                # Construct the path to the video file
                video_path = os.path.join(video_base_dir, directory, f"{filename}_1{speaker}1.masked.mp4")

                if os.path.exists(video_path):
                    frame_rate = get_frame_rate(video_path)
                    if frame_rate is None:
                        print(f"Skipping {video_path} due to frame rate retrieval issue.")
                        continue

                    sentence_start_frame = int(row['start_time'])

                    # Process each gloss segment in parallel
                    gloss_segments = row['gloss'].split()
                    for idx, segment in enumerate(gloss_segments):
                        task = executor.submit(
                            process_gloss_segment, row, segment, idx, video_path, sentence_start_frame, frame_rate
                        )
                        tasks.append(task)

            # Collect results as tasks complete
            for task in as_completed(tasks):
                result = task.result()
                print(result)

process_csv_in_parallel()
print(f"JSON file created/updated at: {json_output_file}")
