import os
import csv
import subprocess
import json

# Base path to the directory containing the videos
video_base_dir = '/vol/vssp/LF_datasets/mixedmode/mein-dgs-korpus/MASKED_VIDEOS/'  # Adjust this if needed
# Output directory where the extracted segments will be saved
output_dir = '/mnt/fast/nobackup/scratch4weeks/ef0036/mdgs/'
# Output path for the JSON file
json_output_file = 'dataset.json'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Path to the CSV file
csv_file = 'mdgs.csv'  # Update this with the actual path

# Dictionary to store JSON data
json_data = {}

# Function to get the frame rate using ffprobe
def get_frame_rate(video_file):
    ffprobe_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_file
    ]
    try:
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        frame_rate_str = result.stdout.strip()
        # Frame rate can be a fraction, e.g., "25/1", so we need to evaluate it
        if '/' in frame_rate_str:
            numerator, denominator = map(int, frame_rate_str.split('/'))
            frame_rate = numerator / denominator
        else:
            frame_rate = float(frame_rate_str)
        return frame_rate
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving frame rate for {video_file}: {e}")
        return None

# Function to extract a video segment using ffmpeg (using frame numbers converted to seconds)
def extract_segment(video_file, start_frame, stop_frame, output_file, frame_rate):
    start_time = start_frame / frame_rate  # Convert frames to seconds
    stop_time = stop_frame / frame_rate    # Convert frames to seconds
    ffmpeg_command = [
        'ffmpeg', '-i', video_file, '-ss', str(start_time), '-to', str(stop_time), '-c', 'copy', output_file
    ]
    subprocess.run(ffmpeg_command, check=True)

# Read the CSV file
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='|')
    for row in reader:
        speaker = row['camera'].lower()  # 'a' or 'b' based on the camera field in lowercase
        filename = row['filename'].split('-')[0][0:-1]  # The long number
        directory = filename  # Get the base directory name (everything before '-')
        
        # Construct the path to the video file
        video_path = os.path.join(video_base_dir, directory, f"{filename}_1{speaker}1.masked.mp4")

        if os.path.exists(video_path):
            # Retrieve the frame rate for the video
            frame_rate = get_frame_rate(video_path)
            if frame_rate is None:
                print(f"Skipping {video_path} due to frame rate retrieval issue.")
                continue

            # Extract start and stop frames
            start_frame = int(row['start_time'])  # Use frame numbers directly
            stop_frame = int(row['stop_time'])    # Use frame numbers directly

            # Create output file path
            output_file = os.path.join(output_dir, f"{row['filename']}_segment_{speaker}.mp4")

            # Extract the segment
            extract_segment(video_path, start_frame, stop_frame, output_file, frame_rate)
            print(f"Extracted: {output_file}")
            
            # Prepare JSON entry
            json_data[output_file] = {
                "folder": output_dir,
                "mplug": row['ger_text'],          # Use 'ger_text' for 'mplug'
                "gloss": row['gloss'],             # Add gloss information
                "ofa": f"A video of a BSL interpreter signing: {row['ger_text']}",
                "sound_mplug": "",
                "raw": "",
                "split": "val"
            }
        else:
            print(f"Video file {video_path} not found.")

# Write the JSON data to a file
with open(json_output_file, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, indent=4, ensure_ascii=False)

print(f"JSON file created: {json_output_file}")
