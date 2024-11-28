import os
import json
import cv2  # OpenCV library for video processing

# Load the JSON file
json_file = '/vol/research/LF_datasets/mixedmode/BOBSL/bobsl-v1-release/spottings/dict_spottings.json'  # Update if necessary
with open(json_file, 'r') as f:
    data = json.load(f)

# Define the path where your videos are stored
video_directory = '/vol/vssp/datasets/mixedmode/BOBSL/bobsl-v1-release/MASKED_VIDEOS_BIGGEST'  # Update if necessary

# Define the output directory where clips will be saved
output_directory = '/mnt/4tb/data/bobslspot_mask_test'  # Update if necessary
os.makedirs(output_directory, exist_ok=True)

# Extract data from the JSON structure
# Iterate over the keys: 'train', 'public_test', and 'val'
for split in ['public_test']:
    for word in data[split].keys():
        # Extract relevant data for the current sample
        file_path = data[split][word]
        files = data[split][word]['names']
        global_times = data[split][word]['global_times']
        probs = data[split][word]["probs"]

        # Iterate over each mouthing time for this video
        for idx, time in enumerate(global_times):
            vid_path = files[idx]  # Use the video name directly as ID
            if probs[idx] < 0.8:
                continue

            if time is None:
                print(f"Sample {word} missing mouthing_time. Skipping.")
                continue

            # Construct the full path to the video file
            video_filename = f"{vid_path}.masked.mp4"  # Adjust extension if necessary
            video_path = os.path.join(video_directory, video_filename)

            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found. Skipping.")
                continue

            # Load the video using OpenCV
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Failed to open video {video_path}. Skipping.")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print(f"FPS is zero for video {video_path}. Skipping.")
                cap.release()
                continue
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps

            # Calculate start and end times for the clip (2 seconds total)
            durations = [0.6, 2, 3]
            if len(word) < 2:
                clip_duration = durations[0]
            if len(word) > 2 and len(word) < 5:
                clip_duration = durations[1]
            else:
                clip_duration = durations[2]

            clip_duration = 0.6  # seconds
            half_clip = clip_duration / 2.0

            start_time = max(0, time)
            end_time = min(duration, time + half_clip)

            # Convert times to frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Set the video position to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Prepare the output video writer
            output_filename = f"{files[idx]}_{word}_{idx}.mp4"  # Include word and index for clarity
            output_path = os.path.join(output_directory, output_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (adjust if necessary)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Read and write frames from start_frame to end_frame
            current_frame = start_frame

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached unexpectedly at frame {current_frame}.")
                    break
                out.write(frame)
                current_frame += 1

            # Release resources
            cap.release()
            out.release()

            print(f"Extracted clip saved to {output_path}")
