import os
import pickle
import cv2  # OpenCV library for video processing

# Load the pickle file
pickle_file = '/vol/vssp/datasets/mixedmode/BOBSL/bobsl-v1-release/annotations/bobsl_mouthing_c2281_verified_mouthing_9263_dict_15782.pkl'  # Update if necessary
with open(pickle_file, 'rb') as f:
    y = pickle.load(f)

# Define the path where your videos are stored
video_directory = '/vol/vssp/datasets/mixedmode/BOBSL/bobsl-v1-release/MASKED_VIDEOS_BIGGEST'  # Update if necessary

# Define the output directory where clips will be saved
output_directory = '/mnt/4tb/data/bobslmouth_mask'  # Update if necessary
os.makedirs(output_directory, exist_ok=True)

# Extract data from the pickle file
video_names = y["videos"]["name"]            # List of video file names
words = y["videos"]["word"]                  # Corresponding words mouthed
mouthing_times = y["videos"]["mouthing_time"]  # Corresponding mouthing times

# Iterate over each video
for idx, video_name in enumerate(video_names):
    # Extract data from the sample
    video_id = video_name.split(".")[0]  # Remove the file extension
    word = words[idx]
    mouthing_time = mouthing_times[idx]

    if mouthing_time is None:
        print(f"Sample {video_name} missing mouthing_time. Skipping.")
        continue

    # Construct the full path to the video file
    video_filename = f"{video_id}.masked.mp4"  # Adjust extension if necessary
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
    clip_duration = 0.6  # seconds
    half_clip = clip_duration / 2.0

    start_time = max(0, mouthing_time - half_clip)
    end_time = min(duration, mouthing_time + half_clip)

    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the video position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepare the output video writer
    output_filename = f"{video_id}_{word}.mp4"  # Include word for clarity
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
