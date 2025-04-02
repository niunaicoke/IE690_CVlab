import os
import cv2
import random


paths = [
    "/home/yulab/Desktop/titanic",
    "/home/yulab/Desktop/lost in translation",
    "/home/yulab/Desktop/fastfurious",
    "/home/yulab/Desktop/conjuring"
]
NUM_FRAMES = 200
BASE_OUTPUT_DIR = "/home/yulab/Desktop/extracted_frames"  # Base output directory


os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
print(f"Frames will be saved to subfolders under: {os.path.abspath(BASE_OUTPUT_DIR)}")


def extract_random_frames(video_path, output_dir):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames!")
        return


    selected_frames = sorted(random.sample(range(total_frames), min(NUM_FRAMES, total_frames)))


    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for i, frame_idx in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{i + 1}.jpg")
            success = cv2.imwrite(frame_filename, frame)
            if not success:
                print(f"Failed to save {frame_filename} (check permissions/disk space)")
        else:
            print(f"Failed to read frame {frame_idx} from {video_path}")

    cap.release()



for path in paths:

    folder_name = os.path.basename(path)


    movie_output_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
    os.makedirs(movie_output_dir, exist_ok=True)


    for video_file in os.listdir(path):
        if video_file.lower().endswith(".mp4"):
            video_path = os.path.join(path, video_file)
            print(f"Processing: {video_path} → Saving to: {movie_output_dir}")
            extract_random_frames(video_path, movie_output_dir)

print("Extraction complete! Check the output directories.")