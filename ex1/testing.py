import cv2  # Import the OpenCV library


def save_frames(frames, output_directory):
    """Save frames as image files."""
    for i, frame in enumerate(frames):
        output_path = f"{output_directory}/frame_{i:04d}.png"  # Name the files as frame_0001.png, frame_0002.png, etc.
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert from RGB to BGR format for OpenCV


def preprocess_for_testing(frames, output_directory):
    save_frames(frames, output_directory)


# save_frames(frames, "./output_frames_video1")  Saves the frames in a directory named output_frames_video1

def get_number_of_frames(frames):
    return len(frames)
