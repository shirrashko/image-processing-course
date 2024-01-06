import sys
import mediapy as mp
import numpy as np
import matplotlib.pyplot as plt

# weight of each color channel when converting to grayscale by the RGB TO YIQ formula
RED_WEIGHT = 0.299
GREEN_WEIGHT = 0.587
BLUE_WEIGHT = 0.114
SQUARE_SIZE = 256  # hyperparameter for dividing the frame into squares


def read_video(video_path):
    """
    Reads a video from the specified path.

    :param video_path: The path to the video file.
    :return: An array of RGB frames in the video.
             Each frame is a 3D array with dimensions [frame_height, frame_width, color_channels].
    """
    return mp.read_video(video_path)


def rgb_to_grayscale(frames):
    """
    Converts RGB frames to grayscale frames using the RGB to YIQ formula.

    :param frames: An array of RGB frames. Each frame is a 3D array with dimensions
                   [frame_height, frame_width, color_channels].
    :return: An array of grayscale frames. Each frame is a 2D array with dimensions
             [frame_height, frame_width], where each element represents the grayscale
             intensity of a pixel at a specific location in the frame.
    """
    return RED_WEIGHT * frames[:, :, :, 0] + GREEN_WEIGHT * frames[:, :, :, 1] + BLUE_WEIGHT * frames[:, :, :, 2]


def normalize_frames(frames):
    """
    Normalizes frames by scaling pixel values to the [0, 1] range.

    :param frames: An array of frames.
    :return: An array of normalized frames.
    """
    return frames / 255.0


def divide_frame_into_squares(frame, square_size=SQUARE_SIZE):
    """
    Divides a frame into smaller squares of specified size.

    :param frame: The frame to be divided.
    :param square_size: The size of the squares (default is set by SQUARE_SIZE).
    :return: A list of smaller squares obtained from the frame.
    """
    rows, cols = frame.shape
    return [frame[i:i + square_size, j:j + square_size]
            for i in range(0, rows, square_size)
            for j in range(0, cols, square_size)]


def cumulative_histogram(square):
    """
    Computes the cumulative histogram of a square from the frame.

    :param square: A square segment of the frame.
    :return: The cumulative histogram of the square.
    """
    hist, _ = np.histogram(square, bins=256, range=(0, 1))
    return np.cumsum(hist)


def vectorial_distance(square1, square2):
    """
    Computes the L1 norm-based vectorial distance between two squares.

    :param square1: The first square segment.
    :param square2: The second square segment.
    :return: The L1 norm-based vectorial distance between the two squares.
    """
    hist1 = cumulative_histogram(square1)
    hist2 = cumulative_histogram(square2)
    return np.sum(np.abs(hist1 - hist2))


def compute_frame_differences(frames, square_size):
    """
    Computes the vectorial differences between each pair of consecutive frames.

    :param frames: An array of frames (grayscale).
    :param square_size: The size of the squares to divide each frame into.
    :return: A list of differences between each pair of consecutive frames.
    """
    differences = []

    for i in range(len(frames) - 1):
        current_frame_squares = divide_frame_into_squares(frames[i], square_size)
        next_frame_squares = divide_frame_into_squares(frames[i + 1], square_size)

        total_diff = sum(vectorial_distance(sq1, sq2) for sq1, sq2 in zip(current_frame_squares, next_frame_squares))
        differences.append(total_diff)

    return differences


def normalize_differences(differences):
    """
    Normalizes the frame differences to a range of [0, 1].

    :param differences: A list of raw frame differences.
    :return: A list of normalized frame differences.
    """
    min_diff = min(differences)
    max_diff = max(differences)
    return [(diff - min_diff) / (max_diff - min_diff) for diff in differences]


def find_max_difference_scene_cut(frames, square_size=SQUARE_SIZE):
    """
    Identifies the scene cut in the video based on the maximum frame difference.

    :param frames: An array of grayscale frames from the video.
    :param square_size: The size of the squares to divide each frame into.
    :return: The index of the frame with the maximum difference and a list of normalized differences.
    """
    differences = compute_frame_differences(frames, square_size)
    normalized_differences = normalize_differences(differences)

    max_diff = max(normalized_differences)
    cut_frame_index = normalized_differences.index(max_diff)

    return cut_frame_index, normalized_differences


def plot_differences(cut_frame_index, differences):
    """
    Plots the frame differences with markers indicating the scene cut.

    :param cut_frame_index: Index of the frame where the scene cut is detected.
    :param differences: A list of normalized frame differences.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(differences, marker='o', markersize=3)
    plt.title('Frame Differences')
    plt.xlabel('Frame Index')
    plt.ylabel('Normalized Difference')
    plt.axvline(x=cut_frame_index, color='r', linestyle='--', label=f'Cut at frame {cut_frame_index}')
    plt.legend()
    plt.show()


def main(video_path, video_type):
    """
    Main function to execute the scene cut detection process.

    :param video_path: The path to the video file.
    :param video_type: The type/category of the video.
    :return: The indices of the frames before and after the scene cut.
    """
    rgb_frames = read_video(video_path)
    grayscale_frames = rgb_to_grayscale(rgb_frames)
    normalized_frames = normalize_frames(grayscale_frames)

    cut_frame_index, differences = find_max_difference_scene_cut(normalized_frames)

    plot_differences(cut_frame_index, differences)

    return cut_frame_index, cut_frame_index + 1


if __name__ == '__main__':
    print(sys.argv[1], sys.argv[2])
    main(sys.argv[1], sys.argv[2])

# TODO: remove loops, understand the way the square works in my program, create graphs for the report
# TODO: run pip3 freeze > requirements.txt again to update the requirements.txt file when finished
