import os

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage import io as sio
from skimage.measure import block_reduce
from tifffile import imsave


def smooth_trim_video(
    image_dir,
    out_name,
    save_dir,
    frame_range,
    smooth_window,
):
    """Function to smooth imaging videos and trim them to a set frame range

    INPUT PARAMETERS
        img_dir - str containing the path to the directory where all the
                image files are located

        out_name - str for the name of the output file

        save_dir - str of the path to where to save the output file

        frame_range - tuple specifying the range of frames of the original
                    video to process

        smooth_window - int specifying how many frames you want to smooth over

        resize - tuple specifying what size you wish to resize the iamge to
                (e.g., 1000 x 1000 pixels)


    """
    FRAMES_PER_FILE = 800
    # Get all the image filenames
    fnames = [img for img in os.listdir(image_dir) if img.endswith(".tif")]
    fnames.sort()

    # Identify the videos to load to match the frame range
    first_frame = frame_range[0]
    last_frame = frame_range[1]
    # Keep track of frames in the files
    temp_frame_tracker = 0
    first_file_idx = None
    last_file_idx = None
    start_frames = None
    end_frames = None
    for i, _ in enumerate(fnames):
        update = temp_frame_tracker + FRAMES_PER_FILE
        if (temp_frame_tracker <= first_frame) and (update >= first_frame):
            first_file_idx = i
            start_frames = temp_frame_tracker
            temp_frame_tracker = temp_frame_tracker + FRAMES_PER_FILE
            continue
        if (temp_frame_tracker <= last_frame) and (update >= last_frame):
            last_file_idx = i
            end_frames = update
            break
        temp_frame_tracker = temp_frame_tracker + FRAMES_PER_FILE

    # Print images and correponding frame range
    print(f"Frames {start_frames} - {end_frames}")
    print(f"Start {first_frame}  End {last_frame}")

    print(f"File List:")
    load_images = []
    for i in range(first_file_idx, last_file_idx + 1):
        load_images.append(fnames[i])
        print(f"{i} : {fnames[i]}")

    # Go through and process the tif files
    trimmed_tifs = []

    for fnum, file in enumerate(load_images):
        tif = sio.imread(os.path.join(image_dir, file), plugin="tifffile")
        # Get only the frames needed
        if fnum == 0:
            start_diff = first_frame - start_frames
            tif = tif[start_diff:, :, :]
        if fnum == len(load_images) - 1:
            curr_start = end_frames - FRAMES_PER_FILE
            end_diff = last_frame - curr_start
            tif = tif[:end_diff, :, :]
        print(tif.shape)
        trimmed_tifs.append(tif)

    # Combine tifs into single tif file
    combined_tif = np.concatenate(trimmed_tifs, axis=0)

    # Perform smoothing
    smoothed_tif = uniform_filter1d(combined_tif, size=smooth_window, axis=0)

    # Remove the padding
    # smoothed_tif = smoothed_tif[PAD_FRAMES:-PAD_FRAMES, :, :]

    # Save the smoothed and trimmed tif file
    save_name = os.path.join(save_dir, out_name)
    # smoothed_tif = smoothed_tif.astype(np.uint8) # Ensure it is uint8
    imsave(save_name, smoothed_tif)

    print("Done Processing")
