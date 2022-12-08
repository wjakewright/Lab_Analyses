import os

import cv2
import numpy as np
from skimage import io as sio
from tifffile import imsave


def multi_plane_processing(
    img_dir, save_dir, out_name, num_plane=2, frame_num=800,
):
    """Function to process multi-plane imaging tiffs and seperate the different planes
        into seperate tiffs and folders
        
        INPUT PARAMETERS
            
            img_dir - str specifying the directory where all the tiffs are located

            save_dir - str specifying the directory where all the new tiffs will be saved

            num_plane - int specifying how many planes are in the loaded tiff files

            frame_num - int specifying how many frames we want in each output tiff file

            pixels - tuple specifying the dimension of the images in pixels
            
    """
    # Get all the image filenames
    fnames = [img for img in os.listdir(img_dir) if img.endswith(".tif")]
    fname_idx = [x.split("_")[-1] for x in fnames]
    fname_idx = [int(x.split(".")[0]) for x in fname_idx]
    fnames = [x for _, x in sorted(zip(fname_idx, fnames))]

    # Set up temporary output for each plane and save directory
    temp_output = {}
    for i in range(num_plane):
        k = f"plane_{i}"
        temp_output[k] = []
        temp_path = os.path.join(save_dir, k)
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)

    # Keep track of file outputs
    file_tracker = np.zeros(num_plane)

    # Process each tif file
    for tif_file in fnames:
        print(tif_file)
        file = sio.imread(os.path.join(img_dir, tif_file), plugin="tifffile")

        # Keep track of planes
        curr_plane = 0

        # Iterate through each frame
        for frame in range(file.shape[0]):
            tif = file[frame, :, :]
            plane_key = f"plane_{curr_plane}"
            temp_output[plane_key].append(tif)

            # Save output if has desired number of frames
            if len(temp_output[plane_key]) == frame_num:
                output_tif = np.array(temp_output[plane_key])
                file_num = str(int(file_tracker[curr_plane]))
                while len(file_num) < 5:
                    file_num = "0" + file_num
                tif_name = f"{out_name}_{plane_key}_{file_num}.tif"
                save_name = os.path.join(save_dir, plane_key, tif_name)
                imsave(save_name, output_tif)
                # Reset temp output
                temp_output[plane_key] = []
                file_tracker[curr_plane] = file_tracker[curr_plane] + 1

            # Change the current plane and continue onto the next frame
            if curr_plane < num_plane - 1:
                curr_plane = curr_plane + 1
            elif curr_plane == num_plane - 1:
                curr_plane = 0

    for i, (key, value) in enumerate(temp_output.items()):
        if len(value) != 0:
            output_tif = np.array(value)
            file_num = str(int(file_tracker[i]))
            while len(file_num) < 5:
                file_num = "0" + file_num
            tif_name = f"{out_name}_{key}_{file_num}.tif"
            save_name = os.path.join(save_dir, key, tif_name)
            imsave(save_name, output_tif)

