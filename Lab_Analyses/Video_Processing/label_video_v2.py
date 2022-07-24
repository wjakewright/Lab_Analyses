import os

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage import io as sio
from skimage.measure import block_reduce
from tifffile import imsave

"""Moduel to process imaging vides for presentation purposes. Allows
    transformation to a heatmap, downsampling, moving average, and frame
    labeling"""


def label_video(
    img_dir,
    labels,
    out_name,
    save_dir,
    avg_rate=None,
    ds_rate=None,
    img_range=None,
    gamma=None,
    resize=None,
    low_thresh=None,
):

    """Function to label specific frames of imaging videos that correspond
        with specific events. Also allows conversion to heatmap, moving average,
        and downsampling.

        Loads each individual tif file, processes it and then saves it before clearing
        memory and doing the next file

        Output are the processed tif files

        CREATOR
            William (Jake) Wright - 03/21/2022
        
        INPUT PARAMETERS
            img_dir - str containing the path to the directory where all the
                        image files are located

            labels - list containing which frames are to be labeled. Each item is a tuple
                        of ints that will label all frames within that range

            out_name - string for the name of the output files

            save_dir - str of the path to where to save the files

            avg_rate - int specifying how many frames you'd like to average over for moving
                        ovgerage smoothing. Optional with default set to None

            ds_rate - int specifying how many frames to average over for downsampling.
                        Optional with default set to none
            
            img_rage - Tuple spcifying the range of image file you wish to load. E.g. (0,4)
                        to load the first 4 images in the dir. Optional with default set to
                        None
            
            gamma - float specifying by what factor you wish to change the brightness of the images
                    using gamma correction. Default is set to None. 

            resize - tuple specifying what size you wish to resize the images to (e.g. 200x200 pixles)
                    Optional with default set to None

            low_thresh - int specifying what intensity you wish to set as the lower threshold. 
                        (0-255). Default is set to None


    """

    # Get all the image filenames
    fnames = [img for img in os.listdir(img_dir) if img.endswith(".tif")]
    fnames.sort()
    # Grab images to process
    if img_range is not None:
        fnames = fnames[img_range[0] : img_range[1]]

    # Set up variable to track how many frames have been processed
    frame_tracker = 0

    # Process each tif image

    for fnum, file in enumerate(fnames):
        frame = sio.imread(os.path.join(img_dir, file), plugin="tifffile")
        out = np.zeros(np.shape(frame))
        # Convert to uint8 and normalize
        curr_file = cv2.normalize(frame, out, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Process each frame
        processed_tifs = []
        for i in range(np.shape(curr_file)[0]):
            tif = curr_file[i, :, :]
            # Adjust brightness
            if gamma is not None:
                tif = gamma_correction(gamma, tif)

            # Threshold image
            if low_thresh is not None:
                _, tif = cv2.threshold(tif, low_thresh, 255, cv2.THRESH_TOZERO)

            # Convert to heatmap
            heat_tif = cv2.applyColorMap(tif, cv2.COLORMAP_INFERNO)
            heat_tif = cv2.cvtColor(heat_tif, cv2.COLOR_RGB2BGR)

            # Adjust image size
            if resize is not None:
                heat_tif = cv2.resize(heat_tif, resize)

            # Ensure data type is uint8
            heat_tif = (heat_tif).astype(np.uint8)
            processed_tifs.append(heat_tif)
            heat_tif = None  ## clearing variable for memory

        # Create moving average
        processed_tifs = np.array(processed_tifs)  # Convert to np.array
        if avg_rate is not None:
            avg_tifs = uniform_filter1d(processed_tifs, size=avg_rate, axis=0)
        else:
            avg_tifs = processed_tifs
        processed_tifs = None

        # Add the labels
        if labels:
            for label in labels:
                a = int(label[0]) - frame_tracker
                b = int(label[1]) - frame_tracker
                im_range = range(len(avg_tifs))
                if a in im_range and b in im_range:
                    print(f"{a} to {b}")
                    avg_tifs[a : b + 1, -50:-1, -50:-1, 0] = 255
                    avg_tifs[a : b + 1, -50:-1, -50:-1, 1:2] = 1
                elif b in im_range and a not in im_range:
                    print(f"{a} to {b}")
                    avg_tifs[0 : b + 1, -50:-1, -50:-1, 0] = 255
                    avg_tifs[0 : b + 1, -50:-1, -50:-1, 1:2] = 1
                elif a in im_range and b not in im_range:
                    print(f"{a} to {b}")
                    avg_tifs[a:-1, -50:-1, -50:-1, 0] = 255
                    avg_tifs[a:-1, -50:-1, -50:-1, 1:2] = 1

        # Update how many frames have been prrocessed
        frame_tracker = frame_tracker + (np.shape(curr_file)[0])

        # Downsample the image
        if ds_rate is not None:
            avg_tifs = block_reduce(
                avg_tifs,
                block_size=(ds_rate, 1, 1, 1),
                func=np.mean,
                cval=np.mean(avg_tifs),
            )

        # Save the tif file
        sname = out_name + f" _{fnum}.tif"
        save_name = os.path.join(save_dir, sname)
        avg_tifs = avg_tifs.astype(np.uint8)  # ensure it is uint8
        imsave(save_name, avg_tifs)
        print(f"Processed Image {fnum+1}")


def gamma_correction(gamma, image):
    """Helper function to perform gamma correction for labeled video"""

    inv_gamma = 1 / gamma
    table = [((i / 255) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(image, table)

