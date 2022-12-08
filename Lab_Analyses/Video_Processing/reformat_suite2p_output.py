import os

import numpy as np
from skimage import io as sio
from tifffile import TiffFile, imsave


def reformat_suite2p_output(img_dir, save_dir, out_name, frame_num=800):
    """Function to reformat motion corrected tifs output from 
        Suite2p
        
        INPUT PARAMETERS
            img_dir - str specifying the directory where all tif
                      files are located
                      
            save_dir - str specifying the directory where the
                        tif files should be saved
            
            out_name - str specifying the basename for new tif files
            
            frame_num - int specifying the desired number of frames
                        in each new tif file
    """

    # Get all the image filenames
    fnames = [img for img in os.listdir(img_dir) if img.endswith(".tif")]
    print(fnames)
    # Setup output dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Set up temporary output
    temp_output = []

    # Keep track of file outputs
    file_tracker = 0

    for tif_file in fnames:
        print(tif_file)
        temp_file = TiffFile(os.path.join(img_dir, tif_file))
        file = []
        for page in temp_file.pages:
            file.append(page.asarray())
        temp_file.close()
        file = np.array(file)
        # iterate through frames
        for frame in range(file.shape[0]):
            tif = file[frame, :, :]
            temp_output.append(tif)

            # Save output if has desired number of frames
            if len(temp_output) == frame_num:
                output_tif = np.array(temp_output)
                file_num = str(int(file_tracker))
                while len(file_num) < 5:
                    file_num = "0" + file_num
                tif_name = f"{out_name}_{file_num}.tif"
                save_name = os.path.join(save_dir, tif_name)
                imsave(save_name, output_tif)
                # Reset temp output
                temp_output = []
                file_tracker = file_tracker + 1

    if len(temp_output) != 0:
        output_tif = np.array(temp_output)
        file_num = str(int(file_tracker))
        while len(file_num) < 5:
            file_num = "0" + file_num
        tif_name = f"{out_name}_{file_num}.tif"
        save_name = os.path.join(save_dir, tif_name)
        imsave(save_name, output_tif)


def reformat_suite2p_zstacks(img_dir, save_dir, out_name):
    """Function to reformat motion corrected tifs of zstack images
        output from Suite2p. Each loaded tif should include all frames
        from the same imaging plane
        
        INPUT PARAMETERS
            img_dir - str specifying the directory where all tif
                        files are located
            
            save_dir - str specifying the directory where the tif 
                        files should be saved to
                        
            out_name - str specifying the name for the output tif
    """
    # Get all the image file names
    fnames = [img for img in os.listdir(img_dir) if img.endswith(".tif")]
    print(fnames)
    # set up save dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Set up output
    avg_stack = []

    for tif_file in fnames:
        # Load the tif file
        temp_file = TiffFile(os.path.join(img_dir, tif_file))
        file = []
        for page in temp_file.pages:
            file.append(page.asarray())
        temp_file.close()
        file = np.array(file)
        print(file.shape)

        # Average the frames
        avg_plane = np.mean(file, axis=0)
        avg_stack.append(avg_plane)

    # Convert output into single array
    avg_stack = np.array(avg_stack)
    avg_stack = avg_stack.astype(np.int16)

    # Save the image
    out_name = out_name + ".tif"
    save_name = os.path.join(save_dir, out_name)
    imsave(save_name, avg_stack)

