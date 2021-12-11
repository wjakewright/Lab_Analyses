import os
from os.path import join as pjoin
import scipy.io as sio
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


'''Module to process imaging videos for presnetation purposes. Allows transformation
    to a heatmap, downsampling, and frame labeling'''

def label_video(img_dir,labels,out_name,image_rate,speed=1,ds_rate=None,img_range=None,scale=None,resize=None,low_thresh=None,
                filter_param=None):
    '''Function to label specific frames of imaging videos that correpsonde
        with specific events. Also allows converstion to a heatmap and downsampling
        
        CREATOR
            William (Jake) Wright   -   12/10/2021
        
        INPUT PARAMETERS
            img_dir - string containing the path to the directory where all image files
                      are located

            labels - list containing which frame are to be labeled. Each item is a tuple of ints 
                     that will label all frames within that range

            out_name - string for the name of the saved video

            image_rate - int specifying the imaging rate of the images being loaded.

            speed - int specifying how what factor you want to speed the video up by. Default
                    is set to 1 for no change in speed.

            ds_rate - int specifying how many frames to average over for downsampling. Default 
                      is set to None for no downsampling
            
            img_range - tuple specifying the range of image files you wish to concatenate.
                        E.g. (0,4) to combine the first 4 image files. Default is set to none for 
                        concatenation of all images. Default is set to None to concat all images
            
            scale - integer spcifying by what factor you wish to enhance the brightness of the images.
                    Default is set to  None to not change the brightness
            
            resize - tuple spcifying what size you wish to resize the image to. (e.g. 500x500 pixels)
                     Default is set to None for no resizing.

            low_thresh = integer specifying what intensity you wish to set as the lower threshold. Note
                         data are of dtype=uint8. Default is set to None for no lower threshold
            
            filter_param - list of the ksize.width, ksize.height, sigmaX, sigmaY for Gaussian Filtering.
                            Utilized by cv2.GaussianBlur(). Default is set to None for no filtering.

            '''
    # Get all image filenames
    images = [img for img in os.listdir(img_dir) if img.endswith('.tif')]
    images = images.sort()
    # Grab and process images
    if img_range is None:
        images = images
    else:
        images = images[img_range[0],img_range[1]+1]
    # Loading the images
    tif_stack = []
    for image in images:
        frame = sio.imread(os.path.join(img_dir,image),plugin='tifffile')
        out = np.zeros(np.shape(frame))
        frame = cv2.normalize(frame,out,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        tif_stack.append(frame)
    # Concatenate all images
    tif_concat = []
    for t in tif_stack:
        for tif in range(np.shape(t)[0]):
            # Convert to heatmap
            i = t[tif,:,:]
            heat = cv2.applyColorMap(i,cv2.COLORMAP_MAGMA)
            heat = cv2.cvtColor(heat,cv2.COLOR_RGB2BGR)
            img = Image.fromarray(heat)
            # Scale brightness
            if scale is not None:
                enhancer = ImageEnhance.Brightness(img)
                im = enhancer.enhance(scale)
            else:
                im = img
            # Adjust image size
            if resize is not None:
                im = im.resize(resize,Image.ANTIALIAS)
            else:
                pass
            # Convert back to array
            img = np.array(im)
            if low_thresh is not None:
                _,img = cv2.threshold(img,low_thresh,cv2.THRESH_TOZERO)
            else:
                pass
            if filter_param is not None:
                img = cv2.GaussianBlur(img,(filter_param[0],filter_param[1]),filter_param[2],filter_param[3])
            else:
                pass

            tif_concat.append(img)
    # Add the labels
    tif_array = np.array(tif_concat)
    for label in labels:
        if label[1] < len(tif_array):
            a = int(label[0])
            b = int(label[1])
            tif_array[a:b+1,-50:-1,-50:-1,0] = 255
            tif_array[a:b+1,-50:-1,-50:-1,1:2] = 1

    # Downsample the video
    if ds_rate is not None:
        avg_tif = block_reduce(np.array(tif_concat),block_size=(ds_rate,1,1,1),func=np.mean,cval=np.mean(np.array(tif_concat)))
    else:
        avg_tif = np.array(tif_concat)
    # Make sure video arrays are of uint8
    avg_tif = (avg_tif).astype(np.uint8)

    # Save the video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_name = out_name + '.avi'
    fps = int(image_rate / ds_rate) * speed
    size = avg_tif[0].shape[:2]
    video = cv2.VideoWriter(video_name,fourcc,fps,size)
    for i,_ in enumerate(avg_tif):
        video.write(cv2.cvtColor(avg_tif[i],cv2.COLOR_RGB2BGR)) # ensuring image is BGR for cv2 ouput
    cv2.destroyAllWindows
    video.release()


            
    