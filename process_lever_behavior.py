"""Module to process lever press behavior from Dispatcher and Ephus outputs

    CREATOR - William (Jake) Wright 1/6/2022"""

import os
import re
from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal


def parse_lever_movement_continuous(xsg_data):
    """Function to parse the lever force into movement and nonmovement epochs
    
        INPUT PARAMETERS
            xsg_data - object containing the data from all the xsglog files. This is 
                        output from load_xsg_continuous() function.
                        
        OUTPUT PARAMETERS
            lever_active - 
            
            lever_force_resample - 
            
            lever_force_smooth - 
            
            lever_velocity_envelope_smooth -
            
    """
    # Set xsg sampling rate
    xsg_sample_rate = 10000  ## Should always be 10kHz
    # Resample the lever traces to 1kHz
    lever_force_resample = sysignal.resample_poly(
        xsg_data.channels["Lever"], up=1, down=10
    )
    butterworth_stop = 5 / 500  # fraction of nyquist (cutoff = 10 Hz)
    butter = sysignal.butter(4, butterworth_stop, "low")
    lever_force_smooth = sysignal.filtfilt(butter[0], butter[1], lever_force_resample)

    # Get lever velocity and smooth
    lever_velocity_resample = np.insert(np.diff(lever_force_smooth), 0, 0, axis=0)
    lever_velocity_resample_smooth = matlab_smooth(lever_velocity_resample, 5).astype(
        np.float64
    )
    lever_velocity_resample_smooth[
        np.argwhere(np.isnan(lever_velocity_resample)).flatten()
    ] = np.nan

    # Get lever velocity envelope
    lever_velocity_hilbert = sysignal(lever_velocity_resample_smooth)
    lever_velocity_envelope = np.sqrt(
        (lever_velocity_hilbert * np.conj(lever_velocity_hilbert))
    )
    # Change window if you would like to smooth envelope velocity
    lever_velocity_envelope_smooth = matlab_smooth(lever_velocity_envelope, 1)

    # Define active parts of the lever
    movethresh = 0.0007
    lever_active = (lever_velocity_envelope_smooth > movethresh).astype(int)
    # Give leeway on both ends to all movements
    movement_leeway = 150  # ms to extend the movment total (half on each end)
    movement_leeway_filt = np.ones(movement_leeway, dtype=int)
    lever_active = np.convolve(lever_active, movement_leeway_filt, "same")
    # Close gaps of detrmined size
    gap_allowance = 500  # in ms
    lever_active_switch = np.diff(
        np.pad(lever_active, pad_width=1, mode="constant", constant_value=0)
    )
    lever_active_starts = np.argwhere(lever_active_switch == 1).flatten()
    lever_active_stops = np.argwhere(lever_active_switch == -1).flatten()
    lever_active_movement_times = lever_active_starts - lever_active_stops


def load_xsg_continuous(dirname):
    """Function to load xsg files output from ephus
        
        INPUT PARAMETERS 
            dirname - string with the path to directory where files are located

        OUTPUT PARAMETERS
            data - dictionary containing an array for each xsglog data file
    """
    # Get all the xsglog file names
    files = [file for file in os.listdir(dirname) if file.endswith(".xsglog")]
    # Get all file names and info
    for file in files:
        fullname = os.path.join(dirname, file)
        fname = file
        name, epoch, _ = parse_xsg_filename(fname)
        byte = os.path.getsize(fullname)
        cdate = os.path.getctime(fullname)
        mdate = os.path.getmtime(fullname)
        file_info = FileInfo(name, dirname, cdate, mdate, byte)

        if file == files[0]:
            name_first = name
            epoch_first = epoch
        else:
            if name_first != name or epoch_first != epoch:
                raise NameError(
                    "Files do not match. Only one epoch is allowed. Delete/move unused files."
                )
    # Create the data object
    data = xsglog_Data(name, epoch, file_info)
    # Load the text file
    fn = name_first + ".txt"
    try:
        with open(os.path.join(dirname, fn), "r") as fid:
            data.txt = fid.read()
    except FileNotFoundError:
        print(f"Could not open {os.path.join(dirname,fn)}")
    # Load and store each xsglog file
    data.channels = {}
    for file in files:
        fn = file.name
        _, _, channel_name = parse_xsg_filename(fn)
        try:
            with open(os.path.join(dirname, fn), "rb") as fid:
                fdata = np.fromfile(fid, np.float64)
        except FileNotFoundError:
            print(f"Could not open {os.path.join(dirname,fn)}")
        data.channels[channel_name] = fdata

    # Check if all the data files are of the same length
    lens = map(len, data.channels.values())
    if len(set(lens)) == 1:
        pass
    else:
        raise ValueError("Mismatched data size!!!")

    return data


def parse_xsg_filename(fname):
    """ Function to parse filenames into substrings
    
        INPUT PARAMETERS
            fname - string of the filename to be parsed
            
    """
    name = re.search("[A-Z]{2}[0-9]{4}", fname).group()
    epoch = re.search("[A-Z]{4}[0-9]{4}", fname).group()
    channel = re.search("_(\w+).xsglog", fname).group()[1:]


def matlab_smooth(data, window):
    """Function to replicate the implementation of matlab smooth function
    
        INPUT PARAMETERS
            data - 1d numpy array
            
            window - int. Must be odd value
            
    """
    out0 = np.convolve(data, np.ones(window, dtype=int), "valid") / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(data[: window - 1])[::2] / r
    stop = (np.cumsum(data[:-window:-1])[::2] / r)[::1]

    return np.concatenate((start, out0, stop))


@dataclass
class FileInfo:
    """Dataclass to store file info"""

    fname: str
    dir: str
    cdate: str
    mdate: str
    bytes: int


@dataclass
class xsglog_Data:
    """Dataclass for storing the xsglog data"""

    name: str
    epoch: str
    file_info: FileInfo
