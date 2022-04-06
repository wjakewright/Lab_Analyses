"""Module to load and process xsg files output from ephus"""

import os
import re
from dataclasses import dataclass

import numpy as np


def load_xsg_continuous(dirname):
    """Function to load xsg files output from ephus
    
        INPUT PARAMETERS
            dirname - string with the path to the dir where files are located
        
        OUTPUT PARAMETERS
            data - xsglog_data dataclass with attributes
                        name - str of file name
                        epoch - str of epoch name
                        file_info - FileInfo dataclass
                        channels - directory with key value pairs for each file name
                                    and file data (np.array)
    """

    # Get all xsglog file names
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
                raise Exception(
                    "Files do not match. Only one epoch is allowed. Delete/move unused files"
                )
    # Create the xsglog data object
    xsg_data = Xsglog_Data(name=name, epoch=epoch, file_info=file_info, channels={})

    # Load the text file
    fn = name_first + ".txt"
    try:
        with open(os.path.join(dirname, fn), "r") as fid:
            xsg_data.txt = fid.read()
    except FileNotFoundError:
        print(f"Could not open {os.path.join(dirname, fn)}")

    for file in files:
        fn = file
        _, _, channel_name = parse_xsg_filename(fn)
        try:
            with open(os.path.join(dirname, fn), "rb") as fid:
                fdata = np.fromfile(fid, np.float64)
        except FileNotFoundError:
            print(f"Could not open {os.path.join(dirname, fn)}")
        xsg_data.channels[channel_name] = fdata

    # Check if all the data files are the same length
    lens = map(len, xsg_data.channels.values())
    if len(set(lens)) != 1:
        raise Exception("Mismatched ata size!!!!")

    return xsg_data


def parse_xsg_filename(fname):
    """Helper function to parse xsglog filenames"""

    name = re.search("[A-Z]{2}[0-9]{4}", fname).group()
    epoch = re.search("[A-Z]{4}[0-9]{4}", fname).group()
    channel = re.search("_(\w+)", fname).group()[1:]

    return name, epoch, channel


# ----------------------------------------------------------------------------
# -----------------------DATACLASSES USED HERE--------------------------------
# ----------------------------------------------------------------------------


@dataclass
class FileInfo:
    """Dataclass to store xsglog file information"""

    file_name: str
    directory: str
    created_date: str
    modified_data: str
    file_size: int


@dataclass
class Xsglog_Data:
    """Dataclass for storing the xsglog data"""

    name: str
    epoch: str
    file_info: FileInfo
    channels: dict
