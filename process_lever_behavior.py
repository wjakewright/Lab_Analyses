"""Module to process lever press behavior from Dispatcher and Ephus outputs

    CREATOR - William (Jake) Wright 1/6/2022"""

import os
from dataclasses import dataclass



def load_xsg_continuous(dirname):
    """Function to load xsg files output from ephus
        
        INPUT PARAMETERS 
            dirname - string with the path to directory where files are located
    """

    files = [file for file in os.listdir(dirname) if file.endswith(".xsglog")]

    for file in files:
        fullname = os.path.join(dirname,file)
        fname = file
        name, epoch = parse_filename(fname)
        byte = os.path.getsize(fullname)
        cdate = os.path.getctime(fullname)
        mdate = os.path.getmtime(fullname)
        file_info = FileInfo(name,dirname,cdate,mdate,byte)

        if file == files[0]:
            name_first = name
            epoch_first = epoch
        else:
            if name_first != name or epoch_first != epoch:
                raise NameError("Files do not match. Only one epoch is allowed. Delete/move unused files.")
    
    data = xsglog_Data(name,epoch,file_info)



def parse_filename(fname):
    """ Function to parse filenames into substrings
    
        INPUT PARAMETERS
            fname - string of the filename to be parsed
            
    """


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


