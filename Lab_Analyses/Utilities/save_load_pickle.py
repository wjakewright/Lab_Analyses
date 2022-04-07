#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle


def save_pickle(fname, data, path=None):
    """save_pickle - function to save python data as a pickle file
    
        CREATOR
            William (Jake) Wright - 10/15/2021
        
        INPUT PARAMETERS
            fname - string specifying the file name
            
            data - data to be saved in the file
            
            path - path location for where the file is to be saved.
                    Default is none to save in current directory.
    """

    if path is None:
        fname = fname
    else:
        fname = os.path.join(path, fname)

    pickle_fname = fname + ".pickle"
    pickle.dump(data, open(pickle_fname, "wb"))


def load_pickle(fname_list, path=None):
    """load_pickle - a function to load a list of pickled files.
    
        CREATOR
            William (Jake) Wright - 10/15/2021
        
        INPUT PARAMETERS
            fname_list - a list of the file names to be loaded. Names must be
                         formated as strings
        
        OUTPUT PARAMETERS
            loaded_files = a list of the loaded files
            
            """
    if path is None:
        fnames = fname_list
    else:
        fnames = []
        for n in fname_list:
            fn = os.path.join(path, n)
            fnames.append(fn)
    loaded_files = []
    for fname in fnames:
        with open(fname + ".pickle", "rb") as f:
            file = pickle.load(f)
        loaded_files.append(file)

    return loaded_files

