#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

def load_pickle(fname_list):
    '''load_pickle - a function to load a list of pickled files.
    
        CREATOR
            William (Jake) Wright - 10/15/2021
        
        INPUT PARAMETERS
            fname_list - a list of the file names to be loaded. Names must be
                         formated as strings
        
        OUTPUT PARAMETERS
            loaded_files = a list of the loaded files
            
            '''
    loaded_files = []
    for fname in fname_list:
        f = open(fname,'rb')
        file = pickle.load(f)
        loaded_files.append(file)
    
    return loaded_files
    

