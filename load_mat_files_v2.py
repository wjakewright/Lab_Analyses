from os.path import join as pjoin
import scipy.io as sio
import numpy as np

'''Module to load .mat files containing behavior and imaging data into
    a python compatible format
    
    CREATOR
        William (Jake) Wright 12/02/2021'''

def load_mat(fname,fname1=None,path=None):
    '''load_mat - Function to load .mat files into an object that is 
                    compatible with python.

        CREATOR
            William (Jake) Wright - 12/02/2021
            
        USAGE
            behavior_dict = load_behavior_mat_file(fname,fname1=None)

        INPUT PARAMETERs
            fname - string of the name of the file you wish to load.

            fname1 - string of the structure in the file you wish to load if it is different
                     than the file name. Optional, with default set to None.

            path - path where your file will be loaded from. Default path is set to
                   'C:\\Users\\Jake\\Desktop\\Processed_Data. Input path string if you
                   wish to change.

        OUTPUT PARAMETERs
            behavior_obj - Object containing the matlab structure with each field
                            being an attribute of the object. Some attributes are
                            additional objects. Check attributes with obj.__dict__.keys()'''

    # In case the name of the structure in the file doesn't match what it is 
    # saved as. 

    if fname1 == None:
        fname1 = fname
    else:
        fname1 = fname1

    # set the path to load the .mat file
    if path == None:
        p = r'C:\Users\Jake\Desktop\Processed_data\matlab_data\to_convert'
    else:
        p = path
    mat_fname = pjoin(p,fname)
    try:
        mat_file = sio.loadmat(mat_fname,squeeze_me=True,struct_as_record=False)
    except FileNotFoundError:
        print(fname + ' not found')
        return
    obj = mat_file[fname1]
    
    return obj


    

