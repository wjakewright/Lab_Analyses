import os


def check_file_exists(path, name, includes):
    """Function to check if a file exists
        
        INPUT PARAMETERS
            path - str to the path where to check if the file exists
            
            name - str of the file name to search for
            
            includes - boolean indicating if the file must match exactly the
                        name input or just included

                        
        OUTPUT PARAMETERS
            exists = boolean indicating if the file exists or not

            exist_files - list of file names that match 
            
    """
    # Get filenames without extensions
    fnames = os.listdir(path)
    exists = False

    exist_files = []
    if includes is True:
        for fname in fnames:
            if name in fname:
                exist_files.append(fname)

    else:
        for fname in fnames:
            if name == fname.split(".")[0]:
                exist_files.append(fname)
    if len(exist_files) > 0:
        exists = True

    return exists, exist_files
