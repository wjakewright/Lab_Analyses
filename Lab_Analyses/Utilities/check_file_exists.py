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
    try:
        fnames = os.listdir(path)
    except FileNotFoundError:
        exists = False
        exist_files = []
        return exists, exist_files

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


def get_existing_files(path, name, includes):
    """Helper function to check and load existing files"""
    exists, exist_files = check_file_exists(path, name, includes)
    if exists is True:
        if len(exist_files) > 1:
            print("")
            print("More than one matching file exists")
            for n, file in enumerate(exist_files):
                print(f"{n}). {file}")
            fnum = input("Which file number would you like to load: ")
            fnum = fnum - 1
            fname = exist_files[fnum]
        else:
            fname = exist_files[0]
    else:
        fname = None

    return fname
