"""Module for loading optogenetic stimulation sessions for opto responsiveness analysis"""


import os
import re

from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def load_opto_sessions(mice, days, keywords=None):
    """Function to load multiple optogenetic stimulation session datasets.
        Can load multiple mice and multiple days
        
        INPUT PARAMETERS
            mice - list of strings with the each mouses id
            
            days - list of lists containing which days to load for each mouse
            
            keywords - a list of keywords that should be in the file names to be loaded
            
        OUTPUT PARAMETERS
    
    """
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    # Initialize the output
    loaded_files = []

    # Go through each mouses data
    for mouse, day in zip(mice, days):
        # Go through each day to be loaded from the mouse
        for d in day:
            b_path = os.path.join(initial_path, mouse, "behavior", d)
            i_path = os.path.join(initial_path, mouse, "imaging", d)
            # check if there is a specific opto directory for that day
            if os.path.isdir(os.path.join(b_path, "opto")):
                b_path = os.path.join(b_path, "opto")
            if os.path.isdir(os.path.join(i_path, "opto")):
                i_path = os.path.join(i_path, "opto")

            all_imaging_files = [x for _, _, x in os.walk(i_path)][0]
            all_behavior_files = [x for _, _, x in os.walk(b_path)][0]

            session_names = []
            session_imaging_files = []
            session_behavior_files = []
            if keywords is None:
                session_names.append(f"{mouse}_{d}")
                session_imaging_files = [all_imaging_files]
                session_behavior_files = [all_behavior_files]
            else:
                for keyword in keywords:
                    sub_i_files = [
                        x
                        for x in all_imaging_files
                        if re.search(f"{keyword}_imaging_data", x)
                    ]
                    sub_b_files = [
                        x
                        for x in all_behavior_files
                        if re.search(f"{keyword}_processed", x)
                    ]
                    session_names.append(f"{mouse}_{d}_{keyword}")
                    session_imaging_files.append(sub_i_files)
                    session_behavior_files.append(sub_b_files)

            # see if there are multiple subsession for each session using different powers
            for image_files, behavior_files in zip(
                session_imaging_files, session_behavior_files
            ):
                if len(image_files) > 1:
                    sub_sessions = []
                    for i in image_files:
                        sub = re.search("[0-9]+mw", i).group()
                        sub_sessions.append(sub)

                    for sub_sess in sub_sessions:
                        i_fname = [x for x in image_files if f"_{sub_sess}" in x]
                        b_fname = [x for x in behavior_files if f"_{sub_sess}" in x]
                        i_file = load_pickle(i_fname, i_path)
                        b_file = load_pickle(b_fname, b_path)
                else:
                    i_file = load_pickle(image_files, i_path)
                    b_file = load_pickle(behavior_files, b_path)
