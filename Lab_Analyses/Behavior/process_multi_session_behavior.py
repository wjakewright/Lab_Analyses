"""Module for analyzing multiple optogenetic sessions from individual mice"""

import os

from Lab_Analyses.Behavior.process_lever_behavior import process_lever_behavior
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def process_multi_session_behavior(
    mouse_id,
    base_path,
    days=None,
    save=False,
):
    """Function to process behavior data when there are multiple sessions
    in a single day

    INPUT PARAMETERS
        mouse_id - str specifying what the mouse's id is

        base_path - str of the base path containing directories with behavioral
                    data in it

        days - a list of strings specifying if only a subset of days contain opto sessions

        save - boolean specifying if the data is to be saved"""

    # Move to the base behavior directory
    mouse_path = os.path.join(base_path, mouse_id)
    os.chdir(mouse_path)

    # Get each day that will be processed
    if days:
        day_dirs = days
    else:
        day_dirs = [
            os.path.join(mouse_path, name)
            for name in os.listdir(".")
            if os.path.isdir(name)
        ]
    # Start processing each day
    for day in day_dirs:
        # Get all the different sessions for each day
        day_name = os.path.basename(day)
        print(f"                           ", end="\r")
        print(f"- {day_name}")
        sess_dirs = [
            os.path.join(day, name)
            for name in os.listdir(day)
            if os.path.isdir(os.path.join(day, name))
        ]
        # Process each day
        for sess in sess_dirs:
            sess_name = os.path.basename(sess)
            print(f"                           ", end="\r")
            print(f"---{sess_name}", end="\r")
            # Check if there are missing files
            fnames = os.listdir(sess)
            xsg_files = [file for file in os.listdir(sess) if file.endswith(".xsglog")]
            if not xsg_files:
                print(f"Session {sess_name} from day {day_name} missing xsg files")
                continue
            d_present = False
            for fname in fnames:
                if "data_@lever2p" in fname:
                    d_present = True
            if d_present == False:
                print(
                    f"Session {sess_name} from day {day_name} missing dispatcher file"
                )
                continue
            # Process the behavior
            p_file = process_lever_behavior(
                mouse_id, sess, imaged=True, save=False, save_suffix=None
            )

            # Save the output
            if save:
                print("Saving data")
                # Set the save path
                initial_path = r"G:\Analyzed_data\individual"
                save_path = os.path.join(initial_path, mouse_id, "behavior", day_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # Make filename
                save_name = f"{mouse_id}_{day_name}_{sess_name}_processed_lever_data"
                # Save the data as a pickle file
                save_pickle(save_name, p_file, save_path)
