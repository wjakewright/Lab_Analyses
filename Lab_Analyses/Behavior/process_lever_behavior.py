"""Module to process lever press behavior output from ephus and dispatcher"""


import os
import re
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Behavior.dispatcher_to_frames_continuous import (
    dispatcher_to_frames_continuous,
)
from Lab_Analyses.Behavior.load_xsg_continuous import load_xsg_continuous
from Lab_Analyses.Behavior.parse_lever_movement_continuous import (
    parse_lever_movement_continuous,
)
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def process_lever_behavior(mouse_id, path, imaged, save=False, save_suffix=None):
    """Function to process lever press behavior data
    
        INPUT PARAMETERS
            mouse_id - str specifying the id of the mouse

            path - string indicating the path where all of the behavior files
                    are located. Should contain all files from dispatcher and ephus
            
            imaged - boolean True or False indicating if the behavioral session
                    was imaged
            
            save - boolean True or Falsse to save the data at the nedd
                    Optional. Default is set to False
            
            save_suffix - string to be appended at the end of the file name.
                          used to indicated any additional information about the session.
                          Default is set to None
                          
        OUTPUT PARAMETERS
            behavior_data - dataclass object containing the behavior data with fields:

                            mouse_id - str with the id of the mouse

                            sess_name - str with the name of the session

                            date - str with the date the data was collected on

                            dispatcher_data - object containing all the native data

                                            from dispatcher directly loaded from matlab
                            
                            xsg_data - Xsglog_Data object containing xsglog data
                            
                            lever_active - np.array of when lever was active binarized
                            
                            lever_force_resample - np. array of the lever force resampled
                                                    to 1kHz
                            
                            lever_force_smooth - np.array of the resampled lever force smooth
                                                with a butterworth filter
                                                
                            lever_velocity_envelope_smooth - np.array of the lever velocity envelope
                                                            calculated with hilbert transformation
                                                            and then smoothed
                            
                            behavior_frames - np.array containing the data for each behavioral trial.
                                              Data for each trial is stored in an object
                            
                            imaged_trials - logical array indicating which trials imaging
                                            was also performed

                            frame_times - np.array with the time (sec) of each imaging frame
                                    
    """
    # Session name
    sess_name = os.path.basename(path)
    # Load xsg data
    xsg_data = load_xsg_continuous(path)
    if xsg_data is None:
        return None

    # parse the lever movement
    (
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
    ) = parse_lever_movement_continuous(xsg_data)

    # Match behavior to imaging frames
    fnames = os.listdir(path)
    # Get dispatcher filename
    dispatcher_fname = []
    for fname in fnames:
        if "data_@lever2p" in fname:
            dispatcher_fname.append(fname)
    # Make sure only one dispatcher file
    if len(dispatcher_fname) > 1:
        raise Exception(
            "More than one dispatcher file found. Move or delete one of the files"
        )

    # Pull dispatcher data
    if imaged is True:
        (
            dispatcher_data,
            behavior_frames,
            imaged_trials,
            frame_times,
        ) = dispatcher_to_frames_continuous(dispatcher_fname[0], path, xsg_data, imaged)

    else:
        dispatcher_data = dispatcher_to_frames_continuous(
            dispatcher_fname[0], path, xsg_data, imaged
        )
        behavior_frames = np.array([])
        imaged_trials = np.array([])
        frame_times = np.array([])

    date = re.search("[0-9]{6}", dispatcher_fname[0]).group()

    behavior_data = Processed_Lever_Data(
        mouse_id,
        sess_name,
        date,
        dispatcher_data,
        xsg_data,
        lever_active,
        lever_force_resample,
        lever_force_smooth,
        lever_velocity_envelope_smooth,
        behavior_frames,
        imaged_trials,
        frame_times,
    )

    # Save the data
    if save is True:
        # Set the save path
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        save_path = os.path.join(initial_path, mouse_id, "behavior", sess_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # Make file name
        if save_suffix is not None:
            save_name = f"{mouse_id}_{sess_name}_processed_lever_data_{save_suffix}"
        else:
            save_name = f"{mouse_id}_{sess_name}_processed_lever_data"
        # Save the data as a pickle file
        save_pickle(save_name, behavior_data, save_path)

    return behavior_data


# ------------------------------------------------------------------------------------------
# ---------------------------------DATACLASS USED-------------------------------------------
# ------------------------------------------------------------------------------------------


@dataclass
class Processed_Lever_Data:
    """Dataclass for storing the final processed lever behavior data output"""

    mouse_id: str
    sess_name: str
    date: str
    dispatcher_data: object
    xsg_data: object
    lever_active: np.ndarray
    lever_force_resample: np.ndarray
    lever_force_smooth: np.ndarray
    lever_velocity_envelope_smooth: np.ndarray
    behavior_frames: np.ndarray
    imaged_trials: np.ndarray
    frame_times: np.ndarray

