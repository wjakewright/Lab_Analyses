"""Module to convert dispatcher behavioral data into corresponding imaging frames"""

import copy
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Behavior.read_bit_code import read_bit_code
from Lab_Analyses.Utilities.load_mat_files import load_mat


def dispatcher_to_frames_continuous(file_name, path, xsg_data, imaged):
    """Function to convert dispatcher behavioral data into frames to match
        with imaging data
        
        INPUT PARAMETERS
            file_name - string containing the file name of the dispatcher file
            
            path - string containing the path where the file is located
            
            xsg_data - object containing the data from all the xsglog files. This
                       is output from the load_xsg_continuous function
                       
            imaged - boolean true or false if behavior data was also imaged
        
        OUTPUT PARAMETERS
            dispatcher_data - object containing all the native data from 
                              dispatcher directly loaded from matlab
            
            behavior_frames - object containing the behavioral data converted to
                              match imaging frames
            
            imaged_trials - np.array logical of which trials were imaged
            
            frame_trials - the time (sec) of each image frame
    
    """
    # Constants
    XSG_SAMPLE_RATE = 10000

    # Load the structure within the dispatcher .mat file
    mat_saved = load_mat(fname=file_name, fname1="saved", path=path)
    mat_saved_autoset = load_mat(fname=file_name, fname1="saved_autoset", path=path)
    mat_saved_history = load_mat(fname=file_name, fname1="saved_history", path=path)
    dispatcher_data = Dispatcher_Data(mat_saved, mat_saved_autoset, mat_saved_history)

    # If not imaged return the dispatcher data
    if imaged is False:
        return dispatcher_data

    # Get behavior frames from dispatcher
    bhv_frames = copy.deepcopy(mat_saved_history.ProtocolsSection_parsed_events)
    imaged_trials = np.zeros(len(bhv_frames))

    # Get frame times (sec) from frame trigger traces
    frame_trace = xsg_data.channels["Frame"]
    frame_times = (
        np.nonzero(
            (frame_trace[1:] > 2.5).astype(int) & (frame_trace[:-1] < 2.5).astype(int)
        )[0]
        + 1
    )
    frame_times = (frame_times + 1) / XSG_SAMPLE_RATE

    # Get trials in raw samples since started
    trial_channel = xsg_data.channels["Trial_number"]
    curr_trial_list = read_bit_code(trial_channel)

    # Loop through trials and find the offsets
    for idx, curr_trial in enumerate(curr_trial_list[:, 1]):
        # Skip if it's the last trial and not completed in behavior
        curr_trial = curr_trial.astype(int) - 1
        if curr_trial >= len(bhv_frames) or curr_trial < 0:
            continue

        # The start time is the rise of the first bitcode
        curr_bhv_start = bhv_frames[curr_trial].states.bitcode[0]
        curr_xsg_bhv_offset = curr_bhv_start - curr_trial_list[idx, 0]

        ## Apply offest to all numbers within the trial
        # Find all fields in overal structure of trial
        curr_fieldnames = bhv_frames[curr_trial]._fieldnames

        # Determine which trials are imaged
        bhv_window = (
            bhv_frames[curr_trial].states.state_0 - curr_xsg_bhv_offset
        )  ## Start_time to Stop_time of behavioral trial (sec)

        # Get the frame times within the current behavioral trial window
        a = (frame_times > bhv_window[0, 1]).astype(int)
        b = (frame_times > bhv_window[1, 0]).astype(int)
        imaged_frames = np.round(
            frame_times[np.nonzero(a & b)[0]] * XSG_SAMPLE_RATE
        ).astype(int)

        # Extract the voltage signals indicating whether imaging frames were captured
        frame_trace_window = frame_trace[imaged_frames]

        if np.sum(frame_trace_window):
            imaged_trials[curr_trial] = 1
        else:
            imaged_trials[curr_trial] = 0

        for curr_field in curr_fieldnames:
            # get subfields
            curr_field_data = getattr(bhv_frames[curr_trial], curr_field)
            curr_subfields = curr_field_data._fieldnames
            # find which subfields are numeric
            curr_numeric_subfields = [
                x
                for x in curr_subfields
                if type(getattr(curr_field_data, x)) == np.ndarray
            ]
            # subtract offest from numeric fields and convert to frames
            for s_field in curr_numeric_subfields:
                # pull subfield data
                s_field_data = getattr(curr_field_data, s_field)
                # compensate for offset
                curr_bhv_times = s_field_data - curr_xsg_bhv_offset
                # convert to frames (get thee closest frame from frame time)
                curr_bhv_frames = np.empty(np.shape(curr_bhv_times)).flatten()
                for index, _ in enumerate(curr_bhv_frames):
                    # get index of closest frame
                    curr_bhv_frames[index] = np.argmin(
                        np.absolute(frame_times - curr_bhv_times.flatten()[index])
                    )
                # Update the current subfield value in the object
                curr_bhv_frames = curr_bhv_frames.reshape(np.shape(curr_bhv_times))
                setattr(curr_field_data, s_field, curr_bhv_frames)
            # Update the current field value in the object
            setattr(bhv_frames[curr_trial], curr_field, curr_field_data)

    return dispatcher_data, bhv_frames, imaged_trials, frame_times


# ---------------------------------------------------------------------------------------------
# ------------------------------------DATACLASS USED-------------------------------------------
# ---------------------------------------------------------------------------------------------


@dataclass
class Dispatcher_Data:
    """Dataclass to store the native dispatcher data loaded from Matlab"""

    saved: object
    autoset: object
    saved_history: object
