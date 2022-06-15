"""Module for determining if ROIs display significant changes in activity upon
    optogenetic stimulation"""

import os
from dataclasses import dataclass
from itertools import compress

import Lab_Analyses.Optogenetics.opto_plotting as plotting
import Lab_Analyses.Utilities.data_utilities as data_utils
import Lab_Analyses.Utilities.test_utilities as test_utils
import numpy as np
import pandas as pd
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def classify_opto_response(
    imaging_data,
    behavior_data,
    session_type,
    method,
    window=[-1, 1],
    vis_window=[-2, 3],
    stim_len=1,
    processed_dFoF=False,
    z_score=False,
    save=False,
):
    """Function to determine if ROIs display significant chagnes in activity upon optogenetic
        stimulation. Performed on a single mouse and single session
        
        INPUT PARAMETERS
            imaging_data - dataclass object of the Activity_Output from the 
                           Activity_Viewer for each mouse.
                           
            behavior_data - dataclass object of the Processed_Lever_Data
                            output from process_lever_behavior.

            session_type - str specifying what type of opto session it is. Used
                            to know where in the behavioral data the opto stim occured

            method - str specifying which method to use to assess significance
                    Accepts test and shuffle
            
            window - tuple specifying the time before and after opto stim onset you want
                    to analyze. Default set to [-1, 1]
            
            vis_window - same as window, but for how much of the trail you wish to visualize
                        for plotting purposes
            
            stim_len - int specifying how long the opto stimulation is delivered for.
                        Default is set to 1

            processed_dFoF - boolean specifying to use the processed dFoF instead of dFoF

            z_score - boolean specifying if you wish to z_score the data
                        
            save - boolean specifying if you wish to save the output or not

        OUTPUT PARAMETERS
            opto_response_output - dataclass object of the all the outputs
            
    """
    # Constant of what types of ROIs are acceptible for analysis
    ROI_TYPES = ["Soma", "Spine", "Dendrite"]

    # Pull out some important variable from the data
    sampling_rate = imaging_data.parameters["Sampling Rate"]
    before_t = window[0]
    after_t = window[1]
    before_f = window[0] * sampling_rate  # in terms of frames
    after_f = window[1] * sampling_rate  # in terms of frames
    vis_before_f = vis_window[0] * sampling_rate
    vis_after_f = vis_window[1] * sampling_rate
    session_len = len(imaging_data.fluorescence["Background"])

    # Get relevant behavioral data
    imaged_trials = behavior_data.imaged_trials == 1
    behavior_frames = list(compress(behavior_data.behavior_frames, imaged_trials))
    stims = []
    if session_type == "pulsed":
        for i in behavior_frames:
            stims.append(i.states.iti2)
        # Determine stimulation length from the iti durations
        stim_len = np.nanmedian([x[1] - x[2] for x in stims])
    else:
        return print("Only coded for pulsed trials right now")
    ## Check stimulation intervals are within imaging period
    longest_win = max([after_f + stim_len, vis_after_f])
    for idx, i in enumerate(stims):
        if i[1] + longest_win > session_len:
            stims = stims[:idx]

    # Oranize the different ROI types together
    ROI_ids = []
    dFoF = np.zeros(session_len).reshape(-1, 1)
    if processed_dFoF is False:
        for key, value in imaging_data.dFoF.items():
            ids = [f"{key} {x+1}" for x in np.arange(value.shape[1])]
            [ROI_ids.append(x) for x in ids]
            dFoF = np.hstack((dFoF, value))
    else:
        for key, value in imaging_data.processed_dFoF.items():
            ids = [f"{key} {x+1}" for x in np.arange(value.shape[1])]
            [ROI_ids.append(x) for x in ids]
            dFoF = np.hstack((dFoF, value))

    dFoF = dFoF[:, 1:]  # Getting rid of the initialized zeros

    if z_score is True:
        dFoF = test_utils.z_score(dFoF)

    # Start analyzing the data
    befores, afters = data_utils.get_before_after_means(
        activity=dFoF,
        timestamps=stims,
        window=window,
        sampling_rate=sampling_rate,
        offset=False,
        single=False,
    )
    new_stims = [i[0] for i in stims]
    roi_stims, roi_means = data_utils.get_trace_mean_sem(
        activity=dFoF,
        timestamps=new_stims,
        window=vis_window,
        sampling_rate=sampling_rate,
        single=False,
    )
    roi_stims = list(roi_stims.values())
    roi_means = list(roi_means.values())
    results_dict, _, _ = test_utils.response_testing(
        imaging=dFoF,
        timestamps=stims,
        window=window,
        sampling_rate=sampling_rate,
        method=method,
    )

    # Put results into the output
    opto_response_output = Opto_Repsonses(
        mouse_id=behavior_data.mouse_id,
        session=behavior_data.sess_name,
        date=behavior_data.date,
        ROI_ids=ROI_ids,
        dFoF=dFoF,
        stims=stims,
        befores=befores,
        afters=afters,
        roi_stims=roi_stims,
        roi_means=roi_means,
        results=results_dict,
    )

    # Save section
    if save is True:
        # Set the save path
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        save_path = os.path.join(
            initial_path, behavior_data.mouse_id, "behavior", behavior_data.date
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Set filename
        save_name = f"{behavior_data.mouse_id}_{behavior_data.date}_{behavior_data.date}_opto_response"
        # Save the data as a pickle file
        save_pickle(save_name, opto_response_output, save_path)

    return opto_response_output


#################### DATACLASSES #######################
@dataclass
class Opto_Repsonses:
    mouse_id: str
    session: str
    date: str
    ROI_ids: list
    dFoF: np.array
    stims: list
    befores: list
    afters: list
    roi_stims: list
    roi_means: list
    results: dict

    def display_results(self, figsizes=None, parameters=None):
        """Method to make plots and display the data
        
            INPUT PARAMETERS
                figsizes - list containing tuples for how big each figures should be
                
                parameters - dict containg the following paramters:
                                title - str
                                hmap_range - tuple
                                cmap - str
                                zeroed - bool
                                sort - bool
                                center - int
        """
        if parameters is None:
            parameters = {
                "title": "default",
                "hmap_range": None,
                "cmap": None,
                "zeroed": False,
                "sort": False,
                "center": None,
            }
        if figsizes is None:
            figsizes = [(7, 8), (10, 10), (10, 10), (4, 5), (10, 10)]

