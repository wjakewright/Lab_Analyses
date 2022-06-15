"""Module for determining if ROIs display significant changes in activity upon
    optogenetic stimulation"""

import os
from itertools import compress

import Lab_Analyses.Utilities.data_utilities as data_utils
import Lab_Analyses.Utilities.test_utilities as test_utils
import numpy as np
import pandas as pd


def classify_opto_response(
    imaging_data,
    behavior_data,
    method,
    window=[-1, 1],
    vis_window=[-2, 3],
    stim_len=1,
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

            method - str specifying which method to use to assess significance
                    Accepts test and shuffle
            
            window - tuple specifying the time before and after opto stim onset you want
                    to analyze. Default set to [-1, 1]
            
            vis_window - same as window, but for how much of the trail you wish to visualize
                        for plotting purposes
            
            stim_len - int specifying how long the opto stimulation is delivered for.
                        Default is set to 1

            z_score - boolean specifying if you wish to z_score the data
                        
            save - boolean specifying if you wish to save the output or not

        OUTPUT PARAMETERS
            opto_response_output - dataclass object of the all the outputs
            
    """

    # Pull out some important variable from the data
    sampling_rate = imaging_data.parameters["Sampling Rate"]
    before_t = window[0]
    after_t = window[1]
    before_f = window[0] * sampling_rate  # in terms of frames
    after_f = window[1] * sampling_rate  # in terms of frames
    vis_before_f = vis_window[0] * sampling_rate
    vis_after_f = vis_window[1] * sampling_rate

    # Get relevant behavioral data
    imaged_trials = behavior_data.imaged_trials == 1
    behavior_frames = list(compress(behavior_data.behavior_frames, imaged_trials))
    stims = []
    for i in behavior_frames:
        stims.append(i.states.iti2)

