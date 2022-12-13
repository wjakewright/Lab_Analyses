import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Spine_Analysis.spine_utilities import (
    batch_spine_volume_norm_constant,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def grouped_coactivity_analysis(
    mice_list,
    day,
    followup=True,
    activity_window=(-2, 4),
    zscore=False,
    volume_norm=False,
    save=False,
    save_path=None,
):
    """Function to handle spine coactivity analysis of dual spine imaging datasets
        across all mice and all FOVs. Stores coactivity data in dataclass to be used
        for subsequent analyses (e.g., plasticity)
        
        INPUT PARAMETERS
            mice_list - list of str specifying mice ids to be analyzed
            
            day - str of the day to be analyzed
            
            followup - boolean specifying if you want to include followup
                        structural imaging sessions
            
            zscore - boolean of whether or not to zscore activity for analysis
            
            save - boolean of whether to save the output or not
            
            save_path - str sepcifying where to save the data
    """

    CLUSTER_DIST = 5
    grouped_data = defaultdict(list)

    if volume_norm:
        glu_constants = batch_spine_volume_norm_constant(mice_list, day, "GluSnFr")
        ca_constants = batch_spine_volume_norm_constant(mice_list, day, "Calcium")

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"--- Analyzing {mouse}")
        if followup is True:
            datasets = load_spine_datasets(mouse, [day], followup=True)
        else:
            datasets = load_spine_datasets(mouse, [day], followup=False)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
            # Set up main dataset and pull relevant followup information
            if followup is True:
                data = dataset[f"Pre {day}"]
                followup_volume = np.array(
                    dataset[f"Post {day}"].corrected_spine_volume
                )
                followup_flags = dataset[f"Post {day}"].spine_flags
            else:
                data = dataset[day]
                followup_volume = None
                followup_flags = None

            # Get corresponding volume_norm constants if relevant
            if volume_norm:
                curr_glu_constants = glu_constants[mouse][FOV]
                curr_ca_constants = ca_constants[mouse][FOV]
                constants = (curr_glu_constants, curr_ca_constants)
            else:
                constants = None

            # Pull relevant data
            ## Imaging and spine group parameters
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            zoom_factor = data.imaging_parameters["Zoom"]
            spine_groupings = data.spine_grouping
            spine_flags = data.spine_flags
            spine_volume = np.array(data.corrected_spine_volume)
            spine_positions = np.array(data.spine_positions)

            ## Spine activity and movement encoding
            spine_activity = data.spine_GluSnFr_activity
            spine_dFoF = data.spine_GluSnFr_processed_dFoF
            spine_calcium = data.spine_calcium_processed_dFoF
            movement_spines = data.movement_spines
            non_movement_spines = [not x for x in movement_spines]
            rwd_movement_spines = data.reward_movement_spines

            ## Dendrite activity and movement encoding
            dendrite_activity = np.zeros(spine_activity.shape)
            dendrite_dFoF = np.zeros(spine_dFoF.shape)
            movement_dendrites = np.zeros(movement_spines.shape).astype(bool)
            rwd_movement_dendrites = np.zeros(rwd_movement_spines.shape).astype(bool)
            for d in range(data.dendrite_calcium_activity.shape[1]):
                if type(spine_groupings[d]) == list:
                    spines = spine_groupings[d]
                else:
                    spines = spine_groupings
                for s in spines:
                    dendrite_activity[:, s] = data.dendrite_calcium_activity[:, d]
                    dendrite_dFoF[:, s] = data.dendrite_calcium_processed_dFoF[:, d]
                    movement_dendrites[s] = data.movement_dendrites[d]
                    rwd_movement_dendrites[s] = data.reward_movement_dendrites
            non_movement_dendrites = [not x for x in movement_dendrites]

            ## Behavioral data
            lever_active = data.lever_active
            lever_force_smooth = data.lever_force_smooth
            lever_active_rwd = data.rewarded_movement_binary
            lever_active_non_rwd = lever_active - lever_active_rwd
            lever_unactive = np.absolute(lever_active - 1)

            # zscore activity if specified
            if zscore:
                spine_dFoF = d_utils.z_score(spine_dFoF)
                spine_calcium = d_utils.z_score(spine_calcium)
                dendrite_dFoF = d_utils.z_score(dendrite_dFoF)

            # Get volumes in um
            pix_to_um = zoom_factor / 2
            spine_volume_um = (np.sqrt(spine_volume) / pix_to_um) ** 2
            followup_volume_um = (np.sqrt(followup_volume) / pix_to_um) ** 2

            # Get activity frequencies
            spine_activity_rate = d_utils.calculate_activity_event_rate(spine_activity)
            dend_activity_rate = d_utils.calculate_activity_event_rate(
                dendrite_activity
            )

            # Perform local spine coactivity analysis

