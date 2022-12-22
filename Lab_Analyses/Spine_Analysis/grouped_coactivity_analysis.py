import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Spine_Analysis.local_spine_coactivity_v2 import (
    local_spine_coactivity_analysis,
)
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
        print("---------------------------------")
        print(f"- Analyzing {mouse}")
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
            print("-- Local Coactivity Analysis")
            (
                distance_bins,
                distance_coactivity_rate,
                distance_coactivity_rate_norm,
                MRS_distance_coactivity_rate,
                MRS_distance_coactivity_rate_norm,
                nMRS_distance_coactivity_rate,
                nMRS_distance_coactivity_rate_norm,
                avg_local_coactivity_rate,
                avg_local_coactivity_rate_norm,
                avg_MRS_local_coactivity_rate,
                avg_MRS_local_coactivity_rate_norm,
                avg_nMRS_local_coactivity_rate,
                avg_nMRS_local_coactivity_rate_norm,
                cluster_score,
                coactive_num,
                MRS_cluster_score,
                MRS_coactive_num,
                nMRS_cluster_score,
                nMRS_coactive_num,
                movement_cluster_score,
                movement_coactive_num,
                nonmovement_cluster_score,
                nonmovement_coactive_num,
                nearby_spine_idxs,
                nearby_coactive_spine_idxs,
                avg_nearby_spine_freq,
                avg_nearby_coactive_spine_freq,
                rel_nearby_spine_freq,
                rel_nearby_coactive_spine_freq,
                frac_nearby_MRSs,
                nearby_coactive_spine_volumes,
                local_coactivity_rate,
                local_coactivity_rate_norm,
                spine_fraction_coactive,
                local_coactivity_matrix,
                spine_coactive_amplitude,
                spine_coactive_calcium,
                spine_coactive_auc,
                spine_coactive_calcium_auc,
                spine_coactive_traces,
                spine_coactive_calcium_traces,
                spine_noncoactive_amplitude,
                spine_noncoactive_calcium,
                spine_noncoactive_auc,
                spine_noncoactive_calcium_auc,
                spine_noncoactive_traces,
                spine_noncoactive_calcium_traces,
                avg_coactive_spine_num,
                sum_nearby_amplitude,
                avg_nearby_amplitude,
                sum_nearby_calcium,
                avg_nearby_calcium,
                sum_nearby_calcium_auc,
                avg_nearby_calcium_auc,
                avg_coactive_num_before,
                sum_nearby_amplitude_before,
                avg_nearby_amplitude_before,
                sum_nearby_calcium_before,
                avg_nearby_calcium_before,
                avg_relative_nearby_onset,
                sum_coactive_binary_traces,
                sum_coactive_spine_traces,
                avg_coactive_spine_traces,
                sum_coactive_calcium_traces,
                avg_coactive_calcium_traces,
                avg_nearby_move_corr,
                avg_nearby_move_reliability,
                avg_nearby_move_specificity,
                avg_nearby_coactivity_rate,
                relative_coactivity_rate,
                frac_local_coactivity_participation,
            ) = local_spine_coactivity_analysis(
                mouse,
                spine_activity,
                spine_dFoF,
                spine_calcium,
                spine_groupings,
                spine_flags,
                spine_volume_um,
                spine_positions,
                movement_spines,
                non_movement_spines,
                lever_active,
                lever_unactive,
                lever_force_smooth,
                activity_window,
                CLUSTER_DIST,
                sampling_rate,
                volume_norm=constants,
            )

