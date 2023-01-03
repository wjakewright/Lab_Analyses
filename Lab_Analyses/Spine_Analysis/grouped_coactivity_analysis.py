import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Spine_Analysis.dendrite_spine_coactivity_analysis import (
    dendrite_spine_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.local_spine_coactivity_v2 import (
    local_spine_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    batch_spine_volume_norm_constant,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_related_activity import movement_related_activity
from Lab_Analyses.Utilities.quantify_movment_quality import quantify_movement_quality


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
                curr_glu_constants = None
                curr_ca_constants = None
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
                    rwd_movement_dendrites[s] = data.reward_movement_dendrites[d]
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
                spine_cluster_score,
                spine_coactive_num,
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
                local_spine_fraction_coactive,
                local_coactivity_matrix,
                local_spine_coactive_amplitude,
                local_spine_coactive_calcium,
                local_spine_coactive_auc,
                local_spine_coactive_calcium_auc,
                local_spine_coactive_traces,
                local_spine_coactive_calcium_traces,
                local_spine_noncoactive_amplitude,
                local_spine_noncoactive_calcium,
                local_spine_noncoactive_auc,
                local_spine_noncoactive_calcium_auc,
                local_spine_noncoactive_traces,
                local_spine_noncoactive_calcium_traces,
                local_avg_coactive_spine_num,
                local_sum_nearby_amplitude,
                local_avg_nearby_amplitude,
                local_sum_nearby_calcium,
                local_avg_nearby_calcium,
                local_sum_nearby_calcium_auc,
                local_avg_nearby_calcium_auc,
                local_avg_coactive_num_before,
                local_sum_nearby_amplitude_before,
                local_avg_nearby_amplitude_before,
                local_sum_nearby_calcium_before,
                local_avg_nearby_calcium_before,
                local_avg_relative_nearby_onset,
                local_sum_coactive_binary_traces,
                local_sum_coactive_spine_traces,
                local_avg_coactive_spine_traces,
                local_sum_coactive_calcium_traces,
                local_avg_coactive_calcium_traces,
                avg_nearby_move_corr,
                avg_nearby_move_reliability,
                avg_nearby_move_specificity,
                avg_nearby_coactivity_rate,
                relative_local_coactivity_rate,
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

            # Perform Dendrite-Spine coactivity analysis
            print("-- Spine-Dendrite Coactivity Analysis")
            (
                spine_dend_coactivity_matrix,
                spine_dend_coactivity_rate,
                spine_dend_coactivity_rate_norm,
                spine_dend_spine_fraction_coactive,
                spine_dend_dend_fraction_coactive,
                spine_dend_spine_coactive_amplitude,
                spine_dend_spine_coactive_calcium,
                spine_dend_spine_coactive_auc,
                spine_dend_spine_coactive_calcium_auc,
                spine_dend_dend_coactive_amplitude,
                spine_dend_dend_coactive_auc,
                spine_dend_relative_onset,
                spine_dend_spine_coactive_traces,
                spine_dend,
                _spine_coactive_calcium_traces,
                spine_dend_dend_coactive_traces,
                conj_coactivity_matrix,
                conj_coactivity_rate,
                conj_coactivity_rate_norm,
                conj_spine_fraction_coactive,
                conj_dend_fraction_coactive,
                conj_spine_coactive_amplitude,
                conj_spine_coactive_calcium,
                conj_spine_coactive_auc,
                conj_spine_coactive_calcium_auc,
                conj_dend_coactive_amplitude,
                conj_dend_coactive_auc,
                conj_relative_onset,
                conj_spine_coactive_traces,
                conj_spine_coactive_calcium_traces,
                conj_dend_coactive_traces,
                nonconj_coactivity_matrix,
                nonconj_coactivity_rate,
                nonconj_coactivity_rate_norm,
                nonconj_spine_fraction_coactive,
                nonconj_dend_fraction_coactive,
                nonconj_spine_coactive_amplitude,
                nonconj_spine_coactive_calcium,
                nonconj_spine_coactive_auc,
                nonconj_spine_coactive_calcium_auc,
                nonconj_dend_coactive_amplitude,
                nonconj_dend_coactive_auc,
                nonconj_relative_onset,
                nonconj_spine_coactive_traces,
                nonconj_spine_coactive_calcium_traces,
                nonconj_dend_coactive_traces,
                spine_dend_distance_coactivity_rate,
                spine_dend_distance_coactivity_rate_norm,
                spine_dend_avg_local_coactivity_rate,
                spine_dend_avg_local_coactivity_rate_norm,
                spine_dend_cluster_score,
                spine_dend_coactive_num,
            ) = dendrite_spine_coactivity_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium,
                dendrite_activity,
                dendrite_dFoF,
                spine_groupings,
                spine_flags,
                spine_positions,
                activity_window=activity_window,
                cluster_dist=CLUSTER_DIST,
                sampling_rate=sampling_rate,
                volume_norm=volume_norm,
            )

            # Analyze movement-related activity
            print("-- Movement-Related Activity Analysis")
            ## Spines
            (
                move_spine_traces,
                move_spine_amplitude,
                move_spine_onset,
            ) = movement_related_activity(
                lever_active,
                spine_dFoF,
                norm=curr_glu_constants,
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            ## Dendrites
            (
                move_dend_traces,
                move_dend_amplitude,
                move_dend_onset,
            ) = movement_related_activity(
                lever_active,
                dendrite_dFoF,
                norm=None,
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )

            # Get movement quality encoding for spines and dendrites
            print("-- Assessing Movement Quality")
            ## Spines
            (
                _,
                spine_movements,
                _,
                spine_move_correlation,
                spine_move_reliability,
                _,
                _,
                spine_move_specificity,
                learned_movement,
            ) = quantify_movement_quality(
                mouse,
                spine_activity,
                lever_active,
                lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            ## Dendrites
            (
                _,
                dend_movements,
                _,
                dend_move_correlation,
                dend_move_reliability,
                _,
                _,
                dend_move_specificity,
                _,
            ) = quantify_movement_quality(
                mouse,
                dendrite_activity,
                lever_active,
                lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            ## Local Coactivity
            (
                _,
                local_movements,
                _,
                local_move_correlation,
                local_move_reliability,
                _,
                _,
                local_move_specificity,
                _,
            ) = quantify_movement_quality(
                mouse,
                local_coactivity_matrix,
                lever_active,
                lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            ## Spine-Dendrite Coactivity
            (
                _,
                spine_dend_movements,
                _,
                spine_dend_move_correlation,
                spine_dend_move_reliability,
                _,
                _,
                spine_dend_move_specificity,
                _,
            ) = quantify_movement_quality(
                mouse,
                spine_dend_coactivity_matrix,
                lever_active,
                lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )

