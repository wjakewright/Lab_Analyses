import os
from collections import defaultdict

import numpy as np

from Lab_Analyses.Spine_Analysis.dendrite_spine_coactivity_analysis import (
    dendrite_spine_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.local_spine_coactivity_v2 import (
    local_spine_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.spine_coactivity_dataclass import Spine_Coactivity_Data
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    batch_spine_volume_norm_constant,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_related_activity import movement_related_activity
from Lab_Analyses.Utilities.quantify_movment_quality import (
    quantify_movement_quality,
    spine_dendrite_movement_similarity,
)


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
            rwd_nonmovement_spines = [not x for x in rwd_movement_spines]

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
            rwd_nonmovement_dendrites = [not x for x in rwd_movement_dendrites]

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
                spine_dend_spine_coactive_calcium_traces,
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

            learned_movement_pattern = [
                learned_movement for i in range(spine_activity.shape[1])
            ]

            # Compare movement encoding between spines and parent dendrite
            rel_spine_vs_dend_move_corr = spine_move_correlation - dend_move_correlation
            (
                spine_to_dend_correlation,
                spine_to_nearby_correlation,
            ) = spine_dendrite_movement_similarity(
                spine_movements, dend_movements, nearby_spine_idxs,
            )

            # Generate FOV and Mouse ID lists
            fovs = [FOV for i in range(spine_activity.shape[1])]
            ids = [mouse for i in range(spine_activity.shape[1])]

            # Store data from this mouse in grouped_data dictionary
            ## Adding general variables
            grouped_data["mouse_id"].append(ids)
            grouped_data["FOVs"].append(fovs)
            grouped_data["spine_flags"].append(spine_flags)
            grouped_data["followup_flags"].append(followup_flags)
            grouped_data["spine_volumes"].append(spine_volume)
            grouped_data["spine_volumes_um"].append(spine_volume_um)
            grouped_data["followup_volumes"].append(followup_volume)
            grouped_data["followup_volumes_um"].append(followup_volume_um)
            grouped_data["movement_spines"].append(movement_spines)
            grouped_data["nonmovement_spines"].append(non_movement_spines)
            grouped_data["rwd_movement_spines"].append(rwd_movement_spines)
            grouped_data["rwd_nonmovement_spines"].append(rwd_nonmovement_spines)
            grouped_data["movement_dendrites"].append(movement_dendrites)
            grouped_data["nonmovement_dendrites"].append(non_movement_dendrites)
            grouped_data["rwd_movement_dendrites"].append(rwd_movement_dendrites)
            grouped_data["rwd_nonmovement_dendrites"].append(rwd_nonmovement_dendrites)
            grouped_data["spine_activity_rate"].append(spine_activity_rate)
            grouped_data["dend_activity_rate"].append(dend_activity_rate)
            ## Adding local coactivity variables
            grouped_data["distance_coactivity_rate"].append(distance_coactivity_rate)
            grouped_data["distance_coactivity_rate_norm"].append(
                distance_coactivity_rate_norm
            )
            grouped_data["MRS_distance_coactivity_rate"].append(
                MRS_distance_coactivity_rate
            )
            grouped_data["MRS_distance_coactivity_rate_norm"].append(
                MRS_distance_coactivity_rate_norm
            )
            grouped_data["nMRS_distance_coactivity_rate"].append(
                nMRS_distance_coactivity_rate
            )
            grouped_data["nMRS_distance_coactivity_rate_norm"].append(
                nMRS_distance_coactivity_rate_norm
            )
            grouped_data["avg_local_coactivity_rate"].append(avg_local_coactivity_rate)
            grouped_data["avg_local_coactivity_rate_norm"].append(
                avg_local_coactivity_rate_norm
            )
            grouped_data["avg_MRS_local_coactivity_rate"].append(
                avg_MRS_local_coactivity_rate
            )
            grouped_data["avg_MRS_local_coactivity_rate_norm"].append(
                avg_MRS_local_coactivity_rate_norm
            )
            grouped_data["avg_nMRS_local_coactivity_rate"].append(
                avg_nMRS_local_coactivity_rate
            )
            grouped_data["avg_nMRS_local_coactivity_rate_norm"].append(
                avg_nMRS_local_coactivity_rate_norm
            )
            grouped_data["spine_cluster_score"].append(spine_cluster_score)
            grouped_data["spine_coactive_num"].append(spine_coactive_num)
            grouped_data["MRS_cluster_score"].append(MRS_cluster_score)
            grouped_data["MRS_coactive_num"].append(MRS_coactive_num)
            grouped_data["nMRS_cluster_score"].append(nMRS_cluster_score)
            grouped_data["nMRS_coactive_num"].append(nMRS_coactive_num)
            grouped_data["movement_cluster_score"].append(movement_cluster_score)
            grouped_data["movement_coactive_num"].append(movement_coactive_num)
            grouped_data["nonmovement_cluster_score"].append(nonmovement_cluster_score)
            grouped_data["nonmovement_coactive_num"].append(nonmovement_coactive_num)
            grouped_data["avg_nearby_spine_rate"].append(avg_nearby_spine_freq)
            grouped_data["avg_nearby_coactive_spine_rate"].append(
                avg_nearby_coactive_spine_freq
            )
            grouped_data["rel_nearby_spine_rate"].append(rel_nearby_spine_freq)
            grouped_data["rel_nearby_coactive_spine_rate"].append(
                rel_nearby_coactive_spine_freq
            )
            grouped_data["frac_nearby_MRSs"].append(frac_nearby_MRSs)
            grouped_data["nearby_coactive_spine_volumes"].append(
                nearby_coactive_spine_volumes
            )
            grouped_data["local_coactivity_rate"].append(local_coactivity_rate)
            grouped_data["local_coactivity_rate_norm"].append(
                local_coactivity_rate_norm
            )
            grouped_data["local_spine_fraction_coactive"].append(
                local_spine_fraction_coactive
            )
            grouped_data["local_spine_coactive_amplitude"].append(
                local_spine_coactive_amplitude
            )
            grouped_data["local_spine_coactive_calcium"].append(
                local_spine_coactive_calcium
            )
            grouped_data["local_spine_coactive_auc"].append(local_spine_coactive_auc)
            grouped_data["local_spine_coactive_calcium_auc"].append(
                local_spine_coactive_calcium_auc
            )
            grouped_data["local_spine_coactive_traces"].append(
                local_spine_coactive_traces
            )
            grouped_data["local_spine_coactive_calcium_traces"].append(
                local_spine_coactive_calcium_traces
            )
            grouped_data["local_spine_noncoactive_amplitude"].append(
                local_spine_noncoactive_amplitude
            )
            grouped_data["local_spine_noncoactive_calcium"].append(
                local_spine_noncoactive_calcium
            )
            grouped_data["local_spine_noncoactive_auc"].append(
                local_spine_noncoactive_auc
            )
            grouped_data["local_spine_noncoactive_calcium_auc"].append(
                local_spine_noncoactive_calcium_auc
            )
            grouped_data["local_spine_noncoactive_traces"].append(
                local_spine_noncoactive_traces
            )
            grouped_data["local_spine_noncoactive_calcium_traces"].append(
                local_spine_noncoactive_calcium_traces
            )
            grouped_data["local_avg_coactive_spine_num"].append(
                local_avg_coactive_spine_num
            )
            grouped_data["local_sum_nearby_amplitude"].append(
                local_sum_nearby_amplitude
            )
            grouped_data["local_avg_nearby_amplitude"].append(
                local_avg_nearby_amplitude
            )
            grouped_data["local_sum_nearby_calcium"].append(local_sum_nearby_calcium)
            grouped_data["local_avg_nearby_calcium"].append(local_avg_nearby_calcium)
            grouped_data["local_sum_nearby_calcium_auc"].append(
                local_sum_nearby_calcium_auc
            )
            grouped_data["local_avg_nearby_calcium_auc"].append(
                local_avg_nearby_calcium_auc
            )
            grouped_data["local_avg_coactive_num_before"].append(
                local_avg_coactive_num_before
            )
            grouped_data["local_sum_nearby_amplitude_before"].append(
                local_sum_nearby_amplitude_before
            )
            grouped_data["local_avg_nearby_amplitude_before"].append(
                local_avg_nearby_amplitude_before
            )
            grouped_data["local_sum_nearby_calcium_before"].append(
                local_sum_nearby_calcium_before
            )
            grouped_data["local_avg_nearby_calcium_before"].append(
                local_avg_nearby_calcium_before
            )
            grouped_data["local_avg_relative_nearby_onset"].append(
                local_avg_relative_nearby_onset
            )
            grouped_data["local_sum_nearby_binary_traces"].append(
                local_sum_coactive_binary_traces
            )
            grouped_data["local_sum_nearby_spine_traces"].append(
                local_sum_coactive_spine_traces
            )
            grouped_data["local_avg_nearby_spine_traces"].append(
                local_avg_coactive_spine_traces
            )
            grouped_data["local_sum_nearby_calcium_traces"].append(
                local_sum_coactive_calcium_traces
            )
            grouped_data["local_avg_nearby_calcium_traces"].append(
                local_avg_coactive_calcium_traces
            )
            grouped_data["avg_nearby_movement_correlation"].append(avg_nearby_move_corr)
            grouped_data["avg_nearby_movement_reliability"].append(
                avg_nearby_move_reliability
            )
            grouped_data["avg_nearby_movement_specificity"].append(
                avg_nearby_move_specificity
            )
            grouped_data["avg_nearby_coactivity_rate"].append(
                avg_nearby_coactivity_rate
            )
            grouped_data["relative_local_coactivity_rate"].append(
                relative_local_coactivity_rate
            )
            grouped_data["frac_local_coactivity_participation"].append(
                frac_local_coactivity_participation
            )
            ## Adding spine-dendrite coactivity variables
            grouped_data["spine_dend_coactivity_rate"].append(
                spine_dend_coactivity_rate
            )
            grouped_data["spine_dend_coactivity_rate_norm"].append(
                spine_dend_coactivity_rate_norm
            )
            grouped_data["spine_dend_spine_fraction_coactive"].append(
                spine_dend_spine_fraction_coactive
            )
            grouped_data["spine_dend_dend_fraction_coactive"].append(
                spine_dend_dend_fraction_coactive
            )
            grouped_data["spine_dend_spine_coactive_amplitude"].append(
                spine_dend_spine_coactive_amplitude
            )
            grouped_data["spine_dend_spine_coactive_calcium"].append(
                spine_dend_spine_coactive_calcium
            )
            grouped_data["spine_dend_spine_coactive_auc"].append(
                spine_dend_spine_coactive_auc
            )
            grouped_data["spine_dend_spine_coactive_calcium_auc"].append(
                spine_dend_spine_coactive_calcium_auc
            )
            grouped_data["spine_dend_dend_coactive_amplitude"].append(
                spine_dend_dend_coactive_amplitude
            )
            grouped_data["spine_dend_dend_coactive_auc"].append(
                spine_dend_dend_coactive_auc
            )
            grouped_data["spine_dend_relative_onset"].append(spine_dend_relative_onset)
            grouped_data["spine_dend_spine_coactive_traces"].append(
                spine_dend_spine_coactive_traces
            )
            grouped_data["spine_dend_spine_coactive_calcium_traces"].append(
                spine_dend_spine_coactive_calcium_traces
            )
            grouped_data["spine_dend_dend_coactive_traces"].append(
                spine_dend_dend_coactive_traces
            )
            grouped_data["conj_coactivity_rate"].append(conj_coactivity_rate)
            grouped_data["conj_coactivity_rate_norm"].append(conj_coactivity_rate_norm)
            grouped_data["conj_spine_fraction_coactive"].append(
                conj_spine_fraction_coactive
            )
            grouped_data["conj_dend_fraction_coactive"].append(
                conj_dend_fraction_coactive
            )
            grouped_data["conj_spine_coactive_amplitude"].append(
                conj_spine_coactive_amplitude
            )
            grouped_data["conj_spine_coactive_calcium"].append(
                conj_spine_coactive_calcium
            )
            grouped_data["conj_spine_coactive_auc"].append(conj_spine_coactive_auc)
            grouped_data["conj_spine_coactive_calcium_auc"].append(
                conj_spine_coactive_calcium_auc
            )
            grouped_data["conj_dend_coactive_amplitude"].append(
                conj_dend_coactive_amplitude
            )
            grouped_data["conj_dend_coactive_auc"].append(conj_dend_coactive_auc)
            grouped_data["conj_relative_onset"].append(conj_relative_onset)
            grouped_data["conj_spine_coactive_traces"].append(
                conj_spine_coactive_traces
            )
            grouped_data["conj_spine_coactive_calcium_traces"].append(
                conj_spine_coactive_calcium_traces
            )
            grouped_data["conj_dend_coactive_traces"].append(conj_dend_coactive_traces)
            grouped_data["nonconj_coactivity_rate"].append(nonconj_coactivity_rate)
            grouped_data["nonconj_coactivity_rate_norm"].append(
                nonconj_coactivity_rate_norm
            )
            grouped_data["nonconj_spine_fraction_coactive"].append(
                nonconj_spine_fraction_coactive
            )
            grouped_data["nonconj_dend_fraction_coactive"].append(
                nonconj_dend_fraction_coactive
            )
            grouped_data["nonconj_spine_coactive_amplitude"].append(
                nonconj_spine_coactive_amplitude
            )
            grouped_data["nonconj_spine_coactive_calcium"].append(
                nonconj_spine_coactive_calcium
            )
            grouped_data["nonconj_spine_coactive_auc"].append(
                nonconj_spine_coactive_auc
            )
            grouped_data["nonconj_spine_coactive_calcium_auc"].append(
                nonconj_spine_coactive_calcium_auc
            )
            grouped_data["nonconj_dend_coactive_amplitude"].append(
                nonconj_dend_coactive_amplitude
            )
            grouped_data["nonconj_dend_coactive_auc"].append(nonconj_dend_coactive_auc)
            grouped_data["nonconj_relative_onset"].append(nonconj_relative_onset)
            grouped_data["nonconj_spine_coactive_traces"].append(
                nonconj_spine_coactive_traces
            )
            grouped_data["nonconj_spine_coactive_calcium_traces"].append(
                nonconj_spine_coactive_calcium_traces
            )
            grouped_data["nonconj_dend_coactive_traces"].append(
                nonconj_dend_coactive_traces
            )
            grouped_data["spine_dend_distance_coactivity_rate"].append(
                spine_dend_distance_coactivity_rate
            )
            grouped_data["spine_dend_distance_coactivity_rate_norm"].append(
                spine_dend_distance_coactivity_rate_norm
            )
            grouped_data["spine_dend_avg_local_coactivity_rate"].append(
                spine_dend_avg_local_coactivity_rate
            )
            grouped_data["spine_dend_avg_local_coactivity_rate_norm"].append(
                spine_dend_avg_local_coactivity_rate_norm
            )
            grouped_data["spine_dend_cluster_score"].append(spine_dend_cluster_score)
            grouped_data["spine_dend_coactive_num"].append(spine_dend_coactive_num)
            ## Adding movement-related activity variables
            grouped_data["movement_spine_traces"].append(move_spine_traces)
            grouped_data["movement_spine_amplitude"].append(move_spine_amplitude)
            grouped_data["movement_spine_onset"].append(move_spine_onset)
            grouped_data["movement_dend_traces"].append(move_dend_traces)
            grouped_data["movement_dend_amplitude"].append(move_dend_amplitude)
            grouped_data["movement_dend_onset"].append(move_dend_onset)
            ## Adding movement quality variables
            grouped_data["learned_movement_pattern"].append(learned_movement_pattern)
            grouped_data["spine_movements"].append(spine_movements)
            grouped_data["spine_movement_correlation"].append(spine_move_correlation)
            grouped_data["spine_movement_reliability"].append(spine_move_reliability)
            grouped_data["spine_movement_specificity"].append(spine_move_specificity)
            grouped_data["dend_movements"].append(dend_movements)
            grouped_data["dend_movement_correlation"].append(dend_move_correlation)
            grouped_data["dend_movement_reliability"].append(dend_move_reliability)
            grouped_data["dend_movement_specificity"].append(dend_move_specificity)
            grouped_data["local_movements"].append(local_movements)
            grouped_data["local_movement_correlation"].append(local_move_correlation)
            grouped_data["local_movement_reliability"].append(local_move_reliability)
            grouped_data["local_movement_specificity"].append(local_move_specificity)
            grouped_data["spine_dend_movements"].append(spine_dend_movements)
            grouped_data["spine_dend_movement_correlation"].append(
                spine_dend_move_correlation
            )
            grouped_data["spine_dend_movement_reliability"].append(
                spine_dend_move_reliability
            )
            grouped_data["spine_dend_movement_specificity"].append(
                spine_dend_move_specificity
            )
            grouped_data["rel_spine_vs_dend_move_corr"].append(
                rel_spine_vs_dend_move_corr
            )
            grouped_data["spine_to_dend_correlation"].append(spine_to_dend_correlation)
            grouped_data["spine_to_nearby_correlation"].append(
                spine_to_nearby_correlation
            )

    # Merge all the data across FOVs and mice
    regrouped_data = {}
    for key, value in grouped_data.items():
        if type(value[0]) == list:
            regrouped_data[key] = [y for x in value for y in x]
        elif type(value[0]) == np.ndarray:
            if len(value[0].shape) == 1:
                regrouped_data[key] = np.concatenate(value)
            elif len(value[0].shape) == 2:
                regrouped_data[key] = np.hstack(value)
        else:
            print(f"{key} has not been added!!!")

    parameters = {
        "Sampling Rate": sampling_rate,
        "Cluster Dist": CLUSTER_DIST,
        "Distance Bins": distance_bins,
        "zscore": zscore,
        "Volume Norm": volume_norm,
        "Activity Window": activity_window,
    }

    # Store data in dataclass for outputting and saving
    spine_coactivity_data = Spine_Coactivity_Data(
        day=day,
        mouse_id=regrouped_data["mouse_id"],
        FOV=regrouped_data["FOVs"],
        paramters=parameters,
        spine_flags=regrouped_data["spine_flags"],
        followup_flags=regrouped_data["followup_flags"],
        spine_volumes=regrouped_data["spine_volumes"],
        spine_volumes_um=regrouped_data["spine_volumes_um"],
        followup_volumes=regrouped_data["followup_volumes"],
        followup_volumes_um=regrouped_data["followup_volumes_um"],
        movement_spines=regrouped_data["movement_spines"],
        nonmovement_spines=regrouped_data["nonmovement_spines"],
        rwd_movement_spines=regrouped_data["rwd_movement_spines"],
        rwd_nonmovement_spines=regrouped_data["rwd_nonmovement_spines"],
        movement_dendrites=regrouped_data["movement_dendrites"],
        nonmovement_dendrites=regrouped_data["nonmovement_dendrites"],
        rwd_movement_dendrites=regrouped_data["rwd_movement_dendrites"],
        rwd_nonmovement_dendrites=regrouped_data["rwd_nonmovement_dendrites"],
        spine_activity_rate=regrouped_data["spine_activity_rate"],
        dend_activity_rate=regrouped_data["dend_activity_rate"],
        distance_coactivity_rate=regrouped_data["distance_coactivity_rate"],
        distance_coactivity_rate_norm=regrouped_data["distance_coactivity_rate_norm"],
        MRS_distance_coactivity_rate=regrouped_data["MRS_distance_coactivity_rate"],
        MRS_distance_coactivity_rate_norm=regrouped_data[
            "MRS_distance_coactivity_rate_norm"
        ],
        nMRS_distance_coactivity_rate=regrouped_data["nMRS_distance_coactivity_rate"],
        nMRS_distance_coactivity_rate_norm=regrouped_data[
            "nMRS_distance_coactivity_rate_norm"
        ],
        avg_local_coactivity_rate=regrouped_data["avg_local_coactivity_rate"],
        avg_local_coactivity_rate_norm=regrouped_data["avg_local_coactivity_rate_norm"],
        avg_MRS_local_coactivity_rate=regrouped_data["avg_MRS_local_coactivity_rate"],
        avg_MRS_local_coactivity_rate_norm=regrouped_data[
            "avg_MRS_local_coactivity_rate_norm"
        ],
        avg_nMRS_local_coactivity_rate=regrouped_data["avg_nMRS_local_coactivity_rate"],
        avg_nMRS_local_coactivity_rate_norm=regrouped_data[
            "avg_nMRS_local_coactivity_rate_norm"
        ],
        spine_cluster_score=regrouped_data["spine_cluster_score"],
        spine_coactive_num=regrouped_data["spine_coactive_num"],
        MRS_cluster_score=regrouped_data["MRS_cluster_score"],
        MRS_coactive_num=regrouped_data["MRS_coactive_num"],
        nMRS_cluster_score=regrouped_data["nMRS_cluster_score"],
        nMRS_coactive_num=regrouped_data["nMRS_coactive_num"],
        movement_cluster_score=regrouped_data["movement_cluster_score"],
        movement_coactive_num=regrouped_data["movement_coactive_num"],
        nonmovement_cluster_score=regrouped_data["nonmovement_cluster_score"],
        nonmovement_coactive_num=regrouped_data["nonmovement_coactive_num"],
        avg_nearby_spine_rate=regrouped_data["avg_nearby_spine_rate"],
        avg_nearby_coactive_spine_rate=regrouped_data["avg_nearby_coactive_spine_rate"],
        rel_nearby_spine_rate=regrouped_data["rel_nearby_spine_rate"],
        rel_nearby_coactive_spine_rate=regrouped_data["rel_nearby_coactive_spine_rate"],
        frac_nearby_MRSs=regrouped_data["frac_nearby_MRSs"],
        nearby_coactive_spine_volumes=regrouped_data["nearby_coactive_spine_volumes"],
        local_coactivity_rate=regrouped_data["local_coactivity_rate"],
        local_coactivity_rate_norm=regrouped_data["local_coactivity_rate_norm"],
        local_spine_fraction_coactive=regrouped_data["local_spine_fraction_coactive"],
        local_spine_coactive_amplitude=regrouped_data["local_spine_coactive_amplitude"],
        local_spine_coactive_calcium=regrouped_data["local_spine_coactive_calcium"],
        local_spine_coactive_auc=regrouped_data["local_spine_coactive_auc"],
        local_spine_coactive_calcium_auc=regrouped_data[
            "local_spine_coactive_calcium_auc"
        ],
        local_spine_coactive_traces=regrouped_data["local_spine_coactive_traces"],
        local_spine_coactive_calcium_traces=regrouped_data[
            "local_spine_coactive_calcium_traces"
        ],
        local_spine_noncoactive_amplitude=regrouped_data[
            "local_spine_noncoactive_amplitude"
        ],
        local_spine_noncoactive_calcium=regrouped_data[
            "local_spine_noncoactive_calcium"
        ],
        local_spine_noncoactive_auc=regrouped_data["local_spine_noncoactive_auc"],
        local_spine_noncoactive_calcium_auc=regrouped_data[
            "local_spine_noncoactive_calcium_auc"
        ],
        local_spine_noncoactive_traces=regrouped_data["local_spine_noncoactive_trace"],
        local_spine_noncoactive_calcium_traces=regrouped_data[
            "local_spine_noncoactive_calcium_traces"
        ],
        local_avg_coactive_spine_num=regrouped_data["local_avg_coactive_spine_num"],
        local_sum_nearby_amplitude=regrouped_data["local_sum_nearby_amplitude"],
        local_avg_nearby_amplitude=regrouped_data["local_avg_nearby_amplitude"],
        local_sum_nearby_calcium=regrouped_data["local_sum_nearby_calcium"],
        local_avg_nearby_calcium=regrouped_data["local_avg_nearby_calcium"],
        local_sum_nearby_calcium_auc=regrouped_data["local_sum_nearby_calcium_auc"],
        local_avg_nearby_calcium_auc=regrouped_data["local_avg_nearby_calcium_auc"],
        local_avg_coactive_num_before=regrouped_data["local_avg_coactive_num_before"],
        local_sum_nearby_amplitude_before=regrouped_data[
            "local_sum_nearby_amplitude_before"
        ],
        local_avg_nearby_amplitude_before=regrouped_data[
            "local_avg_nearby_amplitude_before"
        ],
        local_sum_nearby_calcium_before=regrouped_data[
            "local_sum_nearby_calcium_before"
        ],
        local_avg_nearby_calcium_before=regrouped_data[
            "local_avg_nearby_calcium_before"
        ],
        local_avg_relative_nearby_onset=regrouped_data[
            "local_avg_relative_nearby_onset"
        ],
        local_sum_nearby_binary_traces=regrouped_data["local_sum_nearby_binary_traces"],
        local_sum_nearby_spine_traces=regrouped_data["local_sum_nearby_spine_traces"],
        local_avg_nearby_spine_traces=regrouped_data["local_avg_nearby_spine_traces"],
        local_sum_nearby_calcium_traces=regrouped_data[
            "local_sum_nearby_calcium_traces"
        ],
        local_avg_nearby_calcium_traces=regrouped_data[
            "local_avg_nearby_calcium_traces"
        ],
        avg_nearby_movement_correlation=regrouped_data[
            "avg_nearby_movement_correlation"
        ],
        avg_nearby_movement_reliability=regrouped_data[
            "avg_nearby_movement_reliability"
        ],
        avg_nearby_movement_specificity=regrouped_data[
            "avg_nearby_movement_specificity"
        ],
        avg_nearby_coactive_rate=regrouped_data["avg_nearby_coactive_rate"],
        relative_local_coactivity_rate=regrouped_data["relative_local_coactivity_rate"],
        frac_local_coactivity_participation=regrouped_data[
            "frac_local_coactivity_participation"
        ],
        spine_dend_coactivity_rate=regrouped_data["spine_dend_coactivity_rate"],
        spine_dend_coactivity_rate_norm=regrouped_data[
            "spine_dend_coactivity_rate_norm"
        ],
        spine_dend_spine_fraction_coactive=regrouped_data[
            "spine_dend_spine_fraction_coactive"
        ],
        spine_dend_dend_fraction_coactive=regrouped_data[
            "spine_dend_dend_fraction_coactive"
        ],
        spine_dend_spine_coactive_amplitude=regrouped_data[
            "spine_dend_spine_coactive_amplitude"
        ],
        spine_dend_spine_coactive_calcium=regrouped_data[
            "spine_dend_spine_coactive_calcium"
        ],
        spine_dend_spine_coactive_auc=regrouped_data["spine_dend_spine_coactive_auc"],
        spine_dend_spine_coactive_calcium_auc=regrouped_data[
            "spine_dend_spine_coactive_calcium_auc"
        ],
        spine_dend_dend_coactive_amplitude=regrouped_data[
            "spine_dend_dend_coactive_amplitude"
        ],
        spine_dend_dend_coactive_auc=regrouped_data["spine_dend_dend_coactive_auc"],
        spine_dend_relative_onset=regrouped_data["spine_dend_relative_onset"],
        spine_dend_spine_coactive_traces=regrouped_data[
            "spine_dend_spine_coactive_traces"
        ],
        spine_dend_spine_coactive_calcium_traces=regrouped_data[
            "spine_dend_spine_coactive_calcium_traces"
        ],
        spine_dend_dend_coactive_traces=regrouped_data[
            "spine_dend_dend_coactive_traces"
        ],
        conj_coactivity_rate=regrouped_data["conj_coactivity_rate"],
        conj_coactivity_rate_norm=regrouped_data["conj_coactivity_rate_norm"],
        conj_spine_fraction_coactive=regrouped_data["conj_spine_fraction_coactive"],
        conj_dend_fraction_coactive=regrouped_data["conj_dend_fraction_coactive"],
        conj_spine_coactive_amplitude=regrouped_data["conj_spine_coactive_amplitude"],
        conj_spine_coactive_calcium=regrouped_data["conj_spine_coactive_calcium"],
        conj_spine_coactive_auc=regrouped_data["conj_spine_coactive_auc"],
        conj_spine_coactive_calcium_auc=regrouped_data[
            "conj_spine_coactive_calcium_auc"
        ],
        conj_dend_coactive_amplitude=regrouped_data["conj_dend_coactive_amplitude"],
        conj_dend_coactive_auc=regrouped_data["conj_dend_coactive_auc"],
        conj_relative_onset=regrouped_data["conj_relative_onset"],
        conj_spine_coactive_traces=regrouped_data["conj_spine_coactive_traces"],
        conj_spine_coactive_calcium_traces=regrouped_data[
            "conj_spine_coactive_calcium_traces"
        ],
        conj_dend_coactive_traces=regrouped_data["conj_dend_coactive_traces"],
        nonconj_coactivity_rate=regrouped_data["nonconj_coactivity_rate"],
        nonconj_coactivity_rate_norm=regrouped_data["nonconj_coactivity_rate_norm"],
        nonconj_spine_fraction_coactive=regrouped_data[
            "nonconj_spine_fraction_coactive"
        ],
        nonconj_dend_fraction_coactive=regrouped_data["nonconj_dend_fraction_coactive"],
        nonconj_spine_coactive_amplitude=regrouped_data[
            "nonconj_spine_coactive_amplitude"
        ],
        nonconj_spine_coactive_calcium=regrouped_data["nonconj_spine_coactive_calcium"],
        nonconj_spine_coactive_auc=regrouped_data["nonconj_spine_coactive_auc"],
        nonconj_spine_coactive_calcium_auc=regrouped_data[
            "nonconj_spine_coactive_calcium_auc"
        ],
        nonconj_dend_coactive_amplitude=regrouped_data[
            "nonconj_dend_coactive_amplitude"
        ],
        nonconj_dend_coactive_auc=regrouped_data["nonconj_dend_coactive_auc"],
        nonconj_relative_onset=regrouped_data["nonconj_relative_onset"],
        nonconj_spine_coactive_traces=regrouped_data["nonconj_spine_coactive_traces"],
        nonconj_spine_coactive_calcium_traces=regrouped_data[
            "nonconj_spine_coactive_calcium_traces"
        ],
        nonconj_dend_coactive_traces=regrouped_data["nonconj_dend_coactive_traces"],
        spine_dend_distance_coactivity_rate=regrouped_data[
            "spine_dend_distance_coactivity_rate"
        ],
        spine_dend_distance_coactivity_rate_norm=regrouped_data[
            "spine_dend_distance_coactivity_rate_norm"
        ],
        spine_dend_avg_local_coactivity_rate=regrouped_data[
            "spine_dend_avg_local_coactivity_rate"
        ],
        spine_dend_avg_local_coactivity_rate_norm=regrouped_data[
            "spine_dend_avg_local_coactivity_rate_norm"
        ],
        spine_dend_cluster_score=regrouped_data["spine_dend_cluster_score"],
        spine_dend_coactive_num=regrouped_data["spine_dend_coactive_num"],
        movement_spine_traces=regrouped_data["movement_spine_traces"],
        movement_spine_amplitude=regrouped_data["movement_spine_amplitude"],
        movement_spine_onset=regrouped_data["movement_spine_onset"],
        movement_dend_traces=regrouped_data["movement_dend_traces"],
        movement_dend_amplitude=regrouped_data["movement_dend_amplitude"],
        movement_dend_onset=regrouped_data["movement_dend_onset"],
        learned_movement_pattern=regrouped_data["learned_movement_pattern"],
        spine_movements=regrouped_data["spine_movements"],
        spine_movement_correlation=regrouped_data["spine_movement_correlation"],
        spine_movement_reliability=regrouped_data["spine_movement_reliability"],
        spine_movement_specificity=regrouped_data["spine_movement_specificity"],
        dend_movements=regrouped_data["dend_movements"],
        dend_movement_correlation=regrouped_data["dend_movement_correlation"],
        dend_movement_reliability=regrouped_data["dend_movement_reliability"],
        dend_movement_specificity=regrouped_data["dend_movement_specificity"],
        local_movements=regrouped_data["local_movements"],
        local_movement_correlation=regrouped_data["local_movement_correlation"],
        local_movement_reliability=regrouped_data["local_movement_reliability"],
        local_movement_specificity=regrouped_data["local_movement_specificity"],
        spine_dend_movements=regrouped_data["spine_dend_movements"],
        spine_dend_movement_correlation=regrouped_data[
            "spine_dend_movement_correlation"
        ],
        spine_dend_movement_reliability=regrouped_data[
            "spine_dend_movement_reliability"
        ],
        spine_dend_movement_specificity=regrouped_data[
            "spine_dend_movement_specificity"
        ],
        rel_spine_vs_dend_move_corr=regrouped_data["rel_spine_vs_dend_move_corr"],
        spine_to_dend_correlation=regrouped_data["spine_to_dend_correlation"],
        spine_to_nearby_correlation=regrouped_data["spine_to_nearby_correlation"],
    )
