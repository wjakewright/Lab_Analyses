import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Spine_Analysis.conjunctive_spine_coactivity import (
    conjunctive_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.local_spine_coactivity import (
    local_spine_coactivity_analysis,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import (
    quantify_movement_quality,
    spine_movement_activity,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis.total_spine_coactivity import total_coactivity_analysis
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def grouped_coactivity_analysis(
    mice_list,
    day,
    followup=True,
    movement_epochs=None,
    zscore=False,
    volume_norm=False,
    save=False,
    save_path=None,
):
    """Function to handle coactivity analysis of dual spine imaging datsets across all
        mice and all FOVs. Stores coactivity data in dataclass to be used in subsequent
        analyses (e.g., plasticity)
        
        INPUT PARAMETERS
            mice_list - list of strings specifying mice ids to be analyzed
            
            day - string of the day to analyze
            
            followup - boolean specifying if you want to include followup structural 
                        imaging sessions
            
            movment_epochs - str specifying if you want to analyze only during movements and
                            what type of movements. Accepts 'movement', 'rewarded', 
                            'unrewarded', 'nonmovement' and 'learned'. Default is None to 
                            analyze the entire imaging period
            
            zscore - boolean of whether or not to zscore activity for analysis
                            
            save - boolean of whether to save the output or not
            
            save_path - str specifying where to save the data
            
    """

    grouped_data = defaultdict(list)

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"--- Analyzing {mouse}")
        mouse_data = defaultdict(list)
        if followup is True:
            datasets = load_spine_datasets(mouse, [day], followup=True)
        else:
            datasets = load_spine_datasets(mouse, [day], followup=False)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
            if followup is True:
                data = dataset[f"Pre {day}"]
                followup_volume = dataset[f"Post {day}"].corrected_spine_volume
                followup_flags = dataset[f"Post {day}"].spine_flags
            else:
                data = dataset[day]
                followup_volume = None
                followup_flags = None
            sampling_rate = data.imaging_parameters["Sampling Rate"]
            # Analyze local spine coactivity
            (
                local_distance_coactivity_rate,
                local_distance_bins,
                local_spine_correlation,
                local_coactivity_rate,
                local_coactivity_matrix,
                local_spine_fraction_coactive,
                local_coactive_spine_num,
                local_coactive_spine_volumes,
                local_spine_coactive_amplitude,
                local_nearby_coactive_amplitude,
                local_spine_coactive_calcium,
                local_nearby_coactive_calcium,
                local_spine_coactive_std,
                local_nearby_coactive_std,
                local_spine_coactive_calcium_std,
                local_nearby_coactive_calcium_std,
                local_spine_coactive_calcium_auc,
                local_nearby_coactive_calcium_auc,
                local_pine_coactive_traces,
                local_nearby_coactive_traces,
                local_spine_coactive_calcium_traces,
                local_nearby_coactive_calcium_traces,
            ) = local_spine_coactivity_analysis(
                data,
                movement_epoch=movement_epochs,
                cluster_dist=10,
                sampling_rate=sampling_rate,
                zscore=zscore,
                volume_norm=volume_norm,
            )

            # Analyze global spine-dendrite coactivity
            (
                global_correlation,
                global_coactivity_event_num,
                global_coactivity_event_rate,
                global_spine_fraction_coactive,
                global_dend_fraction_coactive,
                global_spine_coactive_amplitude,
                global_dend_coactive_amplitude,
                global_spine_coactive_calcium,
                global_spine_coactive_std,
                global_dend_coactive_std,
                global_spine_coactive_calcium_std,
                global_dend_coactive_auc,
                global_spine_coactive_calcium_auc,
                global_relative_spine_coactive_amplitude,
                global_relative_dend_coactive_amplitude,
                global_relative_spine_coactive_calcium,
                global_relative_spine_onsets,
                global_dend_triggered_spine_traces,
                global_dend_triggered_dend_traces,
                global_dend_triggered_spine_calcium_traces,
                global_coactive_spine_traces,
                global_coactive_dend_traces,
                global_coactive_spine_calcium_traces,
                global_coactivity_matrix,
            ) = total_coactivity_analysis(
                data,
                movement_epoch=movement_epochs,
                sampling_rate=sampling_rate,
                zscore=zscore,
                volume_norm=volume_norm,
            )

            # Analyze conjunctive local spine and dendrite coactivity
            (
                conjunctive_correlation,
                conj_coactivity_event_num,
                conj_coactivity_event_rate,
                conj_spine_fraction_coactive,
                conj_dend_fraction_coactive,
                conj_coactive_spine_num,
                conj_coactive_spine_volumes,
                conj_spine_coactive_amplitude,
                conj_nearby_coactive_amplitude_sum,
                conj_spine_coactive_calcium,
                conj_nearby_coactive_calcium_sum,
                conj_dend_coactive_amplitude,
                conj_spine_coactive_std,
                conj_nearby_coactive_std,
                conj_spine_coactive_calcium_std,
                conj_nearby_coactive_calcium_std,
                conj_dend_coactive_std,
                conj_spine_coactive_calcium_auc,
                conj_nearby_coactive_calcium_auc_sum,
                conj_dend_coactive_auc,
                conj_relative_spine_dend_onsets,
                conj_coactive_spine_traces,
                conj_coactive_nearby_traces,
                conj_coactive_spine_calcium_traces,
                conj_coactive_nearby_calcium_traces,
                conj_coactive_dend_traces,
                conj_coactivity_matrix,
            ) = conjunctive_coactivity_analysis(
                data,
                movement_epoch=movement_epochs,
                cluster_dist=10,
                sampling_rate=sampling_rate,
                zscore=zscore,
                volume_norm=volume_norm,
            )

            # Analyze movement-related activity
            (
                move_dend_traces,
                move_spine_traces,
                move_dend_amplitude,
                move_dend_std,
                move_spine_amplitude,
                move_spine_std,
                move_dend_onset,
                move_spine_onset,
            ) = spine_movement_activity(
                data,
                rewarded=False,
                zscore=zscore,
                volume_norm=volume_norm,
                sampling_rate=sampling_rate,
                activity_window=(-2, 2),
            )
            (
                rwd_move_dend_traces,
                rwd_move_spine_traces,
                rwd_move_dend_amplitude,
                rwd_move_dend_std,
                rwd_move_spine_amplitude,
                rwd_move_spine_std,
                rwd_move_dend_onset,
                rwd_move_spine_onset,
            ) = spine_movement_activity(
                data,
                rewarded=True,
                zscore=zscore,
                volume_norm=volume_norm,
                sampling_rate=sampling_rate,
                activity_window=(-2, 2),
            )
            # Assess movement quality
            (
                _,
                spine_movements,
                _,
                spine_movement_corr,
                learned_movement,
            ) = quantify_movement_quality(
                mouse,
                data.spine_GluSnFr_activity,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            ### Set up dendrite activity matrix for each spine
            dend_activity_matrix = np.zeros(data.spine_GluSnFr_activity.shape)
            for d in range(data.dendrite_calcium_activity.shape[1]):
                if type(data.spine_grouping[d]) == list:
                    spines = data.spine_grouping[d]
                else:
                    spines = data.spine_grouping
                dend_activity_matrix[:, spines] = data.dendrite_calcium_activity[:, d]
            (_, dend_movements, _, dend_movement_corr, _,) = quantify_movement_quality(
                mouse,
                dend_activity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            (
                _,
                local_movements,
                _,
                local_movement_corr,
                _,
            ) = quantify_movement_quality(
                mouse,
                local_coactivity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            (
                _,
                global_movements,
                _,
                global_movement_corr,
                _,
            ) = quantify_movement_quality(
                mouse,
                global_coactivity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            (_, conj_movements, _, conj_movement_corr, _,) = quantify_movement_quality(
                mouse,
                conj_coactivity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )

            # Get basic activity frequencies
            spine_activity_freq = []
            dend_activity_freq = []
            for s in range(data.spine_GluSnFr_activity.shape[1]):
                duration = len(data.spine_GluSnFr_activity[:, s]) / sampling_rate
                s_events = np.nonzero(np.diff(data.spine_GluSnFr_activity[:, s]) == 1)
                d_events = np.nonzero(np.diff(dend_activity_matrix[:, s]) == 1)
                spine_freq = (s_events / duration) * 60
                dend_freq = (d_events / duration) * 60
                spine_activity_freq.append(spine_freq)
                dend_activity_freq.append(dend_freq)
            spine_activity_freq = np.array(spine_activity_freq)
            dend_activity_freq = np.array(dend_activity_freq)

            fovs = [FOV for i in range(spine_activity_freq)]
            ids = [mouse for i in range(spine_activity_freq)]

            # Store data from this mouse in mouse_data dictionary


################# DATACLASS #####################
@dataclass
class Spine_Coactivity_Data:
    """Dataclass for storing spine data of a single day following coactivity analysis 
        across all mice in a given group"""

    mouse_id: list
    FOV: list
    parameters: dict
    spine_flags: list
    followup_flags: list
    spine_volumes: np.array
    followup_volumes: np.array
    spine_activity_freq: np.array
    dend_activity_freq: np.array
    distance_coactivity_rate: np.array  # 2d
    local_spine_correlation: np.array
    local_coactivity_rate: np.array
    local_spine_fraction_coactive: np.array
    local_coactive_spine_num: np.array
    local_coactive_spine_volumes: np.array
    local_spine_coactive_amplitude: np.array
    local_nearby_coactive_amplitude: np.array
    local_spine_coactive_calcium: np.array
    local_nearby_coactive_calcium: np.array
    local_spine_coactive_std: np.array
    local_nearby_coactive_std: np.array
    local_spine_coactive_calcium_std: np.array
    local_nearby_coactive_calcium_std: np.array
    local_spine_coactive_calcium_auc: np.array
    local_nearby_coactive_calcium_auc: np.array
    local_pine_coactive_traces: list
    local_nearby_coactive_traces: list
    local_spine_coactive_calcium_traces: list
    local_nearby_coactive_calcium_traces: list
    global_correlation: np.array
    global_coactivity_event_num: np.array
    global_coactivity_event_rate: np.array
    global_spine_fraction_coactive: np.array
    global_dend_fraction_coactive: np.array
    global_spine_coactive_amplitude: np.array
    global_dend_coactive_amplitude: np.array
    global_spine_coactive_calcium: np.array
    global_spine_coactive_std: np.array
    global_dend_coactive_std: np.array
    global_spine_coactive_calcium_std: np.array
    global_dend_coactive_auc: np.array
    global_spine_coactive_calcium_auc: np.array
    global_relative_spine_coactive_amplitude: np.array
    global_relative_dend_coactive_amplitude: np.array
    global_relative_spine_coactive_calcium: np.array
    global_relative_spine_onsets: np.array
    global_dend_triggered_spine_traces: list
    global_dend_triggered_dend_traces: list
    global_dend_triggered_spine_calcium_traces: list
    global_coactive_spine_traces: list
    global_coactive_dend_traces: list
    global_coactive_spine_calcium_traces: list
    conjunctive_correlation: np.array
    conj_coactivity_event_num: np.array
    conj_coactivity_event_rate: np.array
    conj_spine_fraction_coactive: np.array
    conj_dend_fraction_coactive: np.array
    conj_coactive_spine_num: np.array
    conj_coactive_spine_volumes: np.array
    conj_spine_coactive_amplitude: np.array
    conj_nearby_coactive_amplitude_sum: np.array
    conj_spine_coactive_calcium: np.array
    conj_nearby_coactive_calcium_sum: np.array
    conj_dend_coactive_amplitude: np.array
    conj_spine_coactive_std: np.array
    conj_nearby_coactive_std: np.array
    conj_spine_coactive_calcium_std: np.array
    conj_nearby_coactive_calcium_std: np.array
    conj_dend_coactive_std: np.array
    conj_spine_coactive_calcium_auc: np.array
    conj_nearby_coactive_calcium_auc_sum: np.array
    conj_dend_coactive_auc: np.array
    conj_relative_spine_dend_onsets: np.array
    conj_coactive_spine_traces: list
    conj_coactive_nearby_traces: list
    conj_coactive_spine_calcium_traces: list
    conj_coactive_nearby_calcium_traces: list
    conj_coactive_dend_traces: list
    move_dend_traces: list
    move_spine_traces: list
    move_dend_amplitude: np.array
    move_dend_std: np.array
    move_spine_amplitude: np.array
    move_spine_std: np.array
    move_dend_onset: np.array
    move_spine_onset: np.array
    rwd_move_dend_traces: list
    rwd_move_spine_traces: list
    rwd_move_dend_amplitude: np.array
    rwd_move_dend_std: np.array
    rwd_move_spine_amplitude: np.array
    rwd_move_spine_std: np.array
    rwd_move_dend_onset: np.array
    rwd_move_spine_onset: np.array
    learned_movement: np.array
    spine_movements: list
    spine_movement_corr: np.array
    dend_movements: list
    dend_movement_corr: np.array
    local_movements: list
    local_movement_corr: np.array
    global_movements: list
    global_movement_corr: np.array
    conj_movements: list
    conj_movement_corr: np.array

