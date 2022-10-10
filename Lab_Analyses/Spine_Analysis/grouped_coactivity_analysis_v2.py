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
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    calculate_dend_spine_freq,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import spine_movement_activity
from Lab_Analyses.Spine_Analysis.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis.total_spine_coactivity import total_coactivity_analysis
from Lab_Analyses.Utilities.quantify_movment_quality import quantify_movement_quality
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def grouped_coactivity_analysis(
    mice_list,
    day,
    followup=True,
    activity_window=(-2, 2),
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
    CLUSTER_DIST = 10
    grouped_data = defaultdict(list)

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"--- Analyzing {mouse}")
        if followup is True:
            datasets = load_spine_datasets(mouse, [day], followup=True)
        else:
            datasets = load_spine_datasets(mouse, [day], followup=False)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
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
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])

            movement_spines = np.array(data.movement_spines)
            rwd_movement_spines = np.array(data.reward_movement_spines)
            movement_dendrites = np.zeros(movement_spines.shape)
            rwd_movement_dendrites = np.zeros(rwd_movement_spines.shape)
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
                local_spine_coactive_traces,
                local_nearby_coactive_traces,
                local_spine_coactive_calcium_traces,
                local_nearby_coactive_calcium_traces,
            ) = local_spine_coactivity_analysis(
                data,
                activity_window=activity_window,
                movement_epoch=movement_epochs,
                cluster_dist=CLUSTER_DIST,
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
                activity_window=activity_window,
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
                activity_window=activity_window,
                movement_epoch=movement_epochs,
                cluster_dist=CLUSTER_DIST,
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
                activity_window=activity_window,
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
                activity_window=activity_window,
            )
            # Assess movement quality
            (
                _,
                spine_movements,
                _,
                spine_movement_corr,
                spine_movement_num,
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
                    for s in spines:
                        dend_activity_matrix[:, s] = data.dendrite_calcium_activity[
                            :, d
                        ]
                        movement_dendrites[s] = data.movement_dendrites[d]
                        rwd_movement_dendrites[s] = data.reward_movement_dendrites[d]
            (
                _,
                dend_movements,
                _,
                dend_movement_corr,
                dend_movement_num,
                _,
            ) = quantify_movement_quality(
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
                local_movement_num,
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
                global_movement_num,
                _,
            ) = quantify_movement_quality(
                mouse,
                global_coactivity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            (
                _,
                conj_movements,
                _,
                conj_movement_corr,
                conj_movement_num,
                _,
            ) = quantify_movement_quality(
                mouse,
                conj_coactivity_matrix,
                data.lever_active,
                data.lever_force_smooth,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )

            # Get basic activity frequencies
            spine_activity_freq, dend_activity_freq = calculate_dend_spine_freq(
                data, movement_epochs, sampling_rate,
            )

            fovs = [FOV for i in range(len(spine_activity_freq))]
            ids = [mouse for i in range(len(spine_activity_freq))]

            # Get volumes in um
            pix_to_um = data.imaging_parameters["Zoom"] / 2
            spine_volume = np.array(data.corrected_spine_volume)
            spine_volume_um = spine_volume / pix_to_um
            followup_volume_um = followup_volume / pix_to_um

            # get non-movement groups
            non_movement_spines = [not x for x in movement_spines]
            non_movement_dendrites = [not x for x in movement_dendrites]
            non_movement_spines = np.array(non_movement_spines)
            non_movement_dendrites = np.array(non_movement_dendrites)

            # Store data from this mouse in grouped_data dictionary
            grouped_data["mouse_id"].append(ids)
            grouped_data["FOV"].append(fovs)
            grouped_data["spine_flags"].append(data.spine_flags)
            grouped_data["followup_flags"].append(followup_flags)
            grouped_data["spine_volumes"].append(spine_volume)
            grouped_data["spine_volumes_um"].append(spine_volume_um)
            grouped_data["followup_volumes"].append(followup_volume)
            grouped_data["followup_volumes_um"].append(followup_volume_um)
            grouped_data["spine_activity_freq"].append(spine_activity_freq)
            grouped_data["dend_activity_freq"].append(dend_activity_freq)
            grouped_data["movement_spines"].append(movement_spines)
            grouped_data["nonmovement_spines"].append(non_movement_spines)
            grouped_data["rwd_movement_spines"].append(rwd_movement_spines)
            grouped_data["movement_dendrites"].append(movement_dendrites)
            grouped_data["nonmovement_dendrites"].append(non_movement_dendrites)
            grouped_data["rwd_movement_dendrites"].append(rwd_movement_dendrites)
            grouped_data["distance_coactivity_rate"].append(
                local_distance_coactivity_rate
            )
            grouped_data["local_spine_correlation"].append(local_spine_correlation)
            grouped_data["local_coactivity_rate"].append(local_coactivity_rate)
            grouped_data["local_spine_fraction_coactive"].append(
                local_spine_fraction_coactive
            )
            grouped_data["local_coactive_spine_num"].append(local_coactive_spine_num)
            grouped_data["local_coactive_spine_volumes"].append(
                local_coactive_spine_volumes
            )
            grouped_data["local_spine_coactive_amplitude"].append(
                local_spine_coactive_amplitude
            )
            grouped_data["local_nearby_coactive_amplitude"].append(
                local_nearby_coactive_amplitude
            )
            grouped_data["local_spine_coactive_calcium"].append(
                local_spine_coactive_calcium
            )
            grouped_data["local_nearby_coactive_calcium"].append(
                local_nearby_coactive_calcium
            )
            grouped_data["local_spine_coactive_std"].append(local_spine_coactive_std)
            grouped_data["local_nearby_coactive_std"].append(local_nearby_coactive_std)
            grouped_data["local_spine_coactive_calcium_std"].append(
                local_spine_coactive_calcium_std
            )
            grouped_data["local_nearby_coactive_calcium_std"].append(
                local_nearby_coactive_calcium_std
            )
            grouped_data["local_spine_coactive_calcium_auc"].append(
                local_spine_coactive_calcium_auc
            )
            grouped_data["local_nearby_coactive_calcium_auc"].append(
                local_nearby_coactive_calcium_auc
            )
            grouped_data["local_spine_coactive_traces"].append(
                local_spine_coactive_traces
            )
            grouped_data["local_nearby_coactive_traces"].append(
                local_nearby_coactive_traces
            )
            grouped_data["local_spine_coactive_calcium_traces"].append(
                local_spine_coactive_calcium_traces
            )
            grouped_data["local_nearby_coactive_calcium_traces"].append(
                local_nearby_coactive_calcium_traces
            )
            grouped_data["global_correlation"].append(global_correlation)
            grouped_data["global_coactivity_event_num"].append(
                global_coactivity_event_num
            )
            grouped_data["global_coactivity_event_rate"].append(
                global_coactivity_event_rate
            )
            grouped_data["global_spine_fraction_coactive"].append(
                global_spine_fraction_coactive
            )
            grouped_data["global_dend_fraction_coactive"].append(
                global_dend_fraction_coactive
            )
            grouped_data["global_spine_coactive_amplitude"].append(
                global_spine_coactive_amplitude
            )
            grouped_data["global_dend_coactive_amplitude"].append(
                global_dend_coactive_amplitude
            )
            grouped_data["global_spine_coactive_calcium"].append(
                global_spine_coactive_calcium
            )
            grouped_data["global_spine_coactive_std"].append(global_spine_coactive_std)
            grouped_data["global_dend_coactive_std"].append(global_dend_coactive_std)
            grouped_data["global_spine_coactive_calcium_std"].append(
                global_spine_coactive_calcium_std
            )
            grouped_data["global_dend_coactive_auc"].append(global_dend_coactive_auc)
            grouped_data["global_spine_coactive_calcium_auc"].append(
                global_spine_coactive_calcium_auc
            )
            grouped_data["global_relative_spine_coactive_amplitude"].append(
                global_relative_spine_coactive_amplitude
            )
            grouped_data["global_relative_dend_coactive_amplitude"].append(
                global_relative_dend_coactive_amplitude
            )
            grouped_data["global_relative_spine_coactive_calcium"].append(
                global_relative_spine_coactive_calcium
            )
            grouped_data["global_relative_spine_onsets"].append(
                global_relative_spine_onsets
            )
            grouped_data["global_dend_triggered_spine_traces"].append(
                global_dend_triggered_spine_traces
            )
            grouped_data["global_dend_triggered_dend_traces"].append(
                global_dend_triggered_dend_traces
            )
            grouped_data["global_dend_triggered_spine_calcium_traces"].append(
                global_dend_triggered_spine_calcium_traces
            )
            grouped_data["global_coactive_spine_traces"].append(
                global_coactive_spine_traces
            )
            grouped_data["global_coactive_dend_traces"].append(
                global_coactive_dend_traces
            )
            grouped_data["global_coactive_spine_calcium_traces"].append(
                global_coactive_spine_calcium_traces
            )
            grouped_data["conjunctive_correlation"].append(conjunctive_correlation)
            grouped_data["conj_coactivity_event_num"].append(conj_coactivity_event_num)
            grouped_data["conj_coactivity_event_rate"].append(
                conj_coactivity_event_rate
            )
            grouped_data["conj_spine_fraction_coactive"].append(
                conj_spine_fraction_coactive
            )
            grouped_data["conj_dend_fraction_coactive"].append(
                conj_dend_fraction_coactive
            )
            grouped_data["conj_coactive_spine_num"].append(conj_coactive_spine_num)
            grouped_data["conj_coactive_spine_volumes"].append(
                conj_coactive_spine_volumes
            )
            grouped_data["conj_spine_coactive_amplitude"].append(
                conj_spine_coactive_amplitude
            )
            grouped_data["conj_nearby_coactive_amplitude_sum"].append(
                conj_nearby_coactive_amplitude_sum
            )
            grouped_data["conj_spine_coactive_calcium"].append(
                conj_spine_coactive_calcium
            )
            grouped_data["conj_nearby_coactive_calcium_sum"].append(
                conj_nearby_coactive_calcium_sum
            )
            grouped_data["conj_dend_coactive_amplitude"].append(
                conj_dend_coactive_amplitude
            )
            grouped_data["conj_spine_coactive_std"].append(conj_spine_coactive_std)
            grouped_data["conj_nearby_coactive_std"].append(conj_nearby_coactive_std)
            grouped_data["conj_spine_coactive_calcium_std"].append(
                conj_spine_coactive_calcium_std
            )
            grouped_data["conj_nearby_coactive_calcium_std"].append(
                conj_nearby_coactive_calcium_std
            )
            grouped_data["conj_dend_coactive_std"].append(conj_dend_coactive_std)
            grouped_data["conj_spine_coactive_calcium_auc"].append(
                conj_spine_coactive_calcium_auc
            )
            grouped_data["conj_nearby_coactive_calcium_auc_sum"].append(
                conj_nearby_coactive_calcium_auc_sum
            )
            grouped_data["conj_dend_coactive_auc"].append(conj_dend_coactive_auc)
            grouped_data["conj_relative_spine_dend_onsets"].append(
                conj_relative_spine_dend_onsets
            )
            grouped_data["conj_coactive_spine_traces"].append(
                conj_coactive_spine_traces
            )
            grouped_data["conj_coactive_nearby_traces"].append(
                conj_coactive_nearby_traces
            )
            grouped_data["conj_coactive_spine_calcium_traces"].append(
                conj_coactive_spine_calcium_traces
            )
            grouped_data["conj_coactive_nearby_calcium_traces"].append(
                conj_coactive_nearby_calcium_traces
            )
            grouped_data["conj_coactive_dend_traces"].append(conj_coactive_dend_traces)
            grouped_data["move_dend_traces"].append(move_dend_traces)
            grouped_data["move_spine_traces"].append(move_spine_traces)
            grouped_data["move_dend_amplitude"].append(move_dend_amplitude)
            grouped_data["move_dend_std"].append(move_dend_std)
            grouped_data["move_spine_amplitude"].append(move_spine_amplitude)
            grouped_data["move_spine_std"].append(move_spine_std)
            grouped_data["move_dend_onset"].append(move_dend_onset)
            grouped_data["move_spine_onset"].append(move_spine_onset)
            grouped_data["rwd_move_dend_traces"].append(rwd_move_dend_traces)
            grouped_data["rwd_move_spine_traces"].append(rwd_move_spine_traces)
            grouped_data["rwd_move_dend_amplitude"].append(rwd_move_dend_amplitude)
            grouped_data["rwd_move_dend_std"].append(rwd_move_dend_std)
            grouped_data["rwd_move_spine_amplitude"].append(rwd_move_spine_amplitude)
            grouped_data["rwd_move_spine_std"].append(rwd_move_spine_std)
            grouped_data["rwd_move_dend_onset"].append(rwd_move_dend_onset)
            grouped_data["rwd_move_spine_onset"].append(rwd_move_spine_onset)
            grouped_data["learned_movement"].append(learned_movement)
            grouped_data["spine_movements"].append(spine_movements)
            grouped_data["spine_movement_corr"].append(spine_movement_corr)
            grouped_data["spine_movement_num"].append(spine_movement_num)
            grouped_data["dend_movements"].append(dend_movements)
            grouped_data["dend_movement_corr"].append(dend_movement_corr)
            grouped_data["dend_movement_num"].append(dend_movement_num)
            grouped_data["local_movements"].append(local_movements)
            grouped_data["local_movement_corr"].append(local_movement_corr)
            grouped_data["local_movement_num"].append(local_movement_num)
            grouped_data["global_movements"].append(global_movements)
            grouped_data["global_movement_corr"].append(global_movement_corr)
            grouped_data["global_movement_num"].append(global_movement_num)
            grouped_data["conj_movements"].append(conj_movements)
            grouped_data["conj_movement_corr"].append(conj_movement_corr)
            grouped_data["conj_movement_num"].append(conj_movement_num)

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
    parameters = {
        "Sampling Rate": sampling_rate,
        "Cluster Dist": CLUSTER_DIST,
        "Distance Bins": local_distance_bins,
        "zscore": zscore,
        "Volume Norm": volume_norm,
        "Movement Epoch": movement_epochs,
        "Activity Window": activity_window,
    }
    # Store data for outputing and saving
    spine_coactivity_data = Spine_Coactivity_Data(
        day=day,
        mouse_id=regrouped_data["mouse_id"],
        FOV=regrouped_data["FOV"],
        parameters=parameters,
        spine_flags=regrouped_data["spine_flags"],
        followup_flags=regrouped_data["followup_flags"],
        spine_volumes=regrouped_data["spine_volumes"],
        spine_volumes_um=regrouped_data["spine_volumes_um"],
        followup_volumes=regrouped_data["followup_volumes"],
        followup_volumes_um=regrouped_data["followup_volumes_um"],
        spine_activity_freq=regrouped_data["spine_activity_freq"],
        dend_activity_freq=regrouped_data["dend_activity_freq"],
        movement_spines=regrouped_data["movement_spines"],
        nonmovement_spines=regrouped_data["nonmovement_spines"],
        rwd_movement_spines=regrouped_data["rwd_movement_spines"],
        movement_dendrites=regrouped_data["movement_dendrites"],
        nonmovement_dendrites=regrouped_data["nonmovement_dendrites"],
        rwd_movement_dendrites=regrouped_data["rwd_movement_dendrites"],
        distance_coactivity_rate=regrouped_data["distance_coactivity_rate"],
        local_spine_correlation=regrouped_data["local_spine_correlation"],
        local_coactivity_rate=regrouped_data["local_coactivity_rate"],
        local_spine_fraction_coactive=regrouped_data["local_spine_fraction_coactive"],
        local_coactive_spine_num=regrouped_data["local_coactive_spine_num"],
        local_coactive_spine_volumes=regrouped_data["local_coactive_spine_volumes"],
        local_spine_coactive_amplitude=regrouped_data["local_spine_coactive_amplitude"],
        local_nearby_coactive_amplitude=regrouped_data[
            "local_nearby_coactive_amplitude"
        ],
        local_spine_coactive_calcium=regrouped_data["local_spine_coactive_calcium"],
        local_nearby_coactive_calcium=regrouped_data["local_nearby_coactive_calcium"],
        local_spine_coactive_std=regrouped_data["local_spine_coactive_std"],
        local_nearby_coactive_std=regrouped_data["local_nearby_coactive_std"],
        local_spine_coactive_calcium_std=regrouped_data[
            "local_spine_coactive_calcium_std"
        ],
        local_nearby_coactive_calcium_std=regrouped_data[
            "local_nearby_coactive_calcium_std"
        ],
        local_spine_coactive_calcium_auc=regrouped_data[
            "local_spine_coactive_calcium_auc"
        ],
        local_nearby_coactive_calcium_auc=regrouped_data[
            "local_nearby_coactive_calcium_auc"
        ],
        local_spine_coactive_traces=regrouped_data["local_spine_coactive_traces"],
        local_nearby_coactive_traces=regrouped_data["local_nearby_coactive_traces"],
        local_spine_coactive_calcium_traces=regrouped_data[
            "local_spine_coactive_calcium_traces"
        ],
        local_nearby_coactive_calcium_traces=regrouped_data[
            "local_nearby_coactive_calcium_traces"
        ],
        global_correlation=regrouped_data["global_correlation"],
        global_coactivity_event_num=regrouped_data["global_coactivity_event_num"],
        global_coactivity_event_rate=regrouped_data["global_coactivity_event_rate"],
        global_spine_fraction_coactive=regrouped_data["global_spine_fraction_coactive"],
        global_dend_fraction_coactive=regrouped_data["global_dend_fraction_coactive"],
        global_spine_coactive_amplitude=regrouped_data[
            "global_spine_coactive_amplitude"
        ],
        global_dend_coactive_amplitude=regrouped_data["global_dend_coactive_amplitude"],
        global_spine_coactive_calcium=regrouped_data["global_spine_coactive_calcium"],
        global_spine_coactive_std=regrouped_data["global_spine_coactive_std"],
        global_dend_coactive_std=regrouped_data["global_dend_coactive_std"],
        global_spine_coactive_calcium_std=regrouped_data[
            "global_spine_coactive_calcium_std"
        ],
        global_dend_coactive_auc=regrouped_data["global_dend_coactive_auc"],
        global_spine_coactive_calcium_auc=regrouped_data[
            "global_spine_coactive_calcium_auc"
        ],
        global_relative_spine_coactive_amplitude=regrouped_data[
            "global_relative_spine_coactive_amplitude"
        ],
        global_relative_dend_coactive_amplitude=regrouped_data[
            "global_relative_dend_coactive_amplitude"
        ],
        global_relative_spine_coactive_calcium=regrouped_data[
            "global_relative_spine_coactive_calcium"
        ],
        global_relative_spine_onsets=regrouped_data["global_relative_spine_onsets"],
        global_dend_triggered_spine_traces=regrouped_data[
            "global_dend_triggered_spine_traces"
        ],
        global_dend_triggered_dend_traces=regrouped_data[
            "global_dend_triggered_dend_traces"
        ],
        global_dend_triggered_spine_calcium_traces=regrouped_data[
            "global_dend_triggered_spine_calcium_traces"
        ],
        global_coactive_spine_traces=regrouped_data["global_coactive_spine_traces"],
        global_coactive_dend_traces=regrouped_data["global_coactive_dend_traces"],
        global_coactive_spine_calcium_traces=regrouped_data[
            "global_coactive_spine_calcium_traces"
        ],
        conjunctive_correlation=regrouped_data["conjunctive_correlation"],
        conj_coactivity_event_num=regrouped_data["conj_coactivity_event_num"],
        conj_coactivity_event_rate=regrouped_data["conj_coactivity_event_rate"],
        conj_spine_fraction_coactive=regrouped_data["conj_spine_fraction_coactive"],
        conj_dend_fraction_coactive=regrouped_data["conj_dend_fraction_coactive"],
        conj_coactive_spine_num=regrouped_data["conj_coactive_spine_num"],
        conj_coactive_spine_volumes=regrouped_data["conj_coactive_spine_volumes"],
        conj_spine_coactive_amplitude=regrouped_data["conj_spine_coactive_amplitude"],
        conj_nearby_coactive_amplitude_sum=regrouped_data[
            "conj_nearby_coactive_amplitude_sum"
        ],
        conj_spine_coactive_calcium=regrouped_data["conj_spine_coactive_calcium"],
        conj_nearby_coactive_calcium_sum=regrouped_data[
            "conj_nearby_coactive_calcium_sum"
        ],
        conj_dend_coactive_amplitude=regrouped_data["conj_dend_coactive_amplitude"],
        conj_spine_coactive_std=regrouped_data["conj_spine_coactive_std"],
        conj_nearby_coactive_std=regrouped_data["conj_nearby_coactive_std"],
        conj_spine_coactive_calcium_std=regrouped_data[
            "conj_spine_coactive_calcium_std"
        ],
        conj_nearby_coactive_calcium_std=regrouped_data[
            "conj_nearby_coactive_calcium_std"
        ],
        conj_dend_coactive_std=regrouped_data["conj_dend_coactive_std"],
        conj_spine_coactive_calcium_auc=regrouped_data[
            "conj_spine_coactive_calcium_auc"
        ],
        conj_nearby_coactive_calcium_auc_sum=regrouped_data[
            "conj_nearby_coactive_calcium_auc_sum"
        ],
        conj_dend_coactive_auc=regrouped_data["conj_dend_coactive_auc"],
        conj_relative_spine_dend_onsets=regrouped_data[
            "conj_relative_spine_dend_onsets"
        ],
        conj_coactive_spine_traces=regrouped_data["conj_coactive_spine_traces"],
        conj_coactive_nearby_traces=regrouped_data["conj_coactive_nearby_traces"],
        conj_coactive_spine_calcium_traces=regrouped_data[
            "conj_coactive_spine_calcium_traces"
        ],
        conj_coactive_nearby_calcium_traces=regrouped_data[
            "conj_coactive_nearby_calcium_traces"
        ],
        conj_coactive_dend_traces=regrouped_data["conj_coactive_dend_traces"],
        move_dend_traces=regrouped_data["move_dend_traces"],
        move_spine_traces=regrouped_data["move_spine_traces"],
        move_dend_amplitude=regrouped_data["move_dend_amplitude"],
        move_dend_std=regrouped_data["move_dend_std"],
        move_spine_amplitude=regrouped_data["move_spine_amplitude"],
        move_spine_std=regrouped_data["move_spine_std"],
        move_dend_onset=regrouped_data["move_dend_onset"],
        move_spine_onset=regrouped_data["move_spine_onset"],
        rwd_move_dend_traces=regrouped_data["rwd_move_dend_traces"],
        rwd_move_spine_traces=regrouped_data["rwd_move_spine_traces"],
        rwd_move_dend_amplitude=regrouped_data["rwd_move_dend_amplitude"],
        rwd_move_dend_std=regrouped_data["rwd_move_dend_std"],
        rwd_move_spine_amplitude=regrouped_data["rwd_move_spine_amplitude"],
        rwd_move_spine_std=regrouped_data["rwd_move_spine_std"],
        rwd_move_dend_onset=regrouped_data["rwd_move_dend_onset"],
        rwd_move_spine_onset=regrouped_data["rwd_move_spine_onset"],
        learned_movement=regrouped_data["learned_movement"],
        spine_movements=regrouped_data["spine_movements"],
        spine_movement_corr=regrouped_data["spine_movement_corr"],
        spine_movement_num=regrouped_data["spine_movement_num"],
        dend_movements=regrouped_data["dend_movements"],
        dend_movement_corr=regrouped_data["dend_movement_corr"],
        dend_movement_num=regrouped_data["dend_movement_num"],
        local_movements=regrouped_data["local_movements"],
        local_movement_corr=regrouped_data["local_movement_corr"],
        local_movement_num=regrouped_data["local_movement_num"],
        global_movements=regrouped_data["global_movements"],
        global_movement_corr=regrouped_data["global_movement_corr"],
        global_movement_num=regrouped_data["global_movement_num"],
        conj_movements=regrouped_data["conj_movements"],
        conj_movement_corr=regrouped_data["conj_movement_corr"],
        conj_movement_num=regrouped_data["conj_movement_num"],
    )

    # Save Section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Desktop\Analyzed_data\grouped"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Set up name to include some parameters
        if movement_epochs is None:
            epoch_name = "session"
        else:
            epoch_name = movement_epochs
        if zscore:
            a_type = "zscore"
        else:
            a_type = "dFoF"
        if volume_norm:
            norm = "_norm"
        else:
            norm = ""

        save_name = f"{day}_{epoch_name}_{a_type}{norm}_spine_coactivity_data"
        save_pickle(save_name, spine_coactivity_data, save_path)

    return spine_coactivity_data


################# DATACLASS #####################
@dataclass
class Spine_Coactivity_Data:
    """Dataclass for storing spine data of a single day following coactivity analysis 
        across all mice in a given group"""

    day: str
    mouse_id: list
    FOV: list
    parameters: dict
    spine_flags: list
    followup_flags: list
    spine_volumes: np.array
    spine_volumes_um: np.array
    followup_volumes: np.array
    followup_volumes_um: np.array
    spine_activity_freq: np.array
    dend_activity_freq: np.array
    movement_spines: np.array
    nonmovement_spines: np.array
    rwd_movement_spines: np.array
    movement_dendrites: np.array
    nonmovement_dendrites: np.array
    rwd_movement_dendrites: np.array
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
    local_spine_coactive_traces: list
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
    spine_movement_num: np.array
    dend_movements: list
    dend_movement_corr: np.array
    dend_movement_num: np.array
    local_movements: list
    local_movement_corr: np.array
    local_movement_num: np.array
    global_movements: list
    global_movement_corr: np.array
    global_movement_num: np.array
    conj_movements: list
    conj_movement_corr: np.array
    conj_movement_num: np.array

