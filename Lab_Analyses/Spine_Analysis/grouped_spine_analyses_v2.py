import os
from collections import defaultdict

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)


def grouped_longitudinal_spine_volume(
    mice_list,
    days=("Early", "Middle", "Late"),
    followup=False,
    corrected=True,
    threshold=0.25,
    exclude="Shaft Spine",
):
    """Function to handle the longitudinal spine volume analysis across all mice
        This function only calculates overall structural plasticity and merges FOVs
        for each mouse

        INPUT PARAMETERS
            mice_list - list of strings specifying mice ids

            days - list of days you wish to analyze. Must be in file name
            
            followup - boolean specifying if you want to include followup
                        imaging sessions in the analysis

            corrected - boolean of whether or not to use the corrected volume estimates

            threshold - float specifying threshold for plasticity classification

            exclude - str specifying spine types to exclude from analysis
                        
        OUTPUT PARAMETERS
    """

    grouped_data = defaultdict(list)

    for mouse in mice_list:
        # Load the datasets for this mouse
        mouse_datasets = load_spine_datasets(mouse, days, followup)
        mouse_data = defaultdict(list)

        for FOV, data in mouse_datasets.items():
            used_days = list(data.keys())
            datasets = list(data.values())

            # Calculate volume changes
            if corrected:
                _, volumes, _ = calculate_volume_change(
                    datasets, used_days, exclude=exclude
                )
            else:
                volumes, _, _ = calculate_volume_change(
                    datasets, used_days, exclude=exclude,
                )
            for key, value in volumes.items():
                mouse_data[key].append(value)

        # Merge FOVs
        for key, value in mouse_data.items():
            if len(value) == 1:
                mouse_data[key] = value[0]
                continue
            mouse_data[key] = np.concatenate(value)

        # Store data for this mouse
        for key, value in mouse_data.items():
            grouped_data[key].append(value)

    # Get mean volume change for each mouse
    total_data = {}
    mouse_mean_volume = {}
    mouse_percent_potentiated = {}
    mouse_percent_depressed = {}
    for key, value in grouped_data.items():
        total_data[key] = np.concatenate(value)
        mouse_mean = []
        mouse_potentiated = []
        mouse_depressed = []
        for v in value:
            mouse_mean.append(np.mean(v))
            potentiated, depressed, _ = classify_plasticity(v, threshold=threshold)
            mouse_potentiated.append(sum(potentiated) / len(potentiated))
            mouse_depressed.append(sum(depressed) / len(depressed))
        mouse_mean_volume[key] = mouse_mean
        mouse_percent_potentiated[key] = mouse_potentiated
        mouse_percent_depressed[key] = mouse_depressed

    return (
        total_data,
        grouped_data,
        mouse_mean_volume,
        mouse_percent_potentiated,
        mouse_percent_depressed,
    )


def movement_spine_volume(
    mice_list,
    day,
    corrected=True,
    threshold=0.25,
    exclude="Shaft Spine",
    rewarded=False,
):
    """Function to compare spine volumes for movement vs. non-movement spines
        Only looks at two sessions. Combines data across all FOVs and mice
        
        INNPUT PARAMETERS
            mice_list - list of strings of mice to load
            
            day - str specifying what session you wish to load. Used to 
                    find the file names
            
            corrected - boolean specifying if you wish to use the corrected spine
                        volume or not
            
            threshold - float specifying the threshold used to classify plastic spines
            
            exclude - str specifying spine type to exclude from the analysis

            rewarded - boolean specifying if you wish to looked at reward movement spines
            
    """
    grouped_data = defaultdict(list)
    for mouse in mice_list:
        mouse_datasets = load_spine_datasets(mouse, [day], followup=True)
        mouse_data = defaultdict(list)

        for data in mouse_datasets.values():
            keys = list(data.keys())
            datasets = list(data.values())

            # Get the volume
            if corrected:
                _, volumes, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude
                )
            else:
                volumes, _, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            # use only the post volumes
            volumes = volumes[keys[1]]
            # Get the movemet idxs
            if not rewarded:
                movement_idxs = np.array(datasets[0].movement_spines)
                spine_activity = np.array(datasets[0].spine_movement_activity["Real"])
            else:
                movement_idxs = np.array(datasets[0].reward_movement_spines)
                spine_activity = np.array(datasets[0].spine_reward_activity["Real"])

            # Refine idxs to only those analyzed
            movement_idxs = movement_idxs[spine_idxs]
            spine_activity = spine_activity[spine_idxs]

            mouse_data["volumes"].append(volumes)
            mouse_data["move_idx"].append(movement_idxs)
            mouse_data["activity"].append(spine_activity)

        # Merge FOVs for this mouse
        for key, value in mouse_data.items():
            if len(value) == 1:
                mouse_data[key] = value[0]
                continue
            mouse_data[key] = np.concatenate(value)

        # Store data for this mouse
        for key, value in mouse_data.items():
            grouped_data[key].append(value)

    # Merged data across mice
    total_data = {}
    for key, value in grouped_data.items():
        total_data[key] = np.concatenate(value)

    # Get all the volumes for movement and nonmovment spines
    movement_volumes = {"movement": [], "nonmovement": []}
    move_vol = total_data["volumes"][total_data["move_idx"]]
    non_idx = [not x for x in total_data["move_idx"]]
    non_move_vol = total_data["volumes"][non_idx]
    movement_volumes["movement"] = move_vol
    movement_volumes["nonmovement"] = non_move_vol

    # Get means and percentages for each mouse
    mouse_means = {"movement": [], "nonmovement": []}
    mouse_potentiated = {"movement": [], "nonmovement": []}
    mouse_depressed = {"movement": [], "nonmovement": []}

    for vol, move in zip(grouped_data["volumes"], grouped_data["move_idx"]):
        non_idx = [not x for x in move]
        move_vols = vol[move]
        non_move_vols = vol[non_idx]
        move_pot, move_dep, _ = classify_plasticity(move_vols, threshold)
        non_pot, non_dep, _ = classify_plasticity(non_move_vols, threshold)
        mouse_means["movement"].append(np.mean(move_vols))
        mouse_means["nonmovement"].append(np.mean(non_move_vols))
        mouse_potentiated["movement"].append(sum(move_pot) / len(move_pot))
        mouse_potentiated["nonmovement"].append(sum(non_pot) / len(non_pot))
        mouse_depressed["movement"].append(sum(move_dep) / len(move_dep))
        mouse_depressed["nonmovement"].append(sum(non_dep) / len(non_dep))

    return total_data, movement_volumes, mouse_means, mouse_potentiated, mouse_depressed