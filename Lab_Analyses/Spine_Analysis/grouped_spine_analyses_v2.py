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
    movement_volumes = defaultdict(list)
    activity_volumes = defaultdict(list)
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
