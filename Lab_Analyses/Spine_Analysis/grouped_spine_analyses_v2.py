import os
from collections import defaultdict

import numpy as np
from Lab_Analyses.Spine_Analysis.global_coactivity import global_coactivity_analysis
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


def coactivity_spine_volume(
    mice_list, day, corrected=True, threshold=0.25, exclude="Shaft Spine",
):
    """Function to compare the analysis spine volume and co-activity. Combines
        data across FOVs and mice
        
        INPUT PARAMETERS
            mice_list - list of str of mice to load
            
            day - str specifying what sessions you wish to load. Used to find
                the file names
                
            corrected - boolean specifying if you wish to use the corrected spine
                        volume or not
                        
            threshold - float specifying the threshold used to classify plastic spines
            
            exclude - str specifying spine type to exclude from analysis
    """

    grouped_data = defaultdict(list)
    merged_group_data = defaultdict(list)
    for mouse in mice_list:
        print(mouse)
        mouse_datasets = load_spine_datasets(mouse, [day], followup=True)
        mouse_data = defaultdict(list)

        for data in mouse_datasets.values():
            keys = list(data.keys())
            datasets = list(data.values())

            spine_groupings = datasets[0].spine_grouping
            # Get the coactivity results
            (
                global_correlation,
                coactivity_rate,
                spine_fraction_coactive,
                dend_fraction_coactive,
                coactive_amplitude,
                coactive_spines,
                coactivity_epoch_trace,
                coactivity_mean_trace,
                dend_mean_sem,
            ) = global_coactivity_analysis(datasets[0], sampling_rate=60)

            # convert coactivity_mean_trace dict to list
            coactivity_mean_trace = list(coactivity_mean_trace.values())
            coactivity_epoch_trace = list(coactivity_epoch_trace.values())

            # Get the spine volumes
            if corrected:
                _, volumes, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            else:
                volumes, _, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            # Use only the post volumes
            volumes = volumes[keys[1]]

            # classify plasticity
            potentiated, depressed, _ = classify_plasticity(volumes, threshold)

            # Get the coactivity only for the stable spines
            global_correlation = global_correlation[spine_idxs]
            coactivity_rate = coactivity_rate[spine_idxs]
            spine_fraction_coactive = spine_fraction_coactive[spine_idxs]
            dend_fraction_coactive = dend_fraction_coactive[spine_idxs]
            coactive_amplitude = coactive_amplitude[spine_idxs]
            coactive_spines = coactive_spines[spine_idxs]
            groupings = []
            if type(spine_groupings[0]) == list:
                for grouping in spine_groupings:
                    g = [x for x in grouping if x in spine_idxs]
                    groupings.append(g)
            else:
                groupings.append(spine_groupings)
            coactivity_epoch_trace = [coactivity_epoch_trace[i] for i in spine_idxs]
            coactivity_mean_trace = [coactivity_mean_trace[i] for i in spine_idxs]

            # store values
            mouse_data["volumes"].append(volumes)
            mouse_data["potentiated"].append(potentiated)
            mouse_data["depressed"].append(depressed)
            mouse_data["correlation"].append(global_correlation)
            mouse_data["coactivity_rate"].append(coactivity_rate)
            mouse_data["spine fraction"].append(spine_fraction_coactive)
            mouse_data["dend fraction"].append(dend_fraction_coactive)
            mouse_data["coactive amplitude"].append(coactive_amplitude)
            mouse_data["coactive spines"].append(coactive_spines)
            mouse_data["epoch traces"].append(coactivity_epoch_trace)
            mouse_data["mean traces"].append(coactivity_mean_trace)
            mouse_data["dend traces"].append(dend_mean_sem)
            mouse_data["grouping"].append(groupings)

        # Merge mouse data across FOVs
        merged_mouse_data = {}
        for key, value in mouse_data.items():
            if (
                key == "grouping"
                or key == "dend traces"
                or key == "mean traces"
                or key == "epoch traces"
            ):
                continue
            if len(value) == 1:
                merged_mouse_data[key] = value[0]
                continue
            merged_mouse_data[key] = np.concatenate(value)

        # store data for this mouse
        for key, value in mouse_data.items():
            grouped_data[key].append(value)
        for key, value in merged_mouse_data.items():
            merged_group_data[key].append(value)

    # Merge data across mice
    total_data = {}
    for key, value in merged_group_data.items():
        total_data[key] = np.concatenate(value)

    return total_data, grouped_data

