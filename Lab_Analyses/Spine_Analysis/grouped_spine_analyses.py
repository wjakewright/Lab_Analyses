"""Module that handles grouped spine analyses across all mice. Utlizes
    other analyses functions"""

import os
from collections import defaultdict

import numpy as np
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def grouped_longitudinal_spine_volume(
    mice_list,
    days=("Early", "Middle", "Late"),
    followup=False,
    corrected=True,
    threshold=0.25,
    exclude=None,
):
    """Function to handle the longitudinal spine volume analysis across all mcie
    
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
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    # Initialize output
    grouped_data = defaultdict(list)

    # Analyze each mouse
    for mouse in mice_list:
        print(f"--------------------------------\nProcessing {mouse}")
        mouse_data = defaultdict(list)

        data_path = os.path.join(initial_path, mouse, "spine_data")
        FOVs = next(os.walk(data_path))[1]

        # Analyze each FOV
        for FOV in FOVs:
            print(f"--- {FOV}")
            # Load the datasets
            FOV_path = os.path.join(data_path, FOV)
            datasets = []
            fnames = next(os.walk(FOV_path))[2]
            for day in days:
                load_name = [x for x in fnames if day in x and "followup" not in x][0]
                data = load_pickle([load_name], path=FOV_path)[0]
                datasets.append(data)
            used_days = list(days)
            # Add the followup datasets if specified
            if followup:
                followup_datasets = []
                for day in days:
                    followup_name = [x for x in fnames if day in x and "followup" in x][
                        0
                    ]
                    followup_data = load_pickle([followup_name], path=FOV_path)[0]
                    followup_datasets.append(followup_data)
                # Merge the followup data in alternating fashion
                temp_data = datasets
                datasets = [
                    sub[item]
                    for item in range(len(followup_datasets))
                    for sub in [temp_data, followup_datasets]
                ]
                pre_days = [f"Pre {day}" for day in days]
                post_days = [f"Post {day}" for day in days]
                used_days = [
                    sub[item]
                    for item in range(len(post_days))
                    for sub in [pre_days, post_days]
                ]

            # Calculate the volumes
            if corrected:
                _, volumes, spine_idxs = calculate_volume_change(
                    datasets, used_days, exclude=exclude
                )
            else:
                volumes, _, spine_idxs = calculate_volume_change(
                    datasets, used_days, exclude=exclude
                )

            # Store data for this FOV
            for key, value in volumes.items():
                mouse_data[key].append(value)

        # Merge FOV data for this mouse
        merged_data = {}
        for key, value in mouse_data.items():
            if len(value) == 1:
                merged_data[key] = value[0]
                continue
            merged_data[key] = np.concatenate(value)

        # Store data for this mouse
        for key, value in merged_data.items():
            grouped_data[key].append(value)

    # get mean volume change for each mouse
    total_data = {}
    mouse_mean_volume = {}
    mouse_percent_potentiated = {}
    mouse_percent_depressed = {}
    for key, value in grouped_data.items():
        mouse_mean = []
        mouse_potentiated = []
        mouse_depressed = []
        for v in value:
            mouse_mean.append(np.mean(v))
            potentiated_spines, depressed_spines, _ = classify_plasticity(
                v, threshold=threshold
            )
            mouse_potentiated.append(sum(potentiated_spines) / len(potentiated_spines))
            mouse_depressed.append(sum(depressed_spines) / len(depressed_spines))
        mouse_mean_volume[key] = mouse_mean
        mouse_percent_potentiated[key] = mouse_potentiated
        mouse_percent_depressed[key] = mouse_depressed
        total_data[key] = np.concatenate(value)

    return (
        total_data,
        grouped_data,
        mouse_mean_volume,
        mouse_percent_potentiated,
        mouse_percent_depressed,
    )
