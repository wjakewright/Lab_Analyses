import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Spine_Analysis.global_coactivity import global_coactivity_analysis
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import (
    assess_movement_quality,
    spine_movement_activity,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def grouped_coactivity_analysis(
    mice_list,
    days=("Early", "Middle", "Late"),
    followup=False,
    movement_epochs=None,
    corrected=True,
    threshold=0.5,
    exclude="Shaft Spine",
):
    """Function to handle the activity and structural analysis of dual spine imaging datasets
        across all mice and all FOVs. Analyzes both short term changes as well as across the 
        entire experiment
        
        INPUT PARAMETERS
            mice_list - list of strings specifying mice ids to be analyzed
            
            days - list of days you wish to analyze. Must be in the file name
            
            followup - boolean specifying if you want to include followup imaging sessions
                        in the analysis. This will influence the short term analyses. 
            
            movements - str specifying if you want on analyze only during movements and of
                        different types of movements. Accepts "all", "rewarded", "unrewarded", 
                        and "nonmovement".
                        Default is None, analyzing the entire imaging period
            
            corrected - boolean of whether or not to use the corrected volume estimates
            
            threshold - float specifying threshold for plasticity classification
            
            exclude - str specifying spine types to exclude from analysis
            
        OUTPUT PARAMETERS
    """

    final_datasets = {}
    print("Analyzing short term data")
    if followup:
        for day in days:
            print(f"- {day}")
            short_term_data = short_term_coactivity_analysis(
                mice_list, day, corrected, movement_epochs, threshold, exclude
            )
            final_datasets[day] = short_term_data
    else:
        for i in len(days[:-1]):
            day = [days[i], days[i + 1]]
            print(f"- {day[1]}")
            short_term_data = short_term_coactivity_analysis(
                mice_list, day, corrected, movement_epochs, threshold, exclude
            )
            final_datasets[day[1]] = short_term_data


def short_term_coactivity_analysis(
    mice_list, day, movement_epochs, corrected, threshold, exclude
):
    """Function to handle the short term analysis of coactivity datasets"""

    grouped_data = defaultdict(list)

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"--- {mouse}")
        if type(day) == str:
            mouse_datasets = load_spine_datasets(mouse, [day], followup=True)
        elif type(day) == list:
            mouse_datasets = load_spine_datasets(mouse, day, followup=False)
        # Analyze each FOV seperately
        for FOV, data in mouse_datasets.items():
            keys = list(data.keys())
            datasets = list(data.values())
            # Get the spine groupings with parent dendrite
            spine_groupings = datasets[0].spine_grouping
            if type(spine_groupings[0]) != list:
                spine_groupings = [spine_groupings]
            # Get the coactivity data
            (
                global_correlation,
                coactivity_rate,
                spine_fraction_coactive,
                dend_fraction_coactive,
                coactive_amplitude,
                coactive_spines,
                coactivity_epoch_traces,
                coactivity_mean_traces,
                dend_mean_sems,
                spine_onsets,
                relative_onsets,
                dend_onsets,
            ) = global_coactivity_analysis(
                datasets[0], movement_epochs, sampling_rate=60
            )

            # convert coactivity trace data to lists
            coactivity_mean_traces = list(coactivity_mean_traces.values())
            coactivity_epoch_traces = list(coactivity_epoch_traces.values())
            ## duplicate the dendrite traces to match with their children spines
            dend_mean_traces = list(np.zeros(len(coactivity_mean_traces)))
            for i, grouping in enumerate(spine_groupings):
                for g in grouping:
                    dend_mean_traces[g] = dend_mean_sems[i]

            # get spine volumes
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

            # Classify the plasticity
            potentiated, depressed, stable = classify_plasticity(volumes, threshold)
            movement_spines = np.array(datasets[0].movement_spines)
            reward_movement_spines = np.array(datasets[0].reward_movement_spines)
            nonreward_movement_spines = movement_spines.astype(
                int
            ) - reward_movement_spines.astype(int)
            nonreward_movement_spines[nonreward_movement_spines == -1] = 0
            nonreward_movement_spines = nonreward_movement_spines.astype(bool)

            # Assess movement activity
            all_befores, all_durings, _, movement_traces = spine_movement_activity(
                datasets[0],
                activity_type="spine_GluSnFr_processed_dFoF",
                exclude=None,
                sampling_rate=60,
                rewarded=False,
            )
            (
                rwd_befores,
                rwd_durings,
                _,
                reward_movement_traces,
            ) = spine_movement_activity(
                data=datasets[0],
                activity_type="spine_GluSnFr_processed_dFoF",
                exclude=None,
                sampling_rate=60,
                rewarded=True,
            )
            movement_amps = [
                np.nanmean(before - after)
                for before, after in zip(all_befores, all_durings)
            ]
            reward_movement_amps = [
                np.nanmean(rwd_before - rwd_after)
                for rwd_before, rwd_after in zip(rwd_befores, rwd_durings)
            ]
            movement_traces = list(movement_traces.values())
            reward_movement_traces = list(reward_movement_traces)

            # Assess movement quality
            (
                spine_movements,
                _,
                spine_movement_correlations,
                learned_movement,
            ) = assess_movement_quality(
                datasets[0],
                activity_type="spine_GluSnFr_activity",
                coactivity=False,
                exclude=None,
                sampling_rate=60,
                rewarded=False,
            )
            (
                coactive_movements,
                _,
                coactive_movement_correlations,
                _,
            ) = assess_movement_quality(
                datasets[0],
                activity_type="spine_GluSnFr_activity",
                coactivity=True,
                exclude=None,
                sampling_rate=60,
                reawrded=False,
            )

