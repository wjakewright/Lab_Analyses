import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Spine_Analysis.global_coactivity import global_coactivity_analysis
from Lab_Analyses.Spine_Analysis.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)


def grouped_coactivity_analysis(
    mice_list,
    days=("Early", "Middle", "Late"),
    followup=False,
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
                mice_list, day, corrected, threshold, exclude
            )
            final_datasets[day] = short_term_data
    else:
        for i in len(days[:-1]):
            day = [days[i], days[i + 1]]
            print(f"- {day[1]}")
            short_term_data = short_term_coactivity_analysis(
                mice_list, day, corrected, threshold, exclude
            )
            final_datasets[day[1]] = short_term_data


def short_term_coactivity_analysis(mice_list, day, corrected, threhold, exclude):
    """Function to handle the short term analysis of coactivity datasets"""

    grouped_data = defaultdict(list)

    for mouse in mice_list:
        print(f"--- {mouse}")
        if type(day) == str:
            mouse_datasets = load_spine_datasets(mouse, [day], followup=True)
        elif type(day) == list:
            mouse_datasets = load_spine_datasets(mouse, day, followup=False)
