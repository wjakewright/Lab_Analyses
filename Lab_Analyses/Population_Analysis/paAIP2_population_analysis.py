import os

import numpy as np

from Lab_Analyses.Population_Analysis import population_utilities as p_utils
from Lab_Analyses.Population_Analysis.assess_MRN_activity import (
    calculate_movement_encoding,
    get_fraction_MRNs,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def paAIP2_population_analysis(
    paAIP2_mice, EGFP_mice, activity_window=(-2, 4), save_ind=False, save_grouped=False,
):
    """Function to analyze experimental and control mice from paAIP2 population
        imaging experiments
        
        INPUT PARAMETERS
            paAIP2_mice - list of str specifying the mice in the paAIP2 group to be
                          analyzed
            
            EGFP_mice - list of str specifying the mice in the EGFP group to be
                        analyzed
            
            activity_window - tuple specifying the window size to analyze activity around

            save_ind - boolean specifying whetehr to save the data for each mouse

            save_grouped - boolean specifying whether or not to group all mice together
                            and save
    """

    all_mice = paAIP2_mice + EGFP_mice
    paAIP2_data = []
    EGFP_data = []

    sessions = ["Early", "Middle", "Late"]

    # Start analyzing each mouse seperately
    for mouse in all_mice:
        print("----------------------------------------")
        print(f"- Analyzing {mouse}")
        # Load the datasets
        datasets = p_utils.load_population_datasets(mouse, sessions)

        # Pull relevant data into lists
        ## Parameters
        imaging_parameters = datasets["Early"].imaging_parameters
        matched = imaging_parameters["Matched"]
        ## Lever related data
        lever_active = [x.lever_active for x in list(datasets.values())]
        lever_force = [x.lever_force_smooth for x in list(datasets.values())]
        lever_active_rwd = [x.rewarded_movement_binary for x in list(datasets.values())]
        ## Cell identifiers
        cell_positions = [x.cell_positions for x in list(datasets.values())]
        movement_cells_dFoF = [x.movement_cells for x in list(datasets.values())]
        silent_cells_dFoF = [x.silent_cells for x in list(datasets.values())]
        rwd_movement_cells_dFoF = [
            x.reward_movement_cells for x in list(datasets.values())
        ]
        movement_cells_spikes = [
            x.movement_cells_spikes for x in list(datasets.values())
        ]
        silent_cells_spikes = [x.silent_cells_spikes for x in list(datasets.values())]
        rwd_movement_cells_spikes = [
            x.reward_movement_spikes for x in list(datasets.values())
        ]
        ## Activity data
        dFoF = [x.processed_dFoF for x in list(datasets.values())]
        spikes = [x.estimated_spikes for x in list(datasets.values())]
        smooth_spikes = [x.processed_estimated_spikes for x in list(datasets.values())]
        activity = [x.activity_trace for x in list(datasets.values())]
        ## z-scored activity
        zscore_dFoF = d_utils.z_score(dFoF)
        zscore_spikes = d_utils.z_score(spikes)
        zscore_smooth_spikes = d_utils.z_score(smooth_spikes)

        # Get overall activity rates for each cell
        cell_activity_rate = {}
        for session, events in zip(sessions, activity):
            cell_activity_rate[session] = d_utils.calculate_activity_event_rate(events)

        # Get fractions of cell types
        fraction_MRNs_dFoF = get_fraction_MRNs(sessions, movement_cells_dFoF)
        fraction_MRNs_spikes = get_fraction_MRNs(sessions, movement_cells_spikes)
        fraction_rMRNs_dFoF = get_fraction_MRNs(sessions, rwd_movement_cells_dFoF)
        fraction_rMRNs_spikes = get_fraction_MRNs(sessions, rwd_movement_cells_spikes)
        fraction_silent_dFoF = get_fraction_MRNs(sessions, silent_cells_dFoF)
        fraction_silent_spikes = get_fraction_MRNs(sessions, silent_cells_spikes)

        # Get the movement encoding properties of cells
        (
            movement_correlation,
            movement_stereotypy,
            movement_reliability,
            movement_specificity,
        ) = calculate_movement_encoding(
            mouse,
            sessions,
            activity,
            lever_active,
            lever_force,
            threshold=0.5,
            corr_duration=0.5,
            sampling_rate=imaging_parameters["Sampling Rate"],
        )

