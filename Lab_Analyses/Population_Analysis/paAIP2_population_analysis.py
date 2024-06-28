import os

import numpy as np

from Lab_Analyses.Population_Analysis import population_utilities as p_utils
from Lab_Analyses.Population_Analysis.assess_MRN_activity import (
    calculate_movement_encoding,
    get_fraction_MRNs,
)
from Lab_Analyses.Population_Analysis.paAIP2_population_dataclass import (
    paAIP2_Population_Data,
)
from Lab_Analyses.Population_Analysis.population_vector_analysis import (
    pca_population_vector_analysis,
    simple_population_vector_analysis,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_related_activity_v2 import (
    get_movement_onsets,
    movement_related_activity,
)


def paAIP2_population_analysis(
    paAIP2_mice,
    EGFP_mice,
    activity_window=(-2, 4),
    save_ind=False,
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

    """
    # only save individual datasets
    all_mice = paAIP2_mice + EGFP_mice

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
        sampling_rate = imaging_parameters["Sampling Rate"]
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
            x.reward_movement_cells_spikes for x in list(datasets.values())
        ]
        ## Activity data
        dFoF = [x.processed_dFoF for x in list(datasets.values())]
        spikes = [x.estimated_spikes for x in list(datasets.values())]
        smooth_spikes = [x.processed_estimated_spikes for x in list(datasets.values())]
        activity = [x.activity_trace for x in list(datasets.values())]
        ## z-scored activity
        zscore_dFoF = [d_utils.z_score(x) for x in dFoF]
        zscore_spikes = [d_utils.z_score(x) for x in spikes]
        zscore_smooth_spikes = [d_utils.z_score(x) for x in smooth_spikes]

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

        # Set up some output dict
        movement_traces_dFoF = {}
        movement_amplitudes_dFoF = {}
        mean_onsets_dFoF = {}
        movement_traces_spikes = {}
        movement_amplitudes_spikes = {}
        mean_onsets_spikes = {}
        individual_mvmt_onsets = {}
        mvmt_onset_jitter = {}
        avg_pop_vector = {}
        event_pop_vectors = {}
        all_similarities = {}
        med_vector_similarity = {}
        med_vector_correlation = {}

        for i, session in enumerate(sessions):
            # Get movement related activity
            mvmt_traces_dFoF, mvmt_amplitudes_dFoF, m_onsets_dFoF = (
                movement_related_activity(
                    lever_active=lever_active[i],
                    activity=activity[i],
                    dFoF=zscore_dFoF[i],
                    norm=None,
                    avg_window=None,
                    sampling_rate=sampling_rate,
                    activity_window=activity_window,
                )
            )
            movement_traces_dFoF[session] = mvmt_traces_dFoF
            movement_amplitudes_dFoF[session] = mvmt_amplitudes_dFoF
            mean_onsets_dFoF[session] = m_onsets_dFoF
            mvmt_traces_spikes, mvmt_amplitudes_spikes, m_onsets_spikes = (
                movement_related_activity(
                    lever_active=lever_active[i],
                    activity=activity[i],
                    dFoF=zscore_spikes[i],
                    norm=None,
                    avg_window=None,
                    sampling_rate=sampling_rate,
                    activity_window=activity_window,
                )
            )
            movement_traces_spikes[session] = mvmt_traces_spikes
            movement_amplitudes_spikes[session] = mvmt_amplitudes_spikes
            mean_onsets_spikes[session] = m_onsets_spikes

            # Get individual movement onsets and jitter
            ind_mvmt_onsets, m_onset_jitter = get_movement_onsets(
                lever_active=lever_active[i],
                activity=activity[i],
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            individual_mvmt_onsets[session] = ind_mvmt_onsets
            mvmt_onset_jitter[session] = m_onset_jitter

            # Analyze population level activity
            ## Needs to be only MRNs
            MRNs = movement_cells_spikes[i]
            pop_vector, evt_pop_vectors, similarities, med_similarity = (
                pca_population_vector_analysis(
                    dFoF=zscore_smooth_spikes[i][:, MRNs],
                    lever_active=lever_active_rwd[i],
                    activity_window=(-0.5, 1.5),
                    sampling_rate=sampling_rate,
                    n_comps=10,
                )
            )
            avg_pop_vector[session] = pop_vector
            event_pop_vectors[session] = evt_pop_vectors
            all_similarities[session] = similarities
            med_vector_similarity[session] = med_similarity

            med_correlation = simple_population_vector_analysis(
                dFoF=zscore_smooth_spikes[i][:, MRNs],
                lever_active=lever_active_rwd[i],
                activity_window=(-0.5, 1.5),
                sampling_rate=sampling_rate,
            )
            med_vector_correlation[session] = med_correlation

        parameters = {
            "Sampling Rate": sampling_rate,
            "Activity Window": activity_window,
        }

        # Store individual data
        ## Check group
        if mouse in paAIP2_mice:
            group = "paAIP2"
        elif mouse in EGFP_mice:
            group = "EGFP"
        ## Store the data
        individual_data = paAIP2_Population_Data(
            mouse_id=mouse,
            group=group,
            sessions=sessions,
            parameters=parameters,
            lever_active=lever_active,
            lever_force=lever_force,
            lever_active_rwd=lever_active_rwd,
            zscore_dFoF=zscore_dFoF,
            zscore_spikes=zscore_spikes,
            zscore_smooth_spikes=zscore_smooth_spikes,
            mvmt_cells_dFoF=movement_cells_dFoF,
            mvmt_cells_spikes=movement_cells_spikes,
            cell_activity_rate=cell_activity_rate,
            fraction_MRNs_dFoF=fraction_MRNs_dFoF,
            fraction_MRNs_spikes=fraction_MRNs_spikes,
            fraction_rMRNs_dFoF=fraction_rMRNs_dFoF,
            fraction_rMRNs_spikes=fraction_rMRNs_spikes,
            fraction_silent_dFoF=fraction_silent_dFoF,
            fraction_silent_spikes=fraction_silent_spikes,
            movement_correlation=movement_correlation,
            movement_stereotypy=movement_stereotypy,
            movement_reliability=movement_reliability,
            movement_specificity=movement_specificity,
            movement_traces_dFoF=movement_traces_dFoF,
            movement_amplitudes_dFoF=movement_amplitudes_dFoF,
            mean_onsets_dFoF=mean_onsets_dFoF,
            movement_traces_spikes=movement_traces_spikes,
            movement_amplitudes_spikes=movement_amplitudes_spikes,
            mean_onsets_spikes=mean_onsets_spikes,
            individual_mvmt_onsets=individual_mvmt_onsets,
            mvmt_onset_jitter=mvmt_onset_jitter,
            avg_pop_vector=avg_pop_vector,
            event_pop_vectors=event_pop_vectors,
            med_vector_similarity=med_vector_similarity,
            med_vector_correlation=med_vector_correlation,
        )
        ## Save individual if specified
        if save_ind:
            individual_data.save()
