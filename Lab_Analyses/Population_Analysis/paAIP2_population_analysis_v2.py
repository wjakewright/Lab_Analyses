import os

import numpy as np

from Lab_Analyses.Population_Analysis import population_utilities as p_utils
from Lab_Analyses.Population_Analysis.assess_MRN_activity import (
    calculate_movement_encoding,
    get_fraction_MRNs,
)
from Lab_Analyses.Population_Analysis.paAIP2_population_dataclass_v2 import (
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
from Lab_Analyses.Utilities.movement_responsiveness_v2 import movement_responsiveness
from Lab_Analyses.Utilities.spike_event_detection import cascade_event_detection


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
        datasets, spike_sets = p_utils.load_population_datasets(mouse, sessions)

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

        ## Activity data
        dFoF = [x.processed_dFoF for x in list(datasets.values())]
        spikes = [x for x in list(spike_sets.values())]
        ## Event detection
        print("--- Event detection")
        activity = []
        floored = []
        for spike in spikes:
            act, flr = cascade_event_detection(spike, cutoff=0.2)
            activity.append(act)
            floored.append(flr)
        ## z-scored activity
        zscore_dFoF = [d_utils.z_score(x) for x in dFoF]
        zscore_spikes = [d_utils.z_score(x) for x in spikes]

        # Get overall activity rates for each cell
        cell_activity_rate = {}
        for session, events in zip(sessions, activity):
            cell_activity_rate[session] = d_utils.calculate_activity_event_rate(events)

        # Determine movement related neurons
        print(f"--- Movement responsiveness")
        movement_cells_spikes = []
        silent_cells_spikes = []
        rwd_movement_cells_spikes = []
        for i, spike in enumerate(spikes):
            mvmt_cells_spikes, sil_cells_spikes, _ = movement_responsiveness(
                spike,
                lever_active[i],
                permutations=1000,
                percentile=99,
            )
            rwd_mvmt_cells_spikes, _, _ = movement_responsiveness(
                spike,
                lever_active_rwd[i],
                permutations=1000,
                percentile=99,
            )
            movement_cells_spikes.append(mvmt_cells_spikes)
            silent_cells_spikes.append(sil_cells_spikes)
            rwd_movement_cells_spikes.append(rwd_mvmt_cells_spikes)

        # Get fractions of cell types
        print("--- Movement related activity")
        fraction_MRNs_spikes = get_fraction_MRNs(sessions, movement_cells_spikes)
        fraction_rMRNs_spikes = get_fraction_MRNs(sessions, rwd_movement_cells_spikes)
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
        movement_traces_spikes = {}
        rwd_movement_traces_spikes = {}
        movement_amplitudes_spikes = {}
        rwd_movement_amplitudes_spikes = {}
        mean_onsets_spikes = {}
        rwd_mean_onsets_spikes = {}
        individual_mvmt_onsets = {}
        individual_rwd_mvmt_onsets = {}
        mvmt_onset_jitter = {}
        rwd_mvmt_onset_jitter = {}
        avg_pop_vector = {}
        event_pop_vectors = {}
        all_similarities = {}
        med_vector_similarity = {}
        med_vector_correlation = {}

        for i, session in enumerate(sessions):
            # Get movement related activity
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

            rwd_mvmt_traces_spikes, rwd_mvmt_amplitudes_spikes, rwd_m_onsets_spikes = (
                movement_related_activity(
                    lever_active=lever_active_rwd[i],
                    activity=activity[i],
                    dFoF=zscore_spikes[i],
                    norm=None,
                    avg_window=None,
                    sampling_rate=sampling_rate,
                    activity_window=activity_window,
                )
            )
            rwd_movement_traces_spikes[session] = rwd_mvmt_traces_spikes
            rwd_movement_amplitudes_spikes[session] = rwd_mvmt_amplitudes_spikes
            rwd_mean_onsets_spikes[session] = rwd_m_onsets_spikes

            # Get individual movement onsets and jitter
            ind_mvmt_onsets, m_onset_jitter = get_movement_onsets(
                lever_active=lever_active[i],
                activity=activity[i],
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            individual_mvmt_onsets[session] = ind_mvmt_onsets
            mvmt_onset_jitter[session] = m_onset_jitter

            ind_rwd_mvmt_onsets, rwd_m_onset_jitter = get_movement_onsets(
                lever_active=lever_active_rwd[i],
                activity=activity[i],
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            individual_rwd_mvmt_onsets[session] = ind_rwd_mvmt_onsets
            rwd_mvmt_onset_jitter[session] = rwd_m_onset_jitter

            # Analyze population level activity
            ## Needs to be only MRNs
            MRNs = movement_cells_spikes[i]
            pop_vector, evt_pop_vectors, similarities, med_similarity = (
                pca_population_vector_analysis(
                    dFoF=zscore_spikes[i][:, MRNs],
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
                dFoF=zscore_spikes[i][:, MRNs],
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
            dFoF=dFoF,
            spikes=spikes,
            zscore_dFoF=zscore_dFoF,
            zscore_spikes=zscore_spikes,
            activity=activity,
            floored=floored,
            mvmt_cells_spikes=movement_cells_spikes,
            rwd_mvmt_cells_spikes=rwd_movement_cells_spikes,
            cell_activity_rate=cell_activity_rate,
            fraction_MRNs_spikes=fraction_MRNs_spikes,
            fraction_rMRNs_spikes=fraction_rMRNs_spikes,
            fraction_silent_spikes=fraction_silent_spikes,
            movement_correlation=movement_correlation,
            movement_stereotypy=movement_stereotypy,
            movement_reliability=movement_reliability,
            movement_specificity=movement_specificity,
            movement_traces_spikes=movement_traces_spikes,
            rwd_movement_traces_spikes=rwd_movement_traces_spikes,
            movement_amplitudes_spikes=movement_amplitudes_spikes,
            rwd_movement_amplitude_spikes=rwd_movement_amplitudes_spikes,
            mean_onsets_spikes=mean_onsets_spikes,
            rwd_mean_onsets_spikes=rwd_mean_onsets_spikes,
            individual_mvmt_onsets=individual_mvmt_onsets,
            individual_rwd_mvmt_onsets=individual_rwd_mvmt_onsets,
            mvmt_onset_jitter=mvmt_onset_jitter,
            rwd_mvmt_onset_jitter=rwd_mvmt_onset_jitter,
            avg_pop_vector=avg_pop_vector,
            event_pop_vectors=event_pop_vectors,
            med_vector_similarity=med_vector_similarity,
            med_vector_correlation=med_vector_correlation,
        )
        ## Save individual if specified
        if save_ind:
            individual_data.save()
