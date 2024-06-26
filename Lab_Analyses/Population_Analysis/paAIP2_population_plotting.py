import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Utilities import data_utilities as d_utils

sns.set()
sns.set_style("ticks")


def plot_paAIP2_population_dynamics(
    EGFP_data_list,
    paAIP2_data_list,
    norm=False,
    mvmt_only=False,
    spikes=True,
    example_pa=None,
    example_gfp=None,
    figsize=(10, 10),
    save=False,
    save_path=None,
):
    """
    Function to compare the population dynamics of EGFP and paAIP2 groups

    INPUT PARAMETERS
        EGFP_data_list - list of paAIP2_Population_Data objects for EGFP mice

        paAIP2_data_list - list of paAIP2_Population_Data objects for paAIP2 mice

        mvmt_only - boolean specifying whether to only include MRNs

        spikes - boolean specifying whether to use data derived from dFoF or spikes

        example_pa - str specifying the id of an example mouse for paAIP2 group

        example_gfp - str specifying the id of an example mouse for EGFP group

        figsize - tuple specifying the size of the figure

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure
    """
    COLORS = ["black", "mediumslateblue"]
    # Grab specific parameters
    sessions = EGFP_data_list[0].sessions
    sampling_rate = EGFP_data_list[0].parameters["Sampling Rate"]
    activity_window = EGFP_data_list[0].parameters["Activity Window"]

    if spikes:
        activity_type = "Estimated spikes (zscore)"
    else:
        activity_type = "\u0394F/F (zscore)"

    # Organize the data
    if spikes:
        (
            EGFP_mvmt_traces,
            EGFP_frac_mvmt,
            EGFP_frac_silent,
            EGFP_frac_rwd,
            EGFP_event_rate,
            EGFP_mvmt_amplitude,
            EGFP_mvmt_avg_onset,
            EGFP_mvmt_onset_jitter,
            EGFP_vector_similarity,
            EGFP_vector_correlation,
        ) = organize_pop_data_spikes(sessions, EGFP_data_list, mvmt_only, norm)
        (
            paAIP_mvmt_traces,
            paAIP_frac_mvmt,
            paAIP_frac_silent,
            paAIP_frac_rwd,
            paAIP_event_rate,
            paAIP_mvmt_amplitude,
            paAIP_mvmt_avg_onset,
            paAIP_mvmt_onset_jitter,
            paAIP_vector_similarity,
            paAIP_vector_correlation,
        ) = organize_pop_data_spikes(sessions, paAIP2_data_list, mvmt_only, norm)
    else:
        (
            EGFP_mvmt_traces,
            EGFP_frac_mvmt,
            EGFP_frac_silent,
            EGFP_frac_rwd,
            EGFP_event_rate,
            EGFP_mvmt_amplitude,
            EGFP_mvmt_avg_onset,
            EGFP_mvmt_onset_jitter,
            EGFP_vector_similarity,
            EGFP_vector_correlation,
        ) = organize_pop_data_dFoF(sessions, EGFP_data_list, mvmt_only, norm)
        (
            paAIP_mvmt_traces,
            paAIP_frac_mvmt,
            paAIP_frac_silent,
            paAIP_frac_rwd,
            paAIP_event_rate,
            paAIP_mvmt_amplitude,
            paAIP_mvmt_avg_onset,
            paAIP_mvmt_onset_jitter,
            paAIP_vector_similarity,
            paAIP_vector_correlation,
        ) = organize_pop_data_dFoF(sessions, paAIP2_data_list, mvmt_only, norm)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDEF
        GHIJK.
        LMNO..
        PQRS..
        TUVW..
        """,
        figsize=figsize,
    )
    fig.suptitle("paAIP2 Population Activity Dynamics")

    ################### Plot data onto the axes ######################
    # Heatmaps
    ## EGFP early
    print(f"EGFP 1: {EGFP_mvmt_traces[0].shape[1]}")
    plot_activity_heatmap(
        EGFP_mvmt_traces[0],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Early",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## EGFP middle
    print(f"EGFP 7: {EGFP_mvmt_traces[1].shape[1]}")
    plot_activity_heatmap(
        EGFP_mvmt_traces[1],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Middle",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## EGFP late
    print(f"EGFP 14: {EGFP_mvmt_traces[2].shape[1]}")
    plot_activity_heatmap(
        EGFP_mvmt_traces[2],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Late",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## paAIP2 early
    print(f"paAIP2 1: {paAIP_mvmt_traces[0].shape[1]}")
    plot_activity_heatmap(
        paAIP_mvmt_traces[0],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Early",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## paAIP2 middle
    print(f"paAIP2 7: {paAIP_mvmt_traces[1].shape[1]}")
    plot_activity_heatmap(
        paAIP_mvmt_traces[1],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Middle",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## paAIP2 late
    print(f"paAIP2 4: {paAIP_mvmt_traces[2].shape[1]}")
    plot_activity_heatmap(
        paAIP_mvmt_traces[2],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Late",
        cbar_label=activity_type,
        hmap_range=(0.4, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["I"],
        save=False,
        save_path=None,
    )

    # Longitudinal quantification
    ## Fraction movement neurons
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_mvmt,
            "paAIP2": paAIP_frac_mvmt,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. MRNs",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Fraction silent neurons
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_silent,
            "paAIP2": paAIP_frac_silent,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. silent cells",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Fraction reward movement neurons
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_rwd,
            "paAIP2": paAIP_frac_rwd,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. rMRNs",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Event rates
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_event_rate,
            "paAIP2": paAIP_event_rate,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Event rate (events/min)",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["L"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Movement event amplitudes
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_amplitude,
            "paAIP2": paAIP_mvmt_amplitude,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle=activity_type,
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["M"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Movement onsets
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_avg_onset,
            "paAIP2": paAIP_mvmt_avg_onset,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Relative onset (s)",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["N"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Movement onset jitter
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_onset_jitter,
            "paAIP2": paAIP_mvmt_onset_jitter,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Relative onset jitter (s)",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["O"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Population vector similarity
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_vector_similarity,
            "paAIP2": paAIP_vector_similarity,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Population vector similarity",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["S"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Population vector correlation
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_vector_correlation,
            "paAIP2": paAIP_vector_correlation,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Population vector correlation (r)",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["W"],
        legend=True,
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, "paAIP2_Population_Dynamics")
        fig.savefig(fname + ".pdf")


def organize_pop_data_spikes(sessions, data_list, mvmt_only=False, norm=False):
    """Helper function to organize the population data for plotting using spikes"""
    # Initialize main outputs
    mvmt_traces = []
    frac_mvmt = []
    frac_silent = []
    frac_rwd = []
    event_rate = []
    mvmt_amplitude = []
    mvmt_avg_onset = []
    mvmt_onset_jitter = []
    vector_similarity = []
    vector_correlation = []

    for i, session in enumerate(sessions):
        # Temp variables
        m_traces = []
        f_mvmt = []
        f_silent = []
        f_rwd = []
        e_rate = []
        m_amplitude = []
        m_avg_onset = []
        m_onset_jitter = []
        v_similarity = []
        v_correlation = []
        for j, data in enumerate(data_list):
            mvmt_cells = data.mvmt_cells_spikes[i]
            # Grab relevant data
            traces = data.movement_traces_spikes[session]
            MRNs = data.fraction_MRNs_spikes[session]
            silent = data.fraction_silent_spikes[session]
            rMRNs = data.fraction_rMRNs_spikes[session]
            rate = data.cell_activity_rate[session]
            amplitude = data.movement_amplitudes_spikes[session]
            avg_onset = data.mean_onsets_spikes[session]
            onset_jitter = data.mvmt_onset_jitter[session]
            similarity = data.med_vector_similarity[session]
            correlation = data.med_vector_correlation[session]
            ## Subselect if movement cells only
            if mvmt_only:
                traces = d_utils.subselect_data_by_idxs(traces, mvmt_cells)
                rate = d_utils.subselect_data_by_idxs(rate, mvmt_cells)
                amplitude = d_utils.subselect_data_by_idxs(amplitude, mvmt_cells)
                avg_onset = d_utils.subselect_data_by_idxs(avg_onset, mvmt_cells)
                onset_jitter = d_utils.subselect_data_by_idxs(onset_jitter, mvmt_cells)

            ## Average traces for individual cells
            means = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
            means = np.vstack(means).T
            ## Store temporary variables
            m_traces.append(means)
            f_mvmt.append(MRNs)
            f_silent.append(silent)
            f_rwd.append(rMRNs)
            e_rate.append(rate)
            m_amplitude.append(amplitude)
            m_avg_onset.append(avg_onset)
            m_onset_jitter.append(onset_jitter)
            v_similarity.append(similarity)
            v_correlation.append(correlation)

        # Concatenate across datasets
        m_traces = np.hstack(m_traces)
        f_mvmt = np.concatenate(f_mvmt)
        f_silent = np.concatenate(f_silent)
        f_rwd = np.concatenate(f_rwd)
        e_rate = np.concatenate(e_rate)
        m_amplitude = np.concatenate(m_amplitude)
        m_avg_onset = np.concatenate(m_avg_onset)
        m_onset_jitter = np.concatenate(m_onset_jitter)
        v_similarity = np.array(v_similarity)
        v_correlation = np.array(v_correlation)

        mvmt_traces.append(m_traces)
        frac_mvmt.append(f_mvmt)
        frac_silent.append(f_silent)
        frac_rwd.append(f_rwd)
        event_rate.append(e_rate)
        mvmt_amplitude.append(m_amplitude)
        mvmt_avg_onset.append(m_avg_onset)
        mvmt_onset_jitter.append(m_onset_jitter)
        vector_similarity.append(v_similarity)
        vector_correlation.append(v_correlation)

    # Convert relevant data into 2d arrays
    frac_mvmt = convert_to_2D(frac_mvmt)
    frac_silent = convert_to_2D(frac_silent)
    frac_rwd = convert_to_2D(frac_rwd)
    event_rate = convert_to_2D(event_rate)
    mvmt_amplitude = convert_to_2D(mvmt_amplitude)
    mvmt_avg_onset = convert_to_2D(mvmt_avg_onset)
    mvmt_onset_jitter = convert_to_2D(mvmt_onset_jitter)
    vector_similarity = convert_to_2D(vector_similarity)
    vector_correlation = convert_to_2D(vector_correlation)

    if norm:
        vector_correlation = [
            d_utils.normalized_relative_difference(
                vector_correlation[0, :], vector_correlation[i, :]
            )
            for i in range(vector_correlation.shape[0])
        ]

    return (
        mvmt_traces,
        frac_mvmt,
        frac_silent,
        frac_rwd,
        event_rate,
        mvmt_amplitude,
        mvmt_avg_onset,
        mvmt_onset_jitter,
        vector_similarity,
        vector_correlation,
    )


def organize_pop_data_dFoF(sessions, data_list, mvmt_only=False, norm=False):
    """Helper function to organize the population data for plotting using spikes"""
    # Initialize main outputs
    mvmt_traces = []
    frac_mvmt = []
    frac_silent = []
    frac_rwd = []
    event_rate = []
    mvmt_amplitude = []
    mvmt_avg_onset = []
    mvmt_onset_jitter = []
    vector_similarity = []
    vector_correlation = []

    for i, session in enumerate(sessions):
        # Temp variables
        m_traces = []
        f_mvmt = []
        f_silent = []
        f_rwd = []
        e_rate = []
        m_amplitude = []
        m_avg_onset = []
        m_onset_jitter = []
        v_similarity = []
        v_correlation = []
        for j, data in enumerate(data_list):
            mvmt_cells = data.mvmt_cells_dFoF[i]
            # Grab relevant data
            traces = data.movement_traces_dFoF[session]
            MRNs = data.fraction_MRNs_dFoF[session]
            silent = data.fraction_silent_dFoF[session]
            rMRNs = data.fraction_rMRNs_dFoF[session]
            rate = data.cell_activity_rate[session]
            amplitude = data.movement_amplitudes_dFoF[session]
            avg_onset = data.mean_onsets_dFoF[session]
            onset_jitter = data.mvmt_onset_jitter[session]
            similarity = data.med_vector_similarity[session]
            correlation = data.med_vector_correlation[session]
            ## Subselect if movement cells only
            if mvmt_only:
                traces = d_utils.subselect_data_by_idxs(traces, mvmt_cells)
                rate = d_utils.subselect_data_by_idxs(rate, mvmt_cells)
                amplitude = d_utils.subselect_data_by_idxs(amplitude, mvmt_cells)
                avg_onset = d_utils.subselect_data_by_idxs(avg_onset, mvmt_cells)
                onset_jitter = d_utils.subselect_data_by_idxs(onset_jitter, mvmt_cells)

            ## Average traces for individual cells
            means = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
            means = np.vstack(means).T
            ## Store temporary variables
            m_traces.append(means)
            f_mvmt.append(MRNs)
            f_silent.append(silent)
            f_rwd.append(rMRNs)
            e_rate.append(rate)
            m_amplitude.append(amplitude)
            m_avg_onset.append(avg_onset)
            m_onset_jitter.append(onset_jitter)
            v_similarity.append(similarity)
            v_correlation.append(correlation)

        # Concatenate across datasets
        m_traces = np.hstack(m_traces)
        f_mvmt = np.concatenate(f_mvmt)
        f_silent = np.concatenate(f_silent)
        f_rwd = np.concatenate(f_rwd)
        e_rate = np.concatenate(e_rate)
        m_amplitude = np.concatenate(m_amplitude)
        m_avg_onset = np.concatenate(m_avg_onset)
        m_onset_jitter = np.concatenate(m_onset_jitter)
        v_similarity = np.array(v_similarity)
        v_correlation = np.array(v_correlation)

        mvmt_traces.append(m_traces)
        frac_mvmt.append(f_mvmt)
        frac_silent.append(f_silent)
        frac_rwd.append(f_rwd)
        event_rate.append(e_rate)
        mvmt_amplitude.append(m_amplitude)
        mvmt_avg_onset.append(m_avg_onset)
        mvmt_onset_jitter.append(m_onset_jitter)
        vector_similarity.append(v_similarity)
        vector_correlation.append(v_correlation)

    # Convert relevant data into 2d arrays
    frac_mvmt = convert_to_2D(frac_mvmt)
    frac_silent = convert_to_2D(frac_silent)
    frac_rwd = convert_to_2D(frac_rwd)
    event_rate = convert_to_2D(event_rate)
    mvmt_amplitude = convert_to_2D(mvmt_amplitude)
    mvmt_avg_onset = convert_to_2D(mvmt_avg_onset)
    mvmt_onset_jitter = convert_to_2D(mvmt_onset_jitter)
    vector_similarity = convert_to_2D(vector_similarity)
    vector_correlation = convert_to_2D(vector_correlation)

    if norm:
        vector_correlation = [
            d_utils.normalized_relative_difference(
                vector_correlation[0, :], vector_correlation[i, :]
            )
            for i in range(vector_correlation.shape[0])
        ]

    return (
        mvmt_traces,
        frac_mvmt,
        frac_silent,
        frac_rwd,
        event_rate,
        mvmt_amplitude,
        mvmt_avg_onset,
        mvmt_onset_jitter,
        vector_similarity,
        vector_correlation,
    )


def convert_to_2D(data_list):
    """Helper function to convert list of 1D arrays of different lengths into
    a 2D array"""
    # Get max value
    max_len = np.max([len(x) for x in data_list])
    padded_arrays = []
    for data in data_list:
        padded_data = d_utils.pad_array_to_length(data, length=max_len, value=np.nan)
        padded_arrays.append(padded_data)

    # Stack arrays
    output_array = np.vstack(padded_arrays)

    return output_array
