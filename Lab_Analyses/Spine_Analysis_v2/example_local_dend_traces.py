import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sysignal
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Spine_Analysis_v2.spine_dendrite_event_analysis import (
    extend_dendrite_activity,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    bin_by_position,
    find_nearby_spines,
    find_present_spines,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import (
    calculate_coactivity,
    get_conservative_coactive_binary,
)

sns.set()
sns.set_style("ticks")


def example_local_dend_traces(
    mice_list, fov_type="apical", example=None, save=False, save_path=None
):
    """Function to get example traces"""
    # Setup position bins
    MAX_DIST = 15
    bin_num = 15
    position_bins = np.linspace(0, MAX_DIST, 16)

    activity_window = (-1, 2)

    example_coactive_traces = []
    example_noncoactive_traces = []

    # Parse example
    if example:
        example_m = example.split(" ")[0]
        example_f = example.split(" ")[1]
        example_s = int(example.split(" ")[2])

    for mouse in mice_list:
        datasets = load_spine_datasets(mouse, ["Early"], fov_type)

        for FOV, dataset in datasets.items():
            data = dataset["Early"]
            spine_activity = data.spine_GluSnFr_activity
            dendrite_activity = data.dendrite_calcium_activity
            poly_dendrite_dFoF = data.poly_dendrite_calcium_processed_dFoF
            poly_dendrite_positions = data.poly_dendrite_positions
            spine_positions = data.spine_positions
            spine_flags = data.spine_flags
            spine_groupings = data.spine_groupings
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])

            nearby_spine_idxs = find_nearby_spines(
                spine_positions, spine_flags, spine_groupings, None, cluster_dist=10
            )
            # Sort out the spine groupings
            if type(spine_groupings[0]) != list:
                spine_groupings = [spine_groupings]

            # Extend dendrite activity to remove potentially decaying events
            ext_dend_activity = extend_dendrite_activity(
                dendrite_activity, 0.2, sampling_rate
            )
            activity_matrix = np.zeros(spine_activity.shape)
            for c in range(spine_activity.shape[1]):
                _, dend_inactive = get_conservative_coactive_binary(
                    spine_activity[:, c], ext_dend_activity[:, c]
                )
                activity_matrix[:, c] = dend_inactive

            present_spines = find_present_spines(spine_flags)

            for i, spines in enumerate(spine_groupings):
                spine_idxs = list(range(activity_matrix.shape[1]))
                curr_dend_dFoF = poly_dendrite_dFoF[i]
                curr_dend_positions = poly_dendrite_positions[i]
                curr_s_activity = activity_matrix[:, spines]
                curr_present = present_spines[spines]
                curr_positions = spine_positions[spines]
                curr_nearby_spines = [nearby_spine_idxs[j] for j in spines]

                dend_median_dFoF = np.nanmedian(curr_dend_dFoF, axis=1)
                corr_dend_dFoF = curr_dend_dFoF - dend_median_dFoF.reshape(-1, 1)

                for spine in range(curr_s_activity.shape[1]):
                    curr_idx = spine_idxs[spines[spine]]
                    if curr_present[spine] == False:
                        continue
                    spine_pos = curr_positions[spine]
                    dend_pos = np.array(curr_dend_positions) - spine_pos
                    dend_pos = np.absolute(dend_pos)
                    if (curr_nearby_spines[spine] is None) or (
                        len(curr_nearby_spines[spine]) == 0
                    ):
                        combined_nearby_activity = np.zeros(curr_s_activity.shape[0])
                    else:
                        combined_nearby_activity = np.nansum(
                            activity_matrix[:, curr_nearby_spines[spine]], axis=1
                        )
                        combined_nearby_activity[combined_nearby_activity > 1] = 1
                    ## Get inactivity
                    combined_nearby_inactivity = 1 - combined_nearby_activity
                    ## Get binary traces
                    _, _, _, _, coactive = calculate_coactivity(
                        curr_s_activity[:, spine],
                        combined_nearby_activity,
                        sampling_rate=sampling_rate,
                        norm_method="mean",
                    )
                    _, _, _, _, noncoactive = calculate_coactivity(
                        curr_s_activity[:, spine],
                        combined_nearby_inactivity,
                        sampling_rate=sampling_rate,
                        norm_method="mean",
                    )
                    ## Timestamps and refine
                    if np.nansum(coactive):
                        coactive_stamps = t_stamps.get_activity_timestamps(coactive)
                        coactive_stamps = [x[0] for x in coactive_stamps]
                        coactive_stamps = t_stamps.refine_activity_timestamps(
                            coactive_stamps,
                            window=activity_window,
                            max_len=len(curr_s_activity[:, spine]),
                            sampling_rate=sampling_rate,
                        )
                    else:
                        coactive_stamps = []
                    if np.nansum(noncoactive):
                        noncoactive_stamps = t_stamps.get_activity_timestamps(
                            noncoactive
                        )
                        noncoactive_stamps = [x[0] for x in noncoactive_stamps]
                        noncoactive_stamps = t_stamps.refine_activity_timestamps(
                            noncoactive_stamps,
                            window=activity_window,
                            max_len=len(curr_s_activity[:, spine]),
                            sampling_rate=sampling_rate,
                        )
                    else:
                        noncoactive_stamps = []
                    # Getting traces and amps
                    temp_coactive_traces = []
                    temp_coactive_amps = []
                    temp_noncoactive_traces = []
                    temp_noncoactive_amps = []
                    for i in range(corr_dend_dFoF.shape[1]):
                        coactive_traces, coactive_amp = get_dend_traces(
                            coactive_stamps,
                            corr_dend_dFoF[:, i],
                            activity_window,
                            sampling_rate,
                        )
                        noncoactive_traces, noncoactive_amp = get_dend_traces(
                            noncoactive_stamps,
                            corr_dend_dFoF[:, i],
                            activity_window,
                            sampling_rate,
                        )
                        temp_coactive_traces.append(coactive_traces)
                        temp_coactive_amps.append(coactive_amp)
                        temp_noncoactive_traces.append(noncoactive_traces)
                        temp_noncoactive_amps.append(noncoactive_amp)

                    sorted_coactive_traces = [
                        x for _, x in sorted(zip(dend_pos, temp_coactive_traces))
                    ]
                    sorted_coactive_amps = np.array(
                        [x for _, x in sorted(zip(dend_pos, temp_coactive_amps))]
                    )
                    sorted_noncoactive_traces = [
                        x for _, x in sorted(zip(dend_pos, temp_noncoactive_traces))
                    ]
                    sorted_noncoactive_amps = np.array(
                        [x for _, x in sorted(zip(dend_pos, temp_noncoactive_amps))]
                    )
                    sorted_positions = np.array(
                        [y for y, _ in sorted(zip(dend_pos, temp_coactive_amps))]
                    )
                    # bin amplitudes by position
                    binned_coactive_amps = bin_by_position(
                        sorted_coactive_amps, sorted_positions, position_bins
                    )
                    binned_noncoactive_amps = bin_by_position(
                        sorted_noncoactive_amps, sorted_positions, position_bins
                    )
                    if binned_coactive_amps[0] > 0.03:
                        print(f"{mouse} {FOV} {curr_idx}")
                        print(
                            f"Pos: {len(sorted_positions)} {len(sorted_coactive_amps)}"
                        )
                        print(f"Coactive: {binned_coactive_amps}")
                        print(f"Noncoactive: {binned_noncoactive_amps}")

                    if example:
                        if (
                            (example_m == mouse)
                            and (example_f == FOV)
                            and (example_s == curr_idx)
                        ):
                            print("Getting example")
                            example_coactive_traces = sorted_coactive_traces
                            example_noncoactive_traces = sorted_noncoactive_traces

    # Prepare the plot
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        """,
        figsize=(14, 8),
    )
    ax_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for i in range(10):
        if i != 0:
            ylim = axes["A"].get_ylim()
        else:
            ylim = None
        ax = ax_list[i]
        coa_traces = example_coactive_traces[i]
        noncoa_traces = example_noncoactive_traces[i]
        coa_means = np.nanmean(coa_traces, axis=1)
        # coa_sems = stats.sem(coa_traces, axis=1, nan_policy="omit")
        coa_sems = np.zeros(len(coa_means))
        noncoa_means = np.nanmean(noncoa_traces, axis=1)
        # noncoa_sems = stats.sem(noncoa_traces, axis=1, nan_policy="omit")
        noncoa_sems = np.zeros(len(noncoa_means))
        ylim = (-0.06, 0.2)

        plot_mean_activity_traces(
            means=[coa_means, noncoa_means],
            sems=[coa_sems, noncoa_sems],
            group_names=["Coactive", "NonCoactive"],
            sampling_rate=sampling_rate,
            activity_window=activity_window,
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["forestgreen", "black"],
            title=i,
            ytitle="dF/F",
            ylim=ylim,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes[ax],
            save=False,
            save_path=None,
        )

        fig.tight_layout()

        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            fname = os.path.join(save_path, "Local_Dendrite_Examples")
            fig.savefig(fname + ".pdf")


def get_dend_traces(timestamps, dend_dFoF, activity_window, sampling_rate):
    """Helper function to get teh timelocked traces and amplitudes of local
    dendrite traces. Works with only ons set of timestamps and a single
    local dendrite dFoF trace
    """
    center_point = np.absolute(activity_window[0]) * sampling_rate
    after_point = sampling_rate

    # Return null values if there are no timestamps
    if len(timestamps) == 0:
        traces = (
            np.zeros(((activity_window[1] - activity_window[0]) * sampling_rate, 1))
            * np.nan
        )
        amplitude = np.nan
        return traces, amplitude

    # Get the traces
    traces, mean = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Activity"],
        timestamps,
        activity_window,
        sampling_rate,
    )
    mean = mean["Activity"][0]
    traces = traces["Activity"]

    # Smooth the mean trace
    mean = sysignal.savgol_filter(mean, 31, 3)

    # Get max amplitude idx
    max_idx = np.argmax(mean)

    # Average amplitude around the max
    # win = int((0.25 * sampling_rate) / 2)
    # amplitude = np.nanmean(mean[max_idx - win : max_idx + win])
    amplitude = np.nanmean(mean[center_point : center_point + after_point])
    base = np.nanmean(mean[center_point - after_point : center_point])
    amplitude = amplitude - base

    return traces, amplitude
