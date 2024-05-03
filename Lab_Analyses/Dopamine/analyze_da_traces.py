import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

import Lab_Analyses.Utilities.data_utilities as d_utils
from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import load_spine_datasets
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import mean_trace_functions as trace_fun

sns.set_style("ticks")


def analyze_da_traces(
    mice_list,
    session,
    fov_type="apical",
    roi_type="dendrite",
    activity_window=(-2, 4),
    zscore=False,
    dend_to_plot=None,
    save=False,
    save_path=None,
):
    """Function to get and analyze DA transients from dendrites

    INPUT PARAMETERS
        mice_list - list of str specifying all the mice to be analyzed

        session - str specifying the session to be analyzed

        fov_type - str specifying whether to analyze apical or basal FOVs

        activity_window - tuple specifying the window around which activity
                        should be analyzed in seconds

        save - boolean of whetehr to save the output figures or not

        save_path - str specifying where to save the figure outputs


    """
    RND_N = 100
    # Setup containers
    identifiers = []
    session_dFoF_traces = []
    session_activity_traces = []
    active_traces = []
    movement_traces = []
    rwd_movement_traces = []
    reward_traces = []
    cue_traces = []
    random_traces = []

    # Analyze each mouse and FOV seperately
    dendrite_tracker = 0
    for mouse in mice_list:
        print(mouse)
        # Load the data
        datasets = load_spine_datasets(mouse, [session], fov_type)

        for FOV, dataset in datasets.items():
            data = dataset[session]

            # Pull relevant data
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            spine_groupings = data.spine_groupings
            spine_activity = data.spine_DA_activity
            spine_dFoF = data.spine_DA_processed_dFoF
            dendrite_activity = data.dendrite_DA_activity
            dendrite_dFoF = data.dendrite_DA_processed_dFoF

            if zscore:
                spine_dFoF = d_utils.z_score(spine_dFoF)
                dendrite_dFoF = [d_utils.z_score(x) for x in dendrite_dFoF]

            lever_active = data.lever_active
            lever_active_rwd = data.rewarded_movement_binary
            binary_cue = data.binary_cue
            reward_delivery = data.reward_delivery

            if type(spine_groupings[0]) != list:
                spine_groupings = [spine_groupings]
            # Analyze each dendrite in the FOV
            for i, grouping in enumerate(spine_groupings):
                dendrite_tracker = dendrite_tracker + i
                identifier = f"{mouse}_{FOV}_dend_{dendrite_tracker}"
                identifiers.append(identifier)
                # Get appropriate activity
                if roi_type == "dendrite":
                    activity = dendrite_activity[i]
                    dFoF = dendrite_dFoF[i]
                else:
                    activity = spine_activity[:, grouping]
                    dFoF = spine_dFoF[:, grouping]

                # Store session traces
                session_dFoF_traces.append(dFoF)
                session_activity_traces.append(activity)

                # Get mean traces
                ## Activity
                activity_timestamps = []
                for roi in range(activity.shape[1]):
                    a_stamps = t_stamps.get_activity_timestamps(activity[:, roi])
                    a_stamps = t_stamps.refine_activity_timestamps(
                        a_stamps, activity_window, activity.shape[0], sampling_rate
                    )
                    a_stamps = [x[0] for x in a_stamps]
                    activity_timestamps.append(a_stamps)
                a_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    activity_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                active_traces.append(a_traces)

                ## Movement
                move_timestamps = t_stamps.get_activity_timestamps(lever_active)
                move_timestamps = t_stamps.refine_activity_timestamps(
                    move_timestamps, activity_window, activity.shape[0], sampling_rate
                )
                move_timestamps = [x[0] for x in move_timestamps]
                movement_timestamps = [
                    move_timestamps for i in range(activity.shape[1])
                ]
                move_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    movement_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                movement_traces.append(move_traces)

                ## Rewarded Movement
                rwd_move_timestamps = t_stamps.get_activity_timestamps(lever_active_rwd)
                rwd_move_timestamps = t_stamps.refine_activity_timestamps(
                    rwd_move_timestamps,
                    activity_window,
                    activity.shape[0],
                    sampling_rate,
                )
                rwd_move_timestamps = [x[0] for x in rwd_move_timestamps]
                rwd_movement_timestamps = [
                    rwd_move_timestamps for i in range(activity.shape[1])
                ]
                rwd_move_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    rwd_movement_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                rwd_movement_traces.append(rwd_move_traces)

                # Reward deliver
                rwd_timestamps = t_stamps.get_activity_timestamps(reward_delivery)
                rwd_timestamps = t_stamps.refine_activity_timestamps(
                    rwd_timestamps,
                    activity_window,
                    activity.shape[0],
                    sampling_rate,
                )
                rwd_timestamps = [x[0] for x in rwd_timestamps]
                reward_timestamps = [rwd_timestamps for i in range(activity.shape[1])]
                rwd_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    reward_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                reward_traces.append(rwd_traces)

                ## Cue delivery
                c_timestamps = t_stamps.get_activity_timestamps(binary_cue)
                c_timestamps = t_stamps.refine_activity_timestamps(
                    c_timestamps,
                    activity_window,
                    activity.shape[0],
                    sampling_rate,
                )
                c_timestamps = [x[0] for x in c_timestamps]
                cue_timestamps = [c_timestamps for i in range(activity.shape[1])]
                c_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    cue_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                cue_traces.append(c_traces)

                # Random traces
                random_timestamps = []
                for roi in range(activity.shape[1]):
                    rnd_timestamps = np.random.choice(
                        activity.shape[0], RND_N, replace=False
                    )
                    rnd_timestamps = t_stamps.refine_activity_timestamps(
                        rnd_timestamps,
                        activity_window,
                        activity.shape[0],
                        sampling_rate,
                    )
                    random_timestamps.append(rnd_timestamps)
                rnd_traces, _, _ = trace_fun.analyze_event_activity(
                    dFoF,
                    random_timestamps,
                    activity_window,
                    center_onset=False,
                    smooth=False,
                    avg_window=1,
                    norm_constant=None,
                    sampling_rate=sampling_rate,
                    peak_required=False,
                )
                ## Store
                random_traces.append(rnd_traces)

    # Perfrom some plotting for each dendrite
    print(f"Number of dendrite: {len(identifiers)}")
    ## Randomly select dendrite to plot if none given
    if dend_to_plot is None:
        dend_to_plot = np.random.choice(len(identifiers), 1)[0]

    print(dend_to_plot)

    plot_identifier = identifiers[dend_to_plot]
    plot_session_activity_traces = session_activity_traces[dend_to_plot]
    plot_session_dFoF_traces = session_dFoF_traces[dend_to_plot]
    plot_active_traces = active_traces[dend_to_plot]
    plot_movement_traces = movement_traces[dend_to_plot]
    plot_rwd_movement_traces = rwd_movement_traces[dend_to_plot]
    plot_reward_traces = reward_traces[dend_to_plot]
    plot_cue_traces = cue_traces[dend_to_plot]
    plot_random_traces = random_traces[dend_to_plot]

    # Plot session dFoF
    sess_fig, sess_ax = plt.subplots(figsize=(12, 16))
    sess_fig.tight_layout()
    sess_ax.set_title(f"Session dFoF {plot_identifier}")
    for i in range(plot_session_dFoF_traces.shape[1]):
        x = np.linspace(
            0,
            len(plot_session_dFoF_traces[:, i]) / sampling_rate,
            len(plot_session_dFoF_traces[:, i]),
        )
        sess_ax.plot(
            x,
            plot_session_dFoF_traces[:, i] + i,
            label=i,
            linewidth=0.5,
            color="mediumslateblue",
        )
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Figures"
        save_name = os.path.join(save_path, f"Session_dFoF_{plot_identifier}")
        sess_fig.savefig(save_name + ".pdf")

    # Plot active traces
    plot_roi_traces(
        roi_traces=plot_active_traces,
        col_num=4,
        title=f"{plot_identifier}_active_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )

    # Plot movement traces
    plot_roi_traces(
        roi_traces=plot_movement_traces,
        col_num=4,
        title=f"{plot_identifier}_movement_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )

    # Plot rwd movement traces
    plot_roi_traces(
        roi_traces=plot_rwd_movement_traces,
        col_num=4,
        title=f"{plot_identifier}_rewarded_movement_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )
    # Plot reward traces
    plot_roi_traces(
        roi_traces=plot_reward_traces,
        col_num=4,
        title=f"{plot_identifier}_reward_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )
    # Plot cue traces
    plot_roi_traces(
        roi_traces=plot_cue_traces,
        col_num=4,
        title=f"{plot_identifier}_cue_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )
    # Plot random traces
    plot_roi_traces(
        roi_traces=plot_random_traces,
        col_num=4,
        title=f"{plot_identifier}_random_traces",
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        save=save,
        save_path=save_path,
    )


def plot_roi_traces(
    roi_traces,
    col_num=4,
    title="default",
    sampling_rate=60,
    activity_window=(-2, 4),
    save=False,
    save_path=None,
):
    """Helper function to plot the traces for each roi"""
    tot = len(roi_traces)
    row_num = tot // col_num
    row_num += tot % col_num
    figsize = (10, 2.5 * row_num)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(title)

    count = 1
    for traces in roi_traces:
        ax = fig.add_subplot(row_num, col_num, count)
        zeroed_activity = d_utils.zero_window(traces, (0, 2), sampling_rate)
        mean_trace = np.nanmean(zeroed_activity, axis=1)
        sem_trace = stats.sem(zeroed_activity, axis=1, nan_policy="omit")
        plot_mean_activity_traces(
            means=mean_trace,
            sems=sem_trace,
            group_names="Spine",
            sampling_rate=sampling_rate,
            activity_window=activity_window,
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors="mediumslateblue",
            title=f"ROI {count}",
            ytitle="dF/F",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            ax=ax,
            save=False,
            save_path=None,
        )
        count += 1

    fig.tight_layout()
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Figures"
        save_name = os.path.join(save_path, f"{title}_responses")
        fig.savefig(save_name + ".pdf")
