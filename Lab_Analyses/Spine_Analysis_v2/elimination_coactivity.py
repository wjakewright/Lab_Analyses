import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
sns.set_style("ticks")

from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2 import spine_utilities as s_utils
from Lab_Analyses.Spine_Analysis_v2.calculate_distance_coactivity_rate import (
    calculate_distance_coactivity_rate,
)
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_spine_dynamics,
)


def elimination_coactivity_analysis(
    mice_list, fov_type="apical", include_inactive=False,
):
    """Function to look at distance coactivity for eliminated spines to compare with 
        Nathan's analysis"""

    eliminated_coactivity = []
    stable_coactivity = []
    eliminated_coactivity_norm = []
    stable_coactivity_norm = []
    spine_density = {"Early": [], "Mid": [], "Late": []}
    fraction_elim = {"Early": [], "Mid": [], "Late": []}
    fraction_new = {"Early": [], "Mid": [], "Late": []}
    spine_volumes = {"Stable": [], "New": [], "Elim": []}
    # Analyze each mouse
    for mouse in mice_list:
        datasets = s_utils.load_spine_datasets(
            mouse, ["Early", "Middle", "Late"], fov_type
        )
        # Analyze each different FOV
        for FOV, dataset in datasets.items():
            early_data = dataset["Early"]
            mid_data = dataset["Middle"]
            late_data = dataset["Late"]

            if include_inactive:
                early_activity = early_data.spine_GluSnFr_activity
                mid_activity = mid_data.spine_GluSnFr_activity
            else:
                early_activity = remove_inactive_spines(
                    early_data.spine_GluSnFr_activity
                )
                mid_activity = remove_inactive_spines(mid_data.spine_GluSnFr_activity)

            # Early Coactivity
            (
                early_coactivity,
                _,
                early_coactivity_norm,
                _,
                bins,
            ) = calculate_distance_coactivity_rate(
                spine_activity=early_activity,
                spine_positions=early_data.spine_positions,
                flags=early_data.spine_flags,
                spine_groupings=early_data.spine_groupings,
                constrain_matrix=None,
                partner_list=None,
                bin_size=5,
                sampling_rate=early_data.imaging_parameters["Sampling Rate"],
                norm_method="mean",
            )
            # Middle Coactivity
            (
                mid_coactivity,
                _,
                mid_coactivity_norm,
                _,
                bins,
            ) = calculate_distance_coactivity_rate(
                spine_activity=mid_activity,
                spine_positions=mid_data.spine_positions,
                flags=mid_data.spine_flags,
                spine_groupings=mid_data.spine_groupings,
                constrain_matrix=None,
                partner_list=None,
                bin_size=5,
                sampling_rate=mid_data.imaging_parameters["Sampling Rate"],
                norm_method="mean",
            )

            # Identify middle session eliminated spines
            flag_list = [
                early_data.spine_flags,
                mid_data.spine_flags,
                late_data.spine_flags,
            ]
            position_list = [
                early_data.spine_positions,
                mid_data.spine_positions,
                late_data.spine_positions,
            ]
            grouping_list = [
                early_data.spine_groupings,
                mid_data.spine_groupings,
                late_data.spine_groupings,
            ]
            elim_corrected_flags = []

            # Have it to identify only newly eliminated spines
            for i, flags in enumerate(flag_list):
                if i != 0:
                    prev_flags = flag_list[i - 1]
                    temp_flags = []
                    for prev, curr in zip(prev_flags, flags):
                        if "Eliminated Spine" in prev and "Eliminated Spine" in curr:
                            temp_flags.append(["Absent"])
                        elif "Absent" in prev and "Absent" not in curr:
                            if "Eliminated Spine" in curr:
                                temp_flags.append(["Absent"])
                            else:
                                temp_flags.append(["New Spine"])
                        else:
                            temp_flags.append(curr)
                    elim_corrected_flags.append(temp_flags)

            mid_elim = np.array(
                s_utils.find_spine_classes(elim_corrected_flags[0], "Eliminated Spine")
            )
            late_elim = np.array(
                s_utils.find_spine_classes(elim_corrected_flags[1], "Eliminated Spine")
            )
            mid_new = np.array(
                s_utils.find_spine_classes(elim_corrected_flags[0], "New Spine")
            )
            late_new = np.array(
                s_utils.find_spine_classes(elim_corrected_flags[1], "New Spine")
            )
            mid_new_not = np.array([not x for x in mid_new])

            new_late_elim = late_elim * mid_new_not

            mid_elim_coactivity = early_coactivity[:, mid_elim]
            late_elim_coactivity = mid_coactivity[:, new_late_elim]
            mid_elim_coactivity_norm = early_coactivity_norm[:, mid_elim]
            late_elim_coactivity_norm = mid_coactivity_norm[:, new_late_elim]

            eliminated_coactivity.append(
                np.hstack((mid_elim_coactivity, late_elim_coactivity))
            )
            eliminated_coactivity_norm.append(
                np.hstack((mid_elim_coactivity_norm, late_elim_coactivity_norm))
            )
            # eliminated_coactivity.append(mid_elim_coactivity)
            # eliminated_coactivity_norm.append(mid_elim_coactivity_norm)

            # Get stable spines
            mid_stable = np.array(
                s_utils.find_stable_spines_across_days(
                    [early_data.spine_flags, mid_data.spine_flags,]
                )
            )
            late_stable = np.array(
                s_utils.find_stable_spines_across_days(
                    [
                        early_data.spine_flags,
                        mid_data.spine_flags,
                        late_data.spine_flags,
                    ]
                )
            )
            mid_stable = mid_stable.astype(bool)
            late_stable = late_stable.astype(bool)

            mid_stable_coactivity = early_coactivity[:, mid_stable]
            late_stable_coactivity = mid_coactivity[:, late_stable]
            mid_stable_coactivity_norm = early_coactivity_norm[:, mid_stable]
            late_stable_coactivity_norm = mid_coactivity_norm[:, late_stable]

            stable_coactivity.append(
                np.hstack((mid_stable_coactivity, late_stable_coactivity))
            )
            stable_coactivity_norm.append(
                np.hstack((mid_stable_coactivity_norm, late_stable_coactivity_norm))
            )

            density, frac_new, frac_elim = calculate_spine_dynamics(
                spine_flag_list=flag_list,
                spine_positions_list=position_list,
                spine_groupings_list=grouping_list,
            )
            days = ["Early", "Mid", "Late"]
            for i, day in enumerate(days):
                spine_density[day].append(density[i])
                fraction_elim[day].append(frac_elim[i])
                fraction_new[day].append(frac_new[i])

            # Get volumes
            mid_zoom = mid_data.imaging_parameters["Zoom"]
            early_zoom = early_data.imaging_parameters["Zoom"]
            late_zoom = late_data.imaging_parameters["Zoom"]
            early_volumes = (
                np.sqrt(np.array(early_data.spine_volume)) / (early_zoom / 2)
            ) ** 2
            mid_volumes = (
                np.sqrt(np.array(mid_data.spine_volume)) / (mid_zoom / 2)
            ) ** 2
            late_volumes = (
                np.sqrt(np.array(late_data.spine_volume)) / (late_zoom / 2)
            ) ** 2

            elim_volume = np.concatenate(
                [early_volumes[mid_elim], early_volumes[new_late_elim]]
            )
            new_volume = np.concatenate([mid_volumes[mid_new], late_volumes[late_new]])
            stable_volume = np.concatenate(
                [mid_volumes[mid_stable], late_volumes[late_stable]]
            )
            spine_volumes["Stable"].append(stable_volume)
            spine_volumes["Elim"].append(elim_volume)
            spine_volumes["New"].append(new_volume)

    # Join all mice/fovs together
    all_elim_coactivity = np.hstack(eliminated_coactivity)
    all_elim_coactivity_norm = np.hstack(eliminated_coactivity_norm)
    all_stable_coactivity = np.hstack(stable_coactivity)
    all_stable_coactivity_norm = np.hstack(stable_coactivity_norm)
    print(all_elim_coactivity.shape)
    print(all_stable_coactivity.shape)
    # reformat the density data
    all_density = {}
    all_elim = {}
    all_new = {}
    for day in ["Early", "Mid", "Late"]:
        all_density[day] = np.concatenate(spine_density[day])
        all_elim[day] = np.concatenate(fraction_elim[day])
        all_new[day] = np.concatenate(fraction_new[day])
    all_density = np.vstack(list(all_density.values()))
    all_elim = np.vstack(list(all_elim.values()))
    all_new = np.vstack(list(all_new.values()))

    # join together the volume data
    joined_volumes = {}
    for key, value in spine_volumes.items():
        joined_volumes[key] = np.concatenate(value)
        print(f"{key}: {np.nanmean(np.concatenate(value))}")

    # Plot results
    fig, axes = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        """,
        figsize=(10, 15),
    )
    fig.subplots_adjust(hspace=1, wspace=0.5)
    plot_multi_line_plot(
        data_dict={"eliminated": all_elim_coactivity, "stable": all_stable_coactivity},
        x_vals=bins[1:],
        plot_ind=False,
        figsize=(7, 5),
        title="Raw Coactivity",
        ytitle="Coactivity Rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=["firebrick", "black"],
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=False,
    )
    plot_multi_line_plot(
        data_dict={
            "eliminated": all_elim_coactivity_norm,
            "stable": all_stable_coactivity_norm,
        },
        x_vals=bins[1:],
        plot_ind=False,
        figsize=(7, 5),
        title="Norm Coactivity",
        ytitle="Norm. Coactivity Rate",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=["firebrick", "black"],
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        legend=True,
        save=False,
        save_path=False,
    )
    plot_multi_line_plot(
        data_dict={"density": all_density},
        x_vals=np.arange(all_density.shape[0]),
        plot_ind=True,
        figsize=(7, 5),
        title="Spine Density",
        ytitle="Spine Density",
        xtitle="Session Num.",
        ylim=None,
        line_color=["mediumblue"],
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        legend=True,
        save=False,
        save_path=False,
    )
    plot_multi_line_plot(
        data_dict={"density": all_new},
        x_vals=np.arange(all_density.shape[0]),
        plot_ind=True,
        figsize=(7, 5),
        title="new spines",
        ytitle="Fraction New Spines",
        xtitle="Session Num.",
        ylim=None,
        line_color=["forestgreen"],
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        legend=True,
        save=False,
        save_path=False,
    )
    plot_multi_line_plot(
        data_dict={"density": all_elim},
        x_vals=np.arange(all_density.shape[0]),
        plot_ind=True,
        figsize=(7, 5),
        title="Eliminated spines",
        ytitle="Fraction Elim. Spines",
        xtitle="Session Num.",
        ylim=None,
        line_color=["firebrick"],
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        legend=True,
        save=False,
        save_path=False,
    )
    plot_swarm_bar_plot(
        data_dict=joined_volumes,
        mean_type="median",
        err_type="CI",
        figsize=(5, 5),
        title="Spine Volumes",
        xtitle=None,
        ytitle="Spine Volume \u03BCm",
        ylim=None,
        b_colors=["grey", "royalblue", "firebrick"],
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=["grey", "royalblue", "firebrick"],
        s_size=5,
        s_alpha=0.8,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    plot_histogram(
        data=[joined_volumes["Stable"], joined_volumes["New"]],
        bins=25,
        stat="probability",
        avlines=None,
        title="Stable and New Spines",
        xtitle="Spine Volume",
        xlim=None,
        figsize=(5, 5),
        color=["grey", "royalblue"],
        alpha=0.6,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    plot_histogram(
        data=[joined_volumes["Stable"], joined_volumes["Elim"]],
        bins=25,
        stat="probability",
        avlines=None,
        title="Stable and Elim Spines",
        xtitle="Spine Volume",
        xlim=None,
        figsize=(5, 5),
        color=["grey", "firebrick"],
        alpha=0.6,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()


def remove_inactive_spines(spine_activity):
    """Helper function to remove inactive spines from analysis"""

    new_spine_activity = np.zeros(spine_activity.shape) * np.nan

    for i in range(spine_activity.shape[1]):
        if np.nansum(spine_activity[:, i]):
            new_spine_activity[:, i] = spine_activity[:, i]

    return new_spine_activity
