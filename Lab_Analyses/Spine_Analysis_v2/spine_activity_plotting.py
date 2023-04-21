import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_basic_features(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 4),
    hist_bins=25,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot the distribution of relative volumes and activity rates
    
        INPUT PARAMETERS
            dataset - Spine_Activity_Data object
            
            followup_dataset - optional Spine_Activity_Data object of the subsequent
                                session to use for volume comparison. Default is None
                                to use the followup volumes in the dataset

            exclude - str specifying a type of spine to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cuttoffs 
                        for classifying plasticity

            hist_bins - int specifying how many bins to plot for the histogram
            
            save - boolean specifying whether to save the figure or not
            
            save_path - str specifying where to save the path
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    spine_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }
    # Pull the relevant data
    initial_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    spine_activity_rate = dataset.spine_activity_rate

    # Calculate spine volumes
    ## Get followup volumes
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.followup_flags
    ## Setup input lists
    volumes = [initial_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    ## Calculate
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude
    )
    delta_volume = delta_volume[-1]

    # Classify plasticity
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False
    )

    # Subselect data
    initial_volumes = d_utils.subselect_data_by_idxs(initial_volumes, spine_idxs)
    spine_activity_rate = d_utils.subselect_data_by_idxs(
        spine_activity_rate, spine_idxs
    )

    # Organize datad dictionaries
    initial_vol_dict = {}
    activity_dict = {}
    count_dict = {}
    for key, value in spine_groups.items():
        spines = eval(value)
        vol = initial_volumes[spines]
        activity = spine_activity_rate[spines]
        initial_vol_dict[key] = vol[~np.isnan(vol)]
        activity_dict[key] = activity[~np.isnan(activity)]
        count_dict[key] = np.sum(spines)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        [["tl", "tm", "tr"], ["bl", "bm", "br"]], figsize=figsize,
    )
    fig.suptitle("Basic Spine Features")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################### Plat data onto the axes #############################
    # Relative volume distributions
    plot_histogram(
        data=delta_volume,
        bins=hist_bins,
        stat="probability",
        avlines=[1],
        title="\u0394 Volume",
        xtitle="\u0394 Volume",
        xlim=None,
        figsize=(5, 5),
        color="mediumblue",
        alpha=0.7,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["tl"],
        save=False,
        save_path=None,
    )

    # Initial volume vs Relative volume correlation
    plot_scatter_correlation(
        x_var=initial_volumes,
        y_var=delta_volume,
        CI=95,
        title="Initial vs \u0394 Volume",
        xtitle="Initial volume \u03BCm",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["tm"],
        save=False,
        save_path=None,
    )

    # Initial volume for spine types
    plot_swarm_bar_plot(
        data_dict=initial_vol_dict,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Initial Volumes",
        xtitle=None,
        ytitle="Initial volume \u03BCm",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["tr"],
        save=False,
        save_path=None,
    )

    # Percentage of different spine types
    plot_pie_chart(
        data_dict=count_dict,
        title="Fraction of plastic spines",
        figsize=(5, 5),
        colors=COLORS,
        alpha=0.7,
        edgecolor="white",
        txt_color="white",
        txt_size=10,
        legend="top",
        donut=0.6,
        linewidth=1.5,
        ax=axes["bl"],
        save=False,
        save_path=None,
    )

    # Spine activity rate vs relative volume
    plot_scatter_correlation(
        x_var=spine_activity_rate,
        y_var=delta_volume,
        CI=95,
        title="Event Rate vs \u0394 Volume",
        xtitle="Event rate (events/min)",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["bm"],
        save=False,
        save_path=None,
    )

    # Activity rates for spine types
    plot_swarm_bar_plot(
        data_dict=activity_dict,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Event rates",
        xtitle=None,
        ytitle="Event rate (events/min)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["br"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_1")
        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ###########################
    if display_stats == False:
        return

    ## Perform the statistics
    if test_type == "parametric":
        vol_f, vol_p, _, vol_test_df = t_utils.ANOVA_1way_posthoc(
            initial_vol_dict, test_method
        )
        activity_f, activity_p, _, activity_test_df = t_utils.ANOVA_1way_posthoc(
            activity_dict, test_method
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        vol_f, vol_p, vol_test_df = t_utils.kruskal_wallis_test(
            initial_vol_dict, "Conover", test_method,
        )
        activity_f, activity_p, activity_test_df = t_utils.kruskal_wallis_test(
            activity_dict, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"
    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic([["left", "right"]], figsize=(8, 4))
    ## Format the first table
    axes2["left"].axis("off")
    axes2["left"].axes("tight")
    axes2["left"].set_title(
        f"Initial Volume {test_title}\nF = {vol_f:.4}   p = {vol_p:.3E}"
    )
    left_table = axes2["left"].table(
        cellText=vol_test_df.values,
        colLabels=vol_test_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    left_table.auto_set_font_size(False)
    left_table.set_fontsize(8)
    ## Format the second table
    axes2["right"].axis("off")
    axes2["right"].axis("tight")
    axes2["right"].set_title(
        f"Event Rate {test_title}\nF = {activity_f:.4}    p = {activity_p:.3E}"
    )
    right_table = axes2["right"].table(
        cellText=activity_test_df.values,
        cellLabels=vol_test_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    right_table.auto_set_font_size(False)
    right_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_1_Stats")
        fig.savefig(fname + ".pdf")

