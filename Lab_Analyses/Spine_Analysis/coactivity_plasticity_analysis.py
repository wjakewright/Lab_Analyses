import os
from collections import defaultdict
from itertools import combinations, compress

import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis import spine_plotting as sp
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import test_utilities as t_utils
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


class Coactivity_Plasticity:
    """Class to handle the analysis of spine plasticity on coactivity datasets"""

    def __init__(
        self, data, thershold, exclude, vol_norm=False, save=False, save_path=None
    ):
        """Initialize the class
            
            INPUT PARAMETERS    
                data - Spine_Coactivity_Data object, or a list of these objects
                
                threshold - float specifying the threshold to consider for plastic spines
                
                exclude - str specifying the spine types to exclude form analysis (e.g., Shaft)
        """

        # Check number of datasets and set up data
        if type(data) == list:
            dataset = data[0]
            subsequent_flags = data[1].spine_flags
            subsequent_volumes = data[1].spine_volumes
        elif isinstance(data, object):
            if data.followup_volumes is not None:
                dataset = data
                subsequent_flags = data.followup_flags
                subsequent_volumes = data.followup_volumes
            else:
                raise Exception("Data must have followup data containing spine volumes")

        self.day = dataset.day
        self.threshold = thershold
        self.exclude = exclude
        self.vol_norm = vol_norm
        self.parameters = dataset.parameters
        self.save = save
        self.save_path = save_path

        self.group_dict = {
            "plastic_spines": ["enlarged_spines", "shrunken_spines", "stable_spines"],
            "movement_spines": ["movement_spines", "nonmovement_spines"],
            "rwd_movement_spines": ["rwd_movement_spines", "rwd_nonmovement_spines"],
            "movement_dendrites": ["movement_dendrites", "nonmovement_dendrites"],
            "rwd_movement_dendrites": [
                "rwd_movement_dendrities",
                "rwd_nonmovement_dendrites",
            ],
        }

        # Analyze the data
        self.analyze_plasticity(dataset, subsequent_flags, subsequent_volumes)

        if save:
            self.save_output()

    def analyze_plasticity(self, dataset, subsequent_flags, subsequent_volumes):
        """Method to calculate spine volume change and classify plasticity"""

        volume_data = [dataset.spine_volumes, subsequent_volumes]
        flag_list = [dataset.spine_flags, subsequent_flags]
        relative_volumes, spine_idxs = calculate_volume_change(
            volume_data, flag_list, norm=self.vol_norm, days=None, exclude=self.exclude,
        )
        relative_volumes = np.array(list(relative_volumes.values())[-1])
        enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
            relative_volumes, self.threshold, norm=self.vol_norm
        )

        # Store volume change and plasticity classifications
        self.relative_volumes = relative_volumes
        self.enlarged_spines = enlarged_spines
        self.shrunken_spines = shrunken_spines
        self.stable_spines = stable_spines

        # Refine coactivity variables for only stable spines and store them
        attributes = list(dataset.__dict__.keys())
        for attribute in attributes:
            # Save attributes that do not need to be refined
            if attribute == "day" or attribute == "parameters":
                variable = getattr(dataset, attribute)
                setattr(self, attribute, variable)
                continue

            # Get the corresponding variable
            variable = getattr(dataset, attribute)

            # Skip variables that are None
            if variable is None:
                setattr(self, attribute, variable)
                continue

            # Refine variable based on stable spine idxs
            if type(variable) == np.ndarray:
                if len(variable.shape) == 1:
                    new_variable = variable[spine_idxs]
                elif len(variable.shape) == 2:
                    new_variable = variable[:, spine_idxs]
            elif type(variable) == list:
                try:
                    new_variable = [variable[i] for i in spine_idxs]
                except IndexError:
                    print(f"{attribute} is an empty list!!! Will skip.")
                    print(variable)
                    continue
            else:
                raise TypeError(f"{attribute} {type(variable)} is incorrect datatype")

            # Store the attribute
            setattr(self, attribute, new_variable)

    def plot_volume_correlation(
        self,
        variable_name,
        volume_type,
        CI=None,
        figsize=(5, 5),
        ytitle=None,
        xlim=None,
        ylim=None,
        face_color="mediumblue",
        edge_color="white",
        edge_width=0.3,
        s_alpha=0.5,
        line_color="mediumblue",
        line_width=1,
        log_trans=True,
        save=False,
        save_path=None,
    ):
        """Method to plot and correlation a given variable against spine volume change"""
        variable = getattr(self, variable_name)

        # Remove nan values
        non_nan = np.nonzero(~np.isnan(variable))[0]
        variable = variable[non_nan]

        # Log transform relative volumes if specified
        if volume_type == "relative_volume":
            if log_trans:
                volume = np.log10(self.relative_volumes)
            else:
                volume = self.relative_volumes
            xtitle = "\u0394" + " spine volume"
        elif volume_type == "volume_um":
            volume = self.spine_volumes_um
            xtitle = "spine area (um)"
        elif volume_type == "volume":
            volume = self.spine_volumes
            xtitle = "spine area (au)"

        volume = volume[non_nan]

        sp.plot_sns_scatter_correlation(
            volume,
            variable,
            CI,
            title=variable_name,
            xtitle=xtitle,
            ytitle=ytitle,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=edge_width,
            s_alpha=s_alpha,
            line_color=line_color,
            line_width=line_width,
            save=save,
            save_path=save_path,
        )

    def plot_group_scatter_plot(
        self,
        variable_name,
        group_type,
        mean_type,
        err_type,
        figsize=(5, 5),
        ytitle=None,
        ylim=None,
        b_colors=["darkorange", "forestgreen", "silver"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=["darkorange", "forestgreen", "silver"],
        s_size=5,
        s_alpha=0.8,
        test_type="parametric",
        test_method="Tukey",
        save=False,
        save_path=None,
    ):
        """Method for plotting the means and individual points for a given variable for 
            specified groups"""

        variable = getattr(self, variable_name)

        # Get the appropriate groups and store data
        data_dict = {}
        spine_groups = self.group_dict[group_type]

        for group in spine_groups:
            group_spines = getattr(self, group)
            group_data = variable[group_spines]
            group_data = group_data[~np.isnan(group_data)]
            data_dict[group] = group_data

        # Plot data
        sp.plot_swarm_bar_plot(
            data_dict,
            mean_type=mean_type,
            err_type=err_type,
            figsize=figsize,
            title=variable_name,
            xtitle=None,
            ytitle=ytitle,
            ylim=ylim,
            b_colors=b_colors,
            b_edgecolors=b_edgecolors,
            b_err_colors=b_err_colors,
            b_width=b_width,
            b_linewidth=b_linewidth,
            b_alpha=b_alpha,
            s_colors=s_colors,
            s_size=s_size,
            s_alpha=s_alpha,
            ahlines=None,
            save=save,
            save_path=save_path,
        )

        if test_type == "parametric":
            f_stat, anova_p, results_table, _ = t_utils.ANOVA_1way_posthoc(
                data_dict, test_method
            )
            print(f"F statistic: {f_stat}     p_value: {anova_p}")
            print(results_table)

        elif test_type == "nonparametric":
            f_stat, kruskal_p, results_table = t_utils.kruskal_wallis_test(
                data_dict, "Conover", test_method,
            )
            print(f"F statistic: {f_stat}     p_value: {kruskal_p}")
            print(results_table)

    def plot_multi_group_scatter_plots(
        self,
        variable_name,
        group_type,
        subgroup_type,
        mean_type,
        err_type,
        figsize=(5, 5),
        ytitle=None,
        ylim=None,
        b_colors=["darkorange", "forestgreen", "silver"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=["darkorange", "forestgreen", "silver"],
        s_size=5,
        s_alpha=0.8,
        test_method="Tukey",
        save=False,
        save_path=None,
    ):
        """Method for plotting means and individual points for a given variable of specified
            groups and subgroups"""

        if ytitle is None:
            ytitle = variable_name

        # Get the data
        variable = getattr(self, variable_name)

        # Setup groups
        group_list = [group_type, subgroup_type]
        groups = self.group_dict[group_type]
        subgroups = self.group_dict[subgroup_type]

        # Divide data into appropriate groups
        data_dict = {}
        for subgroup in subgroups:
            sub_dict = {}
            sg_spines = getattr(self, subgroup)
            for group in groups:
                g_spines = getattr(self, group)
                spines = np.array(sg_spines) * np.array(g_spines)
                spine_data = variable[spines]
                spine_data = spine_data[~np.isnan(spine_data)]
                sub_dict[group] = spine_data
            data_dict[subgroup] = sub_dict

        # Make the plot
        sp.plot_grouped_swarm_bar_plot(
            data_dict,
            group_list,
            mean_type=mean_type,
            err_type=err_type,
            figsize=figsize,
            title=variable_name,
            xtitle=None,
            ytitle=ytitle,
            ylim=ylim,
            b_colors=b_colors,
            b_edgecolors=b_edgecolors,
            b_err_colors=b_err_colors,
            b_width=b_width,
            b_linewidth=b_linewidth,
            b_alpha=b_alpha,
            s_colors=s_colors,
            s_size=s_size,
            s_alpha=s_alpha,
            ahlines=None,
            save=save,
            save_path=save_path,
        )

        anova_results, posthoc_results = t_utils.ANOVA_2way_posthoc(
            data_dict=data_dict,
            groups_list=group_list,
            variable=variable_name,
            method=test_method,
        )

        print(anova_results)
        print(posthoc_results)

    def plot_group_spine_mean_traces(
        self,
        group_type,
        trace_type,
        exclude=None,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["darkorange", "forestgreen", "silver"],
        ylim=None,
        test_method="holm-sidak",
        save=False,
        save_path=None,
    ):
        """Method to plot the mean activity of traces for specified spine groups"""

        traces = getattr(self, trace_type)

        spine_groups = self.group_dict[group_type]

        ind_mean_traces = {}
        mean_traces = []
        sem_traces = []
        used_groups = []
        for group in spine_groups:
            if group in exclude:
                continue
            spines = getattr(self, group)
            group_traces = compress(traces, spines)
            means = [
                np.nanmean(x, axis=1) for x in group_traces if type(x) == np.ndarray
            ]
            means = np.vstack(means)
            ind_mean_traces[group] = means.T
            means = np.unique(means, axis=0)
            group_mean = np.nanmean(means, axis=0)
            group_sem = stats.sem(means, axis=0, nan_policy="omit")
            mean_traces.append(group_mean)
            sem_traces.append(group_sem)
            used_groups.append(group)

        if self.parameters["zscore"]:
            ytitle = "zscore"
        else:
            ytitle = "\u0394" + "F/F\u2080"

        # Perform statistics
        anova_table, posthoc_dict, posthoc_table = t_utils.ANOVA_2way_mixed_posthoc(
            data_dict=ind_mean_traces,
            method=test_method,
            rm_vals=None,
            compare_type="between",
        )
        print(anova_table)
        # print(posthoc_table)

        # Find the significant differences in the traces
        combos = list(combinations(spine_groups, 2))
        significant_lines = defaultdict(list)
        for i, combo in enumerate(combos):
            if ahlines is None:
                break
            test = combo[0] + " vs. " + combo[1]
            group_idxs = [
                i for i, x in enumerate(posthoc_dict["posthoc comparison"]) if x == test
            ]
            sig_idxs = np.nonzero(posthoc_dict["raw p-vals"][group_idxs] <= 0.05)[0]
            # Append none if no significant differences
            if len(sig_idxs) == 0:
                significant_lines[test] = None
                continue
            break_idxs = np.nonzero(np.insert(np.diff(sig_idxs), 0, 0, axis=0) > 1)[0]
            # Append single line if there are no breaks
            if len(break_idxs) == 0:
                significant_lines[test].append((ahlines[i], sig_idxs[0], sig_idxs[-1]))
                continue
            for j, idx in enumerate(break_idxs):
                if j == 0:
                    significant_lines[test].append(
                        (ahlines[i], sig_idxs[0], sig_idxs[idx - 1])
                    )
                    try:
                        significant_lines[test].append(
                            (ahlines[i], sig_idxs[idx], sig_idxs[break_idxs[j + 1] - 1])
                        )
                    except IndexError:
                        significant_lines[test].append(
                            (ahlines[i], sig_idxs[idx], sig_idxs[-1])
                        )
                elif j == len(break_idxs) - 1:
                    significant_lines[test].append(
                        (ahlines[i], sig_idxs[idx], sig_idxs[-1])
                    )
                else:
                    significant_lines[test].append(
                        (ahlines[i], sig_idxs[idx], sig_idxs[break_idxs[j + 1] - 1])
                    )
        sp.plot_mean_activity_traces(
            mean_traces,
            sem_traces,
            used_groups,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            avlines=avlines,
            ahlines=significant_lines,
            figsize=figsize,
            colors=colors,
            title=trace_type,
            ytitle=ytitle,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def plot_ind_spine_mean_traces(
        self,
        trace_type,
        group,
        avlines=None,
        figsize=(10, 4),
        color="darkorange",
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot the mean activity traces for each spine in a given group"""

        traces = getattr(self, trace_type)
        group_spines = getattr(self, group)
        group_traces = compress(traces, group_spines)
        mean_traces = [np.nanmean(x, axis=1) for x in group_traces]
        sem_traces = [stats.sem(x, axis=1, nan_policy="omit") for x in group_traces]

        if self.parameters["zscore"]:
            ytitle = "zscore"
        else:
            ytitle = "\u0394" + "F/F\u2080"

        sp.ind_mean_activity_traces(
            mean_traces,
            sem_traces,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            avlines=avlines,
            figsize=figsize,
            color=color,
            title=f"{trace_type} {group}",
            ytitle=ytitle,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def plot_spine_coactivity_distance(
        self,
        variable_name,
        group_type,
        figsize=(5, 5),
        colors=["darkorange", "forestgreen", "silver"],
        ylim=None,
        ytitle=None,
        test_method="holm-sidak",
        save=False,
        save_path=None,
    ):
        """Method to plot distance-depencent spine coactivity for different groups"""
        coactivity_data = getattr(self, variable_name)
        spine_groups = self.group_dict[group_type]
        bins = self.parameters["Distance Bins"][1:]

        data_dict = {}
        for group in spine_groups:
            spines = getattr(self, group)
            data = coactivity_data[:, spines]
            data_dict[group] = data

        sp.plot_spine_coactivity_distance(
            data_dict=data_dict,
            bins=bins,
            colors=colors,
            title_suff=group_type,
            figsize=figsize,
            ylim=ylim,
            ytitle=ytitle,
            save=save,
            save_path=save_path,
        )

        two_way_mixed_anova, _, posthoc_table = t_utils.ANOVA_2way_mixed_posthoc(
            data_dict, method=test_method, rm_vals=bins, compare_type="between"
        )

        print(two_way_mixed_anova)
        print(posthoc_table)

        # Get individual datasets if applicable
        ind_variable_name = f"ind_{variable_name}"
        if hasattr(self, ind_variable_name):
            individual_data = getattr(self, ind_variable_name)
            for group in spine_groups:
                spines = getattr(self, group)
                ind_data = compress(individual_data, spines)
                pos, value = zip(*ind_data)
                non_nan = np.nonzero(~np.isnan(value))[0]
                value = value[non_nan]
                pos = pos[non_nan]
                sp.plot_sns_scatter_correlation(
                    var1=pos,
                    value=value,
                    CI=None,
                    title=variable_name,
                    xtitle="Distance (um)",
                    ytitle=ytitle,
                    figsize=figsize,
                    xlim=None,
                    ylim=None,
                    marker_size=5,
                    face_color="mediumblue",
                    edge_color="mediumblue",
                    edge_width=0.3,
                    s_alpha=0.5,
                    line_color="mediumblue",
                    line_width=1,
                    save=False,
                    save_path=None,
                )

    def plot_histogram(
        self,
        variable,
        bins,
        max_lim=None,
        group_type=None,
        exclude=None,
        avlines=None,
        figsize=(5, 5),
        color="mediumblue",
        alpha=0.4,
        save=False,
        save_path=None,
    ):
        """method to plot data variables as a histogram"""
        if group_type is None:
            data = getattr(self, variable)
        else:
            groups = self.group_dict[group_type]
            data = []
            for group in groups:
                if group in exclude:
                    continue
                else:
                    d = getattr(self, variable)
                    g = getattr(self, group)
                    group_data = compress(d, g)
                    data.append(group_data)
        if max_lim is not None:
            data = data[data < max_lim]
        sp.plot_histogram(
            data,
            bins,
            avlines,
            variable,
            variable,
            figsize,
            color,
            alpha,
            save,
            save_path,
        )

    def plot_group_mean_heatmaps(
        self,
        trace_type,
        group_type,
        figsize=(4, 5),
        hmap_range=None,
        center=None,
        sorted=False,
        normalize=False,
        cmap="plasma",
        save=False,
        save_path=None,
    ):
        """Method to plot the trial averaged activity across different groups"""

        traces = getattr(self, trace_type)
        spine_groups = self.group_dict[group_type]
        spine_groups.insert(0, "all")

        trace_dict = {}
        for group in spine_groups:
            if group != "all":
                spines = getattr(self, group)
                group_traces = compress(traces, spines)
                means = [
                    np.nanmean(x, axis=1) for x in group_traces if type(x) == np.ndarray
                ]
            else:
                means = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
            means = np.vstack(means)
            means = np.unique(means, axis=0).T
            trace_dict[group] = means

        if self.parameters["zscore"]:
            cbar_label = "zscore"
        else:
            cbar_label = "\u0394" + "F/F\u2080"

        sp.plot_spine_heatmap(
            trace_dict,
            figsize=figsize,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            title=trace_type,
            cbar_label=cbar_label,
            hmap_range=hmap_range,
            center=center,
            sorted=sorted,
            normalize=normalize,
            cmap=cmap,
            save=save,
            save_path=save_path,
        )

    def plot_group_trial_heatmaps(
        self,
        trace_type,
        group,
        figsize=(4, 5),
        hmap_range=None,
        center=None,
        sorted=False,
        normalize=False,
        cmap="plasma",
        save=False,
        save_path=None,
    ):
        """Method to plot individual trial activity of each spine within a given group"""

        traces = getattr(self, trace_type)
        spines = getattr(self, group)
        group_traces = compress(traces, spines)
        data_dict = {}
        for i, s in enumerate(group_traces):
            name = f"Spine {i+1}"
            data_dict[name] = s

        if self.parameters["zscore"]:
            cbar_label = "zscore"
        else:
            cbar_label = "\u0394" + "F/F\u2080"

        sp.plot_spine_heatmap(
            data_dict,
            figsize=figsize,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            title=trace_type,
            cbar_label=cbar_label,
            hmap_range=hmap_range,
            center=center,
            sorted=sorted,
            normalize=normalize,
            cmap=cmap,
            save=save,
            save_path=save_path,
        )

    def save_output(self):
        """Method to save the output"""
        if self.save_path is None:
            save_path = r"C:\Users\Desktop\Analyzed_data\grouped"
        else:
            save_path = self.save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Set up name based on analyzis parameters
        if self.parameters["zscore"]:
            a_type = "zscore"
        else:
            a_type = "dFoF"
        if self.parameters["Volume Norm"]:
            norm = "norm"
        else:
            norm = ""
        thresh = self.threshold

        save_name = f"{self.day}_{a_type}{norm}_{thresh}_coactivity_plasticity_data"
        save_pickle(save_name, self, save_path)
