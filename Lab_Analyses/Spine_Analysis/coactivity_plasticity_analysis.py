import os
from itertools import compress

import numpy as np
from Lab_Analyses.Spine_Analysis import spine_plotting as sp
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities.save_load_pickle import save_pickle
from scipy import stats


class Coactivity_Plasticity:
    """Class to handle the analysis of spine plasticity on coactivity datasets"""

    def __init__(
        self, data, threshold, exclude, vol_norm=False, save=False, save_path=None
    ):
        """Initialize the class
        
            INPUT PARAMETERS
                data - Spine_Coactivity_Data object, or a list of Spine_Coactivity_
                        Data Objects
                
                threshold - float specifying the threshold to consider for plasticity spines
                
                exclude - str specifying spine types to exclude from analysis (e.g., shaft)
        """

        # Check to see if one dataset was input and set up data
        if type(data) == list:
            self.dataset = data[0]
            self.subsequent_flags = data[1].spine_flags
            self.subsequent_volumes = data[1].spine_volumes
        elif isinstance(data, object):
            if data.followup_volumes is not None:
                self.dataset = data
                self.subsequent_flags = data.followup_flags
                self.subsequent_volumes = data.followup_volumes
            else:
                raise Exception("Data must have followup data containing spine volumes")

        self.day = self.dataset.day
        self.threshold = threshold
        self.exclude = exclude
        self.vol_norm = vol_norm
        self.parameters = self.dataset.parameters
        self.save = save
        self.save_path = save_path

        # Analyze the data
        self.analyze_plasticity()

        if save:
            self.save_output()

    def analyze_plasticity(self):
        """Method to calculate spine volume change and classify plasticity"""

        volume_data = [self.dataset.spine_volumes, self.subsequent_volumes]
        flag_list = [self.dataset.spine_flags, self.subsequent_flags]

        relative_volumes, spine_idxs = calculate_volume_change(
            volume_data, flag_list, norm=self.vol_norm, days=None, exclude=self.exclude
        )
        relative_volumes = np.array(list(relative_volumes.values())[-1])
        enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
            relative_volumes, self.threshold
        )

        # Store volume change and plasticity classification
        self.relative_volumes = relative_volumes
        self.enlarged_spines = enlarged_spines
        self.shrunken_spines = shrunken_spines
        self.stable_spines = stable_spines

        # refine some self attributes
        self.subsequent_volumes = self.subsequent_volumes[spine_idxs]
        self.subsequent_flags = [self.subsequent_flags[i] for i in spine_idxs]

        # Refine coactivity variables for only stable spines and store them
        attributes = list(self.dataset.__dict__.keys())
        ## Go through each attribute
        for attribute in attributes:
            # Save attributes that do not need to be refined
            if (
                attribute == "day"
                or attribute == "parameters"
                or attribute == "learned_movement"
            ):
                variable = getattr(self.dataset, attribute)
                setattr(self, attribute, variable)
                continue

            # Get the corresponding variable
            variable = getattr(self.dataset, attribute)

            # Skip variables that are None
            if variable is None:
                setattr(self, attribute, variable)
                continue

            # Refine variable based on spine idxs
            if type(variable) == np.ndarray:
                if len(variable.shape) == 1:
                    new_variable = variable[spine_idxs]
                elif len(variable.shape) == 2:
                    new_variable = variable[:, spine_idxs]
            elif type(variable) == list:
                try:
                    new_variable = [variable[i] for i in spine_idxs]
                except IndexError:
                    print(f"{attribute} is an empty list !!! Will skip.")
                    print(variable)
                    continue
            else:
                raise TypeError(
                    f"{attribute} {type(variable)} is incorrect datatype !!!"
                )
            # Store the attribute
            setattr(self, attribute, new_variable)

    def plot_volume_correlation(
        self,
        variable_name,
        CI=None,
        ytitle=None,
        xlim=None,
        ylim=None,
        face_color="mediumblue",
        edge_color="white",
        edge_width=0.3,
        s_alpha=0.5,
        line_color="mediumblue",
        line_width=1,
        save=False,
        save_path=None,
    ):
        """Method to plot and correlation a given variable against spine volume change"""
        variable = getattr(self, variable_name)
        # Remove nan values
        non_nan = np.nonzero(~np.isnan(variable))[0]
        variable = variable[non_nan]
        xtitle = "\u0394" + " spine volume"
        # Log transform relative volumes
        log_vol = np.log10(self.relative_volumes)
        log_vol = log_vol[non_nan]

        sp.plot_sns_scatter_correlation(
            log_vol,
            variable,
            CI,
            title=variable_name,
            xtitle=xtitle,
            ytitle=ytitle,
            figsize=(5, 5),
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

    def plot_group_scatter_plots(
        self,
        variable_name,
        group_type,
        mean_type,
        err_type,
        marker="o",
        figsize=(5, 5),
        ytitle=None,
        ylim=None,
        colors=["darkorange", "forestgreen", "silver"],
        s_alpha=0.3,
        save=False,
        save_path=None,
    ):
        """Method for plotting the means and individual points for a given variable
            for specified groups"""

        variable = getattr(self, variable_name)
        data_dict = {}
        if group_type == "plastic_spines":
            spine_groups = [
                "enlarged_spines",
                "shrunken_spines",
                "stable_spines",
            ]
        if group_type == "movement_spines":
            spine_groups = ["movement_spines", "nonmovement_spines"]
        if group_type == "rwd_movement_spines":
            spine_groups = ["rwd_movement_spines", "rwd_nonmovement_spines"]

        for group in spine_groups:
            group_spines = getattr(self, group)
            group_data = variable[group_spines]
            group_data = group_data[~np.nan(group_data)]
            data_dict[group] = group_data

        sp.plot_swarm_bar_plot(
            data_dict,
            mean_type,
            err_type,
            marker,
            figsize,
            title=variable_name,
            xtitle=None,
            ytitle=ytitle,
            ylim=ylim,
            linestyle="",
            m_colors=colors,
            s_colors=colors,
            s_alpha=s_alpha,
            save=save,
            save_path=save_path,
        )

    def plot_group_spine_mean_traces(
        self,
        group_type,
        trace_type,
        exclude=None,
        avlines=None,
        figsize=(5, 5),
        colors=["darkorange", "forestgreen", "silver"],
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot the mean activity traces for specified spine groups"""

        traces = getattr(self, trace_type)
        if group_type == "plastic_spines":
            spine_groups = [
                "enlarged_spines",
                "shrunken_spines",
                "stable_spines",
            ]
        if group_type == "movement_spines":
            spine_groups = ["movement_spines", "nonmovement_spines"]
        if group_type == "rwd_movement_spines":
            spine_groups = ["rwd_movement_spines", "rwd_nonmovement_spines"]

        mean_traces = []
        sem_traces = []
        used_groups = []
        for group in spine_groups:
            if group in exclude:
                continue
            spines = getattr(self, group)
            group_traces = compress(traces, spines)
            means = [x.nanmean(axis=1) for x in group_traces if type(x) == np.ndarray]
            means = np.vstack(means)
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

        sp.plot_mean_activity_traces(
            mean_traces,
            sem_traces,
            used_groups,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            avlines=avlines,
            figsize=figsize,
            colors=colors,
            title=trace_type,
            ytitle=ytitle,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def plot_spine_dend_mean_traces(
        self,
        trace_type,
        group_type,
        figsize=(8, 4),
        colors=["firebrick", "mediumblue"],
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot spine vs dendrite activity traces. Plots spines of the same
            group in different subplots"""

        if trace_type == "dend_triggered":
            spine_traces = getattr(self, "global_dend_triggered_spine_traces")
            dend_traces = getattr(self, "global_dend_triggered_dend_traces")
            avlines = None
        elif trace_type == "global":
            spine_traces = getattr(self, "global_coactive_spine_traces")
            dend_traces = getattr(self, "global_coactiive_dend_traces")
            avlines = [(0, x) for x in self.global_relative_spine_onsets]
        elif trace_type == "conj":
            spine_traces = getattr(self, "conj_coactive_spine_traces")
            dend_traces = getattr(self, "conj_coactive_dend_traces")
            avlines = [(0, x) for x in self.conj_relative_spine_dend_onsets]

        if group_type == "plastic":
            spine_groups = [
                "all",
                "enlarged_spines",
                "shrunken_spines",
                "stable_spines",
            ]
        if group_type == "movement":
            spine_groups = [
                "all",
                "movement_spines",
                "nonmovement_spines",
                "rwd_movement_spines",
                "rwd_nonmovement_spines",
            ]

        mean_dict = {}
        sem_dict = {}
        aline_dict = {}
        for group in spine_groups:
            if group != "all":
                # Get spines and average across all events to get mean trace
                group_spines = getattr(self, group)
                s_traces = compress(spine_traces, group_spines)
                s_traces = [
                    x.nanmean(axis=1) for x in s_traces if type(x) == np.ndarray
                ]
                d_traces = compress(dend_traces, group_spines)
                d_traces = [
                    x.nanmean(axis=1) for x in d_traces if type(x) == np.ndarray
                ]
                alines = compress(avlines, group_spines)
            else:
                s_traces = [
                    x.nanmean(axis=1) for x in spine_traces if type(x) == np.ndarray
                ]
                d_traces = [
                    x.nanmean(axis=1) for x in dend_traces if type(x) == np.ndarray
                ]
                alines = avlines
            s_traces = np.vstack(s_traces)
            d_traces = np.vstack(d_traces)
            d_traces = np.unique(d_traces, axis=0)
            s_mean = np.mean(s_traces, axis=0)
            s_sem = stats.sem(s_traces, axis=0)
            d_mean = np.mean(d_traces, axis=0)
            d_sem = stats.sem(d_traces, axis=0)
            mean_dict[group] = [d_mean, s_mean]
            sem_dict[group] = [d_sem, s_sem]
            aline_dict[group] = alines

        if self.parameters["zscore"]:
            ytitle = "zscore"
        else:
            ytitle = "\u0394" + "F/F\u2080"

        sp.plot_multi_mean_activity_traces(
            mean_dict,
            sem_dict,
            trace_type=["dendrite", "spine"],
            activity_window=self.parameters["Activity Window"],
            aveline_dict=aline_dict,
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
        color="mediumblue",
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot mean activity traces for each spine in a given
            spine group"""

        # Get relevant traces
        traces = getattr(self, trace_type)
        group_spines = getattr(self, group)
        group_traces = compress(traces, group_spines)
        mean_traces = [x.mean(axis=1) for x in group_traces]
        sem_traces = [stats.sem(x, axis=1) for x in group_traces]

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
        group_type,
        figsize=(5, 5),
        colors=["darkorange", "forestgreen", "silver"],
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot distance-dependent spine coactivity for different spine groups"""
        coactivity_data = self.distance_coactivity_rate
        bins = self.parameters["Distance Bins"]

        if group_type == "plastic_spines":
            spine_groups = [
                "enlarged_spines",
                "shrunken_spines",
                "stable_spines",
            ]
        if group_type == "movement_spines":
            spine_groups = [
                "movement_spines",
                "nonmovement_spines",
            ]
        if group_type == "rwd_movement_spines":
            spine_groups = [
                "rwd_movement_spines",
                "rwd_nonmovement_spines",
            ]

        group_dict = {}
        for group in spine_groups:
            spines = getattr(self, group)
            group_data = coactivity_data[:, spines]
            group_dict[group] = group_data

        sp.plot_spine_coactivity_distance(
            data_dict=group_dict,
            bins=bins,
            colors=colors,
            title_suff=group_type,
            figsize=figsize,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def plot_histogram(
        self,
        variable,
        bins,
        avlines=None,
        figsize=(5, 5),
        color="mediumblue",
        alpha=0.4,
        save=False,
        save_path=None,
    ):
        """Method to plot a data varialbe as a histogram"""
        data = getattr(self, variable)

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
        """Method to plot the trial averaged activity heatmaps across different spine groups"""

    def save_output(self):
        """Method to save the output"""
        if self.save_path is None:
            save_path = r"C:\Users\Desktop\Analyzed_data\grouped"
        else:
            save_path = self.save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Set up name based on some of the parameters
        if self.parameters["Movement Epoch"] is None:
            epoch_name = "session"
        else:
            epoch_name = self.parameters["Movement Epoch"]
        if self.parameters["zscore"]:
            a_type = "zscore"
        else:
            a_type = "dFoF"
        if self.parameters["Volume Norm"]:
            norm = "_norm"
        else:
            norm = ""
        thresh = self.threshold
        save_name = f"{self.day}_{epoch_name}_{a_type}{norm}_{thresh}_coactivity_plasticity_data"
        save_pickle(save_name, self, save_path)
