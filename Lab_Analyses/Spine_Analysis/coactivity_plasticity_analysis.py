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

    def plot_plastic_spine_groups(
        self,
        variable_name,
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
        """Method for plotting means and individual points for a given variable for each spine
            group"""
        variable = getattr(self, variable_name)

        enlarged_data = variable[self.enlarged_spines]
        enlarged_data = enlarged_data[~np.isnan(enlarged_data)]
        shrunken_data = variable[self.shrunken_spines]
        shrunken_data = shrunken_data[~np.isnan(shrunken_data)]
        stable_data = variable[self.stable_spines]
        stable_data = stable_data[~np.isnan(stable_data)]

        data_dict = {
            "Enlarged": enlarged_data,
            "Shrunken": shrunken_data,
            "Stable": stable_data,
        }

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

    def plot_plastic_spine_mean_traces(
        self,
        trace_type,
        exclude=None,
        avlines=None,
        figsize=(5, 5),
        colors=["darkorange", "forestgreen", "silver"],
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot the mean activity traces for each plastic spine group"""

        # Get the relevant traces
        traces = getattr(self, trace_type)
        enlarged_traces = compress(traces, self.enlarged_spines)
        shrunken_traces = compress(traces, self.shrunken_spines)
        stable_traces = compress(traces, self.stable_spines)
        # Get the means for each spine
        enlarged_mean_traces = [
            x.mean(axis=1) for x in enlarged_traces if type(x) == np.ndarray
        ]
        enlarged_mean_traces = np.vstack(enlarged_mean_traces)
        shrunken_mean_traces = [
            x.mean(axis=1) for x in shrunken_traces if type(x) == np.ndarray
        ]
        shrunken_mean_traces = np.vstack(shrunken_mean_traces)
        stable_mean_traces = [
            x.mean(axis=1) for x in stable_traces if type(x) == np.ndarray
        ]
        stable_mean_traces = np.vstack(stable_mean_traces)
        # Get mean and sem across traces
        enlarged_mean = np.mean(enlarged_mean_traces, axis=0)
        enlarged_sem = stats.sem(enlarged_mean_traces, axis=0)
        shrunken_mean = np.mean(shrunken_mean_traces, axis=0)
        shrunken_sem = stats.sem(shrunken_mean_traces, axis=0)
        stable_mean = np.mean(stable_mean_traces, axis=0)
        stable_sem = stats.sem(stable_mean_traces, axis=0)

        # prepare data for plotting
        spine_groups = ["enlarged", "shrunken", "stable"]
        mean_list = []
        sem_list = []
        plot_colors = []
        for i, group in enumerate(spine_groups):
            if group == exclude:
                continue
            mean_list.append(eval(f"{group}_mean"))
            sem_list.append(eval(f"{group}_sem"))
            plot_colors.append(colors[i])
        if self.parameters["zscore"]:
            ytitle = "zscore"
        else:
            ytitle = "\u0394" + "F/F\u2080"

        # Make the plot
        sp.plot_mean_activity_traces(
            mean_list,
            sem_list,
            sampling_rate=self.parameters["Sampling Rate"],
            activity_window=self.parameters["Activity Window"],
            avlines=avlines,
            figsize=figsize,
            colors=plot_colors,
            title=trace_type,
            ytitle=ytitle,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def plot_spine_dend_mean_traces(
        self,
        trace_type,
        avlines=None,
        figsize=(8, 4),
        colors=["firebrick", "mediumblue"],
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Method to plot spine vs dendrite activity traces. Plots all groups of spines in the 
           same subplot
           
           trace_types accepted include 'dend_triggered', 'global', 'conj'
        """
        if trace_type == "dend_triggered":
            spine_traces = getattr(self, "global_dend_triggered_spine_traces")
            dend_traces = getattr(self, "global_dend_triggered_dend_traces")
        elif trace_type == "global":
            spine_traces = getattr(self, "global_coactive_spine_traces")
            dend_traces = getattr(self, "global_coactiive_dend_traces")
        elif trace_type == "conj":
            spine_traces = getattr(self, "conj_coactive_spine_traces")
            dend_traces = getattr(self, "conj_coactive_dend_traces")

        spine_groups = ["all", "enlarged_spines", "shrunken_spines", "stable_spines"]
        mean_dict = {}
        sem_dict = {}
        for group in spine_groups:
            if group != "all":
                group_spines = getattr(self, group)
                s_traces = compress(spine_traces, group_spines)
                d_traces = compress(dend_traces, group_spines)
            else:
                s_traces = spine_traces
                d_traces = dend_traces
            s_traces = np.vstack(s_traces)
            d_traces = np.vstack(d_traces)
            d_traces = np.unique(d_traces, axis=0)
            s_mean = np.mean(s_traces, axis=0)
            s_sem = stats.sem(s_traces, axis=0)
            d_mean = np.mean(d_traces, axis=0)
            d_sem = stats.sem(d_traces, axis=0)
            mean_dict[group] = [d_mean, s_mean]
            sem_dict[group] = [d_sem, s_sem]

        if self.parameters["zscore"]:
            ytitle = "zscore"
        else:
            ytitle = "\u0394" + "F/F\u2080"

        sp.plot_multi_mean_activity_traces(
            mean_dict,
            sem_dict,
            trace_type=["dendrite", "spine"],
            activity_window=self.parameters["Activity Window"],
            avline_dict=avlines,
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
