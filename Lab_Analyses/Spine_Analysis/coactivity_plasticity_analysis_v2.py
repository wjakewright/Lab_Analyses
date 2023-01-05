import os
from itertools import compress

import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis import spine_plotting as sp
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change, classify_plasticity)
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
            relative_volumes, self.threshold
        )

        # Store volume change and plasticity classifications
        self.relative_volumes = relative_volumes
        self.enlarge_spines = enlarged_spines
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
        sp.plot_swam_bar_plot(
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

    def plot_multi_group_scatter_plots(
        self, 
        variable_name,
        group_type,
        subgroup_type,
        mean_type,
        err_type,
        figsize=(5,5),
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
        group_list = [subgroup_type, group_type]
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
