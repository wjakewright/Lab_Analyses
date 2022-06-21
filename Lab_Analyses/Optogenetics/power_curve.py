"""Module to generate a power curve for optogenetic stimulations across different
    stimulation power levels. Includes visualizations and statistical testing between
    powers"""
import os

import numpy as np
import pandas as pd
from Lab_Analyses.Optogenetics import opto_plotting as plotting
from Lab_Analyses.Utilities import test_utilities as test_utils
from Lab_Analyses.Utilities.save_load_pickle import save_pickle
from PyQt5.QtWidgets import QFileDialog
from scipy import stats
from tabulate import tabulate


class Power_Curve:
    """Class to generate the power curve for optogenetic responses across different
        stimulation powers.
    
    INPUT PARAMETERS
        data - dictionary of Opto_Response objects, wich each one corresponding to a different power

        posttest - str specifying which post test you wish to perform

        save - boolean to specify if you wish to save the output or not

        save_path - str specifying the path of where to save the 
        
    """

    def __init__(self, data, posttest, save=False, save_path=None):

        self.sessions = list(data.keys())
        self.data = list(data.values())
        self.posttest = posttest
        self.save = save
        self.powers = [x.split("_")[0] for x in data.keys()]

        # Get save name
        # Check if this is grouped data
        if self.data[0].ROI_types is None:
            self.rois = "allROIs"
        else:
            if len(self.data[0].ROI_types) > 1:
                sep = "_"
                self.rois = sep.join(self.data[0].ROI_types)
            else:
                self.rois = self.data[0].ROI_types[0]

        if self.data[0].group_name:
            name = [
                x for x in self.data[0].group_name.split("_") if x not in self.powers
            ]
            jname = "_".join(name)
            self.name = f"{jname}_power_curve"
        else:
            sess = self.data[0].session.split("_")[0]
            self.name = (
                f"{self.data[0].mouse_id}_{self.data[0].date}_{sess}_power_curve"
            )

        if self.save is True:
            if save_path is None:
                self.save_path = QFileDialog.getSaveFileName("Save Directory")[0]
            else:
                self.save_path = save_path
            self.save_path = os.path.join(self.save_path, self.rois)

            fig_path = self.save_path.split("\\")
            idx = fig_path.index("Analyzed_data")
            fig_path[idx] = "Figures"
            idx2 = fig_path.index("grouped")
            fig_path[idx2] = "grouped_data"
            self.fig_path = os.path.join("C:\\", *fig_path[1:])
            # make sure paths exist
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            if not os.path.isdir(self.fig_path):
                os.makedirs(self.fig_path)
        else:
            self.fig_path = None

        # Attributes to be defined later
        self.power_diffs = None
        self.power_scatter = None
        self.power_sem = None
        self.percent_sig = None
        self.all_diffs = None
        self.f_stat = None
        self.ANOVA_pval = None
        self.results_table = None
        self.results_df = None
        self.summary_df = None
        self.summary_table = None

        self.generate_power_curve()
        self.display_results()

        if self.save is True:
            self.save_output()

    def generate_power_curve(self):
        """Method to generate the power curve"""

        # Get the before and after differences for each power dataset
        power_diffs = []
        power_scatter = pd.DataFrame()
        power_sem = []
        percent_sig = []
        all_diffs = {}
        for data, power in zip(self.data, self.powers):
            results = data.results["df"]
            diffs = results["diff"]
            all_diffs[power] = diffs
            power_diffs.append(np.mean(diffs))
            power_scatter[power] = np.array(diffs)
            power_sem.append(stats.sem(diffs))
            percent = (results["sig"].sum() / len(results.index)) * 100
            percent_sig.append(percent)

        self.power_diffs = power_diffs
        self.power_scatter = power_scatter
        self.power_sem = power_sem
        self.percent_sig = percent_sig
        self.all_diffs = all_diffs

        # Perform one-way anova
        f_stat, anova_p, results_table, results_df = test_utils.ANOVA_1way_posthoc(
            all_diffs, self.posttest
        )
        self.f_stat = f_stat
        self.ANOVA_pval = anova_p
        self.results_table = results_table
        self.results_df = results_df

        # Summarize some results
        summary_df = pd.DataFrame()
        n = len(power_scatter)
        for diff, sem, p_sig, power in zip(
            power_diffs, power_sem, percent_sig, self.powers
        ):
            summary_df[power] = [diff, sem, p_sig, n]
        summary_df.set_axis(
            ["mean_diff", "sem_diff", "percent_sig", "n"], axis=0, inplace=True
        )
        summary_table = tabulate(summary_df, headers="keys", tablefmt="fancy_grid")
        self.summary_df = summary_df
        self.summary_table = summary_table

    def display_results(self):
        """Function to display the results"""

        # Generate plots
        plotting.plot_power_curve(
            ps=self.powers,
            diffs=self.power_diffs,
            sems=self.power_sem,
            scatter=self.power_scatter,
            percent_sig=self.percent_sig,
            zscore=True,
            save=self.save,
            name=self.name,
            save_path=self.fig_path,
        )

        # Display results
        print("One-Way ANOVA results")
        print(f"F statistic: ", self.f_stat, "\p-value: ", self.ANOVA_pval)
        print("\n")
        print(f"{self.posttest} Posttest Results")
        print(self.results_table)
        print("\n")
        print("Summary Statistics")
        print(self.summary_table)

    def save_output(self):
        """Method to save output it specified"""
        save_pickle(self.name, self, self.save_path)
