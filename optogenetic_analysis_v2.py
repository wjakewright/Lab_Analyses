from itertools import compress

import dataframe_image as dfi
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from tabulate import tabulate

import opto_plotting as plotting
import utilities as util

sns.set()
sns.set_style("ticks")

"""Module containing several classes and functions to analyze optogenetic stimulatino data.
    Tests for ROIs significantly activated by stimulation while also providing a variety
    of visualizations. Designed to be implemented in Jupyter Notebook 
    Population_Optogenetic_Analysis.ipynb, which aids in loading imaging and behavioral 
    data as well as saves the results as an object. The object can be reloaded and methods
    can be used to perform reanalysis
    
    CREATOR
        William (Jake) Wright 12/06/2021"""


class optogenetic_analysis:
    """Class to analyze direct optogenetic stimulation sessions. Tests if ROIs are
        significantly activiated by stimulation. Also includes methods to get
        summarized activity and generate plots for vizuialization
        
        Can be utilized for a single session, power curve, or grouped sessions across
        mice"""

    def __init__(
        self,
        grouped,
        data,
        method,
        sampling_rate=30,
        window=[-2, 2],
        vis_window=None,
        stim_len=1,
        zscore=False,
        spines=False,
    ):
        """Initializing optogenetic_analysis Class.
        
            INPUT PARAMETERS
                grouped - boolean True or False indicating if the data is grouped
                        
                data - contains the imaging and behavioral data. Organization of data
                        depends on whether or not it is grouped.
                        
                        Grouped data will be a list containing data from each mouse.
                        The items in the list will be dictionaries, containing the
                        paired imaging and behavior data.

                        Ungrouped data will be a single dictionary containing the 
                        imaging and behavioral data
                
                method - string specifying which method is to be used to test
                    significance. Currnetly coded to accept:
                        
                        'test' - Performs Wilcoxon Signed-Rank Test
                        'shuff' - Compares the real difference in activity against
                                    a shuffled distribution
                
                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.

                window - list specifying the time before and after opto stim onset you
                        want to analyze. E.g. [-2,2] for 2s before and after. Default
                        set to [-2,2].

                vis_window - same as window, but for visualizing the data only
                            Default is set to none, in which case vis_window will
                            be determined by window and stim_len.

                stim_len - scaler specifying how long opto stim was delivered for.
                            Default is set to 1s.

                zscore - boolean True or False of whether to zscore the data.

                spines - boolean True or False of whether the ROIs are spines.
                """

        # Storing input data
        self.grouped = grouped
        self.data = data
        self.method = method
        self.sampling_rate = sampling_rate
        self.window = window
        self.before_t = window[0]  # before window in time(s)
        self.before_f = window[0] * sampling_rate  # before window in frames
        self.after_t = window[1]
        self.after_f = window[1] * sampling_rate
        self.vis_window = vis_window
        if vis_window is not None:
            self.vis_after_f = vis_window[1] * sampling_rate
        else:
            self.vis_after_f = 0
        self.stim_len = stim_len
        self.stim_len_f = stim_len * sampling_rate
        self.zscore = zscore
        self.spines = spines

        if self.vis_window is None:
            new_window = [self.window[0], self.window[1] + self.stim_len]
        else:
            new_window = self.vis_window
        self.new_window = new_window

        # Relevant imaging and behavioral data
        # Stored by calling pull_data method below
        self.ROIs = []
        self.dFoF = []
        self.i_trials = []
        self.behavior = []
        self.itis = []
        if grouped is True:
            for d in data:
                ROIs, dFoF, i_trials, behavior, itis = self.pull_data(d)
                self.ROIs.append(ROIs)
                self.dFoF.append(dFoF)
                self.i_trials.append(i_trials)
                self.behavior.append(behavior)
                self.itis.append(itis)
        else:
            ROIs, dFoF, i_trials, behavior, itis = self.pull_data(data)
            self.ROIs = ROIs
            self.dFoF = dFoF
            self.i_trials = i_trials
            self.behavior = behavior
            self.itis = itis

        # Additional attributes to be defined
        self.new_ROIs = None  # List of ROIs
        self.results_dict = None  # Dictionary of results for each ROI (keys)
        self.results_df = None  # Dataframe of results for each ROI (indexes)
        self.all_befores = None  # List of before mean activity for each ROI
        self.all_afters = None  # List of after mean activity for each ROI
        self.roi_stim_epochs = (
            None  # Dictionary of activity during each stim epoch for each ROI (keys)
        )
        self.roi_mean_sems = None  # Dictionary of mean and sem activity across stim epochs for each ROI (keys)

        self.analyze()

    def pull_data(self, data_dict):
        """Method to organize and store relevant imaging and behavioral data"""
        imaging = data_dict["imaging"]
        behavior = data_dict["behavior"]

        # Process Imaging data
        if self.spines is False:
            ROIs = []
            for i in list(imaging.SpineROIs[:-1]):
                ROIs.append("Cell " + str(np.floor(i)))
            dFoF = pd.DataFrame(data=imaging.Processed_dFoF.T, columns=ROIs)
        else:
            ROIs = []
            for i in list(imaging.SpineROIs[:-1]):
                ROIs.append("Spine " + str(np.floor(i)))
            dFoF = pd.DataFrame(data=imaging.Processed_dFoF.T, columns=ROIs)
        if self.zscore is True:
            dFoF = util.z_score(dFoF)
        else:
            dFoF = dFoF

        # Process Behavioral data
        i_trials = behavior.Imaged_Trials
        i_trials = i_trials == 1
        behavior = list(compress(behavior.Behavior_Frames, i_trials))
        itis = []
        for i in behavior:
            itis.append(i.states.iti2)

        ## Check iti intervals are consistent and within imaging period
        longest_window = max([self.after_f + self.stim_len_f, self.vis_after_f])
        for i, _ in enumerate(itis):
            if itis[i][1] + longest_window > len(dFoF):
                itis = itis[0 : i - 1]
            else:
                pass
        for i, _ in enumerate(itis):
            if (itis[i][1] - itis[i][0]) - self.stim_len_f == 1:
                itis[i] = [itis[i][0], itis[i][1] - 1]
            elif (itis[i][1] - itis[i][0]) - self.stim_len_f == -1:
                itis[i] = [itis[i][0], itis[i][1] + 1]
            elif (itis[i][1] - itis[i][0]) == self.stim_len_f:
                itis[i] = itis[i]
            else:
                del itis[i]

        return ROIs, dFoF, i_trials, behavior, itis

    def analyze(self, method=None):
        """Method to analyze data"""
        if method is None:
            method = self.method
        else:
            method = method

        if self.grouped is True:
            output = self.group_data_analysis(method)
        else:
            output = self.single_data_analysis(method)

        self.new_ROIs = output[0]
        self.results_dict = output[5]
        self.results_df = output[6]
        self.all_befores = output[1]
        self.all_afters = output[2]
        self.roi_stim_epochs = output[3]
        self.roi_mean_sems = output[4]

    def group_data_analysis(self, method):
        """Method to group data across mice for the main outputs"""
        new_ROIs = []
        all_befores = []
        all_afters = []
        roi_stim_epochs = []
        roi_mean_sems = []
        results_dicts = []

        for i, (dFoF, itis) in enumerate(zip(self.dFoF, self.itis)):
            for col in dFoF.columns:
                name = f"Mouse {i+1} {col}"
                new_ROIs.append(name)
            befores, afters = util.get_before_after_means(
                activity=dFoF,
                timestamps=itis,
                window=self.window,
                sampling_rate=self.sampling_rate,
                offset=False,
                single=False,
            )
            new_timestamps = [i[0] for i in itis]
            roi_stims, roi_means = util.get_trace_mean_sem(
                activity=dFoF,
                timestamps=new_timestamps,
                window=self.new_window,
                sampling_rate=self.sampling_rate,
            )
            roi_stims = list(roi_stims.values())
            roi_means = list(roi_means.values())
            all_befores.append(befores)
            all_afters.append(afters)
            roi_stim_epochs.append(roi_stims)
            roi_mean_sems.append(roi_means)
            r_dict, _, _ = util.significance_testing(
                imaging=dFoF,
                timestamps=itis,
                window=self.window,
                sampling_rate=self.sampling_rate,
                method=method,
            )
            results_dicts.append(r_dict)

        # Grouping the data
        ROIs = new_ROIs
        group_befores = [y for x in all_befores for y in x]
        group_afters = [y for x in all_afters for y in x]
        group_roi_stims = [y for x in roi_stim_epochs for y in x]
        group_roi_stim_epochs = dict(zip(ROIs, group_roi_stims))
        group_roi_means = [y for x in roi_mean_sems for y in x]
        group_roi_mean_sems = dict(zip(ROIs, group_roi_means))
        results_values = [list(result.values()) for result in results_dicts]
        group_results_values = [y for x in results_values for y in x]
        group_results_dict = dict(zip(ROIs, group_results_values))
        group_results_df = pd.DataFrame.from_dict(group_results_dict, orient="index")
        if "shuff_diff" in group_results_df.columns:
            group_results_df = group_results_df.drop(columns=["shuff_diffs"])
        self.un_grouped_roi_means = roi_stim_epochs

        return [
            ROIs,
            group_befores,
            group_afters,
            group_roi_stim_epochs,
            group_roi_mean_sems,
            group_results_dict,
            group_results_df,
        ]

    def single_data_analysis(self, method):
        """Method to analyze a single session from a single mouse"""
        dFoF = self.dFoF
        itis = self.itis
        ROIs = self.ROIs
        all_befores, all_afters = util.get_before_after_means(
            activity=dFoF,
            timestamps=itis,
            window=self.window,
            sampling_rate=self.sampling_rate,
            offset=False,
            single=False,
        )
        new_timestamps = [i[0] for i in itis]
        roi_stim_epochs, roi_mean_sems = util.get_trace_mean_sem(
            activity=dFoF,
            timestamps=new_timestamps,
            window=self.new_window,
            sampling_rate=self.sampling_rate,
        )
        results_dict, results_df, _ = util.significance_testing(
            imaging=dFoF,
            timestamps=itis,
            window=self.window,
            sampling_rate=self.sampling_rate,
            method=method,
        )

        return [
            ROIs,
            all_befores,
            all_afters,
            roi_stim_epochs,
            roi_mean_sems,
            results_dict,
            results_df,
        ]

    def display_results(
        self,
        fig1_size=(7, 8),
        fig2_size=(10, 20),
        fig3_size=(10, 10),
        fig4_size=(4, 5),
        fig5_size=(10, 10),
        title="default",
        save=False,
        hmap_range=None,
    ):
        """Method to display the data and results"""
        if save is True:
            value = input("File Name: \n")
            name = str(value)
        else:
            name = ""
        if self.grouped is True:
            for i, (dFoF, itis), in enumerate(zip(self.dFoF, self.itis)):
                new_name = name + f"Mouse {i+1}"
                plotting.plot_session_activity(
                    dFoF,
                    itis,
                    self.zscore,
                    figsize=fig1_size,
                    title=title,
                    save=save,
                    name=new_name,
                )
        else:
            plotting.plot_session_activity(
                self.dFoF,
                self.itis,
                self.zscore,
                figsize=fig1_size,
                title=title,
                save=save,
                name=name,
            )

        plotting.plot_each_event(
            self.roi_stim_epochs,
            self.new_ROIs,
            figsize=fig2_size,
            title=title,
            save=save,
            name=name,
        )
        plotting.plot_mean_sem(
            self.roi_mean_sems,
            self.new_window,
            self.new_ROIs,
            figsize=fig3_size,
            col_num=4,
            title=title,
            save=save,
            name=name,
        )
        plotting.plot_opto_heatmap(
            self.roi_mean_sems,
            self.zscore,
            self.sampling_rate,
            figsize=fig4_size,
            title=title,
            save=save,
            name=name,
            hmap_range=hmap_range,
        )
        if self.method == "shuff":
            plotting.plot_shuff_distribution(
                self.results_dict,
                self.new_ROIs,
                figsize=fig5_size,
                col_num=4,
                title=title,
                save=save,
                name=name,
            )
        else:
            pass
        display(self.results_df)
        if save is True:
            t_name = str(value) + "_table.png"
            dfi.export(self.results_df, t_name)


class power_curve:
    """Class to generate a power curve for optogenetic stimulations across different
        stimulation power levels. Includes visualizations and statistical testing between
        powers. Dependent on optogenetic_analysis Class"""

    def __init__(
        self,
        grouped,
        data,
        powers,
        method,
        sampling_rate=30,
        window=[2, 2],
        vis_window=None,
        stim_len=1,
        zscore=False,
        spines=False,
    ):
        """Initializing power_curve Class.
        
            INPUT PARAMETERS
                grouped - boolean True or False indicating if the data is grouped
                        
                data - list with each item representing the paired imaing and behavior data
                       for each power. If grouped, each item will be a tuple containing the 
                       different datasets. Can simply input list(zip(dataset1,dataset2)) to 
                       make such a list
                
                powers - list of the powers used
                
                method - string specifying which method is to be used to test
                    significance. Currnetly coded to accept:
                        
                        'test' - Performs Wilcoxon Signed-Rank Test
                        'shuff' - Compares the real difference in activity against
                                    a shuffled distribution
                
                sampling_rate - scaler specifying the image sampling rate. Default
                                is set to 30hz.

                window - list specifying the time before and after opto stim onset you
                        want to analyze. E.g. [-2,2] for 2s before and after. Default
                        set to [-2,2].

                vis_window - same as window, but for visualizing the data only
                            Default is set to none, in which case vis_window will
                            be determined by window and stim_len.

                stim_len - scaler specifying how long opto stim was delivered for.
                            Default is set to 1s.

                zscore - boolean True or False of whether to zscore the data.

                spines - boolean True or False of whether the ROIs are spines."""

        # Store initial inputs
        self.grouped = grouped
        if grouped is True:
            data = [
                list(i) for i in data
            ]  # convert the list of tuples to list of lists
        self.data = data
        self.powers = powers
        self.method = method
        self.sampling_rate = sampling_rate
        self.window = window
        self.vis_window = vis_window
        self.stim_len = stim_len
        self.zscore = zscore
        self.spines = spines

        # Analyze the data
        opto_objs = []
        for dataset in data:
            opto_obj = optogenetic_analysis(
                self.grouped,
                dataset,
                self.method,
                self.sampling_rate,
                self.window,
                self.vis_window,
                self.stim_len,
                self.zscore,
                self.spines,
            )
            opto_objs.append(opto_obj)
        self.opto_objs = opto_objs

    def generate_power_curve(self, method, save=False):
        """Method to generate power curve"""
        if save is True:
            value = input("File Name: \n")
            name = str(value)
        else:
            name = None
        power_diffs = []
        power_scatter = pd.DataFrame()
        power_sem = []
        percent_sig = []
        all_diffs = {}
        for opto_obj, power in zip(self.opto_objs, self.powers):
            results = opto_obj.results_df
            diffs = results["diff"]
            all_diffs[f"{power} mW"] = diffs
            power_diffs.append(np.mean(diffs))
            power_scatter[power] = np.array(diffs)
            power_sem.append(stats.sem(diffs))
            percent = (results["sig"].sum() / len(results.index)) * 100
            percent_sig.append(percent)

        plotting.plot_power_curve(
            self.powers,
            power_diffs,
            power_sem,
            power_scatter,
            percent_sig,
            self.zscore,
            save=save,
            name=name,
        )

        # Summary table
        summary_df = pd.DataFrame()
        n = len(power_scatter)
        for diff, sem, p_sig, power in zip(
            power_diffs, power_sem, percent_sig, self.powers
        ):
            summary_df[f"{power} mW"] = [diff, sem, p_sig, n]
        summary_df.set_axis(
            ["mean_diff", "sem_diff", "percent_sig", "n"], axis=0, inplace=True
        )
        summary_table = tabulate(summary_df, headers="keys", tablefmt="fancy_grid")

        # Perform one-way anova across different powers
        f_stat, anova_p, results_table, results_df = util.ANOVA_1way_bonferroni(
            all_diffs, method
        )

        # Display results
        print("One-Way ANOVA results")
        print(f"F statistic: ", f_stat, "\p-value: ", anova_p)
        print("\n")
        print(method + " Posttest Results")
        print(results_table)
        print("\n")
        print("Summary Statistics")
        print(summary_table)

        if save is True:
            s_name = str(value) + "_summary_table.png"
            r_name = str(value) + "_results_table.png"
            dfi.export(results_df, r_name)
            dfi.export(summary_df, s_name)

    def visualize_session(
        self,
        session,
        fig1_size=(7, 8),
        fig2_size=(10, 20),
        fig3_size=(10, 10),
        fig4_size=(4, 5),
        fig5_size=(10, 10),
        save=False,
        hmap_range=None,
    ):
        """Method to vizualize a single imaging session with a specific power"""
        name = f"{self.powers[session]} mW"
        self.opto_objs[session].display_results(
            fig1_size,
            fig2_size,
            fig3_size,
            fig4_size,
            fig5_size,
            title=name,
            save=save,
            hmap_range=hmap_range,
        )

