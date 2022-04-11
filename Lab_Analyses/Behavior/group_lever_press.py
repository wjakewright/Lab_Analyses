""""Module to analyze lever press behavior across mice within the same experimental group

    CREATOR - William (Jake) Wright 3/7/2022"""


import numpy as np
from Lab_Analyses.Behavior import behavior_plotting as bplot


class Group_Lever_Press:
    """Class for grouped analysis of lever press behavior across mice"""

    def __init__(self, files):
        """Initialize Group_Lever_Press Class

            INPUT PARAMETERS
                files - list containing Mouse_Lever_Behavior dataclass objects for each
                        mouse to be analyzed in the group
        """

        # Store variables and attributes
        self.files = files
        self.mice = []
        for file in files:
            self.mice.append(file.mouse_id)

        # Check that all mice have the same number of sessions
        self.check_same_sessions()

        self.sessions = files[0].sessions

        # Attributes to be defined later
        self.avg_corr_matrix = None
        self.within_sess_corr = None
        self.ind_within_sess_corr = None
        self.across_sess_corr = None
        self.ind_across_sess_corr = None
        self.avg_cue_to_reward = None
        self.ind_cue_to_reward = None
        self.avg_reaction_time = None
        self.ind_reaction_time = None
        self.success_rate = None
        self.ind_success_rate = None

        self.analyze_data()

    def analyze_data(self):
        """Parent function to analyze all key metrics"""

        self.average_correlation_matrix()
        self.analyze_within_sess_corr()
        self.analyze_cross_sess_corr()
        self.analyze_cue_to_reward()
        self.analyze_reaction_time()
        self.analyze_success_rate()

    def average_correlation_matrix(self):
        """Function to average the correlation matrices across mice"""
        # Grab correlation matricies for each mouse
        corr_matrices = [x.correlation_matrix for x in self.files]
        # Concatenate matrices along the 3rd axis
        cat_matrices = np.dstack(tuple(corr_matrices))
        # Get the mean
        mean_corr = np.nanmean(cat_matrices, axis=2)
        # Store result
        self.avg_corr_matrix = mean_corr

    def analyze_within_sess_corr(self):
        """Function get mean and sem within session correlations across sessions"""
        # Get all the within session correlations
        # Store as arrays in a list for each session
        all_within_corr = []
        for i, _ in enumerate(self.sessions):
            within_corr = [file.within_sess_corr[i] for file in self.files]
            all_within_corr.append(np.array(within_corr))

        within_corr_mean_sems = {"session": [], "mean": [], "sem": []}
        for session, corr in zip(self.sessions, all_within_corr):
            corr_mean = np.nanmean(corr)
            corr_sem = np.nanstd(corr, ddof=1) / np.sqrt(corr.size)
            within_corr_mean_sems["session"].append(session)
            within_corr_mean_sems["mean"].append(corr_mean)
            within_corr_mean_sems["sem"].append(corr_sem)

        self.ind_within_sess_corr = np.vstack(all_within_corr)
        self.within_sess_corr = within_corr_mean_sems

    def analyze_cross_sess_corr(self):
        """Function to get mean and sem across session correlations across sessions"""
        all_cross_corr = []
        for i, _ in enumerate(self.sessions[1:]):
            cross_corr = [file.across_sess_corr[i] for file in self.files]
            all_cross_corr.append(np.array(cross_corr))

        across_corr_mean_sems = {"session": [], "mean": [], "sem": []}
        for session, corr in zip(self.sessions[1:], all_cross_corr):
            corr_mean = np.nanmean(corr)
            corr_sem = np.nanstd(corr, ddof=1) / np.sqrt(corr.size)
            across_corr_mean_sems["session"].append(session)
            across_corr_mean_sems["mean"].append(corr_mean)
            across_corr_mean_sems["sem"].append(corr_sem)

        self.ind_across_sess_corr = np.vstack(all_cross_corr)
        self.across_sess_corr = across_corr_mean_sems

    def analyze_cue_to_reward(self):
        """Function to get mean and sem of the cue to reward across sessions"""
        all_cue_to_reward = []
        for i, _ in enumerate(self.sessions):
            cue_to_reward = [file.cue_to_reward[i] for file in self.files]
            all_cue_to_reward.append(np.array(cue_to_reward))

        avg_cue_to_reward = {"session": [], "mean": [], "sem": []}
        for session, ctr in zip(self.sessions, all_cue_to_reward):
            ctr_mean = np.nanmean(ctr)
            ctr_sem = np.nanstd(ctr, ddof=1) / np.sqrt(ctr.size)
            avg_cue_to_reward["session"].append(session)
            avg_cue_to_reward["mean"].append(ctr_mean)
            avg_cue_to_reward["sem"].append(ctr_sem)

        self.ind_cue_to_reward = np.vstack(all_cue_to_reward)
        self.avg_cue_to_reward = avg_cue_to_reward

    def analyze_reaction_time(self):
        """Function to get mean and sem of the movement reaction time across sessions"""
        all_reaction_time = []
        for i, _ in enumerate(self.sessions):
            reaction_time = [file.reaction_time[i] for file in self.files]
            all_reaction_time.append(np.array(reaction_time))

        avg_reaction_time = {"session": [], "mean": [], "sem": []}
        for session, rt in zip(self.sessions, all_reaction_time):
            rt_mean = np.nanmean(rt)
            rt_sem = np.nanstd(rt, ddof=1) / np.sqrt(rt.size)
            avg_reaction_time["session"].append(session)
            avg_reaction_time["mean"].append(rt_mean)
            avg_reaction_time["sem"].append(rt_sem)

        self.ind_reaction_time = np.vstack(all_reaction_time)
        self.avg_reaction_time = avg_reaction_time

    def analyze_success_rate(self):
        """Function to get mean and sem of the success rate across sessions"""
        all_success_rates = []
        for i, _ in enumerate(self.sessions):
            success_rates = [
                (file.rewards[i] / file.trials[i]) * 100 for file in self.files
            ]
            all_success_rates.append(np.array(success_rates))

        avg_success_rates = {"session": [], "mean": [], "sem": []}
        for session, sr in zip(self.sessions, all_success_rates):
            sr_mean = np.nanmean(sr)
            sr_sem = np.nanstd(sr, ddof=1) / np.sqrt(sr.size)
            avg_success_rates["session"].append(session)
            avg_success_rates["mean"].append(sr_mean)
            avg_success_rates["sem"].append(sr_sem)

        self.ind_success_rate = np.vstack(all_success_rates)
        self.success_rate = avg_success_rates

    def plot_average_data(
        self, to_plot=None, colors=None, ylims=None, save=False, save_path=None
    ):
        """Function to plot data
            
            INPUT PARAMETERS
                to_plot - list of strings specifying which plots you wish to plot. 
                        Accepts: 'success rate', 'cue to reward', 'reaction time', 'within correlation', 'across correlation', 'correlation heatmap'
                        Optional with default set to plot all
                        
                colors - dictionary to specify what color to plot each plot, with each key corresponding to the plot type
                        Keys it accepts: "success", "cue_to_reward", "reaction_time", "within", "across", "cmap"
                        Optional with default set to None. Can specify on the ones you wish to change. 
                        
                ylims - dictionary to specify what ylim to set each plot, with each key corresponding to the plot type
                        Keys it accepts: "success", "cue_to_reward", "reaction_time", "within", "across", "cmap"
                        Optional with default set to None. Can specify on the ones you wish to change.

                save - boolean specifying whether or not you want to save the figure

                save_path - str with the path of where to save the data
        """
        if save is True and save_path is None:
            raise Exception("Must specify the save path in order to save the figures")

        # Set up colors and ylims keyword arguments to ensure proper behavior if different partial inputs
        plot_keys = [
            "success",
            "cue_to_reward",
            "reaction_time",
            "within",
            "across",
            "cmap",
        ]
        if colors is None:
            colors = {
                "success": None,
                "cue_to_reward": None,
                "reaction_time": None,
                "within": None,
                "across": None,
                "cmap": None,
            }
        if ylims is None:
            ylims = {
                "success": None,
                "cue_to_reward": None,
                "reaction_time": None,
                "within": None,
                "across": None,
            }
        for key in plot_keys:
            if key not in colors.keys():
                colors[key] = None
            if key not in ylims.keys():
                ylims[key] = None

        if to_plot is None:
            bplot.plot_success_rate(
                self.success_rate["session"],
                self.success_rate["mean"],
                self.success_rate["sem"],
                self.ind_success_rate,
                ylim=ylims["success"],
                color=colors["success"],
                save=save,
                save_path=save_path,
            )
            bplot.plot_movement_reaction_time(
                self.avg_reaction_time["session"],
                self.avg_reaction_time["mean"],
                self.avg_reaction_time["sem"],
                self.ind_reaction_time,
                ylim=ylims["reaction_time"],
                color=colors["reaction_time"],
                save=save,
                save_path=save_path,
            )
            bplot.plot_cue_to_reward(
                self.avg_cue_to_reward["session"],
                self.avg_cue_to_reward["mean"],
                self.avg_cue_to_reward["sem"],
                self.ind_cue_to_reward,
                ylim=ylims["cue_to_reward"],
                color=colors["cue_to_reward"],
                save=save,
                save_path=save_path,
            )
            bplot.plot_movement_corr_matrix(
                self.avg_corr_matrix,
                title="Average Movement Correlation",
                cmap=colors["cmap"],
                save=save,
                save_path=save_path,
            )
            bplot.plot_within_session_corr(
                self.within_sess_corr["session"],
                self.within_sess_corr["mean"],
                self.within_sess_corr["sem"],
                self.ind_within_sess_corr,
                ylim=ylims["within"],
                color=colors["within"],
                save=save,
                save_path=save_path,
            )
            bplot.plot_across_session_corr(
                self.across_sess_corr["session"],
                self.across_sess_corr["mean"],
                self.across_sess_corr["sem"],
                self.ind_across_sess_corr,
                ylim=ylims["across"],
                color=colors["across"],
                save=save,
                save_path=save_path,
            )
        else:
            for x in to_plot:
                if x == "success rate":
                    bplot.plot_success_rate(
                        self.success_rate["session"],
                        self.success_rate["mean"],
                        self.success_rate["sem"],
                        self.ind_success_rate,
                        ylim=ylims["success"],
                        color=colors["success"],
                        save=save,
                        save_path=save_path,
                    )
                elif x == "reaction time":
                    bplot.plot_movement_reaction_time(
                        self.avg_reaction_time["session"],
                        self.avg_reaction_time["mean"],
                        self.avg_reaction_time["sem"],
                        self.ind_reaction_time,
                        ylim=ylims["reaction_time"],
                        color=colors["reaction_time"],
                        save=save,
                        save_path=save_path,
                    )
                elif x == "cue to reward":
                    bplot.plot_cue_to_reward(
                        self.avg_cue_to_reward["session"],
                        self.avg_cue_to_reward["mean"],
                        self.avg_cue_to_reward["sem"],
                        self.ind_cue_to_reward,
                        ylim=ylims["cue_to_reward"],
                        color=colors["cue_to_reward"],
                        save=save,
                        save_path=save_path,
                    )
                elif x == "correlation_heatmap":
                    bplot.plot_movement_corr_matrix(
                        self.avg_corr_matrix,
                        title="Average Movement Correlation",
                        cmap=colors["cmap"],
                        save=save,
                        save_path=save_path,
                    )
                elif x == "within correlation":
                    bplot.plot_within_session_corr(
                        self.within_sess_corr["session"],
                        self.within_sess_corr["mean"],
                        self.within_sess_corr["sem"],
                        self.ind_within_sess_corr,
                        ylim=ylims["within"],
                        color=colors["within"],
                        save=save,
                        save_path=save_path,
                    )
                elif x == "across correlation":
                    bplot.plot_across_session_corr(
                        self.across_sess_corr["session"],
                        self.across_sess_corr["mean"],
                        self.across_sess_corr["sem"],
                        self.ind_across_sess_corr,
                        ylim=ylims["across"],
                        color=colors["across"],
                        save=save,
                        save_path=save_path,
                    )
                else:
                    print(f"{x} is not a valid plot type")

    def check_same_sessions(self):
        """Function to check to make sure that all mice have same number of sessions"""
        sess_nums = [len(x.sessions) for x in self.files]
        values, counts = np.unique(sess_nums, return_counts=True)
        if len(values) > 1:
            diff_idx = [counts.index(x) for x in counts if x != np.max(counts)]
            error_mice = [file.mouse_id for file in np.array(self.files)[diff_idx]]
            majority_num = np.max(counts)
            diff_values = [x for x in np.array(counts)[diff_idx]]
            error_mice_values = zip(error_mice, diff_values)
            print(f"Supposed to have {majority_num} sessions")
            [print(f"{x} has {y} sessions") for x, y in error_mice_values]
            raise ValueError("Mice cannot have different number of sessions!!!!")

