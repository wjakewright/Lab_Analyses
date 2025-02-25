""""Module to analyze lever press behavior across mice within the same experimental group

    CREATOR - William (Jake) Wright 3/7/2022"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from Lab_Analyses.Plotting.plot_general_heatmap import plot_general_heatmap
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot


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

    def plot_data(
        self,
        figsize=(9, 8),
        colors=None,
        ylims=None,
        plot_ind=False,
        save=False,
        save_path=None,
    ):
        """Function to plot data

        INPUT PARAMETERS
            colors - dictionary to specify what color to plot each plot, with each key corresponding to the plot type
                    Keys it accepts: "success", "cue_to_reward", "reaction_time", "within", "across", "cmap"
                    Optional with default set to None. Can specify on the ones you wish to change.

            ylims - dictionary to specify what ylim to set each plot, with each key corresponding to the plot type
                    Keys it accepts: "success", "cue_to_reward", "reaction_time", "within", "across", "cmap"
                    Optional with default set to None. Can specify on the ones you wish to change.

            save - boolean specifying whether or not you want to save the figure

            save_path - str with the path of where to save the data
        """
        # Pull the data
        success_rate = self.ind_success_rate
        success_sessions = self.success_rate["session"]
        reaction_time = self.ind_reaction_time
        reaction_sessions = self.avg_reaction_time["session"]
        cue_to_reward = self.ind_cue_to_reward
        cue_sessions = self.avg_cue_to_reward["session"]
        within_corr = self.ind_within_sess_corr
        within_sessions = self.within_sess_corr["session"]
        across_corr = self.ind_across_sess_corr
        across_sessions = self.across_sess_corr["session"]
        correlation_matrix = self.avg_corr_matrix

        # Perform some correlations
        corr_results = []
        for data in [
            success_rate,
            reaction_time,
            cue_to_reward,
            within_corr,
            across_corr,
        ]:
            x = list(range(data.shape[0]))
            val_1 = []
            val_2 = []
            for i in range(data.shape[1]):
                val_1.append(data[:, i])
                val_2.append(x)
            val_1 = np.array([y for x in val_1 for y in x])
            val_2 = np.array([y for x in val_2 for y in x])
            non_nan = np.nonzero(~np.isnan(val_1))[0]
            val_1 = val_1[non_nan]
            val_2 = val_2[non_nan]
            r, p = stats.pearsonr(val_2, val_1)
            corr_results.append(f"r = {r:.3}  p = {p:.3E}")

        # Construct the figure
        fig, axes = plt.subplot_mosaic(
            [
                ["l_upper", "r_upper"],
                ["l_middle", "r_middle"],
                ["l_bottom", "r_bottom"],
            ],
            figsize=figsize,
        )
        fig.suptitle("Summarized Lever Press Behavior")
        fig.subplots_adjust(hspace=1, wspace=0.5)

        ############# Add data to the plots ##############
        # Succss rate
        plot_multi_line_plot(
            data_dict={"avg": success_rate},
            x_vals=success_sessions,
            plot_ind=plot_ind,
            figsize=(5, 5),
            title=f"Success Rate\n{corr_results[0]}",
            ytitle="Successful trials (%)",
            xtitle="Session",
            ylim=ylims["success"],
            line_color=colors["success"],
            face_color="white",
            m_size=7,
            linewidth=1.5,
            linestyle="-",
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["l_upper"],
            legend=False,
            save=False,
            save_path=None,
        )
        # Reaction time
        plot_multi_line_plot(
            data_dict={"avg": reaction_time},
            x_vals=reaction_sessions,
            plot_ind=plot_ind,
            figsize=(5, 5),
            title=f"Movement Reaction Time\n{corr_results[1]}",
            ytitle="Reaction time (s)",
            xtitle="Session",
            ylim=ylims["reaction_time"],
            line_color=colors["reaction_time"],
            face_color="white",
            m_size=7,
            linewidth=1.5,
            linestyle="-",
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["r_upper"],
            legend=False,
            save=False,
            save_path=None,
        )
        # Cue to reward
        plot_multi_line_plot(
            data_dict={"avg": cue_to_reward},
            x_vals=cue_sessions,
            plot_ind=plot_ind,
            figsize=(5, 5),
            title=f"Cue to Reward Time\n{corr_results[2]}",
            ytitle="Cue to reward time (s)",
            xtitle="Session",
            ylim=ylims["cue_to_reward"],
            line_color=colors["cue_to_reward"],
            face_color="white",
            m_size=7,
            linewidth=1.5,
            linestyle="-",
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["l_middle"],
            legend=False,
            save=False,
            save_path=None,
        )
        # Within session correlation
        plot_multi_line_plot(
            data_dict={"avg": within_corr},
            x_vals=within_sessions,
            plot_ind=plot_ind,
            figsize=(5, 5),
            title=f"Within Session Correlation\n{corr_results[3]}",
            ytitle="Move. Correlation (r)",
            xtitle="Session",
            ylim=ylims["within"],
            line_color=colors["within"],
            face_color="white",
            m_size=7,
            linewidth=1.5,
            linestyle="-",
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["r_middle"],
            legend=False,
            save=False,
            save_path=None,
        )
        # Across session correlation
        plot_multi_line_plot(
            data_dict={"avg": across_corr},
            x_vals=across_sessions,
            plot_ind=plot_ind,
            figsize=(5, 5),
            title=f"Across Session Correlation\n{corr_results[4]}",
            ytitle="Move. Correlation (r)",
            xtitle="Session",
            ylim=ylims["across"],
            line_color=colors["across"],
            face_color="white",
            m_size=7,
            linewidth=1.5,
            linestyle="-",
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["l_bottom"],
            legend=False,
            save=False,
            save_path=None,
        )
        # Movement correlation heatmap
        axes["r_bottom"].set_aspect(aspect="equal", adjustable="box")
        plot_general_heatmap(
            data=correlation_matrix,
            figsize=(5, 5),
            title="Movement Correlation",
            xtitle="Session",
            ytitle="Session",
            cbar_label="Correlation (r)",
            hmap_range=ylims["cmap"],
            center=None,
            cmap=colors["cmap"],
            axis_width=2.5,
            tick_len=3,
            ax=axes["r_bottom"],
            save=False,
            save_path=None,
        )

        fig.tight_layout()

        # Save section
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            fname = os.path.join(save_path, "Summarized_Lever_Press_Data")
            fig.savefig(fname + ".pdf")

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
