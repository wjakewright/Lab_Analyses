"""Module for determining if ROIs display significant changes in activity upon
    optogenetic stimulation"""

import os
import re
from dataclasses import dataclass
from itertools import compress

import dataframe_image as dfi
import Lab_Analyses.Optogenetics.opto_plotting as plotting
import Lab_Analyses.Utilities.data_utilities as data_utils
import Lab_Analyses.Utilities.test_utilities as test_utils
import numpy as np
import pandas as pd
from IPython.display import display
from Lab_Analyses.Utilities.save_load_pickle import save_pickle
from PyQt5.QtWidgets import QFileDialog


def classify_opto_responses(
    imaging_data,
    behavior_data,
    session_type,
    method,
    window=[-1, 1],
    vis_window=[-2, 3],
    stim_len=1,
    processed_dFoF=False,
    z_score=False,
    save=False,
):
    """Function to determine if ROIs display significant chagnes in activity upon optogenetic
        stimulation. Performed on a single mouse and single session
        
        INPUT PARAMETERS
            imaging_data - dataclass object of the Activity_Output from the 
                           Activity_Viewer for each mouse.
                           
            behavior_data - dataclass object of the Processed_Lever_Data
                            output from process_lever_behavior.

            session_type - str specifying what type of opto session it is. Used
                            to know where in the behavioral data the opto stim occured

            method - str specifying which method to use to assess significance
                    Accepts test and shuffle
            
            window - tuple specifying the time before and after opto stim onset you want
                    to analyze. Default set to [-1, 1]
            
            vis_window - same as window, but for how much of the trail you wish to visualize
                        for plotting purposes
            
            stim_len - int specifying how long the opto stimulation is delivered for.
                        Default is set to 1

            processed_dFoF - boolean specifying to use the processed dFoF instead of dFoF

            z_score - boolean specifying if you wish to z_score the data
                        
            save - boolean specifying if you wish to save the output or not

        OUTPUT PARAMETERS
            opto_response_output - dataclass object of the all the outputs
            
    """
    # Constant of what types of ROIs are acceptible for analysis
    ROI_TYPES = ["Soma", "Spine", "Dendrite"]

    # Pull out some important variable from the data
    sampling_rate = imaging_data.parameters["Sampling Rate"]
    before_t = window[0]
    after_t = window[1]
    before_f = window[0] * sampling_rate  # in terms of frames
    after_f = window[1] * sampling_rate  # in terms of frames
    vis_before_f = vis_window[0] * sampling_rate
    vis_after_f = vis_window[1] * sampling_rate
    session_len = len(imaging_data.fluorescence["Background"])

    # Get relevant behavioral data
    imaged_trials = behavior_data.imaged_trials == 1
    behavior_frames = list(compress(behavior_data.behavior_frames, imaged_trials))
    stims = []
    if session_type == "pulsed":
        for i in behavior_frames:
            stims.append(i.states.iti2)
        # Determine stimulation length from the iti durations
        stim_len = np.nanmedian([x[1] - x[2] for x in stims])
    else:
        return print("Only coded for pulsed trials right now")
    ## Check stimulation intervals are within imaging period
    longest_win = max([after_f + stim_len, vis_after_f])
    for idx, i in enumerate(stims):
        if i[1] + longest_win > session_len:
            stims = stims[:idx]

    # Oranize the different ROI types together
    ROI_ids = []
    dFoF = np.zeros(session_len).reshape(-1, 1)
    if processed_dFoF is False:
        for key, value in imaging_data.dFoF.items():
            if key in ROI_TYPES:
                ids = [f"{key} {x+1}" for x in np.arange(value.shape[1])]
                [ROI_ids.append(x) for x in ids]
                dFoF = np.hstack((dFoF, value))
    else:
        for key, value in imaging_data.processed_dFoF.items():
            if key in ROI_TYPES:
                ids = [f"{key} {x+1}" for x in np.arange(value.shape[1])]
                [ROI_ids.append(x) for x in ids]
                dFoF = np.hstack((dFoF, value))

    dFoF = dFoF[:, 1:]  # Getting rid of the initialized zeros

    if z_score is True:
        dFoF = test_utils.z_score(dFoF)

    # Start analyzing the data
    befores, afters = data_utils.get_before_after_means(
        activity=dFoF,
        timestamps=stims,
        window=window,
        sampling_rate=sampling_rate,
        offset=False,
        single=False,
    )
    new_stims = [i[0] for i in stims]
    roi_stims, roi_means = data_utils.get_trace_mean_sem(
        activity=dFoF,
        timestamps=new_stims,
        window=vis_window,
        sampling_rate=sampling_rate,
        single=False,
    )
    roi_stims = list(roi_stims.values())
    roi_means = list(roi_means.values())
    results_dict, results_df, _ = test_utils.response_testing(
        imaging=dFoF,
        timestamps=stims,
        window=window,
        sampling_rate=sampling_rate,
        method=method,
    )

    # Put results into the output
    analysis_settings = {
        "method": method,
        "window": window,
        "vis_window": vis_window,
        "processed": processed_dFoF,
        "stim length": stim_len,
        "z_score": z_score,
    }
    results = {"dict": results_dict, "df": results_df}
    opto_response_output = Opto_Repsonses(
        mouse_id=behavior_data.mouse_id,
        session=behavior_data.sess_name,
        date=behavior_data.date,
        ROI_ids=ROI_ids,
        imaging_parameters=imaging_data.parameters,
        analysis_settings=analysis_settings,
        dFoF=dFoF,
        stims=stims,
        roi_stims=roi_stims,
        roi_means=roi_means,
        results=results,
    )

    # Save section
    if save is True:
        # Set the save path
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        save_path = os.path.join(
            initial_path, behavior_data.mouse_id, "behavior", behavior_data.date
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Set filename
        save_name = f"{behavior_data.mouse_id}_{behavior_data.date}_{behavior_data.date}_opto_response"
        # Save the data as a pickle file
        save_pickle(save_name, opto_response_output, save_path)

    return opto_response_output


def group_opto_responses(data, group_name, save=False, save_path=None):
    """Function to group data across mice for the same session type
    
        INPUT PARAMETERS
            data - list of Opto_Responses dataclasses for each mouse being grouped

            group_name - str specifying the name of the group

            save_path - str specifying where to save the output
            
        OUTPUT PARAMETERS
            grouped_responses - Opto_Responses data class
            
    """
    # first test the different datasets have the same imaging and analysis parameters
    first_sampling_rate = data[0].imaging_parameters["Sampling Rate"]
    first_settings = data[0].analysis_settings

    for dataset in data[1:]:
        if dataset.imaging_parameters["Sampling Rate"] != first_sampling_rate:
            return print(
                f"{dataset.mouse_id}_{dataset.session}_{dataset.date} has a different sampling rate"
            )
        if list(dataset.analysis_settings.values()) != list(first_settings.values()):
            return print(
                f"{dataset.mouse_id}_{dataset.session}_{dataset.date} has a different analysis settings"
            )

    # Start grouping the datasets

    # Put all the mice ids, sessions, and dates into lists
    mouse_ids = [x.mouse_id for x in data]
    sessions = [x.session for x in data]
    dates = [x.date for x in data]
    imaging_parameters = [x.imaging_parmaeters for x in data]
    analysis_settings = [x.analysis_settings for x in data]
    dFoF = [x.dFoF for x in data]
    stims = [x.stims for x in data]

    # Generate new ROIs
    new_ROIs = []
    for dataset in data:
        for roi in dataset.ROI_ids:
            new_roi = f"{dataset.mouse_id}_{dataset.sessoin}_{dataset.date}_{roi}"
            new_ROIs.append(new_roi)

    # Group the data together across mice
    roi_stims = [x.roi_stims for x in data]
    group_roi_stims = [y for x in roi_stims for y in x]
    group_roi_stim_epochs = dict(zip(new_ROIs, group_roi_stims))
    roi_mean_sems = [x.roi_mean_sems for x in data]
    group_roi_means = [y for x in roi_mean_sems for y in x]
    group_roi_mean_sems = dict(zip(new_ROIs, group_roi_means))
    results = [x.results["dict"] for x in data]
    results_values = [list(result.values()) for result in results]
    group_results_values = [y for x in results_values for y in x]
    group_results_dict = dict(zip(new_ROIs, group_results_values))
    group_results_df = pd.DataFrame.from_dict(group_results_dict, orient="index")
    if "shuff_diffs" in group_results_df.columns:
        group_results_df = group_results_df.drop(columns=["shuff_diffs"])
    group_results = {"dict": group_results_dict, "df": group_results_df}

    # Generate the output
    grouped_responses = Opto_Repsonses(
        mouse_id=mouse_ids,
        session=sessions,
        date=dates,
        ROI_ids=new_ROIs,
        imaging_parameters=imaging_parameters,
        analysis_settings=analysis_settings,
        dFoF=dFoF,
        stims=stims,
        roi_stims=group_roi_stim_epochs,
        roi_mean_sems=group_roi_mean_sems,
        results=group_results,
        group_name=group_name,
    )

    if save is True and save_path is None:
        save_path = QFileDialog.getSaveFileName("Save Directory")[0]

    if save is True:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_name = group_name
        save_pickle(save_name, grouped_responses, save_path)

    return grouped_responses


#################### DATACLASSES #######################
@dataclass
class Opto_Repsonses:
    mouse_id: str  # If data is grouped this will be a lsit
    session: str  # If data is grouped this will be a list
    date: str  # If data is grouped this will be a list
    ROI_ids: list
    imaging_parameters: dict  # If data is grouped this will be a list
    analysis_settings: dict  # If data is grouped this will be a list
    dFoF: np.array  # If data is grouped this will be a list
    stims: list
    roi_stims: list
    roi_mean_sems: list
    results: dict
    group_name: str = ""  # Filled in if data are grouped together

    def display_results(
        self, figsizes=None, parameters=None, save=False, save_path=None
    ):
        """Method to make plots and display the data
        
            INPUT PARAMETERS
                figsizes - dict with a tuple for the specified figure
                
                parameters - dict containg the following paramters:
                                title - str
                                hmap_range - tuple
                                cmap - str
                                zeroed - bool
                                sort - bool
                                center - int
                
                save - boolean specifying if you wish to save the figures
        """
        if parameters is None:
            parameters = {
                "title": "default",
                "hmap_range": None,
                "cmap": None,
                "zeroed": False,
                "sort": False,
                "center": None,
            }
        if figsizes is None:
            figsizes = {
                "fig1": (7, 8),
                "fig2": (10, 10),
                "fig3": (10, 10),
                "fig4": (4, 5),
                "fig5": (10, 10),
            }

        # Set up save values
        if save is True and save_path is None:
            save_path = QFileDialog.getSaveFileName("Save Directory")[0]

        # Plot the session activity
        ## If data is grouped it will plot the different mice seperates
        ### Check if there are multiple groups
        multi_mice = list(filter(re.compile("JW").match, self.ROI_ids))

        # Set up data names for plotting and saving purposes
        if multi_mice:
            data_name = self.group_name
        else:
            data_name = f"{self.mouse_id}_{self.session}_{self.date}"

        # If there are multiple datasets, then plot each dataset seperately
        if multi_mice:
            for i, dataset in enumerate(self.dFoF):
                plotting.plot_session_activity(
                    dataset,
                    self.stims[i],
                    self.analysis_settings["z_score"],
                    figsize=figsizes["fig1"],
                    title=parameters["title"],
                    save=save,
                    name=f"{self.mouse_id[i]}_{self.session[i]}_{self.date[i]}",
                    save_path=save_path,
                )
        else:
            plotting.plot_session_activity(
                self.dFoF,
                self.stims,
                self.analysis_settings["z_score"],
                figsize=figsizes["fig1"],
                title=parameters["title"],
                save=save,
                name=data_name,
                save_path=save_path,
            )

        # Plot the trial heatmaps
        plotting.plot_trial_heatmap(
            self.roi_stims,
            self.analysis_settings["z_score"],
            self.imaging_parameters["Sampling Rate"],
            figsize=figsizes["fig2"],
            title=parameters["title"],
            cmap=parameters["cmap"],
            save=save,
            name=data_name,
            hmap_range=parameters["hmap_range"],
            zeroed=parameters["zeroed"],
            sort=False,
            center=parameters["center"],
            save_path=save_path,
        )

        # Plot the trial averaged trace
        plotting.plot_mean_sem(
            self.roi_mean_sems,
            self.analysis_settings["vis_window"],
            self.ROI_ids,
            figsize=figsizes["fig3"],
            col_num=4,
            title=parameters["title"],
            save=save,
            name=data_name,
            save_path=save_path,
        )

        # Plot the mean heatmap
        plotting.plot_mean_heatmap(
            self.roi_means,
            self.analysis_settings["z_score"],
            self.imaging_parameters["Sampling Rate"],
            figsize=figsizes["fig4"],
            title=parameters["title"],
            cmap=parameters["cmap"],
            save=save,
            name=data_name,
            hmap_range=parameters["hmap_range"],
            zeroed=parameters["zeroed"],
            sort=parameters["sort"],
            center=parameters["center"],
            save_path=save_path,
        )

        if self.analysis_settings["method"] == "shuffle":
            plotting.plot_shuff_distribution(
                self.results["dict"],
                self.ROI_ids,
                figsize=figsizes["fig5"],
                col_num=4,
                title=parameters["title"],
                save=save,
                name=data_name,
                save_path=save_path,
            )

        display(self.results["df"])
        if save is True:
            table_name = os.path.join(save_path, data_name + "_table.png")
            dfi.export(self.results["df"], table_name)

