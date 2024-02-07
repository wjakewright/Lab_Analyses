import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def analyze_kir_population_data(
    mouse_list, save=False,
):
    """Function to analyze kir population data
    
        INPUT PARAMETERS
            mouse_list - list of str specifying the mice to be analyzed

            save_grouped - boolean specifying whether or not to group all mice together
                            and save

    """
    analyzed_data = []

    # Analyze each mouse seperately
    for mouse in mouse_list:
        print("----------------------------------------")
        print(f"- Analyzing {mouse}")
        # Load the datasets
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        data_path = os.path.join(initial_path, mouse, "kir_population_data")
        fnames = next(os.walk(data_path))[2]

        # Analyze each FOV
        for fname in fnames:
            data = load_pickle([fname], path=data_path)[0]
            # Pull data
            fov = data.fov
            kir_positive = data.kir_positive
            dFoF = data.dFoF
            processed_dFoF = data.processed_dFoF
            activity = data.activity
            estimated_spikes = data.estimated_spikes
            binned_spikes = data.binned_spikes
            expression_intensity = data.expression_intensity

            # Zscore some of the activity
            zscore_dFoF = d_utils.z_score(dFoF)
            zscore_processed_dFoF = d_utils.z_score(processed_dFoF)
            zscore_spikes = d_utils.z_score(estimated_spikes)
            zscore_binned_spikes = d_utils.z_score(binned_spikes)

            # Calculate the event rate
            event_rates = d_utils.calculate_activity_event_rate(
                activity, data.imaging_parameters["Sampling Rate"]
            )
            # Get chance event rates
            shuff_event_rates = chance_event_rates(
                event_rates, kir_positive, permutations=1000
            )

            # Get event amplitudes
            ## Get timestamps
            events = []
            for i in range(activity.shape[1]):
                curr_activity = activity[:, i]
                curr_activity = curr_activity[~np.isnan(curr_activity)]
                e = get_activity_timestamps(curr_activity)
                e = [x[0] for x in e]
                e = refine_activity_timestamps(
                    e,
                    (-1, 1),
                    len(curr_activity),
                    sampling_rate=data.imaging_parameters["Sampling Rate"],
                )
                events.append(e)
            _, amplitudes, _ = analyze_event_activity(
                dFoF=processed_dFoF,
                timestamps=events,
                activity_window=(-1, 1),
                sampling_rate=data.imaging_parameters["Sampling Rate"],
            )
            shuff_amplitudes = chance_event_rates(
                amplitudes, kir_positive, permutations=1000,
            )

            # Store data
            ind_data = Kir_Activity_Data(
                mouse_id=mouse,
                fov=fov,
                sampling_rate=data.imaging_parameters["Sampling Rate"],
                kir_ids=kir_positive,
                dFoF=dFoF,
                processed_dFoF=processed_dFoF,
                activity=activity,
                estimated_spikes=estimated_spikes,
                binned_spikes=binned_spikes,
                zscore_dFoF=zscore_dFoF,
                zscore_processed_dFoF=zscore_processed_dFoF,
                zscore_spikes=zscore_spikes,
                zscore_binned_spikes=zscore_binned_spikes,
                event_rates=event_rates,
                shuff_event_rates=shuff_event_rates,
                expression_intensity=expression_intensity,
                amplitudes=amplitudes,
                shuff_amplitudes=shuff_amplitudes,
            )

            analyzed_data.append(ind_data)

    # Group the data together
    grouped_data = Grouped_Kir_Activity_Data(analyzed_data)
    if save:
        print("Saving Grouped data")
        save_path = r"C:\Users\Jake\Desktop\Analyzed_data\grouped\Kir_Population_Data"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = "grouped_kir_population_activity_data"
        save_pickle(fname, grouped_data, save_path)

    return grouped_data


def chance_event_rates(event_rates, positive, permutations=1000):
    """Helper function to get chance event rates from randomly
        selected neurons"""
    # Initialize output
    shuff_event_rates = np.zeros((permutations, len(event_rates))) * np.nan
    # Set number of positive neurons
    positive_num = len(np.nonzero(positive)[0])
    # Set up sample indexes
    positive_idxs = np.nonzero(positive)[0]
    sample_idxs = np.nonzero([not x for x in positive])[0]
    # Iterate through each shuffle
    for i in range(permutations):
        idxs = np.random.choice(sample_idxs, positive_num)
        shuff_event_rates[i, positive_idxs] = event_rates[idxs]

    return shuff_event_rates


@dataclass
class Kir_Activity_Data:
    """Dataclass for storing the data of a single FOV or kir population data"""

    mouse_id: str
    fov: str
    sampling_rate: int
    kir_ids: list
    dFoF: np.ndarray
    processed_dFoF: np.ndarray
    activity: np.ndarray
    estimated_spikes: np.ndarray
    binned_spikes: np.ndarray
    zscore_dFoF: np.ndarray
    zscore_processed_dFoF: np.ndarray
    zscore_spikes: np.ndarray
    zscore_binned_spikes: np.ndarray
    event_rates: np.ndarray
    shuff_event_rates: np.ndarray
    expression_intensity: np.ndarray
    amplitudes: np.ndarray
    shuff_amplitudes: np.ndarray


class Grouped_Kir_Activity_Data:
    """Class to group individual Kir_Activity_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class"""
        # Concatenate all the data together
        self.concatenate_data(data_list)

    def concatenate_data(self, data_list):
        """Method to concatenate all the attributes together"""
        attributes = list(data_list[0].__dict__.keys())
        # Iterate through each dataclass
        for i, data in enumerate(data_list):
            for attribute in attributes:
                if (
                    attribute == "mouse_id"
                    or attribute == "fov"
                    or attribute == "sampling_rate"
                ):
                    var = getattr(data, attribute)
                    variable = [var for x in range(data.dFoF.shape[1])]
                else:
                    variable = getattr(data, attribute)
                # Initialize the attribute if first dataset
                if i == 0:
                    setattr(self, attribute, variable)
                # Concatenate the attributes together for subsequent datasets
                else:
                    old_var = getattr(self, attribute)
                    if type(variable) == np.ndarray:
                        if len(variable.shape) == 1:
                            new_var = np.concatenate((old_var, variable))
                        elif len(variable.shape) == 2:
                            ## Check array lengths
                            max_len = np.max([x.shape[0] for x in [old_var, variable]])
                            if max_len == old_var.shape[0]:
                                variable = d_utils.pad_array_to_length(
                                    variable, max_len, axis=0, value=np.nan
                                )
                            else:
                                old_var = d_utils.pad_array_to_length(
                                    old_var, max_len, axis=0, value=np.nan
                                )
                            new_var = np.hstack((old_var, variable))
                    elif type(variable) == list:
                        new_var = old_var + variable
                    # Set the attribute with new values
                    setattr(self, attribute, new_var)

