import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class Dendritic_Coactivity_Data:
    """Dataclass to contain the spine-dendrite coactivity data from individual sessions"""

    # Session information
    mouse_id: str
    FOV: str
    session: str
    parameters: dict
    # Spine and dendrite categories
    spine_flags: list
    followup_flags: list
    spine_volumes: np.ndarray
    followup_volumes: np.ndarray
    movement_spines: np.ndarray
    nonmovement_spines: np.ndarray
    rwd_movement_spines: np.ndarray
    nonrwd_movement_spines: np.ndarray
    movement_dendrites: np.ndarray
    nonmovement_dendrites: np.ndarray
    rwd_movement_dendrites: np.ndarray
    nonrwd_movement_dendrites: np.ndarray
    # Coactivity-related variables
    ## All coactive events
    all_dendrite_coactivity_rate: np.ndarray
    all_dendrite_coactivity_rate_norm: np.ndarray
    all_shuff_dendrite_coactivity_rate: np.ndarray
    all_shuff_dendrite_coactivity_rate_norm: np.ndarray
    all_fraction_dendrite_coactive: np.ndarray
    all_fraction_spine_coactive: np.ndarray
    all_spine_coactive_amplitude: np.ndarray
    all_spine_coactive_calcium_amplitude: np.ndarray
    all_dendrite_coactive_amplitude: np.ndarray
    all_relative_onsets: np.ndarray
    all_spine_coactive_traces: list
    all_spine_coactive_calcium_traces: list
    all_dendrite_coactive_traces: list
    ## Conj coactive events
    conj_dendrite_coactivity_rate: np.ndarray
    conj_dendrite_coactivity_rate_norm: np.ndarray
    conj_shuff_dendrite_coactivity_rate: np.ndarray
    conj_shuff_dendrite_coactivity_rate_norm: np.ndarray
    conj_fraction_dendrite_coactive: np.ndarray
    conj_fraction_spine_coactive: np.ndarray
    conj_spine_coactive_amplitude: np.ndarray
    conj_spine_coactive_calcium_amplitude: np.ndarray
    conj_dendrite_coactive_amplitude: np.ndarray
    conj_relative_onsets: np.ndarray
    conj_spine_coactive_traces: list
    conj_spine_coactive_calcium_traces: list
    conj_dendrite_coactive_traces: list
    ## Nonconj coactive events
    nonconj_dendrite_coactivity_rate: np.ndarray
    nonconj_dendrite_coactivity_rate_norm: np.ndarray
    nonconj_shuff_dendrite_coactivity_rate: np.ndarray
    nonconj_shuff_dendrite_coactivity_rate_norm: np.ndarray
    nonconj_fraction_dendrite_coactive: np.ndarray
    nonconj_fraction_spine_coactive: np.ndarray
    nonconj_spine_coactive_amplitude: np.ndarray
    nonconj_spine_coactive_calcium_amplitude: np.ndarray
    nonconj_dendrite_coactive_amplitude: np.ndarray
    nonconj_relative_onsets: np.ndarray
    nonconj_spine_coactive_traces: list
    nonconj_spine_coactive_calcium_traces: list
    nonconj_dendrite_coactive_traces: list
    fraction_conj_events: np.ndarray
    # Nearby spine coactivity properties
    conj_coactive_spine_num: np.ndarray
    conj_nearby_coactive_spine_amplitude: np.ndarray
    conj_nearby_coactive_spine_calcium: np.ndarray
    conj_nearby_spine_onset: np.ndarray
    conj_nearby_spine_onset_jitter: np.ndarray
    conj_nearby_coactive_spine_traces: list
    conj_nearby_coactive_spine_calcium_traces: list
    # Coactivity properties distribution
    avg_nearby_spine_coactivity_rate: np.ndarray
    shuff_nearby_spine_coactivity_rate: np.ndarray
    cocativity_rate_distribution: np.ndarray
    rel_nearby_spine_coactivity_rate: np.ndarray
    avg_nearby_spine_coactivity_rate_norm: np.ndarray
    shuff_nearby_spine_coactivity_rate_norm: np.ndarray
    coactivity_rate_norm_distribution: np.ndarray
    rel_nearby_spine_coactivity_rate_norm: np.ndarray
    avg_nearby_spine_conj_rate: np.ndarray
    shuff_nearby_spine_conj_rate: np.ndarray
    conj_coactivity_rate_distribution: np.ndarray
    rel_nearby_spine_conj_rate: np.ndarray
    avg_nearby_spine_conj_rate_norm: np.ndarray
    shuff_nearby_spine_conj_rate_norm: np.ndarray
    conj_coactivity_rate_norm_distrubtion: np.ndarray
    rel_nearby_spine_conj_rate_norm: np.ndarray
    avg_nearby_spine_fraction: np.ndarray
    shuff_nearby_spine_fraction: np.ndarray
    spine_fraction_coactive_distribution: np.ndarray
    rel_spine_fraction: np.ndarray
    avg_nearby_dendrite_fraction: np.ndarray
    shuff_nearby_dendrite_fraction: np.ndarray
    dendrite_fraction_coactive_distribution: np.ndarray
    rel_dendrite_fraction: np.ndarray
    conj_avg_nearby_spine_fraction: np.ndarray
    conj_shuff_nearby_spine_fraction: np.ndarray
    conj_spine_fraction_coactive_distribution: np.ndarray
    rel_conj_spine_fraction: np.ndarray
    conj_avg_nearby_dendrite_fraction: np.ndarray
    conj_shuff_nearby_dendrite_fraction: np.ndarray
    conj_dend_fraction_coactive_distribution: np.ndarray
    rel_conj_dendrite_fraction: np.ndarray
    avg_nearby_relative_onset: np.ndarray
    shuff_nearby_relative_onset: np.ndarray
    relative_onset_distribution: np.ndarray
    rel_nearby_relative_onset: np.ndarray
    conj_avg_nearby_relative_onset: np.ndarray
    conj_shuff_nearby_relative_onset: np.ndarray
    conj_relative_onset_distribution: np.ndarray
    rel_conj_nearby_relative_onset: np.ndarray
    # Noncoactive calcium measurements
    noncoactive_spine_calcium_amplitude: np.ndarray
    noncoactive_spine_calcium_traces: list
    conj_fraction_participating: np.ndarray
    nonparticipating_spine_calcium_amplitude: np.ndarray
    nonparticipating_spine_calcium_traces: list
    # Movement encoding
    learned_movement_pattern: np.ndarray
    all_coactive_movements: list
    all_coactive_movement_correlation: np.ndarray
    all_coactive_movement_stereotypy: np.ndarray
    all_coactive_movement_reliability: np.ndarray
    all_coactive_movement_specificity: np.ndarray
    all_coactive_LMP_reliability: np.ndarray
    all_coactive_LMP_specificity: np.ndarray
    conj_movements: list
    conj_movement_correlation: np.ndarray
    conj_movement_stereotypy: np.ndarray
    conj_movement_reliability: np.ndarray
    conj_movement_specificity: np.ndarray
    conj_LMP_reliability: np.ndarray
    conj_LMP_specificity: np.ndarray
    nonconj_movements: list
    nonconj_movement_correlation: np.ndarray
    nonconj_movement_stereotypy: np.ndarray
    nonconj_movement_reliability: np.ndarray
    nonconj_movement_specificity: np.ndarray
    nonconj_LMP_reliability: np.ndarray
    nonconj_LMP_specificity: np.ndarray

    def save(self):
        """method to save the dataclass"""
        initial_path = r"G:\Analyzed_data\individual"
        save_path = os.path.join(
            initial_path, self.mouse_id, "coactivity_data", self.FOV, self.session
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if self.parameters["zscore"]:
            aname = "zscore"
        else:
            aname = "dFoF"
        if self.parameters["Volume Norm"]:
            norm = "_norm"
        else:
            norm = ""
        if self.parameters["partners"]:
            pname = f"_{self.parameters['partners']}"
        else:
            pname = ""
        if self.parameters["movement period"]:
            mname = f"{self.parameters['movement period']}"
        else:
            mname = "session"
        if self.parameters["extended"]:
            ename = "_extended"
        else:
            ename = ""

        fname = f"{self.mouse_id}_{self.session}_{aname}{norm}_{mname}{pname}{ename}_dendritic_coactivity_data"
        save_pickle(fname, self, save_path)


class Grouped_Dendritic_Coactivity_Data:
    """Class to group individual Dendritic_Coactivity_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class

        INPUT PARAMETERS
            data_list - list of Dendritic_Coactivity_Data dataclasses
        """
        # Initialize some of the initial attributes
        self.session = data_list[0].session
        self.parameters = data_list[0].parameters

        # Concatenate all the other attributes togehter
        self.concatenate_data(data_list)

    def concatenate_data(self, data_list):
        """Method to concatenate all the attributes from the different dataclasses
        together and store the attributes in the current class
        """
        # Get a list of all the attributes
        attributes = list(data_list[0].__dict__.keys())
        # Iterate through each dataclass
        for i, data in enumerate(data_list):
            for attribute in attributes:
                if attribute == "session" or attribute == "parameters":
                    continue
                if attribute == "mouse_id" or attribute == "FOV":
                    var = getattr(data, attribute)
                    variable = [var for x in range(len(data.spine_volumes))]
                else:
                    variable = getattr(data, attribute)

                # Initialize the attribute if first dataset
                if i == 0:
                    setattr(self, attribute, variable)
                else:
                    old_var = getattr(self, attribute)
                    if type(variable) == np.ndarray:
                        if len(variable.shape) == 1:
                            new_var = np.concatenate((old_var, variable))
                        elif len(variable.shape) == 2:
                            new_var = np.hstack((old_var, variable))
                    elif type(variable) == list:
                        new_var = old_var + variable
                    # Set the attribute with the new values
                    setattr(self, attribute, new_var)

    def save(self):
        """Method to save the grouped dataclasses"""
        save_path = r"G:\Analyzed_data\grouped\Dual_Spine_Imaging\Coactivity_Data"

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Prepare the save name
        if self.parameters["zscore"]:
            aname = "zscore"
        else:
            aname = "dFoF"
        if self.parameters["Volume Norm"]:
            norm = "_norm"
        else:
            norm = ""
        if self.parameters["partners"]:
            pname = f"_{self.parameters['partners']}"
        else:
            pname = ""
        if self.parameters["movement period"]:
            mname = f"{self.parameters['movement period']}"
        else:
            mname = "session"
        if self.parameters["extended"]:
            ename = "_extended"
        else:
            ename = ""

        fname = f"{self.session}_{self.parameters['FOV type']}_{aname}{norm}_{mname}{pname}{ename}_grouped_dendritic_coactivity_data"
        save_pickle(fname, self, save_path)
