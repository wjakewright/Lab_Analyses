import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class Spine_Activity_Data:
    """Data class to contain spine-centric activity data for an individual
        session
    """

    # Session information
    mouse_id: str
    FOV: str
    session: str
    parameters: dict
    # Spine and Dendrite categorizations
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
    # Activity rates
    spine_activity_rate: np.ndarray
    dendrite_activity_rate: np.ndarray
    # Movement related activity variables
    spine_movement_traces: list
    spine_movement_calcium_traces: list
    spine_movement_amplitude: np.ndarray
    spine_movement_calcium_amplitude: np.ndarray
    spine_movement_onset: np.ndarray
    spine_movement_calcium_onset: np.ndarray
    dendrite_movement_traces: list
    dendrite_movement_amplitude: np.ndarray
    dendrite_movement_onset: np.ndarray
    # Rewarded movement related activity variables
    spine_rwd_movement_traces: list
    spine_rwd_movement_calcium_traces: list
    spine_rwd_movement_amplitude: np.ndarray
    spine_rwd_movement_calcium_amplitude: np.ndarray
    spine_rwd_movement_onset: np.ndarray
    spine_rwd_movement_calcium_onset: np.ndarray
    dendrite_rwd_movement_traces: list
    dendrite_rwd_movement_amplitude: np.ndarray
    dendrite_rwd_movement_onset: np.ndarray
    # Spine movement encoding
    learned_movement_pattern: list
    spine_movements: list
    spine_movement_correlation: np.ndarray
    spine_movement_stereotypy: np.ndarray
    spine_movement_reliability: np.ndarray
    spine_movement_specificity: np.ndarray
    spine_rwd_movements: list
    spine_rwd_movement_correlation: np.ndarray
    spine_rwd_movement_stereotypy: np.ndarray
    spine_rwd_movement_reliability: np.ndarray
    spine_rwd_movement_specificity: np.ndarray
    spine_LMP_reliability: np.ndarray
    spine_LMP_specificity: np.ndarray
    # Dendrite movement encoding
    dendrite_movements: list
    dendrite_movement_correlation: np.ndarray
    dendrite_movement_stereotypy: np.ndarray
    dendrite_movement_reliablility: np.ndarray
    dendrite_movement_specificity: np.ndarray
    dendrite_rwd_movements: list
    dendrite_rwd_movement_correlation: np.ndarray
    dendrite_rwd_movement_stereotypy: np.ndarray
    dendrite_rwd_movement_reliability: np.ndarray
    dendrite_rwd_movement_specificity: np.ndarray
    dendrite_LMP_reliability: np.ndarray
    dendrite_LMP_specificity: np.ndarray

    def save(self):
        """method to save the dataclass"""
        initial_path = r"C:\Users\Desktop\Analyzed_data\individual"
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

        fname = f"{self.mouse_id}_{self.session}_{aname}{norm}_spine_activity_data"
        save_pickle(fname, self, save_path)


class Grouped_Spine_Activity_Data:
    """Class to group individual Spine_Activity_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class
            
            INPUT PARAMETERS
                data_list - list of Spine_Activity_Data dataclasses
        """
        # Initalize some of the initial attributes
        self.session = data_list[0].session
        self.parameters = data_list[0].parameters

        # Concatenate all the other attributes
        self.concatenate_data(data_list)

    def contanetnate_data(self, data_list):
        """Method to concatenate all the attributes from the different dataclass together
            and store as attributes in the current class
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
                    variable = [var for x in range(len(data.spine_volume))]
                else:
                    variable = getattr(data, attribute)

                # Initialize the attribute if first dataset
                if i == 0:
                    setattr(self, attribute, variable)
                # Concatenate the attributes together for subsequent datasets
                else:
                    old_var = getattr(self, attribute)
                    ## This data should contain only 1d arrays
                    if type(variable) == np.ndarray:
                        new_var = np.concatenate((old_var, variable))
                    ## Or lists
                    if type(variable) == list:
                        new_var = old_var + variable  ## Only dealing with two lists
                    # Set the attribute with the new value
                    setattr(self, attribute, new_var)

    def save(self):
        """Method to save the grouped dataclass"""
        save_path = (
            r"C:\Users\Desktop\Analyzed_data\grouped\Dual_Spine_Imaging\Coactivity_Data"
        )
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

        fname = f"{self.session}_{self.parameters['FOV type']}_{aname}{norm}_grouped_spine_activity_data"
        save_pickle(fname, self, save_path)
