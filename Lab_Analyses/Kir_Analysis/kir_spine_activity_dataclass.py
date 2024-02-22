import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class Kir_Spine_Activity_Data:
    """Dataclass to contain activity data for an individual session"""

    # Session information
    mouse_id: str
    FOV: str
    session: str
    parameters: dict
    # Spine categorizations
    spine_flags: list
    followup_flags: list
    spine_volumes: np.ndarray
    followup_volumes: np.ndarray
    dendrite_number: np.ndarray
    movement_spines: np.ndarray
    nonmovement_spines: np.ndarray
    rwd_movement_spines: np.ndarray
    nonrwd_movement_spines: np.ndarray
    # Activity rate
    spine_activity_rate: np.ndarray
    # Movement related activity variables
    spine_movement_traces: list
    spine_movement_amplitude: np.ndarray
    spine_movement_onset: np.ndarray
    spine_nonrwd_movement_traces: list
    spine_nonrwd_movement_amplitude: np.ndarray
    spine_nonrwd_movement_onset: np.ndarray
    spine_rwd_movement_traces: list
    spine_rwd_movement_amplitude: np.ndarray
    spine_rwd_movement_onset: np.ndarray
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
    spine_fraction_rwd_mvmts: np.ndarray

    def save(self):
        """Method to save the data"""
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        save_path = os.path.join(
            initial_path,
            self.mouse_id,
            "kir_data",
            self.FOV,
            self.session,
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if self.parameters["zscore"]:
            aname = "zscore"
        else:
            aname = "dFoF"

        fname = f"{self.mouse_id}_{self.session}_{aname}_kir_activity_data"
        save_pickle(fname, self, save_path)


class Grouped_Kir_Spine_Activity_Data:
    """Class to group individual Kir_Spine_Activity_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class

        INPUT PARAMETERS
            data_list - list of Kir_Spine_Activity_Data dataclasses

        """
        # Initialize some of the initial attributes
        self.session = data_list[0].session
        self.parameters = data_list[0].parameters

        # Concatenate all the other attributes
        self.concatenate_data(data_list)

    def concatenate_data(self, data_list):
        """Method to concatenate all the attributes from differet dataclass together
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
                    variable = [var for x in range(len(data.spine_volumes))]
                else:
                    variable = getattr(data, attribute)

                # Initialize the attribute if first dataset
                if i == 0:
                    setattr(self, attribute, variable)
                # Concatenate the attributes together for subsequent datasets
                else:
                    old_var = getattr(self, attribute)
                    ## This data should only contain 1d arrays
                    if type(variable) == np.ndarray:
                        new_var = np.concatenate((old_var, variable))
                    ## Or lists
                    if type(variable) == list:
                        new_var = old_var + variable
                    # Set the attribute with the new value
                    setattr(self, attribute, new_var)

    def save(self):
        """Method to save the grouped dataclass"""
        save_path = (
            r"C:\Users\Jake\Desktop\Analyzed_data\Kir_Spine_Imaging\Activity_Data"
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Prepare the save name
        if self.parameters["zscore"]:
            aname = "zscore"
        else:
            aname = "dFoF"

        fname = f"{self.session}_{self.parameters['FOV type']}_{aname}_grouped_kir_activity_data"
        save_pickle(fname, self, save_path)
