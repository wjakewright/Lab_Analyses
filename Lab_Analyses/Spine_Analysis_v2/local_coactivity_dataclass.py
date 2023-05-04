import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class Local_Coactivity_Data:
    """Dataclass to contain the local coactivity data from individual sessions
    """

    # Session information
    mouse_id: str
    FOV: str
    session: str
    parameters: dict
    # Spine and dendrite categorizatons
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
    coactive_spines: np.ndarray
    coactive_norm_spines: np.ndarray
    # Local coactivity rates
    distance_coactivity_rate: np.ndarray
    distance_coactivity_rate_norm: np.ndarray
    avg_local_coactivity_rate: np.ndarray
    avg_local_coactivity_rate_norm: np.ndarray
    shuff_local_coactivity_rate: np.ndarray
    shuff_local_coactivity_rate_norm: np.ndarray
    real_vs_shuff_coactivity_diff: np.ndarray
    real_vs_shuff_coactivity_diff_norm: np.ndarray
    # Nearby spine properties
    avg_nearby_spine_rate: np.ndarray
    shuff_nearby_spine_rate: np.ndarray
    spine_activity_rate_distribution: np.ndarray
    avg_nearby_coactivity_rate: np.ndarray
    shuff_nearby_coactivity_rate: np.ndarray
    local_coactivity_rate_distribution: np.ndarray
    avg_nearby_coactivity_rate_norm: np.ndarray
    shuff_nearby_coactivity_rate_norm: np.ndarray
    local_coactivity_rate_norm_distrubution: np.ndarray
    spine_density_distribution: np.ndarray
    MRS_density_distribution: np.ndarray
    avg_local_MRS_density: np.ndarray
    shuff_local_MRS_density: np.ndarray
    rMRS_density_distribution: np.ndarray
    avg_local_rMRS_density: np.ndarray
    shuff_local_rMRS_density: np.ndarray
    avg_nearby_spine_volume: np.ndarray
    shuff_nearby_spine_volume: np.ndarray
    nearby_spine_volume_distribution: np.ndarray
    local_nn_enlarged: np.ndarray
    shuff_nn_enlarged: np.ndarray
    enlarged_spine_distribution: np.ndarray
    local_nn_shrunken: np.ndarray
    shuff_nn_shrunken: np.ndarray
    shrunken_spine_distribution: np.ndarray
    # Coactive and Noncoactive spine events
    spine_coactive_event_num: np.ndarray
    spine_coactive_traces: list
    spine_noncoactive_traces: list
    spine_coactive_calcium_traces: list
    spine_noncoactive_calcium_traces: list
    spine_coactive_amplitude: np.ndarray
    spine_noncoactive_amplitude: np.ndarray
    spine_coactive_calcium_amplitude: np.ndarray
    spine_noncoactive_calcium_amplitude: np.ndarray
    fraction_spine_coactive: np.ndarray
    fraction_coactivity_participation: np.ndarray
    # Nearby spine activity
    coactive_spine_num: np.ndarray
    nearby_coactive_amplitude: np.ndarray
    nearby_coactive_calcium_amplitude: np.ndarray
    nearby_spine_onset: np.ndarray
    nearby_spine_onset_jitter: np.ndarray
    nearby_coactive_traces: list
    nearby_coactive_calcium_traces: list
    # Movement encoding
    learned_movement_pattern: np.ndarray
    nearby_movement_correlation: np.ndarray
    nearby_movement_stereotypy: np.ndarray
    nearby_movement_reliability: np.ndarray
    nearby_movement_specificity: np.ndarray
    nearby_LMP_reliability: np.ndarray
    nearby_LMP_specificity: np.ndarray
    nearby_rwd_movement_correlation: np.ndarray
    nearby_rwd_movement_stereotypy: np.ndarray
    nearby_rwd_movement_reliablity: np.ndarray
    nearby_rwd_movement_specificity: np.ndarray
    coactive_movements: list
    coactive_movement_correlation: np.ndarray
    coactive_movement_stereotypy: np.ndarray
    coactive_movement_reliability: np.ndarray
    coactive_movement_specificity: np.ndarray
    coactive_LMP_reliability: np.ndarray
    coactive_LMP_specificity: np.ndarray
    coactive_rwd_movements: list
    coactive_rwd_movement_correlation: np.ndarray
    coactive_rwd_movement_stereotypy: np.ndarray
    coactive_rwd_movement_reliability: np.ndarray
    coactive_rwd_movement_specificity: np.ndarray
    coactive_fraction_rwd_mvmts: np.ndarray
    # Local dendritic calcium signals
    coactive_local_dend_traces: list
    coactive_local_dend_amplitude: np.ndarray
    noncoactive_local_dend_traces: list
    noncoactive_local_dend_amplitude: np.ndarray
    nearby_local_dend_traces: list
    nearby_local_dend_amplitude: np.ndarray

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
        if self.parameters["partners"]:
            pname = f"_{self.parameters['partners']}"
        else:
            pname = ""
        if self.parameters["movement period"]:
            mname = f"{self.parameters['movement period']}"
        else:
            mname = "session"

        fname = f"{self.mouse_id}_{self.session}_{aname}{norm}_{mname}{pname}_local_coactivity_data"
        save_pickle(fname, self, save_path)


class Grouped_Local_Coactivity_Data:
    """Class to group individual Local_Coactivity_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class
            
            INPUT PARAMETERS
                data_list - list of Local_Coactivity_Data dataclasses
        """
        # Initialize some of the initial attributes
        self.session = data_list[0].session
        self.parameters = data_list[0].parameters

        # Concatenate all the other attributes
        self.concatenate_data(self, data_list)

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
        if self.parameters["partners"]:
            pname = f"_{self.parameters['partners']}"
        else:
            pname = ""
        if self.parameters["movement period"]:
            mname = f"{self.parameters['movement period']}"
        else:
            mname = "session"

        fname = f"{self.session}_{self.parameters['FOV type']}_{aname}{norm}_{mname}{pname}_grouped_local_coactivity_data"
        save_pickle(fname, self, save_path)
