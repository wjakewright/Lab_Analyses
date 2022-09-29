import os

import numpy as np
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)


class Coactivity_Plasticity:
    """Class to handle the analysis of spine plasticity on coactivity datasets"""

    def __init__(self, data, threshold, exclude):
        """Initialize the class
        
            INPUT PARAMETERS
                data - Spine_Coactivity_Data object, or a list of Spine_Coactivity_
                        Data Objects
                
                threshold - float specifying the threshold to consider for plasticity spines
                
                exclude - str specifying spine types to exclude from analysis (e.g., shaft)
        """

        # Check to see if one dataset was input and set up data
        if type(data) == list:
            self.dataset = data[0]
            self.followup_flags = data[1].spine_flags
            self.followup_volumes = data[1].corrected_spine_volume
        elif isinstance(data, object):
            if data.followup_volumes is not None:
                self.dataset = data
                self.followup_flags = data.followup_flags
                self.followup_volumes = data.followup_volumes
            else:
                raise Exception("Data must have followup data containing spine volumes")

        self.day = self.dataset.day
        self.threshold = threshold
        self.exclude = exclude
        self.parameters = self.dataset.parameters

        # Analyze the data
        self.analyze_plasticity()

    def analyze_plasticity(self):
        """Method to calculate spine volume change and classify plasticity"""

        volume_data = [self.dataset.spine_volumes, self.followup_volumes]
        flag_list = [self.dataset.spine_flags, self.followup_flags]

        relative_volumes, spine_idxs = calculate_volume_change(
            volume_data, flag_list, days=None, exclude=self.exclude
        )
        relative_volumes = list(relative_volumes.values())[0]
        enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
            relative_volumes, self.threshold
        )

        # Store volume change and plasticity classification
        self.relative_volumes = relative_volumes
        self.enlarged_spines = enlarged_spines
        self.shrunken_spines = shrunken_spines
        self.stable_spines = stable_spines

        # Refine coactivity variables for only stable spines and store them
        attributes = list(self.dataset.__dict__.keys())
        ## Go through each attribute
        for attribute in attributes:
            if (
                attribute == "day"
                or attribute == "mouse_id"
                or attribute == "parameters"
                or attribute == "learned_movement"
            ):
                continue
            # Refine the attributes data
            variable = getattr(self.dataset, attribute)
            if type(variable) == np.array:
                if len(variable.shape) == 1:
                    new_variable = variable[spine_idxs]
                elif len(variable.shape) == 2:
                    new_variable = variable[:, spine_idxs]
            elif type(variable) == list:
                new_variable = [variable[i] for i in spine_idxs]
            else:
                raise Exception(f"{variable} is incorrect datatype !!!")
            # Store the attribute
            setattr(self, attribute, new_variable)
