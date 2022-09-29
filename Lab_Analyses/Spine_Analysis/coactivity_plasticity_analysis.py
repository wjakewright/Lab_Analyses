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

