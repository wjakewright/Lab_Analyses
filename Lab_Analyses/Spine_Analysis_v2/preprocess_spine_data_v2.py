import os
import re
from copy import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Lab_Analyses.Behavior.align_lever_behavior import align_lever_behavior
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.data_utilities import join_dictionaries, pad_array_to_length
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.get_dFoF import resmooth_dFoF
from Lab_Analyses.Utilities.movement_responsiveness_v2 import movement_responsiveness
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_dual_spine_data(
    mouse_id,
    channels={"GluSnFr": "GreenCh", "Calcium": "RedCh"},
    fov_type="apical",
    redetection=False,
    resmooth=False,
    reprocess=True,
    save=False,
    followup=True,
):
    """Function to handle the initial processing and organization of dual color spine
    activity datasets.Specifically designed to handel GluSnFr and calcium activity

    INPUT PARAMETERS
        mouse_id - str specifying what the mouse's id is

        channels - dictionary of strings for the different types of activity to
                   be processed paired with keywords that will be used to search
                   and load those files

        fov_type - str specifying whether to process apical or basal FOVs

        redetection - boolean specifying whether to redo the event detection

        resmooth - boolean specifying whether to redo the dFoF smoothing

        reprocess - boolean specifying whether to reprocess or try to load the
                    data

        save - boolean specifying if the data is to be saved or not

        followup - boolean specifying if there is a dedicated followup structural
                    file to be included in the data
    """
