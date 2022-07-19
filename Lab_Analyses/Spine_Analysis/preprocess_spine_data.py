"""Module to handle initial processing of spine activity data sets"""

import os
import re

import numpy as np
from Lab_Analyses.Behavior.align_lever_behavior import align_lever_behavior
from Lab_Analyses.Utilities.movement_responsiveness import movement_responsiveness
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle
from yaml import load


def preprocess_dual_spine_data(
    mouse_id,
    channels={"GluSnFr": "GreenCh", "Calcium": "RedCh"},
    save=False,
    structural=False,
):
    """Function to handle the initial processing of dual color spine 
        activity datasets. Specifically designed to handle GluSnFR and 
        calcium activity
        
        INPUT PARAMETERS
            mouse_id - str specifying what the mouse's id is
            
            channels - tuple of strings for the different types of activity
                    to be co-processed. Will use to search for the relevant
                    files
            
            save - boolean specifying if the data is to be saved

            structural - boolean specifying if there is seperate structural data
                        to include
    """
    if len(channels) != 2:
        return "Need to have at least two channels specified"

    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    mouse_path = os.path.join(initial_path, mouse_id)
    imaging_path = os.path.join(mouse_path, "imaging")
    behavior_path = os.path.join(mouse_path, "behavior")

    # Get the number of FOVs imaged for this mouse
    FOVs = next(os.walk(imaging_path))[1]

    # Preprocess each FOV seperately
    for FOV in FOVs:
        FOV_path = os.path.join(imaging_path, FOV)
        # Get the different imaging session periods
        periods = next(os.walk(FOV_path))[1]
        # Reorder the periods if Early, Middle, Late
        if "Early" and "Late" and "Middle" in periods:
            periods = ["Early", "Middle", "Late"]
        # Preprocess each imaging period
        period_data = {}
        for period in periods:
            period_path = os.path.join(FOV_path, period)
            fnames = next(os.walk(period_path))[2]
            fnames = [x for x in fnames if "imaging_data" in x]
            # Get the files
            GluSnFr_fname = os.path.join(
                period_path,
                [
                    x
                    for x in fnames
                    if channels["GluSnFr"] in x and "structural" not in x
                ][0],
            )
            Calcium_fname = os.path.join(
                period_path, [x for x in fnames if channels["Calcium"] in x][0]
            )
            GluSnFr_data = load_pickle([GluSnFr_fname])[0]
            Calcium_data = load_pickle([Calcium_fname])[0]

            if structural:
                structural_fname = os.path.join(
                    period_path, [x for x in fnames if "structural" in x][0]
                )
                structural_data = load_pickle([structural_fname])[0]
            else:
                structural_data = None

            # Get the matching behavioral data
            day = re.search("[0-9]{6}", os.path.basename(GluSnFr_fname)).group()
            matched_b_path = os.path.join(behavior_path, day)
            b_fnames = next(os.walk(matched_b_path))[2]
            b_fname = [x for x in b_fnames if "processed_lever_data" in x][0]
            b_fname = os.path.join(matched_b_path, b_fname)
            behavior_data = load_pickle([b_fname])[0]
            # rewrite the session name (which is by default the date)
            behavior_data.sess_name = period

            # Align the data
            aligned_behavior, aligned_GluSnFr = align_lever_behavior(
                behavior_data, GluSnFr_data, save="both"
            )
            _, aligned_Calcium = align_lever_behavior(
                behavior_data, Calcium_data, save="imaging"
            )

