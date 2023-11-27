import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_dual_plane_soma_dend_data(
    mouse_id,
    data_dir="two_plane",
    file_types={"Dendrite": "dendrite", "Soma": "soma"},
    save=False,
):
    """Function to organize simultaneously imaged dendrite and soma datasets.
        
        INPUT PARAMETERS 
            mouse_id - str specifying what the mouse's id is
            
            data_dir - str specifying the name of the directory where the data
                      files are located
                      
            file_types - dict containing keywords in file names for the Dendrite 
                        and Soma data files
            
            save - boolean specifying whetehr or not to save the data
    """
    # Set up the load path
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    load_path = os.path.join(initial_path, mouse_id, "imaging", data_dir)
    FOVs = next(os.walk(load_path))[1]

    # Process FOVs seperately
    FOV_data = {}
    for FOV in FOVs:
        # Get file names
        FOV_path = os.path.join(load_path, FOV)
        fnames = [file for file in os.listdir(FOV_path) if file.endswith(".pickle")]
        print(fnames)
        dend_fname = [x for x in fnames if file_types["Dendrite"] in x][0]
        soma_fname = [x for x in fnames if file_types["Soma"] in x][0]
        dend_file = os.path.join(FOV_path, dend_fname)
        soma_file = os.path.join(FOV_path, soma_fname)
        print(f"Dendrite: {dend_file}")
        print(f"Soma: {soma_file}")

        # load the data file
        dend_data = load_pickle([dend_file])[0]
        soma_data = load_pickle([soma_file])[0]

        # Group data into output
        dual_soma_dendrite_data = Dual_Soma_Dendrite_Data(
            mouse_id=mouse_id,
            dend_ids=dend_data.ROI_ids["Dendrite"],
            dend_dFoF=dend_data.dFoF["Dendrite"],
            dend_processed_dFoF=dend_data.processed_dFoF["Dendrite"],
            dend_activity=dend_data.activity_trace["Dendrite"],
            dend_floored=dend_data.floored_trace["Dendrite"],
            soma_dFoF=soma_data.dFoF["Soma"],
            soma_processed_dFoF=soma_data.processed_dFoF["Soma"],
            soma_activity=soma_data.activity_trace["Soma"],
            soma_floored=soma_data.floored_trace["Soma"],
            imaging_parameters=dend_data.parameters,
        )

        # Save section
        if save:
            # Set up the save path
            save_path = os.path.join(initial_path, mouse_id, "dual_plane", FOV)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # Make file name and save
            fname = f"{mouse_id}_{FOV}_dual_soma_dend_data"
            save_pickle(fname, dual_soma_dendrite_data, save_path)

        # Store FOV data
        FOV_data[FOV] = dual_soma_dendrite_data

    return FOV_data


@dataclass
class Dual_Soma_Dendrite_Data:
    """Dataclass to contain all the relevant activity data for simultaneously imaged
        somatic and dendritic datasets."""

    mouse_id: str
    dend_ids: list
    dend_dFoF: np.ndarray
    dend_processed_dFoF: np.ndarray
    dend_activity: np.ndarray
    dend_floored: np.ndarray
    soma_dFoF: np.ndarray
    soma_processed_dFoF: np.ndarray
    soma_activity: np.ndarray
    soma_floored: np.ndarray
    imaging_parameters: dict
