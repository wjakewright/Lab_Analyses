import os

import numpy as np

from Lab_Analyses.Dual_Plane.dual_soma_dendrite_dataclass import Dual_Soma_Dendrite_Data
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_dual_plane_data(
    mouse_id,
    data_dir="two_plane",
    file_types={"Dendrite": "dendrite", "Soma": "soma"},
    reprocess=True,
    save=False,
):
    """Function to organize and preprocess simultaneously imaged dendrite
    and soma datasets

    INPUT PARAMETERS
        mouse_id - str specifying what the mouse's id is

        data_dir - str specifying the name of the directory where the data
                    files are located

        file_types - dict containing keywords in the file names for the Dendrie
                    and Soma data files

        reprocess - boolean specifying if you wish to reprocess the data or not

        save - boolean specifying whether or not to save the data
    """
    # set up the load path
    initial_path = r"G:\Analyzed_data\individual"
    load_path = os.path.join(initial_path, mouse_id, "imaging", data_dir)
    FOVs = next(os.walk(load_path))[1]

    # Preprocess FOVs seperately
    FOV_data = {}
    for FOV in FOVs:
        # check reprocessing
        if reprocess is False:
            print(f"Checking if {FOV} file exists....")
            exists = get_existing_files(
                path=os.path.join(initial_path, mouse_id, "dual_plane", FOV),
                name="dual_soma_dend_data",
                includes=True,
            )
            if exists is not None:
                print(f"Loading {FOV} file....")
                dual_soma_dendrite_data = load_pickle(
                    [exists],
                    path=os.path.join(initial_path, mouse_id, "dual_plane", FOV),
                )
                FOV_data[FOV] = dual_soma_dendrite_data
                continue
            else:
                print(f"{FOV} file not found. Processing....")

        # Get file names
        FOV_path = os.path.join(load_path, FOV)
        fnames = [file for file in os.listdir(FOV_path) if file.endswith(".pickle")]
        dend_fname = [x for x in fnames if file_types["Dendrite"] in x][0]
        soma_fname = [x for x in fnames if file_types["Soma"] in x][0]
        dend_file = os.path.join(FOV_path, dend_fname)
        soma_file = os.path.join(FOV_path, soma_fname)

        # Load the datafile
        dend_data = load_pickle([dend_file])[0]
        soma_data = load_pickle([soma_file])[0]

        # Pull the relevant data to output
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
            save_path = os.path.join(initial_path, mouse_id, "dual_plane", FOV)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # Make file name and ave
            fname = f"{mouse_id}_{FOV}_dual_soma_dend_data"
            save_pickle(fname, dual_soma_dendrite_data, save_path)

        # Store FOV data
        FOV_data[FOV] = dual_soma_dendrite_data

    return FOV_data
