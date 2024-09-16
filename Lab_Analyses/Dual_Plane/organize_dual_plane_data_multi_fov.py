import os

import numpy as np

from Lab_Analyses.Dual_Plane.dual_soma_apical_basal_dataclass import (
    Dual_Soma_Apical_Basal_Data,
)
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_dual_plane_data(
    mouse_id,
    data_dir="two_plane",
    file_types={"Apical": "apical", "Basal": "basal", "Soma": "soma"},
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
                name="dual_soma_apical_basal_data",
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
        apical_fname = [x for x in fnames if file_types["Apical"] in x][0]
        basal_fname = [x for x in fnames if file_types["Basal"] in x][0]
        soma_fname = [x for x in fnames if file_types["Soma"] in x][0]
        apical_file = os.path.join(FOV_path, apical_fname)
        basal_file = os.path.join(FOV_path, basal_fname)
        soma_file = os.path.join(FOV_path, soma_fname)

        # Load the datafile
        apical_data = load_pickle([apical_file])[0]
        basal_data = load_pickle([basal_file])[0]
        soma_data = load_pickle([soma_file])[0]

        # Pull the relevant data to output
        dual_soma_apical_basal_data = Dual_Soma_Apical_Basal_Data(
            mouse_id=mouse_id,
            FOV=FOV,
            apical_ids=apical_data.ROI_ids["Dendrite"],
            apical_dFoF=apical_data.dFoF["Dendrite"],
            apical_processed_dFoF=apical_data.processed_dFoF["Dendrite"],
            apical_activity=apical_data.activity_trace["Dendrite"],
            apical_floored=apical_data.floored_trace["Dendrite"],
            basal_ids=basal_data.ROI_ids["Dendrite"],
            basal_dFoF=basal_data.dFoF["Dendrite"],
            basal_processed_dFoF=basal_data.processed_dFoF["Dendrite"],
            basal_activity=basal_data.activity_trace["Dendrite"],
            basal_floored=basal_data.floored_trace["Dendrite"],
            soma_dFoF=soma_data.dFoF["Soma"],
            soma_processed_dFoF=soma_data.processed_dFoF["Soma"],
            soma_activity=soma_data.activity_trace["Soma"],
            soma_floored=soma_data.floored_trace["Soma"],
            imaging_parameters=apical_data.parameters,
        )

        # Save section
        if save:
            save_path = os.path.join(initial_path, mouse_id, "dual_plane", FOV)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # Make file name and ave
            fname = f"{mouse_id}_{FOV}_dual_soma_apical_basal_data"
            save_pickle(fname, dual_soma_apical_basal_data, save_path)

        # Store FOV data
        FOV_data[FOV] = dual_soma_apical_basal_data

    return FOV_data
