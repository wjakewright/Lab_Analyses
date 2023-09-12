import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from Lab_Analyses.Utilities.deconvolve_calcium import oasis
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_kir_population_data(
    mouse_id, channels={"Functional": "GreenCh", "Expression": "RedCh"}, save=False,
):
    """Function to handle the initial processing and organization of the kir population
        data
        
        INPUT PARAMETERS
            mouse_id - str specifying what the mouse's id is 

            channels - dictionary specifying what the functional and expression channels are

            redetection - boolean specifying whether to reperform event detection

            deconvolution - boolean specifying whether to estimate spikes 

            save - boolean specifying whether the save the output data
        
        """
    print(
        f"--------------------------------------------------\nProcessing Mouse {mouse_id}"
    )
    # Check if two channels were input
    if len(channels) != 2:
        return "Need to have at least two channels specified"
    # Set up the paths to load the data from
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    mouse_path = os.path.join(initial_path, mouse_id)
    imaging_path = os.path.join(mouse_path, "imaging")

    # Identify the FOVs
    FOVs = next(os.walk(imaging_path))[1]
    FOVs = [x for x in FOVs if "FOV" in x]

    # Process each FOV seperately
    for FOV in FOVs:
        print(f"- Preprocessing {FOV}")
        FOV_path = os.path.join(imaging_path, FOV)
        fnames = next(os.walk(FOV_path))[2]
        fnames = [x for x in fnames if "imaging_data" in x]
        # Get the specific files
        functional_fname = os.path.join(
            FOV_path, [x for x in fnames if channels["Functional"] in x][0]
        )
        expression_fname = os.path.join(
            FOV_path, [x for x in fnames if channels["Expression"] in x][0]
        )
        functional_data = load_pickle([functional_fname])[0]
        expression_data = load_pickle([expression_fname])[0]

        # Start pulling the relevant data
        roi_flags = functional_data.ROI_flags["Soma"]
        imaging_parameters = functional_data.parameters
        expression_fluorescence = expression_data.processed_fluorescence["Soma"]
        fluorescence = functional_data.processed_fluorescence["Soma"]
        dFoF = functional_data.dFoF["Soma"]
        processed_dFoF = functional_data.processed_dFoF["Soma"]
        # Perform event detection
        activity, floored, _ = event_detection(
            processed_dFoF,
            threshold=3,
            lower_threshold=1,
            lower_limit=0,
            sampling_rate=imaging_parameters["Sampling Rate"],
            filt_poly=1,
            sec_smooth=0.5,
        )
        # Deconvolve calcium
        estimated_spikes = oasis(
            fluo=fluorescence,
            batch_size=500,
            tau=0.7,
            sampling_rate=imaging_parameters["Sampling Rate"],
        )
        binned_spikes = uniform_filter1d(estimated_spikes, 6, axis=0)
        # Estimate expression intensity
        kir_positive = []
        expression_intensity = np.zeros(expression_fluorescence.shape[1]) * np.nan
        for i in range(expression_fluorescence.shape[1]):
            if "High Confidence" not in roi_flags[i]:
                kir_positive.append(False)
                continue
            expression_intensity[i] = np.nanmean(expression_fluorescence[:, i])
            kir_positive.append(True)

        # Store data
        population_data = Kir_Population_Data(
            mouse_id=mouse_id,
            fov=FOV,
            roi_flags=roi_flags,
            kir_positive=kir_positive,
            imaging_parameters=imaging_parameters,
            fluorescence=fluorescence,
            dFoF=dFoF,
            processed_dFoF=processed_dFoF,
            activity=activity,
            floored=floored,
            estimated_spikes=estimated_spikes,
            binned_spikes=binned_spikes,
            expression_intensity=expression_intensity,
        )

        # Save section
        if save:
            save_path = os.path.join(mouse_path, "kir_population_data")
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            fname = f"{mouse_id}_{FOV}_kir_population_data"
            save_pickle(fname, population_data, save_path)


@dataclass
class Kir_Population_Data:
    """Dataclass for containing all the relevant activity data for a single
        kir population dataset"""

    mouse_id: str
    fov: str
    roi_flags: list
    kir_positive: list
    imaging_parameters: dict
    fluorescence: np.ndarray
    dFoF: np.ndarray
    processed_dFoF: np.ndarray
    activity: np.ndarray
    floored: np.ndarray
    estimated_spikes: np.ndarray
    binned_spikes: np.ndarray
    expression_intensity: np.ndarray
