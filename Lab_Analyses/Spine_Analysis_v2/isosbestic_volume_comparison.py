import os

import numpy as np

from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def isosbestic_volume_comparison(mouse_list, save_fig=False, save_path=None):
    """Function to compare volume estimation between 925nm and 810nm
        isosbestic imaging wavelengths

        INPUT PARAMETERS
            mouse_list - list of strings specifying the mice to analyze

            save_fig - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figure    
        
    """
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    # Set up volume variables
    functional_volumes = []
    isosbestic_volumes = []

    # Iterate through each mouse
    for mouse in mouse_list:
        imaging_path = os.path.join(initial_path, mouse, "imaging", "isosbestic")
        # Get individual FOVs
        FOVs = next(os.walk(imaging_path))[1]
        FOVs = [x for x in FOVs if "FOV" in x]

        # Iterate through each FOV
        for FOV in FOVs:
            FOV_path = os.path.join(imaging_path, FOV)
            fnames = next(os.walk(FOV_path))[2]
            func_fname = [x for x in fnames if "functional_imaging_data" in x][0]
            isos_fname = [x for x in fnames if "isosbestic_imaging_data" in x][0]
            func_data = load_pickle([func_fname], path=FOV_path)[0]
            isos_data = load_pickle([isos_fname], path=FOV_path)[0]
            func_vol = np.array(func_data.corrected_spine_volume)
            isos_vol = np.array(isos_data.corrected_spine_volume)
            pix_to_um = func_data.parameters["Zoom"] / 2
            func_vol = (np.sqrt(func_vol) / pix_to_um) ** 2
            isos_vol = (np.sqrt(isos_vol) / pix_to_um) ** 2
            # func_vol[func_vol > 100] = np.nan
            # isos_vol[isos_vol > 100] = np.nan
            functional_volumes.append(func_vol)
            isosbestic_volumes.append(isos_vol)

    # Concatenate the arrays across FOVs
    functional_volumes = np.concatenate(functional_volumes)
    isosbestic_volumes = np.concatenate(isosbestic_volumes)
    print(f"Number of spine: {len(functional_volumes)}")

    # Plot the data
    plot_scatter_correlation(
        isosbestic_volumes,
        functional_volumes,
        CI=95,
        title="Isosbestic Volume Comparison",
        xtitle="810nm Estimated Volume",
        ytitle="925nm Estimated Volume",
        figsize=(5, 5),
        xlim=(0, 8),
        ylim=(0, 8),
        marker_size=15,
        face_color="cmap",
        edge_color="black",
        edge_width=0.0,
        line_color="black",
        s_alpha=0.2,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=4,
        unity=True,
        ax=None,
        save=save_fig,
        save_path=save_path,
    )

