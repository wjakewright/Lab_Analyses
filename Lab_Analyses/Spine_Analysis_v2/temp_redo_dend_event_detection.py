import os

import numpy as np

from Lab_Analyses.Spine_Analysis_v2.preprocess_spine_data import Dual_Channel_Spine_Data
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import load_spine_datasets
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def redo_dend_event_detection(mouse_list, fov_type, session):
    """ "redo event detection for dendrites"""

    for mouse in mouse_list:
        print(f" - Analyzing mouse {mouse}")
        datasets = load_spine_datasets(mouse, [session], fov_type)

        for FOV, dataset in datasets.items():

            data = dataset[session]
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            dend_calcium = data.dendrite_calcium_processed_dFoF
            dendrite_calcium_activity = (
                np.zeros(data.spine_GluSnFr_activity.shape) * np.nan
            )
            dendrite_calcium_floored = (
                np.zeros(data.spine_GluSnFr_floored.shape) * np.nan
            )

            for spine in range(dend_calcium.shape[1]):
                if np.isnan(dend_calcium[:, spine]).all():
                    continue
                da, df, _ = event_detection(
                    dend_calcium[:, spine].reshape(-1, 1),
                    threshold=2,
                    lower_threshold=0,
                    lower_limit=None,
                    sampling_rate=sampling_rate,
                    filt_poly=4,
                    sec_smooth=1,
                )
                dendrite_calcium_activity[:, spine] = da.reshape(1, -1)
                dendrite_calcium_floored[:, spine] = df.reshape(1, -1)

            data.dendrite_calcium_activity = dendrite_calcium_activity
            data.dendrite_calcium_floored = dendrite_calcium_floored
            save_path = os.path.join(
                r"G:\Analyzed_data\individual", mouse, "spine_data", FOV
            )
            fname = f"{data.mouse_id}_{FOV}_{session}_dual_spine_data"
            save_pickle(fname, data, save_path)
