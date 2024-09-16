import os

import numpy as np
from sklearn import preprocessing

from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def preprocess_dual_plane_data(data):
    """Function to preprocess the dual_plane data.

    INPUT PARAMETERS
        data - dual_soma_dend_data dataclass. Output from organize dual_plane_data
    """

    # Pull relevant data out
    sampling_rate = data.imaging_parameters["Sampling Rate"]

    ## Check somatic and dendritic recordings are the same length
    al = len(data.apical_processed_dFoF[:, 0])
    sl = len(data.soma_processed_dFoF)
    bl = len(data.basal_processed_dFoF[:, 0])
    min_len = np.min([al, sl, bl]) - 1
    apical_dFoF = data.apical_processed_dFoF[:min_len, :]
    basal_dFoF = data.basal_processed_dFoF[:min_len, :]
    soma_dFoF = data.soma_processed_dFoF[:min_len]

    # Duplicate somatic dFoF to pair with each dendrite
    a_somatic_dFoF = np.zeros(apical_dFoF.shape)
    b_somatic_dFoF = np.zeros(basal_dFoF.shape)
    for i in range(apical_dFoF.shape[1]):
        a_somatic_dFoF[:, i] = soma_dFoF.flatten()
    for i in range(basal_dFoF.shape[1]):
        b_somatic_dFoF[:, i] = soma_dFoF.flatten()

    # Generate normalized traces
    scalar = preprocessing.MinMaxScaler()
    apical_dFoF_norm = scalar.fit_transform(apical_dFoF)
    basal_dFoF_norm = scalar.fit_transform(basal_dFoF)
    a_somatic_dFoF_norm = scalar.fit_transform(a_somatic_dFoF)
    b_somatic_dFoF_norm = scalar.fit_transform(b_somatic_dFoF)

    # Perform event detection
    apical_activity, _, _ = event_detection(
        apical_dFoF,
        threshold=3,
        lower_threshold=0,
        lower_limit=None,
        sampling_rate=sampling_rate,
        filt_poly=2,
        sec_smooth=0.5,
    )
    basal_activity, _, _ = event_detection(
        basal_dFoF,
        threshold=3,
        lower_threshold=0,
        lower_limit=None,
        sampling_rate=sampling_rate,
        filt_poly=2,
        sec_smooth=0.5,
    )
    a_somatic_activity, _, _ = event_detection(
        a_somatic_dFoF,
        threshold=2,
        lower_threshold=1,
        lower_limit=None,
        sampling_rate=sampling_rate,
        filt_poly=1,
        sec_smooth=0.5,
    )
    b_somatic_activity, _, _ = event_detection(
        b_somatic_dFoF,
        threshold=2,
        lower_threshold=1,
        lower_limit=None,
        sampling_rate=sampling_rate,
        filt_poly=1,
        sec_smooth=0.5,
    )

    # Put relevant data into an output
    output_dict = {
        "apical_dFoF": apical_dFoF,
        "apical_dFoF_norm": apical_dFoF_norm,
        "apical_activity": apical_activity,
        "basal_dFoF": basal_dFoF,
        "basal_dFoF_norm": basal_dFoF_norm,
        "basal_activity": basal_activity,
        "a_somatic_dFoF": a_somatic_dFoF,
        "a_somatic_dFoF_norm": a_somatic_dFoF_norm,
        "a_somatic_activity": a_somatic_activity,
        "b_somatic_dFoF": b_somatic_dFoF,
        "b_somatic_dFoF_norm": b_somatic_dFoF_norm,
        "b_somatic_activity": b_somatic_activity,
        "sampling_rate": sampling_rate,
    }

    mouse = data.mouse_id
    FOV = data.FOV
    fname = f"{mouse}_{FOV}_processed_dual_plane_data"

    initial_path = r"G:\Repository_data\individual"
    save_path = os.path.join(initial_path, mouse, "dual_plane", FOV)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_pickle(fname, output_dict, save_path)

    return output_dict
