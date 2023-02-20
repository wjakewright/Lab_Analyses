import numpy as np
from sklearn import preprocessing

from Lab_Analyses.Utilities.event_detection import event_detection


def preprocess_dual_plane_data(data):
    """Function to preprocess the dual_plane data. 
        
        INPUT PARAMETERS
            data - dual_soma_dend_data dataclass. Output from organize dual_plane_data
    """

    # Pull relevant data out
    sampling_rate = data.imaging_parameters["Sampling Rate"]

    ## Check somatic and dendritic recordings are the same length
    dl = len(data.dend_processed_dFoF[:, 0])
    sl = len(data.soma_processed_dFoF)
    if dl > sl:
        dendrite_dFoF = data.dend_processed_dFoF[:sl, 0]
        soma_dFoF = data.soma_processed_dFoF
    elif dl < sl:
        dendrite_dFoF = data.dend_processed_dFoF
        soma_dFoF = data.soma_processed_dFoF[:dl]
    else:
        dendrite_dFoF = data.dend_processed_dFoF
        soma_dFoF = data.soma_processed_dFoF

    # Duplicate somatic dFoF to pair with each dendrite
    somatic_dFoF = np.zeros(dendrite_dFoF.shape)
    for i in range(dendrite_dFoF.shape[1]):
        somatic_dFoF[:, i] = soma_dFoF.flatten()

    # Generate normalized traces
    scalar = preprocessing.MinMaxScaler()
    dendrite_dFoF_norm = scalar.fit_transform(dendrite_dFoF)
    somatic_dFoF_norm = scalar.fit_transform(somatic_dFoF)

    # Perform event detection
    dendrite_activity, _, _ = event_detection(
        dendrite_dFoF,
        threshold=2,
        lower_threshold=0,
        lower_limit=None,
        sampling_rate=sampling_rate,
    )
    somatic_activity, _, _ = event_detection(
        somatic_dFoF,
        threshold=2,
        lower_threshold=0,
        lower_limit=0,
        sampling_rate=sampling_rate,
    )

    # Put relevant data into an output
    output_dict = {
        "dendrite_dFoF": dendrite_dFoF,
        "dendrite_dFoF_norm": dendrite_dFoF_norm,
        "dendrite_activity": dendrite_activity,
        "somatic_dFoF": somatic_dFoF,
        "somatic_dFoF_norm": somatic_dFoF_norm,
        "somatic_activity": somatic_activity,
        "sampling_rate": sampling_rate,
    }

    return output_dict
