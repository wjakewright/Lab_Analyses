from dataclasses import dataclass

import numpy as np


@dataclass
class Dual_Soma_Apical_Basal_Data:
    """Dataclass to contain all the relevant activity data for simultaneously imaged
    somatic and dendritic datasets
    """

    mouse_id: str
    FOV: str
    apical_ids: list
    apical_dFoF: np.ndarray
    apical_processed_dFoF: np.ndarray
    apical_activity: np.ndarray
    apical_floored: np.ndarray
    basal_ids: list
    basal_dFoF: np.ndarray
    basal_processed_dFoF: np.ndarray
    basal_activity: np.ndarray
    basal_floored: np.ndarray
    soma_dFoF: np.ndarray
    soma_processed_dFoF: np.ndarray
    soma_activity: np.ndarray
    soma_floored: np.ndarray
    imaging_parameters: dict
