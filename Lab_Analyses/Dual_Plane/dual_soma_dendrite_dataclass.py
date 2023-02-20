from dataclasses import dataclass

import numpy as np


@dataclass
class Dual_Soma_Dendrite_Data:
    """Dataclass to contain all the relevant activity data for simultaneously imaged
        somatic and dendritic datasets
    """

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
