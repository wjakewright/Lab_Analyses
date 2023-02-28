from dataclasses import dataclass

import numpy as np


@dataclass
class Dual_Plane_Data:
    """Dataclass to hold the analyzed dual plane data"""

    dendrite_activity: np.ndarray
    somatic_activity: np.ndarray
    dendrite_dFoF: np.ndarray
    somatic_dFoF: np.ndarray
    dendrite_dFoF_norm: np.ndarray
    somatic_dFoF_norm: np.ndarray
    fraction_dendrite_active: np.ndarray
    fraction_somatic_active: np.ndarray
    dendrite_amplitudes: list
    somatic_amplitudes: list
    other_dendrite_amplitudes: list
    dendrite_decay: list
    somatic_decay: list
    dendrite_amplitudes_norm: list
    somatic_amplitudes_norm: list
    other_dendrite_amplitudes_norm: list
    coactive_dendrite_traces: list
    noncoactive_dendrite_traces: list
    coactive_somatic_traces: list
    noncoactive_somatic_traces: list
    coactive_dendrite_traces_norm: list
    noncoactive_dendrite_traces_norm: list
    coactive_somatic_traces_norm: list
    noncoactive_somatic_traces_norm: list
    sampling_rate: int
