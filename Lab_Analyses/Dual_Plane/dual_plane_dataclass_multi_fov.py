from dataclasses import dataclass

import numpy as np


@dataclass
class Dual_Plane_Data_Multi:
    """Dataclass to hold the analyzed dual plane data"""

    apical_activity: np.ndarray
    basal_activity: np.ndarray
    a_somatic_activity: np.ndarray
    b_somatic_activity: np.ndarray
    apical_dFoF: np.ndarray
    basal_dFoF: np.ndarray
    a_somatic_dFoF: np.ndarray
    b_somatic_dFoF: np.ndarray
    apical_dFoF_norm: np.ndarray
    basal_dFoF_norm: np.ndarray
    a_somatic_dFoF_norm: np.ndarray
    b_somatic_dFoF_norm: np.ndarray
    apical_noise: np.ndarray
    basal_noise: np.ndarray
    a_somatic_noise: np.ndarray
    b_somatic_noise: np.ndarray
    fraction_apical_active: np.ndarray
    fraction_basal_active: np.ndarray
    fraction_a_somatic_active: np.ndarray
    fraction_b_somatic_active: np.ndarray
    apical_amplitudes: list
    basal_amplitudes: list
    a_somatic_amplitudes: list
    b_somatic_amplitudes: list
    other_apical_amplitudes: list
    other_basal_amplitudes: list
    other_apical_across_amplitudes: list
    other_basal_across_amplitudes: list
    apical_decay: list
    basal_decay: list
    a_somatic_decay: list
    b_somatic_decay: list
    apical_amplitudes_norm: list
    basal_amplitudes_norm: list
    a_somatic_amplitudes_norm: list
    b_somatic_amplitudes_norm: list
    other_apical_amplitudes_norm: list
    other_basal_amplitudes_norm: list
    other_apical_across_amplitudes_norm: list
    other_basal_across_amplitudes_norm: list
    coactive_apical_traces: list
    noncoactive_apical_traces: list
    coactive_basal_traces: list
    noncoactive_basal_traces: list
    coactive_a_somatic_traces: list
    noncoactive_a_somatic_traces: list
    coactive_b_somatic_traces: list
    noncoactive_b_somatic_traces: list
    coactive_apical_traces_norm: list
    noncoactive_apical_traces_norm: list
    coactive_basal_traces_norm: list
    noncoactive_basal_traces_norm: list
    coactive_a_somatic_traces_norm: list
    noncoactive_a_somatic_traces_norm: list
    coactive_b_somatic_traces_norm: list
    noncoactive_b_somatic_traces_norm: list
    sampling_rate: int
