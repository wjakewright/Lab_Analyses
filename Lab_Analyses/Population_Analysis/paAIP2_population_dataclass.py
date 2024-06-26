import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class paAIP2_Population_Data:
    """Dataclass to contain the paAIP2 population activity data for all sessions
    for an individual mouse"""

    # Mouse information
    mouse_id: str
    group: str
    sessions: list
    parameters: dict
    # Session related variables
    lever_active: list
    lever_force: list
    lever_active_rwd: list
    zscore_dFoF: list
    zscore_spikes: list
    zscore_smooth_spikes: list
    mvmt_cells_dFoF: list
    mvmt_cells_spikes: list
    # Activity related variables
    cell_activity_rate: dict
    fraction_MRNs_dFoF: dict
    fraction_MRNs_spikes: dict
    fraction_rMRNs_dFoF: dict
    fraction_rMRNs_spikes: dict
    fraction_silent_dFoF: dict
    fraction_silent_spikes: dict
    movement_correlation: dict
    movement_stereotypy: dict
    movement_reliability: dict
    movement_specificity: dict
    movement_traces_dFoF: dict
    movement_amplitudes_dFoF: dict
    mean_onsets_dFoF: dict
    movement_traces_spikes: dict
    movement_amplitudes_spikes: dict
    mean_onsets_spikes: dict
    individual_mvmt_onsets: dict
    mvmt_onset_jitter: dict
    avg_pop_vector: dict
    event_pop_vectors: dict
    med_vector_similarity: dict
    med_vector_correlation: dict

    def save(self):
        """Method to save the dataclass"""
        initial_path = r"G:\Analyzed_data\individual"
        save_path = os.path.join(initial_path, self.mouse_id, "paAIP2_activity")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        fname = f"{self.mouse_id}_{self.group}_population_activity_data"
        save_pickle(fname, self, save_path)
