import itertools
import os

import numpy as np

from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def load_population_datasets(mouse_id, sessions):
    """Function to handle loading all of the population datasets for a mouse

    INPUT PARAMETERS
        mouse_id - str specifyiing which mouse to load

        sessions - list of str specifying which sessions to load.
                    Used for file name searching

    OUTPUT PARAMETERS
        mouse_data - dict containing the data for each day

    """
    initial_path = r"G:\Analyzed_data\individual"

    data_path = os.path.join(initial_path, mouse_id, "population_data")
    fnames = next(os.walk(data_path))[2]

    mouse_data = {}
    mouse_spikes = {}
    for session in sessions:
        session_data = session + "_population_data"
        load_name = [x for x in fnames if session_data in x][0]
        data = load_pickle([load_name], path=data_path)[0]
        mouse_data[session] = data
        session_spikes = session + "_population_spikes"
        spike_name = [x for x in fnames if session_spikes in x][0]
        spikes = np.load(os.path.join(data_path, spike_name))
        spikes = np.nan_to_num(spikes, nan=0)
        mouse_spikes[session] = spikes.T

    return mouse_data, mouse_spikes


def calc_pairwise_distances_btw_vectors(array_list):
    """Helper function to calculate the Eucledian distances between
    multidimensional vectors

    INPUT PARAMETERS
        array_list - list of 2d arrays. Rows represent time and columns represent
                    dimensions of the vector.

    OUTPUT PARAMETERS
        similarities - np.array of all pairwise similarities

        med_similarity - float of the median of all similarities


    """
    # Get all combinations of events
    combos = itertools.combinations(array_list, 2)
    # Perform pariwise similarities
    similarities = []
    for combo in combos:
        # Seperate vectors
        a = combo[0]
        b = combo[1]
        # Calculate the distance
        dist = np.linalg.norm(a - b, axis=1)
        # Average the distance across all time points
        avg_dist = np.nanmean(dist)
        # Get the inverse
        similarities.append(1 / avg_dist)

    # Calculate the overall median similarity
    med_similarity = np.nanmedian(similarities)

    return np.array(similarities), med_similarity


def load_analyzed_pop_datasets(mouse_list, group="paAIP2"):
    """Function to load multiple analyzed datasets

    INPUT PARAMETERS
        mouse_list - list of str with mouse ids

    OUTPUT PARAMETERS
        loaded_data - list of analyzed data objects

    """
    initial_path = r"G:\Analyzed_data\individual"
    loaded_data = []
    for mouse in mouse_list:
        load_path = os.path.join(initial_path, mouse, "paAIP2_activity")
        fname = f"{mouse}_{group}_population_activity_data"
        try:
            load_name = os.path.join(load_path, fname)
            data = load_pickle([load_name])[0]
            loaded_data.append(data)
        except FileNotFoundError:
            print(load_name)
            print("File matching input parameters does not exist")

    return loaded_data
