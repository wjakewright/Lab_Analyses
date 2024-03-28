import os

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
    for session in sessions:
        load_name = [x for x in fnames if session in x][0]
        data = load_pickle([load_name], path=data_path)[0]
        mouse_data[session] = data

    return mouse_data
