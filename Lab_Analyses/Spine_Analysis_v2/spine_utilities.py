import os

import numpy as np

from Lab_Analyses.Utilities.data_utilities import pad_array_to_length
from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def find_stable_spines(spine_flags):
    """Function to find stable spines from a single session

    INPUT PARAMETERS
        spine_flags - list of the spine flags

    OUTPUT PARAMETERS
        stable_spines - boolean np.array of whether each spine is stable
    """
    stable_spines = []
    for flag in spine_flags:
        if "New Spine" in flag or "Eliminated Spine" in flag or "Absent" in flag:
            stable_spines.append(False)
            continue
        else:
            stable_spines.append(True)

    return np.array(stable_spines)


def find_stable_spines_across_days(spine_flag_list):
    """Function to find stable spines across multiple sessions

    INPUT PARAMETERS
        spine_flag_list - list of lists containing all the spine flags

    OUTPUT PARAMETERS
        stable_spines - boolean array of whether or not each spine is stable
                        across all sessions
    """
    # Find stable spines for each day
    daily_stable_spines = []
    for spine_flags in spine_flag_list:
        stable = find_stable_spines(spine_flags)
        daily_stable_spines.append(stable)

    # Find stable spines across all days
    ## Pad arrays first
    max_len = np.max([len(x) for x in daily_stable_spines])
    padded_spines = []
    for ds in daily_stable_spines:
        if len(ds) != max_len:
            padded_spines.append(
                pad_array_to_length(ds, max_len, value=False).astype(bool)
            )
        else:
            padded_spines.append(ds)
    stable_spines = np.prod(np.vstack(padded_spines), axis=0)

    return stable_spines


def find_present_spines(spine_flags):
    """Function to find spines present during the imaging session. Excludes eliminated
    and absent spines

    INPUT PARAMETERS
        spine_flags - list of spine flags

    OUTPUT PARAMETERS
        present_spines - boolean array of whether or not each spine is present
    """
    # Initialize output
    present_spines = []
    for flag in spine_flags:
        if "Eliminated Spine" in flag or "Absent" in flag:
            present_spines.append(False)
        else:
            present_spines.append(True)

    return np.array(present_spines)


def find_spine_classes(spine_flags, spine_class):
    """Function to find specific types of spines based on their flags

    INPUT PARAMETERS
        spine_flags - list of the spine flags

        spine_class - str specifying what type of spine you want to find

    OUTPUT PARAMETERS
        classed_spines - boolean array of whether or not each spine is stable
    """
    # Initialize output
    classed_spines = np.zeros(len(spine_flags)).astype(bool)
    # Find the specific spine classes
    for i, spine in enumerate(spine_flags):
        if spine_class in spine:
            classed_spines[i] = True

    return classed_spines


def load_spine_datasets(mouse_id, days, fov_type):
    """Function to handel loading all the spine datasets for a mouse

    INPUT PARAMETERS
        mouse_id - str specifying which mouse to load

        days - list of str specifying which days to load. Used
                to search for filenames

        fov_type - str specifying whether to load apical or basal FOVs

    OUTPUT PARAMETERS
        mouse_data - dict of dict containing data for each FOV (upper dict)
                    for each imaged data (lower dict)
    """
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    data_path = os.path.join(initial_path, mouse_id, "spine_data")
    FOVs = next(os.walk(data_path))[1]
    FOVs = [x for x in FOVs if fov_type in x]

    mouse_data = {}
    for FOV in FOVs:
        FOV_path = os.path.join(data_path, FOV)
        FOV_data = {}
        fnames = next(os.walk(FOV_path))[2]
        for day in days:
            load_name = [x for x in fnames if day in x][0]
            data = load_pickle([load_name], path=FOV_path)[0]
            FOV_data[day] = data
        mouse_data[FOV] = FOV_data

    return mouse_data


def bin_by_position(data, positions, bins, const=None):
    """Helper function to bin pairwise spine data by relative positions"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.nonzero((positions > bins[i]) & (positions <= bins[i + 1]))[0]
            if idxs.size == 0:
                binned_data.append(np.nan)
                continue
            if const is None:
                binned_data.append(np.nanmean(data[idxs]))
            else:
                binned_data.append(np.nansum(data[idxs]) / const)

    return np.array(binned_data)


def find_nearby_spines(
    spine_positions, spine_flags, spine_groupings, partner_list=None, cluster_dist=5
):
    """Function to find the idxs of each spines neighbors

    INPUT PARAMETERS
        spine_positions - np.array of the the position of each spine along the
                         dendrite

        spine_flags - list of the spine flags

        spine_groupings - list containing the spine groupings for each dendrite

        partner_list - boolean list of spine identities you wish to constrain your
                        analysis to

        cluster_dist - int or float specifying the distance that is to be
                        considered nearby

    OUTPUT PARAMETERS
        nearby_spine_idxs - list containing an array of the nearby spine idxs
                            for each spine
    """
    # Sort out the spine groupings to ensure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Find all the present spines
    present_spines = find_present_spines(spine_flags)

    # Setup the output
    nearby_spine_idxs = [None for i in spine_positions]

    # Iterate through each spine grouping
    for spines in spine_groupings:
        curr_positions = spine_positions[spines]
        curr_present = present_spines[spines]
        # Iterate through each spine
        for spine, position in enumerate(curr_positions):
            # Get the relative positions
            relative_positions = curr_positions - position
            relative_positions = np.absolute(relative_positions)
            nearby_spines = np.nonzero(relative_positions <= cluster_dist)[0]
            # remove target spine and any absent spines
            nearby_spines = [
                x for x in nearby_spines if curr_present[x] == True and x != spine
            ]
            if partner_list is not None:
                nearby_spines = [
                    x for x in nearby_spines if partner_list[spines[x]] == True
                ]
            try:
                nearby_spine_idxs[spines[spine]] = np.array(spines)[nearby_spines]
            except TypeError:
                nearby_spine_idxs[spines[spine]] = []

    return nearby_spine_idxs


def parse_movement_nonmovement_spines(movement_spines, rwd_movement_spines):
    """Function to generate additonal arrays for nonmovement spines and
    movement non rewarded spines

    INPUT PARAMETERS
        movement_spines - boolean list/array of whether each spine in movement or not

        rwd_movement_spines - boolean list/array  of whether each spine is
                              rewarded movement encoding or not

    OUTPUT PARAMETERS
        nonmovement_spines - boolean list/array of whether each spine is nonmovement
                             or not

        movement_non_rwd_spines - boolean list/array of whether each spine is
                                 non movement or not
    """
    # Find nonmovement spines
    nonmovement_spines = np.array([not x for x in movement_spines])
    # Find movement but not reward movement spines
    movement_non_rwd_spines = np.array(movement_spines, dtype=int) - np.array(
        rwd_movement_spines, dtype=int
    ).astype(bool)
    movement_non_rwd_spines[movement_non_rwd_spines < 0] = 0

    return nonmovement_spines, movement_non_rwd_spines


def load_analyzed_datasets(
    type,
    grouped=True,
    session="Early",
    norm=True,
    activity_type="dFoF",
    extended=None,
    mouse=None,
    fov=None,
    fov_type="apical",
    period=None,
    partner=None,
):
    """Function to handle loading of analyzed datasets

    INPUT PARAMETERS
        type - str specifying the type of data to load. Will be used for
                searching for files. Accepts 'Activity', 'Local', 'Global'

        grouped - boolean specifying whether to load a grouped or individual
                dataset

        session - str specifying what session to load

        norm - boolean specifying whether to load normalized dataset or not

        activity_type - str specifying what type of activity the dataset
                        used (dFoF or zscore)

        extended - boolean specifying whetehr or not the dataset to load
                    used extended activity periods

        mouse - str specifying which mouse and fov to load. Used only if loading
                individual dataset.

        fov - str specifying the FOV to load. Only used if loading individual
                dataset

        fov_type - str specifying what type of fovs the data come from. eg.
                    apical or basal

        period - str specifying whether to load a dataset that was constrained
                to specific movement periods. Accepts 'movement', 'nonmovement'
                and 'rewarded movement'. Default is None to load unconstrained
                dataset

        partner - str specifying whether to load a dataset that examined coactivity
                for only given partner types. Accepts 'MRS', 'nMRS' and 'rMRS'.
                Default is None to load unrestricted data set

    OUTPUT PARAMETERS
        loaded_dataset - object containing the specified dataset

    """
    if norm:
        n_name = "_norm"
    else:
        n_name = ""
    if period is None:
        period = "session"
    if partner is None:
        pname = ""
    else:
        pname = f"_{partner}"
    if extended is None:
        ename = ""
    else:
        ename = "_extended"

    # Set up the path
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data"
    if grouped:
        load_path = os.path.join(
            initial_path, "grouped", "Dual_Spine_Imaging", "Coactivity_Data"
        )
    else:
        if mouse == None:
            return "Must specify mouse ID if loading individual dataset"
        load_path = os.path.join(
            initial_path,
            "individual",
            mouse,
            "coactivity_data",
            f"{fov}_{fov_type}",
            session,
        )

    # Construct the file name to load
    if type == "Activity":
        if grouped:
            fname = f"{session}_{fov_type}_{activity_type}{n_name}_grouped_spine_activity_data"
        else:
            fname = f"{mouse}_{session}_{activity_type}{n_name}_spine_activity_data"
    elif type == "Local":
        if grouped:
            fname = f"{session}_{fov_type}_{activity_type}{n_name}_{period}{pname}_grouped_local_coactivity_data"
        else:
            fname = f"{mouse}_{session}_{activity_type}{n_name}_{period}{pname}_local_coactivity_data"
    elif type == "Global":
        if grouped:
            fname = f"{session}_{fov_type}_{activity_type}{n_name}_{period}{pname}{ename}_grouped_dendritic_coactivity_data"
        else:
            fname = f"{mouse}_{session}_{activity_type}{n_name}_{period}{pname}{ename}_dendritic_coactivity_data"
    else:
        return f"Does not accept {type} type of dataset to load"

    # Load the data
    try:
        load_name = os.path.join(load_path, fname)
        loaded_data = load_pickle([load_name])[0]
    except FileNotFoundError:
        print(load_name)

        print("File matching input parameters does not exist")

    return loaded_data


def batch_load_individual_analyzed_datasets(
    type,
    session="Early",
    norm=True,
    activity_type="dFoF",
    extended=None,
    mice_list=None,
    fov_type="apical",
    period=None,
    partner=None,
):
    """Heler function to load all the individual datasets for a group of mice"""
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    all_data = []
    for mouse in mice_list:
        # Check and get all the FOVs
        check_path = os.path.join(initial_path, mouse, "coactivity_data")
        FOVs = next(os.walk(check_path))[1]
        FOVs = [x for x in FOVs if fov_type in x]
        FOVs = [x.split("_")[0] for x in FOVs]
        for fov in FOVs:
            loaded_data = load_analyzed_datasets(
                type,
                grouped=False,
                session=session,
                norm=norm,
                activity_type=activity_type,
                extended=extended,
                mouse=mouse,
                fov=fov,
                fov_type=fov_type,
                period=period,
                partner=partner,
            )
            all_data.append(loaded_data)
    return all_data


def calculate_nearby_vs_distance_variable(variable_mat, position_bins, cluster_dist):
    """Helper function to calculate the nearby coactivity rate vs
    the distance coactivity rate for each spine"""
    DISTANT_BIN = position_bins[-1] - cluster_dist
    # Distant is considered further than 30um

    # Get idx for the nearby cluster bin cutoff
    near_bin_idx = np.nonzero(position_bins == cluster_dist)[0][0]
    dist_bin_idx = np.nonzero(position_bins >= DISTANT_BIN)[0][0]

    # Slice near and distant data
    ## Check of the bin size alignes with the cluster
    if near_bin_idx > 0:
        near = variable_mat[:near_bin_idx, :]
        near = np.nanmean(near, axis=0)
    else:
        near = variable_mat[near_bin_idx, :]
    distant = variable_mat[dist_bin_idx:, :]
    distant = np.nanmean(distant, axis=0)

    # Perform calculation
    near_minus_dist = near - distant

    return near_minus_dist
