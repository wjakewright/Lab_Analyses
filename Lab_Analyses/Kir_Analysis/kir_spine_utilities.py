import os

from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def load_analyzed_kir_datasets(
    type,
    grouped=False,
    session="Early",
    activity_type="dFoF",
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

        activity_type - str specifying what type of activity the dataset
                        used (dFoF or zscore)

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
    if partner is None:
        pname = ""
    else:
        pname = f"_{partner}"

    if period is None:
        period = "session"

    # Set up path
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data"
    if grouped:
        load_path = os.path.join(
            initial_path, "grouped", "Kir_Spine_Imaging", "Activity_Data"
        )
    else:
        if mouse == None:
            return "Must specify mouse ID if loading individual dataset"
        load_path = os.path.join(
            initial_path, "individual", mouse, "kir_data", f"{fov}_{fov_type}", session
        )

    # Construct the file name to load
    if type == "Activity":
        if grouped:
            fname = f"{session}_{fov_type}_{activity_type}_grouped_kir_activity_data"
        else:
            fname = f"{mouse}_{session}_{activity_type}_kir_activity_data"
    elif type == "Coactivity":
        if grouped:
            fname = f"{session}_{fov_type}_{activity_type}_{period}{pname}_grouped_kir_coactivity_data"
        else:
            fname = (
                f"{mouse}_{session}_{activity_type}_{period}{pname}_kir_coactivity_data"
            )

    # Load the data
    try:
        load_name = os.path.join(load_path, fname)
        loaded_data = load_pickle([load_name])[0]
    except FileNotFoundError:
        print(load_name)
        print("File matching input parameters does not exist")

    return loaded_data


def batch_load_individual_kir_analyzed_datasets(
    type,
    session="Early",
    activity_type="dFoF",
    mice_list=None,
    fov_type="apical",
    period=None,
    partner=None,
):
    """Helper function to load all the individual datasets for a group of mice"""
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    all_data = []
    for mouse in mice_list:
        # Check and get all the FOVs
        check_path = os.path.join(initial_path, mouse, "kir_data")
        FOVs = next(os.walk(check_path))[1]
        FOVs = [x for x in FOVs if fov_type in x]
        FOVs = [x.split("_")[0] for x in FOVs]
        for fov in FOVs:
            loaded_data = load_analyzed_kir_datasets(
                type,
                grouped=False,
                session=session,
                activity_type=activity_type,
                mouse=mouse,
                fov=fov,
                fov_type=fov_type,
                period=period,
                partner=partner,
            )
            all_data.append(loaded_data)

    return all_data
