import numpy as np

from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def get_fraction_MRNs(sessions, MRN_list):
    """Function to calculate the fraction of MRSNs across multiple sessions
    
    INPUT PARAMETERS
        sessions - list of str specifying the name of each session
        
        MRN_list - list of boolean arrays indentifying each neuron as 
                    belonging to a specific class (e.g., MRNs)
                    
    OUTPUT PARAMETERS
        fraction_MRNs - dict of arrays containing the fraction of MRNs
    """

    fraction_MRNs = {}
    for session, MRNs in zip(sessions, MRN_list):
        fraction = np.nansum(MRNs) / np.sum(~np.isnan(MRNs))  # account for nans
        fraction_MRNs[session] = np.array([fraction])

    return fraction_MRNs


def calculate_movement_encoding(
    mouse_id,
    sessions,
    activity_list,
    lever_active_list,
    lever_force_list,
    threshold=0.5,
    corr_duration=0.5,
    sampling_rate=30,
):
    """Function to calculate movement encoding properties of neurons across sessions
    
        INPUT PARAMETERS
            mouse_id - str specifying the mouse id. Used to pull relevant learned movement

            sessions - list of str specirfying the session names
            
            activity_list - list of the activity matrices for each session
            
            lever_active_list - list of the lever active binary traces
            
            lever_force_list - list of the lever force traces
            
            threshold - float of the correlation threshold for a movement to be considered
                        a learned movement
                        
            corr_duration - float specifying how long (sec) of the movements to correlated
            
            sampling_rate - int specifying the imaging sampling rate

        OUTPUT PARAMETERS
            movement_correlation - dict of arrays containing the LMP corr for each cell for 
                                    each session
            
            movement_stereotypy - dict of arrays containing the stereotypy for each cell 
                                    for each session

            movement_reliability - dict of arrays containing the reliability for each cell
                                    for each session

            movement_specificity - dict of arrays containing the specificity for each cell
                                    for each session
    """

    # Set up the outputs
    movement_correlation = {}
    movement_stereotypy = {}
    movement_reliability = {}
    movement_specificity = {}

    # Iterate through each session
    for session, activity, lever_active, lever_force in zip(
        sessions, activity_list, lever_active_list, lever_force_list
    ):
        (_, corr, stereo, reli, speci, _, _, _,) = quantify_movement_quality(
            mouse_id,
            activity,
            lever_active,
            lever_force,
            threshold=threshold,
            corr_duration=corr_duration,
            sampling_rate=sampling_rate,
        )
        movement_correlation[session] = corr
        movement_stereotypy[session] = stereo
        movement_reliability[session] = reli
        movement_specificity[session] = speci

    return (
        movement_correlation,
        movement_stereotypy,
        movement_reliability,
        movement_specificity,
    )

