import os

import numpy as np

from Lab_Analyses.Population_Analysis import population_utilities as p_utils
from Lab_Analyses.Utilities import data_utilities as d_utils


def paAIP2_population_analysis(
    paAIP2_mice,
    EGFP_mice,
    matched=False,
    activity_window=(-2, 4),
    save_ind=False,
    save_grouped=False,
):
    """Function to analyze experimental and control mice from paAIP2 population
        imaging experiments
        
        INPUT PARAMETERS
            paAIP2_mice - list of str specifying the mice in the paAIP2 group to be
                          analyzed
            
            EGFP_mice - list of str specifying the mice in the EGFP group to be
                        analyzed
            
            matched - boolean specifying whether or not the data are roi matched
                      across days
            
            activity_window - tuple specifying the window size to analyze activity around

            save_ind - boolean specifying whetehr to save the data for each mouse

            save_grouped - boolean specifying whether or not to group all mice together
                            and save
    """

    all_mice = paAIP2_mice + EGFP_mice
    paAIP2_data = []
    EGFP_data = []

