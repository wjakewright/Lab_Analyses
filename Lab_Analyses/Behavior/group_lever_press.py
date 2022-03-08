""""Module to analyze lever press behavior across mice within the same experimental group

    CREATOR - William (Jake) Wright 3/7/2022"""

import numpy as np

class Group_Lever_Press:
    """Class for grouped analysis of lever press behavior across mice"""

    def __init__(self, files):
        """Initialize Group_Lever_Press Class

            INPUT PARAMETERS
                files - list containing Mouse_Lever_Behavior dataclass objects for each
                        mouse to be analyzed in the group
        """

        # Store variables and attributes
        self.files = files
        self.mice = []
        for file in files:
            self.mice.append(file.mouse_id)
        
        # Check that all mice have the same number of sessions
        self.check_same_sessions()

        self.sessions = files[0].sessions

        # Attributes to be defined later
        self.avg_corr_matrix = None
        self.within_sess_corr = None
        self.across_sess_corr = None

    def average_correlation_matrix(self):
        """Function to average the correlation matrices across mice"""
        # Grab correlation matricies for each mouse
        corr_matrices = [x.correlation_matrix for x in self.files]
        # Concatenate matrices along the 3rd axis
        cat_matrices = np.dstack(tuple(corr_matrices))
        # Get the mean
        mean_corr = np.nanmean(cat_matrices, axis=2)
        # Store result
        self.avg_corr_matrix = mean_corr

    
    def analyze_within_sess_corr(self):
        """Function get mean and sem within session correlations across sessions"""
        # Get all the within session correlations
        # Store as arrays in a list for each session
        all_within_corr = []

    
    def check_same_sessions(self):
        """Function to check to make sure that all mice have same number of sessions"""
        sess_nums  = [len(x.sessions) for x in self.files]
        values, counts = np.unique(sess_nums, return_counts=True)
        if len(values) > 1:
            diff_idx = [counts.index(x) for x in counts if x != np.max(counts)]
            error_mice = [file.mouse_id for file in np.array(self.files)[diff_idx]]
            majority_num = np.max(counts)
            diff_values = [x for x in np.array(counts)[diff_idx]]
            error_mice_values = zip(error_mice, diff_values)
            print(f"Supposed to have {majority_num} sessions")
            [print(f"{x} has {y} sessions") for x,y in error_mice_values]
            raise ValueError("Mice cannot have different number of sessions!!!!")
        