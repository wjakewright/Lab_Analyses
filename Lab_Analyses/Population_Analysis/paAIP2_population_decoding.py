import copy
import os
import random

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, svc
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils


class Population_Decoding_Model:
    """Class for decoding rewarded movements from population neural
    activity data
    """

    def __init__(self, model_name):
        """Initialize the model

        INPUT PARAMETERS
            model_name - str specifying the name of the model
                        (e.g. "JW157_paAIP2_Model")

        """

        # Class attributes
        self.model_name = model_name
        self.model_type = None  # Str specifying the type of model

        # Data and features
        self.X = None  # dict (time bin) of 2d array of x values (trial x neuron)
        self.y = None  # 1d array of y values corresponding to x values (trials)
        self.classes = None  # Dict mapping class str to int labels

        # Feature weights / importance (for largest model)
        self.feature_weights = None  # List of feature weights for each CV
        self.feature_importance_train = (
            None  # List of training feature importances for each CV
        )
        self.feature_importance_test = (
            None  # List of testing feature importances for each CV
        )

        # Model results
        self.num_real_train_score = (
            None  # dict (neuron num) of train score arrays (sample x CV)
        )
        self.num_real_test_score = (
            None  # dict (neuron num) of test score arrays (sample x CV)
        )
        self.num_shuff_train_score = (
            None  # dict (neruon num) of train score arrays (sample x CV)
        )
        self.num_shuff_test_score = (
            None  # dict (neuron num) of test score arrays (sample x CV)
        )

        self.time_real_train_score = (
            None  # dict (time bin) of train score arrays (sample x CV)
        )
        self.time_real_test_score = (
            None  # dict (time bin) of test score arrays (sample x CV)
        )
        self.time_shuff_train_score = (
            None  # dict (time bin) of train score arrays (sample x CV)
        )
        self.time_shuff_test_score = (
            None  # dict (time bin) of test score arrays (sample x CV)
        )

        # Other parameters
        self.random_state = 78
        self.bin_size = None  # int of how many frames to average for each bin
        self.mvmt_window = None  # tuple specifying the analysis window around movments
        self.time_points = None  # array of time points used for decoding
        self.score_method = (
            None  # str specifying which method to evaluate model performance
        )
        self.neuron_nums = (
            None  # list of the number of neurons used for different models
        )
        self.neuron_repeats = (
            None  # Int specifying how many time neurons were resampled
        )
        self.shuff_iterations = None  # Int specifying the number of shuffles
        self.cv_num = None  # Int specifying the number of CVs
        self.C_param = None  # float specifying the C parameter for the model

        # Scorer options
        self.scorer_dict = {
            "accuracy": make_scorer(metrics.accuracy_score),
            "balanced_accuracy": make_scorer(metrics.balanced_accuracy_score),
            "average_precision": make_scorer(metrics.average_precision_score),
            "f1": make_scorer(metrics.f1_score),
            "precision": make_scorer(metrics.precision_score),
            "recall": make_scorer(metrics.recall_score),
            "roc_auc": make_scorer(metrics.roc_auc_score),
        }

    def organize_data(
        self,
        activity,
        lever_active,
        mvmt_window,
        bin_size,
        sampling_rate,
    ):
        """Method to organize the data for decoding analysis

        INPUT PARAMETERS
            activity - 2d array of neural activity (time x neurons)

            lever_active - 1d array of binarized lever activity

            mvmt_window - tuple specifying the window around movements to
                            use for decoding (in sec)

            bin_size - int specifying how many frames to bin the activity over
                        for each time point

            sampling_rate - int specifying the imaging frame rate

        """
        self.mvmt_window = mvmt_window
        self.bin_size = bin_size
        # Define the window in frames and get center point
        frame_num = (np.absolute(mvmt_window[0]) * sampling_rate) + (
            np.absolute(mvmt_window[1]) * sampling_rate
        )
        center_point = np.absolute(mvmt_window[0]) * sampling_rate

        # Set up the bin timepoints
        bin_num = int(frame_num / bin_size)
        time_bins = np.linspace(0, frame_num, bin_num).astype(int)
        ## Convert to seconds
        time_bins_sec = (time_bins - center_point) / sampling_rate
        self.time_points = time_bins_sec
        ## Get idxs to average bins over
        time_bin_idxs = [
            (time_bins[i], time_bins[i + 1]) for i in range(len(time_bins) - 1)
        ]

        # Find all mvmt onsets
        timestamps = t_stamps.get_activity_timestamps(lever_active)
        movement_epochs = t_stamps.refine_activity_timestamps(
            timestamps,
            window=mvmt_window,
            max_len=activity.shape[0],
            sampling_rate=sampling_rate,
        )
        ## Use only the onsets
        movement_onsets = [x[0] for x in movement_epochs]

        # Grab activity around movements
        activity_traces, _ = d_utils.get_trace_mean_sem(
            activity,
            ROI_ids=list(range(activity.shape[1])),
            timestamps=movement_onsets,
            window=mvmt_window,
            sampling_rate=sampling_rate,
        )
        activity_traces = list(activity_traces.values())
        # Need to reorganize the traces for each movement
        stacked_activity = np.dstack(activity_traces)
        movement_activity = [
            stacked_activity[:, i, :] for i in range(stacked_activity.shape[1])
        ]
        ## Should be a list of (time x neuron) for each event

        # Get binned activity for each time point
