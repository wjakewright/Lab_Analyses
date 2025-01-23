import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from scipy.ndimage import uniform_filter1d
from sklearn import linear_model, metrics, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils
from Lab_Analyses.Utilities.save_load_pickle import save_pickle

sns.set()
sns.set_style("ticks")


class RWD_Movement_Model:
    """Class for decoding rewarded movements from neural activity for
    individual mice
    """

    def __init__(self, data):
        """Initialization method

        INPUT PARAMETERS
            data - paAIP2_Population_Data dataclass from an individual mouse

        """
        # Class attributes
        self.model_name = f"{data.mouse_id}_{data.group}_rwd_movement_model"
        self.mouse_id = data.mouse_id
        self.session = data.session
        self.group = data.group

        ## Unprocessed data variables
        self.z_score_spikes = data.z_score_spikes
        self.lever_active_rwd = data.lever_active_rwd
        self.MRNs = data.mvmt_cells_spikes

        ## Processed data variables
        self.downsampled_spikes = None
        self.downsampled_lever = None

        self.gapped_spikes = None
        self.gapped_lever = None

        ## Model results
        self.true_model_train_score = None  # 1d array of each CV train results
        self.true_model_test_score = None  # 1d array of each CV test restuls
        self.shuff_model_train_score = None  # 2d array of each CV/shuff train results
        self.shuff_model_test_score = None  # 2d array of each CV/shuff test results

        # Other parameters
        self.model_type = None
        self.cell_num = 100
        self.random_state = 78
        self.classes = None
        self.score_method = None
        self.shuff_iterations = None
        self.resample_num = 10
        self.cv_num = None
        self.C_param = None

        # Scorer options
        self.scorer_dict = {
            "accuracy": make_scorer(metrics.accuracy_score),
            "balanced_accuracy": make_scorer(metrics.balanced_accuracy_score),
            "average_precision": make_scorer(metrics.average_precision_score),
            "precision": make_scorer(metrics.precision_score),
            "recall": make_scorer(metrics.recall_score),
            "f1_score": make_scorer(metrics.f1_score),
            "roc_auc": make_scorer(metrics.roc_auc_score),
        }

        # Preprocess the data
        self.downsample_data()
        self.gap_active_periods()

    def downsample_data(self):
        """method to downsample neural activity and lever trace"""
        print(
            f"Original Data Array Sizes: Lever {len(self.lever_active_rwd)}  Spikes {self.z_score_spikes.shape}"
        )
        FRAMES = 6
        BINS = int(len(self.lever_active_rwd) / FRAMES)

        downsample_spikes = np.array_split(
            self.z_score_spikes, indices_or_sections=BINS, axis=0
        )
        downsample_spikes = np.array([np.nanmean(x, axis=0) for x in downsample_spikes])

        downsample_lever = np.array_split(
            self.lever_active_rwd, indices_or_sections=BINS
        )
        downsample_lever = np.array([np.namean(x) for x in downsample_lever])
        # Ensure it is a binary array
        downsample_lever[downsample_lever > 0] = 1

        # Store values
        self.downsampled_spikes = downsample_spikes
        self.downsampled_lever = downsample_lever

        print(
            f"Downsampled Array Sizes: Lever {len(self.downsampled_lever)}  Spikes {self.downsampled_spikes.shape}"
        )

    def gap_active_periods(self):
        """method to drop periods immediately around rewarded lever presses"""
        ## Extend lever trace by 400ms on either side
        expansion = 12
        exp_constant = np.ones(expansion, dtype=int)
        npad = len(exp_constant) - 1
        l_pad = np.pad(
            self.downsampled_lever, (npad // 2, npad - npad // 2), mode="constant"
        )
        lever_extended = (
            np.convolve(l_pad, exp_constant, "valid").astype(bool).astype(int)
        )

        ## Find the idxs of the extension
        diff_trace = lever_extended - self.downsampled_lever
        good_idxs = np.nonzero(diff_trace == 0)[0]

        gapped_lever = self.downsampled_lever[good_idxs]
        gapped_spikes = self.downsampled_spikes[good_idxs, :]

        self.gapped_spikes = gapped_spikes
        self.gapped_lever = gapped_lever

        print(
            f"Gapped Data Sizes: Lever {len(self.gapped_spikes)}  Spikes {gapped_spikes.shape}"
        )

    def train_real_model(self, classes, score_method, cv_num, model_type, C):
        """Method to train and evaluate the model on real data

        INPUT PARAMETERS
            classes - dict mapping str to the int labels for the classes in y

            score_method - str specifying the score method to use

            cv_num - in specifying the number of cross validations to perform

            model_typ - str specifying the model type to use

            C - float specifying the C parameter for the model training


        """
        self.model_type = model_type
        self.score_method = score_method
        self.classes = classes
        self.cv_num = cv_num
        self.C_param = C

        # Get the scorer
        scorer = self.scorer_dict[score_method]

        idxs = np.arange(self.gapped_spikes.shape[1])

        train_scores_list = []
        test_scores_list = []

        for _ in range(self.resample_num):
            # Subsample only 100 neurons
            curr_idxs = np.random.choice(idxs, size=self.cell_num, replace=False)
            subsampled_x = self.gapped_spikes[:, curr_idxs]
            y = self.gapped_lever

            train_scores, test_scores = train_test_model(
                X=subsampled_x,
                y=y,
                scorer=scorer,
                resample_num=self.resample_num,
                cv_num=cv_num,
                model_type=model_type,
                C=C,
                random_state=self.random_state,
            )
            train_scores_list.append(train_scores)
            test_scores_list.append(test_scores)

        # Store final results
        self.true_model_train_score = np.array(train_scores_list).flatten()
        self.true_model_test_score = np.array(test_scores_list).flatten()

    def train_shuffle_model(self, iterations=10):
        """
        Method to train and evaluate shuffled models to estiamte chance

        INPUT PARAMETERS
            iterations - int specifying how many shuffling iterations to perform

        """
        if self.true_model_test_score is None:
            return "Must train true model first!"

        self.shuff_iterations = iterations
        scorer = self.scorer_dict[self.score_method]

        idxs = np.arange(self.gapped_spikes.shape[1])

        all_shuff_train_scores = []
        all_shuff_test_scores = []

        for i in range(iterations):
            print(f"PERFORMING SHUFFLE {i}")
            # Shuffle data
            shuff_x = copy.deepcopy(self.gapped_spikes)
            shuff_y = copy.deepcopy(self.gapped_lever)
            np.random.shuffle(shuff_y)
            resampled_test_scores = []
            resampled_train_scores = []
            # Subselect the neurons
            for _ in range(self.resample_num):
                curr_idxs = np.random.choice(idxs, size=self.cell_num, replace=False)
                subsampled_x = shuff_x[:, curr_idxs]
                train_scores, test_scores = train_test_model(
                    X=subsampled_x,
                    y=shuff_y,
                    scorer=scorer,
                    resample_num=self.resample_num,
                    cv_num=self.cv_num,
                    model_type=self.model_type,
                    C=self.C,
                    random_state=self.random_state,
                )
                resampled_test_scores.append(test_scores)
                resampled_train_scores.append(train_scores)

            # Store the scores for each shuffle
            all_shuff_train_scores.append(np.array(resampled_train_scores).flatten())
            all_shuff_test_scores.append(np.array(resampled_test_scores).flatten())

        # Reformat scores into 2d arrays
        self.shuff_model_train_score = np.vstack(all_shuff_test_scores)
        self.shuff_model_test_score = np.vstack(all_shuff_train_scores)

    def save(self):
        """method to save the model"""
        initial_path = r"G:\Analyzed_data\individual"
        save_path = os.path.join(initial_path, self.mouse_id, "paAIP2_activity")
        fname = f"{self.mouse_id}_{self.group}_{self.session}_RWD_model"
        save_pickle(fname, self, save_path)


########################### Independent Functions ################################


def train_test_model(
    X,
    y,
    scorer,
    resample_num=1,
    cv_num=10,
    model_type="SVM_linear",
    C=0.1,
    random_state=78,
):
    """Function to train and test models in a cross-validated fashion. Different stages
    of preprocessing and testing are manually controlled to allow greater customization

    INPUT PARAMETERS
        x_values - 2d np.array of the selected neurons activity

        y_values - np.array of the coded labels for y values

        scorer - scorer object to be used for evaluating the model performance

        resample_num - int specifying how many times to resample the data if performed

        cv_num - int specifying how many cross-validations to perform

        model_type - str specifying the type of model to use. Currently coded for
                    "SVM_linear", "SVM_rbf" and "Logistic"

        C - float specifying the degree of regularization for the models

        random_state - int specifying the random state for reproducibility

    OUTPUT PARAMETERS
        train_scores - np.array of the individual train scores

        test_scores - np.array of the individual test scores


    """

    # Set up some final ougputs
    train_scores = []
    test_scores = []

    # Set up some generators
    cv = StratifiedShuffleSplit(
        n_splits=cv_num, test_size=(1 / cv_num), random_state=random_state
    )
    scaler = MinMaxScaler()

    # Set up the resampling method
    sampler = RandomUnderSampler(
        sampling_strategy="not minority",
        random_state=random_state,
    )

    for _ in range(resample_num):
        # resample the data based on y
        X_resample, y_resample = sampler.fit_resample(X, y)

        # Initialize the model
        if model_type == "SVM_linear":
            model = svm.SVC(kernel="linear", class_weight="balanced", C=C)
        elif model_type == "SVM_rbf":
            model = svm.SVC(kernel="rbf", class_weight="balanced", C=C)
        else:
            return f"{model_type} model type is not currently supported"

        # Set up the pipeline
        pipe = Pipeline(steps=[("scaling", scaler), ("classify", model)])

        # Perform the cross-validated training and testing
        cv_results = cross_validate(
            estimator=pipe,
            X=X_resample,
            y=y_resample,
            scoring=scorer,
            cv=cv,
            return_train_score=True,
        )

        test_scores.append(cv_results["test_score"])
        train_scores.append(cv_results["train_score"])

    return train_scores, test_scores
