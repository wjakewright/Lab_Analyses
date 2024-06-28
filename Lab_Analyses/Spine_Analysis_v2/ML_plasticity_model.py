import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn import linear_model, metrics, svm
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import type_of_target

from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils

sns.set()
sns.set_style("ticks")


class ML_Plasticity_Model:
    """Class for ML models to predict synaptic plasticity based on different
    activity-based features
    """

    def __init__(self, model_name):
        """Initialization method.

        INPUT PARAMETERS
            model_name - str specifying the name of the model
                        (e.g., "Apical_Plasticity")

        """
        # Class attributes
        self.model_name = model_name
        self.model_type = None  # Str specifying the type of model
        ## Models
        self.full_models = None  # List of full models from CV
        self.partial_models = None  # Dict of lists for each feature of CV models
        ## Data and features
        self.X = None  # 2d array of x values
        self.X_corrected = None  # 2d array of x values corrected for nan values
        self.y = None  # 1d array of y values corresponding to the x values
        self.features = None  # List str of model features
        self.classes = None  # Dict of mapping class str to int labels
        ## Feature weights/importance
        self.full_model_feature_importance_train = (
            None  # List of training feature importances for each CV
        )
        self.full_model_feature_importance_test = (
            None  # List of testing feature importances for each CV
        )
        ## Model results
        self.full_model_train_score = None  # 1d array of each CV train results
        self.full_model_test_score = None  # 1d array of each CV test results
        self.shuff_model_train_score = None  # 2d array of each CV/shuff train results
        self.shuff_model_test_score = None  # 2d array of each CV/shuff test results
        self.partial_models_train_score = (
            None  # dict of 1d arrays of results for each feature
        )
        self.partial_models_test_score = (
            None  # dict of 1d arrays of results for each feature
        )
        self.full_model_confusion_matricies = (
            None  # list of 2d array of confusion matricies for each cv
        )
        self.shuff_model_confusion_matricies = (
            None  # list of 2d array of average confusion matricies for each shuff
        )

        ## Other parameters
        self.random_state = random.randint(0, 100)
        self.random_state = 78
        print(self.random_state)
        self.score_method = None
        self.shuff_iterations = None
        self.resample = None
        self.resample_num = 10
        self.cv_num = None
        self.C_param = None

        # Scorer options
        self.scorer_dict = {
            "accuracy": make_scorer(metrics.accuracy_score),
            "balanced_accuracy": make_scorer(metrics.balanced_accuracy_score),
            "average_precision": make_scorer(metrics.average_precision_score),
            "f1_micro": make_scorer(metrics.f1_score, average="micro"),
            "f1_macro": make_scorer(metrics.f1_score, average="macro"),
            "precision": make_scorer(metrics.precision_score),
            "recall": make_scorer(metrics.recall_score),
            "roc_auc_ovr": make_scorer(metrics.roc_auc_score, multi_class="ovr"),
            "roc_auc_ovo": make_scorer(metrics.roc_auc_score, multi_class="ovo"),
        }

    def train_full_model(
        self, x_values, y_values, classes, score_method, cv_num, model_type, C, resample
    ):
        """Method to train and evaluate the full model

        INPUT PARAMETERS
            x_values - dictionary of x_values, with each item corresponding to a feature

            y_values - np.array of the coded labels for the spines

            classes - dict mappint str to the int labels for the classes in y

            score_method - str specifying the score method to use

            cv_num - int specifying the number of cross validations to perform

            model_type - str specifying the model type to use

            C - float specifying the C parameter to use for model training

            resample - str specifying whether to resample the data to balance classes. Accepts
                    "under", "over", and None.

        """
        # Perform some data/feature organization
        self.model_type = model_type
        self.features = list(x_values.keys())
        self.score_method = score_method
        self.resample = resample
        self.cv_num = cv_num
        self.C_param = C
        X = np.array(list(x_values.values())).T
        self.X = X
        self.y = y_values
        self.classes = classes

        # Get the scorer
        scorer = self.scorer_dict[score_method]

        # Train and evaluate the model
        (
            X_corrected,
            models,
            feature_importance_train,
            feature_importance_test,
            train_scores,
            test_scores,
            confusion_matricies,
        ) = train_test_model(
            X=X,
            y=y_values,
            scorer=scorer,
            resample=resample,
            resample_num=self.resample_num,
            cv_num=cv_num,
            model_type=model_type,
            C=C,
            random_state=self.random_state,
        )

        # Store outputs
        self.X_corrected = X_corrected
        self.full_models = models
        self.full_model_feature_importance_train = feature_importance_train
        self.full_model_feature_importance_test = feature_importance_test
        self.full_model_train_score = train_scores
        self.full_model_test_score = test_scores
        self.full_model_confusion_matricies = confusion_matricies

    def train_shuffle_model(self, iterations=10):
        """Method to train and evaluate shuffled models to estimate chance

        INPUT PRAMATERS
            iterations - int specifying how many shuffling iterations to perform

        """
        # Make sure true full model has been fitted prior
        if self.full_models is None:
            return "Must train true full model first."
        self.shuff_iterations = iterations

        scorer = self.scorer_dict[self.score_method]

        # Set up outputs
        shuff_train_scores = []
        shuff_test_scores = []
        shuff_confusion_matricies = []

        X_data = self.X_corrected
        y_data = self.y

        # Iteratively fit and evaluate the model
        for i in range(iterations):
            print(f"PERFORMING SHUFFLE {i}")
            ## Shuffle the data
            shuff_x = copy.deepcopy(X_data)
            shuff_y = copy.deepcopy(y_data)
            np.random.shuffle(shuff_y)
            ## Train and evaluate the model
            ### Uses the same paramters used for full model
            (
                _,
                _,
                _,
                _,
                train_scores,
                test_scores,
                confusion_matricies,
            ) = train_test_model(
                X=shuff_x,
                y=shuff_y,
                scorer=scorer,
                resample=self.resample,
                resample_num=self.resample_num,
                cv_num=self.cv_num,
                model_type=self.model_type,
                C=self.C_param,
                random_state=self.random_state,
            )
            shuff_train_scores.append(train_scores)
            shuff_test_scores.append(test_scores)
            shuff_confusion_matricies.append(np.nanmean(confusion_matricies, axis=0))

        # Reshape the scores into 2d arrays and store
        self.shuff_model_train_score = np.vstack(shuff_train_scores)
        self.shuff_model_test_score = np.vstack(shuff_test_scores)
        self.shuff_model_confusion_matricies = shuff_confusion_matricies

    def plot_model_performance(
        self,
        iterations=10,
        color="mediumblue",
        figsize=(5, 5),
        save=False,
        save_path=None,
    ):
        """Method to compare the real full model to the shuffled model performance"""
        # Check the full model is trained and tested
        if self.full_models is None:
            return "Must train true full model first"
        # Perform training and testing on shuffled data if not already done
        if self.shuff_model_train_score is None:
            self.train_shuffle_model(iterations=iterations)

        # Construct the figure
        fig, axes = plt.subplot_mosaic("""ABC""", figsize=figsize)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        # Plot the confusion matricies first
        ## Real data
        real_confusion_matrix = np.nanmean(self.full_model_confusion_matricies, axis=0)
        v_max = np.max(real_confusion_matrix)
        v_min = np.min(real_confusion_matrix)
        real_confusion_disp = ConfusionMatrixDisplay(
            confusion_matrix=real_confusion_matrix,
            display_labels=self.full_models[0].classes_,
        )
        ## Shuff data
        shuff_confusion_matrix = np.nanmean(
            self.shuff_model_confusion_matricies, axis=0
        )
        shuff_confusion_disp = ConfusionMatrixDisplay(
            confusion_matrix=shuff_confusion_matrix,
            display_labels=self.full_models[0].classes_,
        )
        ## Plot the matricies
        real_confusion_disp.plot(
            include_values=True,
            cmap="plasma",
            ax=axes["A"],
            colorbar=True,
            im_kw={"vmax": v_max, "vmin": v_min},
        )
        shuff_confusion_disp.plot(
            include_values=True,
            cmap="plasma",
            ax=axes["B"],
            colorbar=True,
            im_kw={"vmax": v_max, "vmin": v_min},
        )
        axes["A"].set_title("Real data")
        axes["B"].set_title("Shuff data")

        # Plot the real vs shuffle model performance
        plot_swarm_bar_plot(
            data_dict={
                "real": self.full_model_test_score,
                "shuff": self.shuff_model_test_score.flatten(),
            },
            mean_type="mean",
            err_type="sem",
            figsize=figsize,
            title="Real vs. Shuffle",
            xtitle=None,
            ytitle=self.score_method,
            ylim=None,
            b_colors=[color, "darkgrey"],
            b_edgecolors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0.5,
            b_alpha=0.5,
            s_colors=[color, "darkgrey"],
            s_size=5,
            s_alpha=0.9,
            plot_ind=True,
            axis_width=1.5,
            minor_ticks="y",
            tick_len=3,
            ax=axes["C"],
            save=False,
            save_path=None,
        )

        fig.tight_layout()
        # Save section
        if save:
            if save_path is None:
                save_path = r"C:\users\Jake\Desktop\Figures"
            fname = os.path.join(save_path, "ML_Model_Performance")
            fig.savefig(fname + ".svg")

    def plot_feature_weights(self, figsize=(5, 5)):
        """Method to plot the feature weights of the full model"""
        # Get values needed for plotting
        features = self.features
        train_importance = np.vstack(self.full_model_feature_importance_train)
        test_importance = np.vstack(self.full_model_feature_importance_test)

        # Convert feature importances into dicts
        train_importance_dict = {}
        test_importance_dict = {}
        for i, feature in enumerate(features):
            train_importance_dict[feature] = train_importance[:, i]
            test_importance_dict[feature] = test_importance[:, i]

        # Construct the figure
        fig, axes = plt.subplot_mosaic("""AB""", figsize=figsize)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        ## Plot the train importances
        plot_swarm_bar_plot(
            data_dict=train_importance_dict,
            mean_type="mean",
            err_type="sem",
            figsize=figsize,
            title="Train Importances",
            xtitle="Features",
            ytitle="Feature importance",
            ylim=None,
            b_colors="black",
            b_edgecolors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0,
            b_alpha=0.5,
            s_colors="black",
            s_size=5,
            s_alpha=1,
            plot_ind=False,
            minor_ticks="y",
            x_rotation=90,
            ax=axes["A"],
            save=False,
            save_path=None,
        )

        ## Plot the test importances
        plot_swarm_bar_plot(
            data_dict=test_importance_dict,
            mean_type="mean",
            err_type="sem",
            figsize=figsize,
            title="Test Importances",
            xtitle="Features",
            ytitle="Feature importance",
            ylim=None,
            b_colors="black",
            b_edgecolors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0,
            b_alpha=0.5,
            s_colors="black",
            s_size=5,
            s_alpha=1,
            plot_ind=False,
            minor_ticks="y",
            x_rotation=90,
            ax=axes["B"],
            save=False,
            save_path=None,
        )

        fig.tight_layout()


############################## Independent Functions ################################
def train_test_model(
    X,
    y,
    scorer,
    resample=None,
    resample_num=1,
    cv_num=5,
    model_type="SVM_linear",
    C=0.1,
    random_state=8,
):
    """Function to train and test models in a cross-validated fashion. Different stages
    of preprocessing and testing are manually controlled to allow greater customization

    INPUT PARAMETERS
        x_values - dictionary of the x_values, with each item corresponding to a feature

        y_values - np.array of the coded labels for the spines

        scorer - scorer object to be used for evaluating the model performance

        resample - str specifying whether to resample the data to balance classes. Accepts
                    "under", "over", and None.

        resample_num - int specifying how many times to resample the data if performed

        cv_num - int specifying how many cross-validations to perform

        model_type - str specifying the type of model to use. Currently coded for
                    "SVM_linear", "SVM_rbf" and "Logistic"

        C - float specifying the degree of regularization for the models

        random_state - int specifying the random state for reproducibility

    OUTPUT PARAMETERS
        X_corrected - np.array of the corrected X values

        models - list of trained model objects

        feature_importance_train - list of the feature importance for each validation

        feature_importance_test - list of the feature importance for each validation

        train_scores - np.array of the individual train scores

        test_scores - np.array of the individual test scores

        confusion_matrices - list of confusion matrices for each validation

    """
    # Set up som final outputs
    models = []
    feature_importance_train = []
    feature_importance_test = []
    train_scores = []
    test_scores = []
    confusion_matrices = []
    # Set up some generators
    cv = StratifiedShuffleSplit(
        n_splits=cv_num, test_size=(1 / cv_num), random_state=random_state
    )
    scaler = MinMaxScaler()

    # First replace nan values in the x data using KNN Impute
    imputer = KNNImputer(n_neighbors=3, weights="distance", copy=True)
    X_corrected = imputer.fit_transform(X)

    # Set up resampling method
    if resample == "under":
        sampler = RandomUnderSampler(
            sampling_strategy="not minority", random_state=random_state
        )
    elif resample == "over":
        sampler = SMOTE(
            sampling_strategy="not majority", random_state=random_state, k_neighbors=3
        )
    else:
        sampler = None
        resample_num = 1

    for _ in range(resample_num):
        # resample the data
        if resample is not None:
            X_resample, y_resample = sampler.fit_resample(X_corrected, y)
        else:
            X_resample = X_corrected
            y_resample = y
        # Initialize model
        if model_type == "SVM_linear":
            model = svm.SVC(kernel="linear", class_weight="balanced", C=C)
        elif model_type == "SVM_rbf":
            model = svm.SVC(kernel="rbf", class_weight="balanced", C=C)
        elif model_type == "Logistic":
            model = linear_model.LogisticRegression(
                penalty="l2",
                C=C,
                class_weight="balanced",
                solver="saga",
                multi_class="multinomial",
                max_iter=1000,
            )
        else:
            return f"{model_type} model type is not currently supported"

        # Setup the pipeline
        pipe = Pipeline(steps=[("scaling", scaler), ("classify", model)])

        # Perform cross-validated training and testing
        cv_results = cross_validate(
            estimator=pipe,
            X=X_resample,
            y=y_resample,
            scoring=scorer,
            cv=cv,
            return_train_score=True,
            return_estimator=True,
        )
        # Get the splits
        splits = [
            (train, test)
            for i, (train, test) in enumerate(cv.split(X_resample, y_resample))
        ]

        ## Perform additional model evaluation
        for i, (estimator, test_score, train_score) in enumerate(
            zip(
                cv_results["estimator"],
                cv_results["test_score"],
                cv_results["train_score"],
            )
        ):
            ## Store some values
            models.append(estimator)
            test_scores.append(test_score)
            train_scores.append(train_score)
            ## Get current test train splits
            train_split = splits[i][0]
            test_split = splits[i][1]
            ## Generate confusion matrix
            predictions = estimator.predict(X_resample[test_split, :])
            cm = confusion_matrix(
                y_resample[test_split], predictions, labels=estimator.classes_
            )
            confusion_matrices.append(cm)
            ## Determine feature importance
            ### What the model learned using the trainig data
            train_importance = permutation_importance(
                estimator,
                X_resample[train_split, :],
                y_resample[train_split],
                n_repeats=20,
                random_state=random_state,
            )
            ### What explains model performance ont he test data
            test_importance = permutation_importance(
                estimator,
                X_resample[test_split, :],
                y_resample[test_split],
                n_repeats=20,
                random_state=random_state,
            )
            feature_importance_train.append(train_importance.importances_mean)
            feature_importance_test.append(test_importance.importances_mean)

    return (
        X_corrected,
        models,
        feature_importance_train,
        feature_importance_test,
        train_scores,
        test_scores,
        confusion_matrices,
    )


def organize_input_data(
    spine_data,
    local_data,
    global_data,
    spine_features,
    local_features,
    global_features,
    exclude="Shaft Spine",
    threshold=0.3,
):
    """Helper function to organize data into a dictionary that can be input to the SVM class

    INPUT PARAMETERS
        spine_data - spine_activity_dataclass object

        local_data - local_coactivity_dataclass object

        global_data - dendrite_coactivity_dataclass object

        spine_features - list of str specifying the features to grab from the
                        spine_data object

        local_features - list of str specifying the features to grab from the
                        local_data_object

        global_features - list of str specifying the features to grab from the
                            global data object

        exclude - str specifying spine type to exclude from analysis

        threshold - float or tuple of floats specifying the threshold cutoff for
                    classifying plasticity

    OUTPUT PARAMETERS
        x_values - dict containing the x_values for each spine for each features
                    in seperate dict items

        y_values - np.array with the coded identity for each spine

        class_codes - dict containing the spine type label (key) and its
                      coded integer (value)

    """
    # First set up the y values
    ## calculate relative volume and classify plasticity
    spine_volumes = spine_data.spine_volumes
    spine_flags = spine_data.spine_flags
    followup_volumes = spine_data.followup_volumes
    followup_flags = spine_data.followup_flags

    all_volumes = [spine_volumes, followup_volumes]
    all_flags = [spine_flags, followup_flags]

    delta_volume, spine_idxs = calculate_volume_change(
        all_volumes,
        all_flags,
        norm=False,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=False,
    )
    ## Code the spine classes
    class_codes = {"sLTP": 1, "sLTD": 2, "Stable": 3}
    y_values = np.zeros(len(spine_idxs))
    y_values[enlarged_spines] = 1
    y_values[shrunken_spines] = 2
    y_values[stable_spines] = 3
    y_values = y_values.astype(int)
    print(y_values)

    # Set up the x_values
    x_values = {}
    ## Spine-related variables
    if spine_features:
        for feature in spine_features:
            temp_data = getattr(spine_data, feature)
            ## Get values for only the present spines
            temp_data = d_utils.subselect_data_by_idxs(temp_data, spine_idxs)
            x_values[f"s_{feature}"] = temp_data
    ## Local related variables
    if local_features:
        for feature in local_features:
            ## Special cases for distance related values
            if "distance_coactivity_rate" in feature or "distrubtion" in feature:
                temp_data = getattr(local_data, feature)
                temp_data = d_utils.subselect_data_by_idxs(temp_data, spine_idxs)
                bins = getattr(local_data, "parameters")["position bins"]
                for i in range(temp_data.shape[0]):
                    x_values[f"l_{feature}_{bins[i]}"] = temp_data[i, :]
                continue
            if "local_dend_amplitude_dist" in feature:
                temp_data = getattr(local_data, feature)
                temp_data = d_utils.subselect_data_by_idxs(temp_data, spine_idxs)
                bins = getattr(local_data, "parameters")["dendrite position bins"]
                for i in range(temp_data.shape[0]):
                    x_values[f"l_{feature}_{bins[i]}"] = temp_data[i, :]
                continue
            temp_data = getattr(local_data, feature)
            temp_data = d_utils.subselect_data_by_idxs(temp_data, spine_idxs)
            x_values[f"l_{feature}"] = temp_data
    ## Global related variables
    if global_features:
        for feature in global_features:
            ## Not bothering with the distrubtions for now
            temp_data = getattr(global_data, feature)
            temp_data = d_utils.subselect_data_by_idxs(temp_data, spine_idxs)
            x_values[f"g_{feature}"] = temp_data

    return x_values, y_values, class_codes
