import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

sns.set()
sns.set_style("ticks")

from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


class SVM_Plasticity_Model:
    """Class for the SVM model to predict synaptic plasticity based on different
    activity-based features"""

    def __init__(self, model_name):
        """Initialization method. Does not take inputs

        INPUT PARAMETERS
            model_name - str specifying the name of the model
                        (e.g., "Apical_Plasticity")

        """
        # Class attributes
        self.model_name = model_name
        ## Models
        self.full_model = None
        self.partial_models = None
        ## Data and features
        self.X = None
        self.y = None
        self.features = None
        self.classes = None
        self.full_model_parameters = None
        self.full_model_feature_weights = None
        self.partial_models_weights = None
        ## Results
        self.full_model_test_score = None
        self.full_model_train_score = None
        self.full_model_shuffled_test_score = None
        self.full_model_shuffled_train_score = None
        self.partial_models_test_scores = None
        self.partial_models_train_scores = None

        self.random_state = random.randint(0, 100)
        self.score_method = None
        self.shuff_iterations = None

        # Scorer options
        self.scorer_dict = {
            "accuracy": make_scorer(metrics.accuracy_score),
            "balanced_accuracy": make_scorer(metrics.balanced_accuracy_score),
            "average_precision": make_scorer(metrics.average_precision_score),
            "f1_micro": make_scorer(metrics.f1_score, average="micro"),
            "precision": make_scorer(metrics.precision_score),
            "recall": make_scorer(metrics.recall_score),
            "roc_auc_ovr": make_scorer(metrics.roc_auc_score, multi_class="ovr"),
            "roc_auc_ovo": make_scorer(metrics.roc_auc_score, multi_class="ovo"),
        }

    def train_model(self, x_values, y_values, classes, score_method):
        """Method to train the model using grid search cross validation

        INPUT PARAMETERS
            x_values - dictionary of the x_values, with each item corresponding to a feature

            y_values - np.array of the coded labels for the spines

            classes - dict mapping str to the int labels for the classes in y

            score_method - str specifying the socring method to use

        """
        # Organize the data into an array
        ## Store feature names
        self.features = list(x_values.keys())
        self.score_method = score_method
        ## Convert x dict values into array with each column a feature
        X = np.array(list(x_values.values())).T
        ## Store x and y values
        self.X = X
        self.y = y_values
        self.classes = classes

        # print(np.argwhere(np.isnan(X).any(axis=1)))
        # print(np.argwhere(np.isnan(X).any(axis=0)))
        X[np.isnan(X)] = 0

        # Preliminary setup for the grid search
        ## Get the scorer
        scorer = self.scorer_dict[score_method]
        ## Get cross validation method
        cv = StratifiedShuffleSplit(
            n_splits=10, test_size=0.2, random_state=self.random_state
        )
        ## Grid search parameters
        search_params = {"C": np.logspace(-2, 2, 5)}
        ## Specify the classifier
        svc = svm.SVC(kernel="linear", class_weight="balanced")

        # pipe = make_pipeline(
        #    ("scaling", MinMaxScaler()), ("C", "passthrough"), ("classify", svc)
        # )

        # Perform the gridsearch
        clf = GridSearchCV(
            estimator=svc,
            param_grid=search_params,
            scoring=scorer,
            n_jobs=2,
            refit=True,
            cv=cv,
            verbose=2,
            return_train_score=True,
        )
        # clf = make_pipeline(MinMaxScaler(), grid)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        clf.fit(X=X_scaled, y=y_values)

        # Store best model and related variables
        ## Store the estimator
        self.full_model = clf.best_estimator_
        self.full_model_parameters = clf.best_params_
        self.full_model_feature_weights = clf.best_estimator_.coef_
        ## Find best scores
        self.full_model_test_score = clf.best_score_
        self.full_model_train_score = clf.cv_results_["mean_train_score"][
            clf.best_index_
        ]

    def shuffle_classification(self, iterations=10):
        """Method to perform classification on shuffled values to estimate chance

        INPUT PARAMETERS
            iterations - int specifying how many rounds of shuffling to perform.
                        Default is 10.

        """
        # Make sure true model has been fitted prior
        if self.full_model is None:
            return "Must train true model first."
        # Set up dict to variables to store values
        self.shuff_iterations = iterations
        shuff_train_score = []
        shuff_test_score = []

        # Get true model parameters and scorer
        C = self.full_model_parameters["C"]
        scorer = self.scorer_dict[self.score_method]

        # Set up cross-validation method
        cv = StratifiedShuffleSplit(
            n_splits=10, test_size=0.2, random_state=self.random_state
        )

        # Iteratively fit model for each shuffle iteration
        for i in range(iterations):
            print(f"PERFORMING SHUFFLE {i}")
            ## Shuffle data
            shuff_x = copy.deepcopy(self.X)
            shuff_y = copy.deepcopy(self.y)
            np.random.shuffle(shuff_y)
            ## Set up the model and cross validation
            svc = svm.SVC(kernel="linear", C=C)
            clf = make_pipeline(MinMaxScaler(), svc)
            results = cross_validate(
                clf,
                X=shuff_x,
                y=shuff_y,
                cv=cv,
                scoring=scorer,
                return_train_score=True,
            )
            # Store average cross validated scored
            shuff_train_score.append(np.nanmean(results["train_score"]))
            shuff_test_score.append(np.nanmean(results["test_score"]))

        self.full_model_shuffled_train_score = shuff_train_score
        self.full_model_shuffled_test_score = shuff_test_score

    def plot_model_performance(self, iterations=10, color="mediumblue", figsize=(5, 4)):
        """Method to compare the real full model vs shuffled model performance"""
        # Make sure true model has been fitted prior
        if self.full_model is None:
            return "Must train true model first."
        # Perform training on shuffled data if not already done
        if self.full_model_shuffled_test_score is None:
            self.shuffle_classification(iterations=iterations)

        fig, axes = plt.subplot_mosaic("""AB""", figsize=figsize)

        # Plot confusion matrix
        ## Generate some splits
        cv = StratifiedShuffleSplit(
            n_splits=10, test_size=0.2, random_state=self.random_state
        )
        ## Get the average confusion matrix
        confusion_mat = None
        for i, (_, t_split) in enumerate(cv.split(self.X, self.y)):
            predictions = self.full_model.predict(self.X[t_split, :])
            cm = confusion_matrix(
                self.y[t_split], predictions, labels=self.full_model.classes_
            )
            if i == 0:
                confusion_mat = cm
            else:
                confusion_mat = confusion_mat + cm
        confusion_mat = confusion_mat / 10

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat, display_labels=self.full_model.classes_
        )
        disp.plot(include_values=True, cmap="plasma", ax=axes["A"], colorbar=True)

        # Plot the Real vs Shuffle
        plot_swarm_bar_plot(
            data_dict={
                "real": self.full_model_test_score,
                "shuff": self.full_model_shuffled_test_score,
            },
            mean_type="mean",
            err_type="sem",
            figsize=figsize,
            title="Real vs. Shuffle",
            xtitle=None,
            ytitle=self.score_method,
            ylim=None,
            b_colors=[color, "silver"],
            b_edgecolors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0.5,
            b_alpha=0.9,
            s_colors=[color, "silver"],
            s_size=5,
            s_alpha=1,
            plot_ind=False,
            axis_width=1,
            minor_ticks="y",
            tick_len=3,
            ax=axes["B"],
            save=False,
            save_path=None,
        )

        fig.tight_layout()

    def plot_feature_weights(self, absolute=False, figsize=(4, 5)):
        """Method to plot the feature weights of the full model"""
        # Get the values needed for plotting
        classes = list(self.classes.keys())
        features = self.features
        if absolute:
            weights = np.absolute(self.full_model_feature_weights)
            ytitle = "Absolute weight"
        else:
            weights = self.full_model_feature_weights
            ytitle = "Weight"

        class_comparisions = [
            f"{classes[0]} vs. {classes[1]}",
            f"{classes[0]} vs. {classes[2]}",
            f"{classes[1]} vs. {classes[2]}",
        ]

        # Organize the data for plotting
        class_feature_dict = {}
        for i in range(weights.shape[0]):
            class_feature_dict[class_comparisions[i]] = weights[i, :]
        mean_feature_dict = {}
        for i in range(weights.shape[1]):
            mean_feature_dict[features[i]] = np.nanmean(weights[:, i])

        # Perpare for plotting
        fig, axes = plt.subplot_mosaic("""AB""", figsize=figsize)

        x = np.arange(len(features))
        width = 0.25

        ## Plot individual comparisons
        for mult, (comp, data) in enumerate(class_feature_dict.items()):
            offset = width * mult
            axes["A"].bar(x + offset, data, width, label=comp)
        axes["A"].set_ylabel(ytitle)
        axes["A"].set_title("One vs One Feature Weights")
        axes["A"].set_xlabel("Features")
        axes["A"].set_xticks(ticks=x + width, labels=features, rotation="vertical")
        axes["A"].legend(loc="upper left", ncols=len(classes))

        ## Plot the mean weights
        plot_swarm_bar_plot(
            data_dict=mean_feature_dict,
            mean_type="mean",
            err_type="sem",
            figsize=(5, 5),
            title="Average Feature Weights",
            xtitle="Features",
            ytitle=ytitle,
            ylim=None,
            b_colors="black",
            b_edgecolors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0,
            b_alpha=1,
            s_colors="black",
            s_size=5,
            s_alpha=1,
            plot_ind=False,
            axis_width=1.5,
            minor_ticks=None,
            tick_len=3,
            ax=axes["B"],
            save=False,
            save_path=None,
        )

        fig.tight_layout()

    def score_new_data(self, new_x, new_y, colors, figsize=(5, 4)):
        """Method to score new data

        INPUT PARAMETERS
            new_x - dictionary of the x_values, with each item corresponding to a feature

            new_y - np.array of the coded labels for the spines

        """
        # Make sure true model has been fitted prior
        if self.full_model is None:
            return "Must train true model first."
        # Get the scorer
        scorer = self.scorer_dict[self.score_method]

        # Get predictions of the new data
        pred_y = self.full_model.predict(new_x)

        # Score the performance
        score = scorer(y_true=new_y, y_pred=pred_y)

        # Plot the performance vs the full real model
        plot_swarm_bar_plot(
            data_dict={"original": self.full_model_test_score, "new": score},
            mean_type="mean",
            err_type="sem",
            figsize=figsize,
            title=None,
            xtitle=None,
            ytitle=self.score_method,
            ylim=None,
            b_colors=colors,
            b_edge_colors="black",
            b_err_colors="black",
            b_width=0.5,
            b_linewidth=0.5,
            b_alpha=0.9,
            s_colors=colors,
            s_size=5,
            s_alpha=1,
            plot_ind=False,
            axis_width=1,
            minor_ticks="y",
            tick_len=3,
            ax=None,
            save=False,
            save_path=None,
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
