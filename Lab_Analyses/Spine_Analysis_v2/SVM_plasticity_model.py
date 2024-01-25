import random

import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


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

    def train_model(self, x_values, y_values, score_method):
        """Method to train the model using grid search cross validation

        INPUT PARAMETERS
            x_values - dictionary of the x_values, with each item corresponding to a feature

            y_values - np.array of the coded labels for the spines

            score_method - str specifying the socring method to use

        """
        # Organize the data into an array
        ## Store feature names
        self.features = list(x_values.keys())
        self.score_method = score_method
        ## Convert x dict values into array
        X = np.array(list(x_values.values())).T
        ## Store x and y values
        self.X = X
        self.y = y_values

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
        svc = svm.SVC(kernel="linear")

        # Perform the gridsearch
        grid = GridSearchCV(
            estimator=svc,
            param_grid=search_params,
            scoring=scorer,
            n_jobs=2,
            refit=True,
            cv=cv,
            verbose=2,
            return_train_score=True,
        )
        clf = make_pipeline(MinMaxScaler(), grid)
        clf.fit(X=X, y=y_values)

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
        scorer = self.score_method

        # Set up cross-validation method
        cv = StratifiedShuffleSplit(
            n_splits=10, test_size=0.2, random_state=self.random_state
        )

        # Iteratively fit model for each shuffle iteration
        for i in range(iterations):
            print(f"PERFORMING SHUFFLE {i}")
            ## Shuffle data
            shuff_x = np.copy(self.X)
            shuff_y = np.copy(self.y)
            np.random.shuffle(shuff_y)
            ## Set up the model and cross validation
            svc = svm.SVD(kernel="linear", C=C)
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
