import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import Lab_Analyses.Utilities.data_utilities as d_utils
from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import load_spine_datasets
from Lab_Analyses.Utilities.movement_related_activity_v2 import (
    movement_related_activity,
)

sns.set_style("ticks")
