import os
import re
from dataclasses import dataclass
from itertools import compress

import numpy as np

from Lab_Analyses.Optogenetics.synaptic_opto_dataclass import Synaptic_Opto_Data
from Lab_Analyses.Optogenetics.synaptic_opto_responsive import synaptic_opto_responsive
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import refine_activity_timestamps
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def synaptic_opto_processing(
    mouse_id,
    fov_type="apical",
    save=False,
):
    """Function to determine if synapses are responsive to specific optogenetic stimulation
        Note: this does not currently allow incorporation of lever press data. Will need to
        rewrite with code that first aligns the imaging and behvaioral data

    INPUT PARAMETERS
        mouse_id - str specifying what the mouse's id is

        fov_type - str specifying whether to process apical or basal FOVs

        save - boolean specifying if the data is to be saved or not

    """
    # Constants
    ANALYSIS_WIN = (-1, 1)
    VIS_WIN = (-2, 3)

    print(
        f"----------------------------------------------------\nAnalyzing Mouse {mouse_id}"
    )
    initial_path = r"G:\Analyzed_data\individual"
    mouse_path = os.path.join(initial_path, mouse_id)
    imaging_path = os.path.join(mouse_path, "imaging")
    behavior_path = os.path.join(mouse_path, "behavior")

    # Identify the correct FOVs to process
    FOVs = next(os.walk(imaging_path))[1]
    FOVs = [x for x in FOVs if "FOV" in x]
    FOVs = [x for x in FOVs if fov_type in x]

    opto_data_list = []

    # Analyze each FOV seperately
    for FOV in FOVs:
        print(f"- Organizing {FOV} data")
        # Load in the appropriate data
        FOV_path = os.path.join(imaging_path, FOV)

        ## Get the induction session for the activity data
        session = next(os.walk(FOV_path))[1]
        session = [x for x in session if "Induction" in x][0]
        session_path = os.path.join(FOV_path, session)
        fnames = next(os.walk(session_path))[2]
        fnames = [x for x in fnames if "imaging_data" in x][0]
        activity_fname = os.path.join(session_path, fnames)
        activity_data = load_pickle([activity_fname])[0]

        ## Get the matching behavioral file
        day = re.search("[0-9]{6}", os.path.basename(activity_fname)).group()
        matched_b_path = os.path.join(behavior_path, day)
        b_fnames = next(os.walk(matched_b_path))[2]
        b_fname = [x for x in b_fnames if f"{FOV}_processed_lever_data" in x][0]
        b_fname = os.path.join(matched_b_path, b_fname)
        behavior_data = load_pickle([b_fname])[0]

        # Pull some important variables from the activity_data
        sampling_rate = activity_data.parameters["Sampling Rate"]
        spine_flags = activity_data.ROI_flags["Spine"]
        spine_positions = np.array(activity_data.ROI_positions["Spine"])
        spine_groupings = activity_data.parameters["Spine Groupings"]
        if len(spine_groupings) == 0:
            spine_groupings = list(range(activity_data.dFoF["Spine"].shape[1]))
        corrected_spine_volume = np.array(activity_data.corrected_spine_volume)
        dFoF = activity_data.dFoF["Spine"]
        processed_dFoF = activity_data.processed_dFoF["Spine"]
        spine_activity, spine_floored, _ = event_detection(
            processed_dFoF,
            threshold=2,
            lower_threshold=1,
            lower_limit=0.2,
            sampling_rate=sampling_rate,
            filt_poly=4,
            sec_smooth=1,
        )

        # Get the relevant behavioral data
        imaged_trials = behavior_data.imaged_trials == 1
        behavior_frames = list(compress(behavior_data.behavior_frames, imaged_trials))
        stims = []
        for i in behavior_frames:
            stims.append(i.states.iti2.astype(int))
        stim_len = int(np.nanmedian([x[1] - x[0] for x in stims]))
        ## Refine stim onsets to make sure they fit in the imaging sessions
        stims = refine_activity_timestamps(
            timestamps=stims,
            window=VIS_WIN,
            max_len=len(dFoF[:, 0]),
            sampling_rate=sampling_rate,
        )
        stims = [x[0] for x in stims]
        # Zscore some of the activity to use
        z_processed_dFoF = d_utils.z_score(processed_dFoF)

        # Get the activity around each stimulation for each spine
        stim_traces, _ = d_utils.get_trace_mean_sem(
            activity=z_processed_dFoF,
            ROI_ids=list(np.array(list(range(processed_dFoF.shape[1]))).astype(str)),
            timestamps=stims,
            window=VIS_WIN,
            sampling_rate=sampling_rate,
        )
        stim_traces = list(stim_traces.values())

        # Test significance
        diffs, pvalues, ranks, sigs = synaptic_opto_responsive(
            dFoF=z_processed_dFoF,
            timestamps=stims,
            window=ANALYSIS_WIN,
            sampling_rate=sampling_rate,
            smooth=True,
        )

        parameters = {
            "Sampling Rate": sampling_rate,
            "Analysis Window": ANALYSIS_WIN,
            "Visual Window": VIS_WIN,
            "FOV_type": fov_type,
        }

        # Assign dendrite id for each spine
        if type(spine_groupings[0]) != list:
            spine_groupings = [spine_groupings]
        spine_dendrite = np.zeros(z_processed_dFoF.shape[1]) * np.nan
        for i, spines in enumerate(spine_groupings):
            spine_dendrite[spines] = i

        # Store the data in a dataclass
        opto_data = Synaptic_Opto_Data(
            mouse_id=mouse_id,
            FOV=FOV,
            session=session,
            parameters=parameters,
            spine_flags=spine_flags,
            spine_dendrite=spine_dendrite,
            spine_positions=spine_positions,
            spine_volumes=corrected_spine_volume,
            spine_dFoF=dFoF,
            spine_processed_dFoF=processed_dFoF,
            spine_activity=spine_activity,
            spine_floored=spine_floored,
            spine_z_dFoF=z_processed_dFoF,
            stim_timestamps=stims,
            stim_len=stim_len,
            spine_diffs=diffs,
            spine_pvalues=pvalues,
            spine_ranks=ranks,
            responsive_spines=sigs,
            stim_traces=stim_traces,
        )
        opto_data_list.append(opto_data)
        # Save the data
        if save:
            opto_data.save()

    return opto_data_list
