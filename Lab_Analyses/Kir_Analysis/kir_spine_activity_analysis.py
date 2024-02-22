import numpy as np

import Lab_Analyses.Utilities.data_utilities as d_utils
from Lab_Analyses.Kir_Analysis.kir_spine_activity_dataclass import (
    Grouped_Kir_Spine_Activity_Data,
    Kir_Spine_Activity_Data,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    load_spine_datasets,
    parse_movement_nonmovement_spines,
)
from Lab_Analyses.Utilities.movement_related_activity_v2 import (
    movement_related_activity,
)
from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def kir_spine_activity_analysis(
    mice_list,
    session,
    fov_type="apical",
    activity_window=(-2, 4),
    zscore=False,
    save_ind=False,
    save_grouped=False,
):
    """Function to handle the analysis of spine centric analyses of kir spine
    imaging datasets across all mice and FOVs

    INPUT PARAMETERS
        mice_list - list of str specifying all the mice to be analyzed

        session - str specifying the session to be analyzed

        fov_type - str specifying whether to analyze apical or basal FOVs

        activity_window - tuple specifying the window around which activity
                          should be analyzed in seconds

        zscore - boolean of whether or not to zscore the activity for analysis

        save_ind - boolean specifying whether to save data for each FOV

        save_grouped - boolean specifying whether or not to group all FOVs together
                        and save

    """
    # Analyze each mouse seperately
    dendrite_tracker = 0
    analyzed_data = []
    for mouse in mice_list:
        print("----------------------------------------")
        print(f"- Analyzing {mouse}")
        # Load the datasets
        datasets = load_spine_datasets(mouse, [session], fov_type)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
            print(f"-- {FOV}")
            data = dataset[session]

            # Pull relevant data
            ## Some parameters
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            zoom_factor = data.imaging_parameters["Zoom"]
            # Spine identifiers
            spine_groupings = data.spine_groupings
            spine_flags = data.spine_flags
            followup_flags = data.followup_flags
            # Spine volumes
            raw_spine_volume = data.corrected_spine_volume
            raw_followup_volume = data.corrected_followup_volume
            # Spine activity
            spine_activity = data.spine_GluSnFr_activity
            spine_dFoF = data.spine_GluSnFr_processed_dFoF
            ## Spine MRS information
            movement_spines = np.array(data.movement_spines)
            rwd_movement_spines = np.array(data.reward_movement_spines)
            (
                nonmovement_spines,
                nonrwd_movement_spines,
            ) = parse_movement_nonmovement_spines(movement_spines, rwd_movement_spines)
            ## Dendrite variables
            dendrite_number = np.zeros(spine_activity.shape[1]) * np.nan
            for grouping in spine_groupings:
                dendrite_number[grouping] = dendrite_tracker
            dendrite_tracker = dendrite_tracker + 1
            ## Behavioral data
            lever_active = data.lever_active
            lever_force = data.lever_force_smooth
            lever_active_rwd = data.rewarded_movement_binary
            lever_active_nonrwd = lever_active - lever_active_rwd

            # Zscore activity if specified
            if zscore:
                spine_dFoF = d_utils.zscore(spine_dFoF)

            # Get volumes in um
            pix_to_um = zoom_factor / 2
            spine_volumes = (np.sqrt(raw_spine_volume) / pix_to_um) ** 2
            followup_volumes = (np.sqrt(raw_followup_volume) / pix_to_um) ** 2

            # Get activity frequencies
            spine_activity_rate = d_utils.calculate_activity_event_rate(spine_activity)

            # Analyze activity during movements
            (
                spine_movement_traces,
                spine_movement_amplitudes,
                spine_movement_onsets,
            ) = movement_related_activity(
                lever_active=lever_active,
                activity=spine_activity,
                dFoF=spine_dFoF,
                norm=None,
                smooth=True,
                avg_window=None,
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            # Analyze activity during rewarded movements
            (
                spine_rwd_movement_traces,
                spine_rwd_movement_amplitudes,
                spine_rwd_movement_onsets,
            ) = movement_related_activity(
                lever_active=lever_active_rwd,
                activity=spine_activity,
                dFoF=spine_dFoF,
                norm=None,
                smooth=True,
                avg_window=None,
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )
            # Analyze activity during non-rewarded movements
            (
                spine_nonrwd_movement_traces,
                spine_nonrwd_movement_amplitudes,
                spine_nonrwd_movement_onsets,
            ) = movement_related_activity(
                lever_active=lever_active_nonrwd,
                activity=spine_activity,
                dFoF=spine_dFoF,
                norm=None,
                smooth=True,
                avg_window=None,
                sampling_rate=sampling_rate,
                activity_window=activity_window,
            )

            # Analyze movement encoding
            ## All movements
            (
                spine_movements,
                spine_movement_correlation,
                spine_movement_stereotypy,
                spine_movement_reliability,
                spine_movement_specificity,
                spine_LMP_reliability,
                spine_LMP_specificity,
                LMP_trace,
            ) = quantify_movement_quality(
                mouse,
                spine_activity,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            ## Rwd movements
            (
                spine_rwd_movements,
                spine_rwd_movement_correlation,
                spine_rwd_movement_stereotypy,
                spine_rwd_movement_reliability,
                spine_rwd_movement_specificity,
                _,
                _,
                _,
            ) = quantify_movement_quality(
                mouse,
                spine_activity,
                lever_active_rwd,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            spine_fraction_rwd_mvmts = [
                x.shape[0] / y.shape[0]
                for x, y in zip(spine_rwd_movements, spine_movements)
            ]

            LMP = [LMP_trace for i in range(spine_activity.shape[1])]
            learned_movement_pattern = np.stack(LMP).reshape(-1, 1)

            parameters = {
                "Sampling Rate": sampling_rate,
                "zscore": zscore,
                "Activity Window": activity_window,
                "FOV type": fov_type,
            }

            # Store the data
            spine_data = Kir_Spine_Activity_Data(
                mouse_id=mouse,
                FOV=FOV,
                session=session,
                parameters=parameters,
                spine_flags=spine_flags,
                followup_flags=followup_flags,
                spine_volumes=spine_volumes,
                followup_volumes=followup_volumes,
                dendrite_number=dendrite_number,
                movement_spines=movement_spines,
                nonmovement_spines=nonmovement_spines,
                rwd_movement_spines=rwd_movement_spines,
                nonrwd_movement_spines=nonrwd_movement_spines,
                spine_activity_rate=spine_activity_rate,
                spine_movement_traces=spine_movement_traces,
                spine_movement_amplitude=spine_movement_amplitudes,
                spine_movement_onset=spine_movement_onsets,
                spine_nonrwd_movement_traces=spine_nonrwd_movement_traces,
                spine_nonrwd_movement_amplitude=spine_nonrwd_movement_amplitudes,
                spine_nonrwd_movement_onset=spine_nonrwd_movement_onsets,
                spine_rwd_movement_traces=spine_rwd_movement_traces,
                spine_rwd_movement_amplitude=spine_rwd_movement_amplitudes,
                spine_rwd_movement_onset=spine_rwd_movement_onsets,
                learned_movement_pattern=learned_movement_pattern,
                spine_movements=spine_movements,
                spine_movement_correlation=spine_movement_correlation,
                spine_movement_stereotypy=spine_movement_stereotypy,
                spine_movement_reliability=spine_movement_reliability,
                spine_movement_specificity=spine_movement_specificity,
                spine_rwd_movements=spine_rwd_movements,
                spine_rwd_movement_correlation=spine_rwd_movement_correlation,
                spine_rwd_movement_stereotypy=spine_rwd_movement_stereotypy,
                spine_rwd_movement_reliability=spine_rwd_movement_reliability,
                spine_rwd_movement_specificity=spine_rwd_movement_specificity,
                spine_LMP_reliability=spine_LMP_reliability,
                spine_LMP_specificity=spine_LMP_specificity,
                spine_fraction_rwd_mvmts=spine_fraction_rwd_mvmts,
            )

            # Save individual data if specified
            if save_ind:
                spine_data.save()

            # Append to data list
            analyzed_data.append(spine_data)

    # Make grouped ata
    grouped_spine_data = Grouped_Kir_Spine_Activity_Data(analyzed_data)
    if save_grouped:
        grouped_spine_data.save()

    return grouped_spine_data
