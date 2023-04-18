import numpy as np

from Lab_Analyses.Spine_Analysis_v2.dendritic_coactivity_dataclass import (
    Dendritic_Coactivity_Data,
    Grouped_Dendritic_Coactivity_Data,
)
from Lab_Analyses.Spine_Analysis_v2.nearby_coactive_spine_activity import (
    nearby_coactive_spine_activity,
)
from Lab_Analyses.Spine_Analysis_v2.noncoactive_dendrite_analysis import (
    noncoactive_dendrite_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.spine_dendrite_event_analysis import (
    spine_dendrite_event_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    load_spine_datasets,
    parse_movement_nonmovement_spines,
)
from Lab_Analyses.Spine_Analysis_v2.spine_volume_normalization import (
    load_norm_constants,
)
from Lab_Analyses.Spine_Analysis_v2.variable_distance_dependence import (
    variable_distance_dependence,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def dendritic_coactivity_analysis(
    mice_list,
    session,
    fov_type="apical",
    activity_window=(-2, 4),
    cluster_dist=5,
    zscore=False,
    volume_norm=True,
    partners=None,
    movement_period=None,
    extend=None,
    save_ind=False,
    save_grouped=False,
):
    """Function to handle the analysis of dendritic coactivity with spines
    
        INPUT PARAMATERS
            mice_list - list of str specifying all the mice to be analyzed
            
            session - str specifying the session to be analyzed
            
            fov_type - str specifying whether to analyze apical or basal FOVs
            
            activity_window - tuple specifying the window around which the activity
                              should be analyzed (sec)
            
            cluster_dist - int/float specifying the distance to be considered local
            
            zscore - boolean of whether or not to zscore the activity for analysis
            
            volume_norm - boolean of whether or not to normalize activity by spine
                          volume
            
            partners - str specifying whether or not to constrain analysis to only 
                        'MRS' or 'nMRS' partners
            
            movement_period - str specifying whether or not to constrain analysis to
                             specific movement periods
            
            extend - float specifying the duration in seconds to extend the window for 
                     considering coactivity. Default is none for no extension
            
            save_ind - boolean specifying whetehr to save the data for each FOV
            
            save_grouped - boolean specifying whether to save all of the FOVs grouped together
            
    """
    # Get normalization constants
    if volume_norm:
        try:
            glu_constants = load_norm_constants(fov_type, zscore, "GluSnFr")
        except FileNotFoundError:
            return "GluSnFr constants have not been generated!!!"
        try:
            ca_constants = load_norm_constants(fov_type, zscore, "Calcium")
        except FileNotFoundError:
            return "Calcium constants have not been generated!!!"

    # Analyze each mouse seperately
    analyzed_data = []
    for mouse in mice_list:
        print("----------------------------------------")
        print(f"- Analyzing {mouse}")
        # Load the datasets
        datasets = load_spine_datasets(mouse, session, fov_type)

        # Analyze each FOV seperately
        for FOV, dataset in datasets.items():
            print(f"-- {FOV}")
            data = dataset[session]
            if volume_norm:
                curr_glu_constants = glu_constants[mouse][FOV]
                curr_ca_constants = ca_constants[mouse][FOV]
                all_constants = (curr_glu_constants, curr_ca_constants)
            else:
                curr_glu_constants = None
                curr_ca_constants = None
                all_constants = None

            # Pull relevant data
            ## Some parameters
            sampling_rate = int(data.imaging_parameters["Sampling Rate"])
            zoom_factor = data.imaging_parameters["Zoom"]
            ## Spine identifiers
            spine_groupings = data.spine_groupings
            spine_flags = data.spine_flags
            followup_flags = data.followup_flags
            spine_positions = data.spine_positions
            ## Spine Volumes
            raw_spine_volume = data.corrected_spine_volume
            raw_followup_volume = data.corrected_followup_volume
            ## Spine activity
            spine_activity = data.spine_GluSnFr_activity
            spine_dFoF = data.spine_GluSnFr_processed_dFoF
            spine_calcium_dFoF = data.spine_calcium_processed_dFoF
            ## Dendrite activity
            dendrite_activity = data.dendrite_calcium_activity
            dendrite_dFoF = data.dendrite_calcium_processed_dFoF
            ## Movement-related information
            movement_spines = np.array(data.movement_spines)
            rwd_movement_spines = np.array(data.rwd_movement_spines)
            (
                nonmovement_spines,
                nonrwd_movement_spines,
            ) = parse_movement_nonmovement_spines(movement_spines, rwd_movement_spines)
            movement_dendrites = np.array(data.movement_dendrites)
            rwd_movement_dendrites = np.array(data.rwd_movement_dendrites)
            (
                nonmovement_dendrites,
                nonrwd_movement_dendrites,
            ) = parse_movement_nonmovement_spines(
                movement_dendrites, rwd_movement_dendrites
            )
            ## Behavioral data
            lever_active = data.lever_active
            lever_force = data.lever_force
            lever_active_rwd = data.rewarded_movement_binary
            lever_inactive = np.absolute(lever_active - 1)

            # zscore activity if specified
            if zscore:
                spine_dFoF = d_utils.z_score(spine_dFoF)
                spine_calcium_dFoF = d_utils.z_score(spine_calcium_dFoF)
                dendrite_dFoF = d_utils.z_score(dendrite_dFoF)

            # Get volumes in um
            pix_to_um = zoom_factor / 2
            spine_volumes = (np.sqrt(raw_spine_volume) / pix_to_um) ** 2
            followup_volumes = (np.sqrt(raw_followup_volume) / pix_to_um) ** 2

            # Sort out partner list if specified
            if partners == "MRS":
                partner_list = movement_spines
            elif partners == "nMRS":
                partner_list = nonmovement_spines
            elif partners == "rMRS":
                partner_list = rwd_movement_spines
            else:
                partner_list = None

            # Sort out movement period
            if movement_period == "movement":
                constrain_matrix = lever_active
            elif movement_period == "nonmovement":
                constrain_matrix = lever_inactive
            elif movement_period == "rewarded movement":
                constrain_matrix = lever_active_rwd
            else:
                constrain_matrix = None

            # Analyze the spine-dendrite coactivity rates and events
            print(f"---- Assessing spine-dendrite coactivity")
            ## All coactive events
            (
                nearby_spine_idxs,
                all_coactive_binary,
                all_dendrite_coactive_event_num,
                all_dendrite_coactivity_rate,
                all_dendrite_coactivity_rate_norm,
                all_shuff_dendrite_coactivity_rate,
                all_shuff_dendrite_coactivity_rate_norm,
                all_above_chance_coactivity,
                all_above_chance_coactivity_norm,
                all_fraction_dend_coactive,
                all_fraction_spine_coactive,
                all_spine_coactive_amplitude,
                all_spine_coactive_calcium_amplitude,
                all_dendrite_coactive_amplitude,
                all_relative_onsets,
                all_onset_jitter,
                all_spine_coactive_traces,
                all_spine_coactive_calcium_traces,
                all_dendrite_coactive_traces,
            ) = spine_dendrite_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                dendrite_activity,
                dendrite_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=constrain_matrix,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
                extend=extend,
                activity_type="all",
                iterations=1000,
            )
            ## Events with local coactivity
            (
                _,
                conj_coactive_binary,
                conj_dendrite_coactive_event_num,
                conj_dendrite_coactivity_rate,
                conj_dendrite_coactivity_rate_norm,
                conj_shuff_dendrite_coactivity_rate,
                conj_shuff_dendrite_coactivity_rate_norm,
                conj_above_chance_coactivity,
                conj_above_chance_coactivity_norm,
                conj_fraction_dend_coactive,
                conj_fraction_spine_coactive,
                conj_spine_coactive_amplitude,
                conj_spine_coactive_calcium_amplitude,
                conj_dendrite_coactive_amplitude,
                conj_relative_onsets,
                conj_onset_jitter,
                conj_spine_coactive_traces,
                conj_spine_coactive_calcium_traces,
                conj_dendrite_coactive_traces,
            ) = spine_dendrite_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                dendrite_activity,
                dendrite_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=constrain_matrix,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
                extend=extend,
                activity_type="local",
                iterations=1000,
            )
            ## Events without any local coactivity
            (
                _,
                nonconj_coactive_binary,
                nonconj_dendrite_coactive_event_num,
                nonconj_dendrite_coactivity_rate,
                nonconj_dendrite_coactivity_rate_norm,
                nonconj_shuff_dendrite_coactivity_rate,
                nonconj_shuff_dendrite_coactivity_rate_norm,
                nonconj_above_chance_coactivity,
                nonconj_above_chance_coactivity_norm,
                nonconj_fraction_dend_coactive,
                nonconj_fraction_spine_coactive,
                nonconj_spine_coactive_amplitude,
                nonconj_spine_coactive_calcium_amplitude,
                nonconj_dendrite_coactive_amplitude,
                nonconj_relative_onsets,
                nonconj_onset_jitter,
                nonconj_spine_coactive_traces,
                nonconj_spine_coactive_calcium_traces,
                nonconj_dendrite_coactive_traces,
            ) = spine_dendrite_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                dendrite_activity,
                dendrite_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=constrain_matrix,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
                extend=extend,
                activity_type="no local",
                iterations=1000,
            )

            # Calulate fraction of coactivity with local coactivity
            fraction_conj_events = (
                conj_dendrite_coactive_event_num / all_dendrite_coactive_event_num
            )

            # Assess nearby spine activity during conj coactivity events
            print(f"Assessing properties of nearby spines")
            (
                conj_coactive_spine_num,
                conj_nearby_coactive_spine_amplitude,
                conj_nearby_coactive_spine_calcium,
                conj_nearby_spine_onset,
                conj_nearby_spine_onset_jitter,
                conj_nearby_coactive_spine_traces,
                conj_nearby_coactive_spine_calcium_traces,
            ) = nearby_coactive_spine_activity(
                nearby_spine_idxs,
                conj_coactive_binary,
                spine_flags,
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                None,
                all_constants,
                activity_window,
                sampling_rate,
            )

            # Assess distribution of coactivity rates
            ## Coactivity rate
            (
                avg_nearby_spine_coactivity_rate,
                shuff_nearby_spine_coactivity_rate,
                coactivity_rate_distribution,
            ) = variable_distance_dependence(
                all_dendrite_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_nearby_spine_coactivity_rate = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        all_dendrite_coactivity_rate, avg_nearby_spine_coactivity_rate
                    )
                ]
            )
            ## Coactivity rate norm
            (
                avg_nearby_spine_coactivity_rate_norm,
                shuff_nearby_spine_coactivity_rate_norm,
                coactivity_rate_norm_distribution,
            ) = variable_distance_dependence(
                all_dendrite_coactivity_rate_norm,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_nearby_spine_coactivity_rate_norm = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        all_dendrite_coactivity_rate_norm,
                        avg_nearby_spine_coactivity_rate_norm,
                    )
                ]
            )
            ## Conj coactivity rate
            (
                avg_nearby_spine_conj_rate,
                shuff_nearby_spine_conj_rate,
                conj_coactivity_rate_distribution,
            ) = variable_distance_dependence(
                conj_dendrite_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_nearby_spine_conj_rate = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        conj_dendrite_coactivity_rate, avg_nearby_spine_conj_rate,
                    )
                ]
            )
            ## Conj coactivity rate norm
            (
                avg_nearby_spine_conj_rate_norm,
                shuff_nearby_spine_conj_rate_norm,
                conj_coactivity_rate_norm_distribution,
            ) = variable_distance_dependence(
                conj_dendrite_coactivity_rate_norm,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_nearby_spine_conj_rate_norm = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        conj_dendrite_coactivity_rate_norm,
                        avg_nearby_spine_conj_rate_norm,
                    )
                ]
            )
            ## Spine Fraction coactive
            (
                avg_nearby_spine_fraction,
                shuff_nearby_spine_fraction,
                spine_fraction_coactive_distribution,
            ) = variable_distance_dependence(
                all_fraction_spine_coactive,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_spine_fraction = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        all_fraction_spine_coactive, avg_nearby_spine_fraction
                    )
                ]
            )
            ## Dendrite fraction coactive
            (
                avg_nearby_dend_fraction,
                shuff_nearby_dend_fraction,
                dend_fraction_coactive_distribution,
            ) = variable_distance_dependence(
                all_fraction_dend_coactive,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_dend_fraction = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        all_fraction_dend_coactive, avg_nearby_dend_fraction
                    )
                ]
            )
            ## Conj spine fraction coactive
            (
                conj_avg_nearby_spine_fraction,
                conj_shuff_nearby_spine_fraction,
                conj_spine_fraction_coactive_distribution,
            ) = variable_distance_dependence(
                conj_fraction_spine_coactive,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_conj_spine_fraction = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        conj_fraction_spine_coactive, conj_avg_nearby_spine_fraction
                    )
                ]
            )
            ## Conj dend fraction coactive
            (
                conj_avg_nearby_dend_fraction,
                conj_shuff_nearby_dend_fraction,
                conj_dend_fraction_coactive_distribution,
            ) = variable_distance_dependence(
                conj_fraction_dend_coactive,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_conj_dend_fraction = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        conj_fraction_dend_coactive, conj_avg_nearby_dend_fraction
                    )
                ]
            )
            ## Spine relative onsets
            (
                avg_nearby_relative_onset,
                shuff_nearby_relative_onset,
                relative_onset_distribution,
            ) = variable_distance_dependence(
                all_relative_onsets,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_nearby_relative_onset = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(all_relative_onsets, avg_nearby_relative_onset)
                ]
            )
            ## Conj spine relative onsets
            (
                conj_avg_nearby_relative_onset,
                conj_shuff_nearby_relative_onset,
                conj_relative_onset_distribution,
            ) = variable_distance_dependence(
                conj_relative_onsets,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=1000,
            )
            rel_conj_nearby_relative_onset = np.array(
                [
                    (x - y) / (x + y)
                    for x, y in zip(
                        conj_relative_onsets, conj_avg_nearby_relative_onset
                    )
                ]
            )

            # Assess spine calcium when not coactive
            (
                noncoactive_spine_calcium_amplitude,
                noncoactive_spine_calcium_traces,
                conj_fraction_participating,
                nonparticipating_spine_calcium_amplitude,
                nonparticipating_spine_calcium_traces,
            ) = noncoactive_dendrite_analysis(
                all_coactive_binary,
                spine_activity,
                spine_calcium_dFoF,
                dendrite_activity,
                nearby_spine_idxs,
                spine_flags,
                activity_window,
                constrain_matrix=constrain_matrix,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )

            # Assess movement encoding
            ## All coactive events
            (
                all_coactive_movement_correlation,
                all_coactive_movement_stereotypy,
                all_coactive_movement_reliability,
                all_coactive_movement_specificity,
                all_coactive_LMP_reliability,
                all_coactive_LMP_specificity,
                learned_movement_pattern,
            ) = quantify_movement_quality(
                mouse,
                all_coactive_binary,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            ## Conj coactive events
            (
                conj_movement_correlation,
                conj_movement_stereotypy,
                conj_movement_reliability,
                conj_movement_specificity,
                conj_LMP_reliability,
                conj_LMP_specificity,
                _,
            ) = quantify_movement_quality(
                mouse,
                conj_coactive_binary,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            (
                nonconj_movement_correlation,
                nonconj_movement_stereotypy,
                nonconj_movement_reliability,
                nonconj_movement_specificity,
                nonconj_LMP_reliability,
                nonconj_LMP_specificity,
                _,
            ) = quantify_movement_quality(
                mouse,
                nonconj_coactive_binary,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )

            LMP = [learned_movement_pattern for i in range(spine_activity.shape[1])]
            learned_movement_pattern = np.stack(LMP).reshape(-1, 1)

            parameters = {
                "Sampling Rate": sampling_rate,
                "zscore": zscore,
                "Activity Window": activity_window,
                "Volume Norm": volume_norm,
                "FOV type": fov_type,
                "cluster dist": cluster_dist,
                "partners": partners,
                "movement period": movement_period,
                "extended": extend,
            }

            # Store data in dataclass
            dendritic_coactivity_data = Dendritic_Coactivity_Data(
                mouse_id=mouse,
                FOV=FOV,
                session=session,
                parameters=parameters,
                spine_flags=spine_flags,
                followup_flags=followup_flags,
                spine_volumes=spine_volumes,
                followup_volumes=followup_volumes,
                movement_spines=movement_spines,
                nonmovement_spines=nonmovement_spines,
                rwd_movement_spines=rwd_movement_spines,
                nonrwd_movement_spines=nonrwd_movement_spines,
                movement_dendrites=movement_dendrites,
                nonmovement_dendrites=nonmovement_dendrites,
                rwd_movement_dendrites=rwd_movement_dendrites,
                nonrwd_movement_dendrites=nonrwd_movement_dendrites,
                all_dendrite_coactivity_rate=all_dendrite_coactivity_rate,
                all_dendrite_coactivity_rate_norm=all_dendrite_coactivity_rate_norm,
                all_shuff_dendrite_coactivity_rate=all_shuff_dendrite_coactivity_rate,
                all_shuff_dendrite_coactivity_rate_norm=all_shuff_dendrite_coactivity_rate_norm,
                all_above_chance_coactivity=all_above_chance_coactivity,
                all_above_chance_coactivity_norm=all_above_chance_coactivity_norm,
                all_fraction_dendrite_coactive=all_fraction_dend_coactive,
                all_fraction_spine_coactive=all_fraction_spine_coactive,
                all_spine_coactive_amplitude=all_spine_coactive_amplitude,
                all_spine_coactive_calcium_amplitude=all_spine_coactive_calcium_amplitude,
                all_dendrite_coactive_amplitude=all_dendrite_coactive_amplitude,
                all_relative_onsets=all_relative_onsets,
                all_onset_jitter=all_onset_jitter,
                all_spine_coactive_traces=all_spine_coactive_traces,
                all_spine_coactive_calcium_traces=all_spine_coactive_calcium_traces,
                all_dendrite_coactive_traces=all_dendrite_coactive_traces,
                conj_dendrite_coactivity_rate=conj_dendrite_coactivity_rate,
                conj_dendrite_coactivity_rate_norm=conj_dendrite_coactivity_rate_norm,
                conj_shuff_dendrite_coactivity_rate=conj_shuff_dendrite_coactivity_rate,
                conj_shuff_dendrite_coactivity_rate_norm=conj_shuff_dendrite_coactivity_rate_norm,
                conj_above_chance_coactivity=conj_above_chance_coactivity,
                conj_above_chance_coactivity_norm=conj_above_chance_coactivity_norm,
                conj_fraction_dendrite_coactive=conj_fraction_dend_coactive,
                conj_fraction_spine_coactive=conj_fraction_spine_coactive,
                conj_spine_coactive_amplitude=conj_spine_coactive_amplitude,
                conj_spine_coactive_calcium_amplitude=conj_spine_coactive_calcium_amplitude,
                conj_dendrite_coactive_amplitude=conj_dendrite_coactive_amplitude,
                conj_relative_onsets=conj_relative_onsets,
                conj_onset_jitter=conj_onset_jitter,
                conj_spine_coactive_traces=conj_spine_coactive_traces,
                conj_spine_coactive_calcium_traces=conj_spine_coactive_calcium_traces,
                conj_dendrite_coactive_traces=conj_dendrite_coactive_traces,
                nonconj_dendrite_coactivity_rate=nonconj_dendrite_coactivity_rate,
                nonconj_dendrite_coactivity_rate_norm=nonconj_dendrite_coactivity_rate_norm,
                nonconj_shuff_dendrite_coactivity_rate=nonconj_shuff_dendrite_coactivity_rate,
                nonconj_shuff_dendrite_coactivity_rate_norm=nonconj_shuff_dendrite_coactivity_rate_norm,
                nonconj_above_chance_coactivity=nonconj_above_chance_coactivity,
                nonconj_above_chance_coactivity_norm=nonconj_above_chance_coactivity_norm,
                nonconj_fraction_dendrite_coactive=nonconj_fraction_dend_coactive,
                nonconj_fraction_spine_coactive=nonconj_fraction_spine_coactive,
                nonconj_spine_coactive_amplitude=nonconj_spine_coactive_amplitude,
                nonconj_spine_coactive_calcium_amplitude=nonconj_spine_coactive_calcium_amplitude,
                nonconj_dendrite_coactive_amplitude=nonconj_dendrite_coactive_amplitude,
                nonconj_relative_onsets=nonconj_relative_onsets,
                nonconj_onset_jitter=nonconj_onset_jitter,
                nonconj_spine_coactive_traces=nonconj_spine_coactive_traces,
                nonconj_spine_coactive_calcium_traces=nonconj_spine_coactive_calcium_traces,
                nonconj_dendrite_coactive_traces=nonconj_dendrite_coactive_traces,
                fraction_conj_events=fraction_conj_events,
                conj_coactive_spine_num=conj_coactive_spine_num,
                conj_nearby_coactive_spine_amplitude=conj_nearby_coactive_spine_amplitude,
                conj_nearby_coactive_spine_calcium=conj_nearby_coactive_spine_calcium,
                conj_nearby_spine_onset=conj_nearby_spine_onset,
                conj_nearby_spine_onset_jitter=conj_nearby_spine_onset_jitter,
                conj_nearby_coactive_spine_traces=conj_nearby_coactive_spine_traces,
                conj_nearby_coactive_spine_calcium_traces=conj_nearby_coactive_spine_calcium_traces,
                avg_nearby_spine_coactivity_rate=avg_nearby_spine_coactivity_rate,
                shuff_nearby_spine_coactivity_rate=shuff_nearby_spine_coactivity_rate,
                cocativity_rate_distribution=coactivity_rate_distribution,
                rel_nearby_spine_coactivity_rate=rel_nearby_spine_coactivity_rate,
                avg_nearby_spine_coactivity_rate_norm=avg_nearby_spine_coactivity_rate_norm,
                shuff_nearby_spine_coactivity_rate_norm=shuff_nearby_spine_coactivity_rate_norm,
                coactivity_rate_norm_distribution=coactivity_rate_norm_distribution,
                rel_nearby_spine_coactivity_rate=rel_nearby_spine_coactivity_rate,
                avg_nearby_spine_conj_rate=avg_nearby_spine_conj_rate,
                shuff_nearby_spine_conj_rate=shuff_nearby_spine_conj_rate,
                conj_coactivity_rate_distribution=conj_coactivity_rate_distribution,
                rel_nearby_spine_conj_rate=rel_nearby_spine_conj_rate,
                avg_nearby_spine_conj_rate_norm=avg_nearby_spine_conj_rate_norm,
                shuff_nearby_spine_conj_rate_norm=shuff_nearby_spine_conj_rate_norm,
                conj_coactivity_rate_norm_distrubtion=conj_coactivity_rate_norm_distribution,
                rel_nearby_spine_conj_rate_norm=rel_nearby_spine_conj_rate_norm,
                avg_nearby_spine_fraction=avg_nearby_spine_fraction,
                shuff_nearby_spine_fraction=shuff_nearby_spine_fraction,
                spine_fraction_coactive_distribution=spine_fraction_coactive_distribution,
                rel_spine_fraction=rel_spine_fraction,
                avg_nearby_dendrite_fraction=avg_nearby_dend_fraction,
                shuff_nearby_dendrite_fraction=shuff_nearby_dend_fraction,
                dendrite_fraction_coactive_distribution=dend_fraction_coactive_distribution,
                rel_dendrite_fraction=rel_dend_fraction,
                conj_avg_nearby_spine_fraction=conj_avg_nearby_spine_fraction,
                conj_shuff_nearby_spine_fraction=conj_shuff_nearby_spine_fraction,
                conj_spine_fraction_coactive_distribution=conj_spine_fraction_coactive_distribution,
                rel_conj_spine_fraction=rel_conj_spine_fraction,
                conj_avg_nearby_dendrite_fraction=conj_avg_nearby_dend_fraction,
                conj_shuff_nearby_dendrite_fraction=conj_shuff_nearby_dend_fraction,
                conj_dend_fraction_coactive_distribution=conj_dend_fraction_coactive_distribution,
                rel_conj_dendrite_fraction=rel_conj_dend_fraction,
                avg_nearby_relative_onset=avg_nearby_relative_onset,
                shuff_nearby_relative_onset=shuff_nearby_relative_onset,
                relative_onset_distribution=relative_onset_distribution,
                rel_nearby_relative_onset=rel_nearby_relative_onset,
                conj_avg_nearby_relative_onset=conj_avg_nearby_relative_onset,
                conj_shuff_nearby_relative_onset=conj_shuff_nearby_relative_onset,
                conj_relative_onset_distribution=conj_relative_onset_distribution,
                rel_conj_nearby_relative_onset=rel_conj_nearby_relative_onset,
                noncoactive_spine_calcium_amplitude=noncoactive_spine_calcium_amplitude,
                noncoactive_spine_calcium_traces=noncoactive_spine_calcium_traces,
                conj_fraction_participating=conj_fraction_participating,
                nonparticipating_spine_calcium_amplitude=nonparticipating_spine_calcium_amplitude,
                nonparticipating_spine_calcium_traces=nonparticipating_spine_calcium_traces,
                learned_movement_pattern=learned_movement_pattern,
                all_coactive_movement_correlation=all_coactive_movement_correlation,
                all_coactive_movement_stereotypy=all_coactive_movement_stereotypy,
                all_coactive_movement_reliability=all_coactive_movement_reliability,
                all_coactive_movement_specificity=all_coactive_movement_specificity,
                all_coactive_LMP_reliability=all_coactive_LMP_reliability,
                all_coactive_LMP_specificity=all_coactive_LMP_specificity,
                conj_movement_correlation=conj_movement_correlation,
                conj_movement_stereotypy=conj_movement_stereotypy,
                conj_movement_reliability=conj_movement_reliability,
                conj_movement_specificity=conj_movement_specificity,
                conj_LMP_reliability=conj_LMP_reliability,
                conj_LMP_specificity=conj_LMP_specificity,
                nonconj_movement_correlation=nonconj_movement_correlation,
                nonconj_movement_stereotypy=nonconj_movement_stereotypy,
                nonconj_movement_reliability=nonconj_movement_reliability,
                nonconj_movement_specificity=nonconj_movement_specificity,
                nonconj_LMP_reliability=nonconj_LMP_reliability,
                nonconj_LMP_specificity=nonconj_LMP_specificity,
            )
            # Save individual data if specified
            if save_ind:
                dendritic_coactivity_data.save()
            
            # Append data to the list
            analyzed_data.append(dendritic_coactivity_data)
    
    # Make the grouped data
    grouped_dendritic_coactivity_data = Grouped_Dendritic_Coactivity_Data(analyzed_data)
    if save_grouped:
        grouped_dendritic_coactivity_data.save()
    
    return grouped_dendritic_coactivity_data

