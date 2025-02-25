import numpy as np

from Lab_Analyses.Spine_Analysis_v2.coactive_vs_noncoactive_event_analysis import (
    coactive_vs_noncoactive_event_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.distance_coactivity_rate_analysis import (
    distance_coactivity_rate_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.local_coactivity_dataclass import (
    Grouped_Local_Coactivity_Data,
    Local_Coactivity_Data,
)
from Lab_Analyses.Spine_Analysis_v2.local_dendrite_activity import (
    local_dendrite_activity,
)
from Lab_Analyses.Spine_Analysis_v2.nearby_coactive_spine_activity import (
    nearby_coactive_spine_activity,
)
from Lab_Analyses.Spine_Analysis_v2.nearby_spine_density_analysis import (
    nearby_spine_density_analysis,
)
from Lab_Analyses.Spine_Analysis_v2.nearby_spine_movement_quality import (
    neraby_spine_movement_quality,
)
from Lab_Analyses.Spine_Analysis_v2.spatial_local_dendrite_activity import (
    spatial_local_dendrite_activity,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    load_spine_datasets,
    parse_movement_nonmovement_spines,
)
from Lab_Analyses.Spine_Analysis_v2.spine_volume_normalization import (
    load_norm_constants,
)
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Spine_Analysis_v2.variable_distance_dependence import (
    variable_distance_dependence,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import get_conservative_coactive_binary
from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def local_coactivity_analysis(
    mice_list,
    session,
    fov_type="apical",
    activity_window=(-2, 4),
    cluster_dist=5,
    zscore=False,
    volume_norm=True,
    partners=None,
    movement_period=None,
    save_ind=False,
    save_grouped=False,
):
    """Function to handle the analysis of local coactivity between spines

    INPUT PARAMETERS
        mice_list - list of str specifying all the mice to be analyzed

        session - str specifying the session to be analyzed

        fov_type - str specifying whether to analyze apical or basal FOVs

        activity_window - tuple specifying the window around which the activity
                          should be analyzed (sec)

        cluster_dist - int/float speicfying the distance to be considered local

        zscore - boolean of whether or not to zscore the activity for analysis

        volume_norm - boolean of whether or not to normalize activity by
                      spine volume

        partners - str specifying whether or not to contrain analysis to
                    only 'MRS' or 'nMRS' partners

        movement_period - str specifying whether or not to constrain analysis to
                        specific movement epochs

        save_ind - boolean specifying whether to save the data for each FOV

        save_grouped - boolean specifying whether to save all of the FOVs grouped together
    """
    # Get normalization constants
    if volume_norm:
        try:
            glu_constants = load_norm_constants(fov_type, zscore, "GluSnFr")
        except FileNotFoundError:
            return "GluSnFr constants have not be generated!!!"
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
        datasets = load_spine_datasets(mouse, [session], fov_type)

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
            ## Spine movement-related information
            movement_spines = np.array(data.movement_spines)
            rwd_movement_spines = np.array(data.reward_movement_spines)
            (
                nonmovement_spines,
                nonrwd_movement_spines,
            ) = parse_movement_nonmovement_spines(movement_spines, rwd_movement_spines)
            ## Dendrite movement-related information
            movement_dendrites = np.array(data.movement_dendrites)
            rwd_movement_dendrites = np.array(data.reward_movement_dendrites)
            (
                nonmovement_dendrites,
                nonrwd_movement_dendrites,
            ) = parse_movement_nonmovement_spines(
                movement_dendrites, rwd_movement_dendrites
            )
            # Dendrite activity
            dendrite_activity = data.dendrite_calcium_activity
            ## Dendrite poly roi positions and activity
            poly_dendrite_positions = data.poly_dendrite_positions
            poly_dendrite_dFoF = data.poly_dendrite_calcium_processed_dFoF
            ## Behavioral data
            lever_active = data.lever_active
            lever_force = data.lever_force_smooth
            lever_active_rwd = data.rewarded_movement_binary
            lever_inactive = np.absolute(lever_active - 1)
            lever_active_nonrwd = lever_active - lever_active_rwd

            # zscore activity if specified
            if zscore:
                spine_dFoF = d_utils.z_score(spine_dFoF)
                spine_calcium_dFoF = d_utils.z_score(spine_calcium_dFoF)
                poly_dendrite_dFoF = [d_utils.z_score(x) for x in poly_dendrite_dFoF]

            # Get volumes in um
            pix_to_um = zoom_factor / 2
            spine_volumes = (np.sqrt(raw_spine_volume) / pix_to_um) ** 2
            followup_volumes = (np.sqrt(raw_followup_volume) / pix_to_um) ** 2

            # Get spine activity rates
            spine_activity_rate = d_utils.calculate_activity_event_rate(spine_activity)

            # Sort out partner list if specified
            if partners == "MRS":
                partner_list = movement_spines
            elif partners == "nMRS":
                partner_list = nonmovement_spines
            elif partners == "rMRS":
                partner_list == rwd_movement_spines
            else:
                partner_list = None

            # Sort out movement period
            if movement_period == "movement":
                constrain_matrix = lever_active
            elif movement_period == "nonmovement":
                constrain_matrix = lever_inactive
            elif movement_period == "rewarded movement":
                constrain_matrix = lever_active_rwd
            elif movement_period == "nonrewarded movement":
                constrain_matrix = lever_active_nonrwd
            else:
                constrain_matrix = None

            # Get distance-dependent coactivity rates
            print(f"---- Calculating distance-dependent coactivity")
            (
                position_bins,
                distance_coactivity_rate,
                distance_coactivity_rate_norm,
                avg_local_coactivity_rate,
                avg_local_coactivity_rate_norm,
                shuff_local_coactivity_rate,
                shuff_local_coactivity_rate_norm,
                shuff_distance_coactivity_rate,
                shuff_distance_coactivity_rate_norm,
                coactive_spines,
                coactive_spines_norm,
                near_vs_dist_coactivity,
                near_vs_dist_coactivity_norm,
            ) = distance_coactivity_rate_analysis(
                spine_activity,
                spine_positions,
                spine_flags,
                spine_groupings,
                constrain_matrix=constrain_matrix,
                partner_list=partner_list,
                bin_size=5,
                cluster_dist=cluster_dist,
                sampling_rate=sampling_rate,
                norm_method="mean",
                alpha=0.05,
                iterations=10,
            )

            # Assess activity and coactivity of nearby spines
            print(f"---- Assessing properties of nearby spines")
            ## Spine activity
            (
                avg_nearby_spine_rate,
                shuff_nearby_spine_rate,
                spine_activity_rate_distribution,
                near_vs_dist_activity_rate,
            ) = variable_distance_dependence(
                spine_activity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            ## All local coactivity
            (
                avg_nearby_coactivity_rate,
                shuff_nearby_coactivity_rate,
                local_coactivity_rate_distribution,
                near_vs_dist_nearby_coactivity_rate,
            ) = variable_distance_dependence(
                avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            (
                other_spine_relative_coactivity,
                shuff_other_spine_relative_coactivity,
                other_spine_relative_coactivity_distribution,
                near_vs_dist_other_spine_relative_coactivity,
            ) = variable_distance_dependence(
                near_vs_dist_coactivity,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )

            # Assess local MRS density and spine volume
            ## MRS density
            (
                spine_density_distribution,
                MRS_density_distribution,
                avg_local_MRS_density,
                shuff_local_MRS_density,
                _,
            ) = nearby_spine_density_analysis(
                movement_spines,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                iterations=10000,
            )
            ## rMRS density
            (
                _,
                rMRS_density_distribution,
                avg_local_rMRS_density,
                shuff_local_rMRS_density,
                _,
            ) = nearby_spine_density_analysis(
                rwd_movement_spines,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                iterations=10000,
            )
            ## Spine volume
            (
                avg_nearby_spine_volume,
                shuff_nearby_spine_volume,
                nearby_spine_volume_distribution,
                near_vs_dist_volume,
            ) = variable_distance_dependence(
                spine_volumes,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            ## Plasticity
            relative_volumes = np.zeros(spine_activity.shape[1]) * np.nan
            rel_vols, stable_idxs = calculate_volume_change(
                [spine_volumes, followup_volumes],
                [spine_flags, followup_flags],
                norm=False,
                exclude="Shaft Spine",
            )
            relative_volumes[stable_idxs] = rel_vols[-1]
            enlarged, shrunken, stable = classify_plasticity(
                relative_volumes,
                threshold=(0.25, 0.5),
                norm=False,
            )
            enlarged = np.array(enlarged)
            shrunken = np.array(shrunken)
            stable = np.array(stable)
            ## Relative volume
            (
                local_relative_vol,
                shuff_relative_vol,
                relative_vol_distribution,
                near_vs_dist_relative_volume,
            ) = variable_distance_dependence(
                relative_volumes,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )

            ### Enlarged spines
            (
                local_nn_enlarged,
                shuff_nn_enlarged,
                enlarged_spine_distribution,
                _,
            ) = variable_distance_dependence(
                enlarged,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="nearest",
                iterations=10000,
            )
            ### Enlarged spines
            (
                local_nn_shrunken,
                shuff_nn_shrunken,
                shrunken_spine_distribution,
                _,
            ) = variable_distance_dependence(
                shrunken,
                spine_positions,
                spine_flags,
                spine_groupings,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="nearest",
                iterations=10000,
            )

            # Repeat local activity analysis, but for specific spine types
            # Enlarged Spine partners
            ## Activity
            (
                avg_nearby_enlarged_spine_rate,
                shuff_nearby_enlarged_spine_rate,
                enlarged_spine_activity_rate_distribution,
                near_vs_dist_enlarged_activity_rate,
            ) = variable_distance_dependence(
                spine_activity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                partner_list=np.array([not x for x in enlarged]),
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            ## All local coactivity
            (
                avg_nearby_enlarged_coactivity_rate,
                shuff_nearby_enlarged_coactivity_rate,
                enlarged_local_coactivity_rate_distribution,
                near_vs_dist_enlarged_nearby_coactivity_rate,
            ) = variable_distance_dependence(
                avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                partner_list=np.array([not x for x in enlarged]),
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            # Shrunken Spine partners
            ## Activity
            (
                avg_nearby_stable_spine_rate,
                shuff_nearby_stable_spine_rate,
                stable_spine_activity_rate_distribution,
                near_vs_dist_stable_activity_rate,
            ) = variable_distance_dependence(
                spine_activity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                partner_list=stable,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )
            ## All local coactivity
            (
                avg_nearby_stable_coactivity_rate,
                shuff_nearby_stable_coactivity_rate,
                stable_local_coactivity_rate_distribution,
                near_vs_dist_stable_nearby_coactivity_rate,
            ) = variable_distance_dependence(
                avg_local_coactivity_rate,
                spine_positions,
                spine_flags,
                spine_groupings,
                partner_list=stable,
                bin_size=5,
                cluster_dist=cluster_dist,
                method="local",
                iterations=10000,
            )

            # Analyze coactive and noncoactive spine events
            print(f"---- Analyzing coactive and noncoactive events")
            ## All events
            (
                nearby_spine_idxs,
                coactive_binary,
                noncoactive_binary,
                spine_coactive_event_num,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                spine_coactive_onset,
                spine_noncoactive_onset,
                fraction_spine_coactive,
                fraction_coactivity_participation,
            ) = coactive_vs_noncoactive_event_analysis(
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=constrain_matrix,
                partner_list=partner_list,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )
            ## Isolated events
            isolated_matrix = np.zeros(spine_activity.shape)
            for c in range(spine_activity.shape[1]):
                _, dend_inactive = get_conservative_coactive_binary(
                    spine_activity[:, c], dendrite_activity[:, c]
                )
                isolated_matrix[:, c] = dend_inactive

            ## Old method
            ### modify the constrain matrix
            # dendrite_inactive = 1 - dendrite_activity
            # if constrain_matrix is not None:
            #    temp_matrix = constrain_matrix.reshape(-1, 1) * dendrite_inactive
            # else:
            #    temp_matrix = dendrite_inactive

            (
                _,
                _,
                _,
                _,
                spine_coactive_traces,
                spine_noncoactive_traces,
                spine_coactive_calcium_traces,
                spine_noncoactive_calcium_traces,
                spine_coactive_amplitude,
                spine_noncoactive_amplitude,
                spine_coactive_calcium_amplitude,
                spine_noncoactive_calcium_amplitude,
                _,
                _,
                _,
                _,
            ) = coactive_vs_noncoactive_event_analysis(
                isolated_matrix,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_flags,
                spine_positions,
                spine_groupings,
                activity_window=activity_window,
                cluster_dist=cluster_dist,
                constrain_matrix=constrain_matrix,
                partner_list=partner_list,
                sampling_rate=sampling_rate,
                volume_norm=all_constants,
            )

            # Analyze nearby coactive spine activity
            print(f"---- Analyzing nearby spine activity")
            ## All events
            (
                coactive_spine_num,
                nearby_coactive_amplitude,
                nearby_coactive_calcium_amplitude,
                nearby_spine_onset,
                nearby_spine_onset_jitter,
                nearby_coactive_traces,
                nearby_coactive_calcium_traces,
            ) = nearby_coactive_spine_activity(
                nearby_spine_idxs,
                coactive_binary,
                spine_flags,
                spine_activity,
                spine_dFoF,
                spine_calcium_dFoF,
                spine_coactive_onset,
                all_constants,
                activity_window,
                sampling_rate,
            )

            # Assess nearby spine and coactivity movement encoding
            ## Nearby spines
            (
                nearby_movement_correlation,
                nearby_movement_stereotypy,
                nearby_movement_reliability,
                nearby_movement_specificity,
                nearby_LMP_reliability,
                nearby_LMP_speicficity,
            ) = neraby_spine_movement_quality(
                mouse,
                nearby_spine_idxs,
                spine_activity,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            (
                nearby_rwd_movement_correlation,
                nearby_rwd_movement_stereotypy,
                nearby_rwd_movement_reliability,
                nearby_rwd_movement_specificity,
                _,
                _,
            ) = neraby_spine_movement_quality(
                mouse,
                nearby_spine_idxs,
                spine_activity,
                lever_active_rwd,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            ## Local coactivity
            (
                coactive_movements,
                coactive_movement_correlation,
                coactive_movement_stereotypy,
                coactive_movement_reliability,
                coactive_movement_specificity,
                coactive_LMP_reliability,
                coactive_LMP_speicficity,
                learned_movement_pattern,
            ) = quantify_movement_quality(
                mouse,
                coactive_binary,
                lever_active,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            (
                coactive_rwd_movements,
                coactive_rwd_movement_correlation,
                coactive_rwd_movement_stereotypy,
                coactive_rwd_movement_reliability,
                coactive_rwd_movement_specificity,
                _,
                _,
                _,
            ) = quantify_movement_quality(
                mouse,
                coactive_binary,
                lever_active_rwd,
                lever_force,
                threshold=0.5,
                corr_duration=0.5,
                sampling_rate=sampling_rate,
            )
            coactive_frac_rwd_mvmts = [
                x.shape[0] / y.shape[0]
                for x, y in zip(coactive_rwd_movements, coactive_movements)
            ]

            LMP = [learned_movement_pattern for i in range(spine_activity.shape[1])]
            # learned_movement_pattern = np.stack(LMP).reshape(-1, 1)
            learned_movement_pattern = LMP

            # Analyze the local dendritic calcium levels
            print(f"---- Assessing local dendritic calcium")
            ## All periods
            (
                coactive_local_dend_traces,
                coactive_local_dend_amplitude,
                coactive_local_dend_amplitude_dist,
                noncoactive_local_dend_traces,
                noncoactive_local_dend_amplitude,
                noncoactive_local_dend_amplitude_dist,
                dend_position_bins,
            ) = spatial_local_dendrite_activity(
                spine_activity,
                dendrite_activity,
                spine_positions,
                spine_flags,
                spine_groupings,
                nearby_spine_idxs,
                poly_dendrite_dFoF,
                poly_dendrite_positions,
                activity_window=activity_window,
                constrain_matrix=constrain_matrix,
                sampling_rate=sampling_rate,
            )

            parameters = {
                "Sampling Rate": sampling_rate,
                "zscore": zscore,
                "Activity Window": activity_window,
                "Volume Norm": volume_norm,
                "FOV type": fov_type,
                "position bins": position_bins,
                "cluster dist": cluster_dist,
                "partners": partners,
                "movement period": movement_period,
                "dendrite position bins": dend_position_bins,
            }

            # Store the data
            spine_coactivity_data = Local_Coactivity_Data(
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
                distance_coactivity_rate=distance_coactivity_rate,
                shuff_distance_coactivity_rate=shuff_distance_coactivity_rate,
                shuff_distance_coactivity_rate_norm=shuff_distance_coactivity_rate_norm,
                distance_coactivity_rate_norm=distance_coactivity_rate_norm,
                avg_local_coactivity_rate=avg_local_coactivity_rate,
                avg_local_coactivity_rate_norm=avg_local_coactivity_rate_norm,
                shuff_local_coactivity_rate=shuff_local_coactivity_rate,
                shuff_local_coactivity_rate_norm=shuff_local_coactivity_rate_norm,
                coactive_spines=coactive_spines,
                coactive_spines_norm=coactive_spines_norm,
                near_vs_dist_coactivity=near_vs_dist_coactivity,
                near_vs_dist_coactivity_norm=near_vs_dist_coactivity_norm,
                avg_nearby_spine_rate=avg_nearby_spine_rate,
                shuff_nearby_spine_rate=shuff_nearby_spine_rate,
                spine_activity_rate_distribution=spine_activity_rate_distribution,
                near_vs_dist_activity_rate=near_vs_dist_activity_rate,
                avg_nearby_coactivity_rate=avg_nearby_coactivity_rate,
                shuff_nearby_coactivity_rate=shuff_nearby_coactivity_rate,
                local_coactivity_rate_distribution=local_coactivity_rate_distribution,
                near_vs_dist_nearby_coactivity_rate=near_vs_dist_nearby_coactivity_rate,
                other_spine_relative_coactivity=other_spine_relative_coactivity,
                shuff_other_spine_relative_coactivity=shuff_other_spine_relative_coactivity,
                other_spine_relative_coactivity_distrubution=other_spine_relative_coactivity_distribution,
                near_vs_dist_other_spine_relative_coactivity=near_vs_dist_other_spine_relative_coactivity,
                spine_density_distribution=spine_density_distribution,
                MRS_density_distribution=MRS_density_distribution,
                avg_local_MRS_density=avg_local_MRS_density,
                shuff_local_MRS_density=shuff_local_MRS_density,
                rMRS_density_distribution=rMRS_density_distribution,
                avg_local_rMRS_density=avg_local_rMRS_density,
                shuff_local_rMRS_density=shuff_local_rMRS_density,
                avg_nearby_spine_volume=avg_nearby_spine_volume,
                shuff_nearby_spine_volume=shuff_nearby_spine_volume,
                nearby_spine_volume_distribution=nearby_spine_volume_distribution,
                near_vs_dist_volume=near_vs_dist_volume,
                local_relative_vol=local_relative_vol,
                shuff_relative_vol=shuff_relative_vol,
                relative_vol_distribution=relative_vol_distribution,
                near_vs_dist_relative_volume=near_vs_dist_relative_volume,
                local_nn_enlarged=local_nn_enlarged,
                shuff_nn_enlarged=shuff_nn_enlarged,
                enlarged_spine_distribution=enlarged_spine_distribution,
                local_nn_shrunken=local_nn_shrunken,
                shuff_nn_shrunken=shuff_nn_shrunken,
                shrunken_spine_distribution=shrunken_spine_distribution,
                avg_nearby_enlarged_spine_rate=avg_nearby_enlarged_spine_rate,
                shuff_nearby_enlarged_spine_rate=shuff_nearby_enlarged_spine_rate,
                enlarged_spine_activity_rate_distribution=enlarged_spine_activity_rate_distribution,
                near_vs_dist_enlarged_activity_rate=near_vs_dist_enlarged_activity_rate,
                avg_nearby_enlarged_coactivity_rate=avg_nearby_enlarged_coactivity_rate,
                shuff_nearby_enlarged_coactivity_rate=shuff_nearby_enlarged_coactivity_rate,
                enlarged_local_coactivity_rate_distribution=enlarged_local_coactivity_rate_distribution,
                near_vs_dist_enlarged_nearby_coactivity_rate=near_vs_dist_enlarged_nearby_coactivity_rate,
                avg_nearby_stable_spine_rate=avg_nearby_stable_spine_rate,
                shuff_nearby_stable_spine_rate=shuff_nearby_stable_spine_rate,
                stable_spine_activity_rate_distribution=stable_spine_activity_rate_distribution,
                near_vs_dist_stable_activity_rate=near_vs_dist_stable_activity_rate,
                avg_nearby_stable_coactivity_rate=avg_nearby_stable_coactivity_rate,
                shuff_nearby_stable_coactivity_rate=shuff_nearby_stable_coactivity_rate,
                stable_local_coactivity_rate_distribution=stable_local_coactivity_rate_distribution,
                near_vs_dist_stable_nearby_coactivity_rate=near_vs_dist_stable_nearby_coactivity_rate,
                spine_coactive_event_num=spine_coactive_event_num,
                spine_coactive_traces=spine_coactive_traces,
                spine_noncoactive_traces=spine_noncoactive_traces,
                spine_coactive_calcium_traces=spine_coactive_calcium_traces,
                spine_noncoactive_calcium_traces=spine_noncoactive_calcium_traces,
                spine_coactive_amplitude=spine_coactive_amplitude,
                spine_noncoactive_amplitude=spine_noncoactive_amplitude,
                spine_coactive_calcium_amplitude=spine_coactive_calcium_amplitude,
                spine_noncoactive_calcium_amplitude=spine_noncoactive_calcium_amplitude,
                fraction_spine_coactive=fraction_spine_coactive,
                fraction_coactivity_participation=fraction_coactivity_participation,
                coactive_spine_num=coactive_spine_num,
                nearby_coactive_amplitude=nearby_coactive_amplitude,
                nearby_coactive_calcium_amplitude=nearby_coactive_calcium_amplitude,
                nearby_spine_onset=nearby_spine_onset,
                nearby_spine_onset_jitter=nearby_spine_onset_jitter,
                nearby_coactive_traces=nearby_coactive_traces,
                nearby_coactive_calcium_traces=nearby_coactive_calcium_traces,
                learned_movement_pattern=learned_movement_pattern,
                nearby_movement_correlation=nearby_movement_correlation,
                nearby_movement_stereotypy=nearby_movement_stereotypy,
                nearby_movement_reliability=nearby_movement_reliability,
                nearby_movement_specificity=nearby_movement_specificity,
                nearby_LMP_reliability=nearby_LMP_reliability,
                nearby_LMP_specificity=nearby_LMP_speicficity,
                nearby_rwd_movement_correlation=nearby_rwd_movement_correlation,
                nearby_rwd_movement_stereotypy=nearby_rwd_movement_stereotypy,
                nearby_rwd_movement_reliability=nearby_rwd_movement_reliability,
                nearby_rwd_movement_specificity=nearby_rwd_movement_specificity,
                coactive_movements=coactive_movements,
                coactive_movement_correlation=coactive_movement_correlation,
                coactive_movement_stereotypy=coactive_movement_stereotypy,
                coactive_movement_reliability=coactive_movement_reliability,
                coactive_movement_specificity=coactive_movement_specificity,
                coactive_LMP_reliability=coactive_LMP_reliability,
                coactive_LMP_specificity=coactive_LMP_speicficity,
                coactive_rwd_movements=coactive_rwd_movements,
                coactive_rwd_movement_correlation=coactive_rwd_movement_correlation,
                coactive_rwd_movement_stereotypy=coactive_rwd_movement_stereotypy,
                coactive_rwd_movement_reliability=coactive_rwd_movement_reliability,
                coactive_rwd_movement_specificity=coactive_rwd_movement_specificity,
                coactive_fraction_rwd_mvmts=np.array(coactive_frac_rwd_mvmts),
                coactive_local_dend_traces=coactive_local_dend_traces,
                coactive_local_dend_amplitude=coactive_local_dend_amplitude,
                coactive_local_dend_amplitude_dist=coactive_local_dend_amplitude_dist,
                noncoactive_local_dend_traces=noncoactive_local_dend_traces,
                noncoactive_local_dend_amplitude=noncoactive_local_dend_amplitude,
                noncoactive_local_dend_amplitude_dist=noncoactive_local_dend_amplitude_dist,
            )

            # Save individual data if specified
            if save_ind:
                spine_coactivity_data.save()

            # Append data to list
            analyzed_data.append(spine_coactivity_data)

    # Make the grouped data
    grouped_spine_coactivity_data = Grouped_Local_Coactivity_Data(analyzed_data)
    if save_grouped:
        grouped_spine_coactivity_data.save()

    return grouped_spine_coactivity_data
