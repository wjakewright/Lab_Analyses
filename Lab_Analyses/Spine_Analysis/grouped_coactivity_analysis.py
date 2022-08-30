import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Spine_Analysis.global_coactivity import global_coactivity_analysis
from Lab_Analyses.Spine_Analysis.spine_coactivity import spine_coactivity_analysis
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import (
    assess_movement_quality,
    spine_movement_activity,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    load_spine_datasets,
)
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_spine_dynamics,
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def grouped_coactivity_analysis(
    mice_list,
    days=("Early", "Middle", "Late"),
    followup=True,
    movement_epochs=None,
    corrected=True,
    threshold=0.5,
    exclude="Shaft Spine",
    save_short=False,
    save_long=False,
    save_path=None,
):
    """Function to handle the activity and structural analysis of dual spine imaging datasets
        across all mice and all FOVs. Analyzes both short term changes as well as across the 
        entire experiment
        
        INPUT PARAMETERS
            mice_list - list of strings specifying mice ids to be analyzed
            
            days - list of days you wish to analyze. Must be in the file name
            
            followup - boolean specifying if you want to include followup imaging sessions
                        in the analysis. This will influence the short term analyses. 
            
            movement_epochs - str specifying if you want on analyze only during movements and of
                            different types of movements. Accepts "all", "rewarded", "unrewarded", 
                            and "nonmovement".
                            Default is None, analyzing the entire imaging period
            
            corrected - boolean of whether or not to use the corrected volume estimates
            
            threshold - float specifying threshold for plasticity classification
            
            exclude - str specifying spine types to exclude from analysis

            save_short - boolen specifying whether to save the short term data

            save_long - boolean specifying whether to save the long term data

            save_path - str specifying where to save the data
            
        OUTPUT PARAMETERS
    """

    # Perform short term analyses
    short_term_datasets = {}
    print("Performing short term analysis")
    if followup:
        for day in days:
            print(f"- {day}")
            short_term_data = short_term_coactivity_analysis(
                mice_list, day, movement_epochs, corrected, threshold, exclude
            )
            short_term_datasets[day] = short_term_data
    else:
        for i in len(days[:-1]):
            day = [days[i], days[i + 1]]
            print(f"- {day[1]}")
            short_term_data = short_term_coactivity_analysis(
                mice_list, day, movement_epochs, corrected, threshold, exclude
            )
            short_term_datasets[day[1]] = short_term_data

    # Perform longitudinal analyses
    print("Performing longitudinal analysis")
    longitudinal_dataset = longitudinal_coactivity_analysis(
        mice_list, days, corrected, threshold, exclude
    )

    # Save section
    if save_path is None:
        save_path = r"C:\Users\Desktop\Analyzed_data\grouped"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if save_short:
        if movement_epochs is None:
            epoch_name = "session"
        else:
            epoch_name = movement_epochs
        for key, dataset in short_term_datasets.items():
            save_name = f"{epoch_name}_{key}_short_term_coactivity_data"
            save_pickle(save_name, dataset, save_path)
    if save_long:
        save_name = "longitudinal_coactivity_data"
        save_pickle(save_name, longitudinal_dataset, save_path)

    return short_term_datasets, longitudinal_dataset


def longitudinal_coactivity_analysis(mice_list, days, corrected, threshold, exclude):
    """Function to handle the short term"""

    grouped_spine_data = defaultdict(list)
    grouped_mice_data = defaultdict(list)

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"-- {mouse}")
        mouse_data = defaultdict(list)
        mean_mouse_data = defaultdict(list)
        mouse_datasets = load_spine_datasets(mouse, days, followup=False)
        # Analyze each FOV
        for FOV, data in mouse_datasets.items():
            keys = list(data.keys())
            datasets = list(data.values())

            # Perform coactivity analysis
            global_correlations = {}
            mean_global_correlations = {}
            coactivity_rates = {}
            mean_coactivity_rates = {}
            coactivity_amps = {}
            mean_coactivity_amps = {}
            relative_onsets = {}
            fraction_coactive = {}

            fraction_movement = {}
            movement_amps = {}
            mean_movement_amps = {}
            spine_movement_correlations = {}
            mean_spine_movement_correlations = {}

            for key, dataset in zip(keys, datasets):
                (
                    global_correlation,
                    coactivity_rate,
                    _,
                    _,
                    coactive_amplitudes,
                    coactive_spines,
                    _,
                    _,
                    _,
                    _,
                    rel_onsets,
                    _,
                ) = global_coactivity_analysis(
                    dataset, movements=None, sampling_rate=60
                )
                # remove eliminated and excluded spines
                el_spines = find_spine_classes(dataset.spine_flags, "Eliminated Spine")
                el_spines = np.array([not x for x in el_spines])
                select_spines = el_spines
                if exclude:
                    exclude_spines = find_spine_classes(dataset.spine_flags, exclude)
                    exclude_spines = np.array([not x for x in exclude_spines])
                    select_spines = select_spines * exclude_spines
                global_correlation = global_correlation[select_spines]
                coactivity_rate = coactivity_rate[select_spines]
                coactive_amplitudes = coactive_amplitudes[select_spines]
                coactive_spines = coactive_spines[select_spines]
                rel_onsets = rel_onsets[select_spines]
                # Store values for each day
                global_correlations[key] = global_correlation
                mean_global_correlations[key] = np.nanmean(global_correlation)
                coactivity_rates[key] = coactivity_rate
                mean_coactivity_rates[key] = np.nanmean(coactivity_rate)
                coactivity_amps[key] = coactive_amplitudes
                mean_coactivity_amps[key] = np.nanmean(coactive_amplitudes)
                relative_onsets[key] = np.nanmean(rel_onsets)
                fraction_coactive[key] = np.sum(coactive_spines) / len(coactive_spines)

                # Get movement variable data
                move_spines = np.array(dataset.movement_spines)
                all_befores, all_durings, _, _ = spine_movement_activity(
                    dataset,
                    activity_type="spine_GluSnFr_processed_dFoF",
                    exclude=None,
                    sampling_rate=60,
                    rewarded=False,
                )
                move_amps = [
                    np.nanmean(before - after)
                    for before, after in zip(all_befores, all_durings)
                ]
                move_amps = np.array(move_amps)
                (_, _, spine_move_corr, _,) = assess_movement_quality(
                    dataset,
                    activity_type="spine_GluSnFr_activity",
                    coactivity=False,
                    exclude=None,
                    sampling_rate=60,
                    rewarded=False,
                )
                move_spines = move_spines[select_spines]
                frac_move_spines = np.sum(move_spines) / len(move_spines)
                move_amps = move_amps[select_spines]
                spine_move_corr = spine_move_corr[select_spines]
                # Store values
                fraction_movement[key] = frac_move_spines
                movement_amps[key] = move_amps
                mean_movement_amps[key] = np.nanmean(move_amps)
                spine_movement_correlations[key] = spine_move_corr
                mean_spine_movement_correlations[key] = np.nanmean(spine_move_corr)

            # Calculate spine volumes and spine dynamics
            if corrected:
                _, volumes, _ = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            else:
                volumes, _, _ = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            potentiated = {}
            mean_potentiated = {}
            depressed = {}
            mean_depressed = {}
            stable = {}
            mean_stable = {}
            for key, value in volumes.items():
                p, d, s = classify_plasticity(value, threshold)
                potentiated[key] = p
                depressed[key] = d
                stable[key] = s
                mean_potentiated[key] = np.sum(p) / len(p)
                mean_depressed[key] = np.sum(d) / len(d)
                mean_stable[key] = np.sum(s) / len(s)
            (
                spine_density,
                normalized_spine_density,
                fraction_new_spines,
                fraction_eliminated_spines,
            ) = calculate_spine_dynamics(datasets, keys, distance=10)

            # Store values
            ## Spinewise data
            mouse_data["Relative Volumes"].append(volumes)
            mouse_data["Potentiated Spines"].append(potentiated)
            mouse_data["Depressed Spines"].append(depressed)
            mouse_data["Stable Spines"].append(stable)
            mouse_data["Global Correlations"].append(global_correlations)
            mouse_data["Coactivity Rates"].append(coactivity_rates)
            mouse_data["Coactivity Amplitudes"].append(coactivity_amps)
            mouse_data["Movement Amplitudes"].append(movement_amps)
            mouse_data["Movement Quality"].append(spine_movement_correlations)
            ## Averaged data
            mean_volumes = {}
            for key, value in volumes.items():
                mean_volumes[key] = np.nanmean(value)
            mean_mouse_data["Relative Volumes"].append(mean_volumes)
            mean_mouse_data["Potentiated Spines"].append(mean_potentiated)
            mean_mouse_data["Depressed Spines"].append(mean_depressed)
            mean_mouse_data["Stable Spines"].append(mean_stable)
            mean_mouse_data["Spine Density"].append(spine_density)
            mean_mouse_data["Nomalized Spine Density"].append(normalized_spine_density)
            mean_mouse_data["Fraction New Spines"].append(fraction_new_spines)
            mean_mouse_data["Fraction Eliminated Spines"].append(
                fraction_eliminated_spines
            )
            mean_mouse_data["Fraction Coactive"].append(fraction_coactive)
            mean_mouse_data["Gobal Correlations"].append(mean_global_correlations)
            mean_mouse_data["Coactivity Rates"].append(mean_coactivity_rates)
            mean_mouse_data["Coactivity Amplitudes"].append(mean_coactivity_amps)
            mean_mouse_data["Relative Coactivity Onsets"].append(relative_onsets)
            mean_mouse_data["Fraction Movement"].append(fraction_movement)
            mean_mouse_data["Movement Amplitudes"].append(mean_movement_amps)
            mean_mouse_data["Movement Quality"].append(mean_spine_movement_correlations)

        # Condense the mouse_data dictionaries
        ## Spinewise mouse data
        for key, value in mouse_data.items():
            temp_data = {}
            for day in value[0].keys():
                temp_data[day] = np.concatenate(list(d[day] for d in value))
            mouse_data[key] = temp_data
        ## Mean mouse data (Average across FOVs)
        for key, value in mean_mouse_data.items():
            temp_data = {}
            for day in value[0].keys():
                temp_data[day] = np.nanmean([d[day] for d in value])
            mean_mouse_data[key] = temp_data

        # Store data in grouped dictionaries
        for key, value in mouse_data.items():
            grouped_spine_data[key].append(value)
        for key, value in mean_mouse_data.items():
            grouped_mice_data[key].append(value)

    # Condense grouped spine data
    for key, value in grouped_spine_data.items():
        temp_data = {}
        for day in value[0].keys():
            temp_data[day] = np.concatenate(list(d[day] for d in value))
        grouped_spine_data[key] = temp_data
    for key, value in grouped_mice_data.items():
        temp_data = {}
        for day in value[0].keys():
            temp_data[day] = np.array(value)
        grouped_mice_data[key] = temp_data

    # Store in dataclass for output
    longitudinal_data = Longitudinal_Coactivity_Volume_Data(
        mouse_ids=mice_list,
        spine_relative_volume=grouped_spine_data["Relative Volumes"],
        potentiated_spine_ids=grouped_spine_data["Potentiated Spines"],
        depressed_spine_ids=grouped_spine_data["Depressed Spines"],
        stable_spine_ids=grouped_spine_data["Stable Spines"],
        spine_global_correlations=grouped_spine_data["Global Correlations"],
        spine_coactivity_rates=grouped_spine_data["Coactivity Rates"],
        spine_coactivity_amplitudes=grouped_spine_data["Coactivity Amplitudes"],
        spine_movement_amplitudes=grouped_spine_data["Movement Amplitudes"],
        spine_movement_quality=grouped_spine_data["Movement Quality"],
        avg_relative_volume=grouped_mice_data["Relative Volumes"],
        fraction_potentiated=grouped_mice_data["Potentiated Spines"],
        fraction_depressed=grouped_mice_data["Depressed Spines"],
        fraction_stable=grouped_mice_data["Stable Spines"],
        avg_spine_density=grouped_mice_data["Spine Density"],
        normalized_spine_density=grouped_mice_data["Normalized Spine Density"],
        fraction_new_spines=grouped_mice_data["Fraction New Spines"],
        fraction_eliminated_spines=grouped_mice_data["Fraction Eliminated Spines"],
        avg_global_correlation=grouped_mice_data["Global Correlations"],
        avg_coactivity_rates=grouped_mice_data["Coactivity Rates"],
        avg_coactivity_amplitudes=grouped_mice_data["Coactivity Amplitudes"],
        fraction_movement=grouped_mice_data["Fraction Movement"],
        avg_spine_coactivity_onset=grouped_mice_data["Relative Coactivity Onsets"],
        avg_movement_amplitudes=grouped_mice_data["Movement Amplitudes"],
        avg_movement_quality=grouped_mice_data["Movement Quality"],
    )

    return longitudinal_data


def short_term_coactivity_analysis(
    mice_list, day, movement_epochs, corrected, threshold, exclude
):
    """Function to handle the short term analysis of coactivity datasets"""

    grouped_data = defaultdict(list)

    # Analyze each mouse seperately
    for mouse in mice_list:
        print(f"--- {mouse}")
        mouse_data = defaultdict(list)
        if type(day) == str:
            mouse_datasets = load_spine_datasets(mouse, [day], followup=True)
        elif type(day) == list:
            mouse_datasets = load_spine_datasets(mouse, day, followup=False)
        # Analyze each FOV seperately
        for FOV, data in mouse_datasets.items():
            keys = list(data.keys())
            datasets = list(data.values())
            # Get the spine groupings with parent dendrite
            spine_groupings = datasets[0].spine_grouping
            if type(spine_groupings[0]) != list:
                spine_groupings = [spine_groupings]
            # Get the coactivity data
            (
                global_correlation,
                coactivity_rate,
                spine_fraction_coactive,
                dend_fraction_coactive,
                coactive_amplitude,
                coactive_spines,
                coactivity_epoch_traces,
                coactivity_mean_traces,
                dend_mean_sems,
                spine_onsets,
                relative_onsets,
                d_onsets,
            ) = global_coactivity_analysis(
                datasets[0], movement_epochs, sampling_rate=60
            )

            # convert coactivity trace data to lists
            coactivity_mean_traces = list(coactivity_mean_traces.values())
            coactivity_epoch_traces = list(coactivity_epoch_traces.values())
            ## duplicate the dendrite traces to match with their children spines
            dend_mean_traces = list(np.zeros(len(coactivity_mean_traces)))
            dend_onsets = np.zeros(len(spine_onsets))
            for i, grouping in enumerate(spine_groupings):
                for g in grouping:
                    dend_mean_traces[g] = dend_mean_sems[i]
                    dend_onsets[g] = d_onsets[i]

            # get spine volumes
            if corrected:
                _, volumes, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            else:
                volumes, _, spine_idxs = calculate_volume_change(
                    datasets, keys, exclude=exclude,
                )
            # Use only the post volumes
            volumes = volumes[keys[1]]

            # Classify the plasticity
            potentiated, depressed, stable = classify_plasticity(volumes, threshold)
            movement_spines = np.array(datasets[0].movement_spines)
            reward_movement_spines = np.array(datasets[0].reward_movement_spines)
            nonreward_movement_spines = movement_spines.astype(
                int
            ) - reward_movement_spines.astype(int)
            nonreward_movement_spines[nonreward_movement_spines == -1] = 0
            nonreward_movement_spines = nonreward_movement_spines.astype(bool)
            movement_dendrites = np.zeros(len(movement_spines))
            reward_movement_dendrites = np.zeros(len(reward_movement_spines))
            for i, grouping in enumerate(spine_groupings):
                for g in grouping:
                    movement_dendrites[g] = datasets[0].movement_dendrites[i]
                    reward_movement_dendrites[g] = datasets[
                        0
                    ].reward_movement_dendrites[i]

            # Assess movement activity
            all_befores, all_durings, _, movement_traces = spine_movement_activity(
                datasets[0],
                activity_type="spine_GluSnFr_processed_dFoF",
                exclude=None,
                sampling_rate=60,
                rewarded=False,
            )
            (
                rwd_befores,
                rwd_durings,
                _,
                reward_movement_traces,
            ) = spine_movement_activity(
                data=datasets[0],
                activity_type="spine_GluSnFr_processed_dFoF",
                exclude=None,
                sampling_rate=60,
                rewarded=True,
            )
            movement_amps = [
                np.nanmean(after - before)
                for before, after in zip(all_befores, all_durings)
            ]
            reward_movement_amps = [
                np.nanmean(rwd_after - rwd_before)
                for rwd_before, rwd_after in zip(rwd_befores, rwd_durings)
            ]
            movement_amps = np.array(movement_amps)
            reward_movement_amps = np.array(reward_movement_amps)
            movement_traces = list(movement_traces.values())
            reward_movement_traces = list(reward_movement_traces)

            # Assess movement quality
            (
                spine_movements,
                _,
                spine_movement_correlations,
                learned_movement,
            ) = assess_movement_quality(
                datasets[0],
                activity_type="spine_GluSnFr_activity",
                coactivity=False,
                exclude=None,
                sampling_rate=60,
                rewarded=False,
            )
            (
                coactive_movements,
                _,
                coactive_movement_correlations,
                _,
            ) = assess_movement_quality(
                datasets[0],
                activity_type="spine_GluSnFr_activity",
                coactivity=True,
                exclude=None,
                sampling_rate=60,
                rewarded=False,
            )

            # Assess spine coactivity
            spine_coactivity_mat, _ = spine_coactivity_analysis(
                datasets[0].spine_GluSnFr_activity,
                datasets[0].spine_positions,
                datasets[0].spine_flags,
                spine_grouping=spine_groupings,
                bin_size=5,
                sampling_rate=60,
            )
            # Look at only nearby spine coactivity
            spine_coactivity = []
            for i in range(spine_coactivity_mat.shape[1]):
                spine_coactivity.append(
                    np.mean([spine_coactivity_mat[0, i], spine_coactivity_mat[1, i]])
                )
            spine_coactivity = np.array(spine_coactivity)

            # Remove all non-stable spines from varibles
            global_correlation = global_correlation[spine_idxs]
            coactivity_rate = coactivity_rate[spine_idxs]
            spine_fraction_coactive = spine_fraction_coactive[spine_idxs]
            dend_fraction_coactive = dend_fraction_coactive[spine_idxs]
            coactive_amplitude = coactive_amplitude[spine_idxs]
            coactive_spines = coactive_spines[spine_idxs]
            coactivity_epoch_traces = [coactivity_epoch_traces[i] for i in spine_idxs]
            coactivity_mean_traces = [coactivity_mean_traces[i] for i in spine_idxs]
            dend_mean_traces = [dend_mean_traces[i] for i in spine_idxs]
            spine_onsets = spine_onsets[spine_idxs]
            relative_onsets = relative_onsets[spine_idxs]
            dend_onsets = dend_onsets[spine_idxs]
            movement_spines = movement_spines[spine_idxs]
            reward_movement_spines = reward_movement_spines[spine_idxs]
            nonreward_movement_spines = nonreward_movement_spines[spine_idxs]
            movement_dendrites = movement_dendrites[spine_idxs]
            reward_movement_dendrites = reward_movement_dendrites[spine_idxs]
            movement_amps = movement_amps[spine_idxs]
            reward_movement_amps = reward_movement_amps[spine_idxs]
            movement_traces = [movement_traces[i] for i in spine_idxs]
            reward_movement_traces = [reward_movement_traces[i] for i in spine_idxs]
            spine_movements = [spine_movements[i] for i in spine_idxs]
            spine_movement_correlations = spine_movement_correlations[spine_idxs]
            coactive_movements = [coactive_movements[i] for i in spine_idxs]
            coactive_movement_correlations = coactive_movement_correlations[spine_idxs]
            spine_coactivity = spine_coactivity[spine_idxs]

            # Store variables in mouse_data dictionary
            mouse_data["global correlation"].append(global_correlation)
            mouse_data["coactivity rate"].append(coactivity_rate)
            mouse_data["spine fraction coactive"].append(spine_fraction_coactive)
            mouse_data["dend fraction coactive"].append(dend_fraction_coactive)
            mouse_data["coactive amplitude"].append(coactive_amplitude)
            mouse_data["coactive spines"].append(coactive_spines)
            mouse_data["coactivity epoch traces"].append(coactivity_epoch_traces)
            mouse_data["coactivity mean traces"].append(coactivity_mean_traces)
            mouse_data["dend mean traces"].append(dend_mean_traces)
            mouse_data["spine onsets"].append(spine_onsets)
            mouse_data["relative onsets"].append(relative_onsets)
            mouse_data["dend onsets"].append(dend_onsets)
            mouse_data["movement spines"].append(movement_spines)
            mouse_data["reward movement spines"].append(reward_movement_spines)
            mouse_data["nonreward movement spines"].append(nonreward_movement_spines)
            mouse_data["movement dendrites"].append(movement_dendrites)
            mouse_data["reward movement dendrites"].append(reward_movement_dendrites)
            mouse_data["movement amps"].append(movement_amps)
            mouse_data["reward movement amps"].append(reward_movement_amps)
            mouse_data["movement traces"].append(movement_traces)
            mouse_data["reward movement traces"].append(reward_movement_traces)
            mouse_data["spine movements"].append(spine_movements)
            mouse_data["spine movement correlations"].append(
                spine_movement_correlations
            )
            mouse_data["coactive_movements"].append(coactive_movements)
            mouse_data["coactive movement correlations"].append(
                coactive_movement_correlations
            )
            mouse_data["spine coactivity"].append(spine_coactivity)
            mouse_data["volumes"].append(volumes)
            mouse_data["potentiated"].append(potentiated)
            mouse_data["depressed"].append(depressed)
            mouse_data["stable"].append(stable)
            learned_movements = [
                learned_movement for i in range(len(global_correlation))
            ]
            mouse_data["leaned movement"].append(learned_movements)
            fovs = [FOV for i in range(len(global_correlation))]
            ids = [mouse for i in range(len(global_correlation))]
            mouse_data["FOV"].append(fovs)
            mouse_data["mouse_id"].append(ids)

        # Merge FOVs for this mouse
        for key, value in mouse_data.items():
            if len(value) == 1:
                mouse_data[key] = value[0]
                continue
            if type(value) == np.ndarray:
                mouse_data[key] = np.concatenate(value)
            elif type(value) == list:
                mouse_data[key] = [y for x in value for y in x]

        # Store data for this mouse
        for key, value in mouse_data.items():
            grouped_data[key].append(value)

    # Merge data across mice
    for key, value in grouped_data.items():
        if type(value) == np.ndarray:
            grouped_data[key] = np.concatenate(value)
        elif type(value) == list:
            grouped_data[key] = [y for x in value for y in x]

    # Store data in dataclass for output
    short_term_coactivity_data = Short_Term_Coactivity_Volume_Data(
        mouse_ids=grouped_data["mouse_id"],
        FOVs=grouped_data["FOV"],
        volumes=grouped_data["volumes"],
        potentiated_spines=grouped_data["potentiated"],
        depressed_spines=grouped_data["depressed"],
        stable_spines=grouped_data["stable"],
        global_correlation=grouped_data["global correlation"],
        coactivity_rate=grouped_data["coactivity rate"],
        spine_fraction_coactive=grouped_data["spine fraction coactive"],
        dendrite_fraction_coactive=grouped_data["dend fraction coactive"],
        coactive_spines=grouped_data["coactive spines"],
        coactive_amplitude=grouped_data["coactive amplitude"],
        coactivity_epoch_traces=grouped_data["coactivity epoch traces"],
        coactivity_mean_traces=grouped_data["coactivity mean traces"],
        dendrite_mean_traces=grouped_data["dend mean traces"],
        spine_onsets=grouped_data["spine onsets"],
        dendrite_onsets=grouped_data["dend onsets"],
        relative_spine_onsets=grouped_data["relative onsets"],
        movement_spines=grouped_data["movement spines"],
        reward_movement_spines=grouped_data["reward movement spines"],
        nonreward_movement_spines=grouped_data["nonreward movement spines"],
        movement_dendrites=grouped_data["movement dendrites"],
        reward_movement_dendrites=grouped_data["reward movement dendrites"],
        movement_amps=grouped_data["movement amps"],
        reward_movement_amps=grouped_data["reward movement amps"],
        movement_traces=grouped_data["movement traces"],
        reward_movement_traces=grouped_data["reward movement traces"],
        spine_lever_movements=grouped_data["spine movements"],
        spine_movement_correlations=grouped_data["spine movement correlations"],
        coactive_lever_movements=grouped_data["coactive movements"],
        coactive_movement_correlations=grouped_data["coactive movement correlations"],
        learned_movements=grouped_data["learned movement"],
        spine_coactivity=grouped_data["spine coactivity"],
    )

    return short_term_coactivity_data


################# DATACLASSES ###################
@dataclass
class Short_Term_Coactivity_Volume_Data:
    """Dataclass for storing the short term coactivity volume
        analyzed data"""

    mouse_ids: list
    FOVs: list
    volumes: np.ndarray
    potentiated_spines: np.ndarray
    depressed_spines: np.ndarray
    stable_spines: np.ndarray
    global_correlation: np.ndarray
    coactivity_rate: np.ndarray
    spine_fraction_coactive: np.ndarray
    dendrite_fraction_coactive: np.ndarray
    coactive_spines: np.ndarray
    coactive_amplitude: np.ndarray
    coactivity_epoch_traces: list
    coactivity_mean_traces: list
    dendrite_mean_traces: list
    spine_onsets: np.ndarray
    dendrite_onsets: np.ndarray
    relative_spine_onsets: np.ndarray
    movement_spines: np.ndarray
    reward_movement_spines: np.ndarray
    nonreward_movement_spines: np.ndarray
    movement_dendrites: np.ndarray
    reward_movement_dendrites: np.ndarray
    movement_amps: np.ndarray
    reward_movement_amps: np.ndarray
    movement_traces: list
    reward_movement_traces: list
    spine_lever_movements: list
    spine_movement_correlations: np.ndarray
    coactive_lever_movements: list
    coactive_movement_correlations: np.ndarray
    learned_movements: list
    spine_coactivity: np.ndarray


@dataclass
class Longitudinal_Coactivity_Volume_Data:
    """Dataclass for storing the longitudinal coactivity
        volume data analysis"""

    mouse_ids: list
    # spinewise values
    spine_relative_volume: dict
    potentiated_spine_ids: dict
    depressed_spine_ids: dict
    stable_spine_ids: dict
    spine_global_correlations: dict
    spine_coactivity_rates: dict
    spine_coactivity_amplitudes: dict
    spine_movement_amplitudes: dict
    spine_movement_quality: dict
    # mouse averaged data
    avg_relative_volume: dict
    fraction_potentiated: dict
    fraction_depressed: dict
    fraction_stable: dict
    avg_spine_density: dict
    normalized_spine_density: dict
    fraction_new_spines: dict
    fraction_eliminated_spines: dict
    avg_global_correlation: dict
    avg_coactivity_rates: dict
    avg_coactivity_amplitudes: dict
    fraction_movement: dict
    avg_spine_coactivity_onset: dict
    avg_movement_amplitudes: dict
    avg_movement_quality: dict

