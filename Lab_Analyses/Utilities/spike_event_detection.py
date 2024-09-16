import numpy as np


def cascade_event_detection(spikes, cutoff=0.25):
    """Function to descritize spikes into active portions"""
    DERIV_CUTOFF = 0.005

    activity = np.zeros(spikes.shape)
    floored = np.zeros(spikes.shape)

    # Iterate through each ROI
    for i in range(spikes.shape[1]):
        # Find areas above the cutoff
        curr_spikes = spikes[:, i]
        temp_active = np.zeros(len(curr_spikes))
        temp_active[curr_spikes > cutoff] = 1

        # Refine the starts and stops
        ## Find the starts and stops
        pad = np.zeros(1)
        diff = np.diff(np.concatenate((pad, temp_active, pad)))

        start_trace = diff == 1
        stop_trace = diff == -1
        start_idxs = np.nonzero(start_trace)[0]
        stop_idxs = np.nonzero(stop_trace)[0]

        deriv = np.gradient(curr_spikes)

        transitions = []
        for start, stop in zip(start_idxs, stop_idxs):
            trans = refine_start_stop(start, stop, deriv, cutoff=DERIV_CUTOFF)
            transitions.append(trans)

        # Update the trace
        refine_active = np.zeros(len(temp_active))
        for t in transitions:
            refine_active[t[0] : t[1]] = 1

        inactive_idxs = np.nonzero(refine_active == 0)[0]
        floored_trace = np.copy(curr_spikes)
        floored_trace[inactive_idxs] = 0

        activity[:, i] = refine_active
        floored[:, i] = floored_trace

    return activity, floored


def refine_start_stop(start, stop, deriv, cutoff):
    """Helper function to refine onsets and offsets"""
    # Refine start
    start_search = deriv[:start]
    below_cutoff = start_search <= cutoff
    new_start = np.nonzero(below_cutoff)[0][-1]
    # Refine stop
    stop_search = deriv[stop:]
    below_cutoff = stop_search >= -cutoff
    new_stop = np.nonzero(below_cutoff)[0][0] + stop

    return (new_start, new_stop)
