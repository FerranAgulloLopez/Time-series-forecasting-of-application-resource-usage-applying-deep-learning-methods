# Modification of the Autopilot implementation from Claudia Herron Mulet

from os import listdir
import numpy as np
from os.path import isfile, exists, join
import random
import itertools
from bisect import bisect_left


def delta(x, y):
    '''Kroenecker delta implementation:
    Args:
        x (float): number 1 to compute the delta
        y (float): number 2 to compute the delta
    '''
    if np.isclose(x, y, 0.01).all():
        return 1
    else:
        return 0


def overrun_cost(L, d_m, prev, buckets, hist):
    j = bisect_left(buckets, L)  # returns closest highest than limit using binary search like algorithm
    return (1 - d_m) * prev + d_m * sum(hist[j:])


def underrun_cost(L, d_m, prev, buckets, hist):
    j = bisect_left(buckets, L)  # returns closest highest than limit using binary search like algorithm
    return (1 - d_m) * prev + d_m * sum(hist[:j])


def model_limit(d_m, M_m, over_prev, under_prev, L_m_prev, buckets, hist, w_o, w_u, w_delta_L):
    '''Equation 7 of Autopilot
    Args:
        d_m (float): model decay rate
        M_m (float): model margin
        over_prev (dict): overrun cost of previous time (key=limit, value=cost)
        under_prev (dict): undrrun cost of previous time (key=limit, value=cost)
        L_m_prev (float): previous model limit
        buckets (np.array): edges of resource histograms
        hist (np.array): histogram usage to evaluate
    Returns:
        L_m (float): model limit
        overrun (dict):
    '''
    w_sum = np.inf
    L_m_prime = -1
    over_current, under_current = {}, {}
    L_m_prime_prev = L_m_prev - M_m
    for L in buckets:
        over_current[L] = w_o * overrun_cost(L, d_m, over_prev[L], buckets, hist)
        under_current[L] = w_u * underrun_cost(L, d_m, under_prev[L], buckets, hist)
        change_of_limit = w_delta_L * delta(L, L_m_prime_prev)
        if (over_current[L] + under_current[L] + change_of_limit) < w_sum:
            w_sum = over_current[L] + under_current[L] + change_of_limit
            L_m_prime = L
    L_m = L_m_prime + M_m
    return L_m, over_current, under_current


def overrun_model(L_m, buckets, hist):
    j = bisect_left(buckets, L_m)  # returns closest highest than limit using binary search like algorithm
    return sum(hist[j:])


def underrun_model(L_m, buckets, hist):
    j = bisect_left(buckets, L_m)  # returns closest highest than limit using binary search like algorithm
    return sum(hist[:j])


def cost_model(d_m, M_m, over_cost, under_cost, L_m_prev, c_m, buckets, hist, w_o, w_u, w_delta_L, d):
    L_m, over_cost, under_cost = model_limit(d_m, M_m, over_cost, under_cost, L_m_prev, buckets, hist, w_o, w_u, w_delta_L)
    overrun = w_o * overrun_model(L_m, buckets, hist)
    underrun = w_u * underrun_model(L_m, buckets, hist)
    change_of_limit = w_delta_L * delta(L_m, L_m_prev)
    return d * (overrun + underrun + change_of_limit) + (1 - d) * c_m, L_m, over_cost, under_cost


def get_ml_rec(agg_trace, buckets, dm_min, dm_max, d_n_step, Mm_min, Mm_max, M_n_step, w_delta_m, w_delta_L, w_o, w_u, d):
    rec = []
    d_ms = np.linspace(dm_min, dm_max, d_n_step, endpoint=True)
    M_ms = np.linspace(Mm_min, Mm_max, M_n_step, endpoint=True)
    models = list(itertools.product(d_ms, M_ms))
    c_m, L_m = dict.fromkeys(models, 0), dict.fromkeys(models, 0)
    over_cost_m, under_cost_m = dict.fromkeys(models, dict.fromkeys(buckets, 0)), dict.fromkeys(models,
                                                                                                dict.fromkeys(
                                                                                                    buckets, 0))
    m_current = (-1, -1)
    L_current = -1
    for t in range(len(agg_trace)):
        # print("processing window {}/{}".format(t,len(agg_trace)))
        min_cost = np.inf
        m_argmin = (-1, -1)
        for m in models:
            c_m[m], L_m[m], over_cost_m[m], under_cost_m[m] = cost_model(m[0], m[1], over_cost_m[m],
                                                                         under_cost_m[m], L_m[m], c_m[m], buckets,
                                                                         agg_trace[t], w_o, w_u, w_delta_L, d)
            change_of_m = w_delta_m * delta(m_current, m)
            changel_of_L = w_delta_L * delta(L_m[m], L_current)
            tot = c_m[m] + change_of_m + changel_of_L
            if tot < min_cost:
                min_cost = tot
                m_argmin = m
        rec.append(L_m[m_argmin])
    return rec


def get_granular_rec(rec, warmup, RESOLUTION):
    return np.concatenate((np.repeat(np.array(warmup), RESOLUTION), np.repeat(rec, RESOLUTION)))


def get_aggregated_signal(trace, RESOLUTION, CPU_BINS):
    num_windows = int(np.floor((len(trace)) / RESOLUTION))
    agg_signal = []
    for t in range(num_windows):  # for each of the windows
        window_hist = np.histogram(trace[t * RESOLUTION:(t * RESOLUTION + RESOLUTION)], bins=CPU_BINS)[
            0]  # compute the histogram per window
        agg_signal.append(window_hist)
    if len(trace) % RESOLUTION != 0:  # there are some timesteps missing in the aggregated signal
        t += 1
        window_hist = \
            np.histogram(trace[t * RESOLUTION:(t * RESOLUTION + (int(len(trace)) % RESOLUTION))], bins=CPU_BINS)[0]
        agg_signal.append(window_hist)
    return agg_signal


def get_peak_in_hist(hist):
    # get the index of the higher non empty bucket in histogram
    return max([i for i, e in enumerate(hist) if e != 0])


def get_peak_rec(agg_trace, buckets, N):
    rec = []
    for t in range(len(agg_trace)):
        last = max(0, t - (N - 1))
        peak_idx = max([get_peak_in_hist(hist) for hist in agg_trace[last:t + 1]])
        rec.append(buckets[peak_idx])
    return rec


def get_decay(tau, half_life):
    '''Computes the exponentially decaying weight
        Returns:
        exponent on base 2 of - tau divided by the half time
    '''
    return 2 ** -(tau / half_life)


def average_use_of_histogram(hist, buckets, NUM_BUCKETS):
    '''Computes the average use of histograms as in formula 2 of autopilot paper
    '''
    numerator, denominator = 0, 0
    for j in range(NUM_BUCKETS):
        numerator += buckets[j] * hist[j]
        denominator += hist[j]

    return numerator / denominator


def get_wavg_rec(agg_trace, buckets, half_life, NUM_BUCKETS, N):
    rec = []
    denominator = sum([get_decay(tau, half_life) for tau in range(N + 1)])
    for t in range(len(agg_trace)):
        numerator = sum(
            [get_decay(tau, half_life) * average_use_of_histogram(agg_trace[t - tau], buckets, NUM_BUCKETS) for tau in
             range(t + 1)])
        rec.append(numerator / denominator)
    return rec


def get_decayed_hist(agg_trace, t, buckets, prev, half_life):
    if t == 0:
        return buckets * 1 * agg_trace[0]  # decay is 1
    else:
        # print(get_decay(t, half_life))
        return prev + buckets * get_decay(t, half_life) * agg_trace[t]


def get_jp_rec(agg_trace, buckets, half_life, perc, NUM_BUCKETS):
    rec = []
    adj_hist = np.zeros(NUM_BUCKETS)
    for t in range(len(agg_trace)):
        adj_hist = get_decayed_hist(agg_trace, t, buckets, adj_hist, half_life)
        rec.append(np.percentile(adj_hist, perc))
    return rec
