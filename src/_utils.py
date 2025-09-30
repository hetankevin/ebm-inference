# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import heapq
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import count, islice
from math import floor, exp, isfinite, isinf, log
from typing import List, Optional, Tuple

import numpy as np

from ... import develop
from ...utils._native import Native
from ...utils._purify import purify
from ._tensor import restore_missing_value_zeros

_log = logging.getLogger(__name__)


def _midpoint(low: float, high: float) -> float:
    """Return midpoint between `low` and `high` with high numerical accuracy."""
    half_diff = (high - low) / 2
    if isinf(half_diff):
        # first try to subtract then divide since that's more accurate but some float64
        # values will fail eg (max_float - min_float == +inf) so we need to try
        # a less accurate way of dividing first if we detect this.  Dividing
        # first will always succeed, even with the most extreme possible values of
        # max_float / 2 - min_float / 2
        half_diff = high / 2 - low / 2

    # floats have more precision the smaller they are,
    # so use the smaller number as the anchor
    mid = low + half_diff if abs(low) <= abs(high) else high - half_diff

    if mid <= low:
        # this can happen with very small half_diffs that underflow the add/subtract operation
        # if this happens the numbers must be very close together on the order of a float tick.
        # We use lower bound inclusive for our cut discretization, so make the mid == high
        mid = high
    return mid


def convert_categorical_to_continuous(categories):
    # we do automagic detection of feature types by default, and sometimes a feature which
    # was really continuous might have most of it's data as one or two values.  An example would
    # be a feature that we have "0" and "1" in the training data, but "-0.1" and "3.1" are also
    # possible.  If during prediction we see a "3.1" we can magically convert our categories
    # into a continuous range with a cut point at 0.5.  Now "-0.1" goes into the [-inf, 0.5) bin
    # and 3.1 goes into the [0.5, +inf] bin.
    #
    # We can't convert a continuous feature that has cuts back into categoricals
    # since the categorical value could have been anything between the cuts that we know about.

    clusters = defaultdict(list)
    non_float_idxs = set()

    old_min = +np.inf
    old_max = -np.inf
    for category, idx in categories.items():
        try:
            # this strips leading and trailing spaces
            val = float(category)
        except ValueError:
            non_float_idxs.add(idx)
            continue

        if not isfinite(val):
            continue

        old_min = min(old_min, val)
        old_max = max(old_max, val)

        clusters[idx].append(val)

    # there's a super fringe case where two category strings map to the same bin, but
    # one of them is a float and the other is a non-float.  Normally, we'd include the
    # non-float categorical in the unseens, but in this case we'd need to include
    # a part of a bin.  Handling this just adds too much complexity for the benefit
    # and you could argue that the evidence from the other models is indicating that
    # the string should be closer to zero of the weight from the floating point bin
    # so we take the simple route of putting all the weight into the float and none on the
    # non-float.  We still need to remove any indexes though that map to both a float
    # and a non-float, so this line handles that
    non_float_idxs = [idx for idx in non_float_idxs if idx not in clusters]
    non_float_idxs.append(max(categories.values()) + 1)

    if len(clusters) == 0:
        return np.empty(0, np.float64), [[0], [], non_float_idxs], np.nan, np.nan

    cluster_bounds = sorted(
        (min(cluster_list), max(cluster_list)) for cluster_list in clusters.values()
    )

    # TODO: move everything below here into C++ to ensure cross language compatibility
    cuts = []
    _, low = cluster_bounds[0]
    for high, next_low in cluster_bounds[1:]:
        if low < high:
            # if they are equal or if low is higher then we can't separate one cluster
            # from another, so we keep joining them until we can get clean separations
            cuts.append(_midpoint(low, high))
        low = max(low, next_low)
    cuts = np.array(cuts, np.float64)

    mapping = [[] for _ in range(len(cuts) + 3)]
    for old_idx, cluster_list in clusters.items():
        # all the items in a cluster should be binned into the same bins
        new_idx = np.searchsorted(cuts, [min(cluster_list)], side="right")[0] + 1
        mapping[new_idx].append(old_idx)

    mapping[0].append(0)
    mapping[-1] = non_float_idxs

    return cuts, mapping, old_min, old_max


def _create_proportional_tensor(axis_weights):
    # take the per-feature weights and distribute them proportionally to each cell in a tensor

    axis_sums = [weights.sum() for weights in axis_weights]

    # Normally you'd expect each axis to sum to the total weight from the model,
    # so normally they should be identical.  We encourage model editing though, so they may
    # not be identical under some edits.  Also, if the model is a DP model then the weights are
    # probably different due to the noise contribution.  Let's take the geometic mean to compensate.
    total_weight = exp(sum(log(axis_sum) for axis_sum in axis_sums) / len(axis_sums))
    axis_percentages = [
        weights / axis_sum for weights, axis_sum in zip(axis_weights, axis_sums)
    ]

    shape = tuple(map(len, axis_percentages))
    n_cells = np.prod(shape)
    tensor = np.empty(n_cells, np.float64)

    # the last index items are next together in flat memory layout
    axis_percentages.reverse()

    for cell_idx in range(n_cells):
        remainder = cell_idx
        frac = 1.0
        for percentages in axis_percentages:
            bin_idx = remainder % len(percentages)
            remainder //= len(percentages)
            frac *= percentages[bin_idx]
        val = frac * total_weight
        tensor[cell_idx] = val
    return tensor.reshape(shape)


def process_bag_terms(intercept, term_scores, bin_weights):
    native = Native.get_native_singleton()
    for scores, weights in zip(term_scores, bin_weights):
        if develop.get_option("purify_result"):
            new_scores, add_impurities, add_intercept = purify(scores, weights)
            # TODO: benchmark if it is better to add new_impurities to the existing model scores,
            #       or better to discard them.  Discarding might be better if we assume the
            #       non-overfit benefit of the lower dimensional interactions has already been extracted.
            scores[:] = new_scores
            intercept += add_intercept
        elif scores.ndim == weights.ndim:
            temp_scores = scores.flatten()  # ndarray.flatten() makes a copy
            temp_weights = weights.flatten()  # ndarray.flatten() makes a copy

            ignored = ~np.isfinite(temp_scores)
            temp_scores[ignored] = 0.0
            temp_weights[ignored] = 0.0

            if temp_weights.sum() != 0:
                mean = native.flat_mean(temp_scores, temp_weights)
                intercept += mean
                scores -= mean
        else:
            for i in range(scores.shape[-1]):
                temp_scores = scores[..., i].flatten()  # ndarray.flatten() makes a copy
                temp_weights = weights.flatten()  # ndarray.flatten() makes a copy

                ignored = ~np.isfinite(temp_scores)
                temp_scores[ignored] = 0.0
                temp_weights[ignored] = 0.0

                if temp_weights.sum() != 0:
                    mean = native.flat_mean(temp_scores, temp_weights)
                    intercept[i] += mean
                    scores[..., i] -= mean

        # We could apply the algorithm proposed by Xuezhou Zhang here, however that algorithm doesn't work
        # for nominal categoricals since there is no concept of adjacency, so for nominal categoricals we
        # need some way to make the multiclass scores identifiable. Making the scores sum to zero, or alternatively
        # choosing to zero the class that has the highest intercept class score would work. Making the scores
        # sum to zero is less arbitrary than Xuezhou's algorithm when it comes to calculating feature/term
        # importance values, so if we use Xuezhou's algorithm then apply it when generating an explanation
        # instead of here which will make calculating importances faster.

        # if the missing/unseen bin has zero weight then whatever number was generated via boosting is
        # effectively meaningless and can be ignored. Set the value to zero for interpretability reasons

        restore_missing_value_zeros(scores, weights)


@dataclass
class _BinSegment:
    start: int
    end: int
    count: float
    total: float
    total_sq: float
    sse: float
    id: int
    active: bool = True
    best_split: Optional[int] = None
    best_gain: float = 0.0


def _greedy_bin_segments(
    counts: np.ndarray,
    sums: np.ndarray,
    sumsqs: np.ndarray,
    max_leaves: int,
    min_leaf: float,
) -> List[Tuple[int, int, float, float]]:
    """Return greedy piecewise-constant segments for ordered bins.

    Parameters
    ----------
    counts : ndarray
        Per-bin sample counts (post-subsampling).
    sums : ndarray
        Per-bin sum of target residuals.
    sumsqs : ndarray
        Per-bin sum of squared residuals.
    max_leaves : int
        Maximum number of segments to grow.
    min_leaf : float
        Minimum total count allowed per segment.

    Returns
    -------
    list of tuples
        Each tuple contains (start, end, total, count) for a leaf segment.
    """

    nbins = counts.shape[0]
    if nbins == 0:
        return []

    if max_leaves is None or max_leaves <= 0:
        max_leaves = nbins if nbins > 0 else 1
    max_leaves = max(1, int(max_leaves))

    if min_leaf is None or min_leaf <= 0:
        min_leaf = 1.0

    if not np.any(counts):
        return []

    prefix_counts = np.zeros(nbins + 1, dtype=float)
    prefix_counts[1:] = np.cumsum(counts, dtype=float)
    prefix_sums = np.zeros(nbins + 1, dtype=float)
    prefix_sums[1:] = np.cumsum(sums, dtype=float)
    prefix_sumsqs = np.zeros(nbins + 1, dtype=float)
    prefix_sumsqs[1:] = np.cumsum(sumsqs, dtype=float)

    def _compute_sse(cnt: float, sm: float, ssq: float) -> float:
        if cnt <= 0.0:
            return 0.0
        sse_val = ssq - (sm * sm) / cnt
        return sse_val if sse_val > 0.0 else 0.0

    segment_ids = count()

    def _build_segment(start: int, end: int) -> _BinSegment:
        seg_cnt = prefix_counts[end] - prefix_counts[start]
        seg_sum = prefix_sums[end] - prefix_sums[start]
        seg_sumsq = prefix_sumsqs[end] - prefix_sumsqs[start]
        seg_sse = _compute_sse(seg_cnt, seg_sum, seg_sumsq)
        segment = _BinSegment(start, end, seg_cnt, seg_sum, seg_sumsq, seg_sse, next(segment_ids))
        _evaluate(segment)
        return segment

    def _evaluate(segment: _BinSegment) -> None:
        segment.best_split = None
        segment.best_gain = 0.0
        if segment.end - segment.start <= 1 or segment.count < 2 * min_leaf:
            return

        left_cnt = 0.0
        left_sum = 0.0
        left_sumsq = 0.0
        right_cnt = segment.count
        right_sum = segment.total
        right_sumsq = segment.total_sq

        for boundary in range(segment.start, segment.end - 1):
            bin_count = counts[boundary]
            bin_sum = sums[boundary]
            bin_sumsq = sumsqs[boundary]
            left_cnt += bin_count
            left_sum += bin_sum
            left_sumsq += bin_sumsq
            right_cnt -= bin_count
            right_sum -= bin_sum
            right_sumsq -= bin_sumsq

            if left_cnt < min_leaf or right_cnt < min_leaf:
                continue

            left_sse = _compute_sse(left_cnt, left_sum, left_sumsq)
            right_sse = _compute_sse(right_cnt, right_sum, right_sumsq)
            gain = segment.sse - (left_sse + right_sse)

            if gain > segment.best_gain:
                segment.best_gain = float(gain)
                segment.best_split = boundary

    segments = [_build_segment(0, nbins)]
    candidate_heap: List[Tuple[float, int, _BinSegment]] = []

    if segments[0].best_split is not None and segments[0].best_gain > 0.0:
        heapq.heappush(
            candidate_heap,
            (-segments[0].best_gain, segments[0].id, segments[0]),
        )

    while len(segments) < max_leaves and candidate_heap:
        _, _, segment = heapq.heappop(candidate_heap)
        if not segment.active or segment.best_split is None or segment.best_gain <= 0.0:
            continue

        segment.active = False
        split_at = segment.best_split + 1
        left_segment = _build_segment(segment.start, split_at)
        right_segment = _build_segment(split_at, segment.end)

        idx = next(i for i, seg in enumerate(segments) if seg.id == segment.id)
        segments[idx : idx + 1] = [left_segment, right_segment]

        for new_segment in (left_segment, right_segment):
            if new_segment.best_split is not None and new_segment.best_gain > 0.0:
                heapq.heappush(
                    candidate_heap,
                    (-new_segment.best_gain, new_segment.id, new_segment),
                )

    return [
        (segment.start, segment.end, segment.total, segment.count)
        for segment in segments
        if segment.count > 0.0 and segment.end > segment.start
    ]


def process_terms(bagged_intercept, bagged_scores, bin_weights, bag_weights):
    native = Native.get_native_singleton()

    n_bags = len(bag_weights)
    n_terms = len(bin_weights)
    for bag_idx in range(n_bags):
        term_scores = [bagged_tensor[bag_idx] for bagged_tensor in bagged_scores]
        intercept = np.atleast_1d(bagged_intercept[bag_idx])
        process_bag_terms(intercept, term_scores, bin_weights)
        bagged_intercept[bag_idx] = intercept[0] if len(intercept) == 1 else intercept
        for term_idx in range(n_terms):
            bagged_scores[term_idx][bag_idx] = term_scores[term_idx]

    term_scores = []
    standard_deviations = []
    for scores in bagged_scores:
        averaged = native.safe_mean(scores, bag_weights)
        term_scores.append(averaged)
        stddevs = native.safe_stddev(scores, bag_weights)
        standard_deviations.append(stddevs)

    intercept = native.safe_mean(bagged_intercept, bag_weights)

    if bagged_intercept.ndim == 2:
        # multiclass
        # pick the class that we're going to zero
        zero_index = np.argmax(intercept)
        intercept -= intercept[zero_index]
        bagged_intercept -= np.expand_dims(bagged_intercept[..., zero_index], -1)

    return intercept, term_scores, standard_deviations


def generate_term_names(feature_names, term_features):
    return [" & ".join(feature_names[i] for i in grp) for grp in term_features]


def generate_term_types(feature_types, term_features):
    return [
        feature_types[grp[0]] if len(grp) == 1 else "interaction"
        for grp in term_features
    ]


def order_terms(term_features, *args):
    if len(term_features) == 0:
        # in Python if only 1 item exists then the item is returned and not a tuple
        if len(args) == 0:
            return []
        return tuple([] for _ in range(len(args) + 1))
    keys = (
        [len(feature_idxs), *sorted(feature_idxs)] for feature_idxs in term_features
    )
    sorted_items = sorted(zip(keys, term_features, *args))
    ret = tuple(list(x) for x in islice(zip(*sorted_items), 1, None))
    # in Python if only 1 item exists then the item is returned and not a tuple
    return ret if len(ret) >= 2 else ret[0]


def remove_extra_bins(term_features, bins):
    # many features are not used in pairs, so we can simplify the model
    # by removing the extra higher interaction level bins

    highest_levels = [0] * len(bins)
    for feature_idxs in term_features:
        for feature_idx in feature_idxs:
            highest_levels[feature_idx] = max(
                highest_levels[feature_idx], len(feature_idxs)
            )

    for bin_levels, i in zip(bins, highest_levels):
        if i != 0:
            if len(bin_levels) == 0:
                raise Exception("Empty bin cannot be used in a term.")

            i = min(i, len(bin_levels)) - 1
            types = set(map(type, bin_levels))

            if len(types) != 1:
                raise Exception("Inconsistent bin types.")

            if next(iter(types)) == dict:
                key = frozenset(bin_levels[i].items())
                i -= 1
                while 0 <= i:
                    if key != frozenset(bin_levels[i].items()):
                        break
                    i -= 1
            else:
                key = tuple(bin_levels[i])
                i -= 1
                while 0 <= i:
                    if key != tuple(bin_levels[i]):
                        break
                    i -= 1
            i += 2
        del bin_levels[i:]


def convert_to_intervals(cuts):  # pragma: no cover
    cuts = np.array(cuts, dtype=np.float64)
    if cuts.size == 0:
        return [(-np.inf, np.inf)]

    if not np.isfinite(cuts).all():
        msg = "cuts must contain only finite numbers"
        raise Exception(msg)

    intervals = [(-np.inf, cuts[0]), *zip(cuts[:-1], cuts[1:]), (cuts[-1], np.inf)]

    if any(higher <= lower for (lower, higher) in intervals):
        msg = "cuts must contain increasing values"
        raise Exception(msg)

    return intervals


def convert_to_cuts(intervals):  # pragma: no cover
    if len(intervals) == 0:
        msg = "intervals must have at least one interval"
        raise Exception(msg)

    if any(len(x) != 2 for x in intervals):
        msg = "intervals must be a list of tuples"
        raise Exception(msg)

    if intervals[0][0] != -np.inf:
        msg = "intervals must start from -inf"
        raise Exception(msg)

    if intervals[-1][-1] != np.inf:
        msg = "intervals must end with inf"
        raise Exception(msg)

    cuts = [lower for (lower, _) in intervals[1:]]
    cuts_verify = [higher for (_, higher) in intervals[:-1]]

    if np.isnan(cuts).any():
        msg = "intervals cannot contain NaN"
        raise Exception(msg)

    if any(x[0] != x[1] for x in zip(cuts, cuts_verify)):
        msg = "intervals must contain adjacent sections"
        raise Exception(msg)

    if any(higher <= lower for lower, higher in zip(cuts, cuts[1:])):
        msg = "intervals must contain increasing sections"
        raise Exception(msg)

    return cuts


def make_bag(y, n_classes, test_size, rng, is_stratified):
    # all test/train splits should be done with this function to ensure that
    # if we re-generate the train/test splits that they are generated exactly
    # the same as before

    if test_size < 0:  # pragma: no cover
        msg = "test_size must be a positive numeric value."
        raise Exception(msg)
    n_samples = len(y)

    if 1 <= test_size:
        if test_size % 1:
            msg = "If test_size >= 1, test_size should be a whole number."
            raise Exception(msg)
        test_size = int(test_size)
    else:
        # prefer training samples
        test_size = floor(n_samples * test_size)

    if test_size == 0:
        return None

    if n_samples <= test_size:
        msg = "The entire dataset cannot exclusively be validation. There must be some training data."
        raise Exception(msg)

    n_train_samples = n_samples - test_size
    native = Native.get_native_singleton()

    if is_stratified:
        bag = native.sample_without_replacement_stratified(
            rng, n_classes, n_train_samples, test_size, y
        )
    else:
        bag = native.sample_without_replacement(rng, n_train_samples, test_size)

    return bag


# Utility functions for InferableEBMRegressor class
def _eigh_pinv_psd(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Symmetric PSD pseudoinverse via eigendecomp with threshold."""
    w, V = np.linalg.eigh(A)
    wi = np.where(w > tol, 1.0 / w, 0.0)
    return (V * wi) @ V.T


def _quantile_edges(x: np.ndarray, n_bins: int, rng: np.random.Generator, eps: float = 1e-12) -> np.ndarray:
    """Quantile cut edges for numeric x; jitter ties slightly to reduce degenerate bins."""
    x = x.astype(float, copy=False)
    if np.any(np.isnan(x)):
        raise ValueError("NaNs in X are not supported in this prototype.")
    jitter = eps * rng.standard_normal(size=x.shape)
    xj = x + jitter
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(xj, qs) if xj.size > 0 else np.array([])
    # ensure strict monotonicity
    if edges.size > 1:
        for i in range(1, edges.size):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], np.inf)
    return edges


def _digitize_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Digitize values using edges with proper bounds."""
    b = np.digitize(x, edges, right=False)
    return np.clip(b, 0, len(edges))


def _auto_bins_for_numeric(
    n: int,
    min_bins_auto: int = 8,
    max_bins_auto: int = 255,
    scheme: str = "quantile",
) -> int:
    """Auto bin count for numeric features."""

    scheme = (scheme or "quantile").lower()
    if scheme in {"cube", "cuberoot", "cubert"}:
        nb = int(2 * round(n ** (1.0 / 3.0)))
        nb = max(min_bins_auto, min(max_bins_auto - 3, nb))
    elif scheme in {"quantile", "count"}:
        nb = max(min_bins_auto, min(max_bins_auto - 3, n))
    else:
        raise ValueError(f"Unknown auto binning scheme '{scheme}'")
    return nb


def _merge_small_bins(bins_j: np.ndarray, min_count: int) -> np.ndarray:
    """Merge bins with fewer than min_count samples by combining with adjacent bins."""
    def _pass(bins_arr):
        if bins_arr.size == 0:
            return bins_arr
        counts = np.bincount(bins_arr, minlength=int(bins_arr.max()) + 1).astype(int)
        for b in range(counts.size):
            if counts[b] > 0 and counts[b] < min_count:
                if b < counts.size - 1:
                    bins_arr[bins_arr == b] = b + 1
                elif b > 0:
                    bins_arr[bins_arr == b] = b - 1
        return bins_arr
    
    bins_j = _pass(bins_j)
    bins_j = _pass(bins_j)
    if bins_j.size > 0:
        bins_j = np.clip(bins_j, 0, int(bins_j.max()))
    return bins_j


def _fit_adaptive_binning(
    X: np.ndarray,
    n_bins: int,
    min_bins_auto: int = 16,
    max_bins_auto: int = 512,
    rng: np.random.Generator = None,
    auto_bins_scheme: str = "quantile",
) -> List:
    """Fit adaptive binning for features using Algorithm 1 approach."""
    if rng is None:
        rng = np.random.default_rng()
        
    n, p = X.shape
    binning_list = []
    
    for j in range(p):
        col = X[:, j]
        if col.dtype.kind in "OUS":   # categorical
            uniq, counts = np.unique(col, return_counts=True)
            order = np.argsort(-counts, kind="mergesort")
            # map in frequency order; fold tail into last bin if needed
            if n_bins > 0:
                nb = min(n_bins, uniq.size)
            else:
                nb = min(max(min_bins_auto, uniq.size), max_bins_auto)
            
            cat2bin = {}
            for rank, idx in enumerate(order):
                cat2bin[uniq[idx]] = rank if rank < nb - 1 else nb - 1
            
            binning_list.append({
                'is_categorical': True,
                'edges': None,
                'cat2bin': cat2bin,
                'n_bins': nb
            })
        else:  # numeric
            nb = n_bins if n_bins > 0 else _auto_bins_for_numeric(
                n,
                min_bins_auto,
                max_bins_auto,
                scheme=auto_bins_scheme,
            )
            edges = _quantile_edges(col, nb, rng)
            binning_list.append({
                'is_categorical': False,
                'edges': edges,
                'cat2bin': None,
                'n_bins': nb
            })
    
    return binning_list


def _assign_bins(X: np.ndarray, binning_list: List) -> List[np.ndarray]:
    """Assign samples to bins using the fitted binning."""
    n, p = X.shape
    bins_list = []
    
    for j in range(p):
        binning = binning_list[j]
        col = X[:, j]
        
        if binning['is_categorical']:
            out = np.empty(n, dtype=np.int64)
            cat2bin = binning['cat2bin']
            for i, val in enumerate(col):
                out[i] = cat2bin.get(val, 0)  # unseen values -> bin 0
            bins_list.append(out)
        else:
            edges = binning['edges']
            out = _digitize_edges(col.astype(float, copy=False), edges)
            bins_list.append(out)
    
    return bins_list


def _post_fit_recenter(intercept: float, term_scores: List[np.ndarray], 
                      bin_weights: List[np.ndarray], train_bins: List[np.ndarray]) -> float:
    """
    Post-fit exact recenter per feature: guarantees ∑ᵢfₖ(xᵢ)=0 constraint.
    
    This ensures the constraint that the paper uses to define the mean-zero subspace H₀.
    """
    n_samples = len(train_bins[0])
    new_intercept = intercept
    
    for k, (scores, weights, bins) in enumerate(zip(term_scores, bin_weights, train_bins)):
        # Compute train-average of feature k contribution
        if len(scores.shape) == 1:  # main effects
            avg_k = float(np.dot(scores, weights) / n_samples) if n_samples > 0 else 0.0
            if avg_k != 0.0:
                term_scores[k] = scores - avg_k
                new_intercept += avg_k
        else:  # interactions - handle per class if needed
            for class_idx in range(scores.shape[-1]):
                avg_k = float(np.dot(scores[..., class_idx], weights) / n_samples) if n_samples > 0 else 0.0
                if avg_k != 0.0:
                    term_scores[k][..., class_idx] = scores[..., class_idx] - avg_k
                    new_intercept += avg_k
    
    return new_intercept
