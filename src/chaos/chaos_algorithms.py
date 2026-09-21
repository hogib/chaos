"""Core algorithms for chaotic time-series analysis.

Adapted from S. Sarwar, A. Likens, N. Stergiou, S. Mastorakis,
"A nonlinear analysis software toolkit for biomechanical data",
arXiv:2311.06723 (2023), https://arxiv.org/abs/2311.06723

Contents:
    wolf_lye_core, rosenstein_lye_core, fit_log_divergence, samp_ent_core,
    corrdim_core, fnn_core, ami_core.
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from scipy.stats import linregress


def wolf_lye_core(x, fs, tau, dim, evolve):
    """Computes the maximum Lyapunov exponent via the Wolf (1985) method.

    Args:
        x: Normalized signal (1-D array).
        fs: Sampling frequency in Hz.
        tau: Time delay.
        dim: Embedding dimension.
        evolve: Number of evolution steps.

    Returns:
        Tuple ``(out_matrix, LyE)`` where ``out_matrix`` is the per-step detail
        table and ``LyE`` is the scalar maximum Lyapunov exponent.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    SCALEMX = (np.max(x) - np.min(x)) / 10
    ANGLMX = 30 * np.pi / 180
    DT = 1.0 / fs
    ITS, distSUM = 0, 0.0

    N = len(x)
    M = N - (dim - 1) * tau
    NPT = M - evolve
    if NPT < 1:
        return np.zeros((0, 9), dtype=np.float64), np.nan

    Y = np.empty((M, dim), dtype=np.float64)
    for i in range(dim):
        Y[:, i] = x[i * tau: M + i * tau]
    Y = Y[: NPT + evolve, :]
    out = np.zeros((int(np.floor(NPT / evolve) + 1), 9), dtype=np.float64)

    # Excluded candidates are marked with inf rather than nan so the searches
    # below can use argmin, which is several times faster than nanargmin.
    Ydisti = np.sqrt(np.einsum("ij,ij->i", Y[0] - Y[:NPT], Y[0] - Y[:NPT]))
    Ydisti = np.where(Ydisti > 0, Ydisti, np.inf)
    Ydisti[np.arange(max(0, -10), min(NPT, 11))] = np.inf
    current_point_pair = int(np.argmin(Ydisti))
    if not np.isfinite(Ydisti[current_point_pair]):
        return out, np.nan

    thbest, OUTMX, LyE = 0, SCALEMX, 0.0
    for i in range(0, NPT, evolve):
        ep = i + evolve
        safe = current_point_pair + evolve < len(Y) and ep < len(Y)
        pair_ep = (
            current_point_pair + evolve if safe else current_point_pair + evolve - 1
        )

        start_dist = np.linalg.norm(Y[i] - Y[current_point_pair])
        end_dist = np.linalg.norm(Y[ep] - Y[pair_ep])

        # A zero separation at either end makes log2(end/start) infinite and
        # would poison the running mean for every remaining step, so the step
        # is skipped rather than accumulated.
        if start_dist > 0 and end_dist > 0:
            distSUM += np.log2(end_dist / start_dist) / (evolve * DT)
            ITS += 1
            LyE = distSUM / ITS
        out[int(np.floor(i / evolve))] = [
            ITS, i, current_point_pair, start_dist, end_dist, LyE, OUTMX,
            thbest * 180 / np.pi, ANGLMX * 180 / np.pi,
        ]

        if end_dist < SCALEMX:
            current_point_pair += evolve
            if current_point_pair > NPT:
                current_point_pair -= evolve
                current_point_pair, ANGLMX, thbest, OUTMX = _wolf_next_point(
                    1, Y, i, current_point_pair, NPT, evolve, SCALEMX, ANGLMX
                )
        else:
            current_point_pair, ANGLMX, thbest, OUTMX = _wolf_next_point(
                0, Y, i, current_point_pair, NPT, evolve, SCALEMX, ANGLMX
            )

    return out, LyE


def _wolf_next_point(flag, Y, current_point, current_point_pair, NPT, evolve,
                     SCALEMX, ANGLMX):
    """Selects the next nearest neighbour for the Wolf algorithm.

    Args:
        flag: 0 to search for a replacement within the cone, 1 to force one.
        Y: Embedded trajectory matrix, shape ``(NPT + evolve, dim)``.
        current_point: Current reference index.
        current_point_pair: Current nearest-neighbour index.
        NPT: Number of points usable for distance comparisons.
        evolve: Evolution step.
        SCALEMX: Maximum allowed separation distance.
        ANGLMX: Maximum allowed angular deviation (radians).

    Returns:
        ``(next_pt, ANGLMX, thbest, SCALEMX)`` — the new neighbour index, the
        updated angle bound, the chosen angle, and the (unchanged) scale bound.
    """
    ep = current_point + evolve
    diff = Y[ep] - Y[:NPT]
    Yd = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    Yd[np.arange(max(0, ep - 10), min(NPT, ep + 11))] = np.inf

    safe = current_point_pair + evolve < len(Y)
    end_v = (
        Y[current_point_pair + evolve]
        if safe
        else Y[current_point_pair + evolve - 1]
    )
    Vcurr = Y[ep] - end_v
    end_dist = np.linalg.norm(Vcurr)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.abs(diff @ Vcurr / (Yd * end_dist))

    # theta >= ANGLMX is equivalent to cos_t <= cos(ANGLMX) since arccos is
    # monotone decreasing; testing the cosine avoids an arccos over every point.
    thbest = ANGLMX
    with np.errstate(invalid="ignore"):
        pot = np.where((Yd > 0) & (cos_t > np.cos(ANGLMX)), Yd, np.inf)
    next_pt = -1

    if flag == 0:
        cand = int(np.argmin(pot))
        if pot[cand] <= SCALEMX:
            ANGLMX = 30 * np.pi / 180
            thbest = float(np.arccos(np.clip(cos_t[cand], -1.0, 1.0)))
            next_pt = cand

    if next_pt == -1:
        tmp = np.where(Yd > 0, Yd, np.inf)
        fallback = int(np.argmin(tmp))
        if not np.isfinite(tmp[fallback]):
            return current_point_pair, ANGLMX, thbest, SCALEMX
        next_pt = fallback
        thbest = ANGLMX

    return next_pt, ANGLMX, thbest, SCALEMX


def _nearest_neighbours(Y, band):
    """Finds, for every point, its nearest neighbour outside a Theiler band.

    Args:
        Y: Embedded trajectory, shape ``(M, dim)``.
        band: Minimum index separation ``|i - j|`` a neighbour must exceed.

    Returns:
        Integer array of length ``M`` holding the neighbour index of each point.
    """
    M = Y.shape[0]
    # At most 2*band+1 candidates can be rejected by the band, so asking for
    # one more than that guarantees a valid neighbour is in the result set.
    k = min(M, 2 * band + 2)
    _, idx = cKDTree(Y).query(Y, k=k)
    idx = np.asarray(idx).reshape(M, -1)

    rows = np.arange(M)
    outside = np.abs(idx - rows[:, np.newaxis]) > band
    first = outside.argmax(axis=1)
    first[~outside.any(axis=1)] = 0
    return idx[rows, first].astype(np.int64)


def _mean_log_divergence(Y, neighbours):
    """Averages ``log`` of the neighbour separation over all trajectory pairs.

    Equivalent to building the full ``(M, M)`` divergence matrix and taking the
    row means of its positive entries, but accumulates in ``O(M)`` memory.

    Args:
        Y: Embedded trajectory, shape ``(M, dim)``.
        neighbours: Nearest-neighbour index per point (see
            :func:`_nearest_neighbours`).

    Returns:
        Array of length ``M`` with the mean log divergence at each step.
    """
    M = Y.shape[0]
    sum_log = np.zeros(M, dtype=np.float64)
    count = np.zeros(M, dtype=np.int64)

    for i in range(M):
        nn = int(neighbours[i])
        end = min(M - i, M - nn)
        if end <= 0:
            continue
        delta = Y[i : i + end] - Y[nn : nn + end]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        positive = dist > 0
        # Non-positive entries are exactly 0 and log leaves them untouched, so
        # they contribute nothing to the sum and need no extra masking pass.
        np.log(dist, out=dist, where=positive)
        sum_log[:end] += dist
        count[:end] += positive

    return np.divide(
        sum_log, count, out=np.zeros(M, dtype=np.float64), where=count > 0
    )


def fit_log_divergence(ave_ln_div, fs, mean_period, start, end):
    """Least-squares slope of the mean log-divergence curve.

    This is the single definition of the Rosenstein fit window used by both
    :func:`rosenstein_lye_core` and the per-window wrapper, so the two cannot
    drift apart. ``start`` and ``end`` are expressed in *mean periods*, and the
    returned slope is therefore in units of 1/period.

    Args:
        ave_ln_div: Mean log-divergence curve (1-D array).
        fs: Sampling frequency in Hz.
        mean_period: Reciprocal of the dominant frequency in seconds.
        start: Fit-window start, in mean periods.
        end: Fit-window end, in mean periods.

    Returns:
        Tuple ``(slope, r_squared, n_points)``. ``slope`` and ``r_squared`` are
        ``nan`` and ``n_points`` is ``0`` when the window is unusable.
    """
    ave = np.asarray(ave_ln_div, dtype=np.float64).ravel()
    nz = int(np.count_nonzero(ave))

    lo = int(round(start * mean_period * fs))
    hi = int(round(end * mean_period * fs))
    if lo < 0 or hi <= lo or hi > nz:
        return np.nan, np.nan, 0

    t = np.arange(lo, hi + 1, dtype=np.float64) / fs / mean_period
    y = ave[lo : hi + 1]
    if t.size < 2:
        return np.nan, np.nan, int(t.size)

    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return float(slope), r2, int(t.size)


def rosenstein_lye_core(x, fs, tau, dim, slope, mean_period):
    """Computes the short-term Lyapunov exponent via Rosenstein (1993).

    Nearest neighbours come from a :class:`scipy.spatial.cKDTree` query rather
    than a dense distance matrix, and the divergence curve is accumulated in
    linear memory instead of an ``(M, M)`` matrix.

    Args:
        x: Normalized signal (1-D array).
        fs: Sampling frequency in Hz.
        tau: Time delay.
        dim: Embedding dimension.
        slope: Fit window in mean periods. Either ``[short_start, short_end]``
            or ``[short_start, short_end, long_start, long_end]``; only the
            short window is used here.
        mean_period: Reciprocal of the dominant frequency in seconds.

    Returns:
        ``[lye_short, divergence_matrix]`` where ``divergence_matrix`` has rows
        ``[index, nearest_neighbour_index, average_log_divergence]``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    M = N - (dim - 1) * tau
    if M < 2:
        return [np.nan, np.zeros((3, 0), dtype=np.float64)]

    Y = np.empty((M, dim), dtype=np.float64)
    for j in range(dim):
        Y[:, j] = x[j * tau : M + j * tau]

    neighbours = _nearest_neighbours(Y, band=(dim - 1) * tau)
    ave_ln_div = _mean_log_divergence(Y, neighbours)

    lye, _, _ = fit_log_divergence(ave_ln_div, fs, mean_period, slope[0], slope[1])
    return [
        lye,
        np.vstack((np.arange(M, dtype=np.float64), neighbours, ave_ln_div)),
    ]


def samp_ent_core(data, m, r):
    """Computes the Sample Entropy of a signal.

    Uses :func:`scipy.spatial.distance.pdist`, which evaluates each unordered
    pair once instead of filling a full square matrix. Self-matches are absent
    by construction, so no diagonal has to be masked out.

    Args:
        data: Normalized signal (1-D array).
        m: Template length.
        r: Tolerance as a multiple of the signal's standard deviation.

    Returns:
        Sample entropy (float) or ``nan`` if it cannot be computed.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    L = len(data) - m
    if L < 2:
        return np.nan

    R = r * np.std(data, ddof=1)
    idx = np.arange(m + 1)[np.newaxis, :] + np.arange(L)[:, np.newaxis]
    templates = data[idx]

    # Each unordered pair is counted twice to keep the same normalisation as
    # the ordered-pair form; the A/B ratio is unaffected either way.
    Bm = 2 * np.count_nonzero(
        pdist(templates[:, :m], metric="chebyshev") <= R
    ) / (L * L)
    Am = 2 * np.count_nonzero(
        pdist(templates, metric="chebyshev") <= R
    ) / (L * L)
    if Am == 0 or Bm == 0:
        return np.nan
    return float(-np.log(Am / Bm))


def corrdim_core(x, tau, de):
    """Computes the correlation dimension of a signal.

    Args:
        x: Normalized signal (1-D array).
        tau: Time delay.
        de: Embedding dimension.

    Returns:
        Correlation dimension (slope of the log-log correlation integral).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    n = N - (de - 1) * tau

    Y = np.zeros((de, n))
    for i in range(de):
        Y[i, :] = x[i * tau: N - (de - 1 - i) * tau]

    bins = 200
    k = de * tau

    D = cdist(Y[:, k + 1 : n].T, Y[:, 0 : n - k - 1].T, metric="euclidean")
    if D.size == 0:
        return 0.0

    # D is not needed afterwards, so sort the underlying buffer in place and
    # skip an extra full-size copy. Sorting first also makes the radius range
    # below a pair of lookups rather than two more passes over the matrix.
    flat_sorted = D.reshape(-1)
    flat_sorted.sort()

    eps1 = float(flat_sorted[-1])
    # A degenerate (constant) signal collapses every pairwise distance to zero;
    # there is no log-log range to fit and log(0) would warn on every window.
    if not np.isfinite(eps1) or eps1 <= 0:
        return 0.0

    # The blocks compared above overlap, so some pairs are a point against
    # itself and the minimum distance is always exactly 0. Falling back to
    # machine epsilon there would anchor the radius grid to an absolute
    # constant instead of the data, leaving almost every bin below the
    # smallest real distance (and making the result depend on the signal's
    # units). The floor is the smallest *positive* distance instead.
    first_positive = int(np.searchsorted(flat_sorted, 0.0, side="right"))
    if first_positive >= flat_sorted.size:
        return 0.0
    eps2 = float(flat_sorted[first_positive])

    epsilon = np.exp(np.linspace(np.log(eps2), np.log(eps1), bins))
    cumCI = np.searchsorted(flat_sorted, epsilon, side="right").astype(float)

    denom = (n - k) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        CI = np.where(
            cumCI > 0,
            np.log(cumCI / denom) + np.log(n - k - 1),
            np.nan,
        )
    eps = np.log(epsilon)

    fin = np.isfinite(CI)
    if not np.any(fin):
        return 0.0

    lo, hi = np.min(CI[fin]), np.max(CI[fin])
    mid = (lo + hi) / 2
    q = (hi - lo) / 4
    idx = np.where((CI > mid) & (CI < mid + q) & fin)[0]
    if len(idx) < 2:
        return 0.0

    slope, *_ = linregress(eps[idx], CI[idx])
    return float(slope)


def fnn_core(data, tau, max_dim, rtol=15, atol=2, speed=1):
    """Determines the embedding dimension via False Nearest Neighbours.

    Args:
        data: Time series (1-D array).
        tau: Time delay.
        max_dim: Maximum embedding dimension to search.
        rtol: Criterion-1 threshold (recommended: 15).
        atol: Criterion-2 threshold (recommended: 2).
        speed: 1 to stop at the first minimum, 0 to search up to ``max_dim``.

    Returns:
        Tuple ``(dE, dim)`` where ``dE`` is the array of false-neighbour
        ratios and ``dim`` is the found dimension.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data) - tau * max_dim
    RA = np.std(data)
    m_search = 2

    z = np.array([data[0:n]])
    y = np.array([[]])
    indx = np.arange(0, n)
    dim = np.array([])
    dE = np.zeros((max_dim, 1))

    for j in range(max_dim):
        y = np.array([np.append(y, z)]) if j == 0 else np.vstack([y, z])
        z = np.array([data[tau * (j + 1): n + tau * (j + 1)]])
        L = np.zeros(n)

        y_model, z_model, sort_list, node_list = _kd_part(y, z, 512)

        for i in range(len(indx)):
            yq = np.array(y[:, indx[i]])
            b_upper = np.inf * np.ones(np.size(yq))
            b_lower = -b_upper
            pqd = np.inf * np.ones((1, m_search))
            pqr, pqz = np.array([]), np.array([])

            pqd, y_model, z_model, _, _, pqz, _, _, sort_list, node_list = _kd_search(
                0, m_search, yq, pqd, y_model, z_model, 0, pqr, pqz,
                b_upper, b_lower, sort_list, node_list,
            )

            distance = pqz[0] - pqz[1]
            if np.abs(distance) > pqd[1] * rtol:
                L[i] = 1
            if np.sqrt(pqd[1] ** 2 + distance ** 2) / RA > atol:
                L[i] = 1

        dE[j] = np.sum(L) / n

        if speed == 1:
            if j >= 2 and dE[j - 2] > dE[j - 1] and dE[j - 1] < dE[j]:
                dim = j - 1
                break
            if j >= 1 and np.abs(dE[j] - dE[j - 1]) <= 0.001:
                dim = j - 1
                break
            if dE[j] == 0:
                dim = j
                break

    if speed == 0:
        for i in range(len(dE)):
            if np.abs(dE[i - 1] - dE[i]) <= 0.001:
                dim = i - 1
                break

    return dE, int(dim) + 1 if np.size(dim) > 0 else max_dim


def _kd_part(y_in, z_in, bin_size):
    """Builds a kd-tree database used by :func:`_kd_search`.

    Args:
        y_in: State matrix, shape ``(dim, n)``.
        z_in: Target matrix, shape ``(dim, n)``.
        bin_size: Minimum segment size below which the segment is a leaf.

    Returns:
        Tuple ``(y_model, z_model, sort_list, node_list)`` describing the
        partitioned data and tree structure.
    """
    y_model = y_in.copy()
    z_model = z_in.copy()
    d, n_y = y_model.shape
    node_list = np.array([[0, n_y, 0, 0]])
    sort_list = np.array([[0, 0]], dtype=float)
    node, last = 0, 0

    while node <= last:
        segment = np.arange(node_list[node, 0], node_list[node, 1])
        rg = np.amax(y_model, axis=1) - np.amin(y_model, axis=1)

        if np.max(rg) > 0 and len(segment) >= bin_size:
            index = np.argsort(rg)
            yt = y_model[:, segment]
            zt = z_model[:, segment]
            y_index = np.argsort(yt[index[d - 1]])
            y_sort = np.sort(yt[index[d - 1]])
            tlen = len(y_sort)

            cut = (
                y_sort[(tlen + 1) // 2]
                if tlen % 2
                else (y_sort[tlen // 2] + y_sort[tlen // 2 + 1]) / 2
            )
            L = y_sort <= cut
            if np.sum(L) == tlen:
                L = y_sort < cut
                cut = (cut + np.max(y_sort[L])) / 2

            y_model[:, segment] = yt[:, y_index]
            z_model[:, segment[0]: segment[0] + len(segment)] = zt[:, y_index]

            sort_list[node, :] = [index[d - 1], cut]
            node_list[node, 2] = last + 1
            node_list[node, 3] = last + 2
            last += 2
            nl = np.sum(L)
            node_list = np.vstack([
                node_list,
                [segment[0], segment[0] + nl - 1, 0, 0],
                [segment[0] + nl, segment[-1], 0, 0],
            ])
            sort_list = np.vstack([sort_list, [[0, 0], [0, 0]]])

        node += 1

    return y_model, z_model, sort_list, node_list


def _kd_search(node, m_search, yq, pqd, y_model, z_model,
               L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list):
    """Nearest-neighbour search over a kd-tree produced by :func:`_kd_part`.

    Args:
        node: Index of the current kd-tree node.
        m_search: Number of nearest neighbours to keep.
        yq: Query vector.
        pqd: Running distance heap.
        y_model: Partitioned state matrix.
        z_model: Partitioned target matrix.
        L_done: Flag set to 1 when the search can stop early.
        pqr: Running neighbour heap (state).
        pqz: Running neighbour heap (target).
        b_upper: Upper bounding hyperrectangle.
        b_lower: Lower bounding hyperrectangle.
        sort_list: kd-tree split descriptors.
        node_list: kd-tree node descriptors.

    Returns:
        Updated versions of ``(pqd, y_model, z_model, L_done, pqr, pqz,
        b_upper, b_lower, sort_list, node_list)``.
    """
    if L_done == 1:
        return pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list

    if node_list[node, 2] == 0:
        yi = node_list[node, 0:2]
        yt = y_model[:, yi[0]: yi[1]]
        zt = z_model[:, yi[0]: yi[1]]
        d = len(yq)
        dist = np.sqrt(np.sum((yt[:d, :] - yq[:d, np.newaxis]) ** 2, axis=0))

        pqd = np.append(dist, pqd)
        pqr = np.append(yt, pqr)
        pqz = np.append(zt, pqz)
        idx = np.argsort(pqd)
        pqd = np.sort(pqd)

        length = pqz.shape[0]
        if len(idx) > length:
            pqr = pqr[idx[0:length]]
            pqz = pqz[idx[0:length]]
        else:
            pqr = pqr[idx]
            pqz = pqz[idx]

        if len(pqd) > m_search:
            pqd = pqd[:m_search]
        if pqz.shape[0] > m_search:
            pqr = pqr[:m_search]
            pqz = pqz[:m_search]

        if any(
            (np.abs(yq - b_lower) <= pqd[m_search - 1])
            | (np.abs(yq - b_upper) <= pqd[m_search - 1])
        ):
            L_done = 1

        return pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list

    disc = int(sort_list[node, 0])
    part = sort_list[node, 1]

    def _descend(child, bu_val, bl_val, is_upper):
        nonlocal b_upper, b_lower
        if is_upper:
            old = b_upper[disc]
            b_upper[disc] = bu_val
        else:
            old = b_lower[disc]
            b_lower[disc] = bl_val
        res = _kd_search(
            child, m_search, yq, pqd, y_model, z_model,
            L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list,
        )
        if is_upper:
            b_upper[disc] = old
        else:
            b_lower[disc] = old
        return res

    if yq[disc] <= part:
        (pqd, y_model, z_model, L_done, pqr, pqz,
         b_upper, b_lower, sort_list, node_list) = _descend(
            node_list[node, 2], part, b_lower[disc], True
        )
        if not L_done and _overlap(yq, m_search, pqd, b_upper, b_lower):
            b_lower_old = b_lower[disc]
            b_lower[disc] = part
            (pqd, y_model, z_model, L_done, pqr, pqz,
             b_upper, b_lower, sort_list, node_list) = _kd_search(
                node_list[node, 3], m_search, yq, pqd, y_model, z_model,
                L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list,
            )
            b_lower[disc] = b_lower_old
    else:
        (pqd, y_model, z_model, L_done, pqr, pqz,
         b_upper, b_lower, sort_list, node_list) = _descend(
            node_list[node, 3], b_upper[disc], part, False
        )
        if not L_done and _overlap(yq, m_search, pqd, b_upper, b_lower):
            b_upper_old = b_upper[disc]
            b_upper[disc] = part
            (pqd, y_model, z_model, L_done, pqr, pqz,
             b_upper, b_lower, sort_list, node_list) = _kd_search(
                node_list[node, 2], m_search, yq, pqd, y_model, z_model,
                L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list,
            )
            b_upper[disc] = b_upper_old

    return pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list


def _overlap(yq, m_search, pqd, b_upper, b_lower):
    """Tests whether a hypersphere intersects a kd-tree bounding box.

    Args:
        yq: Query vector.
        m_search: Number of nearest neighbours being tracked.
        pqd: Distance heap (only the last entry matters).
        b_upper: Upper bounding hyperrectangle.
        b_lower: Lower bounding hyperrectangle.

    Returns:
        ``1`` if the sphere intersects the box, ``0`` otherwise.
    """
    dist = pqd[m_search - 1] ** 2
    s = 0.0
    for i in range(len(yq)):
        if yq[i] < b_lower[i]:
            s += (yq[i] - b_lower[i]) ** 2
            if s > dist:
                return 0
        elif yq[i] > b_upper[i]:
            s += (yq[i] - b_upper[i]) ** 2
            if s > dist:
                return 0
    return 1


def ami_core(data, max_lag, n_bins=0):
    """Computes the Average Mutual Information (AMI) function.

    Used to select a phase-space delay as the first AMI minimum.

    Args:
        data: Time series (1-D array).
        max_lag: Maximum lag step.
        n_bins: Histogram bin count. If ``0``, selects automatically via
            Scott's (1979) rule.

    Returns:
        Tuple ``(tau_matrix, v_AMI)``. ``tau_matrix`` holds the first-minimum
        points as ``[lag, ami_value]`` rows. ``v_AMI`` is a ``(2, max_lag)``
        array with row 0 = lag and row 1 = AMI value.
    """
    eps = np.finfo(float).eps
    data = np.asarray(data, dtype=np.float64).ravel()
    N = len(data)

    if n_bins == 0:
        bins = int(np.ceil(
            (np.max(data) - np.min(data))
            / (3.49 * np.nanstd(data) * N ** (-1 / 3))
        ))
    else:
        bins = int(n_bins)

    d = data - data.min()
    y = np.array(np.floor(d / (d.max() / (bins - eps))), dtype=int)

    overlap = N - max_lag
    increment = 1.0 / overlap
    pA = sp.csr_matrix(
        (np.full(overlap, increment), (y[:overlap], np.ones(overlap, dtype=int)))
    ).toarray()[:, 1]

    v = np.zeros((2, max_lag))

    for lag in range(max_lag):
        v[0, lag] = lag
        pB = sp.csr_matrix(
            (np.full(overlap, increment), (y[lag: overlap + lag], np.ones(overlap, dtype=int)))
        ).toarray()[:, 1]

        pAB = sp.csr_matrix(
            (np.full(overlap, increment), (y[:overlap], y[lag: overlap + lag]))
        )
        A, B = np.nonzero(pAB)
        AB = pAB.data
        denom = np.multiply(pA[A], pB[B])
        denom[denom == 0] = eps
        v[1, lag] = float(np.sum(AB * np.log2(AB / denom)))

    tau_rows = []
    for i in range(1, v.shape[1] - 1):
        if v[1, i - 1] >= v[1, i] and v[1, i] <= v[1, i + 1]:
            tau_rows.append([i, v[1, i]])

    tau_matrix = np.array(tau_rows) if tau_rows else np.full((1, 2), -1.0)

    threshold = 0.2 * v[1, 0]
    for i in range(v.shape[1]):
        if v[1, i] < threshold:
            if len(tau_matrix) > 0 and tau_matrix[0, 0] != -1:
                tau_matrix[0, 1] = i
            break

    return tau_matrix, v
