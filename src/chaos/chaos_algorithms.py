"""Core algorithms for chaotic time-series analysis.

Adapted from S. Sarwar, A. Likens, N. Stergiou, S. Mastorakis,
"A nonlinear analysis software toolkit for biomechanical data",
arXiv:2311.06723 (2023), https://arxiv.org/abs/2311.06723

Contents:
    wolf_lye_core, rosenstein_lye_core, samp_ent_core, corrdim_core,
    fnn_core, ami_core.
"""

import numpy as np
import numpy.polynomial.polynomial as _poly
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
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
    Y = np.empty((M, dim), dtype=np.float64)
    for i in range(dim):
        Y[:, i] = x[i * tau: M + i * tau]

    NPT = N - (dim - 1) * tau - evolve
    Y = Y[: NPT + evolve, :]
    out = np.zeros((int(np.floor(NPT / evolve) + 1), 9), dtype=object)

    Ydisti = np.sqrt(np.sum((Y[0] - Y[:NPT]) ** 2, axis=1))
    excl = np.arange(max(0, -10), min(NPT, 11))
    Ydisti[Ydisti <= 0] = np.nan
    Ydisti[excl] = np.nan
    current_point_pair = int(np.nanargmin(Ydisti))

    thbest, OUTMX, LyE = 0, SCALEMX, 0.0
    for i in range(0, NPT, evolve):
        ep = i + evolve
        safe = current_point_pair + evolve < len(Y) and ep < len(Y)
        pair_ep = (
            current_point_pair + evolve if safe else current_point_pair + evolve - 1
        )

        start_dist = np.linalg.norm(Y[i] - Y[current_point_pair])
        end_dist = np.linalg.norm(Y[ep] - Y[pair_ep])

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
    Yd = np.sqrt(np.sum(diff ** 2, axis=1))
    excl = np.arange(max(0, ep - 10), min(NPT, ep + 11))
    Yd[excl] = np.nan

    safe = current_point_pair + evolve < len(Y)
    end_v = (
        Y[current_point_pair + evolve]
        if safe
        else Y[current_point_pair + evolve - 1]
    )
    Vcurr = Y[ep] - end_v
    end_dist = np.linalg.norm(Vcurr)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.abs(np.sum(Vcurr * diff, axis=1) / (Yd * end_dist))
        theta = np.arccos(np.clip(cos_t, -1.0, 1.0))

    thbest = ANGLMX
    pot = np.where((Yd <= 0) | (theta >= ANGLMX), np.nan, Yd)
    next_pt = -1

    if flag == 0:
        order = np.argsort(np.where(np.isnan(pot), np.inf, pot))
        cand = int(order[0])
        if not np.isnan(pot[cand]) and pot[cand] <= SCALEMX:
            ANGLMX = 30 * np.pi / 180
            thbest = float(np.abs(theta[cand]))
            next_pt = cand
        else:
            flag = 1

    if next_pt == -1:
        tmp = np.where(Yd <= 0, np.nan, Yd)
        next_pt = int(np.nanargmin(tmp))
        thbest = ANGLMX

    return next_pt, ANGLMX, thbest, SCALEMX


def rosenstein_lye_core(x, fs, tau, dim, slope, mean_period):
    """Computes the short-term Lyapunov exponent via Rosenstein (1993).

    Nearest neighbours are found with a :class:`scipy.spatial.cKDTree`
    query instead of a dense distance matrix, which is much faster for
    moderately long signals.

    Args:
        x: Normalized signal (1-D array).
        fs: Sampling frequency in Hz.
        tau: Time delay.
        dim: Embedding dimension.
        slope: Slope window ``[short_start, short_end]`` in periods.
        mean_period: Reciprocal of the dominant frequency in seconds.

    Returns:
        ``[lye_short, divergence_matrix]`` where ``divergence_matrix`` has rows
        ``[index, nearest_neighbour_index, average_log_divergence]``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    M = N - (dim - 1) * tau
    Y = np.empty((M, dim), dtype=np.float64)
    for j in range(dim):
        Y[:, j] = x[j * tau: M + j * tau]

    band = (dim - 1) * tau
    k_query = min(M, 2 * band + 2)
    tree = cKDTree(Y)
    _, idx = tree.query(Y, k=k_query)
    if idx.ndim == 1:
        idx = idx[:, np.newaxis]

    IND2 = np.empty(M, dtype=np.int64)
    for i in range(M):
        row = idx[i]
        valid = np.abs(row - i) > band
        IND2[i] = row[int(np.argmax(valid))] if valid.any() else row[0]

    DM = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        nn = IND2[i]
        end = min(M - i, M - nn)
        if end > 0:
            DM[:end, i] = np.sqrt(
                np.sum((Y[i : i + end] - Y[nn : nn + end]) ** 2, axis=1)
            )

    AveLnDiv = np.zeros(M, dtype=np.float64)
    for i in range(M):
        pos = DM[i, DM[i, :] > 0]
        if len(pos):
            AveLnDiv[i] = np.mean(np.log(pos))

    time = np.arange(len(AveLnDiv)) / fs / mean_period
    nz = int(np.count_nonzero(AveLnDiv))

    def _fit(lo, hi):
        if hi <= nz:
            c = _poly.polyfit(time[lo : hi + 1], AveLnDiv[lo : hi + 1], 1)
            return float(c[1])
        return np.nan

    sL = [
        0 if slope[0] == 0 else round(slope[0] * mean_period * fs),
        round(slope[1] * mean_period * fs),
    ]

    return [
        _fit(sL[0], sL[1]),
        np.vstack((np.arange(M), IND2, AveLnDiv)),
    ]


def samp_ent_core(data, m, r):
    """Computes the Sample Entropy of a signal.

    Uses :func:`scipy.spatial.distance.cdist` with the Chebyshev metric rather
    than a 3-D broadcast, which is significantly faster.

    Args:
        data: Normalized signal (1-D array).
        m: Template length.
        r: Tolerance as a multiple of the signal's standard deviation.

    Returns:
        Sample entropy (float) or ``nan`` if it cannot be computed.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    R = r * np.std(data, ddof=1)
    L = len(data) - m
    idx = np.arange(m + 1)[np.newaxis, :] + np.arange(L)[:, np.newaxis]
    templates = data[idx]

    dist_m1 = cdist(templates, templates, metric="chebyshev")
    dist_m = cdist(templates[:, :m], templates[:, :m], metric="chebyshev")

    np.fill_diagonal(dist_m, R + 1.0)
    np.fill_diagonal(dist_m1, R + 1.0)

    Bm = (dist_m <= R).sum() / (L * L)
    Am = (dist_m1 <= R).sum() / (L * L)
    if Am == 0 or Bm == 0:
        return np.nan
    return float(-np.log(Am / Bm))


def corrdim_core(x, tau, de):
    """Computes the correlation dimension of a signal.

    The pairwise distance matrix is built once with :func:`scipy.spatial.distance.cdist`
    instead of recomputing it twice in Python loops.

    Args:
        x: Normalized signal (1-D array).
        tau: Time delay.
        de: Embedding dimension.

    Returns:
        Correlation dimension (slope of the log–log correlation integral).
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

    eps1 = float(D.max())
    eps2 = float(D.min())
    if eps2 == 0:
        eps2 = np.finfo(float).eps

    epsilon = np.exp(np.linspace(np.log(eps2), np.log(eps1), bins))
    flat_sorted = np.sort(D.ravel())
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
