# ----------------- LOAD PACKAGES -----------------
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad          # kept as fallback only
from scipy.special import spence  # used for closed-form I / Xi
from typing import Dict, Any, Callable, Optional, List, Tuple, Union, Sequence
from collections import Counter, OrderedDict
import pickle, gzip, os
import matplotlib.pyplot as plt
import math, time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm, ListedColormap
import tqdm

def build_E_marks(P1_vals,P2_vals,P3_vals):
    E_marks: List[Dict[str, Any]] = [] 
    for e1, p1 in P1_vals.items():
        for e2, p2 in P2_vals.items():
            for e3, p3 in P3_vals.items():
                nu = float(p1 * p2 * p3)
                if e3 == -1:
                    E_marks.append({'kind': 'market', 'eta': float(e1), 'rho': None, 'nu': float(nu)})
                elif e3 == 1:
                    E_marks.append({'kind': 'limit_buy', 'eta': None, 'rho': float(e2), 'nu': float(nu)})
                elif e3 == 2:
                    E_marks.append({'kind': 'limit_sell', 'eta': None, 'rho': float(e2), 'nu': float(nu)})
                else:
                    raise ValueError('Unexpected e3')
    return E_marks

def f(liquidity: float, theta_f: float, kappa: float, lam_ref: float) -> float:
    """Arrival intensity for limit orders"""
    return theta_f * np.exp(-kappa * (float(max(liquidity, 0.0))-lam_ref))

def g(liquidity: float, theta_g: float, kappa: float, lam_ref: float) -> float:
    """Arrival intensity market orders"""
    return theta_g * np.exp(kappa * (float(max(liquidity, 0.0))-lam_ref))

# ----------------- PRICE IMPACT & IMPACT COST -----------------
# iota(z, lam, iota0) = 0.01 * ln(1 + lam - z) / (lam - z)  for lam > 0, z < lam
#                       iota0                                  for lam <= 0 or z >= lam
# The log-kernel coefficient is always 0.01; iota0 is only the flat fallback.
# Closed-form anti-derivative uses the dilogarithm Li_2(x) = spence(1 - x).

import math
from scipy.special import spence  # spence(u) = Li2(1-u)

def _Li2(x: float) -> float:
    """Standard Dilogarithm Li2(x) using SciPy's spence."""
    return float(spence(1.0 - float(x)))

def I(delta: float, lam: float, iota0: float, iota1: float = 0.01) -> float:
    """
    Price Impact Function: 
    I = sgn(delta) * integral_0^|delta| iota(lambda - z) dz
    """
    D = abs(delta)
    if D == 0.0:
        return 0.0
    
    L = float(lam)
    s = math.copysign(1.0, delta)
    
    # Case: lambda <= 0, iota is always iota0
    if L <= 0.0:
        return s * (iota0 * D)
    
    # Case: |delta| <= lambda, iota is always iota1 * ln(1+ell)/ell
    if D <= L:
        # Integral results in iota1 * [Li2(D-L) - Li2(-L)]
        # This is positive because Li2 is increasing on the negative axis
        return s * iota1 * (_Li2(D - L) - _Li2(-L))
    
    # Case: |delta| > lambda, iota transitions from iota1 to iota0 at z=lambda
    else:
        # Part 1: z in [0, L] -> iota1 * (-Li2(-L))
        # Part 2: z in [L, D] -> iota0 * (D - L)
        pos_part = iota1 * (-_Li2(-L))
        const_part = iota0 * (D - L)
        return s * (pos_part + const_part)

def Xi(delta: float, lam: float, iota0: float, iota1: float = 0.01) -> float:
    """
    Impact Cost Function:
    Xi = integral_0^|delta| (|delta| - z) * iota(lambda - z) dz
    """
    D = abs(delta)
    if D == 0.0:
        return 0.0
    
    L = float(lam)
    
    # Case: lambda <= 0, integral of (D-z)*iota0
    if L <= 0.0:
        return 0.5 * iota0 * D * D
    
    # Case: |delta| <= lambda
    if D <= L:
        arg = 1.0 + L - D
        # Derived term from integral: 
        # (D-L)*[Li2(D-L) - Li2(-L)] + (1+L)ln(1+L) - (1+L-D)ln(1+L-D) - D
        val = ((D - L) * (_Li2(D - L) - _Li2(-L))
               + (1.0 + L) * math.log(1.0 + L)
               - arg * math.log(arg)
               - D)
        return iota1 * val
    
    # Case: |delta| > lambda
    else:
        # Part 1 (z in [0, L]): (D-L)*integral_iota1 + integral_iota1_linear
        # Results in: (D-L)*(-Li2(-L)) + (1+L)ln(1+L) - L
        part1 = iota1 * ((D - L) * (-_Li2(-L)) + (1.0 + L) * math.log(1.0 + L) - L)
        # Part 2 (z in [L, D]): Triangle integral of iota0
        part2 = 0.5 * iota0 * (D - L)**2
        return part1 + part2

# ---- file helpers for experiment data ----
def _data_path_for_experiment(meta: Dict[str, Any]) -> str:
    """Return canonical path for experiment data file."""
    exp = str(meta.get('experiment', 'unnamed'))
    return f"./data_{exp}.pkl.gz"

# ---- small helper to produce stage file path ----
def _stage_path_for_experiment(meta_or_name: Any, stage: str) -> str:
    """Return canonical path for a stage-specific experiment file."""
    if isinstance(meta_or_name, str):
        exp = str(meta_or_name)
    else:
        exp = str(meta_or_name.get('experiment', 'unnamed'))
    base = f"./data_{exp}"
    return f"{base}_{stage}.pkl.gz"

# ---- save only selected keys (atomic move) ----
def save_stage_file(path: str, payload: Dict[str, Any]) -> None:
    """Write payload dict for a stage into gzip/pickle (atomic-ish)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    with gzip.open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)

# ---- load stage file (returns {} if missing or corrupt) ----
def load_stage_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with gzip.open(path, 'rb') as f:
            d = pickle.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}

# ---- merge stage files into one data dict (non-destructive) ----
def load_experiment_stages(meta_or_name: Any) -> Dict[str, Any]:
    """Load all available stage files and merge into single dict."""
    stages = ['meta', 'transitions', 'qvi', 'report']
    data = {}
    # stage 'meta' stored in the base path (legacy); try base first
    base_path = _data_path_for_experiment(meta_or_name) if not isinstance(meta_or_name, str) else f"./data_{meta_or_name}.pkl.gz"
    # attempt to load legacy full file if it exists - this keeps backward compatibility
    if os.path.exists(base_path):
        try:
            with gzip.open(base_path, 'rb') as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                data.update(payload)
        except Exception:
            pass

    # load stage-specific files and update (stage files override)
    for s in ['transitions', 'qvi', 'report']:
        p = _stage_path_for_experiment(meta_or_name, s)
        stage_payload = load_stage_file(p)
        if stage_payload:
            data.update(stage_payload)

    # finally try to load meta-only stage if present
    meta_path = _stage_path_for_experiment(meta_or_name, 'meta')
    meta_payload = load_stage_file(meta_path)
    if meta_payload:
        data.update(meta_payload)

    return data

# ----------------- HELPER FUNCTIONS (META-DRIVEN) -----------------

def idx_from_q_lam(q_val: float, lam_b_val: float, lam_s_val: float, meta: dict) -> int:
    """
    Map (q, lam_buy, lam_sell) -> flattened index with ordering: i = iq * (Nb * Ns) + ib * Ns + is_
    corresponding to the nearest grid node (clamped to grid bounds) in each dimension and
    returns the index as if the 3D array were flattened with q as the slowest-changing axis
    """
    qs = meta['qs']
    lams_b = meta['lams_b']
    lams_s = meta['lams_s']

    # clamp inputs to the grid bounds
    q_c = float(np.clip(q_val, qs[0], qs[-1]))
    lb_c = float(np.clip(lam_b_val, lams_b[0], lams_b[-1]))
    ls_c = float(np.clip(lam_s_val, lams_s[0], lams_s[-1]))

    # nearest grid indices
    iq = int(np.argmin(np.abs(qs - q_c)))
    ib = int(np.argmin(np.abs(lams_b - lb_c)))
    is_ = int(np.argmin(np.abs(lams_s - ls_c)))

    Nb = lams_b.size
    Ns = lams_s.size

    return iq * (Nb * Ns) + ib * Ns + is_


def trilinear_interpolation_weights(q_t: float, lb_t: float, ls_t: float,
                                    meta: dict,
                                    tol: float = 1e-12) -> Tuple[List[int], List[float]]:
    """
    Return indices and weights for trilinear interpolation onto (qs x lams_b x lams_s) grid.
    - Handles exact matches, linear/bilinear degeneracies and general interior case.
    - Returns list of at most 8 indices and corresponding weights summing to 1.

    Behaviour summary:
      1. Clamp the target point (q_t, lb_t, ls_t) to the grid bounding box.
      2. If the target exactly matches a grid node (within tol) in all 3 dims -> return that node with weight 1.
      3. If it matches exactly in 2 dims, reduce to linear interpolation in the remaining dim -> return 2 indices + weights.
      4. If it matches exactly in 1 dim, reduce to bilinear interpolation in the other two dims -> return up to 4 indices + weights.
      5. Otherwise, perform full trilinear interpolation on the cube defined by the lower indices in each dim -> return 8 corner indices + weights.
      6. Degenerate spacing (zero distance between consecutive nodes) is guarded against by returning the nearest single node.
    """
    qs = np.asarray(meta['qs'], dtype=float)
    lams_b = np.asarray(meta['lams_b'], dtype=float)
    lams_s = np.asarray(meta['lams_s'], dtype=float)

    Nq = qs.size
    Nb = lams_b.size
    Ns = lams_s.size

    # clamp target coordinates
    q_c = float(np.clip(q_t, qs[0], qs[-1]))
    lb_c = float(np.clip(lb_t, lams_b[0], lams_b[-1]))
    ls_c = float(np.clip(ls_t, lams_s[0], lams_s[-1]))

    # detect exact matches
    q_idx_close = np.where(np.isclose(qs, q_c, atol=tol, rtol=0.0))[0]
    lb_idx_close = np.where(np.isclose(lams_b, lb_c, atol=tol, rtol=0.0))[0]
    ls_idx_close = np.where(np.isclose(lams_s, ls_c, atol=tol, rtol=0.0))[0]

    # exact match in all dimensions
    if q_idx_close.size > 0 and lb_idx_close.size > 0 and ls_idx_close.size > 0:
        iq = int(q_idx_close[0])
        ib = int(lb_idx_close[0])
        is_ = int(ls_idx_close[0])
        return [iq * (Nb * Ns) + ib * Ns + is_], [1.0]

    def lower_index(arr, val):
        idx = int(np.searchsorted(arr, val) - 1)
        if idx < 0:
            idx = 0
        if idx > arr.size - 2:
            idx = arr.size - 2
        return idx

    # q & lb exact → linear in s
    if q_idx_close.size > 0 and lb_idx_close.size > 0:
        iq = int(q_idx_close[0])
        ib = int(lb_idx_close[0])

        if np.isclose(ls_c, lams_s[-1], atol=tol, rtol=0.0):
            is_ = Ns - 1
            return [iq * (Nb * Ns) + ib * Ns + is_], [1.0]

        i_s = lower_index(lams_s, ls_c)
        s1, s2 = lams_s[i_s], lams_s[i_s + 1]

        if s2 == s1:
            return [iq * (Nb * Ns) + ib * Ns + i_s], [1.0]

        w2 = (ls_c - s1) / (s2 - s1)
        w1 = 1.0 - w2

        idx1 = iq * (Nb * Ns) + ib * Ns + i_s
        idx2 = iq * (Nb * Ns) + ib * Ns + (i_s + 1)
        return [idx1, idx2], [w1, w2]

    # q & ls exact → linear in lb
    if q_idx_close.size > 0 and ls_idx_close.size > 0:
        iq = int(q_idx_close[0])
        is_ = int(ls_idx_close[0])

        if lb_c == lams_b[-1]:
            ib = Nb - 1
            return [iq * (Nb * Ns) + ib * Ns + is_], [1.0]

        i_b = lower_index(lams_b, lb_c)
        b1, b2 = lams_b[i_b], lams_b[i_b + 1]

        if b2 == b1:
            return [iq * (Nb * Ns) + i_b * Ns + is_], [1.0]

        w2 = (lb_c - b1) / (b2 - b1)
        w1 = 1.0 - w2

        idx1 = iq * (Nb * Ns) + i_b * Ns + is_
        idx2 = iq * (Nb * Ns) + (i_b + 1) * Ns + is_
        return [idx1, idx2], [w1, w2]

    # lb & ls exact → linear in q
    if lb_idx_close.size > 0 and ls_idx_close.size > 0:
        ib = int(lb_idx_close[0])
        is_ = int(ls_idx_close[0])

        if q_c == qs[-1]:
            iq = Nq - 1
            return [iq * (Nb * Ns) + ib * Ns + is_], [1.0]

        i_q = lower_index(qs, q_c)
        q1, q2 = qs[i_q], qs[i_q + 1]

        if q2 == q1:
            return [i_q * (Nb * Ns) + ib * Ns + is_], [1.0]

        w2 = (q_c - q1) / (q2 - q1)
        w1 = 1.0 - w2

        idx1 = i_q * (Nb * Ns) + ib * Ns + is_
        idx2 = (i_q + 1) * (Nb * Ns) + ib * Ns + is_
        return [idx1, idx2], [w1, w2]

    # full trilinear interpolation
    i_q = lower_index(qs, q_c)
    i_b = lower_index(lams_b, lb_c)
    i_s = lower_index(lams_s, ls_c)

    q1, q2 = qs[i_q], qs[i_q + 1]
    b1, b2 = lams_b[i_b], lams_b[i_b + 1]
    s1, s2 = lams_s[i_s], lams_s[i_s + 1]

    if (q2 == q1) or (b2 == b1) or (s2 == s1):
        iq = int(np.argmin(np.abs(qs - q_c)))
        ib = int(np.argmin(np.abs(lams_b - lb_c)))
        is_ = int(np.argmin(np.abs(lams_s - ls_c)))
        return [iq * (Nb * Ns) + ib * Ns + is_], [1.0]

    wq2 = (q_c - q1) / (q2 - q1)
    wq1 = 1.0 - wq2
    wb2 = (lb_c - b1) / (b2 - b1)
    wb1 = 1.0 - wb2
    ws2 = (ls_c - s1) / (s2 - s1)
    ws1 = 1.0 - ws2

    idx000 = i_q * (Nb * Ns) + i_b * Ns + i_s
    idx001 = i_q * (Nb * Ns) + i_b * Ns + (i_s + 1)
    idx010 = i_q * (Nb * Ns) + (i_b + 1) * Ns + i_s
    idx011 = i_q * (Nb * Ns) + (i_b + 1) * Ns + (i_s + 1)
    idx100 = (i_q + 1) * (Nb * Ns) + i_b * Ns + i_s
    idx101 = (i_q + 1) * (Nb * Ns) + i_b * Ns + (i_s + 1)
    idx110 = (i_q + 1) * (Nb * Ns) + (i_b + 1) * Ns + i_s
    idx111 = (i_q + 1) * (Nb * Ns) + (i_b + 1) * Ns + (i_s + 1)

    w000 = wq1 * wb1 * ws1
    w001 = wq1 * wb1 * ws2
    w010 = wq1 * wb2 * ws1
    w011 = wq1 * wb2 * ws2
    w100 = wq2 * wb1 * ws1
    w101 = wq2 * wb1 * ws2
    w110 = wq2 * wb2 * ws1
    w111 = wq2 * wb2 * ws2

    return (
        [idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111],
        [w000, w001, w010, w011, w100, w101, w110, w111]
    )

# ----------------- POST-TRADE -----------------
def post_trade(node_state: Tuple[float, float, float],
               action: Dict[str, Any],
               meta: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """
    Given (q, lam_b, lam_s) and an action, return:
      (q_new, lam_b_new, lam_s_new, baseW, delta_p, agent_q_change)

    Action types and relevant sizes:
      - {'type':'impulse', 'm': m}
      - {'type':'external_market', 'eta': eta, 'lambda_a_buy': lb_a, 'lambda_a_sell': ls_a}
      - {'type':'external_limit', 'rho': rho, 'side': 'buy'|'sell', ...}
    """
    
    iota0 = meta['iota0']
    alpha = meta['alpha']
    backend_mode = meta['backend_mode']
    zeta = meta['zeta']

    q = node_state[0]
    lam_b_pos = node_state[1]
    lam_s_pos = node_state[2]
    
    # small epsilon used to avoid division by zero or degenerate denominators
    eps = 1e-14

    # default outputs (if nothing executes)
    delta_p = 0.0            # price impact caused by the action
    agent_q_change = 0.0     # change in agent inventory (q_new - q)

    # ----------------- Impulse actions (agent-originated market orders) -----------------
    if action['type'] == 'impulse':
        m = float(action.get('m', 0.0))

        if backend_mode == 'OFF':
            if m > 0:
                exec_m = float(min(abs(m), lam_s_pos))
                exec_m = +exec_m
            elif m < 0:
                exec_m = -float(min(abs(m), lam_b_pos))
            else:
                exec_m = 0.0
        else:
            exec_m = m

        q_new = q + exec_m
        agent_q_change = q_new - q

        if exec_m > 0:
            lam_consumed = min(abs(exec_m), lam_s_pos)
            lam_s_new = lam_s_pos - lam_consumed
            lam_b_new = lam_b_pos
            delta_p = I(exec_m, lam_s_pos, iota0)
            impact_cost = Xi(exec_m, lam_s_pos, iota0)

        elif exec_m < 0:
            lam_consumed = min(abs(exec_m), lam_b_pos)
            lam_b_new = lam_b_pos - lam_consumed
            lam_s_new = lam_s_pos
            delta_p = I(exec_m, lam_b_pos, iota0)
            impact_cost = Xi(exec_m, lam_b_pos, iota0)

        else:
            lam_s_new = lam_s_pos; lam_b_new = lam_b_pos
            delta_p = 0.0; impact_cost = 0.0

        spread_cost = zeta * abs(exec_m)

        exponent = (q_new) * delta_p - (spread_cost + impact_cost)
        baseW = math.exp(-alpha * exponent)

        return q_new, lam_b_new, lam_s_new, baseW, delta_p, agent_q_change

    # ----------------- External market orders -----------------
    elif action['type'] == 'external_market':
        eta = float(action.get('eta', 0.0))
        lam_a_buy = float(action.get('lambda_a_buy', 0.0)); lam_a_sell = float(action.get('lambda_a_sell', 0.0))
        lam_a_buy_pos = max(lam_a_buy, 0.0); lam_a_sell_pos = max(lam_a_sell, 0.0)

        if eta == 0.0:
            return q, lam_b_pos, lam_s_pos, 1.0, 0.0, 0.0

        if eta > 0:
            denom = lam_s_pos + lam_a_sell_pos
            executed_total = float(min(abs(eta), denom)) if denom > eps else 0.0

            if denom <= eps or executed_total <= 0.0:
                lam_s_new = lam_s_pos; lam_b_new = lam_b_pos
                agent_fill = 0.0
            else:
                agent_fill = (lam_a_sell_pos / denom) * executed_total
                lam_reduction_e = min((lam_s_pos / denom) * executed_total, lam_s_pos)
                lam_reduction_a = min((lam_a_sell_pos / denom) * executed_total, lam_a_sell_pos)
                lam_s_new = lam_s_pos - lam_reduction_e
                lam_b_new = lam_b_pos

            q_new = q - agent_fill
            agent_q_change = q_new - q
            delta_p = I(executed_total, denom, iota0) if executed_total != 0.0 else 0.0

            baseW = math.exp(-alpha * (q_new * delta_p))

            return q_new, lam_b_new, lam_s_new, baseW, delta_p, agent_q_change

        else:
            denom = lam_b_pos + lam_a_buy_pos
            executed_total = float(min(abs(eta), denom)) if denom > eps else 0.0

            if denom <= eps or executed_total <= 0.0:
                lam_b_new = lam_b_pos; lam_s_new = lam_s_pos
                agent_fill = 0.0
            else:
                agent_fill = (lam_a_buy_pos / denom) * executed_total
                lam_reduction_e = min((lam_b_pos / denom) * executed_total, lam_b_pos)
                lam_reduction_a = min((lam_a_buy_pos / denom) * executed_total, lam_a_buy_pos)
                lam_b_new = lam_b_pos - lam_reduction_e
                lam_s_new = lam_s_pos

            q_new = q + agent_fill
            agent_q_change = q_new - q

            delta_p = I(-executed_total, denom, iota0) if executed_total != 0.0 else 0.0
            baseW = math.exp(-alpha * (q_new * delta_p))

            return q_new, lam_b_new, lam_s_new, baseW, delta_p, agent_q_change

    # ----------------- External limit order events -----------------
    elif action['type'] == 'external_limit':
        rho = float(action.get('rho', 0.0))
        side = action.get('side', 'sell')
        if side == 'sell':
            add = max(rho, 0.0)
            rem = min(max(-rho, 0.0), lam_s_pos)
            lam_s_new = lam_s_pos + add - rem
            lam_b_new = lam_b_pos
            q_new = q
        else:
            add = max(rho, 0.0)
            rem = min(max(-rho, 0.0), lam_b_pos)
            lam_b_new = lam_b_pos + add - rem
            lam_s_new = lam_s_pos
            q_new = q

        baseW = 1.0
        delta_p = 0.0
        agent_q_change = 0.0
        return q_new, lam_b_new, lam_s_new, baseW, delta_p, agent_q_change

    else:
        return q, lam_b_pos, lam_s_pos, 1.0, 0.0, 0.0


# =====================================================================
# MODULE-LEVEL WORKER: one inventory slice for precompute_transitions
# (must be at module level for joblib/loky pickling)
# =====================================================================
def _precompute_iq(
    iq: int,
    meta: Dict[str, Any],
    candidate_pairs: List[Tuple[float, float]],
    f_rate_buy: np.ndarray,   # (Nb, K)  limit-buy arrival rates
    f_rate_sell: np.ndarray,  # (Ns, K)  limit-sell arrival rates
    g_rate_buy: np.ndarray,   # (Nb, K)  market-buy arrival rates
    g_rate_sell: np.ndarray,  # (Ns, K)  market-sell arrival rates
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute impulse_transitions, transitions, and allowed_controls_map for
    all (ib, is_) nodes at fixed inventory level iq.
    Designed to be called in parallel by joblib.Parallel over iq values.

    Key optimisations over the serial original:
      - Limit-order post-trade states are computed *once* per (ib, is_, mark)
        and reused for all K controls (baseW=1 for limit orders; state is
        independent of the agent's quote size).
      - Admissibility is checked by a vectorised numpy pass over all K controls.
    """
    qs     = meta['qs']
    lams_b = meta['lams_b']
    lams_s = meta['lams_s']
    Nb, Ns = lams_b.size, lams_s.size
    K      = len(candidate_pairs)

    E_marks         = meta['E_marks']
    impulse_set     = meta['impulse_set']
    prune_threshold = float(meta['prune_threshold'])
    eps_fill        = float(meta['eps_fill'])
    iota0           = float(meta['iota0'])
    alpha           = float(meta['alpha'])
    zeta            = float(meta['zeta'])
    backend_mode    = str(meta.get('backend_mode', 'OFF'))
    kappa_hat       = float(meta.get('kappa_hat', 0.0))
    # generator_mode='exo_only' removes the agent's displayed quotes from the
    # exogenous arrival-rate kernel (both the f/g depth arguments and the VI
    # tilt). Fills, admissibility, and post_trade are deliberately unchanged.
    generator_mode  = str(meta.get('generator_mode', 'full'))
    if generator_mode not in ('full', 'exo_only'):
        raise ValueError(f"Unknown generator_mode={generator_mode!r}")
    agent_depth_on  = 0.0 if generator_mode == 'exo_only' else 1.0
    q_min           = float(qs[0])
    q_max           = float(qs[-1])
    q               = float(qs[iq])

    la_buy_arr  = np.array([max(float(b), 0.0) for b, _ in candidate_pairs])  # (K,)
    la_sell_arr = np.array([max(float(s), 0.0) for _, s in candidate_pairs])  # (K,)

    # Split E_marks by kind (used in vectorised admissibility check)
    sell_market_marks = [m for m in E_marks if m['kind'] == 'market' and float(m['eta']) > 0]
    buy_market_marks  = [m for m in E_marks if m['kind'] == 'market' and float(m['eta']) < 0]

    part_trans:   Dict = {}
    part_imp:     Dict = {}
    part_allowed: Dict = {}

    for ib in range(Nb):
        lam_b_pos = max(float(lams_b[ib]), 0.0)
        for is_ in range(Ns):
            lam_s_pos = max(float(lams_s[is_]), 0.0)
            node = iq * (Nb * Ns) + ib * Ns + is_

            # ----------------------------------------------------------
            # (A) IMPULSE TRANSITIONS
            # ----------------------------------------------------------
            imp_list: List[Tuple] = []
            for m_f in impulse_set:
                m_f = float(m_f)
                if backend_mode == 'OFF':
                    if m_f > 0:
                        exec_m = min(m_f, lam_s_pos)
                        # keep even if exec_m==0 (matches original: null impulse stored)
                        q_t = q + exec_m
                        if q_t < q_min - 1e-12 or q_t > q_max + 1e-12:
                            continue
                        lam_s_t = lam_s_pos - exec_m
                        lam_b_t = lam_b_pos
                        delta_p     = I(exec_m, lam_s_pos, iota0)
                        impact_cost = Xi(exec_m, lam_s_pos, iota0)
                    elif m_f < 0:
                        exec_m = -min(-m_f, lam_b_pos)
                        # keep even if exec_m==0 (matches original: null impulse stored)
                        q_t = q + exec_m
                        if q_t < q_min - 1e-12 or q_t > q_max + 1e-12:
                            continue
                        lam_b_t = lam_b_pos - (-exec_m)
                        lam_s_t = lam_s_pos
                        delta_p     = I(exec_m, lam_b_pos, iota0)
                        impact_cost = Xi(exec_m, lam_b_pos, iota0)
                    else:
                        continue
                    spread_cost = zeta * abs(exec_m)
                    exponent    = q_t * delta_p - (spread_cost + impact_cost)
                    baseW       = math.exp(-alpha * exponent)
                    agent_q_change = exec_m
                    idxs, wts = trilinear_interpolation_weights(q_t, lam_b_t, lam_s_t, meta)
                else:
                    # backend_mode != 'OFF' – delegate to post_trade
                    q_after = q + m_f
                    if q_after < q_min - 1e-12 or q_after > q_max + 1e-12:
                        continue
                    act = {'type': 'impulse', 'm': m_f}
                    q_t, lam_b_t, lam_s_t, baseW, _dp, agent_q_change = post_trade(
                        (q, lam_b_pos, lam_s_pos), act, meta)
                    idxs, wts = trilinear_interpolation_weights(
                        q_t, float(lam_b_t), float(lam_s_t), meta)
                imp_list.append((idxs, wts, baseW, float(agent_q_change)))
            part_imp[node] = imp_list

            # ----------------------------------------------------------
            # (B) ADMISSIBILITY – vectorised over all K controls at once
            # ----------------------------------------------------------
            admissible_mask = np.ones(K, dtype=bool)
            for mark in sell_market_marks:          # eta > 0
                eta     = abs(float(mark['eta']))
                denom_s = lam_s_pos + la_sell_arr   # (K,)
                valid   = denom_s > eps_fill
                exec_t  = np.where(valid, np.minimum(eta, denom_s), 0.0)
                fill    = np.where(valid, la_sell_arr / np.where(denom_s > 0, denom_s, 1.0) * exec_t, 0.0)
                q_p     = q - fill
                admissible_mask &= (q_p >= q_min - 1e-12) & (q_p <= q_max + 1e-12)
            for mark in buy_market_marks:           # eta < 0
                eta     = abs(float(mark['eta']))
                denom_b = lam_b_pos + la_buy_arr    # (K,)
                valid   = denom_b > eps_fill
                exec_t  = np.where(valid, np.minimum(eta, denom_b), 0.0)
                fill    = np.where(valid, la_buy_arr / np.where(denom_b > 0, denom_b, 1.0) * exec_t, 0.0)
                q_p     = q + fill
                admissible_mask &= (q_p >= q_min - 1e-12) & (q_p <= q_max + 1e-12)
            allowed_list = [int(k) for k in np.where(admissible_mask)[0]]
            part_allowed[node] = allowed_list

            # ----------------------------------------------------------
            # (C) LIMIT-ORDER POST-STATES (independent of k – precompute once)
            # ----------------------------------------------------------
            lim_buy_iw:  Dict[int, Tuple] = {}
            lim_sell_iw: Dict[int, Tuple] = {}
            for midx, mark in enumerate(E_marks):
                if mark['kind'] == 'limit_buy':
                    rho = float(mark['rho'])
                    add = max(rho,  0.0)
                    rem = min(max(-rho, 0.0), lam_b_pos)
                    lb_t = lam_b_pos + add - rem
                    lim_buy_iw[midx] = trilinear_interpolation_weights(q, lb_t, lam_s_pos, meta)
                elif mark['kind'] == 'limit_sell':
                    rho = float(mark['rho'])
                    add = max(rho,  0.0)
                    rem = min(max(-rho, 0.0), lam_s_pos)
                    ls_t = lam_s_pos + add - rem
                    lim_sell_iw[midx] = trilinear_interpolation_weights(q, lam_b_pos, ls_t, meta)

            # ----------------------------------------------------------
            # VI correction factors for extended arrival rates (kappa_hat)
            # lambda^VI = (lam_b_total) - (lam_s_total) per control k
            # ----------------------------------------------------------
            if kappa_hat != 0.0:
                lam_bid_total = lam_b_pos + agent_depth_on * la_buy_arr    # (K,)
                lam_ask_total = lam_s_pos + agent_depth_on * la_sell_arr   # (K,)
                lam_sum = lam_bid_total + lam_ask_total            # (K,)
                # lambda^VI = (lam_bid - lam_ask) / (lam_bid + lam_ask), in [-1, 1]
                lam_vi_arr = np.where(lam_sum > 0.0,
                                      (lam_bid_total - lam_ask_total) / lam_sum,
                                      0.0)                         # (K,)
                vi_corr_pos = np.exp( kappa_hat * lam_vi_arr)   # (K,)
                vi_corr_neg = np.exp(-kappa_hat * lam_vi_arr)   # (K,)

            # ----------------------------------------------------------
            # (D) CONTINUOUS TRANSITIONS for each admissible control
            # ----------------------------------------------------------
            for k_idx in allowed_list:
                lam_a_buy_pos  = float(la_buy_arr[k_idx])
                lam_a_sell_pos = float(la_sell_arr[k_idx])
                trans_list: List[Tuple] = []

                for midx, mark in enumerate(E_marks):
                    nu   = float(mark['nu'])
                    kind = mark['kind']

                    if kind == 'limit_buy':
                        kappa_rate = nu * float(f_rate_buy[ib, k_idx])
                        # kappa_hat extension: cancellations (rho<0) use f^- with VI correction
                        if kappa_hat != 0.0 and float(mark['rho']) < 0:
                            kappa_rate *= float(vi_corr_pos[k_idx])
                        if kappa_rate <= 0.0:
                            continue
                        idxs, wts = lim_buy_iw[midx]
                        if abs(kappa_rate) * sum(wts) < prune_threshold:
                            continue
                        trans_list.append((idxs, wts, 1.0, kappa_rate))

                    elif kind == 'limit_sell':
                        kappa_rate = nu * float(f_rate_sell[is_, k_idx])
                        # kappa_hat extension: cancellations (rho<0) use f^- with VI correction
                        if kappa_hat != 0.0 and float(mark['rho']) < 0:
                            kappa_rate *= float(vi_corr_neg[k_idx])
                        if kappa_rate <= 0.0:
                            continue
                        idxs, wts = lim_sell_iw[midx]
                        if abs(kappa_rate) * sum(wts) < prune_threshold:
                            continue
                        trans_list.append((idxs, wts, 1.0, kappa_rate))

                    else:  # market order
                        eta = float(mark['eta'])
                        if eta > 0:
                            kappa_rate = nu * float(g_rate_sell[is_, k_idx])
                            # kappa_hat extension: g(lam_ask, +lam_VI)
                            if kappa_hat != 0.0:
                                kappa_rate *= float(vi_corr_pos[k_idx])
                        else:
                            kappa_rate = nu * float(g_rate_buy[ib, k_idx])
                            # kappa_hat extension: g(lam_bid, -lam_VI)
                            if kappa_hat != 0.0:
                                kappa_rate *= float(vi_corr_neg[k_idx])
                        if kappa_rate <= 0.0:
                            continue
                        act = {'type': 'external_market', 'eta': eta,
                               'lambda_a_buy': lam_a_buy_pos, 'lambda_a_sell': lam_a_sell_pos}
                        q_t, lb_t, ls_t, baseW, _dp, _aq = post_trade(
                            (q, lam_b_pos, lam_s_pos), act, meta)
                        idxs, wts = trilinear_interpolation_weights(
                            q_t, float(lb_t), float(ls_t), meta)
                        if abs(kappa_rate * baseW) * sum(wts) < prune_threshold:
                            continue
                        trans_list.append((idxs, wts, baseW, kappa_rate))

                if trans_list:
                    part_trans[(node, k_idx)] = trans_list

    return part_trans, part_imp, part_allowed


# =====================================================================
# PRECOMPUTE TRANSITIONS  (vectorised + parallel)
# =====================================================================
def precompute_transitions(meta: Dict[str, Any]) -> Tuple[Dict, Dict, int, List[Tuple[float, float]], Dict[int, List[int]]]:
    """
    Precompute transitions on the 3D grid (q, lam_b, lam_s).

    Speedups vs. original:
      - f/g arrival-rate arrays computed once via numpy broadcasting (Nb×K and Ns×K).
      - Limit-order post-state interpolation weights precomputed once per
        (ib, is_, mark), shared across all K controls.
      - Admissibility check vectorised over all K controls with numpy.
      - Outer iq loop iterates sequentially (joblib removed to avoid
        memory pressure from forking large numpy arrays).
    """
    qs                 = meta['qs']
    lams_b             = meta['lams_b']
    lams_s             = meta['lams_s']
    lambda_a_buy_grid  = meta['lambda_a_buy_grid']
    lambda_a_sell_grid = meta['lambda_a_sell_grid']

    theta_f = float(meta['theta_f'])
    theta_g = float(meta['theta_g'])
    kappa   = float(meta['kappa'])
    lam_ref = float(meta['lam_ref'])

    Nq = qs.size; Nb = lams_b.size; Ns = lams_s.size
    N  = int(Nq * Nb * Ns)

    candidate_pairs = [(float(b), float(s))
                       for b in lambda_a_buy_grid
                       for s in lambda_a_sell_grid]

    # ------------------------------------------------------------------
    # Pre-compute arrival-rate tables by broadcasting over (grid, k_idx)
    # ------------------------------------------------------------------
    lb_pos = np.maximum(lams_b, 0.0)   # (Nb,)
    ls_pos = np.maximum(lams_s, 0.0)   # (Ns,)
    la_buy  = np.array([max(float(b), 0.0) for b, _ in candidate_pairs])  # (K,)
    la_sell = np.array([max(float(s), 0.0) for _, s in candidate_pairs])  # (K,)

    # generator_mode='exo_only': the agent's quotes do not enter the exogenous
    # arrival-rate kernel — rate tables collapse to external depth only
    # (constant across the K controls). Fill mechanics remain control-dependent.
    generator_mode = str(meta.get('generator_mode', 'full'))
    if generator_mode not in ('full', 'exo_only'):
        raise ValueError(f"Unknown generator_mode={generator_mode!r}")
    agent_depth_on = 0.0 if generator_mode == 'exo_only' else 1.0

    lam_b_total = lb_pos[:, None] + agent_depth_on * la_buy[None, :]    # (Nb, K)
    lam_s_total = ls_pos[:, None] + agent_depth_on * la_sell[None, :]   # (Ns, K)

    f_rate_buy  = theta_f * np.exp(-kappa * (lam_b_total - lam_ref))  # (Nb, K)
    f_rate_sell = theta_f * np.exp(-kappa * (lam_s_total - lam_ref))  # (Ns, K)
    g_rate_buy  = theta_g * np.exp( kappa * (lam_b_total - lam_ref))  # (Nb, K)
    g_rate_sell = theta_g * np.exp( kappa * (lam_s_total - lam_ref))  # (Ns, K)

    transitions:          Dict = {}
    impulse_transitions:  Dict = {}
    allowed_controls_map: Dict = {}

    for iq in tqdm.tqdm(range(Nq), desc='precompute_transitions'):
        part_trans, part_imp, part_allowed = _precompute_iq(
            iq, meta, candidate_pairs,
            f_rate_buy, f_rate_sell, g_rate_buy, g_rate_sell
        )
        transitions.update(part_trans)
        impulse_transitions.update(part_imp)
        allowed_controls_map.update(part_allowed)

    return transitions, impulse_transitions, N, candidate_pairs, allowed_controls_map

def solve_qvi(meta: Dict[str, Any], data: Dict[str, Any],  max_policy_iters: int, imp_tol: float) -> (np.ndarray, Dict[str, Any]):
    """
    Forward-in-time solver using the exact CTMC one-step operator (implicit in u_{n+1}),
    with Howard-style policy iteration per time slice and the refined impulse-value
    evaluation.

    Returns:
      u_ts: ndarray shape (n_steps+1, N) of u at times [0, dt, 2dt, ..., n_steps*dt]
      info: dict with policy_idx (final), policy_pairs, impulse_choice_m, policy_history,
            impulse_history, forced_impulse_history, u_history (same as u_ts).
    Returns:
      - u_ts, info  (exact same shapes/contents as your original function did)
    """
    
    transitions = data['transitions']
    impulse_transitions = data['impulse_transitions']
    N = int(data['N'])
    qs = np.asarray(meta['qs'], dtype=float)
    lams_b = np.asarray(meta['lams_b'], dtype=float)
    lams_s = np.asarray(meta['lams_s'], dtype=float)
    candidate_pairs = list(data['candidate_pairs'])
    impulse_set = np.asarray(meta['impulse_set'], dtype=float)
    T = float(meta['T'])
    dt = float(meta['dt'])
    allowed_controls_map = data['allowed_controls_map']

    # pull scalar parameters from meta (fall back to globals if you prefer, but meta is authoritative)
    alpha = float(meta.get('alpha', globals().get('alpha', 0.0)))
    zeta = float(meta.get('zeta', globals().get('zeta', 0.0)))
    iota0 = float(meta.get('iota0', globals().get('iota0', 0.0)))

    # ---------- Begin solver body (IDENTICAL LOGIC to your original function) ----------
    Nq = qs.size; Nb = lams_b.size; Ns = lams_s.size

    n_steps = int(np.ceil(T / dt))
    u_ts = np.zeros((n_steps + 1, N), dtype=float)

    # Build initial u0
    u0 = np.zeros(N, dtype=float)
    for iq in range(Nq):
        for ib in range(Nb):
            for is_ in range(Ns):
                i = iq * (Nb * Ns) + ib * Ns + is_
                qv = float(qs[iq])
                lb = float(lams_b[ib]); ls = float(lams_s[is_])
                if qv > 0:
                    terminal_l = lb
                else:
                    terminal_l = ls
                u0[i] = - np.exp(alpha * (zeta * abs(qv) + Xi(qv, terminal_l, iota0)))

    u_ts[0, :] = u0.copy()

    policy_idx = np.full(N, -1, dtype=int)
    for i in range(N):
        allowed = allowed_controls_map.get(i, [])
        policy_idx[i] = int(allowed[0]) if (allowed and len(allowed) > 0) else -1

    policy_history = np.full((n_steps, N), -1, dtype=int)
    impulse_choice_history = np.zeros((n_steps, N), dtype=float)
    forced_impulse_history = np.zeros((n_steps, N), dtype=bool)

    # ------------------------------------------------------------------
    # Pre-flatten transition data once (before the time loop)
    #   _flat_K[(n,k)]      = total jump rate K_i
    #   _exp_neg_Kdt[(n,k)] = exp(-K_i * dt)   <-- cached; dt is constant
    #   _flat_j[(n,k)]      = int array of destination indices
    #   _flat_kw[(n,k)]     = float array: baseW_m * kappa_m * wt per entry
    # ------------------------------------------------------------------
    _flat_K:      Dict[Tuple[int,int], float]      = {}
    _exp_neg_Kdt: Dict[Tuple[int,int], float]      = {}
    _flat_j:      Dict[Tuple[int,int], np.ndarray] = {}
    _flat_kw:     Dict[Tuple[int,int], np.ndarray] = {}

    for (nd, ki), tlist in transitions.items():
        K_tot = sum(float(kap) for _, _, _, kap in tlist)
        if K_tot <= 0.0:
            continue
        js: List[int]   = []
        cs: List[float] = []
        for (idxs, wts, bW, kap) in tlist:
            bWf = float(bW); kapf = float(kap)
            for j, wt in zip(idxs, wts):
                js.append(int(j)); cs.append(bWf * kapf * float(wt))
        _flat_K[nd, ki]      = K_tot
        _exp_neg_Kdt[nd, ki] = math.exp(-K_tot * dt)
        _flat_j[nd, ki]      = np.array(js,  dtype=np.intp)
        _flat_kw[nd, ki]     = np.array(cs,  dtype=float)

    # Pre-allocated COO buffers (upper bound on nnz per Howard iteration)
    _max_nnz = N + sum(len(v) for v in _flat_j.values())
    _row_buf  = np.empty(_max_nnz, dtype=np.intp)
    _col_buf  = np.empty(_max_nnz, dtype=np.intp)
    _dat_buf  = np.empty(_max_nnz, dtype=float)

    def build_SK_cache(u_vec: np.ndarray) -> Dict:
        """Fast SK cache: S_ik = kw_ik @ u_vec[j_ik]  (numpy dot, no Python loop)."""
        SK: Dict = {}
        for node, allowed_list in allowed_controls_map.items():
            if not allowed_list:
                continue
            for k_idx in allowed_list:
                key = (int(node), int(k_idx))
                j_arr = _flat_j.get(key)
                if j_arr is None:
                    SK[key] = (0.0, 0.0)
                else:
                    SK[key] = (_flat_K[key],
                               float(np.dot(_flat_kw[key], u_vec[j_arr])))
        return SK

    # LU factorisation cache – reused across time steps when policy is unchanged
    _lu_policy_key: Optional[bytes] = None
    _lu_solver = None   # callable returned by spla.factorized

    nonconverged_steps = 0

    for step in tqdm.tqdm(range(n_steps)):
        u_rhs = u_ts[step, :].copy()
        forced_mask = np.zeros(N, dtype=bool)
        forced_values = np.zeros(N, dtype=float)
        impulse_choice_m = np.zeros(N, dtype=float)

        converged = False
        u_next = None
        u_prev_iter = None
        last_howard_resid = float('inf')
        last_n_policy_flips = -1
        last_n_mask_flips = -1
        for it in range(max_policy_iters):
            # ----------------------------------------------------------
            # Assemble linear system using pre-allocated COO buffers
            # ----------------------------------------------------------
            b   = np.zeros(N, dtype=float)
            nnz = 0

            for i in range(N):
                if forced_mask[i]:
                    _row_buf[nnz] = i; _col_buf[nnz] = i; _dat_buf[nnz] = 1.0; nnz += 1
                    b[i] = forced_values[i]
                    continue

                k_idx = int(policy_idx[i])
                key   = (i, k_idx)
                j_arr = _flat_j.get(key)
                if j_arr is None:
                    _row_buf[nnz] = i; _col_buf[nnz] = i; _dat_buf[nnz] = 1.0; nnz += 1
                    b[i] = u_rhs[i]
                    continue

                K_tot = _flat_K[key]
                expm  = _exp_neg_Kdt[key]          # exp(-K*dt) pre-cached
                b[i]  = expm * u_rhs[i]

                _row_buf[nnz] = i; _col_buf[nnz] = i; _dat_buf[nnz] = 1.0; nnz += 1

                pref_over_K            = (1.0 - expm) / K_tot
                kw_arr                 = _flat_kw[key]
                n_e                    = len(j_arr)
                _row_buf[nnz:nnz+n_e]  = i
                _col_buf[nnz:nnz+n_e]  = j_arr
                _dat_buf[nnz:nnz+n_e]  = -pref_over_K * kw_arr
                nnz += n_e

            # Solve: reuse LU factorisation when policy+mask unchanged
            policy_key = policy_idx.tobytes() + forced_mask.tobytes()
            if policy_key == _lu_policy_key and _lu_solver is not None:
                u_next = _lu_solver(b)
            else:
                A = sp.csr_matrix(
                    (_dat_buf[:nnz], (_row_buf[:nnz], _col_buf[:nnz])),
                    shape=(N, N))
                try:
                    _lu_solver     = spla.factorized(A)
                    _lu_policy_key = policy_key
                    u_next         = _lu_solver(b)
                except Exception:
                    _lu_solver     = None
                    _lu_policy_key = None
                    u_next, info_iter = spla.gmres(A, b, tol=1e-10)
                    if info_iter != 0:
                        raise RuntimeError(
                            "Linear solve failed; gmres info=" + str(info_iter))

            # Howard residual between consecutive iterates (diagnostic only)
            if u_prev_iter is not None:
                last_howard_resid = float(np.max(np.abs(u_next - u_prev_iter)))
            u_prev_iter = u_next.copy()

            SK_cache = build_SK_cache(u_next)

            impulse_vals = np.full(N, -np.inf, dtype=float)
            impulse_m_choice = np.zeros(N, dtype=float)

            post_node_best_cache = {}

            for i in range(N):
                trans_imp_list = impulse_transitions.get(int(i), [])
                if not trans_imp_list:
                    continue
                best_val = -np.inf
                best_m = 0.0
                for (idxs, wts, baseW, m) in trans_imp_list:
                    cont_agg = 0.0
                    for j, wt in zip(idxs, wts):
                        j = int(j)
                        if j in post_node_best_cache:
                            best_cont_j = post_node_best_cache[j]
                        else:
                            allowed_post = allowed_controls_map.get(int(j), [])
                            if not allowed_post:
                                best_cont_j = float(u_next[j])
                            else:
                                best_cont_j = -np.inf
                                for k_after in allowed_post:
                                    K_jk, S_jk = SK_cache.get((int(j), int(k_after)), (0.0, 0.0))
                                    if K_jk <= 0.0:
                                        cont_jk = float(u_next[j])
                                    else:
                                        expm_j = _exp_neg_Kdt.get(
                                            (int(j), int(k_after)),
                                            math.exp(-K_jk * dt))
                                        weighted = float(S_jk) / float(K_jk)
                                        cont_jk = expm_j * float(u_next[j]) + (1.0 - expm_j) * weighted
                                    if cont_jk > best_cont_j:
                                        best_cont_j = cont_jk
                                if best_cont_j == -np.inf:
                                    best_cont_j = float(u_next[j])
                            post_node_best_cache[j] = best_cont_j
                        cont_agg += float(wt) * float(best_cont_j)
                    val = float(baseW) * float(cont_agg)
                    if val > best_val:
                        best_val = val
                        best_m = float(m)
                if best_val == -np.inf:
                    impulse_vals[i] = -np.inf
                    impulse_m_choice[i] = 0.0
                else:
                    impulse_vals[i] = float(best_val)
                    impulse_m_choice[i] = float(best_m)

            new_policy_idx = policy_idx.copy()
            for i in range(N):
                allowed_list = allowed_controls_map.get(int(i), [])
                if not allowed_list:
                    new_policy_idx[i] = -1
                    continue
                best_k = new_policy_idx[i]
                best_val = -np.inf
                for k_idx in allowed_list:
                    K_i, S_i = SK_cache.get((int(i), int(k_idx)), (0.0, 0.0))
                    if K_i <= 0.0:
                        candidate = float(u_next[i])
                    else:
                        expm_i = _exp_neg_Kdt.get(
                            (int(i), int(k_idx)), math.exp(-K_i * dt))
                        weighted  = S_i / K_i
                        candidate = expm_i * float(u_next[i]) + (1.0 - expm_i) * weighted
                    if candidate > best_val:
                        best_val = candidate
                        best_k = int(k_idx)
                new_policy_idx[i] = int(best_k)

            prefer_impulse_mask = np.zeros(N, dtype=bool)
            for i in range(N):
                k_try = int(new_policy_idx[i]) if new_policy_idx[i] != -1 else -1
                if k_try == -1:
                    cont_val = float(u_next[i])
                else:
                    K_i_try, S_i_try = SK_cache.get((int(i), int(k_try)), (0.0, 0.0))
                    if K_i_try <= 0.0:
                        cont_val = float(u_next[i])
                    else:
                        expm_try = _exp_neg_Kdt.get(
                            (int(i), int(k_try)), math.exp(-K_i_try * dt))
                        cont_val = expm_try * float(u_next[i]) + (1.0 - expm_try) * (S_i_try / K_i_try)
                if impulse_vals[i] > cont_val + imp_tol:
                    prefer_impulse_mask[i] = True

            policies_unchanged = np.array_equal(new_policy_idx, policy_idx)
            impulses_unchanged = np.array_equal(prefer_impulse_mask, forced_mask)
            u_change = np.linalg.norm(u_next - u_ts[step, :], ord=np.inf)
            last_n_policy_flips = int(np.sum(new_policy_idx != policy_idx))
            last_n_mask_flips = int(np.sum(prefer_impulse_mask != forced_mask))

            if policies_unchanged and impulses_unchanged:
                u_ts[step + 1, :] = u_next.copy()
                policy_idx = new_policy_idx.copy()
                forced_mask = prefer_impulse_mask.copy()
                forced_values = np.where(forced_mask, impulse_vals, 0.0)
                impulse_choice_m = impulse_m_choice.copy()
                converged = True
                break

            policy_idx = new_policy_idx.copy()
            forced_mask = prefer_impulse_mask.copy()
            forced_values = np.where(forced_mask, impulse_vals, 0.0)
            impulse_choice_m = impulse_m_choice.copy()

        if not converged:
            nonconverged_steps += 1
            tqdm.tqdm.write(
                f"[howard] step {step}: cap {max_policy_iters} hit — "
                f"policy flips {last_n_policy_flips}, "
                f"impulse-mask flips {last_n_mask_flips}, "
                f"max|du| last iter {last_howard_resid:.3e}")
            if u_next is not None:
                u_ts[step + 1, :] = u_next.copy()
            else:
                u_ts[step + 1, :] = u_rhs.copy()

        policy_history[step, :] = policy_idx.copy()
        impulse_choice_history[step, :] = impulse_choice_m.copy()
        forced_impulse_history[step, :] = forced_mask.copy()

    if nonconverged_steps:
        print(f"[howard] {nonconverged_steps}/{n_steps} steps hit the "
              f"{max_policy_iters}-iteration cap (last iterate accepted).",
              flush=True)

    policy_pairs = [ candidate_pairs[k] if (k != -1 and 0 <= k < len(candidate_pairs)) else (0.0, 0.0) for k in policy_idx.tolist() ]

    return u_ts, policy_idx, policy_pairs, impulse_choice_m, policy_history, impulse_choice_history, forced_impulse_history

def report(meta: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified report that precomputes:
      - decision_maps: per-slice arrays (n_slices x Nq x Nb x Ns) for best continuous
                       choices and impulse/continuation advantages (matching old precompute_decision_map keys)
      - single_reports: a list of plotting-style reports (matching build_plot_report keys) for every slice

    Time interpretation: BACKWARD from maturity. u_ts[0] is the terminal
    condition, so slice s corresponds to time REMAINING = s*dt. Consumers
    looking up a policy at forward simulation time t must index with
    remaining = T - t (see lookup_policy_state in cluster_diagnostics.py).
    """
    u_ts = data['u_ts']
    policy_idx = data['policy_idx']
    policy_pairs = data['policy_pairs']
    impulse_choice_m = data['impulse_choice_m']
    policy_history = data['policy_history']
    impulse_choice_history = data['impulse_history']
    forced_impulse_history = data['forced_impulse_history']
    u_history = u_ts
    
    transitions = data['transitions']
    impulse_transitions = data['impulse_transitions']
    N = data['N']
    candidate_pairs = data['candidate_pairs'] 
    allowed_controls_map = data['allowed_controls_map']
    
    T= meta['T']
    dt = meta['dt']
    iota0 = meta['iota0']
    qs = meta['qs']
    lams_b = meta['lams_b']
    lams_s = meta['lams_s']
    impulse_set = meta['impulse_set']
    Nq, Nb, Ns = qs.size, lams_b.size, lams_s.size

    u_arr = np.asarray(u_ts, dtype=float)
    if u_arr.ndim == 1:
        n_slices = 1
        u_history = u_arr.reshape((1, N))
    elif u_arr.ndim == 2:
        n_slices = int(u_arr.shape[0])
        if u_arr.shape[1] != N:
            raise ValueError(f"u_history layout mismatch: expected second dim {N} but got {u_arr.shape[1]}")
        u_history = u_arr
    else:
        raise ValueError("u_ts must be 1D or 2D array")

    # slice times (elapsed)
    slice_times = np.array([float(s) * float(dt) for s in range(n_slices)], dtype=float)

    # helper index flattening
    def node_index(iq:int, ib:int, is_int:int) -> int:
        return int(iq) * (Nb * Ns) + int(ib) * Ns + int(is_int)

    # policy alignment helper (policy_idx can be 1D or 2D history)
    ph = np.asarray(policy_idx)
    def _get_policy_for_slice(s: int):
        if ph.ndim == 1:
            return ph
        if ph.ndim == 2:
            ph_n = ph.shape[0]
            if ph_n == n_slices:
                return ph[int(s), :]
            if ph_n == max(0, n_slices - 1):
                if s == 0:
                    return ph[0, :]
                else:
                    return ph[int(s) - 1, :]
            idx = min(max(0, s), ph_n - 1)
            return ph[int(idx), :]
        raise ValueError("policy_idx has unsupported shape")

    # allocate decision maps (mirrors precompute_decision_map)
    shape = (n_slices, Nq, Nb, Ns)
    best_cont_k = np.full(shape, -1, dtype=int)
    best_cont_lam_a_buy = np.full(shape, np.nan, dtype=float)
    best_cont_lam_a_sell = np.full(shape, np.nan, dtype=float)
    best_imp_m = np.full(shape, np.nan, dtype=float)
    imp_adv = np.full(shape, np.nan, dtype=float)
    cont_adv = np.full(shape, np.nan, dtype=float)

    # list to hold single-slice plotting reports for each slice
    single_reports: List[Dict[str, Any]] = [None] * n_slices  # type: ignore

    # ------------------------------------------------------------------
    # Pre-build flat transition structures once (shared across all slices)
    # _r_flat_K[(n,k)]      = total jump rate K
    # _r_exp_neg_Kdt[(n,k)] = exp(-K * dt)
    # _r_flat_j[(n,k)]      = int index array
    # _r_flat_kw[(n,k)]     = baseW * kappa * wt float array
    # ------------------------------------------------------------------
    _r_flat_K:      Dict[Tuple[int,int], float]      = {}
    _r_exp_neg_Kdt: Dict[Tuple[int,int], float]      = {}
    _r_flat_j:      Dict[Tuple[int,int], np.ndarray] = {}
    _r_flat_kw:     Dict[Tuple[int,int], np.ndarray] = {}

    _dt_r = float(dt) if dt is not None else 0.0
    for (nd, ki), tlist in transitions.items():
        K_tot = sum(float(kap) for _, _, _, kap in tlist)
        if K_tot <= 0.0:
            continue
        js_r: List[int]   = []
        cs_r: List[float] = []
        for (idxs, wts, bW, kap) in tlist:
            bWf = float(bW); kapf = float(kap)
            for j, wt in zip(idxs, wts):
                js_r.append(int(j)); cs_r.append(bWf * kapf * float(wt))
        _r_flat_K[nd, ki]      = K_tot
        _r_exp_neg_Kdt[nd, ki] = math.exp(-K_tot * _dt_r)
        _r_flat_j[nd, ki]      = np.array(js_r, dtype=np.intp)
        _r_flat_kw[nd, ki]     = np.array(cs_r, dtype=float)

    # main per-slice computation
    for s in tqdm.tqdm(range(n_slices)):
        u_slice = u_history[int(s), :].astype(float)
        policy_slice = _get_policy_for_slice(s)

        # Build SK_cache: (node,k)->(K,S)  using numpy dot (no Python inner loop)
        SK_cache: Dict[Tuple[int,int], Tuple[float,float]] = {}
        for node, allowed in allowed_controls_map.items():
            if not allowed:
                continue
            for k_idx in allowed:
                key = (int(node), int(k_idx))
                j_arr = _r_flat_j.get(key)
                if j_arr is None:
                    SK_cache[key] = (0.0, 0.0)
                else:
                    SK_cache[key] = (
                        _r_flat_K[key],
                        float(np.dot(_r_flat_kw[key], u_slice[j_arr])))

        # cache best continuation per post-node j
        best_cont_cache: Dict[int, float] = {}
        def best_cont_for_postnode(j_idx: int) -> float:
            j = int(j_idx)
            if j in best_cont_cache:
                return best_cont_cache[j]
            allowed_post = allowed_controls_map.get(j, [])
            if not allowed_post:
                val = float(u_slice[j])
                best_cont_cache[j] = val
                return val
            best_val = -np.inf
            for k_after in allowed_post:
                K_jk, S_jk = SK_cache.get((int(j), int(k_after)), (0.0, 0.0))
                if K_jk <= 0.0:
                    cont_jk = float(u_slice[j])
                else:
                    expm_j = _r_exp_neg_Kdt.get(
                        (int(j), int(k_after)), math.exp(-K_jk * _dt_r))
                    weighted = float(S_jk) / float(K_jk)
                    cont_jk = expm_j * float(u_slice[j]) + (1.0 - expm_j) * weighted
                if cont_jk > best_val:
                    best_val = cont_jk
            if best_val == -np.inf:
                best_val = float(u_slice[j])
            best_cont_cache[j] = float(best_val)
            return best_val

        # local containers for this slice
        best_imp_val_local = np.full((Nq, Nb, Ns), -np.inf, dtype=float)
        best_imp_m_local = np.full((Nq, Nb, Ns), np.nan, dtype=float)
        best_cont_val_local = np.full((Nq, Nb, Ns), -np.inf, dtype=float)
        best_cont_k_local = np.full((Nq, Nb, Ns), -1, dtype=int)
        best_cont_lam_a_buy_local = np.full((Nq, Nb, Ns), np.nan, dtype=float)
        best_cont_lam_a_sell_local = np.full((Nq, Nb, Ns), np.nan, dtype=float)

        # compute per-node values
        for iq in range(Nq):
            for ib in range(Nb):
                for is_ in range(Ns):
                    node = node_index(iq, ib, is_)

                    # impulses (refined)
                    tlist = impulse_transitions.get(node, [])
                    if tlist:
                        best = -np.inf; bestm = np.nan
                        for (idxs, wts, baseW, m) in tlist:
                            cont_agg = 0.0
                            for j, wt in zip(idxs, wts):
                                cont_agg += float(wt) * float(best_cont_for_postnode(int(j)))
                            val = float(baseW) * float(cont_agg)
                            if val > best:
                                best = val; bestm = float(m)
                        best_imp_val_local[iq, ib, is_] = best
                        best_imp_m_local[iq, ib, is_] = bestm
                    else:
                        best_imp_val_local[iq, ib, is_] = -np.inf
                        best_imp_m_local[iq, ib, is_] = np.nan

                    # continuous: best over allowed controls
                    allowed = allowed_controls_map.get(node, [])
                    if not allowed:
                        best_cont_val_local[iq, ib, is_] = float(u_slice[node])
                        best_cont_k_local[iq, ib, is_] = -1
                        best_cont_lam_a_buy_local[iq, ib, is_] = np.nan
                        best_cont_lam_a_sell_local[iq, ib, is_] = np.nan
                    else:
                        bestv = -np.inf; bestk = -1
                        for k_idx in allowed:
                            K_i, S_i = SK_cache.get((int(node), int(k_idx)), (0.0, 0.0))
                            if K_i <= 0.0:
                                valk = float(u_slice[node])
                            else:
                                expm = _r_exp_neg_Kdt.get(
                                    (int(node), int(k_idx)), math.exp(-K_i * _dt_r))
                                valk = expm * float(u_slice[node]) + (1.0 - expm) * (float(S_i) / float(K_i))
                            if valk > bestv:
                                bestv = valk; bestk = int(k_idx)
                        best_cont_val_local[iq, ib, is_] = bestv
                        best_cont_k_local[iq, ib, is_] = bestk
                        if bestk != -1:
                            try:
                                lam_ab, lam_as = candidate_pairs[int(bestk)]
                                best_cont_lam_a_buy_local[iq, ib, is_] = float(lam_ab)
                                best_cont_lam_a_sell_local[iq, ib, is_] = float(lam_as)
                            except Exception:
                                best_cont_lam_a_buy_local[iq, ib, is_] = np.nan
                                best_cont_lam_a_sell_local[iq, ib, is_] = np.nan

        # fill decision maps arrays for slice s
        best_cont_k[s, :, :, :] = best_cont_k_local
        best_cont_lam_a_buy[s, :, :, :] = best_cont_lam_a_buy_local
        best_cont_lam_a_sell[s, :, :, :] = best_cont_lam_a_sell_local
        best_imp_m[s, :, :, :] = best_imp_m_local

        # compute adv arrays (imp_adv, cont_adv)
        # u_slice has layout [iq*(Nb*Ns) + ib*Ns + is_] = C-order reshape
        u_grid_local = u_slice.reshape(Nq, Nb, Ns)
        imp_adv_local = best_imp_val_local - u_grid_local
        cont_adv_local = best_cont_val_local - u_grid_local
        imp_adv[s, :, :, :] = imp_adv_local
        cont_adv[s, :, :, :] = cont_adv_local

        # build the single-slice plotting-style report (matching build_plot_report keys)
        best_cont_k_grid = best_cont_k_local.copy()
        best_cont_lam_a_buy_grid = best_cont_lam_a_buy_local.copy()
        best_cont_lam_a_sell_grid = best_cont_lam_a_sell_local.copy()
        best_imp_m_grid = best_imp_m_local.copy()
        imp_adv_grid = imp_adv_local.copy()
        cont_adv_grid = cont_adv_local.copy()

        best_cont_val_grid = cont_adv_grid + u_grid_local
        best_imp_val_grid = imp_adv_grid + u_grid_local
        best_of = np.maximum(best_cont_val_grid, best_imp_val_grid)

        res_grid = best_of - u_grid_local
        imp_minus_cont_grid = best_imp_val_grid - best_cont_val_grid

        lam_s_zero_idx = int(np.argmin(np.abs(lams_s - 0.0)))
        lam_b_zero_idx = int(np.argmin(np.abs(lams_b - 0.0)))
        q_zero_idx = int(np.argmin(np.abs(qs - 0.0)))
        chosen_time = float(slice_times[s]) if slice_times is not None else None

        single_reports[s] = {
            'qs': qs, 'lams_b': lams_b, 'lams_s': lams_s,
            'Nq': Nq, 'Nb': Nb, 'Ns': Ns,
            'res_grid': res_grid,
            'imp_adv_grid': imp_adv_grid,
            'cont_adv_grid': cont_adv_grid,
            'imp_minus_cont_grid': imp_minus_cont_grid,
            'best_imp_m_grid': best_imp_m_grid,
            'best_cont_lam_a_buy_grid': best_cont_lam_a_buy_grid,
            'best_cont_lam_a_sell_grid': best_cont_lam_a_sell_grid,
            'best_cont_k_grid': best_cont_k_grid,
            'allowed_controls_map': allowed_controls_map,
            'candidate_pairs': candidate_pairs,
            'impulse_set': np.asarray(impulse_set) if impulse_set is not None else np.array([]),
            'lam_s_zero_idx': lam_s_zero_idx,
            'lam_b_zero_idx': lam_b_zero_idx,
            'q_zero_idx': q_zero_idx,
            'chosen_time': chosen_time,
            'chosen_slice_idx': int(s)
        }

    # package decision_maps (older format)
    decision_maps = {
        'slice_times': slice_times,
        'n_slices': n_slices,
        'qs': qs, 'lams_b': lams_b, 'lams_s': lams_s,
        'best_cont_k': best_cont_k,
        'best_cont_lam_a_buy': best_cont_lam_a_buy,
        'best_cont_lam_a_sell': best_cont_lam_a_sell,
        'best_imp_m': best_imp_m,
        'imp_adv': imp_adv,
        'cont_adv': cont_adv
    }

    out = {
        'decision_maps': decision_maps,
        'single_reports': single_reports,
        'slice_times': slice_times,
        'n_slices': n_slices,
        'qs': qs, 'lams_b': lams_b, 'lams_s': lams_s,
        'candidate_pairs': candidate_pairs,
        'allowed_controls_map': allowed_controls_map,
        'impulse_set': np.asarray(impulse_set) if impulse_set is not None else np.array([])
    }
    return decision_maps, single_reports, slice_times, n_slices

def run_qvi_pipeline(
    experiment : Dict[str, Any], 
    force_recompute: bool = False,
    verbose: bool = True,
    max_policy_iters: int = 50,
    imp_tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Handles the end-to-end QVI flow with a detailed meta-parameter summary.

    NOTE: This version uses stage-specific files via load_experiment_stages()
    and save_stage_file()/_stage_path_for_experiment(...) so large arrays are
    persisted per-stage (avoids huge single-file serialization spikes).
    """

    # 1. Resolve Meta and Path (always load merged stage files)
    if isinstance(experiment, str):
        exp_name = experiment
        data = load_experiment_stages(experiment)   # experiment may be name or meta dict
        data_path = f"./data_{exp_name}.pkl.gz"     # legacy base path (for messages)
        if not data or 'meta' not in data:
            raise ValueError(f"Experiment '{exp_name}' not found or missing 'meta' in {data_path}")
        meta = data['meta']
    else:
        meta = experiment
        exp_name = str(meta.get('experiment', 'unnamed'))
        # load merged stage files for this meta dict as well
        data = load_experiment_stages(meta)
        data_path = _data_path_for_experiment(meta)

    # 2. Verbose Meta Printing
    if verbose:
        print(f"\n{'='*25} Experiment: {exp_name} {'='*25}")
        params = [
            'experiment', 'T', 'dt', 'alpha', 'p0', 'iota0', 'zeta', 'backend_mode',
            'q_min', 'q_max', 'lb_min', 'lb_max', 'ls_min', 'ls_max', 'qs', 
            'lams_b', 'lams_s', 'lambda_a_buy_grid', 'lambda_a_sell_grid', 
            'impulse_set', 'theta_f', 'theta_g', 'lam_ref', 'kappa', 'kappa_hat',
            'P1_vals', 'P2_vals', 'P3_vals'
        ]
        for p in params:
            val = meta.get(p, 'N/A')
            
            # Custom formatting for the P-value dictionaries
            if isinstance(val, dict):
                items = [f"{k}:{v}" for k, v in val.items()]
                print(f"{p: <20}: {{ {', '.join(items)} }}")
            
            # Summarize large arrays/lists (grids)
            elif isinstance(val, (np.ndarray, list)) and len(val) > 10:
                v_min, v_max = (min(val), max(val)) if len(val) > 0 else (0, 0)
                print(f"{p: <20}: Array/List (len: {len(val)}, range: [{v_min}, {v_max}])")
            
            # Standard scalars
            else:
                print(f"{p: <20}: {val}")
        print(f"{'='*65}\n")

    # 3. Sync Meta in the data container (save only meta stage if needed)
    if not data or force_recompute:
        data = {'meta': meta}
    elif 'meta' not in data:
        data['meta'] = meta
        # save only the small meta stage file (avoid writing the huge merged file)
        save_stage_file(_stage_path_for_experiment(meta, 'meta'), {'meta': meta})

    recomputed_previous = force_recompute

    # ---- STAGE 1: Transitions ----
    transition_keys = ('transitions', 'impulse_transitions', 'N', 'candidate_pairs', 'allowed_controls_map')
    needs_trans = recomputed_previous or not all(k in data for k in transition_keys)
    
    if needs_trans:
        print(f"[{exp_name}] Computing transitions...")
        results = precompute_transitions(meta)
        stage_payload = {'meta': meta}
        stage_payload.update(dict(zip(transition_keys, results)))
        # save stage-specific file (atomic)
        save_stage_file(_stage_path_for_experiment(meta, 'transitions'), stage_payload)
        # update in-memory merged data for downstream stages
        data.update(stage_payload)
        recomputed_previous = True 
    else:
        print(f"[{exp_name}] Transitions present.")

    # ---- STAGE 2: QVI Solver ----
    qvi_keys = ('u_ts', 'policy_idx', 'policy_pairs', 'impulse_choice_m', 
                'policy_history', 'impulse_history', 'forced_impulse_history')
    needs_qvi = recomputed_previous or not all(k in data for k in qvi_keys)

    if needs_qvi:
        print(f"[{exp_name}] Solving QVI...")
        results = solve_qvi(meta, data, max_policy_iters=max_policy_iters, imp_tol=imp_tol)
        stage_payload = dict(zip(qvi_keys, results))
        # include meta for consistency / easy inspection
        stage_payload['meta'] = meta
        save_stage_file(_stage_path_for_experiment(meta, 'qvi'), stage_payload)
        data.update(stage_payload)
        recomputed_previous = True
    else:
        print(f"[{exp_name}] QVI solution present.")

    # ---- STAGE 3: Unified Report ----
    report_keys = ('decision_maps', 'single_reports', 'slice_times', 'n_slices')
    needs_report = recomputed_previous or not all(k in data for k in report_keys)

    if needs_report:
        print(f"[{exp_name}] Generating reports...")
        results = report(meta, data)
        stage_payload = dict(zip(report_keys, results))
        stage_payload['meta'] = meta
        save_stage_file(_stage_path_for_experiment(meta, 'report'), stage_payload)
        data.update(stage_payload)
    else:
        print(f"[{exp_name}] Report present.")

    print(f"[{exp_name}] Pipeline complete.\n")
    return data