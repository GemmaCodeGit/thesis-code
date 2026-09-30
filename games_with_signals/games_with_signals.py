"""Two-type mean field equilibrium with jump signals.

Numerical companion to the thesis chapter *Jump Signals in Optimal Investment*
(P. Bank, G. Sedrakjan, "How much should we care about what others know? Jump
signals in optimal investment under relative performance concerns"). The
script implements the case study of the section *Case Studies and
Financial-Economic Discussion*:

* two investor types A and B trade one stock with log-normal price shocks;
* before a shock, each type receives a categorical signal about a noisy
  version of the shock (with probability ``ps`` and quality ``rho``);
* the representative investor's best response to a signal-driven mean field
  strategy is known in closed form up to a one-dimensional optimisation
  (thesis eq. (mf_optimal_zero) and (mf_optimal_nonzero));
* a mean field equilibrium is found by iterating the best-response map;
* alternative type environments are compared with a symmetric reference
  environment through the certainty equivalent
  ``x_0^A = exp(M^{A,alt} - M^{A,ref})`` with ``M`` as in thesis eq. (M_mf).

Structure of the file
---------------------
1. Market and investor parameters.
2. Auxiliary functions (jump size, normal cdf/pdf, file helpers).
3. Mean field equilibrium: the mean jump multiplier ``mean_jumps``, the two
   best-response objectives, ``compute_optimal_response`` and the fixed-point
   iteration ``mean_field_equilibrium``; the value constant ``M``.
   The saved ``.npz`` files hold the swept parameter values, the type
   environments, the equilibrium profiles and the certainty equivalents.
4. Experiment drivers: ``alternative`` (one-dimensional parameter sweep) and
   ``alternative_2d`` (two-dimensional grid); both run in a process pool and
   cache their result as ``.npz``.
5. The scenario grid solved for the thesis (``__main__``).

Running the file solves all six scenarios (theta in {0.5, 1}, alpha in
{0.5, 2, 4}) into ``scenario_outputs/`` using all available cores. This is
slow (hours per scenario at the resolutions used in the thesis).

Notation. A *type environment* ``t`` is a structured array with one row per
type (row 0 = type A, row 1 = type B) and fields ``p`` (population share),
``ps`` (signal frequency), ``rho`` (signal quality), ``alpha`` (relative risk
aversion) and ``theta`` (relative performance concern). A *strategy profile*
``pi`` is a ``(2, 7)`` array: ``pi[i, z]`` is the fraction of wealth type
``i`` invests after signal ``z``; index ``6`` (``zero_idx``) is the no-signal
position, indices ``0..5`` correspond to ``jump_signals``.
"""

import numpy as np
import copy
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

import warnings
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=IntegrationWarning)
from functools import lru_cache

############################################################
## PARAMETERS OF THE MARKET AND THE INVESTORS             ##
############################################################

# Common market parameters (identical for both types): interest rate r,
# drift kappa and volatility sigma of the diffusive part, and the log-normal
# jump size eta(e) = exp(sigma_hat * e + kappa_hat - sigma_hat^2 / 2) - 1
# driven by a standard normal mark e. sigma_hat is the jump volatility,
# kappa_hat the jump drift (0 = jumps with zero log-mean).
r = 0

kappa = 0.08
kappa_hat = 0.0
sigma = 0.3
sigma_hat = 0.1

# Jump intensity lambda of the common price shocks and the half-width acc of
# the "small" and "medium" signal categories, see integral_bounds().
lam = 10
acc = 0.5

# Signal alphabet. The thesis uses {-inf, -1, -0.5, 0.5, 1, +inf}; the code
# labels the two unbounded categories by +-1.5. A signal z with |z| < 1.5
# says that the perturbed mark lies in an interval of width acc, the signals
# +-1.5 say that it lies beyond +-1 (the "large" shocks).
jump_signals = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
min_signal = jump_signals[0]
max_signal = jump_signals[-1]

# Position of the no-signal strategy inside a strategy vector pi[i, :] and
# total number of positions per type (6 signals + no signal).
zero_idx = 6
total_strategies = 7

# Reference type environment of the thesis: p^A = p^B = 0.5, signal frequency
# ps = 0.5, signal quality rho = 0.5. theta and alpha are set per scenario.
default_base_params = {'p': 0.5, 'ps': 0.5, 'rho': 0.5}

# Masks over jump_signals: interior positive / interior negative / top / bottom
# category. They select the four cases of integral_bounds() in vectorised form.
MASK_POS = (jump_signals > min_signal) & (jump_signals < max_signal) & (jump_signals > 0)
MASK_NEG = (jump_signals > min_signal) & (jump_signals < max_signal) & (jump_signals < 0)
MASK_MAX = (jump_signals >= max_signal)
MASK_MIN = (jump_signals <= min_signal)

#########################
## AUXILIARY FUNCTIONS ##
#########################

exp_term = np.exp(kappa_hat - (1 / 2) * (sigma_hat ** 2))

def eta(e):
    """Relative jump size eta(e) of the stock for the common mark e ~ N(0, 1)."""
    return np.exp(sigma_hat * e) * exp_term - 1

def jump(e1, x, exponent):
    """(1 + x * eta(e1))^exponent: post-jump wealth factor of a fraction x
    invested in the stock, raised to the power used by the CRRA utility."""
    return (1 + x * eta(e1)) ** exponent

def fast_norm_cdf(x):
    """Logistic approximation of the standard normal cdf (max. error ~1.4e-2
    on the tails, much smaller near 0). Used inside the numerical integrals
    where scipy's norm.cdf would dominate the run time."""
    return 0.5 * (1 + np.tanh(np.sqrt(np.pi / 8) * x))

def fast_norm_pdf(x):
    """Standard normal density."""
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def to_structured(t_list):
    """Convert a list of two parameter dicts into the structured type array
    (row 0 = type A, row 1 = type B) used throughout."""
    return np.array(
        [(entry['p'], entry['ps'], entry['rho'], entry['alpha'], entry['theta']) for entry in t_list],
        dtype=[('p', 'f8'), ('ps', 'f8'), ('rho', 'f8'), ('alpha', 'f8'), ('theta', 'f8')]
    )

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def ensure_parent_directory(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_directory(parent_dir)

def scenario_path(output_dir, filename):
    return os.path.join(output_dir, filename)

def format_scenario_value(value):
    """0.5 -> '0p5', 2.0 -> '2' (folder-name safe)."""
    numeric_value = float(value)
    if np.isclose(numeric_value, round(numeric_value)):
        return str(int(round(numeric_value)))
    return str(numeric_value).replace('.', 'p')

def scenario_folder_name(theta, alpha):
    """Output folder of a scenario, e.g. theta_0p5_alpha_2."""
    return f"theta_{format_scenario_value(theta)}_alpha_{format_scenario_value(alpha)}"

############################
## MEAN FIELD EQUILIBRIUM ##
############################

def integral_bounds(sig, e1, rho_1):
    """Interval I(z, e^c) of the idiosyncratic mark, thesis Section 'A Single
    Stock Model ...'.

    Given the common mark e1 and a signal sig of a type with quality rho_1,
    the perturbed mark z = rho_1 * e1 + sqrt(1 - rho_1^2) * e_i lies in the
    category of sig iff the idiosyncratic mark e_i lies in the returned
    interval. The unbounded categories (+-1.5) are truncated at +-5 standard
    deviations.
    """
    sqrt_term_1 = 1 / np.sqrt(1 - (rho_1 ** 2))

    if min_signal < sig < max_signal:
        if sig > 0:
            lower_integral_bound = (sig - acc - rho_1 * e1) * sqrt_term_1
            upper_integral_bound = (sig - rho_1 * e1) * sqrt_term_1
        else:
            lower_integral_bound = (sig - rho_1 * e1) * sqrt_term_1
            upper_integral_bound = (sig + acc - rho_1 * e1) * sqrt_term_1
    elif sig >= max_signal:
        lower_integral_bound = (max_signal - acc - rho_1 * e1) * sqrt_term_1
        upper_integral_bound = 5
    else:
        lower_integral_bound = -5
        upper_integral_bound = (min_signal + acc - rho_1 * e1) * sqrt_term_1

    return lower_integral_bound, upper_integral_bound

def idiosyncratic_mean(e, rho, pi, type_number):
    """Signal-averaged log wealth factor of one type, given the common mark e.

    Computes sum_z log(1 + pi[type](z) * eta(e)) * N_{0,1}(I(z, e)) over the
    six non-zero signals z, i.e. the integral over the idiosyncratic mark in
    thesis eq. (compute_mean). Vectorised version of integral_bounds().
    """
    sqrt_term = 1.0 / np.sqrt(1.0 - rho ** 2)
    base = (jump_signals - rho * e) * sqrt_term
    acc_term = acc * sqrt_term

    lower_bounds = np.empty_like(jump_signals, dtype=float)
    upper_bounds = np.empty_like(jump_signals, dtype=float)

    lower_bounds[MASK_POS] = base[MASK_POS] - acc_term
    upper_bounds[MASK_POS] = base[MASK_POS]

    lower_bounds[MASK_NEG] = base[MASK_NEG]
    upper_bounds[MASK_NEG] = base[MASK_NEG] + acc_term

    lower_bounds[MASK_MAX] = (max_signal - acc - rho * e) * sqrt_term
    upper_bounds[MASK_MAX] = 5.0

    lower_bounds[MASK_MIN] = -5.0
    upper_bounds[MASK_MIN] = (min_signal + acc - rho * e) * sqrt_term

    norm_cdf_diff = fast_norm_cdf(upper_bounds) - fast_norm_cdf(lower_bounds)

    pi_eta_e = pi[type_number][:6] * eta(e)
    log_terms = np.log1p(pi_eta_e)

    return np.dot(log_terms, norm_cdf_diff)

def mean_jumps(e, t, pi):
    """Mean jump multiplier of the population, thesis eq. (compute_mean):

        exp{ sum_t p^t [ (1 - ps^t) log(1 + pi^t(0) eta(e))
                         + ps^t * idiosyncratic_mean(e, rho^t, pi, t) ] }.

    This is the geometric mean of the post-jump wealth factors across the
    population when the common mark is e.
    """
    eta_e = eta(e)
    pi_0_eta_e = pi[0][zero_idx] * eta_e
    pi_1_eta_e = pi[1][zero_idx] * eta_e

    type_0 = (1 - t[0]['ps']) * np.log1p(pi_0_eta_e) + t[0]['ps'] * idiosyncratic_mean(e, t[0]['rho'], pi, 0)
    type_1 = (1 - t[1]['ps']) * np.log1p(pi_1_eta_e) + t[1]['ps'] * idiosyncratic_mean(e, t[1]['rho'], pi, 1)

    return np.exp(t[0]['p'] * type_0 + t[1]['p'] * type_1)

# --- Best-response objectives (negated, for scipy minimisation) ---

def zero_signal_objective(x, type_number, pi, t, cached_m_jumps):
    """Negative of the no-signal objective in thesis eq. (mf_optimal_zero)
    for a fraction x invested by type ``type_number``:

        x (kappa - r) - alpha sigma^2 x^2 / 2 - theta (1 - alpha) sigma^2 x * avg_pi_0
        + lambda (1 - ps) / (1 - alpha) * ( E[(1 + x eta)^{1-alpha} * mean_jumps^{-theta(1-alpha)}] - 1 ).

    The expectation over the common mark e ~ N(0, 1) is a quadrature on
    [-5, 5]. ``cached_m_jumps`` evaluates mean_jumps(e, t, pi) (memoised, as
    it is the same for every candidate x).
    """
    alpha = t[type_number]['alpha']
    theta = t[type_number]['theta']
    exponent = -theta * (1 - alpha)
    sigma_sq = sigma**2

    avg_pi_0 = t[0]['p'] * pi[0][zero_idx] + t[1]['p'] * pi[1][zero_idx]
    A = (kappa - r) * x - (0.5 * alpha * sigma_sq * (x**2)) + exponent * sigma_sq * x * avg_pi_0

    def integrand_zero_signal(e1):
        m_jumps = cached_m_jumps(e1) ** exponent
        return fast_norm_pdf(e1) * jump(e1, x, 1 - alpha) * m_jumps

    integral = quad(integrand_zero_signal, -5, 5, epsabs=1e-5, epsrel=1e-5, limit=200)[0]
    return -(A + (lam * (1 - t[type_number]['ps']) / (1 - alpha)) * (integral - 1))

def non_zero_signal_objective(x, sig, type_number, pi, t, cached_m_jumps):
    """Negative of the signal objective in thesis eq. (mf_optimal_nonzero)
    for a fraction x invested by type ``type_number`` after signal ``sig``:

        1 / (1 - alpha) * E[ ((1 + x eta)^{1-alpha} mean_jumps^{-theta(1-alpha)} - 1)
                             * N_{0,1}(I(sig, e)) ] / N_{0,1}(I(sig)).

    The conditional law of the common mark given the signal is the standard
    normal law reweighted by N_{0,1}(I(sig, e)) (see integral_bounds) and
    normalised by normalization(sig).
    """
    alpha = t[type_number]['alpha']
    theta = t[type_number]['theta']
    exponent = -theta * (1 - alpha)

    def integrand_non_zero_signal(e1):
        lower_integral_bound, upper_integral_bound = integral_bounds(sig, e1, t[type_number]['rho'])
        upper = fast_norm_cdf(upper_integral_bound)
        lower = fast_norm_cdf(lower_integral_bound)

        m_jumps = cached_m_jumps(e1) ** exponent
        i_jumps = jump(e1, x, 1 - alpha)
        density = fast_norm_pdf(e1)

        return density * (i_jumps * m_jumps - 1) * (upper - lower)

    integral = quad(integrand_non_zero_signal, -5, 5, epsabs=1e-5, epsrel=1e-5, limit=200)[0]
    normal = 1 / normalization(sig)
    return -((1 / (1 - alpha)) * normal * integral)

def normalization(z):
    """Unconditional probability N_{0,1}(I(z)) that the perturbed mark falls
    into the category of signal z (the signal frequency is lam * ps * this)."""
    if min_signal < z < max_signal:
        if z > 0:
            return norm.cdf(z) - norm.cdf(z - acc)
        else:
            return norm.cdf(z + acc) - norm.cdf(z)
    elif z >= max_signal:
        return 1 - norm.cdf(max_signal - acc)
    else:
        return norm.cdf(min_signal + acc)

def compute_optimal_response(z, i, pi, t):
    """Best response of type i to the profile pi for signal index z
    (z == zero_idx: no signal). Bounded scalar optimisation on [0, 1]."""
    @lru_cache(maxsize=1024)
    def cached_m_jumps(e1):
        return mean_jumps(e1, t, pi)

    if z == zero_idx:
        result = minimize_scalar(zero_signal_objective, 0.5, bounds=(0,1), method='bounded', args=(i, pi, t, cached_m_jumps), options={'xatol': 1e-8}).x
    else:
        sig = jump_signals[z]
        result = minimize_scalar(non_zero_signal_objective, 0.5, bounds=(0,1), method='bounded', args=(sig, i, pi, t, cached_m_jumps), options={'xatol': 1e-8}).x

    return result

def mean_field_equilibrium(t):
    """Mean field equilibrium for the type environment t by best-response
    iteration (Gauss-Seidel: each updated position is used immediately).
    Starts from pi = 0.5 everywhere and stops when the maximal change of a
    position drops below 1e-8 or after max_inter + 1 sweeps."""
    pi = np.full((2, total_strategies), 0.5)
    pi_new = np.zeros((2, total_strategies))

    epsilon = 1e-8
    j = 0
    max_inter = 10
    current_error = np.inf

    while current_error >= epsilon and j <= max_inter:
        pi_new = pi.copy()

        for i in range(2):
            for z in range(total_strategies):
                pi[i, z] = compute_optimal_response(z, i, pi, t)

        current_error = np.max(np.abs(pi - pi_new))
        j += 1

    return pi

def M(t, i, pi):
    """Value constant M^t_pi of type i, thesis eq. (M_mf), evaluated at the
    profile pi (the value function is v^t(x, xbar, T) = u_t(x, xbar) exp(M T)
    up to the T-scaling handled there).

    Pieces: -theta * avg_tau_pi (mean growth rate of the population, tau^t =
    r + pi^t(0)(kappa - r) - sigma^2 pi^t(0)^2 / 2), the diffusion term
    theta^2 (1 - alpha) sigma^2 avg_pi_0^2 / 2, the no-signal objective at
    pi[i](0), and the signal objectives at pi[i](z) weighted by the signal
    frequencies lam * ps * N_{0,1}(I(z)).
    """
    alpha = t[i]['alpha']
    theta = t[i]['theta']

    avg_pi_0 = t[0]['p'] * pi[0][zero_idx] + t[1]['p'] * pi[1][zero_idx]

    tau_0 = r + pi[0][zero_idx] * (kappa - r) - 0.5 * (sigma**2) * (pi[0][zero_idx]**2)
    tau_1 = r + pi[1][zero_idx] * (kappa - r) - 0.5 * (sigma**2) * (pi[1][zero_idx]**2)
    avg_tau_pi = t[0]['p'] * tau_0 + t[1]['p'] * tau_1

    M_val = 0
    M_val += -theta * avg_tau_pi
    M_val += 0.5 * (theta**2) * (1 - alpha) * (sigma**2) * (avg_pi_0**2)

    def simple_m_jumps(e1):
        return mean_jumps(e1, t, pi)

    M_val += -zero_signal_objective(pi[i][zero_idx], i, pi, t, simple_m_jumps)

    for z in range(6):
        sig = jump_signals[z]
        normal = normalization(sig)
        M_val += lam * t[i]['ps'] * normal * (-non_zero_signal_objective(pi[i][z], sig, i, pi, t, simple_m_jumps))

    return M_val

##########################################################################
## EQUILIBRIA AND CERTAINTY EQUIVALENT IN ALTERNATIVE ENVIRONMENTS       ##
##########################################################################
#
# The reference environment is symmetric (both types carry the same
# parameters). An alternative environment changes one or two parameters of
# one type (or the population share p, which is always adjusted for both
# types so that p^A + p^B = 1). For each alternative environment we solve the
# equilibrium and record the certainty equivalent of type A,
#
#     c = x_0^A = exp(M^{A,alt} - M^{A,ref}),
#
# i.e. the initial capital type A would need in the reference environment to
# be as well off as in the alternative one (c < 1: type A is worse off).

def compute_single_point(x, type_ref, agent, var, ref_M):
    """Solve one alternative environment: parameter ``var`` of type ``agent``
    set to x (for var == 'p' the other type gets 1 - x). Returns the
    environment, the equilibrium profile and the certainty equivalent c."""
    t_list = copy.deepcopy(type_ref)

    if var == 'p':
        t_list[agent]['p'] = x
        t_list[1 - agent]['p'] = 1 - x
    else:
        t_list[agent][var] = x

    alt_t = to_structured(t_list)
    alt_pi = mean_field_equilibrium(alt_t)
    c = np.exp(M(alt_t, 0, alt_pi) - ref_M)
    return {var: x, 't': alt_t, 'pi': alt_pi, 'c': c}

def alternative(filename, type_ref, var, var_values, ref_M):
    """One-dimensional sweep of parameter ``var`` of type B over var_values.
    Skips the computation if ``filename`` exists. Saves var, t, pi, c."""
    ensure_parent_directory(filename)
    if os.path.exists(filename):
        print(f"  [Skip] '{filename}' already exists. Skipping computation.")
        return

    print(f"  [Compute] Generating '{filename}'...")
    agent = 1
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_single_point, x, type_ref, agent, var, ref_M): x for x in var_values}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(var_values), desc=f"Evaluating {var}"):
            results.append(future.result())

    results.sort(key=lambda item: item[var])
    np.savez(filename, var=np.array([res[var] for res in results]), t=np.array([res['t'] for res in results]),
             pi=np.array([res['pi'] for res in results]), c=np.array([res['c'] for res in results]))

def compute_single_point_2d(x, y, type_ref, agent_x, var_x, agent_y, var_y, ref_M):
    """Solve one alternative environment with two parameters changed:
    var_x of type agent_x set to x and var_y of type agent_y set to y."""
    t_list = copy.deepcopy(type_ref)

    for val, var, agent in [(x, var_x, agent_x), (y, var_y, agent_y)]:
        if var == 'p':
            t_list[agent]['p'] = val
            t_list[1 - agent]['p'] = 1 - val
        else:
            t_list[agent][var] = val

    alt_t = to_structured(t_list)
    alt_pi = mean_field_equilibrium(alt_t)
    c = np.exp(M(alt_t, 0, alt_pi) - ref_M)
    return x, y, c

def alternative_2d(filename, type_ref, var_x, x_values, agent_x, var_y, y_values, agent_y, ref_M):
    """Two-dimensional grid sweep. Saves the meshgrids x_grid, y_grid
    (indexing='ij') and the certainty equivalent c_grid."""
    ensure_parent_directory(filename)
    if os.path.exists(filename):
        print(f"  [Skip] '{filename}' already exists. Skipping computation.")
        return

    print(f"  [Compute] Generating '{filename}' (2D Grid)...")
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')
    c_grid = np.zeros_like(x_grid, dtype=float)
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_single_point_2d, x, y, type_ref, agent_x, var_x, agent_y, var_y, ref_M)
                   for x in x_values for y in y_values]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{var_x} x {var_y}"):
            results.append(future.result())

    for x, y, c in results:
        i, j = np.where(x_values == x)[0][0], np.where(y_values == y)[0][0]
        c_grid[i, j] = c

    np.savez(filename, x_grid=x_grid, y_grid=y_grid, c_grid=c_grid, var_x=var_x, var_y=var_y)

##########################
##   EXPERIMENT SUITE   ##
##########################

def run_all_experiments(output_dir, ref_t_list, ref_M, res_1d, res_2d):
    """All parameter sweeps of one scenario. Type B is the deviating type
    unless the agent index says otherwise (agent 0 = type A).

    One-dimensional (res_1d points):
      exp1        ps^B in [0, 1]
      exp2        rho^B in [0, 0.99]
      exp3        theta^B in [0, 1]
      exp4/exp5   theta^B at ps^B = 0.1 / 0.9
      exp6/exp7   theta^B at rho^B = 0.1 / 0.9
      exp_new3_k  p^B in [0.1, 0.9] at rho^B = 0.1 / 0.5 / 0.9
      exp_new4_k  rho^B in [0, 0.9] at ps^B = 0.1 / 0.5 / 0.9

    Two-dimensional (res_2d x res_2d grids):
      exp_new1        rho^B x ps^B
      exp_new2        p^B x ps^B
      exp_new5        theta^B x rho^B
      exp_new6        p^B x rho^B
      exp_act4_parity rho^A x rho^B
      exp_new7        rho^A x rho^B (same grid as exp_act4_parity)
      exp_new8        ps^A x ps^B
      exp_new7_cross  rho^B x ps^A
      exp_new8_cross  ps^B x rho^A
    """
    alternative(scenario_path(output_dir, 'exp1.npz'), ref_t_list, 'ps', np.linspace(0, 1, res_1d), ref_M)
    alternative(scenario_path(output_dir, 'exp2.npz'), ref_t_list, 'rho', np.linspace(0, 0.99, res_1d), ref_M)
    alternative(scenario_path(output_dir, 'exp3.npz'), ref_t_list, 'theta', np.linspace(0, 1, res_1d), ref_M)

    t_list_4 = copy.deepcopy(ref_t_list); t_list_4[1]['ps'] = 0.1
    alternative(scenario_path(output_dir, 'exp4.npz'), t_list_4, 'theta', np.linspace(0, 1, res_1d), ref_M)
    t_list_5 = copy.deepcopy(ref_t_list); t_list_5[1]['ps'] = 0.9
    alternative(scenario_path(output_dir, 'exp5.npz'), t_list_5, 'theta', np.linspace(0, 1, res_1d), ref_M)

    t_list_6 = copy.deepcopy(ref_t_list); t_list_6[1]['rho'] = 0.1
    alternative(scenario_path(output_dir, 'exp6.npz'), t_list_6, 'theta', np.linspace(0, 1, res_1d), ref_M)
    t_list_7 = copy.deepcopy(ref_t_list); t_list_7[1]['rho'] = 0.9
    alternative(scenario_path(output_dir, 'exp7.npz'), t_list_7, 'theta', np.linspace(0, 1, res_1d), ref_M)

    alternative_2d(scenario_path(output_dir, 'exp_new1.npz'), ref_t_list, 'rho', np.linspace(0, 0.99, res_2d), 1, 'ps', np.linspace(0, 1, res_2d), 1, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new2.npz'), ref_t_list, 'p', np.linspace(0.1, 0.9, res_2d), 1, 'ps', np.linspace(0, 1, res_2d), 1, ref_M)

    t_n3_1 = copy.deepcopy(ref_t_list); t_n3_1[1]['rho'] = 0.1
    alternative(scenario_path(output_dir, 'exp_new3_1.npz'), t_n3_1, 'p', np.linspace(0.1, 0.9, res_1d), ref_M)
    t_n3_2 = copy.deepcopy(ref_t_list); t_n3_2[1]['rho'] = 0.5
    alternative(scenario_path(output_dir, 'exp_new3_2.npz'), t_n3_2, 'p', np.linspace(0.1, 0.9, res_1d), ref_M)
    t_n3_3 = copy.deepcopy(ref_t_list); t_n3_3[1]['rho'] = 0.9
    alternative(scenario_path(output_dir, 'exp_new3_3.npz'), t_n3_3, 'p', np.linspace(0.1, 0.9, res_1d), ref_M)

    t_n4_1 = copy.deepcopy(ref_t_list); t_n4_1[1]['ps'] = 0.1
    alternative(scenario_path(output_dir, 'exp_new4_1.npz'), t_n4_1, 'rho', np.linspace(0, 0.9, res_1d), ref_M)
    t_n4_2 = copy.deepcopy(ref_t_list); t_n4_2[1]['ps'] = 0.5
    alternative(scenario_path(output_dir, 'exp_new4_2.npz'), t_n4_2, 'rho', np.linspace(0, 0.9, res_1d), ref_M)
    t_n4_3 = copy.deepcopy(ref_t_list); t_n4_3[1]['ps'] = 0.9
    alternative(scenario_path(output_dir, 'exp_new4_3.npz'), t_n4_3, 'rho', np.linspace(0, 0.9, res_1d), ref_M)

    alternative_2d(scenario_path(output_dir, 'exp_new5.npz'), ref_t_list, 'theta', np.linspace(0, 1, res_2d), 1, 'rho', np.linspace(0, 0.99, res_2d), 1, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new6.npz'), ref_t_list, 'p', np.linspace(0.1, 0.99, res_2d), 1, 'rho', np.linspace(0, 0.99, res_2d), 1, ref_M)

    alternative_2d(scenario_path(output_dir, 'exp_act4_parity.npz'), ref_t_list, 'rho', np.linspace(0, 0.99, res_2d), 0, 'rho', np.linspace(0, 0.99, res_2d), 1, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new7.npz'), ref_t_list, 'rho', np.linspace(0, 0.99, res_2d), 0, 'rho', np.linspace(0, 0.99, res_2d), 1, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new8.npz'), ref_t_list, 'ps', np.linspace(0, 1, res_2d), 0, 'ps', np.linspace(0, 1, res_2d), 1, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new7_cross.npz'), ref_t_list, 'rho', np.linspace(0, 0.99, res_2d), 1, 'ps', np.linspace(0, 1, res_2d), 0, ref_M)
    alternative_2d(scenario_path(output_dir, 'exp_new8_cross.npz'), ref_t_list, 'ps', np.linspace(0, 1, res_2d), 1, 'rho', np.linspace(0, 0.99, res_2d), 0, ref_M)

def run_scenario(output_dir, base_params, res_1d, res_2d):
    """Solve one scenario: reference equilibrium (cached in
    ref_equilibrium.npz), then all experiments."""
    ensure_directory(output_dir)

    ref_t_list = [base_params.copy(), base_params.copy()]
    ref_t_array = to_structured(ref_t_list)
    ref_filename = scenario_path(output_dir, 'ref_equilibrium.npz')

    print(f"\n=== Scenario: {os.path.basename(output_dir)} ===")
    print(f"Reference parameters: theta={base_params['theta']}, alpha={base_params['alpha']}")

    if os.path.exists(ref_filename):
        print(f"  [Skip] '{ref_filename}' already exists. Loading reference equilibrium.")
        ref_data = np.load(ref_filename)
        ref_pi = ref_data['pi']
        ref_M = float(ref_data['M'])
    else:
        print("  [Compute] Computing reference equilibrium...")
        ref_pi = mean_field_equilibrium(ref_t_array)
        ref_M = M(ref_t_array, 0, ref_pi)
        np.savez(ref_filename, pi=ref_pi, M=ref_M)

    print("Running experiments... (This will utilize multiple CPU cores if computing)")
    run_all_experiments(output_dir, ref_t_list, ref_M, res_1d, res_2d)
    print("Experiments complete.")

####################
##   EXECUTION    ##
####################

if __name__ == '__main__':
    # The six scenarios of the thesis: relative performance concern theta in
    # {0.5, 1} and relative risk aversion alpha in {0.5, 2, 4}, applied to
    # both types in the reference environment. res_1d points per 1D sweep,
    # res_2d x res_2d points per grid.
    base_params = default_base_params.copy()
    theta_values = [0.5, 1.0]
    alpha_values = [0.5, 2.0, 4.0]
    scenario_root = 'scenario_outputs'

    res_1d = 100
    res_2d = 30

    ensure_directory(scenario_root)
    for theta_value in theta_values:
        for alpha_value in alpha_values:
            scenario_params = base_params.copy()
            scenario_params['theta'] = theta_value
            scenario_params['alpha'] = alpha_value
            run_scenario(
                scenario_path(scenario_root, scenario_folder_name(theta_value, alpha_value)),
                scenario_params,
                res_1d,
                res_2d,
            )
