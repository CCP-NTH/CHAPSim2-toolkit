"""Verify the Ruu (uu11) budget terms in operations.py against reference data.

The reference is the Vreman & Kuerten Chan180_FD2 channel-flow dataset. The
mean profiles, Reynolds stresses, gradients and correlations it provides are
fed through the budget functions, and the terms they return are compared with
the dataset's own budget file.

Code assumes homogeneity in x and z directions.
"""

import contextlib
import io
import types
from pathlib import Path

import numpy as np

from chapsim2_toolkit import operations as op
from chapsim2_toolkit import utils as ut
from chapsim2_toolkit.turb_stats import BudgetComputer, Config

DATA = Path(__file__).parent / "reference_data" / "Chan180_FD2_all"


def test_ruu_budget_matches_chan180():
    # ── constants ──────────────────────────────────────────────────────────
    Re_tau = 180
    Re     = Re_tau                   # Re = 1/nu used by operations.py

    # ── load data ──────────────────────────────────────────────────────────
    basic_u = np.loadtxt(DATA / "Chan180_FD2_basic_u.txt", comments="%")
    basic_v = np.loadtxt(DATA / "Chan180_FD2_basic_v.txt", comments="%")
    basic_w = np.loadtxt(DATA / "Chan180_FD2_basic_w.txt", comments="%")
    basic_p = np.loadtxt(DATA / "Chan180_FD2_basic_p.txt", comments="%")
    grad_yc = np.loadtxt(DATA / "Chan180_FD2_grad_yc.txt", comments="%")
    grad_ys = np.loadtxt(DATA / "Chan180_FD2_grad_ys.txt", comments="%")
    budget  = np.loadtxt(DATA / "Chan180_FD2_budget_Ruu.txt", comments="%")

    # ── grids ──────────────────────────────────────────────────────────────
    yp_c = basic_u[:, 0]             # y+ cell-centres (128 pts)
    yp_s = grad_ys[:, 0]             # y+ staggered    (129 pts)
    y_c  = yp_c / Re_tau             # physical y
    N    = len(yp_c)
    zeros = np.zeros(N)

    # interpolate staggered → cell-centre
    def s2c(f_s, yp_src=None):
        if yp_src is None:
            yp_src = yp_s
        return np.interp(yp_c, yp_src, f_s)

    # ═══════════════════════════════════════════════════════════════════════
    # EXTRACT RAW QUANTITIES
    # ═══════════════════════════════════════════════════════════════════════
    # Mean velocities (on yc)
    yp_v = basic_v[:, 0]  # y+ for basic_v (staggered, 128 pts w/o wall)
    U = basic_u[:, 1];  V = s2c(basic_v[:, 1], yp_v);  W = basic_w[:, 1]

    # Reynolds stresses <u'_i u'_j>
    uu = basic_u[:, 2]**2;  vv = s2c(basic_v[:, 2]**2, yp_v);  ww = basic_w[:, 2]**2
    uv = s2c(basic_v[:, 5], yp_v);  uw = basic_u[:, 6];  vw = s2c(basic_v[:, 6], yp_v)

    # Mean velocity gradient: only d<u>/dy nonzero in channel flow
    dudy = s2c(grad_ys[:, 1])

    # Dissipation-tensor diagonal for (0,0): sum_k <(du'/dx_k)^2>
    diss_11 = grad_yc[:, 3]**2 + s2c(grad_ys[:, 4])**2 + grad_yc[:, 5]**2

    # Triple correlation <u'u'v'> (for turbulent diffusion)
    uuv = basic_u[:, 5]

    # Pressure-velocity <p'u'_i>
    pu = basic_p[:, 3];  pv = basic_p[:, 4];  pw = basic_p[:, 5]

    # Pressure-strain correlations <p' du'_i/dx_j> (only diagonals available)
    pdudx_11 = basic_p[:, 6]   # <p' du'/dx>
    pdudx_22 = basic_p[:, 7]   # <p' dv'/dy>
    pdudx_33 = basic_p[:, 8]   # <p' dw'/dz>

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD tke_comp_dict  (same format as op.compute_TKE_components output)
    # ═══════════════════════════════════════════════════════════════════════

    # Reynolds stress tensor (3,3,N)
    reynolds_stress_tensor = np.array([
        [uu, uv, uw],
        [uv, vv, vw],
        [uw, vw, ww],
    ])

    # Mean velocity gradient tensor (3,3,N) — only dU/dy nonzero
    mean_velocity_grad_tensor = np.zeros((3, 3, N))
    mean_velocity_grad_tensor[0, 1] = dudy

    # Dissipation tensor (3,3,N) — only (0,0) needed for uu11
    dissipation_tensor = np.zeros((3, 3, N))
    dissipation_tensor[0, 0] = diss_11

    # Turbulent convection tensors: d<u'_i u'_j u'_k>/dx_k per direction
    # Only y-direction (x2) nonzero in channel flow
    turb_conv_tensor_x1 = np.zeros((3, 3, N))
    turb_conv_tensor_x2 = np.zeros((3, 3, N))
    turb_conv_tensor_x2[0, 0] = np.gradient(uuv, y_c)   # d<u'u'v'>/dy
    turb_conv_tensor_x3 = np.zeros((3, 3, N))

    # Laplacian of Reynolds stresses: d²<u'_i u'_j>/dy² (only y matters)
    lap_re_stress_x2 = np.zeros((3, 3, N))
    lap_re_stress_x2[0, 0] = op.second_derivative(uu, y_c, axis=0)

    # Pressure-velocity fluctuation gradient: d<p'u'_i>/dx_j (only y nonzero)
    press_vel_grad = np.zeros((3, 3, N))
    press_vel_grad[0, 1] = np.gradient(pu, y_c)

    # Pressure-strain correlation tensor <p' du'_i/dx_j> (3,3,N)
    # Chan180 data provides diagonal terms directly as fluctuation correlations
    pressure_strain_tensor = np.zeros((3, 3, N))
    pressure_strain_tensor[0, 0] = pdudx_11
    pressure_strain_tensor[1, 1] = pdudx_22
    pressure_strain_tensor[2, 2] = pdudx_33

    # Assemble the dictionary
    tke_comp_dict = {
        'U1': U, 'U2': V, 'U3': W,
        'pr': basic_p[:, 1],
        'TKE': 0.5 * (uu + vv + ww),
        'mean_velocity_grad_tensor':        mean_velocity_grad_tensor,
        'fluc_velocity_grad_tensor':        np.zeros((3, 3, N)),
        'reynolds_stress_tensor':           reynolds_stress_tensor,
        'dissipation_tensor':               dissipation_tensor,
        'turb_conv_tensor_x1':              turb_conv_tensor_x1,
        'turb_conv_tensor_x2':              turb_conv_tensor_x2,
        'turb_conv_tensor_x3':              turb_conv_tensor_x3,
        'lap_re_stress_tensor_x1':          np.zeros((3, 3, N)),
        'lap_re_stress_tensor_x2':          lap_re_stress_x2,
        'lap_re_stress_tensor_x3':          np.zeros((3, 3, N)),
        'pr_prime':                         zeros,
        'press_velocity_fluc_grad_tensor':  press_vel_grad,
        'pressure_strain_tensor':           pressure_strain_tensor,
        'mean_conv_tensor_x1':              np.zeros((3, 3, N)),
        'mean_conv_tensor_x2':              np.zeros((3, 3, N)),
        'mean_conv_tensor_x3':              np.zeros((3, 3, N)),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # CALL operations.py BUDGET FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════
    production   = next(iter(op.compute_production(tke_comp_dict, 'uu11').values()))
    dissipation  = next(iter(op.compute_dissipation(Re, tke_comp_dict, 'uu11').values()))
    turb_diff    = next(iter(op.compute_turbulent_convection(tke_comp_dict, 'uu11').values()))
    visc_diff    = next(iter(op.compute_viscous_diffusion(Re, tke_comp_dict, 'uu11').values()))
    press_trans  = next(iter(op.compute_pressure_transport(tke_comp_dict, 'uu11').values()))
    press_strain = next(iter(op.compute_pressure_strain(tke_comp_dict, 'uu11').values()))

    # ═══════════════════════════════════════════════════════════════════════
    # REFERENCE VALUES
    # ═══════════════════════════════════════════════════════════════════════
    ref_prod = budget[:, 1];  ref_pstrain = budget[:, 2];  ref_diss = budget[:, 3]
    ref_turb = budget[:, 4];  ref_pdiff   = budget[:, 5];  ref_visc = budget[:, 6]

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARE (skip boundary points where FD stencils are unreliable)
    # ═══════════════════════════════════════════════════════════════════════
    # Skip boundary points where FD stencils are unreliable. Each tolerance is
    # a fraction of the term's peak, plus a small absolute floor for the
    # pressure transport, whose reference is zero to round-off.
    s = slice(2, -2)
    terms = [
        # name,                computed,     reference,  relative tolerance
        ("Production",          production,   ref_prod,    5e-3),
        ("Dissipation",         dissipation,  ref_diss,    1e-3),
        ("Turbulent diffusion", turb_diff,    ref_turb,    2e-2),
        ("Viscous diffusion",   visc_diff,    ref_visc,    1e-3),
        ("Pressure transport",  press_trans,  ref_pdiff,   1e-3),
        ("Pressure strain",     press_strain, ref_pstrain, 1e-3),
    ]
    for name, computed, reference, tolerance in terms:
        err  = np.abs(computed[s] - reference[s]).max()
        rmax = np.abs(reference[s]).max()
        assert err <= tolerance * rmax + 1e-9, \
            f"{name}: max|err| = {err:.3e} against a peak of {rmax:.3e}"


# ── term selection and the Joule split, on synthetic profiles ─────────────────
NY = 24
N_STUART = 0.4


def mhd_profiles(seed=0):
    """Every variable a budget can read, as random wall-normal profiles."""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(NY) for name in ut._BASE_REQUIRED_VARS}


def budget(terms, component='total', raw=None):
    """A BudgetComputer for an MHD case, run once on `raw`; returns it and what it printed."""
    config = Config.from_module(types.SimpleNamespace(
        cases=['c'], timesteps=['1'], Re=[180.0], mhd_on=True, stuart_number=N_STUART,
        re_stress_budget_on=True, re_stress_component=component, re_stress_budget_terms=terms,
        average_x_direction=True, average_z_direction=True))
    loader = types.SimpleNamespace(get_raw_dict=lambda case, ts: raw or mhd_profiles(),
                                   y_coords=np.linspace(-1.0, 1.0, NY))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        computer = BudgetComputer(config)
        computer.compute_for_case('c', '1', loader)
    return computer, printed.getvalue()


def test_joule_dissipation_is_the_u_cross_b_work():
    """J_ij against N(<u'_i f_j> + <u'_j f_i>), f = (u' x B) x B, from velocity samples."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal((5000, 3)) @ rng.standard_normal((3, 3))      # correlated fluctuations
    B = np.array([0.3, 0.8, -0.5])                                          # any field, not a unit one
    f = np.cross(np.cross(u, B), B)
    work = N_STUART * (u.T @ f + f.T @ u) / len(u)
    d = {'reynolds_stress_tensor': u.T @ u / len(u)}
    for c, (i, j) in {'uu11': (0, 0), 'uu12': (0, 1), 'uu13': (0, 2), 'uu22': (1, 1), 'uu33': (2, 2)}.items():
        assert np.isclose(op.compute_joule_dissipation(B, N_STUART, d, c)[f'joule_{c}'], work[i, j]), c
    assert np.isclose(op.compute_joule_dissipation(B, N_STUART, d)['joule'], 0.5 * np.trace(work)), \
        'the TKE takes half the trace, like the other terms'


def test_joule_split_of_the_lorentz_term():
    with contextlib.redirect_stdout(io.StringIO()):
        d = op.compute_budget_components(mhd_profiles(), np.linspace(-1.0, 1.0, NY), average_z=True, average_x=True)
    R, B = d['reynolds_stress_tensor'], [0.0, 1.0, 0.0]                    # a wall-normal field
    joule = {}
    for c in ('total', 'uu11', 'uu12', 'uu22', 'uu33'):
        lorentz = next(iter(op.compute_mhd_term(B, N_STUART, d, c).values()))
        joule[c] = next(iter(op.compute_joule_dissipation(B, N_STUART, d, c).values()))
        potential = op.compute_electric_potential_term(B, N_STUART, d, c)['electric_potential']
        assert np.allclose(joule[c] + potential, lorentz), f'{c}: Joule + electric potential = Lorentz term'
    assert np.allclose(joule['uu11'], -2 * N_STUART * R[0, 0]) and np.allclose(joule['uu33'], -2 * N_STUART * R[2, 2])
    assert np.allclose(joule['uu22'], 0.0), 'nothing is damped along the field'
    assert np.allclose(joule['uu12'], -N_STUART * R[0, 1])
    assert np.allclose(joule['total'], -N_STUART * (R[0, 0] + R[2, 2])), "TKE: -N <|u' x B|^2>"


def test_budget_plots_only_the_selected_terms():
    computer, _ = budget(['joule', 'dissipation', 'balance'])
    assert computer.terms == ['dissipation', 'joule', 'balance'], 'the selection, in plotting order'
    raw = computer.raw_results
    assert np.allclose(raw['balance'][('c', '1')], raw['dissipation'][('c', '1')] + raw['joule'][('c', '1')]), \
        'the balance sums the selected terms'

    for component in ('total', 'uu11'):
        everything, _ = budget(None, component=component)
        assert everything.terms == BudgetComputer.terms_for(component, False, True), 'no selection means every term'
        assert {'joule', 'electric_potential'} <= set(everything.terms), f'{component}: the Lorentz term is split'
    assert budget('dissipation, balance')[0].terms == ['dissipation', 'balance'], 'a string list works too'

    skipped, printed = budget(['joule', 'buoyancy'], component='uu11')
    assert skipped.terms == ['joule'] and "'buoyancy' is not a term of this uu11 budget" in printed, \
        'a term this budget does not have (buoyancy without thermal statistics) is named and skipped'

    no_currents = {k: v for k, v in mhd_profiles().items() if not k.startswith('ju')}
    unavailable, _ = budget(['joule', 'electric_potential'], raw=no_currents)
    assert np.all(np.isnan(unavailable.raw_results['electric_potential'][('c', '1')])), \
        'without <u_i j_l> the electric potential term is marked unavailable, not shown as -J'
    assert not np.any(np.isnan(unavailable.raw_results['joule'][('c', '1')])), \
        'while the Joule dissipation needs only the Reynolds stresses'
