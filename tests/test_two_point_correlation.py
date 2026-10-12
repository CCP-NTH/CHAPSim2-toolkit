"""The spanwise two-point correlation against analytic cases.

The kernels in operations.py are checked directly, and the whole
TwoPointCorrelationComputer path -- XDMF discovery, fluctuation construction,
x-station selection, slab chunking, centreline folding, half-channel reduction
-- is checked end to end on a synthetic case written to a temporary directory
in the same layout CHAPSim2 produces
(case/2_visu/{xdmf,data,mesh}).

The synthetic field is a single spanwise Fourier mode,

    u(z, y, x) = U(y, x) + A(y, x) cos(k z)
    v(z, y, x) = V(y, x) + B(y, x) cos(k z + phi)

whose correlations are known exactly:

    rho_uu(dz) = cos(k dz),  rho_uv(dz) = cos(k dz + phi),  L_z = 1/k

so every number below has an analytic target rather than a reference dataset.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

from _harness import attribute_xml, write_bin, write_xdmf
from chapsim2_toolkit import operations as op
from chapsim2_toolkit.turb_stats import TwoPointCorrelationComputer

# ── synthetic case definition ──────────────────────────────────────────
NZ, NY, NX = 48, 17, 12
LZ = 2.0 * np.pi
MODE = 1
K = 2.0 * np.pi * MODE / LZ          # spanwise wavenumber of the planted mode
PHI = np.pi / 3.0                    # u-v phase lag -> rho_uv(0) = cos(PHI) = 0.5
CASE, TIMESTEP = 'synthetic', '000100'


# ── writing a synthetic CHAPSim2 visu case ─────────────────────────────
def build_case(root, with_t_avg=True, t_avg_letters=('u1', 'u2'), with_tsp_avg=False,
               asymmetric=False, t_avg_offset=0.0):
    """Write the synthetic case and return its exact fields for comparison.

    t_avg_letters picks which mean velocities the t_avg file holds, with_tsp_avg
    adds a spanwise-averaged (zi1) file holding u1 only, asymmetric makes the
    u' amplitude differ between the lower and upper walls, and t_avg_offset
    shifts the t_avg means away from the snapshot's own spanwise mean.
    """
    visu = root / CASE / '2_visu' / 'xdmf'
    data = root / CASE / '2_visu' / 'data'
    mesh = root / CASE / '2_visu' / 'mesh'
    for directory in (visu, data, mesh):
        directory.mkdir(parents=True)

    x_nodes = np.linspace(0.0, 4.0, NX + 1)
    y_nodes = np.linspace(-1.0, 1.0, NY + 1)
    z_nodes = np.linspace(0.0, LZ, NZ + 1)
    for axis, nodes in (('x', x_nodes), ('y', y_nodes), ('z', z_nodes)):
        write_bin(mesh / f'domain1_grid_{axis}.bin', nodes, header_bytes=4)

    y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    z_cell = 0.5 * (z_nodes[:-1] + z_nodes[1:])

    # Amplitudes vary with both y and x, so a wrongly-sliced station or a
    # mis-broadcast mean shows up as a wrong answer rather than passing by luck.
    amp_u = 1.0 + 0.5 * np.cos(np.pi * y_cell)[:, None] + 0.1 * x_cell[None, :]
    amp_v = 0.4 + 0.2 * y_cell[:, None] ** 2 + 0.05 * x_cell[None, :]
    mean_u = 10.0 * (1.0 - y_cell ** 2)[:, None] + x_cell[None, :]
    mean_v = -3.0 + 0.5 * y_cell[:, None] - 0.2 * x_cell[None, :]
    if asymmetric:
        # Like a one-side-heated channel: the upper wall's turbulence is much weaker.
        amp_u = amp_u * (1.0 - 0.8 * y_cell[:, None])

    wave = np.cos(K * z_cell)[:, None, None]
    wave_shift = np.cos(K * z_cell + PHI)[:, None, None]
    u_inst = mean_u[None, :, :] + amp_u[None, :, :] * wave
    v_inst = mean_v[None, :, :] + amp_v[None, :, :] * wave_shift

    write_bin(data / f'domain1_qx_ccc_{TIMESTEP}.bin', u_inst)
    write_bin(data / f'domain1_qy_ccc_{TIMESTEP}.bin', v_inst)
    write_xdmf(visu / f'domain1_flow_{TIMESTEP}.xdmf', 'flow', (NZ + 1, NY + 1, NX + 1),
                [attribute_xml('qx_ccc', f'domain1_qx_ccc_{TIMESTEP}.bin', (NZ, NY, NX)),
                 attribute_xml('qy_ccc', f'domain1_qy_ccc_{TIMESTEP}.bin', (NZ, NY, NX))])

    if with_t_avg:
        means = {'u1': mean_u, 'u2': mean_v}
        for name in t_avg_letters:
            write_bin(data / f'domain1_t_avg_{name}_{TIMESTEP}.bin',
                       np.broadcast_to(means[name][None, :, :] + t_avg_offset, (NZ, NY, NX)))
        write_xdmf(visu / f'domain1_t_avg_flow_{TIMESTEP}.xdmf', 't_avg_flow',
                    (NZ + 1, NY + 1, NX + 1),
                    [attribute_xml(f't_avg_{name}', f'domain1_t_avg_{name}_{TIMESTEP}.bin', (NZ, NY, NX))
                     for name in t_avg_letters])

    if with_tsp_avg:
        write_bin(data / f'domain1_tsp_avg_u1_zi1_{TIMESTEP}.bin', mean_u[None, :, :])
        write_xdmf(visu / f'domain1_tsp_avg_flow_zi1_{TIMESTEP}.xdmf', 'tsp_avg_flow',
                    (2, NY + 1, NX + 1),
                    [attribute_xml('tsp_avg_u1', f'domain1_tsp_avg_u1_zi1_{TIMESTEP}.bin', (1, NY, NX))])

    return dict(y_cell=y_cell, x_cell=x_cell, z_cell=z_cell,
                amp_u=amp_u, amp_v=amp_v, u_inst=u_inst, v_inst=v_inst)


def run(root, **kwargs):
    """Run the computer over the synthetic case and return its results dict."""
    options = dict(components='uu', y_coords_str='', x_coords_str='',
                   timesteps=[TIMESTEP], mean_mode='t_avg')
    options.update(kwargs)
    computer = TwoPointCorrelationComputer(str(root), **options)
    if not computer.compute_for_case(CASE, TIMESTEP):
        return None
    return computer.processed_results[(CASE, TIMESTEP)]


def run_quietly(root, **kwargs):
    """run(), also returning everything it printed."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        out = run(root, **kwargs)
    return out, printed.getvalue()


# Each test builds its own case in a temporary directory: the case is tiny,
# and it keeps every test runnable on its own under the bundled runner.

def test_kernels_match_analytic_answers():
    z = np.arange(NZ) * LZ / NZ
    signal = np.broadcast_to(np.cos(K * z)[:, None], (NZ, 3)).copy()
    R = op.compute_two_point_correlation_z(signal, signal)
    rho = op.normalise_correlation(R)
    sep = z[:R.shape[0]]
    assert np.allclose(rho[:, 0], np.cos(K * sep), atol=1e-12), 'rho of a single mode is cos(k dz)'
    L = op.compute_integral_length_scale(rho, sep)
    assert abs(L[0] - 1.0 / K) < 0.01 / K, 'integral scale of that mode is 1/k'
    L_neg = op.compute_integral_length_scale(-rho, sep)
    assert np.isclose(L_neg[0], L[0]), 'a correlation starting negative has the same L (not 0)'
    y_odd = np.linspace(-1.0, 1.0, 9)
    R_odd = np.cos(K * sep)[:, None] * y_odd[None, :]
    assert np.allclose(op.symmetry_average_correlation(R_odd, 1, axis=1), R_odd), \
        'folding an odd-in-y R_uv with the odd sign keeps it'
    assert np.allclose(op.compute_two_point_correlation_z(signal, signal, max_sep=7),
                       np.stack([(signal * np.roll(signal, -d, axis=0)).mean(axis=0) for d in range(7)])), \
        'periodic FFT estimator == direct wrap-around sum'


def test_correlation_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exact = build_case(root)
        sep_exact = exact['z_cell'][:NZ // 2 + 1] - exact['z_cell'][0]

        out = run(root, components='uu,uv', x_coords_str='0.5, 3.5')
        assert out is not None and set(out) == {'uu', 'uv'}, 'computer produced both components'

        uu = out['uu']
        assert np.allclose(uu['sep'], sep_exact), 'separation axis matches the z grid'
        assert uu['R'].shape == (NZ // 2 + 1, NY, 2), 'two x stations selected'
        assert np.allclose(uu['rho'][0], 1.0), 'rho_uu(0) == 1 everywhere'
        target = np.cos(K * sep_exact)[:, None, None]
        assert np.allclose(uu['rho'], target, atol=1e-10), 'rho_uu(dz) == cos(k dz) at every y and x'
        # The cell nearest each requested station, so the check does not depend on
        # where the request happens to land in the grid.
        station_idx = [int(np.argmin(np.abs(exact['x_cell'] - v))) for v in (0.5, 3.5)]
        assert np.allclose(uu['R'][0], 0.5 * exact['amp_u'][:, station_idx] ** 2), \
            "R_uu(0) == <u' u'> of the planted field"

        uv = out['uv']
        target_uv = np.cos(K * sep_exact + PHI)[:, None, None]
        assert np.allclose(uv['rho'], target_uv, atol=1e-10), 'rho_uv(dz) == cos(k dz + phi)'
        assert np.allclose(uv['rho'][0], np.cos(PHI)), 'rho_uv(0) == cos(phi) = 0.5'
        assert np.abs(uv['rho']).max() <= 1.0 + 1e-12, 'cross-correlation stays bounded by 1'

        assert np.allclose(uu['L'], 1.0 / K, rtol=0.01), 'L_z == 1/k at every y and x'

        snap = run(root, components='uu', x_coords_str='0.5, 3.5', mean_mode='snapshot')
        assert np.allclose(snap['uu']['rho'], uu['rho'], atol=1e-10), \
            'snapshot mean gives the same rho as the t_avg mean (the planted mean is exact)'
        assert not np.allclose(uu['R'][0], (exact['u_inst'] ** 2).mean(axis=0)[:, station_idx]), \
            'but the raw R differs from an unsubtracted mean'


def test_x_station_handling():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exact = build_case(root)
        mid = run(root, components='uu')
        assert mid['uu']['R'].shape[2] == 1, 'no x stations given -> one mid-domain station'
        assert np.isclose(mid['uu']['x'][0], exact['x_cell'][NX // 2]), 'mid station is nx//2'
        avg = run(root, components='uu', average_x=True)
        assert avg['uu']['R'].shape[2] == 1, 'average_x -> a single x-averaged profile'
        assert np.allclose(avg['uu']['R'][0, :, 0], (0.5 * exact['amp_u'] ** 2).mean(axis=1), atol=1e-10), \
            'x-averaged R == mean of all stations'


def test_slab_chunking_matches_single_slab():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root)
        original_chunk = TwoPointCorrelationComputer._X_CHUNK
        try:
            TwoPointCorrelationComputer._X_CHUNK = 5     # forces 3 slabs over 12 columns
            chunked = run(root, components='uu,uv', average_x=True)
            TwoPointCorrelationComputer._X_CHUNK = 10_000
            single = run(root, components='uu,uv', average_x=True)
        finally:
            TwoPointCorrelationComputer._X_CHUNK = original_chunk
        assert np.allclose(chunked['uu']['R'], single['uu']['R'], atol=1e-12)
        assert np.allclose(chunked['uv']['R'], single['uv']['R'], atol=1e-12)


def test_half_channel_selection():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root)
        full = run(root, components='uu,uv', x_coords_str='0.5')
        uu_R = full['uu']['R']
        assert uu_R.shape[1] == NY, 'full channel keeps every y row, unfolded'

        half_len = NY - NY // 2
        lower = run(root, components='uu', x_coords_str='0.5', half_channel_side='lower')
        upper = run(root, components='uu', x_coords_str='0.5', half_channel_side='upper')
        both = run(root, components='uu', x_coords_str='0.5', half_channel_side='average')
        assert lower['uu']['R'].shape[1] == half_len, 'lower half keeps n - n//2 rows'
        assert np.allclose(lower['uu']['R'][0, :, 0], uu_R[0, :half_len, 0]), 'lower half is the y=-1 side'
        assert np.allclose(upper['uu']['R'][0, :, 0], np.flip(uu_R[0, :, 0])[:half_len]), \
            'upper half is the flipped y=+1 side'
        assert np.isclose(lower['uu']['y'][0], upper['uu']['y'][0]), 'both halves start at the wall'
        assert lower['uu']['y'][0] < lower['uu']['y'][-1], 'y is wall distance, increasing inward'
        assert np.allclose(both['uu']['R'][0, :, 0], op.symmetric_average(uu_R[0, :, 0])), \
            "'average' side averages the two halves"
        # 'average' folds R_uv with the (-1)^1 sign; the planted R_uv is even in
        # y (not the odd profile a real channel has), so that fold must cancel it.
        uv_avg = run(root, components='uv', x_coords_str='0.5', half_channel_side='average')
        assert np.allclose(uv_avg['uv']['R'], 0.0, atol=1e-12), \
            "'average' folds uv with the odd sign (an even R_uv cancels)"


def test_symmetry_check_before_folding():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root)
        _, printed = run_quietly(root, components='uu', x_coords_str='0.5', half_channel_side='average')
        assert 'WARNING' not in printed, 'a symmetric flow folds without a warning'

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root, asymmetric=True)
        _, printed = run_quietly(root, components='uu', x_coords_str='0.5', half_channel_side='average')
        assert 'not symmetric' in printed, 'an asymmetric flow is flagged when folded'
        _, printed = run_quietly(root, components='uu', x_coords_str='0.5', half_channel_side='lower')
        assert 'WARNING' not in printed, 'and is not flagged when only one wall is used'


def test_max_separation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root)
        one = run(root, components='uu', x_coords_str='0.5', max_sep=1)
        assert one['uu']['R'].shape[0] == 2 and np.isclose(one['uu']['sep'][-1], LZ / NZ), \
            'max_sep counts cells: 1 cell -> separations 0 and dz'
        negative, _ = run_quietly(root, components='uu', x_coords_str='0.5', max_sep=-3)
        assert negative is not None and negative['uu']['R'].shape[0] == NZ // 2 + 1, \
            'a negative max_sep falls back to half the span instead of raising'
        too_big = run(root, components='uu', x_coords_str='0.5', max_sep=10 * NZ)
        assert too_big['uu']['R'].shape[0] == NZ, 'max_sep beyond the domain is clamped to nz - 1 cells'
        _, printed = run_quietly(root, components='uu', x_coords_str='0.5', max_sep=3)
        assert 'L is truncated' in printed, 'a separation too short to reach rho = 0 is reported'


def test_missing_t_avg_falls_back_to_snapshot_mean():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root, with_t_avg=False)
        fallback = run(root, components='uu', x_coords_str='0.5')
        assert fallback is not None and np.allclose(fallback['uu']['rho'][0], 1.0)
        assert fallback['uu']['mean'] == 'u: snapshot mean'


def test_missing_t_avg_v_leaves_u_mean_alone():
    # The t_avg mean is offset from the snapshot's own spanwise mean, so the
    # two decompositions give different R and a silent switch would show.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root, t_avg_letters=('u1',), t_avg_offset=0.3)
        alone = run(root, components='uu', x_coords_str='0.5')
        with_v = run(root, components='uu,uv', x_coords_str='0.5')
        snapshot = run(root, components='uu', x_coords_str='0.5', mean_mode='snapshot')
        assert np.allclose(with_v['uu']['R'], alone['uu']['R'])
        assert not np.allclose(with_v['uu']['R'], snapshot['uu']['R'])
        assert with_v['uu']['mean'] == 'u: t_avg'
        assert with_v['uv']['mean'] == 'u: t_avg, v: snapshot mean'


def test_tsp_avg_mean_is_used_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root)
        reference = run(root, components='uu,uv', x_coords_str='0.5, 3.5')

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_case(root, with_tsp_avg=True)
        tsp = run(root, components='uu,uv', x_coords_str='0.5, 3.5')
        assert tsp['uv']['mean'] == 'u: tsp_avg, v: t_avg', \
            'the small tsp_avg mean is used first, t_avg only for what it lacks'
        assert np.allclose(tsp['uu']['R'], reference['uu']['R'])
        assert np.allclose(tsp['uv']['R'], reference['uv']['R'])

        # A second snapshot of the same field with no averaged files of its own.
        second = '000200'
        xdmf_dir = root / CASE / '2_visu' / 'xdmf'
        flow = (xdmf_dir / f'domain1_flow_{TIMESTEP}.xdmf').read_text()
        (xdmf_dir / f'domain1_flow_{second}.xdmf').write_text(flow)
        computer = TwoPointCorrelationComputer(str(root), 'uu', '', '0.5', timesteps=[TIMESTEP, second])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            computer.compute_for_case(CASE, 'avg')
        assert 'mixes' in printed.getvalue(), 'averaging snapshots that removed different means is flagged'
        assert computer.processed_results[(CASE, 'avg')]['uu']['mean'] == 'mixed'


def test_missing_case_is_reported_not_raised():
    with tempfile.TemporaryDirectory() as tmp:
        assert run(Path(tmp) / 'nonexistent', components='uu') is None


def test_spectrum_consistency_wiener_khinchin():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exact = build_case(root)
        line = exact['u_inst'][:, 5, 3]
        line = line - line.mean()
        k_axis, E = op.compute_1d_spectrum(line, dx=float(LZ / NZ))
        R_line = op.compute_two_point_correlation_z(line, line)
        assert np.isclose(E.sum(), R_line[0]), 'sum(E(k)) == R(0)'
        assert np.isclose(k_axis[int(np.argmax(E))], K), 'spectral peak sits at the planted wavenumber'
