"""The 1D energy spectrum against analytic cases.

The whole SpectrumComputer path -- XDMF discovery, y and x-station selection,
line averaging, dropping the k = 0 and Nyquist modes, averaging over
snapshots -- is checked end to end on synthetic cases written in the layout
CHAPSim2 produces (case/2_visu/*.xdmf + case/1_data/*.bin).

The planted fields are single Fourier modes, e.g.

    u(z, y, x) = U(y, x) + A(y, x) cos(kz z)

whose energy per mode is known exactly: A^2/2 at kz, nothing in any other
mode -- so every number below has an analytic target.
"""

import contextlib
import io
import tempfile
import types
from pathlib import Path

import numpy as np

from _harness import attribute_xml, write_bin, write_xdmf
from chapsim2_toolkit import operations as op
from chapsim2_toolkit.turb_stats import (Config, SpectrumComputer, TurbulenceStatsPipeline, PlotConfig,
                                         TurbulencePlotter, create_data_loader)

# ── synthetic case definition ──────────────────────────────────────────
NZ, NY, NX = 48, 17, 40
LZ, LX = 2.0 * np.pi, 4.0 * np.pi
MZ, MX = 3, 5                        # mode numbers of the planted waves
KZ = 2.0 * np.pi * MZ / LZ
KX = 2.0 * np.pi * MX / LX
PHI = np.pi / 3.0
CASE = 'synthetic'


# ── writing a synthetic CHAPSim2 visu case ─────────────────────────────
def write_grid(root):
    """Create the case folders and grid files; returns the cell-centre coordinates."""
    data = root / CASE / '1_data'
    data.mkdir(parents=True)
    (root / CASE / '2_visu').mkdir(parents=True)
    nodes = {'x': np.linspace(0.0, LX, NX + 1),
             'y': np.linspace(-1.0, 1.0, NY + 1),
             'z': np.linspace(0.0, LZ, NZ + 1)}
    for axis, values in nodes.items():
        write_bin(data / f'domain1_grid_{axis}.bin', values, header_bytes=4)
    return {axis: 0.5 * (values[:-1] + values[1:]) for axis, values in nodes.items()}


def write_snapshot(root, timestep, u_inst, v_inst=None, w_inst=None):
    """Write one flow file holding qx_ccc (and qy_ccc, qz_ccc if given), each laid out (nz, ny, nx)."""
    attributes = []
    for name, field in (('qx_ccc', u_inst), ('qy_ccc', v_inst), ('qz_ccc', w_inst)):
        if field is None:
            continue
        bin_name = f'domain1_{name}_{timestep}.bin'
        write_bin(root / CASE / '1_data' / bin_name, field)
        attributes.append(attribute_xml(name, bin_name, field.shape))
    write_xdmf(root / CASE / '2_visu' / f'domain1_flow_{timestep}.xdmf', 'flow',
               (NZ + 1, NY + 1, NX + 1), attributes)


def run(root, timestep, **kwargs):
    """Run the computer over the synthetic case and return its results dict."""
    options = dict(direction='z', y_coords_str='-0.5, 0.5')
    options.update(kwargs)
    computer = SpectrumComputer(str(root), **options)
    if not computer.compute_for_case(CASE, timestep):
        return None
    return computer.processed_results[(CASE, timestep)]


def run_quietly(root, timestep, **kwargs):
    """run(), also returning everything it printed."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
        out = run(root, timestep, **kwargs)
    return out, printed.getvalue()


def z_wave_snapshot(root, timestep, amp, mean=5.0):
    """Write a spanwise wave of y-dependent amplitude, uniform in x."""
    z = np.linspace(0.0, LZ, NZ + 1)
    z = 0.5 * (z[:-1] + z[1:])
    write_snapshot(root, timestep, mean + amp[None, :, None] * np.cos(KZ * z)[:, None, None]
                   * np.ones((1, 1, NX)))


# Each test builds its own case in a temporary directory: the cases are tiny,
# and it keeps every test runnable on its own under the bundled runner.

def test_kernel():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((5, 64))
    signal -= signal.mean(axis=-1, keepdims=True)
    k_axis, E = op.compute_1d_spectrum(signal, dx=0.1)
    assert np.allclose(E.sum(axis=-1), (signal ** 2).mean(axis=-1)), 'Parseval: sum(E) == mean(f^2)'
    assert np.isclose(k_axis[1], 2.0 * np.pi / (64 * 0.1)), 'k is in rad/length'


def test_z_spectrum_at_x_stations():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cells = write_grid(root)
        y, x, z = cells['y'], cells['x'], cells['z']

        # A spanwise wave whose amplitude varies with y and x, on a mean that
        # varies with both, so a wrongly-picked station or a mean leaking into
        # the spectrum shows up as a wrong answer.
        amp = 1.0 + 0.5 * np.cos(np.pi * y)[:, None] + 0.1 * x[None, :]
        mean = 10.0 * (1.0 - y ** 2)[:, None] + x[None, :]
        write_snapshot(root, '000100', mean[None] + amp[None] * np.cos(KZ * z)[:, None, None])

        out, _ = run_quietly(root, '000100', x_coords_str='1.0, 9.0')
        assert out is not None, 'computer produced a result'
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        x_idx = [int(np.argmin(np.abs(x - v))) for v in (1.0, 9.0)]
        assert np.allclose(out['y'], y[y_idx]), 'y locations are the nearest cells'
        assert np.allclose(out['x'], x[x_idx]), 'x stations are the nearest cells'
        assert len(out['k']) == NZ // 2 - 1 and np.isclose(out['k'][0], 2.0 * np.pi / LZ), \
            'k = 0 and Nyquist dropped: modes 1 .. nz/2 - 1'
        assert out['E']['u'].shape == (2, 2, NZ // 2 - 1), 'E has shape (n_y, n_x, n_k)'
        target = 0.5 * amp[np.ix_(y_idx, x_idx)] ** 2
        assert np.allclose(out['E']['u'][:, :, MZ - 1], target), 'E(kz) == A^2/2 at every y and station'
        others = np.delete(out['E']['u'], MZ - 1, axis=2)
        assert np.abs(others).max() < 1e-20, 'no energy in any other mode (the mean is not leaked)'


def test_x_station_handling():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cells = write_grid(root)
        y, x, z = cells['y'], cells['x'], cells['z']
        amp = 1.0 + 0.5 * np.cos(np.pi * y)[:, None] + 0.1 * x[None, :]
        mean = 10.0 * (1.0 - y ** 2)[:, None] + x[None, :]
        write_snapshot(root, '000100', mean[None] + amp[None] * np.cos(KZ * z)[:, None, None])
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        x_idx = [int(np.argmin(np.abs(x - 1.0)))]

        mid, printed = run_quietly(root, '000100')
        assert np.allclose(mid['x'], [x[NX // 2]]), 'no x stations -> one mid-domain station'
        assert 'mid-domain station' in printed, 'and that is reported'
        avg, _ = run_quietly(root, '000100', average_x=True)
        assert avg['E']['u'].shape[1] == 1 and len(avg['x']) == 0, \
            'average_x -> one x-averaged spectrum, no station'
        assert np.allclose(avg['E']['u'][:, 0, MZ - 1], 0.5 * (amp[y_idx] ** 2).mean(axis=1)), \
            'x-averaged E(kz) == mean over x of A^2/2'
        both, _ = run_quietly(root, '000100', x_coords_str='1.0', average_x=True)
        assert np.allclose(both['x'], [x[x_idx[0]]]), 'given x stations take precedence over average_x'


def test_x_spectrum():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cells = write_grid(root)
        y, x, z = cells['y'], cells['x'], cells['z']
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]

        # A streamwise wave plus a spanwise one. Along x the spanwise wave is
        # constant, so only B shows up in an x spectrum.
        amp_x = 0.3 + 0.2 * y ** 2
        amp_z = 0.7 - 0.1 * y
        u_x = ((10.0 * (1.0 - y ** 2))[None, :, None]
               + amp_x[None, :, None] * np.cos(KX * x + PHI)[None, None, :]
               + amp_z[None, :, None] * np.cos(KZ * z)[:, None, None])
        write_snapshot(root, '000200', u_x)

        out_x, printed = run_quietly(root, '000200', direction='x', average_x=True)
        assert len(out_x['k']) == NX // 2 - 1 and np.isclose(out_x['k'][0], 2.0 * np.pi / LX), \
            'modes 1 .. nx/2 - 1 along x'
        assert out_x['E']['u'].shape == (2, 1, NX // 2 - 1) and len(out_x['x']) == 0, \
            'averaged over z: one spectrum per y, no station'
        assert np.allclose(out_x['E']['u'][:, 0, MX - 1], 0.5 * amp_x[y_idx] ** 2), 'E(kx) == B^2/2'
        others = np.delete(out_x['E']['u'], MX - 1, axis=2)
        assert np.abs(others).max() < 1e-20, 'no energy in any other kx'
        assert 'WARNING' not in printed, 'no periodicity warning when x is declared homogeneous'
        _, printed = run_quietly(root, '000200', direction='x')
        assert 'WARNING' in printed, "periodicity warning with 'Average x direction' off"
        out_z, _ = run_quietly(root, '000200', x_coords_str='3.0')
        assert np.allclose(out_z['E']['u'][:, 0, MZ - 1], 0.5 * amp_z[y_idx] ** 2), \
            'the same snapshot along z holds only the spanwise wave'


def test_nyquist_mode_of_cell_centred_field():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_grid(root)
        # qx_ccc is the average of the two x-face values, which cancels the
        # x-direction Nyquist mode exactly.
        q = np.random.default_rng(1).standard_normal((NZ, NY, NX))
        u_ccc = 0.5 * (q + np.roll(q, -1, axis=2))
        write_snapshot(root, '000500', u_ccc)
        line = u_ccc[:, 4, :] - u_ccc[:, 4, :].mean(axis=-1, keepdims=True)
        full = op.compute_1d_spectrum(line, dx=LX / NX)[1].mean(axis=0)
        assert full[-1] < 1e-25 * full.max(), "cell-centring zeroes the full spectrum's Nyquist mode"
        out_n, _ = run_quietly(root, '000500', direction='x', average_x=True)
        assert np.isclose(out_n['k'][-1], (NX // 2 - 1) * 2.0 * np.pi / LX), 'the computer leaves it out'
        assert out_n['E']['u'].min() > 1e-6 * out_n['E']['u'].max(), \
            'so the spectrum stays within a few decades on a log axis'


def test_v_and_w_components():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cells = write_grid(root)
        y, z = cells['y'], cells['z']
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]

        # Each component carries a spanwise wave in its own mode, so a
        # component read from the wrong variable lands in the wrong mode.
        amp = {'u': 1.0 + 0.2 * y, 'v': 0.5 - 0.1 * y, 'w': 0.8 + 0.3 * y ** 2}
        mode = {'u': MZ, 'v': MZ + 2, 'w': MZ + 4}
        wave = {c: amp[c][None, :, None] * np.cos(2.0 * np.pi * mode[c] * z / LZ)[:, None, None]
                * np.ones((1, 1, NX)) for c in 'uvw'}
        write_snapshot(root, '000700', 5.0 + wave['u'], wave['v'], wave['w'])

        out, printed = run_quietly(root, '000700', components='w, u, v', x_coords_str='2.0')
        assert list(out['E']) == ['w', 'u', 'v'], 'one spectrum per requested component, in the order given'
        assert printed.splitlines().count('Data loaded for synthetic/domain1_flow_000700.xdmf') == 1, \
            'one load line for the snapshot, not one per component'
        for c in 'uvw':
            assert np.allclose(out['E'][c][:, 0, mode[c] - 1], 0.5 * amp[c][y_idx] ** 2), \
                f'E_{c}{c}(kz) == A^2/2 of its own component'
            others = np.delete(out['E'][c], mode[c] - 1, axis=2)
            assert np.abs(others).max() < 1e-20, f'and nothing in any other mode of E_{c}{c}'

        default, _ = run_quietly(root, '000700', x_coords_str='2.0')
        assert list(default['E']) == ['u'], "components default to 'u'"
        mixed, printed = run_quietly(root, '000700', components='V,u,v,uv', x_coords_str='2.0')
        assert list(mixed['E']) == ['v', 'u'] and "'uv'" in printed, \
            'case-insensitive, duplicates dropped, an unrecognised token reported and ignored'
        blank, printed = run_quietly(root, '000700', components='', x_coords_str='2.0')
        assert list(blank['E']) == ['u'] and "defaulting to 'u'" in printed, "no valid component -> 'u'"

        write_snapshot(root, '000800', 5.0 + wave['u'])  # qx_ccc only
        missing, printed = run_quietly(root, '000800', components='u,w', x_coords_str='2.0')
        assert missing is None and 'Error: No data found for synthetic/domain1_flow_000800.xdmf (qz_ccc missing)' \
            in printed, \
            'a component missing from the file is reported and the snapshot left out'

        computer = SpectrumComputer(str(root), 'z', '-0.5, 0.5', '2.0', components='u,v,w')
        with contextlib.redirect_stdout(io.StringIO()):
            computer.compute_for_case(CASE, '000700')
        config = Config.from_module(types.SimpleNamespace(folder_path=str(root), cases=[CASE],
                                                          timesteps=['000700']))
        ax = TurbulencePlotter(config, PlotConfig(), None).plot_spectrum(computer).axes[0]
        curves = [line for line in ax.get_lines() if line.get_label() != '$k^{-5/3}$']
        assert len(curves) == 6, 'one curve per component and y location'
        assert all(f'{c}{c} ' in line.get_label() for c, line in zip('uuvvww', curves)), \
            'each labelled with its component'
        assert all(f'E_{{{c}{c}}}' in ax.get_ylabel() for c in 'uvw'), 'and the y axis names them all'


def test_averaging_over_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        y = write_grid(root)['y']
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        amp_1, amp_2 = 1.0 + 0.2 * y, 2.0 - 0.3 * y
        z_wave_snapshot(root, '000300', amp_1)
        z_wave_snapshot(root, '000400', amp_2)
        target_avg = 0.25 * (amp_1[y_idx] ** 2 + amp_2[y_idx] ** 2)

        out_avg, printed = run_quietly(root, 'avg', timesteps=['000300', '000400'], x_coords_str='2.0')
        assert out_avg is not None and np.allclose(out_avg['E']['u'][:, 0, MZ - 1], target_avg), \
            "timestep 'avg' averages the configured snapshots"
        assert 'averaged 2 snapshot(s)' in printed, 'and says how many it used'
        out_gap, printed = run_quietly(root, 'avg', timesteps=['000300', '999999', '000400'],
                                       x_coords_str='2.0')
        assert 'Error: No data found for synthetic/domain1_flow_999999.xdmf (no such file)' in printed \
            and out_gap is not None \
            and np.allclose(out_gap['E']['u'][:, 0, MZ - 1], target_avg), \
            'a missing snapshot is reported and left out'
        none, printed = run_quietly(root, 'avg', timesteps=[])
        assert none is None and 'no usable snapshots' in printed, "'avg' with no snapshots fails cleanly"


def test_degraded_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        y = write_grid(root)['y']
        z_wave_snapshot(root, '000100', 1.0 + 0.2 * y)
        write_snapshot(root, '000600', np.ones((1, NY, NX)))
        flat, printed = run_quietly(root, '000600')
        assert flat is None and 'full 3D' in printed, 'a 2D slice is reported, not raised'
        no_y, printed = run_quietly(root, '000100', y_coords_str='')
        assert no_y is None and 'no y-coordinates' in printed, 'no y locations -> skipped'
        missing, _ = run_quietly(root / 'nonexistent', '000100')
        assert missing is None, 'a missing case is reported, not raised'


def test_pipeline_with_average_over_timesteps():
    """Through the pipeline, as the GUI runs it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        y = write_grid(root)['y']
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        amp_1, amp_2 = 1.0 + 0.2 * y, 2.0 - 0.3 * y
        z_wave_snapshot(root, '000300', amp_1)
        z_wave_snapshot(root, '000400', amp_2)
        target_avg = 0.25 * (amp_1[y_idx] ** 2 + amp_2[y_idx] ** 2)

        module = types.SimpleNamespace(
            folder_path=str(root), input_format='visu', cases=[CASE],
            timesteps=['000300', '000400'], average_over_timesteps=True,
            ux_velocity_on=False, save_fig=False, display_fig=False,
            spectrum_on=True, profile_direction='z',
            x_profile_y_coords='-0.5, 0.5', slice_coords='2.0, 8.0',
        )
        config = Config.from_module(module)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            loader = create_data_loader(config)
            loader.load_all()
            pipeline = TurbulenceStatsPipeline(config, loader)
            pipeline.compute_all()
            pipeline.process_all()
            fig = TurbulencePlotter(config, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        spectra = pipeline.spectrum_computer.processed_results
        assert (CASE, 'avg') in spectra, "the loader's 'avg' timestep yields a spectrum"
        assert np.allclose(spectra[(CASE, 'avg')]['E']['u'][:, 0, MZ - 1], target_avg), \
            'of the averaged snapshots'
        lines = fig.axes[0].get_lines() if fig is not None else []
        curves = [line for line in lines if line.get_label() != '$k^{-5/3}$']
        reference = [line for line in lines if line.get_label() == '$k^{-5/3}$']
        assert len(curves) == 4, 'one plotted curve per y location and x station'
        assert len(reference) == 1, 'plus one k^-5/3 reference line, on by default'

        # k^-5/3 reference line
        k_ref, E_ref = reference[0].get_xdata(), reference[0].get_ydata()
        slope = np.polyfit(np.log(k_ref), np.log(E_ref), 1)[0]
        assert np.isclose(slope, -5.0 / 3.0), 'its slope is -5/3'
        k_all = np.concatenate([line.get_xdata() for line in curves])
        assert np.allclose(np.log(k_ref[[0, -1]]), np.log(k_all.min()) + np.array([0.25, 0.75])
                           * np.log(k_all.max() / k_all.min())), 'it spans the middle half of the log k range'
        for line in curves:
            k, E = line.get_xdata(), line.get_ydata()
            inside = (k >= k_ref[0]) & (k <= k_ref[-1])
            assert np.all(E[inside] <= E_ref[0] * (k[inside] / k_ref[0]) ** (-5.0 / 3.0)), \
                'and lies above every curve there'

        config_off = Config.from_module(types.SimpleNamespace(**{**vars(module), 'spectrum_kolmogorov_ref_on': False}))
        fig_off = TurbulencePlotter(config_off, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        labels_off = [line.get_label() for line in fig_off.axes[0].get_lines()]
        assert '$k^{-5/3}$' not in labels_off and len(labels_off) == 4, \
            'spectrum_kolmogorov_ref_on = False leaves only the curves'

        # k^-3 reference line, off by default
        assert '$k^{-3}$' not in [line.get_label() for line in lines], 'no k^-3 line unless asked for'
        config_k3 = Config.from_module(types.SimpleNamespace(**{**vars(module), 'spectrum_k_minus3_ref_on': True}))
        fig_k3 = TurbulencePlotter(config_k3, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        by_label = {line.get_label(): line for line in fig_k3.axes[0].get_lines()}
        assert len(by_label) == 6, 'spectrum_k_minus3_ref_on adds a line alongside the k^-5/3 one'
        k_3, E_3 = by_label['$k^{-3}$'].get_xdata(), by_label['$k^{-3}$'].get_ydata()
        assert np.isclose(np.polyfit(np.log(k_3), np.log(E_3), 1)[0], -3.0), 'its slope is -3'
        assert np.allclose(k_3, k_ref), 'over the same middle half of the log k range'
        for line in curves:
            k, E = line.get_xdata(), line.get_ydata()
            inside = (k >= k_3[0]) & (k <= k_3[-1])
            assert np.all(E[inside] <= E_3[0] * (k[inside] / k_3[0]) ** -3.0), 'above every curve there'
        assert np.allclose(by_label['$k^{-5/3}$'].get_ydata(), E_ref), \
            'and placed from the curves alone: the k^-5/3 line is where it was without it'
        config_k3_only = Config.from_module(types.SimpleNamespace(
            **{**vars(module), 'spectrum_k_minus3_ref_on': True, 'spectrum_kolmogorov_ref_on': False}))
        fig_k3_only = TurbulencePlotter(config_k3_only, PlotConfig(), loader).plot_spectrum(pipeline.spectrum_computer)
        k3_only = [line for line in fig_k3_only.axes[0].get_lines() if line.get_label() == '$k^{-3}$']
        assert len(k3_only) == 1 and np.allclose(k3_only[0].get_ydata(), E_3), \
            'nor does the k^-5/3 line move the k^-3 one'


def test_pipeline_with_instantaneous_data_type():
    """Data type 'inst': the averaged fields are left unread, and the spectrum runs as before."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        y = write_grid(root)['y']
        y_idx = [int(np.argmin(np.abs(y - v))) for v in (-0.5, 0.5)]
        amp_1, amp_2 = 1.0 + 0.2 * y, 2.0 - 0.3 * y
        z_wave_snapshot(root, '000300', amp_1)
        z_wave_snapshot(root, '000400', amp_2)
        # A time-averaged file beside each snapshot, which t_avg reads and inst must not.
        # The snapshots also get a pressure: 'pr' is the averaged name too once its
        # t_avg_ prefix is stripped, so a loader reading them would take it for the mean.
        for timestep in ('000300', '000400'):
            u1 = np.ones((NZ, NY, NX))
            write_bin(root / CASE / '1_data' / f'domain1_pr_{timestep}.bin', u1)
            write_xdmf(root / CASE / '2_visu' / f'domain1_flow_{timestep}.xdmf', 'flow', (NZ + 1, NY + 1, NX + 1),
                       [attribute_xml('qx_ccc', f'domain1_qx_ccc_{timestep}.bin', u1.shape),
                        attribute_xml('pr', f'domain1_pr_{timestep}.bin', u1.shape)])
            bin_name = f'domain1_t_avg_u1_{timestep}.bin'
            write_bin(root / CASE / '1_data' / bin_name, u1)
            write_xdmf(root / CASE / '2_visu' / f'domain1_t_avg_flow_{timestep}.xdmf', 't_avg_flow',
                       (NZ + 1, NY + 1, NX + 1), [attribute_xml('t_avg_u1', bin_name, u1.shape)])

        def run_pipeline(data_type, **switches):
            module = types.SimpleNamespace(
                folder_path=str(root), input_format='visu', xdmf_data_type=data_type, cases=[CASE],
                timesteps=['000300', '000400'], average_over_timesteps=True, average_z_direction=False,
                ux_velocity_on=True, u_prime_sq_on=True, save_fig=False, display_fig=False,
                spectrum_on=True, profile_direction='z',
                x_profile_y_coords='-0.5, 0.5', slice_coords='2.0', **switches,
            )
            config = Config.from_module(module)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
                loader = create_data_loader(config)
                loader.load_all()
                pipeline = TurbulenceStatsPipeline(config, loader)
                pipeline.compute_all()
                pipeline.process_all()
            return loader, pipeline, printed.getvalue()

        loader, _, printed = run_pipeline('t_avg')
        assert any('u1' in fields for fields in loader.data.values()), 't_avg reads the averaged file'
        loading = [line for line in printed.splitlines() if 'Data loaded' in line or 'No data found' in line]
        assert loading == [f'Data loaded for synthetic/domain1_{kind}_{timestep}.xdmf'
                           for kind in ('t_avg_flow', 'flow') for timestep in ('000300', '000400')], \
            'one line per file read: the averaged files, then the snapshots the spectrum reads'

        # The budget asks for every variable, which would otherwise read every instantaneous field.
        loader, pipeline, printed = run_pipeline('inst', re_stress_budget_on=True)
        assert loader.data_types == ['inst'] and not loader.data, \
            "'inst' reads no fields into the loader, even with stress and budget statistics on"
        assert loader.timesteps == ['avg'], 'but still collapses the timesteps for averaging over them'
        loading = [line for line in printed.splitlines() if 'Data loaded' in line or 'No data found' in line]
        assert loading == [f'Data loaded for synthetic/domain1_flow_{timestep}.xdmf'
                           for timestep in ('000300', '000400')], 'only the snapshots are read'
        assert pipeline.statistics == [] and 'skipped: ux_velocity' in printed \
            and pipeline.budget_computer is None, \
            'the averaged statistics are reported as skipped'
        assert np.allclose(pipeline.spectrum_computer.processed_results[(CASE, 'avg')]['E']['u'][:, 0, MZ - 1],
                           0.25 * (amp_1[y_idx] ** 2 + amp_2[y_idx] ** 2)), \
            'and the spectrum is averaged over the snapshots as before'


def test_direction_and_coordinates_come_from_the_profile_settings():
    """Spectra and correlation have no settings of their own for these: a
    combination they cannot use is refused before anything is read."""
    def config(**settings):
        module = types.SimpleNamespace(folder_path='', cases=[CASE], timesteps=['000300'], **settings)
        with contextlib.redirect_stdout(io.StringIO()):
            return Config.from_module(module)

    def refused(**settings):
        try:
            config(**settings)
        except ValueError as exc:
            return str(exc)
        raise AssertionError(f'accepted {settings}')

    both = dict(slice_coords='2.0', x_profile_y_coords='-0.5')
    for switch in ('spectrum_on', 'two_point_corr_on'):
        for direction in ('y', 'x & y'):
            assert "need profile_direction 'z'" in refused(**{switch: True}, profile_direction=direction, **both), \
                f"{switch} with profile_direction '{direction}'"
        for given in ({}, {'slice_coords': '2.0'}, {'x_profile_y_coords': '-0.5'}):
            assert 'you must provide both x and y coordinates for two-point correlation or spectra in z' \
                in refused(**{switch: True}, profile_direction='z', **given), f'{switch} in z with only {given}'
        config(**{switch: True}, profile_direction='z', **both)  # accepted

    assert 'only implemented in z' in refused(two_point_corr_on=True, profile_direction='x', **both)
    assert 'y coordinates for spectra in x' in refused(spectrum_on=True, profile_direction='x',
                                                       slice_coords='2.0')
    config(spectrum_on=True, profile_direction='x', x_profile_y_coords='-0.5')  # x stations not needed
    config(profile_direction='z')  # and nothing is asked of a run with neither on

    computer = TurbulenceStatsPipeline(config(spectrum_on=True, profile_direction='x', **both),
                                       None).spectrum_computer
    assert computer.direction == 'x' and computer.y_coords == [-0.5] and computer.x_coords == [], \
        'an x spectrum takes the y coords and is given no x stations'
