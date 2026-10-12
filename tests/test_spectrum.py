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


def write_snapshot(root, timestep, u_inst):
    """Write one instantaneous qx_ccc field, laid out (nz, ny, nx), as a flow file."""
    bin_name = f'domain1_qx_ccc_{timestep}.bin'
    write_bin(root / CASE / '1_data' / bin_name, u_inst)
    write_xdmf(root / CASE / '2_visu' / f'domain1_flow_{timestep}.xdmf', 'flow',
               (NZ + 1, NY + 1, NX + 1), [attribute_xml('qx_ccc', bin_name, u_inst.shape)])


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
        assert out['E'].shape == (2, 2, NZ // 2 - 1), 'E has shape (n_y, n_x, n_k)'
        target = 0.5 * amp[np.ix_(y_idx, x_idx)] ** 2
        assert np.allclose(out['E'][:, :, MZ - 1], target), 'E(kz) == A^2/2 at every y and station'
        others = np.delete(out['E'], MZ - 1, axis=2)
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
        assert avg['E'].shape[1] == 1 and len(avg['x']) == 0, \
            'average_x -> one x-averaged spectrum, no station'
        assert np.allclose(avg['E'][:, 0, MZ - 1], 0.5 * (amp[y_idx] ** 2).mean(axis=1)), \
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
        assert out_x['E'].shape == (2, 1, NX // 2 - 1) and len(out_x['x']) == 0, \
            'averaged over z: one spectrum per y, no station'
        assert np.allclose(out_x['E'][:, 0, MX - 1], 0.5 * amp_x[y_idx] ** 2), 'E(kx) == B^2/2'
        others = np.delete(out_x['E'], MX - 1, axis=2)
        assert np.abs(others).max() < 1e-20, 'no energy in any other kx'
        assert 'WARNING' not in printed, 'no periodicity warning when x is declared homogeneous'
        _, printed = run_quietly(root, '000200', direction='x')
        assert 'WARNING' in printed, "periodicity warning with 'Average x direction' off"
        out_z, _ = run_quietly(root, '000200', x_coords_str='3.0')
        assert np.allclose(out_z['E'][:, 0, MZ - 1], 0.5 * amp_z[y_idx] ** 2), \
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
        assert out_n['E'].min() > 1e-6 * out_n['E'].max(), \
            'so the spectrum stays within a few decades on a log axis'


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
        assert out_avg is not None and np.allclose(out_avg['E'][:, 0, MZ - 1], target_avg), \
            "timestep 'avg' averages the configured snapshots"
        assert 'averaged 2 snapshot(s)' in printed, 'and says how many it used'
        out_gap, printed = run_quietly(root, 'avg', timesteps=['000300', '999999', '000400'],
                                       x_coords_str='2.0')
        assert 'Missing instantaneous flow file' in printed and out_gap is not None \
            and np.allclose(out_gap['E'][:, 0, MZ - 1], target_avg), \
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
            spectrum_on=True, spectrum_direction='z',
            spectrum_y_coords='-0.5, 0.5', spectrum_x_coords='2.0, 8.0',
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
        assert np.allclose(spectra[(CASE, 'avg')]['E'][:, 0, MZ - 1], target_avg), \
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
