"""Text shown by the toolkit's Help tab.

Keep the bodies plain text at <= 78 columns: they are rendered in a fixed
pitch font, unwrapped, so the reader can rely on the alignment.
"""

__all__ = ['TOPICS', 'VERSION']

from chapsim2_toolkit import __version__ as VERSION


_ABOUT = """\
CHAPSim2 Toolkit
================

A post-processing and pre-processing front end for the CHAPSim2 DNS solver,
built on NumPy, Matplotlib and PyVista.

  Version   {version}
  Licence   BSD 3-Clause
  Source    https://github.com/weiwangstfc/CHAPSim2-toolkit

Copyright (c) 2025-2026, Science and Technology Facilities Council (STFC),
UKRI, and The University of Sheffield.

Written by Alex Old, with Wei Wang (co-author and maintainer), under the
CCP-NTH project.

What it reads
-------------
The output written by current CHAPSim2: XDMF descriptors with binary field
data, the ASCII statistics tables, and the monitor logs. Both Cartesian
(channel, duct) and cylindrical (pipe, annulus) cases are supported, with
and without heat transfer and MHD.

Older runs that kept every visualisation file flat in 2_visu/ still load,
but current output is what the toolkit is built around.

The five working tabs
---------------------
  Mesh Analysis           check a mesh against DNS resolution limits
                          before submitting the job
  Monitoring Points       probe and bulk histories over a run
  Slice Visualisation     2-D contour maps of any field
  3D Visualisation        slices, isosurfaces and volume rendering
  Turbulence Statistics   profiles, Reynolds stresses, budgets,
                          spectra, two-point correlations

Everything here is also available as standalone scripts for HPC applications 
- see README.md. The scripts and the GUI share the same reader
and the same plotting code, so they produce the same numbers.
""".format(version=VERSION)


_GETTING_STARTED = """\
How to use
==========

1. Choose the case, once
------------------------
Use the Case bar at the top of the window. Point it at a CHAPSim2 case
folder - the directory holding 1_data/, 2_visu/, 3_monitor/. Any
subdirectory of the case will do; it is normalised to the case root.

The line underneath summarises what was found:

    pipe_scp_inout_qw | timesteps: 0, 20, 40, 60 | input_chapsim.ini

If it says "no visualisation output", or the timestep you want is missing, 
the path is wrong or data is otherwise unavailable. Choosing a case sets up 
all the tabs at once. Every tab still has its own path field, so one can be 
pointed at a different case if you want to compare.

2. Pick a tab
----------------------
Each tab has its controls on the left and its output on the right, with a
strip of headline numbers above the figure. The tab-by-tab topics in this
list cover the details.

3. Watch the console
--------------------
The narrow strip on the right edge is the console; the dot appears when
there is new output. Click the arrow to open it. Everything the tools
print goes there - what was loaded, what was skipped and why, and the full
traceback if something fails.

Where figures go
----------------
Slice and 3D save on request, through their own Save buttons.

Turbulence Statistics writes every figure it produces. By default they go
to turb_stats_plots/ beside the toolkit; set an Output directory in the
Output section to send them elsewhere or save using the figure canvas.
"""


_LAYOUT = """\
Case folder layout
==================

A CHAPSim2 case directory:

    <case>/
      input_chapsim.ini     the solver input - read by Mesh Analysis
      1_data/               restart and checkpoint binaries, spectra
      2_visu/
        xdmf/               .xdmf descriptors - the index into everything
        data/               field binaries (.bin) and ASCII tables
        mesh/               grid coordinates
      3_monitor/            probe and history logs
      4_check/              mesh and property tables written by the solver

Point the Case bar at <case>. Any of the subdirectories also work.

What the data types mean
------------------------
  inst       one instantaneous snapshot, 3D
  t_avg      averaged in time, 3D
  tsp_avg    averaged in time and over periodic directions, 1D or 2D
  2d_slice   a pre-written plane, can be instantaneous or time averaged

The Slice and 3D tabs report what variables each group holds.

Time-and-space averages come in two shapes, depending on the case:

  one periodic direction   a 2-D plane, tagged with the averaged-out
                           direction (zi1 for a spanwise-periodic channel)
  two periodic directions  a 1-D ASCII profile table in 2_visu/data/

Both are read automatically; you do not need to know which you have.

Pipes and annuli
----------------
These are solved in (x, r, theta), not (x, y, z). The toolkit recovers
those axes and labels them accordingly: a profile axis reads r, and a
cross-section is drawn unrolled as (r, theta) rather than as a disc. The
wall-normal statistics work exactly as they do for a channel.
"""


_MESH = """\
Mesh Analysis
=============

Checks a mesh against DNS resolution requirements before you submit the
job. It rebuilds the wall-normal grid exactly as the solver does, so the
spacings reported are the ones the solver will use.

Using it
--------
Choosing a case loads its input_chapsim.ini automatically. Otherwise an 
empty template is the default and can generate an input file after adjusting 
mesh properties according to requirements.

Reading the headline strip
--------------------------
  Re_tau         friction Reynolds number from an empirical correlation
  dy+ wall       first cell in wall units; DNS wants <= 1
  dy+ centre     coarsest wall-normal spacing
  dx+, dz+       streamwise and spanwise spacing; <= 10 and <= 5
  max growth     largest cell-to-cell ratio; keep under about 1.2
  diff. number   diffusion number for the chosen time step
  total cells    the size of the job you are about to submit

Values are coloured against those limits. Please note these are advisory, a 
full mesh convergence study is advised.

Notes
-----
The case fixes some extents: a channel is y in [-1, 1], a pipe is r in
[0, 1] with Lz = 2*pi. Those boxes are greyed out because the solver
overrides them. For an annulus, "y bottom" is the inner radius, so it
stays editable and must lie between 0 and 1.

The predicted Re_tau comes from a correlation for a developed flow. A
short or still-developing run will measure something lower; that is the
run, not the mesh.
"""


_MONITOR = """\
Monitoring Points
=================

Plots what the solver recorded as it ran: probe points and the bulk and
mass-conservation histories. This is how you check a run is healthy and
has converged to a statistically steady state.

Using it
--------
Choosing a case points this tab at it; 3_monitor is found underneath.
Set how many probe points to read, then Run.

  Sample factor         plot every nth row - a long run holds 10^5+ rows
                        per trace and drawing them all is slow
  Running avg. window   overlay a centred running mean; useful for
                        spotting a drift under the noise
  Auto y-lim            clip the axis when a trace genuinely diverges,
                        so one blow-up does not flatten the rest

Figures
-------
  Pt N              u, v, w, pressure, pressure correction, temperature
                    at one probe
  Bulk Quantities   mass conservation, kinetic energy, pressure, bulk
                    velocity, bulk mass flux, bulk enthalpy and
                    temperature
  Change History    mass residuals, Poisson diagnostics, total mass and
                    its drift, kinetic energy change rate

Columns are read from each file's own header rather than by position, so
a monitor file that gains or loses a column still plots the right
quantity.

What to look for
----------------
Bulk quantities should be steady and mass conservation should be approximately 
machine precision. Depending on the flow modelled, monitoring points should have 
a stable running average.
"""


_SLICE = """\
Slice Visualisation
===================

2-D contour maps of any field, either cut from a 3-D snapshot or read
from a plane the solver wrote.

Steps
-----
1. Choose the case. The tab scans and reports what each data type holds,
   then selects a type and timestep that exist.
2. Check Data type, Physics and Timestep.
3. Click "Load variables". The list fills.
4. Select a variable in the list.
5. Set the Plane and Slice index - the coordinate for that index is shown
   beside it - and click Plot.

Reading a pre-written plane is much faster than loading a full 3-D
field. Set Data type to 2d_slice and put a label such as yi24 in the
Slice label box; the available labels are listed when the case is
scanned. They are read straight out of the timestep's slice bundle.

Options
-------
  Fluctuation    u' = u_inst - u_t_avg, needs a t_avg file for the same
                 timestep; the path is filled in if one exists
  Vorticity      one component, computed from qx_ccc, qy_ccc, qz_ccc
  x crop         restrict the streamwise range, for xy and xz planes
  Colour scale   auto, symmetric about zero, or a range you give.
                 Symmetric is the honest choice for a signed quantity:
                 it keeps zero at the middle of the colour map

The headline strip reports the grid, the coordinate system, which plane
was cut, where, and the min, max and mean of what is drawn.

Variable names
--------------
  qx_ccc, qy_ccc, qz_ccc   instantaneous velocity at cell centres
  gx_ccc, ...              mass flux, for variable-density cases
  pr, phi                  pressure and pressure correction
  u1, u2, u3, uu11, ...    averaged statistics, in tsp_avg and t_avg data
"""


_VISU3D = """\
3D Visualisation
================

Interactive 3-D views through PyVista: slice planes, isosurfaces and
volume rendering. Needs the pyvista package.

Steps
-----
1. Choose the case, then Load variables and select one.
2. Pick a Mode and click Render.
3. Drag to orbit, right-drag to pan, scroll to zoom. "Screenshot" writes
   the current view to a file.

  Stride   read every nth cell. Start at 2 or 4 on a large grid of more than 
  50 million points. 

Statistics
----------
  Fluctuation    u' against the matching t_avg field
  Q-criterion    vortex identification; isosurface mode is the useful one
  Vorticity      one component

Volume rendering refuses to start above a cell-count limit - raise the
stride and try again. It is also the slowest mode by a wide margin;
isosurfaces are usually the better way to look at structure.
"""


_STATS = """\
Turbulence Statistics
=====================

The main post-processing tab: mean profiles, Reynolds stresses, budget
terms, heat transfer statistics, spectra and two-point correlations.

Steps
-----
1. Choose the case. The case folder names, the latest timestep and
   everything the run itself recorded are filled in - see "From the
   case file" below.
2. Check the values and set those the input file does
   not record: e.g the wall heat flux.
3. Tick the statistics you want.
4. Run. Different figure types appear in the selector above the plot.

From the case file
------------------
If an input_chapsim.ini file is in the case directory, the toolkit will 
auto-fill most fields in the post-processing configuration. Otherwise a 
blank template is the default, configurations can be saved and re-loaded 
as 'config.py' files using the 'save' and 'load' buttons. Loading a case 
also checks that the output is the output against the input file loaded.

Fluid properties
----------------
These follow the solver's own model, so the numbers here are the numbers
the run used:

  supercritical water, CO2   interpolated from the NIST table, taken from
                             the case folder when it has one
  liquid metals              sodium, lead, bismuth, LBE, lithium, FLiBe
                             and PbLi-17, from correlations in T

Everything is SI: K, kg/m3, Pa s, W/(m K), J/(kg K), J/kg, 1/K, Pa.

Outside a fluid's valid range the properties come back as NaN and plot as
a gap, rather than as a confident extrapolation.

Input
-----
  Input format   xdmf for binary output, text for the ASCII profile
                 tables
  Data type      tsp_avg for time-and-space averages, t_avg for 3-D
                 time averages that still need averaging here
  Geometry       labels the wall-normal axis and hides controls that do
                 not apply - a pipe has one wall and an axis, so there
                 is no second side to choose

Averaging
---------
Average over a direction only where the flow is periodic. 
For a case with an inlet and an outlet, leave the streamwise directon 
unaveraged and use the profile and slice coordinate boxes to pick 
stations.

The headline strip
------------------
  Re_bulk    Specified bulk reference value
  Re_tau     measured from the wall velocity gradient of the loaded profile
  u_tau      friction velocity
  tau_w      wall shear stress

Normalisation
-------------
The u_tau used for normalisation is measured from the profile, not from
a correlation, so it reflects the state of the run. For a pipe the wall
gradient is taken at the wall, not at the axis.

Output
------
Save figures writes to the Output directory, or to turb_stats_plots/
beside the toolkit if none is set.

Load config.py / Save config.py exchange settings with the standalone
turb_stats.py script, so a case set up here can be re-run without a
display, on a cluster.
"""


_TROUBLE = """\
When something does not work
============================

Plot does nothing
-----------------
Check if a variable is selected. Click one in the Variables list; 
loading them does not select one.

Re_tau looks far too small
--------------------------
Usually the run has not developed. Compare against the Mesh Analysis
prediction, which assumes a developed flow, and check in Monitoring
Points that the kinetic energy has plateaued.

Nothing happens after Run
-------------------------
Open the console. Loading a large 3-D field may take a few minutes, 
and any error appears there with its traceback rather than in a dialog.

A pipe cross-section is drawn as a rectangle
--------------------------------------------
This is the current implementation. The plane is (r, theta) - a radius 
against an angle - so it is drawn unrolled and labelled r and theta.

"""


#: (title, body) in the order the Help tab lists them.
TOPICS = [
    ('About',                 _ABOUT),
    ('How to use',            _GETTING_STARTED),
    ('Case folder layout',    _LAYOUT),
    ('Mesh Analysis',         _MESH),
    ('Monitoring Points',     _MONITOR),
    ('Slice Visualisation',   _SLICE),
    ('3D Visualisation',      _VISU3D),
    ('Turbulence Statistics', _STATS),
    ('Troubleshooting',       _TROUBLE),
]
