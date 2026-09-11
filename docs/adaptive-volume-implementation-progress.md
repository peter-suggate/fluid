# Adaptive volume implementation progress

Implementation follows [the production plan](adaptive-geometric-volume-plan.md).

## Copy and default cutover

The complete adaptive-mass method tree and its sparse-world adapter/device library
were copied literally. The new adaptive-volume method is installed as the default
in the UI, built-in adaptive scene profiles, harnesses, benchmarks and tests.
Original adaptive-mass remains available by explicit selection.

The [copy provenance](../artifacts/adaptive-volume-copy/README.md) records 136
byte-equal files before wiring edits and a runtime dependency audit finding no
executable route from the new method into the original method/adapter pair.
Internal names and filenames were intentionally retained for this first step.

The first complete canonical Dawn run passed 16/17 lanes. All behavior checks
and both frame-time checks passed. The page-budget lane's test reported success
at 29.58 seconds, but process completion exceeded its unchanged 30-second timeout.
See [initial gate receipt](../artifacts/adaptive-volume-copy/dawn-gate-initial.json).
The unchanged page-budget lane passed on focused repeat in 26.856 seconds.
This is still not a passing full gate in one run.

The explicitly selected original and production copy both passed two repeated
eight-step Dawn captures. Density, velocity, pressure, divergence and accepted
topology were bit-identical. The field digest was
`e28fd5754223eafbc1fe5b52412bd7f778bb99c2cf0d8ad83700480ab8c41aab`.
These receipts predate numerical changes and establish copy parity only.
The copy measured 29.491 ms for mini32 and 84.083 ms for mini64, within the
unchanged 40/110 ms limits.
No new unit tests were written or run; the literal copy retains existing test files.
Validation uses Dawn; TypeScript and import/install
checks are supplementary static checks. Existing static failures were observed,
including copied diagnostics with errors matching their original source.

## Numerical implementation

At this intermediate milestone the production method retained CM12 transport. Geometric planes were consumed
by pressure and presentation. The first Dawn run caught a WGSL reserved identifier
(fixed), followed by a reset-waterline error of 0.845 cells against the unchanged
0.4-cell limit. Mini32 measured 56.164 ms against the unchanged 40 ms limit.
Correcting supported-plane blending reduced the reset-waterline error to
0.0000763 cells and passed the focused hydrostatic lane. Separate raw/support
plane caches brought mini32 to 29.491 ms, passing its focused lane.
The combined gate subsequently passed 15/17 lanes, as recorded below.

Physical subface and bounded transport helpers now execute in production, as
recorded below. Independent cut-solid helpers still await integration. The
volume-authority cutover has not yet passed its acceptance checks.

Remaining acceptance work includes pressure/transport compatibility, complete
swept residency, moving-solid displacement, live outside-domain injection, and
large-step accuracy/cost. Transfer and source algorithms are integrated, with
broader production validation still required.

## Volume transport cutover in progress

Small surface discrepancies remain deferred at the user's direction. The M1
combined gate passed 15/17 lanes. No surface limit, physical limit, or timing
ceiling has been relaxed.

Production now uses physical liquid amount V and capacity C. The legacy scalar
storage holds V/full-cell-volume. Native liquid pressure precedes paired
physical-subface transport, with PLIC swept-prism high flux and a bounded FCT
correction. CM12 beta/gamma transport, diffusion, sharpening, density repair,
and excess-density pressure feedback no longer execute in the normal frame.

The GPU chooses synchronized transport microsteps, currently limited to 128 per
outer step. Prepressure sparse activation reserves receiver support. Exact-zero
retirement preserves dilute liquid. Authored shared-face retirement requires an
independent exterior fallback; connected dry authored leaves may consequently
remain resident. Dynamic rows now use the CPU builder's full-face dual weight,
and active fine-only dynamic seams retain their required refinement floor.
A structural witness records face coverage for every accepted cell. Every initial
nonzero volume or source and every nonzero candidate volume must have all six
physical faces before commit. Unused dry backing may remain incomplete until
swept support is reserved; the solver does not activate an entire air catalogue
solely to close unused dry cells.

Topology refinement now owns each parent collectively: PLIC child intersections
for supported open parents, capacity-weighted allocation for unresolved/cut
parents, and deterministic bounded allocation of floating-point remainder.
Coarsening sums child volumes with compensation. Storage error is audited at the
existing 8epsilon capacity bound; it is not claimed to vanish exactly in f32.
Static Q8 capacities now derive from integer open occupancy, giving exact closed
and fully open endpoints rather than subtracting a rounded fraction from one.

### Bounded transport through initially empty cells

Requiring all surrounding air to be divergence-free proved unnecessarily
restrictive. A closed-domain diagnostic found two disconnected air components
with opposite integrated flux imbalance, caused by freezing every
liquid-adjacent row. Secondary air and primary residual-refinement solves have
therefore been removed from execution.

The low-order volume flux now uses a shared receiver factor. Available incoming
volume includes the cell's actual outgoing transfers. Synchronized monotone
factor iterations preserve incompressible full-cell through-flow, while bounding
compressive flow into initially empty/newly wetted cells. Each face still owns
one amount gathered with opposite signs by its endpoints. The original 8epsilon
capacity audit and separate validate/commit dispatches remain; authority is not
clamped and a nonconverged low state fails before publication.

The first two-frame closed-box Dawn probe now passes: no invalid accepted cells,
no boundary outflow, relative volume-balance error 2.07e-8 after the second frame,
and maximum fill 1 + one f32 epsilon. Receipt:
`artifacts/adaptive-volume-copy/dawn-volume-cfl-limited.json`.
This is a small-CFL smoke, not large-step acceptance. Its initial nested fixed
schedule cost about 1.05 seconds GPU transport despite needing only one microstep
and one limiter pass. A flattened 512-packet GPU phase schedule with indirect
singleton/cell/face dispatch gating now executes. A later two-frame above-nozzle
probe measured 83–85 ms total GPU, including 57–58 ms transport. Empty packet
dispatch cost remains substantial; this is not within the mini32 performance gate.
A frozen-envelope single-pass experiment preserved the source probe's numerical
result but increased total GPU time to 93–94 ms (transport 66–68 ms), so its
per-packet dispatch change was reverted. Receipt:
`artifacts/adaptive-volume-copy/dawn-volume-onepass-source.json`.

### Validation and outstanding algorithms

The earlier full volume gate passed 4/17 lanes. The second complete volume gate
passed 7/17 in 346.98 seconds:
`artifacts/adaptive-volume-copy/dawn-volume-second-full-gate.json`.
Later browser inspection found a live Fluid Lab tab rendering and rebuilding
shaders. Consequently timing comparisons and timeout attribution from this
period require an isolated repeat; they must not be treated as clean performance
evidence. The browser tab is now closed for isolated Dawn measurements.
Live liquid injection (now checked with the actual accepted global volume),
outside-tank collapse and mini64 min8 surface pass, alongside failure halt,
mixed topology, clipped transfer and generation storage. Mini32 now reaches
frame 8 but exhausts the 512-packet schedule before all 17 microsteps complete;
mini32 performance, hydrostatics and page budget hit unchanged timeouts. Mini64
still reports missing physical face coverage. Surface failures remain deferred;
other failures included pressure/air compatibility, missing face coverage,
negative static capacity roundoff, rigid overlap, and live fluid insertion.
The latest fixes have not yet rerun that full matrix. The original successful
eight-step corner diagnostic conserved 2048 units to 6.52e-8 relative error with
zero invalid cells/outflow; it still failed the unchanged surface symmetry limit.

Accepted-volume QA sums all accepted sparse cells, including pages outside the
original authored box. The production Dawn CFL probe is
`tools/probe-adaptive-volume-cfl-dawn.ts`. It reports actual method/source identity,
volume/capacity bounds, sources/outflow, completed physical time, and stage costs.
No unit tests were added or run.

Hose injection now freezes capacity-weighted rates before native pressure and
adds their volume only in accepted transport microsteps. Source components must
have a pressure anchor; sealed full components retain pending volume. The old
post-step jet injection and postprojection velocity overwrite are removed.
A two-frame submerged-nozzle Dawn probe passed with requested = emitted = 0.246300876
fine-cell-volume units, pending = 0, no invalid cells and relative balance error
4.66e-9. Receipt: `artifacts/adaptive-volume-copy/dawn-volume-submerged-source.json`.
The same two-frame above-surface source probe passed with pending = 0, no invalid
cells, and relative balance error 3.19e-8. Receipt:
`artifacts/adaptive-volume-copy/dawn-volume-above-source.json`.

The moving-solid GCL baseline is implemented: previous poses
are rasterized on the current topology, old/new capacities are retained, pressure
uses their geometric change and one mean physical face aperture, transport uses
interpolated donor/receiver capacities, and successful completion publishes the
new geometry. Candidate capacities are rasterized before transfer. Previous poses
persist through resident replacement. The first focused rigid Dawn run halted
at frame 11: a partial cell with V=0.413, Cold=1 and Cnew=0.375 was excluded from
pressure, so its required displacement was absent. The targeted receipt is
`artifacts/adaptive-volume-copy/dawn-volume-motion-rigid-qa.json`; the failure
log retains cell membership and old/new capacity. Closing wet cells now enter
pressure regardless of the former half-full cutoff. Moving geometry uses an
implicit donor-fill solve on the same shared physical subfaces, while final
volume still comes from the actual flux gather. The next per-frame Dawn
diagnostic advanced through frame 11 and stopped at frame 12 on a fully closing
cell's 1.86e-9 rounding remainder. Adjusting a single f32 face flux could not
represent an exact terminal sum in all cells. A second shared face component
now carries that bounded rounding remainder; the same two components are
gathered by both endpoints. Per-frame Dawn passed fully closing-cell events
through frame 15 with exact zero in closed cells and no invalid cells. A shared
zero-capacity endpoint aperture mask fixes sampled open faces attached to
permanently closed cells. Moving transport still needs bounded treatment of
newly filling receivers; the complete rigid lane remains failing.
Sparse reservation now includes a swept material box and one face-support
margin. The postprojection envelope and exact structural coverage audit stay
strict; only transfers strictly beyond the physical envelope may be closed as
numerical tails. This gets mini32 past its earlier frame-3 missing-face failure,
but mini64 still has a coverage failure after topology changes.
Cut cells still use the copied scalar aperture/capacity approximation. Independent
cut-solid and split-transport helper files are not evidence that those algorithms
execute. The full geometric-volume cutover is not yet accepted.

### Transport performance work

At the user's request, an Astra agent is optimizing compiled transport topology.
A compact signed cell-to-subface list is built once per outer step and replaces
nine recurring nested incidence/row/subface walks. Before/after Dawn captures
have bit-identical density, velocity, pressure and divergence field hashes,
plus identical accepted-volume receipts. Initial timing samples were affected
by browser GPU contention. The isolated matched two-frame large-step comparison
measured transport 78.12/88.34 ms before and 75.37/72.48 ms after (3.5%/18.0%
reductions; about 11% across those two frames). This is a small diagnostic sample,
not a passing production performance gate. Receipts:
`dawn-volume-adjacency-isolated-before.json` and
`dawn-volume-adjacency-isolated-after.json` under `artifacts/adaptive-volume-copy/`.
The dedicated GPU indirect-argument publisher is also retained. It replaces
513 command-buffer copies/pass transitions with same-pass GPU publications,
retaining all 512 fresh packet argument generations and early-stop decisions.
Against the isolated compiled-adjacency baseline, transport decreased from
75.37/72.48 ms to 49.74/52.23 ms; total GPU time decreased from 99.48/97.58 ms
to 74.71/76.87 ms. Fields and full accepted-volume receipts remained bit-identical.
Receipt: `artifacts/adaptive-volume-copy/dawn-volume-gpu-args-after-large.json`.
These two optimizations leave the volume equations and bounds unchanged. The
fixed dispatch tail was addressed by the subsequent continuation increment.


The next performance increment replaces the production fixed packet schedule
with small GPU submissions and a 16-byte completion receipt. The equations,
128-substep limit, 128 limiter iterations per substep and final volume checks
remain unchanged. The renderer retains the preceding image until the entire
physical frame is available; Dawn callers explicitly await that same boundary.
The transport timestamp spans continuation waits and is therefore elapsed
transport latency, not solely GPU execution occupancy. Matched two-frame Dawn comparisons now pass, with bit-identical final field
hashes and full accepted-volume receipts against the fixed schedule on the same
topology implementation. Large-step transport latency fell from 51.51/50.99 ms
to 27.26/26.94 ms (about 47%); frame wall time fell from 187.15/174.68 ms to
74.38/58.69 ms. The actual limiter work stayed at 46/64 passes, while submitted
packets fell from 512 each frame to 48/64. Small source steps also passed exact
parity: transport 44.50/44.63 ms versus 15.53/31.92 ms, with 8 packets submitted.
Receipts: `dawn-volume-current512-{small,large}.json` and
`dawn-volume-chunk8-{small,large}.json` in `artifacts/adaptive-volume-copy/`.
These are focused diagnostic samples; the unchanged full regression gate remains
the acceptance boundary.

The topology face-coverage guard now certifies cells in parallel and seals its
global decision in a separate dispatch. A structural-area correction also
prevents closed wall apertures from falsely refusing refinement. In three-frame
mini64 diagnostics, parallel certification reduced resolution planning from
182.65/290.65/263.65 ms to 4.92/4.13/3.80 ms. GPU-produced phase-specific indirect
arguments subsequently skip setup and commit kernels during limiter-only
packets: median transport decreased from 287.83 to 258.87 ms, and median total
time from 354.29 to 327.02 ms. Both runs completed the same 12 microsteps and
821 limiter passes, with the same reported bounds and outflow. These short
mini64 probes do not contain full field hashes or constitute a performance pass.
Receipts: `dawn-volume-chunk8-mini64-debug.json`,
`dawn-volume-parallel-guard-mini64.json`, and
`dawn-volume-phase-dispatch-mini64.json` under `artifacts/adaptive-volume-copy/`.

The final two-frame large-step capture after these performance changes retained
identical density, velocity, pressure and divergence hashes and identical full
accepted-volume receipts against both the earlier GPU-argument and chunk-eight
captures. Transport measured 24.12/27.72 ms, total timestamps 55.90/52.04 ms and
frame wall time 68.25/60.44 ms; source fingerprints were unchanged during capture.
Receipt: `dawn-volume-final-performance-large.json`. Separate experiments should
not have their percentage gains added together.

Mini64 still exceeds the unchanged 110 ms ceiling. Its backing generation also
pauses for roughly 15 seconds, separately from transport. The next performance
priorities are the 821 actual limiter passes and repeated gathers, generation
construction/compilation, and status-map overhead from roughly 103 eight-packet
chunks. Increasing chunk size after incomplete submissions is a candidate only;
it has not been implemented or measured.

The complete unchanged canonical gate after this performance increment passed
7/17 lanes in 438.67 seconds, within its 480-second suite budget. Passing lanes:
failure halt, mixed topology, clipped transfer, generation storage, hydrostatic
adaptivity, mini64 min8 surface, and live liquid injection. Symmetry and the
authored region surface checks still fail their existing limits. Page budget,
mini32 correctness, both performance lanes, both far-wall lanes and live rigid
coupling timed out. Outside-tank collapse exited with native `SIGSEGV`, without
a numerical failure receipt. No ceilings or timeouts were changed. The passing
focused transport probes do not supersede this failing acceptance result.
Receipt: `artifacts/adaptive-volume-copy/dawn-volume-performance-full-gate.json`;
full log: `/tmp/fluid-geometric-performance-full-gate.log`.

### Current moving-solid acceptance limit

The moving-solid low-flux branch solves bounded shared face fluxes using cell
dual potentials, addressing competing compulsory displacement and optional
incoming flow. The closing-cell allocator now routes bounded rounding residuals
through a component tree of shared faces, so an entirely closing neighbor need
not absorb the residual. It has an explicit 128-cell component workspace and
does not yet handle every saturated directed closing route. No scalar clamp
is used to hide unresolved volume.

The preceding focused rigid run reached frame 13, generation 2, and failed
`initializeGeometricVolumeCells` at owner 1942: volume 0.0750884 in zero capacity.
That exposed copied overlap averaging spreading a coarse parent's liquid into
children closed at the accepted old pose.

Generation transfer now voxelizes target old-pose capacity before source-owned
allocation. Open parents use PLIC intersections; cut parents currently use
capacity-weighted allocation. Coarsening gathers extensive amounts, including
accepted signed roundoff. Per-source residual and global compensated-sum audits
check conservation separately from capacity. A refused candidate retains the
preceding resident and reports the reason. Velocity uses positive observed
parcel weights, so signed roundoff cancellation cannot amplify its magnitude.
Rigid collision bindings switch only after transfer validation. The subsequent
rigid admission diagnostic exposed another defect: eight samples scaled to a
coarse cell reported capacity 8 while its children summed to 5 at the same pose.
Rigid occupancy now uses the same eight samples per finest lattice voxel at
every rung, with sample-wise union across bodies. Reaction attribution uses
those same locations. The focused rigid test passes all 30 steps, volume
retention, finite dynamics, and buoyancy ordering in 15.56 seconds (log:
`/tmp/fluid-geometric-rigid-nested-capacity.log`). Exact clipped-solid
intersections and joint static/rigid aperture integration remain fidelity work;
the existing product of static and rigid averages is not a general exact union.

### Three-second ladder correctness increment

The static low-flux limiter now certifies the actual gathered state and freezes
the exact factor generation that passed that audit. It no longer changes
already admissible factors before publication. Static propagation has a 1,024
iteration work ceiling; the moving-solid ceiling remains 128, and all volume
bounds remain unchanged. Production continuation accounts for that work budget.
Focused symmetric-2D and first-split-bundle runs both reached 90 complete steps
at three seconds, with relative volume balance error about 1.94e-7. These focused
receipts do not certify the entire ladder or supersede the canonical gate.

Frontier preparation exposed an ITR1 owner collision: a coarse cell can own
several fine seam patches. Positive ownership is now a compiled CSR list, and
the BFA projection consumes every owned row across every accepted side reference.
The default generation byte reservation now accounts for permitted topology
growth instead of three copies of the initial tiny resident. Explicit byte caps
remain authoritative. A separate transfer-receipt bug requested an overlapping
mapped buffer range twice; both numeric views now share one mapping.

The external frontier scene exposed a separate false bridge: nominal page
adjacency connected a clipped cell ending at y=12 to one beginning at y=16.
Composite interfaces now match actual physical face coordinates and clipped
tangential overlap. The exposed faces retain ordinary sparse-air boundaries;
no nonexistent face is fabricated across the gap. Frontier-create then reached
three seconds with volume balance error 3.30e-7 after recorded outflow and zero
nonfinite dynamics across all accepted cells, including outside the authored tank.

`npm run test:dawn:sparse-cm12:ladder-3s` executes all fifteen catalog ladder
scenes with their production profile, complete-frame waits, accepted clocks,
finite-field checks, full accepted-cell finite dynamics, capacity bounds, and volume/source/outflow balance. It
records source fingerprints and separates rejected work from the last accepted
frame. The first frozen-source matrix reached three seconds in fourteen of the
fifteen scenes. Long-dam reached 1.1 seconds with valid accepted volume and finite
dynamics, then exhausted the admission watchdog while compiling its next
topology. Receipt: `artifacts/adaptive-volume-copy/ladder-3s-final-matrix.json`.
The device cache had retained every completed generation's pipeline family;
it now retains only two recent completed families per category, while preserving
pending compilation and live resident ownership. Validation of that follow-up
and the complete ladder remains in progress.

The bounded-cache diagnostic advanced to 1.2 seconds, but later compilation
again slowed severely. `vmmap` reported 2,936 thread stacks and `ps -M` nearly
3,000 live threads, approximately one per pipeline request. This matches the
[Dawn worker-pool counter defect](https://github.com/google/dawn/commit/f29668745ec56f89106877887928e63c2004e433),
fixed upstream on June 8, after the installed `webgpu` 0.4.0 release. The
`webgpu` dependency is now 0.6.0 (Dawn revision
`55c03af2b97acd886d6d65bd345c214c4408ed40`, which contains that fix). The retry
held at 17 process threads through hundreds of compilations. Asynchronous
compilation and physics limits are unchanged.

That release also exposes a dedicated-staging upload defect: writes larger
than 4 MiB whose length is 4 modulo 8 abort when Dawn compares an 8-byte-rounded
mapped span with the source span. The captured trigger was the 5,065,388-byte
resident activity buffer. Resident initialization now uploads exact byte ranges
in chunks of at most 4 MiB, including state seeding and topology arenas. It adds
no padding and changes no buffer contents. The shared helper also covers
generation-transfer metadata and other geometric bulk uploads; the second
captured abort was the 9,294,460-byte overlap metadata buffer.

The full isolated ladder repeat now passes **15/15 scenes**, each completing
90 accepted paper steps to three seconds. Every scene and the full matrix have
unchanged source fingerprints (`6c0eaab0f8af0e47c3b35ce5ae6c21fab3072379804b54522cc15ac6c39624f2`).
All accepted cells pass the unchanged capacity and finite-dynamics checks.
Long-dam finishes with 108,815 accepted cells, zero invalid/nonfinite cells,
zero recorded boundary outflow, and relative volume-balance error 9.22e-6.
The other fourteen scenes have at most 3.30e-7 relative balance error.
Receipt: `artifacts/adaptive-volume-copy/ladder-3s-validated-matrix.json`.

This is a correctness result, not a performance acceptance: long-dam required
1,171.5 seconds wall time and the full matrix 1,546.6 seconds. Frequent topology
families still compile about 230 pipelines each. The unchanged canonical
17-lane gate was paused at the user's request during the terrain lane after
the ladder passed. There is no completed canonical result for this increment;
the partial log is `/tmp/fluid-geometric-correctness-full-gate.log`. The user
reported lumpy surfaces and dissipative flow and proposed a matched comparison
against another method before further implementation. Passing the ladder's
volume/finite-dynamics checks does not establish surface or energy fidelity.

### Dam reference and face-advection dissipation (12 September 2026)

Added the default Sparse Geometric Ritter flume to the UI and a standalone
Dawn measurement command, `npm run probe:dawn:adaptive-volume:dam-front`.
The probe records accepted-volume columns, matching bin-averaged Ritter depth
and discharge, threshold arrival brackets, energy estimates, raw pressure
support, volume/bounds/finite-dynamics checks and source fingerprints. It does
not label completion as analytic-fidelity acceptance. Reference assumptions,
coordinates, receipts and measured values are in `docs/geometric-dam-reference.md`.

Identified and fixed an unconditional face-to-cell-to-face averaging filter.
A manufactured near-zero-step vortex retained only 72.86% of energy before the
change, exactly matching the predicted filter; source staggered sampling retains
99.9999996%. The production predictor now advects the source face field while
using extended collocated velocity for trajectories. Physical overlapping face
patches select mixed-resolution samples; unsupported and uncertified exterior
samples and moving cut faces retain the existing extension fallback. Uniform
coarse and mixed-resolution Dawn identity controls pass. UI pipeline copy now
describes the actual field and trajectory sources.

The matched dam improves modestly at the unchanged 1/30 s production step:
5%-depth arrival at the 0.4 m sensor is 0.4993 s, formerly 0.5211 s; the matching
Ritter observation is 0.3037 s. The 2 s run observes wall impact and retains
bounded volume. A large-step smoke also passes two 0.2 s steps with actual
Courant numbers up to 49.63 and volume balance error below 6.74e−7.

The remaining large-step error is material: halving the timestep moves the
0.7 s front from 0.5913 m to 0.8137 m (Ritter 0.9216 m). Fixed-finest controls
show the same strong temporal sensitivity, so coarsening is not the dominant
cause. Current volume microsteps reuse the outer frame's momentum, pressure
membership and coefficients while new cells become wet. Bounds and geometric
CFL subcycling are not a coupled momentum/pressure time integrator. Coherent
physical microepochs, with correct time, banks, geometry and source accounting,
remain required to address this limitation without merely reducing the UI step.

A trial that seeded velocity from all positive accepted volume was withdrawn:
it fixed isolated thin-sheet translation but promoted unprojected gravity-driven
air velocities and severely regressed the dam. The existing sub-isovalue-motion
regression remains known red; only its Dawn provider lifetime and explanation
were updated. No pressure threshold, physical bound, canonical lane, timing
ceiling or test tolerance was changed. The mandatory full Dawn gate completed
in 454.3 s: 8/17 lanes passed, seven exceeded their unchanged timeouts, and
symmetric expansion and min8 region surface publication failed correctness
thresholds. The earlier paused gate already showed those two correctness
failures, but supplied no complete matched timing baseline. Full receipt:
`artifacts/analytic-motion/staggered-dawn-regression.json`.

### Minimal all-fine analytic isolation (12 September 2026)

Added `geometric-uniform-translation` to the UI and
`npm run probe:dawn:adaptive-volume:translation`. The scene contains two all-fine
8³ bricks and a 6 m/s plug, without gravity, viscosity or surface tension. Exact
cell-integrated translation is the reference. Both CFL 2 and fractional CFL 5/3
controls pass with essentially unchanged velocity and kinetic energy through
all captured production stages. This supplies a small positive correctness
baseline before introducing more complicated flows.

A detached-box variant preserves momentum, energy, centroid and column volume
but fails shape: its L1 error oscillates between roundoff at half-cell positions
and 3.125% at whole-cell positions. The error does not accumulate over the
observed 12 smaller steps. It is a separate geometric reconstruction symptom,
not evidence of energy dissipation. Details and receipts:
`docs/geometric-uniform-translation.md`. No new production physics change was
made for these isolation runs.

The four-brick, all-fine steady Euler vortex then reproduces real damping with
no moving free surface. Over the same 1/60 s interval, total energy loss is
8.46%, 5.73%, and 6.12% at dt 1/60, 1/120, and 1/240. Pressure-projection losses
halve with dt, while accumulated preparation/interpolation losses increase;
smaller steps alone are not a monotonic cure. Pressure solves converge. A
near-zero-step control retains 99.99996% energy. This narrows the generic loss
to temporal advection/projection splitting and spatial velocity interpolation;
it prevents treating the dam's frozen wetting/pressure coupling as a complete
explanation. Details: `docs/geometric-steady-vortex.md`. Production fixes to
these two numerical mechanisms remain future work; this increment establishes
the small analytic baselines and stage attribution first.

### First method concern: interface-normal reconstruction

Working the cases in order confirmed the detached-box error at individual GPU
planes and subface fluxes. The LS volume-fraction gradient tilts the half-filled
corner normal to `(±2,-1,0)/√5`. A half-cell sweep then transports 1/16 through
the front face instead of zero and 7/16 through the rear instead of 1/2; FCT
accepts these bounds-valid fluxes. Those exact captured values account for the
entire 3.125% L1 error. An independent straight-plane stencil calculation also
shows a 14.62° normal error, so the concern extends beyond corners.

Stopped at this method concern as requested: the interface-normal estimator
needs a reconstruction design review. Sparse bricks and conservative shared
flux accounting are not implicated by this result. No production physics or
acceptance tolerance changed. See `docs/geometric-corner-reconstruction.md`
and `artifacts/analytic-motion/corner-transport-detail.json`.

### First reconstruction and interpolation corrections

The stopping conclusion above is now superseded for certified 2D uniform
extrusions. A volume-consistent 3×3 height candidate reconstructs eight exact
axis, oblique, signed and offset plane cases to at most 1.71e−6 degrees normal
error. The detached-box translation now returns to roundoff-level shape error:
4.58e−16 at the accepted clock after two half-cell steps, with unchanged
velocity and kinetic energy. This establishes the specific corner correction
without claiming that the existing general 3D fallback is fixed. Receipts:
`artifacts/analytic-motion/plane-fit-2d.json`,
`corner-fit-detail.json`, and `corner-fit-cfl2.json`.

Certified complete, open, uniform interior MAC stencils now use
component-bounded tensor cubic velocity sampling; other stencils keep the
linear path. Across the fixed 1/60 s vortex interval, total energy loss falls
from 8.4582% to 7.9765% at dt 1/60, from 5.7334% to 4.7825% at dt 1/120, and
from 6.1249% to 4.3674% at dt 1/240. Projection remains the dominant 1/60 s
loss and still requires a time-coupled correction. See
`docs/geometric-steady-vortex.md` and the
`artifacts/analytic-motion/vortex-cubic-*.json` receipts.
