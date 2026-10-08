# NB-FLIP surface authority and large timesteps

The experimental `uniform-narrow-band-flip` method follows Ferstl et al.
Sections 3.1–3.3 for surface authority and sampling. `uniform-volume` retains
its conservative transport, surface-volume constraint and pressure recovery.

## Step contract

1. Advect persistent particles and the interior level set with matching RK4
   characteristics of the accepted extended velocity. Trajectory subdivisions
   target half an h cell of travel; the global pressure timestep is unchanged.
   Both paths report a sticky error above 256 subdivisions rather than hanging
   or silently truncating the trajectory. Particle collision walks remain.
2. Reconstruct the particle surface and union it with the interior eroded by h
   in sampled regions. Uncovered, explicitly coarse regions retain Eulerian
   surface tracking. Only the zero-crossing neighbourhood needs erosion;
   repeatedly adding h to unreinitialized deep distances would hollow the bulk.
3. Reinitialize distance and measure cell occupancy from that surface. There is
   no independent conservative volume transport, dust cleanup, surface-volume
   shift, ghost-phi drain or volume-recovery pressure source in this method.
   The occupancy field remains available to existing pressure, refinement and
   presentation consumers, but it is derived geometry, not a mass target.
4. Use particle velocity at faces within depth r=2h, and the semi-Lagrangian
   velocity deeper inside. The switch is sharp. With quadratic transfer, keep
   particles to R=5h: the 3h inner collar exceeds the kernel's 1.5 sqrt(3) h
   footprint on a cubic lattice. The optional coarse-face transfer also samples
   at the h particle scale. Forces and existing 4h global / fine-band pressure
   follow, with matched pre/post liquid-face extrapolation for FLIP increments.
5. Update particle velocities, then retire deep or excess inner samples and
   seed underfilled inner-band cells from the projected velocity. The outer h
   is protected; no spacing relaxation moves it. Newly admitted refinement
   regions bootstrap their surface samples once. Retired region coverage cannot
   continue eroding a surface after its particles have been removed.
6. Only samples within the outer 2h request swept fine support. The inner
   particle collar does not keep simulation tiles fine by itself. Residency
   prediction uses accepted grid motion, matching supported-particle advection,
   rather than the potentially different FLIP residual velocity. End-of-step
   membership and pre-erosion bulk distances are also guarded by actual
   zero-crossing cells: a bounded local
   distance propagation rejects stale shallow interior phi. The guard measures
   distance to crossing-cell boxes conservatively, with at most a cell diagonal
   of slack; near the surface the redistanced phi still sets the 5h threshold.
   This prevents old surface samples from spreading mandatory refinement into
   the bulk, and prevents their retirement from hollowing old coverage.
   Explicit Full/Requested layouts retain their existing semantics.

The centroid reconstruction is calibrated to preserve a flat eight-samples-per-
cell lattice. A nearest-sample support bound prevents false liquid in empty gaps;
inscribed sample spheres retain sheet corners that a centroid alone excludes.
Weighted offsets are accumulated relative to the queried vertex to avoid
cancellation from large absolute coordinates.

## Diagnostics

`solver.narrowBandFlipInfo` includes:

- `beforeMaxOutside`: maximum positive particle distance to the advected grid
  surface, in h cells, before union.
- `afterMaxOutside`: the same measurement after reconstruction/redistancing.
- `outsideSurface`: number farther than 0.5h outside that surface.
- `unsupported`: number without resolved liquid velocity support.
- `deepInterior`: number deeper than R before end-of-step retirement.

These describe the existing particles before reseeding. No volume shift follows
reconstruction, so the post-reconstruction distance is also the final surface
comparison. Diagnostic maxima are observations, not a proof of timestep accuracy.
In particular, the paper's sufficient gap allowance of h must be checked for
highly deforming flows; RK tracing alone does not make arbitrary timesteps exact.

Subgrid unresolved samples retain the existing gravity fallback and PIC re-entry.
They are counted explicitly as unsupported. This is an extension to the paper,
not evidence that their surrounding liquid is resolved. The optional all-4h
experiment remains resolution-limited: its h-sized droplets cannot be faithfully
represented by 4h surface vertices. Fine coverage is the accuracy reference.

## Verification contract

The GPU tests check that volume transport, cleanup and target-volume correction
are never invoked in NB-FLIP, occupancy equals reconstructed geometry, and the
pressure recovery source is zero even with direct runtime overrides. They also
cover resting planes, live refinement, detached sheets, unsupported droplet
re-entry, free-fall acceleration, particle budgets, transfer kernels and extension
reuse. A manufactured translating drop uses 100 ms pressure steps (CFL_h=2.56)
and compares its endpoint with 50 ms steps, without requiring pressure subcycling.
A manufactured stale-distance case checks deep-particle retirement and return
from Full to a bounded dynamic fine band. The gravity-driven hydrostatic check permits 0.05h surface drift and 0.2%
volume drift over 20 steps; observed drift is about 0.025h / 0.16%. Tightening
pressure residual tolerance did not remove that reconstruction/sampling error.
The zero-gravity lattice equilibrium still retains its strict 1e-5 bounds.

The old experimental mass assertions measured a separate conservative V even
when it disagreed with the displayed surface. They are replaced by exact
occupancy/geometry agreement and a 2% short-trajectory volume-error bound for the
fine-band dam. Resting-pool and static refinement checks retain their strict
bounds. All-4h surface-volume drift is reported rather than hidden by conserving
an unrelated V field. No maintained Uniform Geometric threshold is changed.

## Figure 9 residency and performance

The 900-step, 60 Hz Figure 9 regression finishes at 15 s with 1,025 of 6,500
fully wet tiles fine (15.8%), 465,786 particles and no reseeding budget overflow.
The preceding membership-only guard still hollowed old covered regions and hit
1,048,576 particles. Applying the geometric guard before erosion as well as at
resampling removed that feedback. This regression checks coarse bulk retention;
it is not a claim of exact long-run mass conservation.

A six-frame timestamp window at frames 43–48, using 17 ms global steps, measured
39.1 ms mean wall time and 35.1 ms GPU work. The largest relevant groups were:

| Work | GPU ms/step |
| --- | ---: |
| Particle and level-set advection | 9.0 |
| Particle surface reconstruction | 4.2 |
| Geometric band guards, both rebuilds | 4.2 |
| Particle-to-grid transfer | 2.3 |
| Unused conservative-volume cell traces | 2.4 |

NB-FLIP now omits the last item entirely; momentum writes that scratch field
before its next reader. A subsequent short run measured 38.6 ms mean wall time,
so the measured end-to-end gain was modest, rather than the entire isolated
pass duration. These short windows include timestamp quantization, scheduling
variation and nondeterministic particle accumulation order.

The next experiments, not implemented here, should be:

1. Adopt the paper's trilinear transfer and R=3h/r=2h instead of quadratic
   transfer and R=5h. A thick planar band then has 40% less particle volume;
   thin structures remain fully sampled. This deliberately uses the paper's
   empirical overlap choice rather than a worst-case diagonal kernel envelope.
2. Keep matching RK4 traces but allow travel of about 2h per subdivision in
   smooth velocity fields, with tighter subdivisions when their curvature/error
   estimate requires it. The current half-cell rule spends many velocity samples
   on nearly straight trajectories. Collision walks and the pressure timestep
   remain separate. A varying-velocity large-CFL regression is needed before
   adopting this change; constant translation alone cannot validate it.
3. Restrict exact geometric guards to the active particle/fine band plus its
   dependency and swept-motion halo. Keep reliable geometric membership at the
   surface; preserve the liquid sign in the distant coarse bulk. Every halo must
   cover the chosen timestep and all propagation passes, including new sources.

Pressure tolerances and outer-surface particle retention should remain intact.
