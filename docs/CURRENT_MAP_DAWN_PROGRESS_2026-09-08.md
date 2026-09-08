# Current spatial field: Dawn production investigation

## Checkpoint status

The user stopped this approach as a production dead end and requested a commit
before restoring the pre-8-September coarse-first implementation. This is an
unfinished experimental checkpoint, not an accepted solver or surface fix.
No rollback is included here. The pre-day reference is commit `57b6ae39`.

The latest point-only seed-cut optimization is preserved as work in progress.
It omits unused map Jacobians, but Metal produces small non-bitwise differences:
native density differs in 13 cells at step 4 (maximum 3.725e-9) and 22 cells at
step 6 (maximum 7.451e-9); velocity remains bitwise identical. Independent
point-evaluation validation was still in progress when work stopped. The
impact quadrature depth probe is also unfinished. The canonical regression
gate has no passing receipt; prior runs exhausted their wall-clock budgets.

The quarter falling-ball profile now selects `current-map` in production. Its
paper timestep remains exactly 1/30 s. The objective is not yet achieved: the
actual simulation must preserve the detached sphere and a flat pool until near
impact, with density and displayed geometry supplied by the same field.

The current authority is `rho(x) = q_seed(X(x)) det(DX(x))`. Certified C2
increment maps compose without repeatedly resampling the cumulative map. Native
density and momentum restrict quadrature of this field. The shipping implicit
surface samples it directly. No prescribed trajectory or fitted geometry is
used. Increment storage currently has an explicit 32-step capacity limit.

## Reproduced production defects

The actual authored quarter scene uses 32 × 24 × 32 finest supports, 0.05 m cell
width, sphere radius 0.25 m and centre y=0.9125 m, pool height 0.4 m, gravity
−9.80665 m/s². The correct transport-before-force reference after n steps is
`y = 0.9125 - 9.80665 (1/30)^2 n(n-1)/2`.

* V9 showed missing accepted map increments despite increasing host step counts.
  Its steps 2, 4 and 6 native density files were byte-identical. Accepted map
  counts at steps 0/2/4/6/7 were 0/2/3/4/4.
* V10–V16 reproduced the missing publication in Dawn. Per-stage copies showed
  zero shared simulation faults. Explicit compute-pass boundaries alone did not
  fix it. Retrying the same publication pipelines and bindings in a separate
  command buffer succeeded. A standalone 912 MB arena publication oracle also
  passed, excluding basic archive address/size arithmetic.
* Actual GPU failure-checkpoint copies between heavy map phases improved
  publication but did not establish complete execution: V17 published through
  step 6 then skipped step 7.
* V17 step 6 contained stale fine integrals. All 64 worst independently audited
  supports retained their step-4 density and momentum words exactly. Example
  support (14,20,14) stored density 0.98291087 versus independently integrated
  accepted density about 0.004326. Native restriction matched those stale fine
  values; it was not the source of the discrepancy. The accepted spatial field
  still integrated to approximately the original total mass.
* V18 added per-kernel GPU completion counts. On step 3, integration completed
  only 17,280 of 24,576 supports. Every other required kernel completed its full
  count. The new admission check rejected the frame with
  `CURRENT_FIELD_INCOMPLETE_DISPATCH`, operands `[124,12,17280,24576]`.
  This is evidence of incomplete heavy integration execution, not a proven
  diagnosis of a particular driver fault. Private array bounds were audited.

V19 distributes each support's existing adaptive quadrature across a GPU
workgroup. Gauss nodes, saturation-root cuts, tolerances, adaptive depth and
fallback axes are retained. Both production coarse and fully fine arms now
complete every required dispatch through step 7. Native means match the accepted
fine measures, with total liquid amount approximately 1.0897775 m³; independent
audits no longer find stale fine values. Publication checks completion before
native writes and verifies the commit receipt before presentation.

Step 8 rejects unresolved impact quadrature, not incomplete dispatch. Support
11791, at (15,8,15), reports y-momentum error 0.22869068 against tolerance
0.00735775. Independent increasingly resolved quadrature confirms difficult
convergence there. Production depth remains 3 and tolerances are unchanged.

## Remaining shape defect

V19 coarse and fine independently sampled half-density surfaces agree on the
remaining preimpact defect. At step 6 the sphere's maximum radial error is
3.585 mm and the pool depression is 0.998 mm, despite nominal sphere clearance
of 99.056 mm. At step 7 radial errors span −11.886 to +7.372 mm and pool
depression reaches approximately 3.956 mm. This is a field transport defect,
not mesh tessellation or adaptive restriction.

The actual frozen VEX at step 6 shows why. The old sphere velocity is the correct
−1.634442 m/s, while the pool is effectively stationary. The dry gap blends
between these velocities. Frozen Eulerian RK2 backtraces the expected new
sphere bottom to y=0.549217567 m instead of its old position 0.553537222 m: a
4.320 mm shortfall. Cubic quasi-interpolation adds another 0.473 mm error.
At the pool surface, RK2 moves the departure only 0.332 µm, but the spline
overshoots by −0.496 mm and its local determinant is 0.940494. Thus both
future-air velocity sampling and spline overshoot require correction. Native
projected sphere velocities remain essentially uniform freefall. No replacement
transport operator has yet been accepted on this evidence alone.

## Performance pass

The production quarter scene is currently too slow. Dawn profiling records
unsplit frame wall times of 554, 2445 and 4069 ms for frames 1–3. A separate
instrumented run, splitting dispatches into timestamped passes, attributes
253, 1644 and 2967 ms to cooperative fine-measure integration, reaching 94% of
frame-3 GPU time. Orientation certification costs roughly 94 ms, all 64
velocity extension sweeps together 20–30 ms, and native restriction about
0.2 ms. Instrumented execution is 15–24% faster than the original grouping,
so these GPU figures must not be presented as exact unsplit timings.

The first production optimization precomputes the identical fine-support range
bounds in a separate parallel dispatch rather than making 63 workgroup lanes
wait while lane 0 computes them. It retains the field, bounds, quadrature and
completion checks. Its instrumented A/B completes through frame 6 with exactly
matching native density and velocity at frames 2, 4 and 6. Instrumented wall
times for frames 1–3 fall from 475/1879/3207 ms to 269/559/873 ms; frame 6 takes
1946 ms, including roughly 500 ms for range bounds and 1243 ms for quadrature.
The unsplit production A/B also exactly matches density and velocity through
frame 6. Its frame times are 597/1410/1974/2568/3224/4021 ms, so actual frame-2
and frame-3 improvements are 42% and 51%, respectively. Instrumentation changes
scheduling substantially; its larger speedup must not be claimed for production.
An explicit compute-pass boundary between range production and quadrature also
passes exact native density and velocity comparisons at frames 2/4/6. Shipping
frame times become 350/678/1043/1391/2058/2207 ms. Thus frame 3 improves from
4069 to 1043 ms (74%) without changing the calculation. Both changes are in
production. These are individual sequential captures, not statistical timing
baselines. Performance remains insufficient for interaction.
The profiling tool is `tools/profile-current-map-dawn.ts`; baseline receipts are
under `artifacts/current-map/performance/quarter-{split,unsplit}.json`.

## Reproduction and independent checks

Run only one Dawn process at a time, with the GPU-active browser closed:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts \
  --arm=coarse --regions=authored --steps=0,2,4,6,7,8 \
  --transport=current-map --out=artifacts/current-map/quarter/next/coarse
```

Use `--arm=fine` for the fully fine comparison. `--regions=handoff` explicitly
selects the older authored local slab diagnostic; it is not the user's default
production configuration. The capture checks one map publication per encoded
step, records per-stage completion and failure receipts, and saves actual
shipping GPU triangles plus native fields. Map checkpoints are packed and
losslessly compressed, retaining all accepted increments and both working maps
and measure banks. Metadata relocates archive/footer offsets explicitly.

Independent CPU tools:

* `tools/analyze-current-map-capture.py`: native restriction and mesh diagnostics.
* `tools/analyze-current-map-ray-field.py`: accepted-density half-level roots
  against the discrete-gravity reference, independent of mesh tessellation.
* `tools/audit-current-map-measure.py`: independent tensor quadrature, exact
  previous-bank word comparison and native restriction audit.

V17's apparently small visible errors did not constitute success: at step 6 the
sphere reference radial error reached 3.49 mm and the pool moved about 1 mm while
its nominal sphere clearance was still 99 mm. Its stale fine integrals explain
why visual inspection alone was insufficient.

The canonical `npm run test:dawn:sparse-cm12` remains mandatory. Earlier runs
exhausted wall-clock budgets; those are not numerical passes and no thresholds
have been relaxed. Browser use is reserved for a significant Dawn milestone.
