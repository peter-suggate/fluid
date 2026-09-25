# Uniform Geometric runtime resolution

## Objective and constraints

Support a live, global h ↔ 4h simulation switch as the first step toward very
large scenes with mixed coarse and fine cells around the surface. Both modes
advance the complete simulation over the same physical domain. Existing 4³
work maps and coarse velocity sampling do not constitute a coarse simulation.

Switching must not reset the scene, allocate GPU resources, compile pipelines,
or reconstruct the solver. It does require a GPU state-transfer transaction.
Returning to h reconstructs the current coarse solution; it cannot recover
detail discarded during coarse evolution. A transfer touches fine-sized state,
so realtime switching at the largest size remains a measured acceptance gate.

Normal fine-mode performance must not regress significantly. No background
coarse simulation, continuous restriction, additional fine-grid passes, or
per-cell resolution lookup is allowed. Additional total GPU memory should stay
within a low-single-digit percentage; use 3% as the initial engineering budget
and report exceptions rather than silently increasing it. Retaining fine
allocations makes allocation-free return possible, but does not reduce resident
memory while coarse mode is active. Sparse residency is a subsequent project.

## Execution and resource architecture

Prepare two execution configurations at initialization: logical dimensions,
physical spacing, bindings, dispatch plans, geometry, and required specialized
pipelines. Select one configuration at a complete step boundary. Keep a single
scene, world bounds, clock, terrain and rigid-body system. Initially support
3D dimensions divisible by four; reject unsupported configurations explicitly.
Do not pad or rescale physical boundaries to satisfy divisibility.

Retain a small coarse persistent state alongside the existing fine state.
Audit actual lifetimes before classifying fields: conservative liquid volume,
vertex surface geometry, MAC velocities including negative boundary faces,
solid capacities, pressure warm starts and numerical history. Only one state
advances. The inactive fine fields become stale, not a second authority.

Share extension, transport and pressure temporaries through one scratch buffer
sized to the maximum requirement of either configuration, not their sum. Each
configuration retains its own offsets and dimensions. Shared backing has one
owner; borrowed arena views must not destroy it. Transfers need disjoint live
source/destination storage and must never treat stale scratch as persistent
state. Geometry, conditioning buffers, render publication and pipeline caches
also belong in the allocation census; 1/64 cell count is not a total-memory
overhead estimate.

The current constructor in `lib/methods/uniform/webgpu-uniform-reference.ts`
binds dimensions, resources and execution together. Separate resource planning
from allocation first, then make execution configurations consume shared
ownership explicitly. Do not instantiate two independent solver objects.

## Transfer contract

Fine → coarse: sum physical liquid volume in each 4³ group; restrict integrated
MAC flux across the 4² corresponding boundary faces using open-face areas;
construct coarse solid capacities and surface geometry; project the new
velocity. Account for cell-volume units explicitly. Level-set averaging alone
does not preserve liquid mass. Thin walls and disconnected cavities can change
topology under coarsening: define representability checks before exposing the
switch for arbitrary terrain and bodies.

Coarse → fine: distribute physical liquid volume conservatively within fine
open capacities, reconstruct a volume-consistent surface, prolong velocity
with matching aggregate face flux, then project on the fine geometry. Transfer
or invalidate pressure warm starts and numerical histories explicitly. The
restriction/prolongation contract must measure energy and momentum changes;
flux consistency alone does not guarantee their conservation.

Regenerate active sets, tile maps, extension state, pressure coefficients and
convergence histories for the destination. Commit the active configuration and
publication together after transfer. Queue serialization supplies ordering;
no synchronous CPU readback belongs on the switching path. Edits made during
coarse evolution must be reflected in fine geometry before refinement.

## Coarse evolution and presentation

Run all stages at 4h: extension, conservative transport, surface evolution,
forces, solids, pressure and projection. Start with the same timestep to isolate
spatial savings. Review physical versus cell-scaled tolerances and band widths.
A larger stable timestep is a later optimization, not part of the initial
speedup claim. Roughly 64 times fewer cells does not imply 64 times faster frames.

Publish and render the active lattice with its own dimensions and world-space
spacing. Avoid a full fine-grid expansion every coarse frame. Rendering and
fixed dispatch costs must appear separately in benchmarks.

## Implementation sequence and acceptance

1. Extract allocation-free scratch layouts and permit explicitly borrowed
   backing. Test capacity, offsets and ownership. Capture the actual allocation
   census; preserve existing fine allocation sizes and shader addressing.
2. Prepare h/4h execution configurations and coarse persistent fields. Verify
   the memory budget, device limits and zero extra fine-mode dispatches before
   wiring a user-facing toggle.
3. Implement and validate transfers, including geometry and projection, using
   stationary pools, translation, nonuniform velocity, thin features and solids.
4. Connect step-boundary requests and active-resolution publication. Exercise
   repeated toggles and edits without resets, allocations or recompilation.
5. Measure sequential fresh-run ABBA fine baseline/enabled comparisons, coarse
   steady-state stage times, both transition directions, peak GPU allocation,
   and rendering cost at representative and largest supported sizes. Predeclare
   a fine frame-time tolerance (initial target 2%, with repeat runs to resolve
   noise); do not waive a reproducible slowdown. Record absolute transition
   latency against the target interactive frame budget.

Numerical acceptance includes physical volume, capacity bounds, divergence,
hydrostatic stability, surface displacement, energy change, boundary flux and
repeated-toggle drift. Validate coarse evolution against a directly initialized
coarse control as well as transfer-specific oracles. GPU tests run sequentially
under the repository WebGPU lease and never concurrently with the browser.
Run `npm run test:dawn:sparse-cm12` after substantial shared presentation,
terrain-boundary or live-edit changes, per AGENTS.md; do not alter its ceilings.

## Path to mixed resolution

Keep transfer operators and field ownership reusable per block. This milestone
has no spatial coarse/fine seams. A later mixed-resolution solver needs a
composite pressure operator, conservative interface flux authority, graded
transitions and refine/coarsen hysteresis. Global switching validates state
transfer and resource architecture, not those additional interface mechanics.

## Current implementation: switching both ways

`WebGPUUniformReferenceOptions.prepareCoarseSimulation` opts into preparing the
second configuration at startup. `requestCoarseSimulation()` requests activation
at the next complete step boundary; `activateCoarseSimulation()` can also commit
at an already idle boundary and returns false while an adaptive frame is pending.
`requestFineSimulation()` requests the return at a completed coarse-step boundary;
`activateFineSimulation()` performs it immediately when idle. `simulationCellScale`
reports 1 or 4. The runtime adapter accepts `coarseSimulation: "on"` or `"off"`.

The UI exposes preparation and live switching in solver setup and the simulation
panel (`S`). Preparation restarts once; the subsequent Switch to coarse / Switch
to fine button preserves time and the current liquid. Disable preparation is
explicitly labelled as a restart. Unsupported geometry disables preparation.

The root retains two execution configurations using the existing encoder class,
with explicit shared scratch, rigid-system and rigid-exchange ownership. The
coarse configuration is private, has no independent advancing loop, and uses the
root's time at activation. It does not construct another independent solver or
copy scene state during the switch. All allocation and compilation happen during
preparation, including a GPU warmup of restriction and projection over disposable
coarse startup state. Coarse persistent fields stay resident; fine fields remain frozen
after activation. Public field, grid and diagnostic access follows the selected
configuration, while asynchronous callbacks update their own execution's info.

The first supported geometry is a native-storage 3D box without terrain,
interior authored solid edits, rigid bodies or inlet. Each axis must be at least
32 cells and divisible by four, the fine configuration must support the shared
arena, and the coarsened lattice must cover exactly the same physical domain.
Unsupported preparation and subsequent solid/scene edits fail explicitly.
Preparation enforces a 3% additional-memory budget using the existing allocation
census; GPU tests independently count actual buffer and texture payload bytes.

The GPU transaction restricts V in physical-volume units, area-averages each
positive and negative MAC face, retains physical-distance samples at coincident
vertices, applies the existing global volume-constrained surface correction,
rebuilds destination authority/targets, projects velocity and publishes coarse
fields. Transfer projection covers the whole coarse lattice without relying on
old tile certificates. No simulation time or step is consumed. Coarse work maps
are regenerated by the first coarse advance. Pressure and diagnostics receipts
are reserved at startup, including for later runtime pressure-schedule changes.

Implemented files:

- `lib/methods/uniform/uniform-scratch-arena.ts`: allocation-free layouts and
  borrowed-buffer ownership with capacity and usage validation.
- `lib/methods/uniform/uniform-resolution-transfer.ts`: prepared restriction
  pipelines and bindings, with native field-shape validation.
- `lib/methods/uniform/webgpu-uniform-reference.ts`: configuration preparation,
  step-boundary activation, projection and public active-field routing.
- `tests/uniform-runtime-coarse-dawn.test.ts`: evolving fine-to-coarse proof,
  allocation/compilation checks, physical-volume and pressure checks, and an
  independent nonuniform MAC/volume/phi transfer oracle. Set
  `UNIFORM_COARSE_TEST_SIZE=128` for the larger proof (default 64).
- `tools/benchmark-uniform-runtime-coarse-dawn.ts`: fresh-run fine-mode ABBA
  benchmark with the predeclared 2% tolerance.

Fine return copies the current coarse cell-volume fraction into its 64 children,
interpolates MAC velocity along each face normal with constant tangential values,
and trilinearly interpolates vertex phi in physical metres. Thus restriction of
prolonged state recovers each parent volume and boundary face flux. The destination
then reconstructs surface authority and projects velocity. Fine detail discarded
by coarsening cannot be recovered; this is a conservative starting reconstruction.
Both configurations retain preallocated pressure/diagnostics receipts. Repeated
switches reuse all fields, scratch, pipelines and bindings.

Remaining: general geometry representability, energy/momentum and hydrostatic
quality characterization across round trips, live geometry edits,
render-frame benchmarking, and composite
coarse/fine seams. A successful solver test does not certify browser attachment
or realtime transition latency for arbitrary scene sizes.

Validation and measured results are recorded in
`docs/benchmarks/uniform-runtime-coarse-2026-09-25.md`.
