# Adaptive geometric volume: incremental production implementation plan

Status: implementation in progress, 2026-09-11. See
[implementation progress](adaptive-volume-implementation-progress.md) for measured
results and remaining work.
Updated with the user's production, default-selection, Dawn testing and agent
requirements.

The first deliverable is a literal copy of adaptive-mass, immediately installed
as the default production fluid method. Every subsequent increment changes that
production implementation and leaves a functioning simulation the user can inspect.

Proposed identity: adaptive-volume. UI label: **Sparse Geometric**. Initially
describe it accurately as running the copied CM12 algorithm.

## Working agreement

- Literally copy the source files, not a wrapper, subclass, re-export, preset
  or alias of the original implementation.
- Verify byte equality before minimum identity/wiring edits. Preserve internal
  names and filenames initially where safely scoped.
- **Immediate production cutover for every implemented increment.** No deferred
  promotion, preview-only delivery, shadow solver, experimental flag, historical
  stage selector or separate approval/promotion step.
- **New method is the default everywhere:** UI, scenes/presets, workers,
  harnesses, benchmarks, tests and ordinary validation commands.
- Original adaptive-mass may remain explicitly selectable for comparison.
  It is never an implicit default, automatic fallback or substitute test target.
- Finish and report the working production copy before numerical changes.
- Preserve supported scenes, adaptive bricks, large outer-step operation and
  live editing throughout. Defer sophistication or optimization rather than
  shipping a restricted replacement as the default.
- Deliver small coherent increments with a directly runnable scene, visual
  evidence where relevant, correctness receipts and measured cost.
- Reset/reinitialize explicitly across a state-representation change. Never
  silently reinterpret running CM12 density as bounded geometric volume.
- **Avoid unit tests. Dawn is the validation path about 99% of the time.**
  Prefer actual production solver/scene integration and regression runs.
  Browser use is limited to UI wiring and visual inspection.

## Implementation agents

Coupled numerical implementation uses **Astra Medium sub-agents**: model
gpt-6-astra, reasoning effort medium. Routine work uses **GPT-5.6 Sol High**
sub-agents, following the updated user preference. Give each agent the plan,
concrete ownership, shared contracts and acceptance conditions. The coordinator owns integration, production wiring
and end-to-end verification.

Assign independent bounded work with disjoint file ownership. Milestone 0 can
use source/runtime copying, UI/default wiring, and Dawn/harness-default migration
assignments. Agree identities and paths first. Complete copying before another
agent edits the destination files; do not overwrite concurrent work.

Serialize dependent edits and GPU runs. One GPU owner coordinates all Dawn work;
other agents can work on CPU-side code concurrently. Do not assign unit-test
development or speculative competing numerical implementations. For a coupled
change, fix state/flux/generation contracts before delegating components.

## Copy scope and wiring

The entry is lib/methods/adaptive-mass/method.ts. Installation and default
selection come from lib/methods/index.ts. The solver imports the external CM12
sparse-world adapter, which constructs the original resident and uses a device
library. Copying the method folder alone would still execute original code.

| Source | Destination |
| --- | --- |
| Entire lib/methods/adaptive-mass/ tree | lib/methods/adaptive-volume/ |
| lib/sparse-world/internal/cm12-adapter.ts | lib/sparse-world/internal/adaptive-volume-adapter.ts |
| lib/sparse-world/internal/cm12-device-library.ts | lib/sparse-world/internal/adaptive-volume-device-library.ts |

Rewire the copied solver, adapter and device library to each other and the
copied resident. Audit transitive dependencies for any remaining executable
route into adaptive-mass. Keep ordinary core utilities, public contracts,
renderer, scene model and rigid-body services shared. Record shared numerical
constants and copy them locally before changing their behavior.

Audit:

- Method/harness/resource/feature identities, failure attribution, pipeline
  graphs, diagnostics, and caches or registrations that could collide.
- FeatureSlot.tsx, copied adaptivity controls, gravity UI, voxel/liquid edit
  checks and mirrored edits in the simulation controller.
- UI startup/reset, scene selection, URL hydration, settings persistence,
  worker initialization and comparison-pane defaults.
- Built-in profiles and test/benchmark fixtures selecting adaptive-mass:
  migrate to adaptive-volume while preserving parameter values.
- Existing explicit user-selected historical methods remain explicit; missing
  or default selections resolve to the new method.
- Harness factories, Dawn defaults, direct solver/resident imports and benchmark
  factories. Changing only the registry default does not retarget direct imports.
- Module-boundary/install checks: add the new method zone and required
  composition/adapter edges while retaining sibling-method isolation.

## Milestone 0 — literal copy becomes production default

1. Record revision, working-tree status, inventory and hashes. Copy byte for
   byte and verify equality before editing.
2. Apply only identity, wiring and default changes; record departures from the copy.
3. Make normal application and test execution construct the copied resident
   and shader generator. Capture evidence of the actual implementation selected.
4. Verify play/pause, reset, scenes, refinement, gravity, field views, comparison
   panes and live liquid/solid edits. Use Dawn for solver behavior and brief
   browser inspection for UI behavior.
5. Compare initial state and short deterministic evolution against explicitly
   selected original CM12 at identical settings: fields, topology, pressure
   and publication. Expect bit equality where deterministic; establish original
   repeatability before assigning tolerances to nondeterministic fields.
6. Run the canonical production Dawn gate with the new default. Retain coverage,
   behavioral limits and timing ceilings.

**Visible result:** opening the application or running ordinary tests uses
Sparse Geometric, with copied CM12 dynamics and existing capabilities intact.

**Exit:** copy provenance, isolated execution, default-routing evidence, parity,
Dawn receipts and comparable timing. Deliver before numerical work.

## Milestone 1 — production interface geometry

Introduce geometry services through real production consumers, in small
increments. Build consistent narrow-band signed distance/reconstruction from the
current CM12 implicit surface, then switch presentation and pressure-interface
geometry over in the smallest coherent steps. Support mixed-resolution boundaries
and solids from their first production use.

Liquid amount still uses CM12 here. Do not call this geometric conservation or
obtain a supposedly bounded liquid volume by clamping excess density. The
geometry serves actual rendering/pressure; it is not an optional preview or
independent fluid simulation.

Use production Dawn still-pool, moving-interface and terrain runs to measure
waterline, normals, pressure stability and stage cost after each consumer change.
Where consumers cannot change independently without inconsistency, integrate
the minimum coupled set together.

**Visible result:** default fluid uses new geometry for real surface/pressure
behavior while remaining a functioning CM12-transport fluid.

**Exit:** stable hydrostatics, supported scenes, consistent physical/rendered
waterline, no seam regression and measured production cost.

## Milestone 2 — coherent geometric-volume authority cutover

Implement the smallest complete geometric-volume fluid, keeping copied velocity
advection and pressure-solver machinery where compatible. Sub-agents own bounded
components under shared contracts. Integrate the coherent bundle directly into
production; do not ship partially compatible authorities.

### State and reconstruction

Initialize liquid volume V from authored liquid intersected with open geometry.
Capacity C accounts for solids; physical mass is constant liquid density times V.
Construct PLIC planes with normals from the interface and offsets matching V.
Reconcile/redistance phi from volume-constrained geometry without changing V.
Phi must not become a second authority over liquid amount.

### Transport and pressure

Use bounded geometric face fluxes with an explicit transport CFL condition.
Each physical subface owns one transfer with equal/opposite volume updates.
Coarse-face transfer equals the sum of fine-subface transfers. Validate bounded
fractions and full-cell preservation separately from flux cancellation.

Check that projected row velocities supply the discretely divergence-free
physical fluxes required by transport; provide compatible pressure/flux treatment
where they do not. Use geometry for membership, free-surface pressure distances
and publication. Remove CM12 beta/gamma transport, sharpening, excess-density
pressure feedback and capacity repair from the volume-authority path. Do not
hide invalid transport with clamping or excess redistribution.

Preserve large outer steps immediately using conservative synchronized transport
substeps, supported velocity extension and complete swept residency. Define the
ordering of forces, velocity prediction, interface transport and projection.
Avoid a full pressure solve per transport microstep where validated integration
permits it. Later milestones optimize this working baseline.

### Adaptivity, terrain and editing belong in this cutover

- Refinement intersects the parent reconstruction with children. Coarsening
  sums child volumes; reject demotions that cannot represent the child interface.
- Transfer face state consistently and restore flux/divergence compatibility
  after topology changes.
- Reserve swept donors/receivers and reconstruction support before execution.
  Sparse growth cannot rely on the former visible-density threshold.
- Publish volume, phi, reconstruction, pressure membership and presentation
  from the same accepted generation.
- Use solid geometry for clipped intersections, not scalar capacity alone.
  Moving solids require wall velocity and swept solid-volume accounting;
  a capacity decrease cannot erase displaced liquid.
- Preserve live liquid edits/inflows and record authored additions/removals and
  boundary outflow separately from conservation error.

**Visible result:** default geometric-volume dynamics supports adaptive bricks,
large outer steps, terrain and live edits. No restricted regular-grid production
mode or automatic CM12 fallback.

**Exit:** production Dawn pools/dams, seam crossing, re-rung/growth, hillside flow,
liquid/solid edits, bounded volume and source balance, waterline, symmetry, energy
behavior and complete stage cost. Prescribed-motion scenes test translation,
rotation and deformation through production execution rather than unit tests.

This bundle is larger than a scalar swap because its authorities are coupled.
Keep first-version numerical order and optimization ambitions modest. Deliver
independently useful prerequisites through Milestone 1 as small production
increments, maintaining frequent visible progress. Do not defer existing
capabilities until after this cutover.

## Milestone 3 — improve the working volume solver

Deliver each improvement directly to production, one at a time:

1. Reconstruction continuity, thin-feature preservation and curvature.
2. Adaptivity policy, conservative handoff accuracy and cost.
3. Pressure/flux consistency and momentum transport where measured defects
   justify changes. Conserved mass is not a claim of conserved momentum/energy.
4. Solid intersection fidelity and moving-solid behavior.
5. Regular-tile execution and specialized seam/cut-cell work reduction.

Each delivery has a concrete production scene and Dawn regression receipts.
Existing supported capabilities remain available throughout.

## Milestone 4 — large-step accuracy and performance

Measure outer CFL 1, 2, 4, 8 and 16, with 25 as a stress target, against smaller
outer steps at matched physical time/resolution. Record actual dt, transport
substeps, surface error, volume bounds/balance, memory and GPU stage time.
These are test points, not promises of equal accuracy or constant cost.

Improve synchronized subcycling incrementally. Reduce the outer step when
impact/force accuracy requires it; handle explicit surface tension and small
cut-cell stability explicitly. Consider multi-cell remapping or local temporal
adaptivity only after measurements establish a bottleneck. Each implemented
improvement cuts over directly in production.

## Validation and delivery

**About 99% of testing uses Dawn. Avoid new unit tests.** Use actual production
solver/scene integration, regression lanes and GPU measurements. The literal
copy retains existing files, including tests; it does not call for expanding
a duplicate unit-test suite. Browser checks cover UI wiring and visual review.
TypeScript, module-boundary and installation checks remain routine static checks.

Keep the canonical command required by AGENTS.md:

    npm run test:dawn:sparse-cm12

Migrate its normal method/factory/import targets to the new production method.
Preserve every lane's scope and current behavioral/timing ceilings. Do not
weaken checks, delete coverage or raise ceilings to pass a cutover.
CM12-specific algorithm checks that eventually cease to describe production
must be explicitly replaced by checks of the corresponding new invariant,
with the changed meaning documented.

The default gate must demonstrably execute the new resident. Original CM12 is
only an explicit comparison target. Use the current baseline policy in
docs/SPARSE_CM12_DAWN_REGRESSION.md; attribute pre-existing failures through
source-matched evidence.

All Dawn runs are serial under the repository WebGPU lease with browser
simulation stopped. One agent owns GPU execution. Resume browser inspection
afterward. Use comparison panes visually; benchmark one simulation at a time.

Each delivery records source identity, scene/settings, physical duration,
actual resident, remaining CM12 components, relevant visual evidence,
volume/surface/energy/symmetry measures, startup cost and steady-state time/memory.
Report regressions plainly. Total volume alone is not evidence of good dynamics.

## Immediate implementation scope

The user has requested implementation of the full plan. Work proceeds through
the milestones using Astra Medium sub-agents. The literal copy and default
cutover precede numerical changes; each numerical increment must keep a
functioning production simulation and satisfy its Dawn acceptance checks.
