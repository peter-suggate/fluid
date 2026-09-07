# Coarse-first surface adaptation

The default **Coarse-first · energy + curvature** criterion starts from B1 and
adds required resolution. The existing activity and surface-distance modes
remain available in the toolstrip chevron beside the solver. Primary energy,
curvature and lookahead inputs appear beside it in the same row. All sliders
are in **SIM → Activity + resolution → Criterion**. Open
**Coarse-first · ball into still pool** in the validation scene catalog.
Changing the criterion is live; reset to compare construction-time topology.

The [symmetric expansion A/B](symmetric-expansion-coarse-first-physics-2026-09-06.md)
documents the accepted-owner curvature correction, equal-volume stencils, and
the latest regression results. The measurements below record the original
coarse-first rollout.

## Research and design

[Ando and Batty, 2020, sections 6.1–6.6](https://cs.uwaterloo.ca/~c2batty/papers/Ando2020/Ando2020.pdf)
provide the closest precedent: surface adaptivity, geometric/velocity sizing,
propagation away from interfaces, temporal coherence and progressive
construction. Their sizing function includes liquid and solid geometry and
velocity derivatives. This implementation uses a normal-cone estimate and
specific kinetic energy instead; it does not claim to reproduce their solver.

[Basilisk's adaptive wavelet discussion](https://basilisk.fr/sandbox/Antoonvh/the_adaptive_wavelet_algorithm)
illustrates selecting resolution from reconstruction error with separate
refinement/coarsening decisions. The
[curvature adaptation experiment](https://basilisk.fr/sandbox/Antoonvh/rc.c)
also illustrates why curvature alone needs temporal/feature protection. Here,
thin-fluid guards, incoming motion, conservative transfer, and accepted-output
representability receipts supply that protection.

The hypothesis is that a flat free surface carries no intrinsic finest-cell
requirement. A moving or curved feature supplies a requirement; 2:1 grading
then propagates the required support. Gravity potential energy is deliberately
excluded: a deep hydrostatic pool should not refine because it is deep.

## Implemented policy

- **Initialization:** sample the authored density gradient, including a halo,
  before discarding fine information. Let `D` be the diameter of the
  componentwise normal bounds. Select the first dyadic resolution `r` satisfying
  `D / r <= curvatureTolerance`. Constant normals select B1; varying or opposing
  normals retain curved bodies and small disconnected features. The existing
  static-solid restriction floor and authored cell-size regions also apply.
  Construct density by exact volume restriction and grade the complete wet/air
  atlas. No volume identity, scene name, or distance to the test ball is used.
- **Accepted-state geometry:** resolve the accepted owner directly, then measure
  normal bounds over interface cells and a one-cell halo for B1 leaves. Each
  gradient uses equal-width, aligned virtual control volumes, restricting finer
  donors before differencing. This keeps a flat surface flat across 2:1 seams.
  Stencils intersecting solid cut cells defer to the independent solid/coupling
  floor. Finer leaves already supply multiple normals across their represented
  interface. Store the resulting curvature floor in activity reason bits 16–20.
  This is a conservative normal-variation estimator, not a signed differential
  curvature suitable for a surface-tension force.
- **Energy:** maximum wet-cell `E = 0.5 * |u_metres_per_second|²`. The finest
  threshold defaults to 8 m²/s². A rung `r` has speed threshold
  `sqrt(2 * E_finest) * r / B`; thresholds are converted to the existing travel
  uniform using the current timestep and finest physical cell width. This
  applies in bulk as well as at the interface.
- **Prediction:** a surface or transport-support receiver searches a bounded
  neighbourhood of accepted liquid sources. Each source supplies its wet-mass
  weighted velocity. Test the swept source/receiver bounding boxes over the
  configured horizon; require a graded rung proportional to predicted travel
  divided by current gap plus one brick width. Search and decisions read an
  immutable activity snapshot. Prediction adds reason bit 8192 to a promotion.
- **Selection:** maximum of curvature, energy, incoming motion, solid geometry,
  moving-body requirements and thin/injected-liquid guards. Surface membership
  alone contributes no floor. Still-water air support uses B2: the nine-cell velocity extension needs
  fewer physical pages while remaining 2:1 with B1 liquid. It contributes no
  B8 floor, which would make the entire pool B4 again.
- **Merging:** descend one rung after the configured number of valid epochs.
  Exposed surfaces still require a fresh generation-stamped output-space proof,
  with the existing displacement and normal tolerances. Refinement bypasses
  that delay. The usual conservative transfer and atomic topology publication
  remain authoritative.

Changing policy controls invalidates incremental measurements, and prevents
old-tolerance surface receipts from authorizing a merge on that frame. The
history vector formerly used for density moments holds mean liquid velocity
in coarse-first mode; the legacy mode retains its original meaning.

## Controls

| Toolstrip input | Default | Meaning |
| --- | ---: | --- |
| Finest kinetic energy | 8 m²/s² | Energy required for the finest rung |
| Curvature tolerance | 0.25 | Permitted normal variation per cell |
| Impact lookahead | 0.5 s | Accepted-velocity prediction horizon; zero disables prediction |
| Impact search radius | 3 bricks | Spatial reach; bounded to 1–6, cost grows cubically |
| Surface proof persistence | 2 epochs | Valid epochs before a merge |

Surface displacement/normal tolerances, topology cadence/budget, thin-feature
and residency controls remain available. Inapplicable legacy travel/score
controls and the initial fine-band control are disabled for this criterion.
The existing authored refinement regions still constrain the result.

## Validation and observed work reduction

The fixture is a 6.4 × 4.8 × 6.4 m tank on a 128 × 96 × 128 lattice, with water
1.6 m deep and a radius-0.25 m liquid ball centred at height 3.65 m. It has no
refinement regions. The 1.8 m drop gap gives ballistic contact around 0.61 s.

The full-size Dawn impact run starts with 39,852 active cells. All four central
receivers reach B8 by 0.40 s; all 128 sampled far-side surface bricks remain B1
before contact. Relative mass error after 1.25 s is 0.0686%. A separate GPU
test refines adjacent 0.8 m bulk cells in place, conserving mass and enforcing
2:1 face transitions without a topology generation replacement.

Earlier construction comparison on the 96 × 48 × 96 fixture (before the larger,
deeper scene and bulk macro cover):

| Criterion | Resident bricks including air | Accepted cells | B1 pool-surface bricks |
| --- | ---: | ---: | ---: |
| Existing activity | 688 | 223,372 | 0 / 144 |
| Coarse-first | 864 | 30,880 | 124 / 144 |

That is **7.23× fewer initial accepted cells**, not a measured 7.23× frame-rate
increase. The larger air brick count follows the existing fixed-cell-depth
velocity-extension support requirement. A still pool without the ball starts
with 864 accepted cells, all B1.

Focused tests check exact initial mass agreement with an all-fine reference,
B8 curved-liquid initialization, method/control normalization, hydrostatic B1
retention, accepted receiver refinement before impact, distant B1 retention
before impact, finite fields, and mass after impact. The Dawn observations
include all four symmetric central receivers.

The latest canonical regression completed within its wall-clock budget, with
14 of 16 lanes passing. Mini32 measured 29.69 ms (40 ms ceiling). Mini64
measured 66.13 ms (50 ms ceiling); its largest costs were pressure solve
(22.61 ms) and face preparation (14.75 ms). The tall-hills lane exited with a
native SIGSEGV in the combined run, then passed its unchanged isolated lane.
The mini64 performance failure remains open; its ceiling is unchanged.

The new still-pool test retains B1 surface coverage but fails its strict mass
check: 0.001077% drift after 19 steps versus a 0.001% ceiling. The disturbed
surface recovery test passes: central receivers return to B2 while still
moving, with 0.0250% mass drift after 150 steps. Both the CPU macro-incidence
regression and the native bulk rerung regression pass. The new mode is the
configured default, but these remaining validation failures prevent claiming
a fully green production gate.

```sh
node --import tsx --test tests/sparse-cm12-coarse-first.test.ts
npm run test:dawn:sparse-cm12:coarse-first
npm run test:dawn:sparse-cm12
```

## Limits

The new criterion currently reaches one cell per ordinary brick at the surface;
uniform fully wet siblings can merge into larger bulk cells while retaining a
base-brick surface band and physical 2:1 grading. It does not merge surface
coverage. Existing macro/world
allocation and publication constraints still apply. GPU-grown pages retain the
existing fixed fine graph. Predictive reach is explicitly bounded by search
radius, and velocity averaging can underpredict counterflow inside a source
brick. Local energy, curvature and thin-feature requirements remain independent.

The all-rung host catalogue is a separate large-scene memory limit. Temporary
incidence objects are now packed directly as CSR; exact integer cell/interior
face keys replace temporary strings; row ordering reuses owned records; and
build caches are released before final packing. Admission subtracts accepted
mutable work already counted in the all-rung bound, without raising its limits. The compact B2 air band keeps the larger fixture within the fixed catalogue
budget: 2,064 resident leaves, including 2,000 base leaves and 64 bulk macros,
with 40,032 accepted cells at reset. Initially active leaves up to span two
receive all-rung GPU backing; inactive structural guards do not. Bulk cells
can therefore refine in place, with closure across every face patch enforcing
physical 2:1 grading. ITR1 incidence entries use the full 32-bit word to support
macro interfaces with more than sixteen terms. Cases beyond that budget use bounded
generation preparation at the configured topology cadence. Earlier,
a 128 × 48 × 128 version exhausted the default 4 GB Node heap during topology
construction. Reducing accepted cell work alone does not remove that catalogue cost.
No timing ceiling, regression threshold, or heap limit was raised to hide it.

The terrain regression exposed negative child density during an existing
open-volume-corrected B4-to-B8 transfer. A conservative positivity limiter now
contracts reconstruction deviations about the parent open-volume mean. This
retains parent mass rather than clipping away negative mass independently.

## Editing resolution while paused

Adding, moving, resizing, or changing a refinement region's Min/Max controls
now requests an asynchronous topology publication from the live solver. Pause
at any point, add a region, and set Min and Max to the same cell size to compare
surface reconstructions at fixed simulation time. The next draw extracts the
new surface as soon as its conservative transfer has completed. No reset or
manual physics step is required.

The editor clamps accepted cell widths to the edited bounds and closes physical
2:1 grading. It does not run transport, pressure, gravity, sharpening, rigid
motion, or advance the activity clock. Relaxing a bound permits the current
resolution to remain; ordinary automatic adaptation resumes with physics.
Topology freeze still holds accepted cells. Surface shape can change under
coarsening even though liquid volume is conserved; this makes transfer and
reconstruction movement observable separately from physical motion.

`applySceneUniforms` retains its inexpensive policy-upload contract for normal
simulation drivers. The renderer follows it with `refreshSceneTopology`. Backed
rungs use the existing resident candidate transaction: classify the edited bounds,
close grading, transfer the changed cells/faces, validate, publish, and repaint.
This reuses GPU storage and returns only a four-byte backing receipt to the host;
it does not read the fluid fields back or reconstruct the world. Rapid edits
coalesce, and pending editor work suspends physics admission.

The current storage ABI still limits dynamically created pages to B8 and leaves
some macro changes without resident candidate backing. An unsupported edit keeps
the complete accepted topology intact before requesting the existing generation
preparation path. That compatibility path retains the existing candidate catalogue
and physical growth-pool ceiling instead of prebuilding extra rungs for every new
air page. Existing memory, leaf and cell budgets remain enforced. No larger
editor allocation allowance is introduced.

This fallback is **not the target architecture for vast scenes**. Demand-created,
complete candidate backing for dynamic leaves and macro changes must replace it;
the transaction must reserve only the changed region and its grading/seam support,
then publish and reclaim those pages without rebuilding the whole resident world.
The new in-place edit path is the fast path for currently backed topology, not a
claim that arbitrary vast-scene topology creation is complete.

Run `npm run test:dawn:sparse-cm12:paused-regions` for flat and curved surfaces,
coarsening/refinement at reset and after a physics step, rapid superseding edits,
partial regions, volume conservation, and resumption. Renderer publication is
covered by `tests/webgpu-renderer-live-fluid-edit.test.ts`.
