# Fluid surface handoff — 8 September 2026

## Read this first

**Production evolved surfaces are still incorrect. No replacement transport or
translation-only production cutover was implemented.** The user stopped this
session and requested this handoff immediately, concerned that discussion of
an affine/translation path was becoming a shape-preserving shortcut.

The next session must focus on **the actual UI/production falling-ball scenes**,
first fully fine, then adaptive. A sphere should remain spherical during
undisturbed freefall and the disconnected pool should remain flat. Establish
that from the simulated density and motion, not by forcing the expected shape.
The previous session spent too much time on isolated mathematical prototypes.
Their passing tests do not resolve the visible regression.

User requirements, in priority order:

- Identify and fix the root cause in real production scenes. Both coarse and
  all-fine A/B surfaces were wrong; all-fine is not a correctness reference.
- No mesh smoothing, surface projection, shape fitting, analytic falling-ball
  overlay, prescribed-gravity animation, or translation-only workaround.
- New work should reach production/UI so the user can verify it. Do not keep
  accumulating disconnected prototypes and call that a completed milestone.
- One current spatial density authority must supply both native solver amounts
  and the displayed implicit surface. Refining a native cell only restricts
  that same field; it must not invent new geometry.
- Preserve subtle curvature and intentional sharp branches. Pursue a durable
  method, with compiled neighbor access and GPU execution for large work.
- Compare against independent analytic cases, then real scenes; run core
  regressions periodically. Do not weaken thresholds or timing ceilings.
- Use Astra subagents, medium normally; extra-high was explicitly authorized.
- Commit validated owned progress, avoiding other tasks' changes.

## Workspace and stopped state

Repository: `/Users/petersuggate/code/me/fluid`, shared checkout, branch `main`.
Code HEAD when this handoff was written: **`ca1da5ce`**. The handoff itself may
be the following documentation commit. No production shader edits occurred
during the final architecture discussion.

All three subagents were interrupted on the user's stop request. Root checked
the process inventory: no Dawn/capture process remained, and
`/tmp/fluid-webgpu-exclusive.lock/owner.json` was absent. The other SVO task was
notified that our GPU reservation is released. Recheck before acquiring GPU.

Repository `AGENTS.md` requires `npm run test:dawn:sparse-cm12` after large
simulation, topology, presentation, terrain or live-edit changes. Never run
Dawn alongside another Dawn process or a GPU-active browser. Other tasks share
this checkout. Do not reset it or stage all changes.

## What is wrong in production — established evidence

The live retained support representation is

```text
q_K(x) = a_K * q_seed(x) + b_K.
```

Its update sees old/new native means but no departure coordinate or transported
spatial derivatives. Initially dry supports stay spatially constant as liquid
arrives. Normals in an old ramp remain parallel to the original seed normals.
Means can match while the surface is pinned, stepped or distorted.

A full-fine isolated sphere with **actual production transport and prescribed
uniform velocity**, before meshing, established three distinct defects:

1. Native CM12 center-mean transport diffuses the shape. A half-cell translation
   becomes `.5*rho_i + .5*rho_(i-x)` (then `.25/.5/.25` after two steps), rather
   than integrals of the translated spatial density. Actual GPU values match
   those stencils to `4.47e-8`. Mean errors reach `.1077/.1619`; global M0 still
   agrees to roughly `3e-9`.
2. The retained lift then creates density face jumps `.3613/.4152` after one/two
   steps. Native/retained amount agreement does not diagnose that geometry loss.
3. `cm12RetainedDensityPhiAtFine` has false-zero endpoint shortcuts. In the first
   step, 144 exactly-zero published samples have actual density unequal to .5;
   the worst discrepancy is .5. Removing only that shortcut cannot restore the
   missing spatial transport or eliminate real half-density plateaus.

Detailed evidence:

- [Imposed-flow diagnosis](retained-imposed-flow-diagnosis-2026-09-08.md)
- [Authority contract and negative controls](retained-density-motion-authority-design-2026-09-08.md)
- [Production dependency audit](retained-density-production-cutover-audit-2026-09-08.md)
- Raw old-field capture: `artifacts/retained-imposed-flow/sphere-full-fine/`
- Old/new/analytic density plot:
  `artifacts/retained-imposed-flow/comparison/density-comparison.png`

The actual quarter-scene coarse/all-fine A/B is in
`artifacts/retained-visual-ab/quarter/`, at steps 0, 6, 15 and 30. Both evolved
surfaces are wrong; the coarse arm also develops asymmetry and weaker impact.
At 1 s, coarse/fine kinetic energies were 38.88/92.65 J; coarse COM X drift was
−9.297 mm versus approximately −0.000021 mm fine. These were shipping GPU meshes,
not remeshed analytic surfaces. See [A/B report](retained-visual-physics-ab-2026-09-08.md).

## Exact real scenes and the newest unfinished diagnostic

Scene IDs:

- `coarse-first-pool-impact-quarter`
- `coarse-first-pool-impact-half`
- Full-pool memory/regression scene: `coarse-first-pool-impact`

Original local min/max-8 query: **`0_0_0_25_66.6667_100_8_8`**. Use the actual
query parser, not idealized replacement bounds. In the quarter it gives
`min=(-.8,0,-.8), max=(-.4,.8,.8)`. The half's literal height snaps differently.
Fully fine means min=max=1 over the entire physical domain; verify actual
accepted native widths, including runtime leaves. Use balanced/coarse-first
with paper time step for the falling-ball comparison.

Two **uncommitted** files were added immediately before stopping:

- `tools/capture-retained-falling-velocity-dawn.ts`
- `tools/retained-falling-velocity-analysis.ts`

These read real gravity-driven production state without prescribing velocity,
pressure, gamma, density or retained coefficients. The fine arm completed six
steps and twelve snapshots in 21.056 s before the stop request was processed:

```text
/tmp/fluid-retained-falling-velocity-fine.log
artifacts/retained-falling-velocity/quarter/fine/
  configuration.json, provenance.json, trace.json, completed.json
  step-N-transport-velocity-extension.{bin,json}
  step-N-velocity-projection.{bin,json}
```

**The diagnostic and its aggregate conclusions have not been independently
reviewed.** The adaptive arm was not run. Do not treat its reported large halo
velocity/strain residuals as proven physical defects: the current aggregation
includes native samples without checking the VEX validity/depth mask, and uses
the effective-velocity plane for both stage labels. Verify that plane's exact
meaning at projection, current native ownership, scalar parity, sample validity
and geometric coverage before making claims. Raw bank velocities were also
saved. A sphere bulk and a diffuse/air halo must not be conflated.

Production stage ordering matters: scalar transport precedes force addition.
First-step VEX can correctly be zero; the first gravity response appears after
force/projection and in the following VEX. Compare against the solver's actual
discrete time ordering, not continuous gravity at a mismatched time.

The intended commands, only after reviewing that adapter and taking the GPU
lease, are:

```bash
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
  node --import tsx tools/capture-retained-falling-velocity-dawn.ts --arm=fine
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
  node --import tsx tools/capture-retained-falling-velocity-dawn.ts --arm=coarse
```

## Production integration seams — not an implemented solution

The dependency audit identified three required changes that belong together:

1. Consume immutable actual velocity after `transport-velocity-extension`,
   before conservative gather overwrites effective transport velocity
   (`webgpu-sparse-cm12-resident.ts`, around line 6515; locate by stage name).
2. Replace `advanceRetainedDensitySupport` / `a*q_seed+b`, and the old
   `targetMean` comparison in native integration. The transported current field
   must author native density, not be fitted to old CM12 scalar targets.
3. Publish point queries from the same accepted current field. Replacing only
   `cm12RetainedDensityPhiAtFine` would be a display workaround.

Momentum, gamma diffusion, sharpening, solids and injection are additional
writers/consumers. They must operate consistently on the new measure. Merely
overwriting native rho after old gather is not a validated physics cutover.
Relevant code is `webgpu-sparse-cm12-resident.{ts,wgsl.ts}` and
`sparse-cm12-retained-scene-density.ts` under `lib/methods/adaptive-mass/`.

**No final general carrier has been selected.** Do not inherit exploratory
suggestions as approved architecture:

- A global affine map cannot describe a stationary pool and falling sphere.
  Per-component translations were discussed, not implemented or validated, and
  the user specifically objected to pursuing a translation hack. Do not resume
  that as the production solution.
- A current inverse map with `rho=q0(X)*det(DX)` was only discussed. Conservation
  requires injectivity, orientation and complete material coverage, not just a
  positive sampled determinant. C1 map interpolation is insufficient for smooth
  normals of that same rho because Jacobian derivatives can jump; at least C2
  would be needed near a regular interface. No production implementation exists.
- CM12 gamma is the cumulative row-sum correction of its discrete transport
  matrix, **not** a stored physical Jacobian. Substituting `gamma=J` and rendering
  `rho/J` would change the model and separate geometry from the current density.
  It is not a compatible drop-in or an approved direction.

The original paper is `docs/papers/massConservingLiquids.txt`, particularly
§§3.4–3.8. Distinguish descriptions in papers/documents from the user's request.

## Useful completed work, with strict scope limits

| Commit | Work | What it does not establish |
| --- | --- | --- |
| `21c503b0` | Current quadric GPU coefficients and same-field amounts publish atomically; coverage/integration failures and host publication race tested | General deforming field, production transport, momentum or impacts |
| `dc46b89c` | Matched GPU quadric/old production/analytic comparison; q error <7.1e-7 and relative M0 error <2e-7 in two prescribed translations | Shipping falling-ball correctness |
| `9d92974e` | GPU uniform native-VEX recognizer, immutable pre-gather capture, sealed generation/coverage proof | Real gravity's nonuniform flow or production cutover |
| `6412bcbb` | Tensor CSL4 GPU, five cases including bounded-range rejection and rollback | General saturated/branched liquid transport |
| `01d41221`, `36e8571b` | CPU C2 MAC divergence/face-flux interpolation and periodic Hamiltonian pair-map oracles | Native VEX conversion, GPU implementation or production physics |
| `f3f9eae2` | Counterexample: arbitrary saturated Galerkin moment fitting may have no finite C2 potential solution | A selected alternative |
| `3f0dcb04` | 450-frame recorded live-editor replay passes current atomic wet-overlap rejection contract | Accepted insertion of all recorded wet strokes |
| `ca1da5ce` | Remove seven unused warm-up pipeline entries; actual dispatch availability test | Fixing canonical startup deadlines |

Quadric GPU: `/tmp/fluid-quadratic-pullback-gpu-9.log`, 17/17 subcases, 2.869 s.
Uniform VEX GPU: `/tmp/fluid-uniform-vex-map-gpu-2.log`, 21/21 subcases.
Actual prescribed native VEX: `/tmp/fluid-native-uniform-vex-capture-2.log`, two
steps, 5.689 s; map `B=I,t=(-.02500000037252903,0,0) m`. The first failed capture
had incorrectly expected collecting FCA phase; corrected checks require actual
sealed authority, not relaxed validation.

There are useful mathematical failures in
`docs/latent-hierarchy-feasibility-review-2026-09-08.md` and
`docs/galerkin-potential-feasibility-2026-09-08.md`. Do not repeat a large research
ladder before returning to the live scene. A newer six-test 1D weak-evolution
oracle remains uncommitted and experimental; it does not solve strong saturation
or production curved-feature transport.

## Regression, memory and initial-shape status

Latest unchanged canonical gate:
`/tmp/fluid-current-field-canonical-1.log`, **180.043 s; five passed, six timed
out, six unrun**. No passing canonical receipt. Timeouts: symmetry, topology
page budget, hydrostatics, mini32 correctness, mini32 performance and mini64
performance (the latter had only 12.300 s of remaining suite time).

Initial quarter/half geometry and thirteen paused partitions passed earlier:
`/tmp/fluid-retained-pool-production-6.log`, two cases, 64.1 s. Initial implicit
pool plane errors are about 1e-16 m; sphere polygon errors remain finite
tessellation errors. These do not prove evolved shape correctness. Historical
mesh work (`0cabd42d`) predates the user's explicit no-repair steering; do not
extend that approach to conceal transport errors.

GPU topology memory improvements **are production**: `e78daf19` interned catalog
expansion and `cfde4126` GPU generation intersection replace large repeated CPU
geometry objects. The earlier failing 1,836-leaf catalog dropped to 166 reference
builds; CPU preparation measured 403.5 MB peak heap versus >2.48 GB previously.
Typed certification shadows remain CPU memory. Full-pool completion remains open.

Newest full-pool attempt `/tmp/fluid-full-pool-gpu-catalog-current-1.log` reached
step 7, with two accepted replacement generations, without OOM. Root stopped it
during step 8; it is **incomplete**. Several earlier replacement attempts threw
and were handled as deferred work; the old probe did not print deferral details.
Heap at step 7 was approximately 1.46 GB; this is not a memory/performance pass.
The probe was then edited to print deferral/allocation/heap-limit information and
check simulation health. Its retry did not start: it encountered the stopped
process's stale lease. That lease was subsequently removed after confirming the
owner PID was gone. Do not claim 30-step completion.

An important QA correction: `dynamicLiquidMassFineCells` in the world-growth
receipt sums `max(A,B)`, not the accepted bank, and can falsely report mass
growth during conservative motion. Use GPU-selected accepted density, e.g.
`readDiagnosticFields(true)`, including runtime leaves.

## Uncommitted work to preserve and review

Owned by this stopped surface task:

- `tools/implicit-density/uniform-vex-map-gpu.ts`: partial 8-word coverage
  certificate footer, not GPU validated after `9d92974e`.
- `tools/implicit-density/uniform-vex-quadratic-coupling-gpu.ts`: untracked,
  incomplete GPU admission helper; not imported, integrated or GPU validated.
- `tools/implicit-density/weak-segment-density-line-oracle.ts`,
  `tests/weak-segment-density-line-oracle.test.ts`,
  `docs/weak-segment-density-line-decision-2026-09-08.md`: agent reports six CPU
  tests/strict types pass; no general 3D or production relevance established.
- `tools/probe-sparse-cm12-full-pool-advance.ts`: diagnostic additions described
  above; modified version did not acquire GPU.
- The two real falling-velocity capture/analysis files listed earlier.

Other tasks' changes: `app/globals.css`, `components/EntityOptions.tsx`,
`components/OakTreeEditor.tsx`, `docs/OAK_V2.md`, `lib/core/editor-entity.ts`,
`lib/core/oak-tree-controls.ts`, `tests/host-transport-status.test.ts`,
`tests/oak-tree-controls.test.ts`, and the untracked
`tests/sparse-cm12-retained-preparation-recipe.test.ts`. Preserve them. The
untracked `tools/__pycache__/` is generated. Always inspect fresh git status.

The editor task's completed `68aad476` and SVO task's `afc42ccc` are unrelated
owned commits. Public solid-edit preparation now uses asynchronous atomic
preflight/apply; do not restore the obsolete synchronous replay assumptions.

## First steps for the next session

1. Reproduce the visible production defect with the exact scene/settings above.
   Review the new real-gravity raw snapshots and adapter first; establish valid
   velocity samples and the first frame/stage at which spatial geometry changes
   incorrectly. Then capture the adaptive arm. Keep the analytic sphere a QA
   reference only.
2. Select and implement a genuine current-field transport method at the three
   production seams together. Do not treat an imposed uniform-flow test or an
   affine recognizer as a general fluid solver, and do not force a sphere shape
   against the actual simulated deformation.
3. Demonstrate the resulting production density, surface and native amounts in
   the user's UI, fully fine and adaptive, before extending isolated research.
   Track shape, pool flatness, mass, momentum and timing independently. State
   explicitly any unsupported physics or failed gate.

The user asked to continue in another session. This document preserves the
state; it does not authorize the speculative translation/Jacobian proposals.
