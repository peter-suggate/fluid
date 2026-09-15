# Sparse geometric 3-D surface diagnosis

This note records the September 2026 Dawn investigation of
`sparse-cm12-symmetric-expansion` with the adaptive level-set volume path. It
separates measured defects from hypotheses and distinguishes two kinds of
reference:

- **Independent analytic check:** zero gravity and zero initial velocity must
  leave the authored box, volume, density, phi, and velocity unchanged.
- **Shared-method finest reference:** the same 3-D implementation with every
  active surface brick forced to B8. Completed post-presentation receipts prove
  that state only for the then-active bricks; transient prephysics support
  topology is not forced. This measures resolution sensitivity, but it is not
  an analytic oracle and does not validate the shared 3-D transport or contour
  method.

The ordinary adaptive arm and the forced-finest arm must be run separately.
Dawn uses the repository-wide WebGPU lease; do not run either arm alongside a
browser or another Dawn process.

## Reproduction

Capture the analytic stationary case, the moving adaptive case, and the
shared-method finest reference:

```bash
mkdir -p artifacts/sparse-geometric-surface

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-sparse-geometric-symmetric-surface-dawn.ts \
  --arm=adaptive --gravity=0 --steps=2 --stages=1 \
  --output=artifacts/sparse-geometric-surface/static-adaptive.json

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-sparse-geometric-symmetric-surface-dawn.ts \
  --arm=adaptive --steps=20 \
  --output=artifacts/sparse-geometric-surface/adaptive.json

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-sparse-geometric-symmetric-surface-dawn.ts \
  --arm=all-fine --steps=20 \
  --output=artifacts/sparse-geometric-surface/all-fine.json

node --import tsx tools/compare-sparse-geometric-symmetric-surface.ts \
  --before=artifacts/sparse-geometric-surface/baseline-adaptive.json \
  --after=artifacts/sparse-geometric-surface/adaptive.json \
  --steps=0,1,3,7,10 \
  --output=artifacts/sparse-geometric-surface/comparison.json \
  --comparison-svg=artifacts/sparse-geometric-surface/front.svg
```

Focused regression tests are:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx --test tests/sparse-geometric-surface-evolution-dawn.test.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx --test --test-name-pattern='VEX keeps projected effective velocity' \
  tests/sparse-cm12-trace-gravity-dawn.test.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx --test tests/sparse-cm12-velocity-extension-rebuild-dawn.test.ts
```

After focused diagnosis, run the repository gate once:

```bash
npm run test:dawn:sparse-cm12
```

## Confirmed initial and stationary state

The scene is a `32 x 16 x 32` finest-cell domain with `dt = 1/30 s`. The
authored liquid is the centered footprint `[8,24) x [8,24)`, height 8, and
volume 2048 fine-cell cubes. Generation 1 has four active B8 bricks, 2048
active cells, and 2601 unique adaptive vertices. The detailed initial QA found
1792 negative and 809 exactly-zero vertex phi values. It did **not** find an
all-zero field.

The pressure solve is also present and healthy in the first moving frame. A
captured pre-fix moving frame had 2048 pressure cells, 6400 pressure rows,
maximum pressure about 1963.7 Pa, relative residual between `4.45e-7` and
`5.62e-7` across the diagnostic runs, and maximum projected velocities
`(|u|,|v|,|w|) = (0.458896, 0.325712, 0.458896) m/s`. Its D4 velocity error was
`5.96e-8 m/s`. Earlier reports of zero horizontal velocity came from a stale
resident receipt, not the pressure field.

With gravity disabled, the analytic check now preserves the complete state for
two steps: volume stays exactly 2048, maximum density and published-phi changes
are zero, maximum velocity and trace displacement are zero, D4 errors are zero,
and no adaptive level-set or transport fault is reported. The stage capture
also remains symmetric before and after redistance and publishes no liquid
outside the authored footprint.

## Confirmed defects and fixes

### Authored brick geometry used a physical wall as a free surface

The brick-seed path used a closed box SDF. Its bottom face coincides with the
closed physical floor, so it introduced an artificial `phi = 0` surface where
the liquid should continue into the wall. It also formed world coordinates by
adding a rounded negative origin after scaling, which gave opposite,
grid-aligned faces different floating-point signs.

Brick seeds now use the same wall-aware `volumeExpression` as explicit box
geometry, and lattice coordinates are centered before scaling. This changes
only the existing authored-sample expression.

### Exact-knot interpolation let zero-weight corners change support

`lsvSampleCellOrdinal` accumulated support from all eight corners even when a
corner's trilinear weight was exactly zero. At a transferred box corner, one
reflection had `phi = 0` with metric support while its mirror inherited deep-air
support from an irrelevant corner. Redistance then skipped the mirror seed and
changed the symmetric pair from `sqrt(8)` to `sqrt(10)`.

The sampler now skips a zero-weight corner before loading its vertex, phi, or
support. The focused transfer/redistance box fixture is symmetric after this
change. Constraint compilation already excludes weights at or below `1e-7`, so
the constraint projector does not need an analogous change.

### Frontier continuation selected the phi closest to zero

When a new receiver page was admitted, `lsvExtendFromSlot` chose the axial
probe whose extrapolated `abs(phi)` was smallest. A farther probe could cross a
signed-distance medial ridge and project its one-sided gradient tangentially
into dry space. The choice depended on extrapolated value rather than the
physical validity radius of the old support.

Frontier continuation now chooses the nearest old support distance, with
`abs(phi)` only breaking equal-distance ties. At an exact axial probe knot it
prefers the incident cell on the advancing side, then takes the signed slope
between that cell's two axis faces. This preserves an oblique affine field and
does not depend on whichever transverse incident cell the generic owner lookup
happens to return. The two face samples read no more corner values than the old
source sample plus full 3-D gradient reconstruction.

### A sparse diagonal characteristic discarded valid phase evidence

After the contour fixes, a later moving frame exposed a separate support-edge
case. The accepted metric vertex at `[3,7,1]` had `phi = 3.3166249`, while its
clipped RK2 departure was `[3.0016434,8.0601530,0.9899294]`, only `1.0602026`
fine cells away. The diagonal departure had no represented interpolation cell,
and the old advection fallback considered only already-deep source vertices,
so it faulted despite a direct distance certificate that the characteristic
could not reach the interface.

Advection now consumes the travelled distance from any finite non-absent
source clearance. It accepts the fallback only when
`abs(sourcePhi) - travel > 1e-5`, stores that conservative signed remainder,
and marks it as deep phase support. It never publishes the unsampled endpoint
as metric or creates a zero crossing. This uses values already loaded by the
existing failure branch.

### Post-pressure VEX restored the previous frame's velocity

The first VEX build correctly began from the accepted source velocity. After
pressure projection, collocation published the current velocity to the
effective transport plane, but the second VEX initialization immediately
reloaded `sourceCellVelocity()`. On the first frame that bank is still zero, so
the transport receipt reported zero trace displacement and density remained
unchanged despite a nonzero, symmetric projected velocity. The following frame
then used the previous projection, producing a one-frame lag.

A hooked VEX initializer now reads one native `vec4` from its effective-plane
authority. Hookless standalone fixtures retain the source-bank path. The Dawn
fixture deliberately supplies conflicting stale source and projected plane
values and confirms that the projected value is retained; both the focused VEX
fixture and the production stationary test pass.

### Collocation used volume density to decide whether to publish velocity

Pressure membership, VEX seeding, and phi advection use the accepted level-set
phase, but collocation previously published projected effective velocity only
when conservative density exceeded the liquid isovalue. A cut cell may be
level-set liquid while its volume fraction is below that threshold, leaving phi
advection with a different velocity authority from pressure.

Collocation now uses `pressureAcceptedCellMember(id)`, the frozen membership of
the pressure solve it just projected.

## Relation to the 2-D level-set path

The 2-D Rust implementation uses level-set membership for both velocity
extension and pressure (`phi <= 0` in `numerics.rs`). It also commits projected
transport support before its final pre-transport extension. The 3-D resident
does the equivalent support publication inside
`encodeTopologyEditTransaction(..., "prepare", ..., projectedTransportFrontier=true)`:
despite the name, that call authorizes and publishes the shadow topology,
accepts frontier pages, refreshes indirect work, rebuilds compiled topology and
LSV, and only then returns. The first corrected moving frame has 16 active B8
bricks and 8192 accepted cells. There is no evidence that an extra pressure pass
or a host-side transaction reorder is needed.

The 2-D `phi <= 0` versus 3-D cell-center `phi < 0` distinction remains visible
in source, but the captured first frame disproves the proposed failure mode of
an empty liquid/pressure set. It is not changed as part of these fixes.

### Remaining conservative-volume limitation

The moving run conserves total accepted volume but does not keep every cell
inside its open-volume capacity. On the first corrected frame, 896 cells were
over capacity and the largest excess was 0.083104 fine-cell cubes. The final
receiver marginal residual was 0.154789, while the final donor residual was
only `3.58e-7`. This is the expected signature of the current fixed balancing
order: the third donor normalization makes donor columns conservative, but it
perturbs receiver rows after their last normalization. A full donor can then
gather more than one receiver capacity even though total mass is conserved.

The 2-D code uses the same three row/column rounds, so the round count alone is
not a dimensional defect. Its raw weights differ materially: 2-D traces four
corners and the centre, clips the resulting deformed footprint, and only then
balances. The 3-D path traces one centre and rigidly translates an axis-aligned
box. Neighbouring boxes can overlap or leave gaps under a velocity gradient,
which gives the fixed marginal rounds a harder matrix before balancing.

Two same-schedule prototypes were measured and rejected. Tracing the six
area-averaged face bounds reduced the first-frame maximum density from 1.08310
to 1.06262, but density then grew faster, reaching 10.57 by frame 10, and one
box inverted at frame 11. Treating unsupported old-space overlap as a virtual
zero-density source left frame-one density unchanged and was slightly worse by
frame three. Neither prototype remains in production. Increasing the balancing
round count, clipping volume, or adding a corrective pass would conflict with
the fixed-work requirement and would conceal rather than establish the missing
3-D geometric remap.

The existing excess-volume pressure source then turns local `V > capacity`
into positive divergence. Phi follows that expanding velocity while total V
stays fixed, so repeated remap compression appears as a growing phi-enclosed
region. Replacing this source with signed `V - C H(phi)` feedback was not made:
2-D treats that mismatch as a diagnostic, and such a change would force phi to
hide the transport error without making the remap bounded.

## Work-count audit

No confirmed fix adds a compute pass, iteration, full-field copy, or host
round-trip:

- wall-aware initialization and centered scaling replace one expression;
- exact-knot interpolation skips irrelevant corner reads;
- nearest-support continuation replaces the full 3-D gradient reconstruction
  with two weighted axis-face samples and uses at most the existing eight
  incident-owner queries;
- diagonal phase certification consumes already-loaded source clearance only
  after ordinary sampling and frontier continuation fail;
- the hooked VEX initializer replaces three scalar state reads with one native
  `vec4` plane read;
- collocation replaces one wet predicate with the already-published pressure
  membership predicate.

The probes, stage lenses, and QA readbacks are test-only.

When `lsvAdvectPhi` halts, its existing four-word sticky failure receipt now
stores the fault mask followed by the clipped characteristic departure in
fine-cell coordinates. The surface probe joins that receipt to the preceding
accepted checkpoint in `failureDetail`, including the owner vertex position,
phi, support, source bank, level-set generation, cell/vertex counts, and
accepted topology summary. This adds no production dispatch or buffer; it
reuses words that were previously reserved zeros. The complete preceding
vertex field remains in that checkpoint for neighbourhood analysis.

## Historical moving evolution from the support-fixed build

These measurements come from the intermediate `support-fixed-adaptive` build,
before the later directional frontier changes were combined. They demonstrate
the effect of the fixes present in that build; they are not the final combined
result. That intermediate probe completed 30 steps without a validation error,
transport fault, or adaptive-level-set fault. The original baseline halted in
`lsvAdvectPhi` at frame 8. Accepted volume in the intermediate run was
2048.00009 at step 1, 2048.00011 at step 3, 2048.00026 at step 7, and
2048.00037 at step 10: the largest relative drift before wall contact was
`1.79e-7`.

The birdseye audit thresholds cell-centre phi into a phase mask, samples that
mask bilinearly along 128 fixed rays at 1/8-cell intervals, and reports
high-frequency radial roughness after removing smooth angular modes through
12. It is a footprint/jaggedness measure, not a direct subcell phi-zero
crossing or a renderer contour. This distinguishes grid jaggedness from the
intended four-sided shape; it does not fit or expect a circle. The projected
published-phi phase-mask front remained D4-exact through step 7. Its
ground-front roughness RMS was 0.108 cell at step 1, 0.069 at step 3, 0.119 at
step 7, and 0.152 at step 10. The step-10 angular D4 difference was one radial
sample, 0.125 cell. In the original baseline, the projected published-phi
phase-mask front was already
3.75 cells out of D4 at step 1, 4.375 at step 3, and 2.5 at step 7; that run
then halted.

The intermediate build's upper zero crossing expanded from the authored `[8,24)` square
to `[7,25)` at step 1 while remaining exactly symmetric. Its maximum reflected
height error was 0 at step 1, 0.00147 cell at step 3, 0.0101 at step 7, and
0.00595 at step 10. Velocity D4 error was `5.96e-8 m/s` at step 1,
`0.00181 m/s` at step 3, `0.0138 m/s` at step 7, and `0.0141 m/s` at step 10.
The surface therefore evolves and spreads in the first frame rather than
remaining frozen behind the pressure solve.

That run stayed entirely B8 at steps 0, 1, 7, and 10; step 3 contained four B4
and twelve B8 bricks. A new post-fix forced-finest run was not captured at this
checkpoint, so these measurements establish analytic stationary correctness,
conservation, continued evolution, and early front quality, but do not claim a
final adaptive-versus-finest error. The checked-in offline comparison is
`artifacts/sparse-geometric-surface/before-after-comparison.json`; its companion
three-panel SVG shows integrated-density, ground-phi phase-mask, and projected
published-phi phase-mask outlines at the selected equal steps.

The subsequent directional frontier build remained close to D4 through step 8
but halted in `lsvAdvectPhi` at frame 9. Its last accepted level-set image was
generation 3 with 9,801 vertices and 8,192 cells; fault owner 4,895 was the
metric-support vertex at fine coordinate `[1,7,22]`. This result is under
investigation and superseded the intermediate build's long-run claim. The
retained phase-certification change then advanced the directional level-set
core cleanly through the old failure and through frame 10.

A separate face-box departure experiment subsequently failed the
geometric-volume transport `buildCoupling` contract at frame 11 (owner cell
4,899) and was rejected. That owner is a transport cell and must not be
interpreted as an adaptive-level-set vertex. The retained directional and
phase-certification changes do not yet have a successful final long-run probe.

The 2-D implementation's deformed-footprint behavior and its explicit contour
reconstruction may eventually provide a stronger independent comparison for a
moving 3-D slice. That comparison has not yet been established. Differences in
footprint or contour machinery are limitations of the current benchmark, not
confirmed 3-D defects and not reasons to add work to the runtime.
