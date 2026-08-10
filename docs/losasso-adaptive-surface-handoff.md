# Losasso adaptive-surface scaling handoff

> **Superseded 2026-08-09 by
> [losasso-fully-adaptive-surface-handoff.md](losasso-fully-adaptive-surface-handoff.md)**,
> which carries the plan forward. This document remains as the record of the
> baseline measurements and the rejected experiments.

## Goal

Make refinement-region minimum cell sizes apply to the complete Losasso
method: free-surface transport, redistance, volume correction, velocity
extension, and rendering. The target is the water-box dam break with a
full-tank minimum cell size of 4³:

- it should be substantially faster than normal refinement;
- its surface should be smooth at the authored coarse scale;
- normal refinement must not change visibly or numerically;
- symmetric-expansion must not worsen;
- the final method must have one adaptive authority, not conditioned physics
  paths or dense fallbacks.

The reference is
[losasso-2004-octree-water-smoke.txt](papers/losasso-2004-octree-water-smoke.txt),
especially Sections 3 and 6. It stores scalar fields at octree nodes,
constructs node velocities at the coarsest neighboring-face scale,
semi-Lagrangian advects phi at those nodes, and fast-marches on the adaptive
structure.

## Repository state

Work began at commit 61ff946. Relevant preceding commits:

- 9dec9da publishes exact accepted adaptive surface nodes.
- 61ff946 publishes unique accepted adaptive leaf edges.
- 8cd9f86 advects the coarse surface from nodal velocity.
- c75e023 retires the older coarse MAC transport seam.

At handoff there is one uncommitted production change in
lib/webgpu-octree-losasso-coarse-phi.wgsl.ts. Compact-row gradients now sample
at plus/minus the row span and divide by that span. A size-1 row follows the
original arithmetic and coordinates exactly.

## Dawn reproduction

Normal refinement:

    FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=0 WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/probe-dam-surface-shape.ts

Full-tank 4³ minimum:

    FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=4 FLUID_REFINEMENT_REGION_SCOPE=full WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/probe-dam-surface-shape.ts

Pass attribution:

    FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS=8 FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS=8 FLUID_REFINEMENT_REGION_FLOOR=4 FLUID_REFINEMENT_REGION_SCOPE=full WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/benchmark-power-dam.ts --lane=ui --steps=20 --pass-timestamps --isolate-pass-labels

Symmetry:

    npm run test:webgpu:symmetric-expansion:one-step

Do not run Dawn concurrently with browser WebGPU. An interrupted isolated run
can leave /tmp/fluid-webgpu-exclusive.lock; verify the recorded owner PID no
longer exists before removing that exact stale lock.

## Baseline A/B at 0.248 s

| Metric | Normal | Full-tank 4³ |
|---|---:|---:|
| Pressure-required rows | 1,296 | 24 |
| Pressure iterations | 19 | 5 |
| Accepted topology leaves | 4,665 | 178 |
| Adaptive nodes | 5,717 | 357 |
| Adaptive edges | 16,493 | 935 |
| Dense predicted surface cells | 6,912 | 6,912 |
| Extension-band faces | 13,383 | 10,711 |
| Interior surface ridge | 0.0908 cells | 0.3195 cells |
| Approximate probe wall time | 1.80 s | 1.55 s |

Normal's strict surface-profile fingerprint is:

    7.9741, 7.9557, 7.9185, 7.8612, 7.7853, 7.6994,
    8.0649, 7.9334, 7.7714, 7.5591, 7.7536, 7.5016,
    5.7620, 3.6283, 2.7147, 2.1398, 1.6887, 1.3762,
    0.9203, null, null, null, null, null

The full-tank 4³ baseline begins:

    9.1532, 9.1217, 9.0616, 8.9857, 8.9052, 9.4727,
    9.4008, 9.2669, 9.0518, 8.8476, 9.3373, 8.7063,
    7.0446, 4.3371, 3.2299, 2.2426, 1.0072, ...

Normal must remain byte-for-byte identical at this checkpoint. Comparing only
ridge or volume is insufficient: several rejected experiments preserved one
summary metric while visibly changing the profile.

## Performance diagnosis

The pressure side already scales well. The surface side does not:

1. WebGPUOctreeCoarseSummary still predicts all 6,912 finest cells.
2. It owns three dense cell-centred phi banks.
3. It runs dense seeding, repeated dense Jacobi/Eikonal redistance, dense
   volume reduction/correction, and dense-complement publication.
4. WebGPUOctreeLosassoExtensionBand stages the entire finest MAC lattice twice
   per advance, then stages the entire finest nodal lattice twice.
5. In a coarse region each finest MAC invocation reconstructs velocity through
   the compact coarse-face hierarchy. A coarse run is therefore more expensive
   per staged face than a normal run.

Measured isolated GPU passes:

| Pass | Normal | Full-tank 4³ |
|---|---:|---:|
| Accepted finest-MAC staging | about 0.67 ms | about 1.40 ms |
| Predictor finest-MAC staging | about 0.63 ms | about 1.18 ms |
| Dense surface prediction | about 0.14 ms | about 0.14 ms |
| Accepted nodal staging | about 0.05 ms | about 0.05 ms |
| Predictor nodal staging | about 0.05 ms | about 0.05 ms |

Topology shrinks by roughly 16 times and pressure rows by roughly 54 times,
but surface work remains finest-domain-sized. This is why end-to-end speed
improves only slightly.

## Representation mismatch

The current method has two incompatible surface representations:

- accepted topology already publishes a compact adaptive node and edge graph;
- transported phi, redistance, correction, and the compatibility tail remain
  finest cell-centred.

Compact coarse rows then sample that dense tracker. In particular, trackerPhi
previously estimated a size-4 row gradient with plus/minus one finest-cell
probes. That imports finest-scale slope and noise into a coarse affine row.
The surviving change corrects the stencil to plus/minus header.size.

That correction is necessary but not sufficient. The Dawn dense-surface probe
is almost unchanged because the dense tracker remains authoritative.

## Rejected experiments

All changes described below were fully reverted.

### Adaptive transport followed by dense redistance

An adaptive-node phi buffer was transported and trilinearly materialized into
the dense banks before the existing redistance and correction sequence.

The coarse surface initially became more monotone, but pre-correction volume
collapsed and the normal reservoir profile visibly rose. Repeated
adaptive-to-dense-to-adaptive remapping filtered the field every advance. The
dense compatibility view had become authoritative again.

### Persistent adaptive transport without adaptive redistance

A persistent node field with generation tags and topology-handoff
interpolation was tried. Dense redistance was removed, but no adaptive
edge-based redistance replaced it.

Normal changed from roughly eight cells high to roughly ten, lost about 35
liquid cells, and full-tank 4³ ridge grew to about 0.97 cells. Persistent nodes
alone are insufficient: transport, redistance, and volume control must move
together.

### Coarse velocity interpolation with dense phi

Velocity for coarse surface cells was interpolated from the owning leaf's
coarse corner nodes while dense phi and redistance remained.

Normal stayed exact, but full-tank 4³ ridge worsened from 0.3195 to 0.4601
cells. Dense redistance regenerated finest-scale structure after transport.

### Post-hoc affine projection

Interface cells in coarse rows were exposed through a leaf-scale affine phi
reconstruction while the internal dense tracker remained unchanged.

Normal could be held exact by restricting projection to coarse interface rows,
but full-tank 4³ ridge worsened to about 0.82 cells. Presentation-only
smoothing created disagreement among compact rows, dense air-side samples, and
the internal dense field.

### Finest-MAC interior-owner shortcut

Strict interior planes of a coarse pressure leaf bypassed the initial compact
face search and reconstructed directly from the owner.

Normal stayed exact and the two expensive staging passes improved by about
four percent. Coarse dynamics changed slightly because extension records can
coexist on those planes. It was reverted because it was not semantics-neutral
and was too small to justify landing before the surface representation.

## Current surviving increment

trackerPhi(position, span) now:

- samples the dense tracker at position plus/minus span;
- divides the central difference by span times physicalCellSize;
- is called with header.size;
- is exactly the old operation when header.size is one.

Validation:

- normal dam-break profile and counters are exactly unchanged;
- full-tank 4³ remains stable, although its dense ridge is not yet improved;
- npm run test:webgpu:symmetric-expansion:one-step passes;
- the symmetry run reports exact volume and topology symmetry with clean
  WebGPU validation.

## Required next architecture

The existing adaptive nodes and edges need to become the sole Losasso surface
graph:

1. Store persistent double-buffered phi on adaptive node identities.
2. At topology changes, initialize refined nodes by interpolation from the
   prior accepted adaptive leaf. Retained and coarsened nodes reuse shared
   identities.
3. Construct node velocities using the coarsest adjacent-face scale from
   Section 3.
4. Semi-Lagrangian advect only published adaptive nodes.
5. Seed and propagate signed distance over unique adaptive edges. Edge length
   is span times h; use a deterministic ping-pong minimum schedule or an
   equivalent exact schedule.
6. Measure and correct volume on adaptive leaves or nodes. Correction must
   update the adaptive authority, not a materialized dense view.
7. Derive pressure-row phi, intervals, gradients, and rendering data from the
   finalized adaptive field.
8. Materialize dense data only for consumers that still require the legacy
   ABI. It must never feed the next transport or redistance step.
9. Replace the four full finest-lattice velocity staging passes with adaptive
   node scheduling after normal equivalence is established.

This must land as one coherent authority transition. Individual components can
be developed and measured in shadow, but switching only transport, only
redistance, or only publication has already been shown to regress.

## Increment gates

For every increment:

1. Run normal dam break at 0.248 s and compare the complete profile and
   receipts with the fingerprint above.
2. Run full-tank 4³ at 0.248 s and require a lower ridge or a measured workload
   reduction without volume or topology damage.
3. Run test:webgpu:symmetric-expansion:one-step; use the longer lane when the
   new authority persists across multiple steps.
4. Capture pass timestamps only after semantic gates pass.
5. Revert immediately if normal changes or the coarse ridge worsens.

A faster pressure solve is not evidence of surface scaling. Surface work must
track adaptive node, edge, and leaf counts rather than domain volume.
