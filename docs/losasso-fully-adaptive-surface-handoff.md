# Losasso fully-adaptive surface handoff

Supersedes `docs/losasso-adaptive-surface-handoff.md` (kept for its experiment
log; its baseline tables are reproduced here). Written 2026-08-09 at HEAD
61ff946 plus one uncommitted change to
`lib/webgpu-octree-losasso-coarse-phi.wgsl.ts`.

## Goal

Make refinement-region minimum cell sizes apply to the complete Losasso
method: free-surface transport, redistance, volume correction, velocity
extension, and rendering. Target: the water-box dam break with a full-tank
minimum cell size of 4³ must be substantially faster than normal refinement,
smooth at the authored coarse scale, with normal refinement unregressed,
symmetric-expansion not worsened, and **one adaptive authority** — no
conditioned physics paths, no dense fallbacks feeding physics.

Reference: `docs/papers/losasso-2004-octree-water-smoke.txt`, Sections 3
and 6.

## Non-negotiable principles

1. **Structural work reduction only.** Dispatch-level optimization never
   improves this lane (standing direction, confirmed repeatedly: the
   interior-owner shortcut measured 4% and was rejected; indirect-dispatch
   plumbing exists and moved nothing). Every work package below must make
   dense passes *cease to exist*, with surface work proportional to adaptive
   node/edge/leaf counts, not `nx·ny·nz`.
2. **The authority invariant.** Whichever representation feeds the *next*
   advance's transport is the authority; everything else is a view. Every
   rejected experiment (§ Why increments failed) violated this: a component
   was switched while the dense field remained upstream of the next step, so
   the dense authority filtered the adaptive result back out every advance.
3. **Views never feed physics.** A materialized dense lattice may exist for
   legacy ABI consumers, but nothing in transport, redistance, volume,
   band membership, ghosts, or refinement evidence may read it.

## Why every previous increment failed — the actual difficulty

The lane has **three** surface representations and the dense one is
authoritative:

| Representation | Owner | Status |
|---|---|---|
| Dense finest-lattice phi (3 banks + 2 activity masks) | `WebGPUOctreeCoarseSummary` (`lib/webgpu-octree-coarse-summary.ts:155-162`) | **Sole authority** |
| Compact rows `{phi,min,max,flags}` + gradient | `WebGPUOctreeLosassoCoarsePhiExchange` | Derived view (resampled from dense every advance) |
| Adaptive nodes + unique leaf edges | `WebGPUOctreeAdaptiveNodes` (`lib/webgpu-octree-adaptive-nodes.ts`) | Published, counted — **consumed by nothing** |

`grep source.nodes|source.edges` across `lib/ tools/ tests/` finds only the
publisher and the CPU receipt reader. The adaptive graph is a census, not a
field.

The rejected experiments, restated as violations of principle 2:

- **Adaptive transport + dense redistance**: adaptive→dense→adaptive
  remapping every advance is a low-pass filter; dense redistance regenerated
  finest-scale structure and became authoritative again. Volume collapsed,
  normal profile rose.
- **Persistent nodes without adaptive redistance**: transport, redistance and
  volume control are one coupled loop; removing dense redistance without an
  edge-based replacement broke both lanes.
- **Coarse nodal velocity + dense phi**: same — dense redistance undid the
  coarse transport (ridge 0.3195 → 0.4601).
- **Post-hoc affine projection**: presentation-only smoothing made compact
  rows, air-side samples and the internal dense field disagree (ridge 0.82).
- **Interior-owner staging shortcut**: a dispatch optimization; rejected on
  principle even though normal stayed exact.

Conclusion, which the rest of this doc operationalizes: **the authority flip
is atomic** — transport, redistance, volume control, row/ghost/band/evidence
derivation all switch to the adaptive graph in one commit. Components are
developed and validated in *shadow* first, where they can be gated
byte-identically, then flipped together under a re-bless protocol (§ Gates),
then the dense machinery is structurally retired.

## Machinery map (verified at HEAD, file:line)

### The dense pipeline that must die

`WebGPUOctreeCoarseSummary.encode` (`lib/webgpu-octree-coarse-summary.ts:354-463`),
run at the top of every advance (`lib/webgpu-octree.ts:5546-5548`). Every
pass below dispatches over `nx·ny·nz` finest cells (via
`air.layout.ownerDirectoryCellCapacity`) regardless of topology:

| Pass | WGSL | Role |
|---|---|---|
| `predictSummaryCells` | :823 | RK2 semi-Lagrangian phi transport per finest cell, nodal-velocity trilinear |
| `seedDenseRedistance` | :883 | seeds bank 2: ±h·max(dims) sentinel, 0 at zero, sub-cell crossing at sign flips |
| `redistanceScratchToOutput`/`OutputToScratch` ×N | :901-931 | Eikonal upwind Jacobi sweeps, N = `planOctreeRedistanceSweeps` (:126) |
| `summarizeDenseVolume` | :953 | fixed-point occupancy Σ round(4096·clamp(.5−φ/h,0,1)) |
| `prepareVolumeCorrection` | :960 | correction scalar → state[15], clamp ±0.5h |
| `correctAndAggregateSummaryCells` | :973 | applies correction in directional halo; scatters min/max into the B4 hierarchy that feeds refinement evidence |
| `publishDenseComplement` | :1029 | writes the dense tail `entries[rowCapacity+cell]` of `volumeDirectory`, sets generation bit 0x40000000 |
| `correctCoarseDirectory` | :1046 | overwrites compact-row phi/min/max from the dense banks (size³ loop per row) |

Plus rigid displacement measurement (domain-sized) and `publishSummary`'s
bank flip gated on `state[18] == domainVolume` — the completeness-receipt
pattern the adaptive publication must inherit.

### The staging pipeline that must die

`WebGPUOctreeLosassoExtensionBand` stages the sparse compact W7 band into
dense arenas so dense consumers can index without a hash probe:

| Pass | Dispatch | Measured (normal / 4³) |
|---|---|---:|
| `stageLosassoVelocityLattice` | full finest MAC lattice | 0.67 / **1.40 ms** |
| `stageLosassoPredictorVelocityLattice` | full finest MAC lattice | 0.63 / **1.18 ms** |
| `stageLosassoNodalVelocityLattice` | full nodal lattice | 0.05 / 0.05 ms |
| `stageLosassoPredictorNodalVelocityLattice` | full nodal lattice | 0.05 / 0.05 ms |

(`lib/webgpu-octree-losasso-extension-band.ts:497-582`; WGSL :299-342.)
Coarsening makes staging **more** expensive per face: a finest-MAC invocation
inside a coarse leaf falls through `containingCompact` into the
`stagedOwnerVelocity` / `stagedNearbyOwningLeafVelocity` hierarchy search
(WGSL :167-262). Also domain-sized: `clearLosassoStagedCoarseOwners` and
`publishLosassoCoarseAirBandFaces` (all finest MAC faces, :422-437).

Nodal velocity today (`stagedNodalSample`, WGSL :281-297) averages the four
finest quadrant MAC faces per axis from the staged raw bank — it is a
finest-scale construction and *depends on* the expensive MAC staging. The
paper's Section 3 coarsest-adjacent-face construction is not yet implemented.

### The adaptive graph that exists

`lib/webgpu-octree-adaptive-nodes.ts` — five passes, encoded first every
tracker advance (`webgpu-octree-coarse-summary.ts:356`):

- **Node identity is already stable and geometric**: a node *is* its linear
  index in the `(nx+1)(ny+1)(nz+1)` corner lattice (:244-251). Published iff
  it is a corner of ≥1 resident owner leaf (:252-277). This is exactly the
  identity a persistent phi bank needs.
- **Edges** are `vec4u(nodeItemA, nodeItemB, span, axis)` (:302-346),
  positive-axis, low-node-claimed, deduped by a per-node `(axis, level)`
  bitmask; **different-span edges at the same node/axis are deliberately both
  published** — that is the hanging-node structure, already encoded.
- Control block validated against the owner generation
  (`nodeControl[1] == ownerPageArena[7]`, fail-closed, :280-300) — the right
  epoch hook.
- Gaps: compaction slot order is a racing `atomicAdd` (slot ≠ stable), there
  is no item→slot inverse, everything is rebuilt each advance, and
  `publishAdaptiveNodes` itself dispatches over the **full** node lattice
  with 8 owner-page probes per node — domain-sized work (only
  `publishAdaptiveEdges` is adaptive-sized, indirect from the node count).

### Who reads phi today (the flip's blast radius)

Dense-complement readers **upstream of physics** — these three must move to
the adaptive graph *in the flip commit*, or the view becomes authoritative
again:

1. `advectLosassoCoarsePhi` → `trackerPhi` (compact-row transport input,
   `lib/webgpu-octree-losasso-coarse-phi.wgsl.ts:63-104`).
2. `publishLosassoCoarseAirBandFaces` (extension-band membership, gates
   velocity extension, `extension-band.wgsl.ts:411-437`).
3. `publishLosassoCoarseOnlyGhosts` → `trackerValue`/`trackerRawValue`
   (air-side ghost sample and rigid-carve discrimination → the pressure
   matrix, `coarse-phi.wgsl.ts:121-151`).

Refinement evidence reads the B4 hierarchy built by
`correctAndAggregateSummaryCells` (via `fineLeafSummary`,
`lib/webgpu-octree.ts:8874-8960`) and the compact arena (via
`correctedCoarsePhi`, :7814-7854) — hierarchy aggregation must be re-fed
from the adaptive field at the flip.

Pure-view readers (may keep reading a materialized tail): renderer coarse-1
classifier (`lib/webgpu-water-global-fine-classify.ts:175-201` →
`sampleCoarseOctreePhi`, which already has a sparse-row-walk fallback at
`lib/webgpu-octree-power-coarse-levelset.ts:525-526`), probe readbacks
(`tools/probe-dam-surface-shape.ts:202-230`), QA volume field
(`tools/webgpu-smoke-readbacks.ts:2756-2837`).

### Handoff precedents already in the tree

- **`WebGPUOctreeLosassoVelocityMigration`**
  (`lib/webgpu-octree-losasso-velocity-migration.wgsl.ts`) — the closest
  analogue for node-phi topology handoff: identity-hashed old snapshot,
  dyadic-ancestor walk (`containingOld`), and refinement reconstruction from
  the old parent (`coarsenedOld`) instead of manufacturing zeros; fails
  closed on absent publication.
- **`warmLosassoCoarsePhi`** (`coarse-phi.wgsl.ts:388-412`) — (cell,size)
  hash carry across the double-buffered arena. Note `priorRow`/`priorPhi`/
  `seedPhi` in `octreeLosassoCoarseOnlyPhiWGSL` (:48-62) are **declared but
  never called** — dead remnants of the reverted persistent-transport
  experiment, and exactly the identity-keyed prior lookup a node field needs.
- **Dense bank flip** (`coarse-summary.ts:1006-1022`) — completeness must be
  part of the receipt (`state[18] == domainVolume` before flipping), or
  consumers cannot distinguish a fresh surface from a held one.
- **Row delta map** (`webgpu-octree.ts:10391-10432`) and fine page carry
  (`webgpu-octree-fine-levelset-topology.ts:2535-2557`) for the general
  identity-across-topology discipline.

## The paper's method, mapped to this codebase

Losasso 2004 stores **all scalars including phi at nodes** (§3). That is not
an implementation detail — it is what makes the method adaptive: nodes and
edges are shared across leaf scales, so one field serves every resolution
without remapping.

| Paper element | Codebase target |
|---|---|
| phi at octree nodes | persistent bank indexed by the existing lattice-item node identity |
| Nodal velocity at coarsest-adjacent-face scale (§3) | new per-published-node construction reading the compact face directory directly (replaces `stagedNodalSample` + MAC staging) |
| T-junction constraints: edge nodes lerped from edge endpoints, face nodes averaged from 4 corners (§3) | new constraint pass driven by the multi-span edge records |
| Refinement: new edge node = avg of 2 neighbors, face node = avg of 4 corners; coarsening: nodal values unchanged/deleted (§3) | node-phi topology handoff pass (velocity-migration pattern) |
| SL advect phi at nodes with nodal velocities (§6) | node-parallel transport pass over published nodes |
| Fast marching on the adaptive structure; missing T-junction directions trivially ignored because the mesh coarsens away from the interface (§6) | iterated min-only upwind over the published edge graph, per-axis edge lengths = span·h |
| Velocity extrapolation: extrapolate **nodal** velocities, then compute face velocities; skip unreachable nodes until a neighbor updates (§6) | replaces the span-1 air-band face records and the Jacobi face extension |
| Particle level set for mass (§6) | not present on this lane; the localized volume corrector remains the volume authority |

## Target architecture

### Data

- `nodePhi`: one buffer, two banks (A/B) indexed by **node lattice item**
  (not compaction slot — slots race). Per-item validity via a
  generation-stamped word (no clears; compare stamp ==
  `ownerPageArena[7]` generation). Memory is node-lattice-sized (capacity,
  not work — capacity is cheap; work must be adaptive-sized).
- `nodeVelocity` (and extension flags): same indexing, one bank, rebuilt per
  advance from faces.
- The existing `nodes`/`edges` compact lists drive **dispatch** (indirect
  from the counts); item-indexed banks make an inverse map unnecessary.
- Constraint classification per published node: independent | edge-hanging
  (slaved to 2 nodes) | face-hanging (slaved to 4 nodes), derivable from the
  8 owner-page probes plus the multi-span edge structure.

### Per-advance surface update (all dispatches ∝ published counts)

1. **Handoff** (only when owner generation changed): for each published
   node, reuse the prior-generation value at the same item if stamped valid;
   otherwise interpolate from the prior accepted containing leaf (multilinear
   over its 8 corner items — the `coarsenedOld` pattern). Stamp generation.
2. **Nodal velocity**: per published node, per axis, gather incident faces
   from the compact face directory at the coarsest adjacent face scale (§3);
   **fail open through the coarse-face hierarchy, never closed to zero** —
   valid-zero fabrication is the root cause of both the memoryless-coarse
   defect and the refinement-region freeze (see Contracts).
3. **Transport**: RK2 semi-Lagrangian per published *independent* node;
   departure-point sampling = containing-leaf multilinear of bank-A node phi;
   closed-wall departure clamp with the +exitDistance outward-slope contract.
   Write bank B.
4. **Constraints**: slave hanging nodes (edge-lerp, face-average). Paper
   constrains all variables; re-apply after each field-mutating stage.
5. **Redistance**: seed per edge — sign change along an edge yields the
   sub-edge crossing distance `|φa|/(|φa|+|φb|)·span·h` at both endpoints;
   non-interface nodes seed the ±(reach sentinel). Then iterated min-only
   upwind sweeps over per-axis edge neighbors (Eikonal quadratic with
   per-axis h = span·h; a missing axis is ignored, per the paper). Sweep
   count derived from the protection width exactly as
   `planOctreeRedistanceSweeps` does (odd, ping-pong, floored) — the reach
   contract is unchanged, only the iteration domain shrinks from `nx·ny·nz`
   cells to published nodes. Seed from the sentinel, **never** from the
   advected magnitude (measured regression: min-only sweeps decay a
   compressed field; interior coarsening bled 1192→128 size-2 leaves).
   Closed walls contribute virtual-air samples at node+h with unit outward
   slope (openTop exempt) so a film flush against a wall is representable.
6. **Volume**: measure occupancy per accepted row from its corner node phi
   (fixed-point integer atomics weighted by size³ — order-independent by
   construction, which is what the D4 exact-volume gate requires; watch u32
   overflow at ≥2.4M-cell domains — split hi/lo words). Correct node phi in
   the directional interface halo. Fix the known counted-set ≠ applied-set
   overshoot: count with the same predicate that applies
   (`coarse-summary.ts:957-959` vs :980). Keep mode 1 (directional,
   localized); the global uniform offset stays dead (it regrows wall films).
7. **Derivation** (same commit as the flip):
   - Rows: row phi = leaf-center multilinear of 8 corners; interval =
     min/max over the 8 corners — **exact** for a multilinear field, and it
     structurally replaces today's size³ dense scan; gradient = multilinear
     gradient at center (replaces `trackerPhi`'s ±span central difference).
   - Ghosts: air-side sample = containing-leaf multilinear of node phi;
     `theta`/dual arithmetic unchanged. Rigid carve discrimination
     (today's `trackerRawValue`) needs a carved/raw pair — carry a raw copy
     through the corrector as the dense path does.
   - Refinement evidence: aggregate per-leaf corner min/max into the B4
     hierarchy entries (work ∝ leaves·levels, replaces the dense
     per-cell scatter).
   - Band membership: interface rows (corner interval straddling zero) emit
     their leaf-scale faces — replaces the all-finest-MAC-faces enumeration
     *and* the span-1 air records (see step 8).
8. **Extension** (paper §6, replaces the air-band + Jacobi face extension):
   extrapolate nodal velocities outward over the edge graph in phi order
   (iterated upwind, same sweep machinery as redistance; unreachable nodes
   wait, per the paper), then compute air-band face velocities as corner
   averages **at each face's own leaf scale**. This deletes the span-1
   valid-zero records that shadow the coarse reconstruction — the mechanism
   behind the refinement-region freeze — instead of patching them.
9. **View materialization** (legacy ABI only): one pass writes the dense
   complement tail from node phi for renderer/probe/QA readers. Physics
   never reads it (enforceable: after the flip, the only shaders binding the
   tail are the classifier and probes). Later (WP5) the renderer's
   already-present sparse-row fallback can retire the view entirely.

### Publication becomes row-driven (structural, WP4)

Replace the lattice-sweep `publishAdaptiveNodes` with emission from accepted
rows: each row emits its 8 corners; first-claim via a generation-stamped
per-item claim word (no clearing pass — stale stamps are simply not current).
Work ∝ 8·rows (24 rows on the 4³ tank vs 6,912+ lattice probes today). Edges
stay low-node-claimed as now. The census oracle
(`censusOctreeTopologyLeaves`, `lib/webgpu-octree.ts:701-775`) already
validates counts either way.

## Gates and the re-bless protocol

The byte-identity gate cannot survive the flip: node-centred phi is a
different discretization from cell-centred phi, so the normal fingerprint
*will* change when the authority changes. Pretending otherwise is what makes
every increment look like a regression. The protocol:

**Shadow phase (WP1–WP3): byte-identity holds.** Shadow passes write only
adaptive buffers and receipts; nothing consumed changes. Gate every
increment on:
1. Normal dam break at 0.248 s: complete profile fingerprint byte-identical
   (table below).
2. Shadow comparison receipts: max/RMS |nodePhi − densePhi(node)| within the
   protection band, published per advance in the probe JSON. Convergence
   criterion for flip readiness: bounded and non-growing over the run on
   normal, full-tank 4³, and mini dam.
3. `npm run test:webgpu:symmetric-expansion:one-step` passes and the shadow
   receipts themselves are D4-symmetric (reflected node pairs bit-equal —
   this is where canonical weights / fixed-point folds get proven, cheaply).

**Flip commit (WP3): re-bless.** Byte-identity is replaced, for that one
commit, by quality equivalence on the normal lane:
- surface-profile delta vs fingerprint bounded (propose: max |Δ| ≤ 0.15
  cells, no systematic rise of the reservoir mean — the rejected experiments
  failed at ~2 cells and ~35 lost liquid cells, so this bar has real teeth);
- volume drift over 0.248 s within the current lane's envelope;
- `tools/benchmark-interface-band-adaptivity.ts` gate still passes;
- free-fall oracles (`ceiling-slab-drop`, `corner-brick-drop`) not worse —
  these pin the wall contracts the new redistance/transport must carry;
- symmetric-expansion one-step: volume and diagonal still **exactly** 0;
- full-tank 4³: ridge ≤ 0.3195 cells (must not be worse than baseline) and
  dense-pass census shows the dense pipeline no longer encoded.
The new normal fingerprint is then recorded and byte-identity resumes for
all subsequent work.

**Never gate on ridge or volume alone** — several rejected experiments held
one summary metric while visibly changing the profile.

## Work packages

### WP0 — hygiene (do first, trivial)

- `tests/octree-fluid-gated-boundaries.test.ts:146`: expected literal needs
  `topologyNodes: 8` (census gained the field in 9a5821d; fails today).
- `tests/webgpu-octree-losasso-regressions.test.ts:17-19`: regex still
  expects `trackerPhi\(centre\)`; the uncommitted span change breaks it
  (fails today). Update to match `trackerPhi\(centre,header\.size\)`.
- The surviving `trackerPhi(position, span)` increment has a wall bug: it
  probes `centre ± span` but `trackerValue` clamps to the lattice, so near a
  wall the two probes sit at asymmetric distances while the divisor stays
  `radius` — biased gradients exactly where wall behavior is fragile. Divide
  by the actual clamped high−low distance instead. (Also note the stencil is
  a full row-width beyond each face; the half-width `±span/2` variant stays
  inside the row's own support and should be A/B'd in shadow.)
- Delete the dead bindings 13/14/16 wiring in
  `lib/webgpu-octree-losasso-dynamics.ts:172-174` left by c75e023.

### WP1 — graph foundations (shadow, byte-identical)

- Persistent `nodePhi` A/B banks + generation stamps, item-indexed.
- Hanging-node classification + constraint pass (edge-lerp / face-average),
  with a receipt counter: max constraint residual must be exactly 0 after
  the pass, and the count of each class published to the probe.
- Handoff pass (reuse-or-interpolate-from-prior-leaf) on owner-generation
  change.
- Seed bank A once from the dense field (bootstrap only; delete at WP5).
- Row-driven publication can land here or at WP4 — it is independent of the
  authority question.

### WP2 — shadow physics (shadow, byte-identical)

- Nodal velocity at coarsest-adjacent-face scale, fail-open.
- Node SL transport with the wall departure contract.
- Edge redistance with virtual-air walls and reach-derived odd sweep count.
- Leaf volume measurement + localized correction applied to node phi.
- All of it dead-ends in comparison receipts. Iterate here until the WP3
  readiness criteria hold on normal, full-tank 4³, mini dam, and the D4 lane.
  This is where the real numerical debugging happens, at zero risk.

### WP3 — the authority flip (one commit)

Switch, together: row derivation (`advectLosassoCoarsePhi` becomes
node-sampling or is replaced), ghost air-side sampling, band membership,
refinement-evidence aggregation, volume authority; stop encoding
`predictSummaryCells`, `seedDenseRedistance`, the sweeps,
`summarizeDenseVolume`/`prepareVolumeCorrection`/`correctAndAggregate…`,
`correctCoarseDirectory`; add the view-materialization pass for the tail.
Re-bless per protocol. Keep the dense code compiled but unencoded for one
commit (fast revert), then delete.

### WP4 — structural retirement of the staging complex

- Nodal velocity now comes from WP2's construction → retire both finest-MAC
  staging passes and both nodal staging passes; face advection
  (`advectLosassoFaces` binding 15) reads the node-velocity bank directly.
- Replace `publishLosassoCoarseAirBandFaces` + Jacobi face extension with
  §6 nodal extrapolation over the edge graph + leaf-scale corner-average
  faces. This is also the fix for the refinement-region freeze (span-1
  valid-zero shadowing) — verify with the drawn-region scene that froze.
- Row-driven node publication (if not landed at WP1).
- Free the three dense phi banks, activity masks, and staged arenas.

### WP5 — acceptance and view retirement

- Full-tank 4³ targets: surface passes' items ∝ 357 nodes / 935 edges / 178
  leaves (not 6,912 cells / 21k MAC faces); ridge materially below 0.3195;
  end-to-end advance time reflecting the ~16× topology shrink.
- Pass timestamps only after all semantic gates pass.
- Optionally retire the dense view: renderer falls to the sparse row walk
  (`sampleCoarseOctreePhi` fallback) — needs a quality look first.
- Port rigid-coupling reads (displacement measurement, carve) onto the
  adaptive field before enabling on rigid scenes; until then gate those
  scenes on the old path (this is the one temporary conditioned path, and it
  is scene-scoped, not physics-scoped).

## Contracts checklist (each has burned us before)

- **Fail open, never closed-to-zero**: any velocity lookup that returns
  valid-zero for a plane/node it cannot resolve turns the coarse interior
  into memoryless hydrostatics (band-width diagnosis) and freezes held
  regions (refinement-region freeze). Total nodal reconstruction or explicit
  invalid — never fabricated zero.
- **Wall contracts** (the coarse lane never received the fine lane's July
  fixes): outward-slope departure clamp; virtual-air redistance samples
  across closed walls at +h; lid-separation hysteresis is solver-side and
  unaffected, but validate with the free-fall 2×2 oracles, not dam metrics.
- **Reach ≥ every consumer's read distance**: refinement evidence reads |phi|
  out to `bandCells + (gradingLayers−1)·maxLeafSize` cells; derive sweep
  counts from that width (the five-sweep constant made the ladder read a
  constant and coarsening decayed).
- **Min-only monotone**: seed sentinels, never advected magnitudes.
- **Volume**: localized directional correction only; corrector's counted set
  must equal its applied set; target volume semantics (`state[14]`/`[36]`)
  carried over; rigid displacement is measured but must not bias the target.
- **Symmetry**: all reductions via order-independent folds (fixed-point
  integer atomics or the exact-accumulator patterns); per-node work keyed by
  lattice item, never by racing compaction slot; quantize interpolation
  weights as `quantize()`/`canonicalWeight` do. The D4 lane demands volume
  and diagonal **exactly** 0.
- **Epoch discipline**: every adaptive publication generation-stamped and
  fail-closed (`nodeControl[1] == ownerPageArena[7]` pattern), with a
  completeness receipt (published count == expected) so consumers can tell
  fresh from held.
- **Binding budget**: 10 storage buffers per stage. `predictSummaryCells`
  sat at exactly 10 before c75e023 freed four. Budget each new pass on
  paper first; fold banks into one buffer with offsets (dense banks and the
  nodal arena both already do this).
- **Leaf ≤ tile** (`octreeLosassoLeafCeiling`): unchanged, but any new
  per-tile pass inherits the invariant.
- **A leaf's row identity is (cell,size)** — reuse the delta map and
  `warmLosassoCoarsePhi` discipline for anything row-keyed; never rotate a
  published buffer object (`coarse-phi.ts:301-307`, pinned by tests).

## What NOT to do

- No dispatch shaving, indirect-dispatch conversions, or per-invocation
  shortcuts as ends in themselves (interior-owner experiment: rejected).
- No presentation-only smoothing of coarse rows (affine projection:
  rejected — views must agree because they derive from one authority, not
  because they are individually patched).
- No partial authority switches (three rejected experiments).
- No re-enabling the global uniform volume offset.
- No judging coarse-lane physics on symmetric-expansion alone — its D4 reds
  at steps 9/34 are pre-existing, and its water is too shallow for the deep
  coarsening paths to fire.

## Reproduction

Normal:

    FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=0 WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/probe-dam-surface-shape.ts

Full-tank 4³:

    FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=4 FLUID_REFINEMENT_REGION_SCOPE=full WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/probe-dam-surface-shape.ts

Pass attribution (only after semantic gates):

    FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS=8 FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS=8 FLUID_REFINEMENT_REGION_FLOOR=4 FLUID_REFINEMENT_REGION_SCOPE=full WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx tools/benchmark-power-dam.ts --lane=ui --steps=20 --pass-timestamps --isolate-pass-labels

Symmetry:

    npm run test:webgpu:symmetric-expansion:one-step

Census lanes: `FLUID_EXTENSION_BAND_CENSUS=1`, `FLUID_STAGED_OWNER_CENSUS=1`
on the probe; adaptive receipt fields (`nodeCount`, `edgeCount`, errors,
published flags) land in the probe JSON per sample.

Do not run Dawn concurrently with browser WebGPU. An interrupted isolated
run can leave `/tmp/fluid-webgpu-exclusive.lock`; verify the recorded owner
PID is dead before removing that exact stale lock.

## Baseline at 0.248 s (unchanged from prior doc)

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

Normal strict surface-profile fingerprint (byte-for-byte through WP2):

    7.9741, 7.9557, 7.9185, 7.8612, 7.7853, 7.6994,
    8.0649, 7.9334, 7.7714, 7.5591, 7.7536, 7.5016,
    5.7620, 3.6283, 2.7147, 2.1398, 1.6887, 1.3762,
    0.9203, null, null, null, null, null

Full-tank 4³ baseline begins:

    9.1532, 9.1217, 9.0616, 8.9857, 8.9052, 9.4727,
    9.4008, 9.2669, 9.0518, 8.8476, 9.3373, 8.7063,
    7.0446, 4.3371, 3.2299, 2.2426, 1.0072, ...

## Open questions (decide during WP2, not before)

1. Sweep-count law on the graph: near-interface edges are span-4 on the
   coarse tank, so each sweep covers 4 cells of reach — the finest-cell
   formula is a safe over-count; measure whether a span-aware count matters
   (it should not, node passes are tiny).
2. Occupancy from corner phi: leaf-center clamp vs 8-octant subsampling —
   pick by volume-drift comparison against the dense oracle in shadow.
3. `±span` vs `±span/2` gradient stencil (WP0 note) — resolve in shadow.
4. Whether the renderer's sparse-row fallback is visually acceptable at 4³
   (decides if the view pass survives WP5).
5. How MacCormack's predictor interacts with node transport — the paper is
   first-order SL only; the lane's literature note (Selle 2008 via
   Ando-Batty) argues against MacCormack near liquid surfaces anyway.
   Proposal: flip with first-order SL on nodes; treat any sharpness loss as
   a separate, later axis.
