# Domain-less narrow-band FLIP: remaining work

Written 14 October 2026. Companion to `domainless-nbflip-2026-10-13.md`, which
holds the design and the record of what has landed. This file is the work
list: what is ready, what gates it, and the order.

Correctness comes first. The performance push starts when Peter confirms the
picture is right.

## Where things stand

Committed as `26c9efb9` on 14 October (everything through the trilinear far
value and the open-top headroom), then, at the Z1 gate, the box deletion and
the pipeline compile fix (commit after `26c9efb9`; Peter: commit at each
passing gate without asking). The other session's edits stay out of both. Checked
(type-check clean, unit suite green, app responds):

- Page-keyed primitives, the census, the solid page directory.
- Container against lattice: a scene that is not whole 32-cell pages is
  solid-padded on its high sides.
- Pressure coarse levels keep enclosed air (the settled padded tank holds).
- The thin-film pressure fix and the bed-slope lane.
- Dam break: side-face dents fixed, top-edge strips fixed, foot terraces
  halved at steps 6 to 8.
- **Step-7 bundle, landed 14 October:** residency, velocity extension, wall
  reach, particle clamps, pressure root rows, wall slots and the solid record
  are keyed by pool block or tile key. Type-check clean, unit suite 887 of
  887. On the device: settled padded tank holds (liquid 3072 over 30 steps),
  water slide grows 18 to 54 pages over 60 steps with no fatal, trough clean
  to step 55.
- **Known regression from that bundle, cause found:** the trough dam break
  stops at step 55 to 57 (about 0.95 s), as liquid reaches the top tile row
  under the open top. The tree before the bundle ran on (it deleted liquid
  at the top plane). Cause, shown on the device: the address helpers clamp
  to the box. A tile stored above the open top is read and written through
  the top row's address, so the velocity extension published a slot nothing
  had written (−67,717.5, a stage's scratch value). The same clamp is in
  about 125 places across 40 files (cell, tile and vertex addresses, the
  detail fields' loads and stores). Storing anything outside the box is
  unsound until those are gone, which is item 6. See item 1c for what was
  done instead.
- **h band halo by block, landed 14 October** (`uniform-pressure-band.ts`).
  Gate: settled tank liquid 3072 at every step, water slide 8,183 at step 60
  (18 to 54 pages), trough 64,233 at step 20 and 115,901 at step 40 (the
  bundle alone: 64,255 and 115,869).
- **Pressure levels by page, landed 14 October** (type-check clean, unit
  suite 886 of 886, app responds)**:** the 8h, 16h and 32h
  levels and the coarsest solve are keyed by block. Its first device run
  failed (pressure not converging at frame 0); cause: the tiled smoother read
  the page table at the block records' offset, so those jobs stored nothing.
  Fixed, then held launch by launch against a host reference on the device
  (trough 85 of 85 launches; letters 127 of 128, the last a residual at
  rounding level). Gate: settled tank 3072 throughout, water slide 8,168 at
  step 60, letters 12 steps clean.

- **Trilinear far value, landed 14 October** (`uniform-mixed-extension.ts`,
  `uniform-mixed-extension-pair.wgsl.ts`, the host twin, two probes): a face
  no sweep reached takes the trilinear combination of the known 4h cells
  around it instead of the nearest source box. No pass, binding or pipeline
  added; one barrier and 27 staged cells where there were two and 81. Gate
  (scratch tree, the same bytes): kernel probe 0 failures; settled tank
  3072; trough liquid 31,868 at step 40; water slide 8,194 at step 60; dam
  break 24 steps, no stop. In the working tree: type-check clean, unit suite
  890 of 890, app responds. `tools/probe-uniform-extension-publish.ts` had
  its anchors repaired. `tools/probe-extension-direct-snapshot.ts` is left
  as it was: it already stopped at its first anchor before this change
  (names removed by paging) and three other tools import from it.

Built in scratch trees, not yet in the working tree:

| Package | What it does | State |
|---|---|---|
| Renderer window list | Surface windows launched over pool blocks | Host model and kernel probe pass. Its agent was stopped before the full-against-windows extraction check finished; not restarted |

## How the work is run

From the afternoon of 14 October: one agent does all the remaining work,
serially, straight in the working tree, in larger increments with one
type-check and one short single-scene device run each (Peter's instruction;
the four-scene gate runs once, after the box is deleted). No parallel packages, no scratch
trees, no landing step. Its order: land the trilinear far value; the dam
break (item 3); delete the box (item 6); the trough's liquid total
(item 1c). It then stops for Peter. The rules below were the morning's and
are kept for the record.

The device was shared by five jobs at once this morning and every one of them
waited. From here:

1. **One device queue, run by the coordinator.** Agents do not take the GPU
   lock. They verify on the CPU (host models, type-check, shader validation)
   and hand back the exact device commands they want run.
2. **One gate per landing, one hold of the lock.** Each package lands on a
   single short batch: trough, settled tank, water slide, letters, basin. No
   repeats unless a number is off.
3. **Two exceptions, one at a time:** the two dam-break investigations need
   the device to find anything. They run short runs and never together.
4. **Land on the gate, not on completeness.** A package that passes its gate
   lands the same hour. Follow-ups go on this list, not into the package.

## Order of work

### 0. Pipeline build speed (first, 14 October)

Peter: "make pipelines fast". The letters build takes 255 s on HEAD and
1,320 s on the Z1 tree. The time is Apple's Metal backend, not Tint and
not robustness (advect alone: 94 s, 91 s without robustness). The cause is
inlining fan-out of the velocity sampler. `umSampleVelocity` inlines
`umSampleVelocity1` 9 times (3 axes × the weighted path's two call sites
plus one). Advect's 4 traces × 2 samples bring that to 144 copies; momentum
repeats the pattern ×4. Marking only `umSampleVelocity1` noinline in the
dumped Metal source takes advect from 98 s to 3.1 s. The fix: one call site
per heavy function (loops over axes, traces and stages), and the same for
Z1's holds. Target: no pipeline over about 5 s, and a letters build under a
minute. Tools in the scratchpad: `pipes/one.mjs` (compile or dump one
module), `pipes/swift/mtlc` (time Metal's backend on Metal source) and
`pipes/callgraph.py` (inlined copies per function).

### 1. Land the step-7 bundle (stack + pressure root): done

- Landed. Gate: settled padded tank and water slide pass; trough fails at
  step 56 (see above).
- Three unit test files or cases were deleted whole, because what they
  tested is gone: `uniform-mixed-root-owners.test.ts`,
  `uniform-pressure-compilation.test.ts`, and the certificate-source case in
  `uniform-page-pool.test.ts`.

### 1b. Land the pressure levels by page: done

- Gate passed (above). Deletes `uniform-pressure-plan.test.ts` (its subject
  is removed). Adds `tools/probe-root-levels.ts`, which holds every pressure
  launch to a host reference on a scene's own pool.
- Not yet shown on a device: these levels solving liquid above an open top.
  The trough stops before any tile above the brim is wet (item 1c).

### 1c. The trough's stop at step 56: open tops get headroom inside the lattice

- Cause: box clamps in the address helpers (above). A patch for the one
  site that produced the bad value exists and runs the trough clean to step
  56, where the pool is then exhausted: the residency certificate asks for a
  second layer of pages above the box while the liquid is still below the
  brim. Patching sites one at a time is the wrong shape; the patch is kept
  as material for item 6.
- **Done instead (in the working tree, type-check clean, unit suite 886 of
  886):** a scene with an open top gets 64 cells of air above the container
  inside its lattice (`LATTICE_OPEN_TOP_HEADROOM_CELLS`), and the lattice
  itself is a closed box. Nothing is stored outside the box. Only the trough
  family has an open top; every other scene is unchanged.
- What this means for the picture: liquid that leaves the open top rises
  between invisible walls and meets a ceiling 64 cells up. The trough's
  splash is expected to stay well under that.
- Gate passed: trough 120 steps at 1/60 s, no stop, liquid finite
  throughout. The lattice is 4x4x2 pages (was 4x2x2), pool 34 blocks, at
  most 28 pages resident after the seed, peak 3,038 h tiles. Memory held
  rises from 464 to 554 MiB.
- Seen in that run, not yet looked at: the liquid total climbs, +2.2 % at
  step 40, +3.3 % at step 60, +5.3 % at step 120. The first 40 steps match
  the runs before headroom, so it is not the headroom. It needs a run at
  1/30 s beside this one before it is called a volume gain.
- **Decision for Peter:** this is an interim. The alternatives are a named
  stop when liquid reaches the lattice top, an outflow there, or true growth
  above the box (item 6 then item 6's open boundary). Say if 64 cells or the
  ceiling is the wrong choice.

### 2. Land the renderer window list: paused

- The agent running its last check was stopped; I have not restarted it.
  The patch exists (`k9/patch.py` in the scratchpad) and needs re-cutting
  against the working tree, then the check below.
- Gate: the extraction tool shows the full scan and the window list give the
  same triangles on the trough, water slide, dam break and a padded tank, in
  both the patched and unpatched trees; and its count of h tiles beside
  unstored pages settles the one open assumption.
- Also turns a silent mesh-capacity overflow into a loud error.
- Frame time and the drawn picture are Peter's to check after landing.

### 3. Dam break (`high-resolution-dam-break`)

- **Fatal at about frame 17** ("closure of 17 tiles, over the cap of 16"):
  traced (`tools/probe-nbflip-speed.ts`, now in the tree). No kernel is
  wrong. The value is a real face velocity: after the front hits the far
  walls the h pressure band puts 25 to 47 cells a step on liquid faces that
  look into air beside liquid at impact pressure, and the corner jet is about
  16 m/s at every time step tried, which is 45 cells a step at 1/60 s. The
  residency cap is 44. The solver before paging made the same speeds (60
  cells a step at step 16) and had no cap, so it ran on. Halving the step
  gives a largest closure of 11 and cuts the liquid loss from 4.1 % to
  0.9 %. **Done:** the residency reach is 8 pages instead of 4 (one
  constant; no work added while speeds are normal), which restores what the
  scene did before paging. At the app's 1/60 s it now runs 30 steps with no
  stop (liquid 5.8 % down at step 24, recovering to 4.2 % at step 30). At
  1/30 s, which the app does not use, it loses 12.7 % by step 10 and goes
  non-finite at step 11. Still open: the one real defect the trace found, a
  one-cell enclosed air pocket taking in 124 cell volumes in a step.
- **Foot terraces, investigated:** the terrace is in rows that are already
  h, not in 4h rows. A foot tile that is h before the flow starts stays
  clean; one that turns h while the flow is already deforming it is level
  with the all-h reference at that step and 3.6 to 3.8 times rougher four
  steps later. No refinement profile within shape tolerance 0.5 fixes it,
  with up to 76 % more h tiles. One change is ready: the far value read
  trilinearly instead of from the nearest coarse source (better on both
  arms at steps 11 to 12, 5 to 15 % worse on the default arm at steps 7 to
  10, and cheaper: 27 cells and one barrier against 81 and two). It lands
  on its own gate. Next agent (5A): why a tile activated mid-flow ends
  ragged. Two candidates: the velocity a sample is seeded with at
  activation, and the add-only union of samples and grid surface ahead of a
  front crossing more than a tile a step. "Shape tolerance 0" is not the
  fix: the 3 October decision was to allow 4h at the surface and fix what
  it exposes.
  **14 October, 5A: not fixed, cause narrowed.** Thirteen sample-kernel
  experiments (FLIP part off, outer layer every step or never, collar
  erasing, collar hot, a hot ring with a collar beyond it, phi carried by
  its foot out to 8 cells) leave the default at 0.47 to 1.43 against 0.63;
  the reference is 0.23. Ruled out on its own: FLIP noise, the outer layer,
  collar-versus-hot authority, the air extension in the phi advect. Left:
  the reference's samples are seeded at rest and carried, the default's are
  seeded mid-flow from a phi the advect has roughened and then hold that
  shape. Evidence and table: `$S/reports/solo-dambreak.md`.
- **A 32-cell component of the run-out sheet: not paging** (14 October).
  The tree before paging gives the same (locked32 0.156 against 0.160,
  sheet rms 0.047 to 0.137 between steps 8 and 9 against 0.044 to 0.136).
  The metric folds each 32-cell window's residual after a quadratic fit, so
  the sheet's own large-scale shape reads as "locked" (its period-16 bin is
  0.173 of 0.195).
- **Liquid dips 4 to 6 %** around steps 16 to 24 at 1/60 s. It grows with
  the time step (0.85 %, 2.3 % and 3.2 % at 0.24 s for 1/120, 1/60 and
  1/30 s) and the solver before paging lost the same (7.5 % at step 24).
  **Found 14 October: the phi advect stage, a scheme limit.** A ledger by
  stage at steps 12 to 14 loses 2,034, 5,012 and 6,511 cells in the advect
  and nothing in any other stage (reconstruction gives 5 to 7 % back); the
  all-h reference loses the same, and its samples fall with it, so it is
  not an accounting gap. Semi-Lagrangian phi advection is not
  conservative: the stream within one step's travel (six cells) of the wall
  is compressed out at the impact. Two remedies, for Peter, below.

### 3b. Continuous particle contact and slope clearance

- Requested by Peter on 14 October. Use `nbflip-slope-clearance`: a finite
  3.072 L release above a continuous 1:16 incline, with a lower catch basin.
- Replace cell-face snapping at terrain contact with contact against the
  continuous heightfield and its normal. Preserve tangential motion and
  remove only inward normal velocity. Keep terrain contact, solid occupancy,
  pressure capacity and particle/grid transfer consistent; a smooth picture
  alone does not fix a staircase collision boundary.
- Establish the current baseline with `tools/probe-slope-clearance.ts`, then
  diagnose and improve the solver until the slope contains zero liquid
  volume and zero FLIP samples within six simulated seconds. The ideal
  frictionless transit is about 2.45 s; six seconds allows spreading and the
  drainage tail in this free-slip, zero-surface-tension model.
- Retain the released water in the basin: seed error and total volume drift
  each at most 1%. Do not delete residue, add a downhill impulse, or relax the
  clearance criterion to make the test pass. Report the clearance time and
  volume history; confirm at 60 and 30 Hz, with flat-floor and still-basin
  controls. Run GPU gates serially.
- Status: continuous scene geometry is authored; continuous contact and
  measured clearance are not yet established. Work in progress in this chat.

### 4. 4h cache by block

- Gate passed, landing 14 October: its kernel probe against the host
  (11,732 points, worst difference 1.8e-7), trough (particles 64,126 at
  step 20 and 116,137 at step 40, liquid 31,870), water slide (liquid 8,159
  at step 60, 54 pages), letters (12 steps, liquid 1,224,968, 86,464
  particles). No fatal in any.
- The changed passes' minima on letters were 6.03 ms before and 5.96 ms
  after in the agent's runs. A clean frame-time comparison waits for the
  performance push.

### 5. Finish re-keying (each a small package on the landed bundle)

- h band halo on block face planes: landed (above).
- Join and static masks: looked at, nothing changed. Three parts were
  already by tile key. What is left (the builder's static masks, the host
  join plane, the static h plane, the solid tile mask) is written by the
  host by box tile, and the host cannot name a tile key because blocks are
  assigned on the GPU. A tile outside the box reads no bit, so nothing is
  wrong today. They move to item 6 as a small table of region boxes
  (checked: the same bits as today over 15.9 million tiles), with two
  choices to make there: where the coarse mask's solid exception is applied,
  and whether a voxel edit's joins are exact tiles or its bounding boxes.
- The extension's low-face predicate.
- A whole-solid page with stored pages across two axes: today a named fatal
  in the pressure root. No scene has one; a terrain scene would.
- Renderer and overlay box guards that share the full-scan samplers.

### 6. Delete the box

- Remove every `UM_BOX*`, `umTileInBox`, `boxTiles` and the device gates;
  the compiler is the check.
- Remove the box clamps in the address helpers (`umCellTexel`,
  `umTileTexel`, `umVertexTexel`, the `Near` forms, the detail fields' loads
  and ring store) and every caller that relies on them. This is the bulk of
  the item. Until it lands nothing may be stored outside the box (item 1c).
- **Designed 14 October, in progress in a scratch tree.** Counted: 111
  shader lines in 22 files name the box, and 310 lines in 42 files on the
  wider host sweep. The larger half is silent: the four `Texel` forms clamp
  to the box and every field load goes through one (187 loads, 46 of them
  at a computed neighbour). Seven packages, files disjoint except the first
  and last:

  | | Package | Files | Needs |
  |---|---|---|---|
  | A | Primitives, additive (page-relative clamps taking the reader's own cell; no shader changes) | 2 | none |
  | B | Sampling libraries | 4 | A |
  | C | Velocity extension (the trough fix, generalised) | 3 | none |
  | D | Single-site kernels and host field checks | 17 | A |
  | E | Masks as a 32-entry region table; solid and join planes by tile key | 3 | lands with F |
  | F | Host: sizes and callers | 9 | A, E |
  | G | Delete the box names; the compiler is then the check | 6, 9 tools, 4 tests | all |

  A is written and checked on the CPU in a scratch tree; C and E were
  stopped part-way. The single agent brings A over and does the rest in
  the order A, C, B, D, E with F, G.
- **14 October, evening: A to G are in the repository.** What changed from
  the design:
  - Every address text ends with the page-table forms (no module includes
    them by name): `umPointCell`; `umPointBounds(p, margin)` (a sample's
    walls: per axis the nearest SOLID page face within the margin, from at
    most seven page words behind an early-out for a point deep in its page);
    `umBoundedCell`; `umHold`, a private holding a sample's walls, set once
    at a sampler's entry and read by every tap; `umVertexCellAbove` and
    `umVertexCellBelow` (one page word behind a page-face branch); and
    `umPointHeld`. The surface module's own copy of `umPointCell` is
    deleted. The scratch tree's per-tap holds (`umVertexAuthorityCell`,
    `umVertexHeldCell`, `umTapHeld`) did not land.
  - B does not use the design's `…NearAt` forms: most includers of the
    sampling libraries carry no block neighbour rows. A sample's walls are
    resolved once (`umPointBounds`) and the taps are clamped to them; at an
    AIR page nothing holds, as before. A zero-weight far tap is no longer
    loaded in the general fine sampler.
  - D: risk 3 is decided as "a tile resolves a vertex on its high planes
    unless a tile across them (face, edge or corner) is stored". The host
    field checks compare the field's class (`uniformDetailClass`); the 4h
    cache checks keep their block extent (not a box). Forces' 4h home and
    the remap and surface-volume 27-tile loops follow the design.
  - E and F: the builder's static masks are a 32-entry region table
    (`uniformMixedRegionTable`, `setRegions`); the solid and join planes are
    written by tile key (`uniformMixedTileWord`, a host tile plane read by
    `umHostSolid`/`umHostJoined`). The request path's static tiles go
    through the region table too; under narrow-band FLIP that path is dead.
  - G: no solver module names a box. The topology, solid record, root
    owners, particle order, NB-FLIP refine, detail ring and dense reference
    modules build their address without one, so every Texel form resolves
    a coordinate through the page table (off the stored pages: the AIR or
    SOLID block). The detail rewriter finds a shader's layout by its tile
    key count (`UM_TILES`) instead of `UM_D`; `udRingStore` reads the tile
    key and answers false off the stored tiles; the face loads' box clamp is
    gone; the run forms lost their box term; the solid stub's
    `umSolidValid` is "not on a SOLID page"; the dense reference clamps its
    own texture's extent. Only the renderer's presentation module still
    names a box (by design, item 2). The host's t=0 build arrays
    (`uniformBoxTileIndex` and friends) stay box-shaped, as planned.
  - Pipeline compile time (found while landing B): Metal inlines every
    call, and the velocity sampler had 144 inlined copies in `advect`. Each
    heavy sampler now has one call site (loops over axes, faces, stages and
    sides; no work added). The letters build went from 255 s to 45.5 s (the
    last pipeline; no pipeline over 6.2 s while three compile at once, the
    slowest 2.4 to 3.5 s alone). The NB-FLIP seed kernels got the same
    treatment (a compile-only, bit-identical edit in the particle file:
    seedInitial 9.2 s to 2.4 s alone).
- Gate, once, at the end: settled tank (exact), trough, water slide,
  letters, each also with the SOLID sentinel block poisoned to 1e30 so that
  a load the old clamp was silently holding shows up as a loud failure.
- **Gate run 14 October, evening (A to G in the repository):** settled tank
  liquid 3,072 and particles 3,584 at every printed step (exact); trough
  particles 47,672 / 64,094 / 88,981 / 115,975 and liquid 31,072 / 31,347 /
  31,702 / 31,879 at steps 10 to 40; letters step 12 liquid 1,224,968,
  86,483 particles; every run `fatal none`, `errors []`. Water slide step
  60: liquid 7,988, 66 pages, 2,003 h tiles, 82,262 particles. That is
  outside the old range (8,160 to 8,200, 54 to 57 pages), but the slide is
  no longer the scene those numbers were taken on: the other session has
  since replaced its voxel steps with a procedural solid (continuous floor;
  `nbflip-water-slide-scene.ts`, edited 16:33), and the h tiles differ from
  step 1 (246 against 244). Not separated from this item's changes on the
  device; the slide needs new reference numbers from that session.
- Tripwire (scratch edit, reverted): every detail field's SOLID sentinel
  block filled with 1e30 at creation. Trough step 40: particles 115,591,
  liquid 31,878; slide step 60: liquid 7,991, 72 pages; no fatal, no error,
  both within run-to-run range of the clean runs. Only detail fields were
  poisoned (the textures made through the detail storage), not the other
  block-addressed buffers. Unit suite: 820 pass, 0 fail; no test deleted.
- Stays box-shaped after this, on purpose: the published pool record (the
  other session's overlay reads it), the renderer's copy of the address
  (item 2), and the host's t=0 build arrays.
- Still blocking a page outside the box after this: pages outside are
  seeded SOLID, page coordinates are unsigned (no lattice origin), low
  faces are "coordinate below 0", world-to-cell uses the container's
  corner, the solid mask is sized by box pages. Those are the lattice
  origin and edge-fatal step below, after which the open-top headroom can
  go.
- The page overlay's published record is box-shaped (kept that way so the
  other session's domain-pages overlay works). It is re-keyed here, with
  that session, so the overlay can show pages claimed outside the box.
- Then: lattice origin and an edge fatal (first point at which a scene can
  be authored without a tank), and host build arrays sized by the liquid's
  pages.

### 7. Performance push (after Peter confirms correctness)

- Target scene: `nbflip-figure-8-letters`, against 21.25 ms box and 28.33 ms
  full.
- Known levers: resolve the vertex address once per tile workgroup;
  lane-uniform runs; the 16h smoothing launch width (194 workgroups now,
  60 before).
- Padding cost: a padded scene holds two layers of padding tiles at h. If it
  matters, treat a whole solid tile's face as a wall.

### Trough liquid total (measured 14 October, evening)

- `uniform-trough-dam-break`, 2 s simulated, two repeats an arm. At 1/60 s
  the liquid total is +0.7 / +2.3 / +3.7 to +4.0 / +2.7 to +3.0 / +6.0 to
  +6.5 / +5.2 % at 1/3, 2/3, 1, 4/3, 5/3 and 2 s (repeats agree to 0.4 %).
  At 1/30 s: +0.8 / +1.0 / +2.5 to +2.9 / +3.1 to +3.3 / -1.6 to -0.4 /
  +0.5 to +2.2 %, swinging 1 to 3 % between prints five steps apart.
- The gain follows the number of steps more than the simulated time, so a
  stage adds liquid (or the count misreads) per step. The 1/30 swings are
  too fast for the flow, which points at the count as at least part of it.
  No stage ledger was run and nothing was changed.

## Waiting on Peter

- **Trough liquid gain** (above): say whether to run the per-stage ledger
  (1/60 s, steps 40 to 60, with an independent recount of the same state).
- **Water slide reference numbers**: the other session's procedural slide
  replaced the voxel one; the old gate range (8,160 to 8,200 liquid, 54 to
  57 pages at step 60) no longer applies. Today: 7,988 / 7,991 liquid, 66 /
  72 pages.
- **The second screenshot** (water slide with two grid planes in the air)
  arrived with no text. If it was meant for this session, say what to look
  at.
- **Renderer window list** (item 2): resume or leave.
- **Dam-break foot terrace** (item 3): only tolerance 0 is clean so far, at
  +76 % h tiles and 2.4 times the samples at step 12. Say whether the front
  is worth that, or whether to try a seeding rule (untested: seed the whole
  surface at t=0 under the default criteria, to show whether sample age
  alone decides). No more dam-break work until then.
- Decided 14 October 16:40: step sizes are never limited (realtime), so
  the impact loss's step-limit remedy is dead and the volume-source remedy
  waits to be asked for; open tops keep the 64-cell headroom and true
  growth or outflow is not built (lattice origin and growth out of scope);
  unit tests get no effort (the ten added files were deleted, the
  pressure-compilation test is not rebuilt); commits happen at each passing
  gate without asking.

## Small things noted, not scheduled

- The storage's `capture` (used by `probe-nbflip-speed.ts` and the dent
  probe) places texels through the host's seed table, so pages claimed after
  the seed read as block 0. Wrong for any scene that grows; fine for the dam
  break, whose pages are all resident from the seed.
- `createEmptyScene` builds its shell before setting the cell size.
- Letters particle count is about 1.5 % lower since the census change.
- Flat-arm ramp volume drift; level-floor film rule.
- The dam break's top edge is rounded at the 50 % budget (4h cannot hold a
  convex edge); sharp at budget 100.
