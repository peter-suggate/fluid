# Cross-stage duplicated work in the pour's late frames

Scene: `nbflip-figure-7-pour` at 1/60 s, f200–212, about 1.3M particles and 10k fine tiles, on the M1 Max. On 3e1db823 a late frame costs 57.6 ms of pass time plus 8.2 ms idle; the target is 17 ms.

Peter, 2026-10-20: "make sure we're taking big swings with the goal of cutting out duplicated work or similar looking work across stages." This document lists the work that two stages compute from the same inputs, ranked by the ms a merge deletes. Pass times are late-frame means from base5 (41d214c1) or base8 (3e1db823); either is noted where they differ.

## Landed this session

| Commit | Change | Pass | Before | After |
|---|---|---|---|---|
| 81e94cb3 | The level set traced back in one step of any length | advectTrace | 0.74 | 0.47 |
| 41d214c1 | Plane corners in three lean launches | remapCells plus the plane launches | 1.43 | 0.31 |
| cbb2cf03 | Geometry's 4h turns in lanes of their own; exact same-sign certificate | geometryRegional | 1.63 | 1.31 |

The certificate in cbb2cf03 was checked bit for bit against umSurfaceTarget: about 530k certified owners a frame, no mismatch.

## Nulls (don't retry)

- **A per-tile cell table for coupleFine** (capacity, open and sample count in workgroup memory, a job's 216 cells instead of 16 lookups a vertex): coupleFine was 2.6 ms in both arms of an alternating-frame A/B. Counts per late frame:
  - 605k vertices;
  - 79% heated;
  - 51% gather.

  Almost every vertex does the full work. The earlier knock-out of the buried test and the held loop (−0.69 ms) measured occupancy, not gathers.
- **The particle move's pieces,** by alternating-frame knock-outs of advectParticles (3.38 ms):

  | Knocked out | Saving (ms) |
  |---|---|
  | gridSupported | 0.43 |
  | depth | 0.40 |
  | trace stages 2–3 | 0.46 |
  | solid blocked test | 0.10 |

  No single piece dominates; G2P, the record traffic and the appends make up the other 2 ms.
- **transferRefine folded into transferDeposit (#5 below)**, the refine's per-sample body run in deposit's classify loop on the record already loaded and its flush after each tile:

  | Pass | Before (ms) | After (ms) |
  |---|---|---|
  | transferDeposit | 1.60 | 3.05 |
  | transferRefine | 0.68 | (deleted) |

  Net +0.76 ms. The refine code in the scatter loop costs the 192-lane deposit its occupancy. This is the fat-kernel rule again (lean split kernels over compact lists beat fat merged ones): a cross-stage merge has to delete work, with lean kernels reading a compact product, not fuse two kernels' bodies. The same rule rules out merging classify with localSpeed (#3) into one kernel.

## Ranked merges

### 1. Retire the all-4h pressure mirror: −2.5 to −2.8 ms, exact up to summation order (pressure; offered to the pressure session)

**Duplicated work.** For a 4h-held tile, the simulation's 4h owner and the root's pressure owner are the same owner, with the same texels (uniform-mixed-surface-geometry.ts:33–39). The step still copies faces into the pressure layout and back, evaluates geometry for both ownerships and runs a root rhs and project.

**What it would delete (base8, ms):**

| Pass | ms |
|---|---|
| transfer to pressure | 0.46 |
| pressure rhs | 0.24 |
| pressure project | 0.72 |
| transfer to simulation | 0.27 |
| pressure-ownership authority dispatches | part of 1.57 |
| geometry's pressure outputs | about 0.1–0.2 |

**Sketch:**
- Index native rows by the simulation ownership's 4h owner slot.
- Use a lean rhs over the simulation's 4h owner list that reads faces in place.
- Hold native pressure in a simulation-owned 4h buffer allocated at capacity.
- Use a lean 4h-face project.

**Check first:**
- whether the two authorities cut 4h tiles by the same rule;
- wall slots;
- freeze;
- whether presentation and warm start read root pressure.

### 2. The snapshot chain: −1.7 to −2.2 ms

- **(a) The old field as a second channel of the pair's hierarchy and publish: exact, −0.5 to −0.7 ms.**
  - The slot snapshot builds its own hierarchy (restrictBand 0.12, restrict 0.09, sweep 0.12, prolong 0.22) and publish (snapshotFine 0.38, snapshotCoarse 0.21).
  - These use the same selections as the current field's (uniform-mixed-extension.ts:1504–1540).
- **(b) u\* remapped as a channel of the paired remapFaces: a physics change behind an NB-FLIP param, −1.2 to −1.5 ms.**
  - Today the NB snapshot builds a second sampling infrastructure for u\*: snapshot cache 0.58, unitVelocityTaps 0.33, and the record pass 0.60 that writes `before`.
  - The head's cache and taps then compute the same weights again for G2P (uniform-narrow-band-flip.ts:1260–1267, 1477–1488).
  - With two-channel head taps, G2P would sample u\* and u_new with one set of weights, and the `before` field and the record pass would go.
  - This overlaps the particle doc's "snapshot fold into G2P"; count only one of the two.

### 3. Census and local speed from producer tile summaries: exact, −0.9 to −1.1 ms (plus −0.5 for local speed)

**Duplicated work.**
- At the head, census classify (1.13) and classifyCoarse (0.31) reload each tile's phi corners, the 125 vertices for resolution error on mixed tiles, and the ±face velocities (uniform-mixed-dynamic.ts:624–660, 740–810, 892–950).
- Nothing writes that phi after geometry at the end of the previous step.
- Geometry's work-control compare already loads all 125 vertices a tile (uniform-surface-work-control.ts:297–343).
- The faces were last written by the tail publish.

**Sketch.**
- The compare emits per-tile summaries: crossing mask, inside/deep bits, value range and resolution error.
- The publish emits per-axis face extents.
- classify becomes a lean pass over the summaries.
- localSpeed (1.22, about 125 phi loads a tile) can take the same extents, or become change-driven over the remap's dilated list.

**Risks:**
- Held regions skip the compare, so their summaries must stay valid.
- The face set must match `umFaceLow`'s skips.
- Freeze steps write the head.

### 4. One paired remapFaces launch for the extension and velocity fields: exact, −0.55 to −0.7 ms

- The two launches (about 0.65 each) walk the same listed tiles and the same addresses.
- The extension launch reads the scratch velocity that the velocity launch writes. Pairing them needs the velocity remap's output kept apart from the extension's input for the duration of the launch.

### 5. transferRefine folded into transferDeposit: measured +0.76 ms (null, see Nulls)

transferRefine (0.67–0.75) walks deposit's occupied-tile list and record ranges again (uniform-particle-brick.wgsl.ts:173 vs :228). Deposit would append refine's inputs to a compact list instead.

### 6. Seam face address table: 0 to 1.5 ms, knock-out first

The same umOwnerAt/umFace walks (uniform-mixed-faces.wgsl.ts:17–60) feed remapFaces, publishSeams, the pair seam sweeps, the forces seam launch, transferSeams and the hanging taps.

**Shared static face apertures** (pressure session, in progress): umPressureFaceV's full solid evaluation is about 0.6 of pRows' 1.77 ms. A cached aperture per h face (3 × 64 a tile, rebuilt when stale) will also serve these call sites:
- remap's beside, cut-weight and open-face tests;
- the transfers;
- geometry.

### 7. Body forces re-deriving face fractions: about −0.5 ms, unverified

## Dropped (inputs differ between the two "duplicates")

- **Crossing search vs arena search:** advected vs coupled phi. But see "Reconstruct crossing search" below: its outputs may be read only by couple's bulkDepth before the arena rebuilds them.
- **Surface advect vs the arena redistance** (the advect's ~6.2 ms against arenaStore's rewrite):
  - arenaStore replaces the advected magnitude with sign·distance only at non-interface h vertices of arena bricks.
  - Interface vertices, every 4h canonical corner, h vertices outside the arena and values ≥6h past reach remain advected state.
  - The sign alone needs the departure point, so every vertex within travel plus stencil of the zero must be traced anyway. That is 56% of h vertices at f100 (brick-arena-core §6.3).
  - The exact cut left is a certified-sign hold of deep vertices: same-sign data over the tile's signed reach box (localSpeed plus spread), min|φ| ≥ 16·max h for every threshold reader. That is about −0.4 ms, or about −0.85 ms with a 6h floor behind a param.
  - Kept as a minor lever.

### Reconstruct crossing search: −1.0 to −1.2 ms, behind a param (under audit)

**What duplicates it.** The batch (depthTiles, depthSeeds, reuse, depthNearestY/Z, depthLists) finds each cell's nearest crossing on the advected phi. Its bank B, masks and reach bits appear to be read only by couple's bulkDepth before the arena search rebuilds them on the coupled phi. bulkDepth deepens negative bulk values: `min(phi, -distance to the nearest crossing box)`.

**Candidate replacement.** Take bulkDepth from the previous step's arena outputs (reach bits and bank B, the nearest crossing of the last final phi), shortened by the tile's travel this step.
- **geometryChanged vs geometryRegional:** the surface stages write phi in between.
- **The two authority runs:** phi changes between them.
- **The five phi-resolve runs:** each follows a different writer.
- **Surface advect vs particle move:** they sample different points and already share the cache and taps.

## Owners

- **Pressure session:** #1 and the face-aperture cache.
- **This session:** #2, #3, #4 and #5, and switching the remap and transfer call sites to the aperture cache once it lands.

Entries 1–5 together are about 6–7.5 ms. That leaves the frame well outside 2× its 17 ms target, so the structural levers (one MG over the h and 4h bricks, extension as one resident pass, the hardware-shaped tile context) remain necessary after these.
