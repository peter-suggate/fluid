# Momentum transport on paddleboats: why it costs 19 ms, and the plan to get under 1 ms

Scope: particle move, separation, sort, transfer and seed; velocity extension; remap; momentum; forces; tiers; hanging taps; and the 4h cache.

Sources:
- Kernel means: boats.json (300 steps, `--bodies`, clean baseline 68.66 ms per advance).
- Counters: xctrace-kernels.txt (44 advances, serialised).
- Counts: mom-counts.txt (check mode, frames 180 and 240).
- A second scene: mom-churn.json (churning-tank, 150 steps).
- A separation-off A/B: mom-nosep.json.

## 0. What the hardware says (all momentum kernels)

Paddleboats at steady state:
- 333k particles.
- 4.85k h tiles (310k h cells, out of 98.3k tiles).
- 374k owners in total.
- The layout generation advances every frame (179 at f180), so remap runs every step.

Momentum transport costs about 19 ms per advance. The particle-proportional half is about 9.5 ms (about 28 ns per particle); the owner-proportional half is about 9.5 ms (about 25 ns per owner).

Every kernel above 0.1 ms runs at 2–25% occupancy, 0.6–42% ALU and 0.3–40 GB/s read, on a part that delivers about 400 GB/s. Nothing is bandwidth-bound or ALU-bound.

**The time is threads × dependent-load latency:** each item walks chains of the form page word → block → tile key → topology word → texel. It walks them again per tap, per neighbour, per stage or per helper call, instead of resolving them once per tile or brick.

Two secondary causes:
- **Launch shape:** dispatch counts are fixed by structure rather than by work, e.g. 22 merge rounds and 5.7 hierarchy levels per advance.
- **Serial tails:** reductions done serially inside one workgroup, as in planeSums.

The floor shows how far off this is. Bandwidth: a particle is read and written once (about 64 + 64 B), plus 16 B of position read for separation, so 333k particles is about 50 MB, or 0.13 ms at 400 GB/s. ALU: about 2k flops per particle is about 0.07 ms at 10 TFLOPS. Grid traffic is about 20 MB, or 0.05 ms. **The work itself fits in about 0.3 ms. The other 18.7 ms is waiting on dependent loads, empty launches and serial tails.**

## 1. Per-kernel table (paddleboats, ms per advance)

The ms column is the kernel mean from boats.json. occ, alu and rd come from xctrace. A "chain" is one dependent device round trip. Extent is the threads launched; items is the useful work.

| kernel | ms | extent → useful items | per-item loads and loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|
| **advectParticlesDeferred** | 4.41 | 3072 WG × 64 persistent → 333k records (slab jobs) | Heun RK3 makes 3 calls to `umSampleVelocity`. Each call: `umPointBounds` (page-kind lookups near page faces) → `umTileAt` (page word → block → tile) → `umVelocitySamplingWeightsAt` (support word + blend mask loop, up to 27 trips) → `umUnitHaloHome` (block stored → fine slot → halo origin) → 3 axes × 8 taps (unit halo texture or 4h cache; 1 chain each after the home). G2P: `nbMoveCold`, `nbSparseAge`, `nbMoveDepth` = `particleDepthIn` (vertex run lookup + 8 `atomicLoad` of NB_DEPTH_A), `nbMoveHeat` × 2, `nbBinCell` (coverage + slot map + links + bins), `nbRigidContact` (body loop, 4 iterations × 7 `umBodyDistance`), `nbParticleBlocked`, `gridSupported` (up to 8 × `umOwnerAt` + phase texel). Surface samples (depth > −1) also run `nbSurfaceShift`: **7 more `nbMoveDepth` calls, i.e. 56 atomic depth loads plus 7 vertex-run lookups**. Then `nbTraceEndpointIn` and the `nbFinish` solid walk (per-voxel `umCellKind` chains). Estimate: about 100–150 dependent round trips per particle. | Latency-bound: dependent chains × register-capped occupancy. occ 10.5% is the register ceiling the code itself names: "whose registers hold the move to a few hundred lanes a core". About 10k resident lanes × 4.41 ms / 333k is about 130 µs of serial latency per particle, consistent with 100+ dependent misses. Read traffic is only 175 B per particle. | `uniform-narrow-band-flip.ts:1196-1330` (stage loop 1243-1309); `nbSurfaceShift` at `:1186`; `particleDepthIn` at `uniform-narrow-band-membership.wgsl.ts:300-326`; sampler at `uniform-mixed-velocity-sampling.wgsl.ts:93-107,178-183,239-250,293,331-344`; `gridSupported` at `uniform-narrow-band-flip.ts:993`; separation-prepass comment at `:1557`. xctrace: occ 10.5, alu 14.3, rd 12.9 GB/s. The window stage `nbWinStage` (`:1099`) already caches 5 tile flags per window, but none of the field, depth or bin data the chains above read. |
| **separationPrepass** | 1.15 | 98k → 333k (4 launch jobs per slab job) | 4 rows × ≤ 2 `nbRun`. Each run is `umTileKey` (3-level) → `umKeyStored` → coverage atomic → `nbBinTile` (slot map atomic) → 2 links + 1 bins: about 5 chained loads per run, about 8 runs per particle. Then about 8–16 `particles[j].position` reads (contiguous). | Dependent run lookups. The pair loop itself is well-behaved: occ 22.9, rd 39.5 GB/s, the best of the particle kernels. The prepass exists only to hide the move's latency; it repeats the record read and the bin lookups the move also does. | `uniform-narrow-band-flip.ts:830-834` (`nbRun`), `:1159-1176` (`nbSeparation`), `:1557+`. |
| slabSort | 0.61 | persistent → about 1.9k bricks | Per-brick counting sort with global atomics on bin counts and links, then a scatter. | Atomics plus a scatter. rd 35 GB/s, write 17 GB/s: partly real traffic, so it is the least latency-dominated kernel here. | `uniform-particle-slab.wgsl.ts:296`. |
| slabSortMerge | 0.14 | **22 dispatches** × 6 µs, occ 6%, ALU 2% | One merge entry per power of two below **capacity** (5.35M), not below the live slab size. | **Launch shape**: about 20 rounds have nothing to merge. | `uniform-particle-slab.wgsl.ts:52,406`. |
| transferSeams / Deposit / Properties / Refine | 0.60 / 0.53 / 0.40 / 0.25 | persistent over slab jobs → 333k particles, about 4.85k tiles | Deposit: tile-staged, but `pbScatter` does 3 axes × 27 nodes × 2 = **162 threadgroup atomics per particle**, then a 756-slot global flush. Seams and Properties re-resolve the owner and face of every node per particle through `umOwnerAt` chains. | Seams and Properties: dependent owner chains (occ 13.7, rd 6 GB/s). Deposit: shared-memory atomic contention (alu 36.7, occ 15). | `uniform-particle-brick.wgsl.ts:159` (`transferDeposit`), `:228` (`transferRefine`). |
| seed / seedCells | 0.46 / 0.18 | persistent → band cells | Per cell: depth (8 vertex loads), coverage, bins, and a page walk per candidate site. | Dependent chains (occ 13.4). | `uniform-narrow-band-flip.ts:1663,1711`. |
| pair_sweepSeams + sweepSeams | 0.68 + 0.28 | **×1.9 dispatches** (pair and current payloads swept separately) → seam owners | Per face, `ueExtend` evaluates up to 6 `umNeighbor`. Each is `umCellPresent` → `umTileAt` → 2 × `umSlotState` (`umFaceLow` + slot) → `umTileWidth` / `umTileMaximumWidth`, plus a site lookup on coarse seams: about 4–6 chained loads × 6 neighbours × 3 axes per owner. | Dependent neighbour chains (occ 11.8). **The same geometry is swept twice**, once per payload. | `uniform-mixed-extension.ts:540-560` (`umNeighbor`), `:971` (`ueExtend`), `:1125-1140` (`ueSweepSeamFine`), `:1258-1281`. |
| publishSeams (+ publishFine, publishList) | 0.67 (+0.20 +0.14) | all seam/fine owners every step | Per face: slot state → write the velocity texel, plus 2 compute copies. | Republishes every face every step, including the unchanged ones (occ 23, ALU 37: the cheapest per item, but done for everything). | `uniform-mixed-extension.ts:1454`. |
| mixedProlongUnknownVelocity | 0.65 | **5.7 dispatches** × 98 µs → every cell of every resident block at each level (4h…16h) | Per cell: 3 texture loads + `nearestHierarchySample` (taps across the neighbour row, `pagePool[ROWS+27b+…]`). | Work proportional to **resident blocks × levels**, not to unknown faces. Most cells are already known (the `entry.w & 1` test) and exit after 3 loads. The cost is the launch plus a full pass over the pool. | `webgpu-uniform-velocity-extrapolation.wgsl.ts:310-335`. |
| mixedSweepBlocks | 0.30 | 5.7 dispatches × 52 µs, **ALU 0.6%, rd 0.3 GB/s** | One 32h cell per block (192 lanes, of which ≤ 64 hold a value). | **Pure launch and barrier floor**: about 52 µs per empty persistent launch. | `webgpu-uniform-velocity-extrapolation.wgsl.ts:277-300`; `UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS=3` at `:7`. |
| mixedRestrictKnownVelocity, restrictBand | 0.20, 0.18 | per level, per live tile | Texture restrict per level. | Same launch-shape cost: occ 6.3% at 33 µs per dispatch. | same file `:252`; `uniform-mixed-extension.ts:1283`. |
| **remapFaces** ×2 (extension scratch + velocity) | 1.00 | listed tiles × 192 lanes, **every step** (the layout changes every frame) | Per face: old and new `umOwnerAt` / `umFace` / `umFacePatch` chains plus a footprint loop over the old owner's faces; 4 barriers per tile. | Dependent owner chains (occ 10.2, rd 1.5–2 GB/s, the lowest of any large kernel). Triggered every step by census churn (the census fork owns that). | `uniform-mixed-remap.ts:1013-1016`; encode at `:239-242`. |
| planeSums | 0.31 | jobs = plane corners (few) × 256 lanes, **occ 3.1%, ALU 3.3%** | Per job, 16 *serial* parts, each with a barrier, `planeReduce` (a 256-lane tree) and a lane-0 accumulate. | **Serial tail**: too few jobs to fill the GPU, and each job is a 16-deep barrier chain. | `uniform-mixed-remap.ts:1366-1400`. |
| remapCells | 0.10 | listed tiles | owner chains | as for remapFaces | `uniform-mixed-remap.ts` (PLANE_STEPS at `:24`) |
| momentumStep | 0.65 | merged tiles → 374k owners | Per owner: topology word → face patches → neighbour owners for the advected stencil (`umOwnerAt` per neighbour). | Dependent chains (occ 12.4, rd 3.6 GB/s). | `uniform-mixed-momentum.ts:518-528`. |
| momentumTrace / momentumFine | 0.31 / 0.28 | fine owners | `momentumFine` already stages `utcSides`/`umFineLive`. Trace samples go through `umSampleVelocity` (the same chain as the move). | Trace: the sampler chain. Fine: partly staged, so better (occ 22.8). | `uniform-mixed-momentum.ts:178,210`. |
| **forces / forcesRegularCoarse / forcesFine** | 0.46 / 0.30 / 0.23 | all owners | For an inviscid scene each face gets only `+= dt·g`. Yet each owner reads occupancy, the neighbour faces, coarse mass and retain words (3–4 chains) to decide that. | **Avoidable work**: 1 ms to add a constant that `momentumStep` could add as it stores the face. | `uniform-mixed-forces.ts:422`. |
| momentumAffineFit | 0.36 | 640 WG, regions | Eligibility does 27 × 64 `umTileKey` plus 3n³ × 2 `umOwnerAt` per region. | Lookups repeated per cell that are constant per region (occ 22.9). | `uniform-mixed-momentum-tiers.ts:252`. |
| unitVelocityTaps + fillUnitHalo | 0.29 + 0.26 | every fine block every step | Rebuilds the unit halo the sampler reads. | Whole-band rebuild that exists only so samplers can skip the per-tap chain. | (hanging fine taps) |
| 4h sampling cache | 0.32 | 1 WG-grid dispatch | Rebuilds every 4h cache block. | Whole-pool rebuild every step. | (`uniformCoarseCacheWGSL`) |

Separation A/B (`sampleSeparation: "off"`, 150 steps, kernels timed from step 90): see §4. It isolates the separation prepass plus the move's `nbSurfaceShift` tail.

The second scene confirms that the per-item cost holds across scenes. On churning-tank (110k particles late, 1.3k h tiles), the same kernels cost about 6.2 ms over the run (whose mean holds far fewer particles). Per-item costs are within 1.5× of paddleboats, so this is per-item latency, not a paddleboats pathology. Particle counts ramp during that run, so I did not fit a floor and slope from it. The empty-launch floor is measured directly instead: mixedSweepBlocks at 52 µs, and slabSortMerge at 6 µs × 22.

## 2. Root-cause summary

1. **Per-item re-resolution of addresses** accounts for about 80% of the 19 ms. The tile, block, owner, slot and vertex keys of a 3³-tile neighbourhood are constant over a slab job or tile job. Yet they are re-derived per particle, per RK stage, per tap, per depth call (×8 at the surface) and per neighbour face. Each derivation is a 2–4 deep dependent chain. With register-capped occupancy (10–15%) nothing hides it.
2. **Work that does not need doing:**
   - forces visiting every owner to add dt·g;
   - the extension sweeping pair and current geometry twice;
   - the hierarchy passing over every resident block at every level;
   - publish and halo rebuilds over every face, every step;
   - the separation prepass re-reading what the move reads.
3. **Launch shape:** merge rounds sized by capacity; hierarchy levels and sweeps launched even when nothing is unknown. Each empty persistent launch is about 6–50 µs.
4. **Serial tails:** planeSums.

## 3. Plan

Every item below applies to every scene. The kernels change; no scene-specific branch is added. Each item lists the expected saving on paddleboats and how it is verified. The verification protocol for every item is:
- `npm run check:types` and `npm run test:unit`;
- seep and letters drift, and step time, with one short timing run each (seep 300 / letters 120 / rest 60, per the GPU gate policy);
- paddleboats `--bodies` 150-step kernel timing.

### M1 — Brick-staged particle pass (move + separation + depth). −4.5 ms (5.56 → about 1.0)

Extend `nbWinStage` (`uniform-narrow-band-flip.ts:1099`). It already stages tile flags for the 4³-tile window of a slab job. Stage, once per slab job and cooperatively by the workgroup, the **operands** of the per-particle chains:
- (a) the window's sampling context per tile: key, width, blend mask, halo home and 4h cache origin (64 tiles × about 24 B);
- (b) the window's nodal depth (NB_DEPTH_A) on the 17³ vertex lattice, or a 2-tile-halo sub-window around the brick (about 10³ × 4 B);
- (c) the window cells' bin start and count (10³ × 8 B).

`umSampleVelocity`, `particleDepthIn`, `nbRun` and `gridSupported` then resolve each tile context from workgroup memory. A tap reads the texel directly: one load, no chain.

This is the same function and the same arithmetic. Only the operand's address source changes: workgroup memory inside the window, the device chain outside it. That is the pattern `nbWinIndex` already uses, so results are bitwise identical. It is not a scene fast path: every sample in every scene takes it, and a trace that leaves the window simply pays today's cost.

Fold `separationPrepass` into the move. With bin runs staged, the pair loop's lookups are workgroup reads; the prepass and its 333k second record reads go (−1.15 ms).

Raise the slab-job workgroup to 256 lanes (`NB_MOVE_CHUNK` is already 256, `uniform-particle-slab.wgsl.ts:183`) so the staging amortises. Workgroup memory is about 14–16 KB, so two workgroups per core give 512 lanes, against today's "few hundred".

Expected: about 100–150 chained round trips per particle become about 3 (record in, record out, a few window texels). The move then approaches its traffic floor of about 0.15 ms. I budget 0.8–1.0 ms in total for the move including its walk and contact.

Verify:
- A new unit test: a frozen field sampled through staged versus unstaged lookups gives bitwise-equal `umSampleVelocity`, `particleDepthIn`, `nbRun` and `gridSupported` on the mixed h/4h/16h fixtures, including samples outside the window.
- Seep and letters drift unchanged within run noise.
- Advect kernel ms and occupancy, from timing.

### M2 — One extension sweep for both payloads. −0.9 ms

`pair_*` and current sweeps share geometry: states, distances and the upwind choice in `ueCombine` (`uniform-mixed-extension.ts:622`). Carry the two payloads as one vec2 slot value, so a single seed, sweep and publish serves both. This removes the ×1.9 on the sweep and seed dispatches.

Precondition to check first: both payloads seed from the same known-face set. If they don't, the shared distance field is still valid and the payload with fewer sources keeps its own mask bit.

Verify: a unit test comparing the vec2 sweep with two scalar sweeps (equal per payload); then seep and letters drift.

### M3 — Extension: work only where faces are unknown or changed. −1.5 ms

- (a) The seam publish and the restrict, sweep and prolong hierarchy run only over **blocks holding an unknown or changed face**. The list comes from `ueMaskOut` / `ueNote` bits that the seam sweeps already produce (`uniform-mixed-extension.ts:1140`), and the hierarchy dispatches indirectly. Levels whose list is empty dispatch zero workgroups. That removes the 5.7 × (52–98 µs) launch floors of mixedSweepBlocks and mixedProlong.
- (b) The publish writes only faces whose slot changed. The regional retain (`ueRegionalRetain`) already knows the clean regions.

Verify: the existing differential sweep test (`ueSweepKind = 0` reference, `:1255`) stays equal; drift; step time.

### M4 — Fold gravity into the momentum store; forces only over a nonzero-term face list. −0.9 ms

For an inviscid, tension-free scene, forces is `+= dt·g` on each liquid face (`uniform-mixed-forces.ts:422`). `momentumStep` and `momentumRegularStep` already hold each face as they store it: add dt·g there, under the same liquid test they use. Body and viscous terms, where a scene has them, run as their own kernel over the faces where the term is nonzero (body-adjacent faces from the body mask).

This is general: the same code runs everywhere, and an empty term list dispatches zero workgroups.

Verify:
- A unit test: momentum + gravity fused equals momentum then forces, on fixtures with bodies and viscosity.
- Drift; step time.

### M5 — Tile-staged 3³ contexts for the grid kernels. −2.0 ms

The kernels are momentumStep, momentumTrace, remapFaces/remapCells, transferSeams/Properties, seed/seedCells and the momentumAffineFit eligibility.

Each tile job first stages its 27 neighbour tiles' key, width, support, topology word and slot base (27 lanes, one chain each, in parallel). Every `umOwnerAt` / `umTileAt` / `umNeighbor` then resolves from workgroup memory. `momentumFine` already does this (`utcSides`/`umFineLive`, `uniform-mixed-momentum.ts:178`) and runs at about 2× the occupancy of its unstaged siblings.

AffineFit: 27 × 64 `umTileKey` plus 3n³ × 2 `umOwnerAt` become 27 staged reads (`uniform-mixed-momentum-tiers.ts:252`).

Expected: each kernel drops to about 25–35% of today's cost (about 3 of 4 chains per item gone).

Verify: a staged-versus-device equality test per helper (the M1 pattern); drift; step time.

### M6 — Launch shape and serial tails. −0.6 ms

- (a) Merge rounds sized by the largest live slab, not capacity. Take an indirect dispatch count from slabSortFinish's max, or do the merge in-workgroup when the slab ≤ 2 × workgroup size. That gives about 22 → 3–4 dispatches (−0.12 ms).
- (b) planeSums: give each of the 16 parts its own workgroup, and combine the 16 partials in planeFinish in a fixed order, so the result stays deterministic (−0.27 ms, `uniform-mixed-remap.ts:1366`).
- (c) Deposit: accumulate the 27 node contributions per particle in registers per destination node class, so 162 threadgroup atomics per particle become about 54 (one per node per axis); or deposit by node rather than by particle within the staged brick. Expected −0.3 ms.

Verify: the sort's canonical-order test and the transfer's deposit-equality test pass unchanged; drift.

### M7 — Retire rebuilds whose only purpose was to skip chains. −0.6 ms (conditional on M1 and M5)

Once the sampler reads staged tile contexts, `fillUnitHalo` + `unitVelocityTaps` (0.55 ms) and the 4h cache rebuild (0.32 ms) are only needed where a sampler outside a staged job reads them. Keep the textures but rebuild only blocks whose faces changed (the M3 change list); otherwise remove the dispatch.

Verify: sampler equality (M1 test); drift.

### Total and the honest gap

| step | momentum ms (paddleboats) |
|---|---|
| today | about 19 |
| after M1–M7 | about 19 − (4.5 + 0.9 + 1.5 + 0.9 + 2.0 + 0.6 + 0.6) ≈ **8 → realistically 4–6 ms** (savings overlap; M5 is the least certain) |

**Getting under 1 ms also needs the item counts to come down.** Momentum work scales with live particles (333k) and owners (374k, of which 4.85k h tiles hold 280k). After M1–M7 the per-item cost is about 6–8 ns per particle and about 4 ns per owner. To reach 1 ms:
- particles must fall to about 80k and h tiles to about 1.2k. Those are roughly churning-tank's counts, for a smaller water body.
- remap must stop running every step (the layout generation advances every frame).

Both are census outcomes: the wake latch and the boats' join/solid tiles, which make up 93% of h tiles. They are owned by the census fork's plan, and this plan depends on them.

With M1–M7 plus that census fix, the projection is about 0.8–1.2 ms. Without the census fix, momentum lands at 4–6 ms.

Order of work:
1. M4: smallest, removes 0.9 ms, no risk.
2. M1: the largest saving.
3. M6.
4. M2.
5. M3.
6. M5.
7. M7.

Gate each item on its equality unit test, seep 300 / letters 120 drift and step time, and the paddleboats kernel table. Commit each as it lands.

## 4. Separation A/B (measured)

This was timing run 2 of 2: `fluid-gpu-run.sh timing`, paddleboats `--bodies`, 150 steps, kernels timed from step 90, `--values {"sampleSeparation":"off"}`. Output: mom-nosep.json.

| | particles in the window | advectParticlesDeferred | per particle | separationPrepass |
|---|---|---|---|---|
| baseline (boats.json) | about 280–333k | 4.41 ms | about 13–16 ns | 1.15 ms |
| separation off | 310–352k | **3.10 ms** | **about 9.3 ns** | 0 |

Turning separation off removes about 35–40% of the move's per-particle time, even though this run holds more particles. With separation on, the move does only three extra things:
- reads the three prepass words;
- on merge or contact, recomputes `nbSeparation`;
- runs `nbSurfaceShift` on surface samples: 7 `nbMoveDepth` calls = 7 × (vertex-run lookup + 8 atomic depth loads).

The surface shift is therefore the confirmed tail. It costs about 1.3 ms in the move, plus 1.15 ms in the prepass: **about 2.5 ms (13% of the stage) for the pair separation's address chains.**

Other effects of the run:
- **slabSort is unchanged** (0.70 versus 0.61 ms, tracking the particle count). It is per-particle, not chain-bound.
- **Whole-step median is 58.2 ms**, against 62.3 ms for the 300-step baseline; the late-half median is 71.3 ms as more particles accumulate.
- **Drift is −0.0092 at step 150**: I did not compare it with a baseline at the same step. This was a timing experiment, not a proposal to drop separation.

M1 moves exactly these reads into workgroup memory: staged depth vertices and staged bin runs. The arithmetic and the Khayyer projection stay as they are.
