# Uniform Geometric on the M1 Max: where the machine goes, and the plan

Lead analysis over the seven evidence reports in this directory. Method id is
`uniform-volume`. HEAD `918fc273`. Every number is measured or counted in the
cited report; anything that is my inference says so.

| report | what it holds |
|---|---|
| `xctrace-capture.md` | mini64 Metal System Trace, counters, CPU flame |
| `xctrace-capture-large.md` | fig7 128³ capture, per-level multigrid, CPU/GPU overlap (section 6) |
| `static-core.md`, `static-volume.md`, `static-pressure.md` | shader reads of every live kernel |
| `static-encode-and-prior-art.md` | per-advance API census, prior measurements, measured negatives |
| `activity-state-audit.md` | the seven activity structures and who reads them |
| `derived-state-rework-audit.md` | derived quantities recomputed instead of compiled |

## 1. The diagnosis in one page

mini64, shipping regime (untraced, passes not isolated): **38.75 ms per advance.**

| | per advance |
|---|--:|
| GPU busy | 21.8 ms |
| main-thread CPU (tracing overhead removed) | about 17 ms |
| sum | 38.8 ms |
| WebGPU passes | 755 (652 of them pressure) |
| Metal compute encoders after Dawn merges passes | 26 |

While busy, the GPU runs at **13.5% occupancy and 10.1% ALU**: about 4% of the
machine. Read bandwidth is 13.7 GB/s on a 400 GB/s part. So the method is not
bandwidth-bound and not arithmetic-bound. Three different things bind it, and
each kernel belongs to exactly one of them:

**A. Launch-floor-bound (pressure, bookkeeping).** 88% of GPU firings are under
20 µs and they are 37% of busy time. Pressure averages 13.7 µs per dispatch in
the shipping regime regardless of level size; half the plan's passes are on
levels holding 1,280 cells in total. `mgSmoothColour` runs at 7.3% occupancy.
Faster kernels do nothing here. Only fewer dispatches help.

**B. Dependent-load-chain-bound (the big per-cell kernels).** Phi advection
(2.96 ms, 16.5% occupancy), projection (1.53), velocity advection (1.30),
`uvBuildEdges` (0.98), `mgBuildFinestRhs` (0.75), sharpening propose (0.68).
These launch the whole grid, and each thread then walks long chains of
storage-buffer loads whose addresses depend on earlier loads: the packed solid
bitfield (10 loads per open-fraction query, 40 per face query), atomic class
words, runtime div/mod on uniform dims. Threads spend their life waiting on
loads, which is what low occupancy with low limiter readings looks like. The
lever is fewer dependent loads per thread: compile the answer once, read it
with one texel.

**C. Genuinely memory-limited (two kernels only).** `uvNormalizeRows` (2.10 ms,
last-level-cache limiter 78.8%) and `uvNormalizeDonors` (0.95 ms, buffer-write
limiter 97.9%, occupancy 89%). These are the only places where the 80-byte
edge record and the 24-byte donor sum size actually set the speed.

Two consequences shape the ranking. First, f16 arithmetic and SIMD-width
tuning buy nothing anywhere: pressure does 0.14 FLOP per byte loaded. Second,
CPU and GPU cost are the same order and their sum equals the harness wall, but
section 6 shows that is a harness fence: in the app an advance costs
max(CPU, GPU), which is the GPU at both sizes.

## 2. Rework: what is computed again and again

This is the second brief ("reduce rework, pre-compile state once per frame or
per change"). The audit's central fact, which I checked in the source:

> `cellOpenFraction` (V_i) and `pressureFaceData` (V_face) depend only on the
> solid bitfield, the terrain texture and rigid bodies. In a body-free scene
> they are constant for the whole run. They are evaluated about **185 and 18
> times per cell per advance**: roughly 2,600 storage loads per cell per
> advance to re-derive constants.

Worst sites: `project` about 42 open-fraction evaluations per cell, `uvAdvectPhi`
about 34 (32 of them an 8-corner contact probe that almost always returns
identity), `uvProposeSharpen` 48 across its 8 rounds, `uvBuildEdges` 24 (16 of
them the same cell), `mgBuildFinestRhs` 9 plus 6 face queries (240 loads).
Those five kernels are 6.9 ms of the 21.8 ms GPU busy.

Three details make it worse than it needs to be:

- **Half of every open-fraction query is dead.** `cellOpenFraction` proves
  `staticSolidVoxelOccupied(p)` false, then calls `cellSolidFraction`, whose
  first line tests it again (`ref.wgsl:365` and `:687`). The same doubling sits
  in `faceOpenFraction`. Five of ten loads, every call, every kernel.
- **The compiler cannot hoist any of it.** The bitfield lives in `activeScratch`,
  a `read_write` binding, and Tint emits unrestricted device pointers, so any
  buffer store in the kernel forces may-alias. Source-level load counts are
  close to real here, not upper bounds.
- **The cache already exists.** `velocityD` holds all three V_face values,
  written this advance by `storeExtrapolationAuthority`. The extension reads it
  across 20 passes. Pressure topology, the RHS build and projection do not
  have it in their layout, so they recompute it 15 times per cell. The
  pressure layout has a free storage slot and both layouts have texture room.

Other rework, smaller:

- `sampleVelocity(p)` reads the same tile-class word three times (once per
  component, atomically) and loads 24 texels that cover a mean of 13.9
  distinct ones.
- The 4h coarse velocity table is read through `atomic<u32>` (8 atomic loads
  per component in far air) although its only writer in the advance is a plain
  store in an earlier pass.
- `uvOpen(id)` is resident in `gammaB` at the moment `uvBuildEdges` needs it 24
  times per cell, but the group binds `gammaA`.
- Three full-field copies per advance (`output→vertexPhiField`, `gammaB→gammaA`,
  `volumeB→volumeA`) look like binding-parity flips.
- At 64³ and below, sharpening runs its uncached arm: the large lane caches
  what the default lane recomputes.

## 3. Activity: four censuses, no launch ever skipped

From `activity-state-audit.md`:

- Four structures answer "where is the liquid" every advance from the same
  inputs against the same 4h constant: the two-level tile classes, the
  sharpening tile map (a provable subset of the tile seed), the phi-region
  bounding box (3 passes, 2 copies, the only indirect dispatch outside
  pressure), and the 32³ page flags (an OR-coarsening of tile bits already in
  memory, computed twice).
- **The page flags feed no solver kernel.** Under native records
  `uvEdgeAddress` is the identity. The only reader is the field-view overlay.
  The mark passes launch densely with 63 of 64 threads retiring at once, and
  `uvCompactPages` is one serial thread.
- **Sparsity never removes a launch.** Every stage but phi dispatches the full
  grid and exits per thread. `uvGather`, advection and projection still store
  on the dead arm, so they cannot simply not run.
- **Pages are the wrong dispatch unit.** Garden: 1.0% of tiles live, 17.8% of
  pages live. The exploitable unit is the 4³ tile; pages are a residency and
  overlay summary of the tile map.
- **Pressure consumes no activity at all**, and it is the largest stage.
- Tile classes sit in an atomic storage buffer reached through runtime divides.
  Storage buffers are at 10 of 10; sampled textures are at 12 and storage
  textures at 6, so a `ceil(n/4)³` `r32uint` class texture addressed by
  `id >> 2` fits with no divide, no atomic and no float barrier.

## 4. The plan

Ordered by expected wall-time return per unit of risk. "Exact" means
bit-identical output is achievable and should be the acceptance test. Expected
returns are my estimates from the measured task times; each item names the
measurement that confirms or kills it.

### WP1. Compile geometry once (regime B; exact)

1. Delete the duplicated `staticSolidVoxelOccupied` test in `cellOpenFraction`
   and `faceOpenFraction`; drop the dead `open` local in `pressureSurfacePhi`
   when volume pressure rows are off. Pure subtraction.
2. Bind `faceOpenIn` (`velocityD`) into the pressure-input and projection
   layouts and read V_face from it. Removes about 600 loads per cell.
3. Add a V_i field: `r32float`, written at construction, on scene edit, and
   over a rigid body's dilated bounding box when it moves. Every
   `cellOpenFraction(id)` at an in-range cell becomes one `textureLoad`. Same
   f32, so exact.
4. Add one static bit per 4³ tile: "no solid, terrain or body within reach".
   In such tiles V_i and V_face are exactly 1, the embedded-contact probe is
   identity, and the solid-velocity queries vanish. Carry it in the tile-class
   texture of WP3.
5. Move the solid bitfield to a `read` binding so what remains can be hoisted.

Expected: a third to a half off the five kernels named in section 2, so 2 to
3 ms of 21.8 ms GPU busy at 64³, scaling with cell count at 128³ where these
stages dominate. Confirm with an interleaved xctrace A/B on `Advect page
vertex phi` and `Uniform pressure projection` after step 1 alone; if step 1
moves nothing, the load-chain model is wrong and the rest of WP1 is suspect.

### WP2. Stop paying per pass on the CPU (exact)

Probe results (`scratchpad/probe-pass-batching.mts`, `probe-bindgroup-cost.mts`):
a storage write in one dispatch and a sampled read of the same texture in the
next, inside one compute pass, is valid and correct across 2,000 dispatches.
The comment at `webgpu-uniform-pressure-multigrid.ts:448` is wrong: the usage
scope is the dispatch, not the pass. With a 27-binding group 0, one pass per
dispatch costs 8.7 µs of CPU; one pass for the chain with group 0 bound once
costs 3.5 µs. GPU time does not change, because Dawn already merges passes
into encoders.

1. One compute pass per multigrid cycle (or per stage), group 0 bound once.
2. Give multigrid a minimal group 0. The flame graph shows Dawn walking all 29
   bindings per dispatch (`SyncScopeUsageTracker::AddBindGroup`).
3. Fold the 16 four-byte `clearBuffer` blits into `mgCheckCycleConvergence`.
   Each one ends a compute encoder.
4. Stop encoding the 163-pass recovery finish every advance, and gate
   `mgSaveAccepted` / `mgRestoreRejected`, from the same lagged evidence that
   already sizes the cycle budget. That is a quarter of the encoded pressure
   passes at budget 1, plus 18 MB of copies.

Expected: 4 to 6 ms of CPU per advance at 64³ from 1 to 3, and 2 to 3 ms of
GPU plus 1.5 ms of CPU from 4. Item 4 changes which passes run, not what they
compute; it is exact whenever the finish would have been a no-op, and needs
the same one-advance recovery story the cycle budget already has.

### WP3. One census, one class texture, launches that shrink

1. One census pass at the head of the step writes the `r32uint` tile-class
   texture: FINE, SHELL, TRANSPORT, SHARPEN, PHI, plus the static solid-free
   bit from WP1. Sharpen tiles, the phi region and page flags become bits or
   derived reductions of it. Reconcile the `>` versus `≥` dust-floor mismatch
   deliberately rather than by accident.
2. Run page mark and compact passes only while the overlay is visible.
3. Read the class once per thread, not once per sample: `sampleVelocity` loads
   it once for three components; de-atomicise the 4h coarse table.
4. Build per-class compact tile lists in the census and dispatch them
   **directly, host-sized from a one-advance-lagged count with margin**, the
   kernel guarding `index >= count`. This is the only data-dependent launch
   scheme that has paid here; GPU-indirect launches cost +40.7% on a whole
   advance. Stages that store on their dead arm need the union with the
   previous set, as the old solve window did, or a persistent output.
5. Give pressure the map: at L0 and L1, dispatch liquid tiles plus a one-tile
   halo rather than the level. `uniform-geometric-tall-air` measured the
   full-level dispatch at 57% of the air cost.

Items 1 to 3 are exact and mostly remove passes and loads. Items 4 and 5 are
where airy scenes (fig7 at 10 to 53% live tiles, garden at 1%) get their
multiple; at mini64, which is 99.6% live, they return nothing. Exactness for 4
holds if skipped tiles were provably no-ops; for 5 it needs air rows to be
identity rows, which they are.

### WP4. Pressure dispatch count (regime A)

1. Specialise `mgActiveId` to `vec3i(gid)` in the dense shader as the paged
   shader already does: 5 dead storage loads per invocation. Exact.
2. Colour-compacted red-black: dispatch half the threads, and stop the
   wrong-colour lanes doing a pass-through store. Removes the 16/16 divergence
   inside every SIMD group. Exact if done in place; needs care with the
   ping-pong.
3. Collapse the coarse levels. Half the plan's passes touch 1,280 cells. An
   8³ level fits one workgroup: sweep it inside one dispatch using workgroup
   memory and barriers instead of 12 dispatches at the launch floor. Probe it
   first on a standalone kernel: `mgSolveCoarsest` today is serial at 0.2%
   occupancy and costs 1.41 ms per advance, so a serial extension would lose,
   and `home-halo-is-an-occupancy-cliff` says large workgroup arrays have
   their own cliff.
4. AoS row record for the smoother (16 loads to 11) and drop the allocated but
   never-read V, phi and residual parities: 88 to 96 bytes per cell of 3D textures today.

Expected: item 3 is the large one, up to half of pressure's dispatches, so 3
to 4 ms of GPU and as much CPU at 64³. It is not exact unless the sweep order
is preserved, which a barrier-stepped red-black can do.

### WP5. Kernel subtraction (exact, cheap, do alongside)

- Specialise the dead 512-iteration `uvAgreementShift` loop and the
  64-iteration `uvSeedPhi` out of `uvAdvectPhi` with `override` constants.
  `never-taken-branch-priced-the-kernel` is precedent: this is the top single
  candidate for the 2.96 ms task.
- Hoist the tile-membership test out of `uvLimitedFlux` (called 6 times per
  thread, 13 atomics).
- Load `uvTarget`'s 8 vertices once instead of 64 taps.
- Enable the cached sharpening arm at 64³ and below.
- Flip bindings instead of the three full-field copies.
- Remove the 5,120-byte `var<workgroup>` in surface-volume if the reduction
  can be restructured; measure first.
- Diagnose `FIM indirect update 2` (0.72 ms at 2.7% occupancy) and the
  unconverged FIM front at 2 sweeps.

### WP6. Data size and packing (regime C)

Only `uvNormalizeRows` and `uvNormalizeDonors` are limiter-bound, 3 ms
together, so this is last at 64³ and rises with grid size.

- Hot/cold split of the 80-byte edge record; the minimal layout is 10 to 22
  bytes. Weights as unorm16 are not exact; donor indices as packed 3-bit
  offsets are.
- Donor sums: four full clears per advance (25 MB at 64³, 201 MB at 128³).
  Clear only live tiles from the class map, or tag with a generation.
- `rgba32float` velocity wastes `.w`; splitting into planes is likely neutral
  or negative because the three staggered footprints overlap. Do not do it.

### Not worth doing

f16 arithmetic; SIMD-width or workgroup-size tuning; per-page dispatch;
GPU-indirect launches for stage sizing; dimension specialisation through
`override` (105 to about 270 pipelines against a 7.2 s startup); anything
justified by pass count alone on the GPU side, since Dawn already merges 29
passes per encoder.

## 5. Order of work

1. WP1 step 1 and WP5 first bullet: two subtractions, one interleaved xctrace
   A/B. They test the regime-B model cheaply.
2. WP2 items 1 to 3: mechanical, exact, CPU-side.
3. WP1 steps 2 to 5 with WP3 items 1 to 3: the class texture and the geometry
   fields share a census pass and a binding change.
4. WP2 item 4, then WP4 item 3 behind a standalone probe.
5. WP3 items 4 and 5 on fig7 and garden, where they matter.
6. WP6.

Gate for every step: V conservation, no NaNs, the existing uniform Dawn lanes,
and paired interleaved timing. Exact items additionally compare state hashes.

## 6. Large scene (fig7 128³) and the CPU/GPU question

From `xctrace-capture-large.md`.

**An app advance costs max(CPU, GPU); only the harness pays CPU + GPU.**
`webgpu-smoke-executor.ts:3179` awaits `awaitFrameCompletion()` after every
advance, and the solver implements that as a full queue drain, so
`FLUID_AWAIT_EVERY_STEPS` is dead for this method. Unfenced, the same advances
cost 21.75 ms (mini64) and 51.36 ms (fig7), which is GPU time alone; fenced
they cost 33.87 and 64.24, matching the harness within 1%. CPU is 11.2 µs per
encoded pass, flat across scenes. The renderer never fences per frame.

Consequences for the plan:

- Section 1's "CPU is on the critical path" holds for harness lanes only. In
  the app the advance is GPU-bound at both sizes, so WP2 (CPU) drops below
  every GPU lever. It still frees 3 to 4 ms of main-thread time per advance.
- Harness timings of GPU-side changes are diluted by the CPU term. Measure GPU
  changes with the unfenced drain probe (`scratchpad/overlap-probe.mts`,
  `PROBE_ARMS=drain`), which also pins the pressure budget so a diverged
  trajectory cannot change the pass count.

At 128³ the frame moves from "a thousand tiny launches" to "ten big kernels
plus a thousand tiny launches": firings over 1 ms are 1% of firings and 36% of
busy time. Family shares, fig7 against mini64: pressure V-cycles 31.9% vs
48.2%, surface publication 15.5% vs 5.7%, pressure setup 11.3% vs 5.1%, volume
gather 10.7% vs 12.1%, extension 9.9% vs 5.8%. `mgSmoothColour` at L0 alone is
8.33 of 14.87 ms; coarse levels never drop below 6 to 10 µs per firing.
Single-dispatch kernels worth attacking next: `Total surface volume: measure`
3.4 ms (last-level-cache limited, the 5 KB workgroup array), `Advect page
vertex phi` 3.3 ms, `rho-prime and face authority` 2.1 ms, `Phi support
census` 1.4 ms.

## 7. Measured results (2026-09-21)

Unfenced GPU-bound ms per advance, 58 advances after 2 warm-up, pass count
pinned, arms interleaved, 2 reps (reps agree within 0.5%; min shown). One
contended rep (Chrome on the GPU) read 40.9 ms and was discarded: contention
only adds, so use the minimum.

| change | mini64 | fig7 128³ | exact vs HEAD | kept |
|---|--:|--:|---|---|
| baseline (HEAD) | 23.41 | 54.65 | | |
| drop dead `open` local in `pressureSurfacePhi` + dead face re-test | −0.9 | −2.4 | bit-identical | yes |
| `cellOpenFraction` stops re-testing the static voxel | −0.84 | −1.6 | rounding-level | yes |
| pressure setup and projection read V_face from `velocityD` | −1.0 | −2.4 | value-identical between arms | yes |
| lean `uvAdvectPhi` (agreement loops compiled out) | 0.0 | 0.0 | bit-identical | **no, reverted** |
| one compute pass per multigrid run, group 0 bound once | 0.0 GPU | 0.0 GPU | bit-identical | yes |
| **all kept changes** | **20.95 (−10.5%)** | **49.00 (−10.3%)** | | |

CPU inside `advanceTo`, from the batching change alone: 8.4 → 5.1 ms (mini64),
10.8 → 6.5 ms (fig7); encoded passes 753 → 115 and 938 → 119. Harness wall on
mini64, 120 steps: 35.2 → 28.5 ms per advance.

What the measurements say about the model:

- **The load-chain model holds.** Removing ten dependent loads from
  `pressurePhi` paid 4%; removing 600 loads per cell of V_face recomputation
  paid another 4.7%. Neither touched a single arithmetic operation.
- **The dead-loop hypothesis is refuted.** A branch on a uniform is coherent
  across every thread, and the 512- and 64-iteration bodies behind it cost
  nothing measurable. `never-taken-branch-priced-the-kernel` does not
  generalise to this kernel. WP5's first bullet is withdrawn.
- **Bit-exactness against HEAD is not a usable bar for shader edits here.**
  The `cellOpenFraction` change returns the same value by construction (0 or 1
  in a body-free, terrain-free scene), yet the state differs by 4e-8 relative
  after four steps and then diverges chaotically. Metal re-fuses the callers'
  float math when the loads move (`atomic-load-is-an-fp-barrier`). The bar used
  instead: the all-off arm stays bit-identical to HEAD, arms that only differ
  in a runtime flag match each other, and 120-step health is equivalent
  (volume drift −0.21% vs −0.19%, pressure converged, max speed 3.06 vs 3.04).
- Gates, all changes on: `test:dawn:uniform-volume` and `uniform-pressure`
  fail the same four tests they fail at HEAD (mini32 far-wall conservation and
  its parent, the slice overlay, Figure 9 frame 9); boundaries, native stages
  and garden pass.

The arms live behind `lib/methods/uniform/uniform-ab-switch.ts`
(`FLUID_UNIFORM_AB_OFF=opentest,facetest,openlocal,batch,facecache`, or `all`).
It is temporary and should go once the program's changes are accepted.
Isolated xctrace captures should pass `FLUID_UNIFORM_AB_OFF=batch`, or set
`FLUID_UNIFORM_MG_LEVEL_LABELS=1`, to keep one labelled pass per kernel.

Next, in order: the V_i field and the static solid-free tile bit (WP1 steps 3
and 4), which attack the remaining 185 open-fraction evaluations per cell; then
`Total surface volume: measure`; then the L0 smoother (WP4 items 1 and 2).
