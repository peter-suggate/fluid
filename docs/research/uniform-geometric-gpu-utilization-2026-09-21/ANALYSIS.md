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

## 8. Second round (2026-09-21, after commit 84a0c558)

Same protocol, plus a third scene with real solids and terrain:
`hero-garden-hose` (144×96×96). Each row is one switch flipped against
everything else on; min of 2–3 interleaved reps.

| change | switch | mini64 | fig7 128³ | garden | exact | kept |
|---|---|--:|--:|--:|---|---|
| solid-bitfield header from host constants, not read back | `solidheader` | −0.57 (−2.7%) | −1.06 (−2.2%) | −0.97 (−3.1%) | see below | yes |
| `measure`: all-zero band corners ⇒ one phi load | `measurelean` | −0.03 | −0.75 (−1.6%) | n/m | bit-identical | yes |
| `measure`: K cells per thread ahead of the tree (K=4/8/32) | — | +0.3/+1.0/+1.8 | −0.3/+0.2/+1.8 | | rounding-level | **no, reverted** |
| multigrid `mgActiveId` compiled to `gid` on a full lattice | `mgstaticid` | −0.66 (−3.2%) | −0.90 (−1.9%) | n/m | bit-identical | yes |
| main shader window header compiled out on a full lattice | `staticid` | −0.30 (−1.5%) | −0.24 (−0.5%) | n/m | bit-identical | yes |
| in-place red-black smoother over half the lattice | `inplace` | −0.06 | −2.68 (−5.8%) | −1.88 (−6.5%) | bit-identical | yes |
| **everything, vs HEAD (`all`)** | | **23.52 → 19.41 (−17.5%)** | **55.05 → 43.27 (−21.4%)** | **36.14 → 27.22 (−24.7%)** | | |

What changed in the picture:

- **The V_i texture (WP1 step 3) is withdrawn.** In a body-free scene
  `cellOpenFraction` is a bounds test and one bit out of a packed word. The
  cost was never the evaluation, it was `staticSolidVoxelOccupied` re-reading
  its own four-word header (magic, three shape words) through a `read_write`
  binding on every call: five dependent loads where one is needed. The header
  is the host's constant (the solver always packs the field over the lattice
  plus a one-cell halo), so it now comes from `params.dimsDt`. A 4-byte-per-cell
  texture would have been 32× the bytes of the bitfield it replaced. A cached
  field is still the right answer for rigid-body scenes (eight corner tests per
  body per call, no broad phase), and only there.
- **With the header fix in, the whole set is bit-identical to HEAD again** on
  mini64 (120 steps: volume 94032.829, max speed 3.0352, residual, FIM count
  all equal) and on fig7 (60 steps). The "rounding-level" drift recorded in
  section 7 came from the `cellOpenFraction` edit interacting with those header
  loads; it is gone. Garden, which exercises the solid and V_face paths for
  real, still differs from HEAD at rounding level (max speed 2.13 vs 1.99 at 60
  steps) and passes its gate.
- **`measure` is latency-bound, not traffic-bound.** Folding K cells into each
  thread cuts the 80-byte-per-leaf workgroup tree by K and made the pass
  slower at every K. The pass is 2M short threads whose cost is the serial
  chain inside each one; lengthening the chain loses more than thinning the
  tree gains. The remaining lever is not visiting the 97% of cells outside the
  band at all, which needs the WP3 band list.
- **Header read-back is a pattern, not an instance.** `mgActiveId` read five
  storage words per thread, in every level kernel, to add an origin the host
  seeds at zero and nothing rewrites when the active region is off (always,
  under paged domains). The main shader's `activeId`/`activeVertexId`/
  `uvWindowMin/Max`/`pressureWindowLattice` did the same. Both are now compiled
  out for full-lattice solvers. The phi kernels are deliberately left alone:
  `phiRegion` rebinds slot 29 to a census-driven window of its own.
- **The smoother no longer copies.** A six-neighbour update of one colour reads
  only the other colour, so both colours share one `read_write` r32float
  texture and each pass dispatches only its own colour (x spans half the
  lattice). `mgSmoothColour` visited all cells of both colours twice per sweep
  just to carry the ping-pong. Every sweep-exit value is unchanged. Guarded to
  full-lattice, dense-storage, 3D scenes without depth symmetry (its colouring
  ignores z, so z-neighbours would race); a live edit that introduces depth
  symmetry rebuilds the plan on the ping-pong smoother.

Gates with everything on: unchanged from HEAD's known-red set (section 7);
boundaries 7/7, native stages 1/1, garden 1/1.

Switches added: `solidheader`, `measurelean`, `mgstaticid`, `staticid`,
`inplace`.

### 8.1 Third round: a fresh capture, then the pressure launches and loads

Fresh xctrace capture of fig7 with everything above on
(`artifacts/xctrace-uniform-geometric-fig7-after/`, isolated regime, 41.12 ms
attributed per advance):

| task | ms/advance | note |
|---|--:|---|
| pressure, all levels | 16.46 | 832 calls. L0 9.25, L1 1.58, L2 1.07, L3 1.45, L4 1.63, L5 1.22, L6 0.25 |
| of which smoothers | 11.23 | L0 alone 5.70 = 152 × 38 µs at 47% occupancy |
| of which `mgBuildFinestTopology` / `mgBuildFinestRhs` | 0.46 / 0.42 | were 3.4 / 1.1 before `facecache` |
| `Total surface volume: measure` | 2.63 | |
| advect page vertex phi | 2.54 | |
| `uvNormalizeRows` | 1.87 | |
| semi-Lagrangian advection | 1.64 | |
| rho-prime and face authority | 1.52 | |
| phi support census | 1.36 | |
| `uvFinishDonorSums` | 1.25 | |
| refresh corrected surface targets | 1.16 | |
| `uvNormalizeDonors` | 1.01 | |

Pressure is 40% of the frame and L1–L6 together cost 7.2 ms for a few percent
of the cells: those levels are launch-bound. Two changes followed, same
protocol:

| change | switch | mini64 | fig7 128³ | garden | exact | kept |
|---|---|--:|--:|--:|---|---|
| one dispatch per coarse smoothing visit (12 colour passes in one workgroup, `textureBarrier` between colours) | `fusevisit` | 19.43 → 18.92 (−2.6%) | 43.36 → 42.75 (−1.4%) | n/m | bit-identical | yes |
| six neighbour liquid flags packed into the baked coefficient `w` | `liquidmask` | 18.79 → 18.26 (−2.8%) | −0.4 to −1.8% | 0 to −1.1% | bit-identical | yes |

- **The fused visit wins only where a lane is short.** `mgSmoothVisitInPlace`
  runs a level's whole visit (`2 × sweeps` colour passes) as one workgroup of
  1024 lanes striding the colour's cells, with `textureBarrier()` where the
  pass boundaries were and the skip flag read once through
  `workgroupUniformLoad`. It replaces 12 launches by one, but each lane then
  runs its cells serially. At a 8192-cell cap with 256 lanes it *lost* 7.7% on
  mini64; it pays at ≤ ~25 serial updates per lane, hence the 1000-cell cap
  (`FLUID_UNIFORM_FUSED_VISIT_CELLS`) with 1024 lanes. That bounds what
  launch-fusing can ever return here: the shipping frame's launch floor is
  about 6 µs of GPU time, not the 7–13 µs the isolated capture shows, and one
  serial cell update costs ~0.25 µs. CPU encode drops ~1 ms (387 pressure
  passes on mini64). Not used under indirect cycle dispatch, which would launch
  it once per tile record.
- **The flag mask pays on the small scene only.** With the mask an update
  reads four coefficient texels, not seven, in both smoothers and in `mgApply`
  (every residual kernel). mini64 −2.8%; the two large scenes move by about
  their noise. Independent texel loads are not what bounds the L0 smoother: its
  cost is the dependent chain (coefficient → pressure → divide → store) per
  thread, the same regime-B shape as the rest of the frame.
- **Value-identical is not bit-identical.** The first mask version also
  selected the coefficient by hand (`n` odd ⇒ this cell's texel, else the
  neighbour's). Same values in, and mini64 looked 5.4% faster, but Metal
  reassociated the two six-term sums, the native fields stopped matching the
  paged arms bit for bit (`uniform-pressure-layout-dawn`, native-stages
  1.9e-5), and max speed was 7.16 vs 6.47 m/s after 16 steps. Bisected by
  swapping halves of the expression: the flag source is free to change, the
  coefficient must keep coming through `mgCoefficient`. Any future edit to the
  smoother or `mgApply` should be fingerprinted at 16 steps before it is timed.
- WGSL traps met on the way: `pass` is a reserved word (silent "Invalid
  ShaderModule"); `textureBarrier()` and `read_write` storage textures work in
  Dawn/Metal.

### 8.2 Dead launches: the recovery finish

Histogramming the L0 smoother's 152 dispatches per advance in the capture gave
two populations: ~24 at ~128 µs (the work) and **~128 at 17–20 µs that do
nothing**. Those are the recovery finish: 8 batches × 8 sweeps × 2 colours,
always encoded, gated on the GPU by a flag that only a rejected cycle sets. A
census over 150 steps each of mini64, fig7 and the garden found recovery live
in **0 of 450** steps. Each dead launch still spawns 1.1M threads to read the
gate and return (~11 µs above the ~6 µs launch floor), and each batch's
`mgSaveAccepted` re-copies a field onto an identical copy (33 µs of real
traffic, 8 times). About 3 ms of fig7's 41.

The lagged cycle budget already exists because "a skipped pass still costs its
launch floor"; the finish was the part it never reached. Not encoding it would
change results on exactly the frames that need it, so instead the finish gets a
second launch shape with identical arithmetic:

- `mgSmoothRowInPlace`: one thread per run of 8 same-colour cells along x
  (`MG_ROW_SEGMENT`), calling the same `mgSmoothCellInPlace`.
- `mgSaveAcceptedQuiet` / `mgRestoreRejectedQuiet`: the two commits, same
  segmenting, returning before any copy while word 22 (recovery entered) is
  clear. Until a cycle is rejected the accepted copy already equals the working
  field, so both are value no-ops there.
- The host picks per step from the lagged readback (now 28 bytes, word 22
  included): per-cell launches when the last observed step entered recovery or
  no sample has landed (HEAD's behaviour), segmented launches otherwise. The
  choice is between two launches of the same arithmetic, so a stale answer
  costs time and nothing else. Fixed-budget mode has no readback and keeps the
  per-cell launches.

| quiet finish | mini64 | fig7 128³ | garden | exact |
|---|--:|--:|--:|---|
| segment 8 (kept) | 18.25 → 18.09 (−0.9%) | 42.00 → 40.13 (−4.4%) | 26.72 → 25.60 (−4.2%) | bit-identical |
| whole rows (65 cells/thread) | | 40.36 | | bit-identical |
| segment 8 on *live* sweeps (experiment) | +3.2 ms | +4.0 ms (live sweep ≈ 2.3×) | | |
| whole rows on live sweeps (experiment) | +6.4 ms | +11.5 ms (live sweep ≈ 4–5×) | | |

Whole rows are 0.2 ms cheaper dead and twice as slow live. With segment 8 a
surprise recovery costs its first one or two frames ~2.3× on the finish
(≈ +20 ms on fig7) before the readback flips the launch back; with whole rows
that hitch would be ≈ +60 ms, on an impact frame. The same experiment is the
reason this shape is *only* for launches expected dead: serial cells per thread
lose on live work here, as they did in `measure`.

Live-path proof: `tests/uniform-pressure-safety-dawn.test.ts` injects faults
that force recovery (64 sweeps, exhausted). A scratch copy printing the
pressure stats per layout × fault × frame was identical in all 24 cases with
the quiet launches forced and off, and a pipeline counter confirmed 1024 quiet
sweep launches and 64 of each quiet commit actually ran.

Not done, needs a decision: not encoding the finish at all when the last
sample was clean would return a further ~1 ms per advance on mini64 (160
launches × the floor), but a frame where recovery first becomes necessary would
go without it. That weakens a safety net, so it is left alone.

### 8.3 Withdrawn: one class load per sample point

`sampleVelocity(p)` calls `sampleVelocityComponent` three times and each opens
with the same atomic two-level class load for the same `p` (section 2 listed
this). Loading it once is value-identical and was measured two ways:

| variant | mini64 | fig7 128³ | garden | exact |
|---|--:|--:|--:|---|
| load once, branch once around all three components | −2.8% | −1.2% | −2.7% | no: mini64 and garden diverge by step 16 |
| load once, branch kept inside each component | −1.6% | −1.1% | −2.1% | bit-identical on all three scenes for 60 steps, **but `uniform-native-stages` goes red** |

The second variant looked exact and is not. That gate compares the native
stages against a domain-free arm compiled from different source; the two sit
2.6e-6 apart at HEAD against a 1e-5 tolerance, and the hoist moved one arm's
rounding enough for `long-dam` frame 7 to reach 1.9e-5. Confirmed by flipping
only this switch (green off, red on). Keeping it would have meant widening that
tolerance for ~1.5%, so it is reverted, and the sampler is byte-for-byte HEAD.

What this adds to the rule from 8.1: a 16-step fingerprint on the three probe
scenes is necessary, not sufficient. **Any edit to a kernel both arms compile
has to pass `test:dawn:uniform-native-stages` and the layout oracle before its
timing means anything.**

### 8.4 Dead workgroups: skip the reduction tree, not the launch

Three dense passes end every 4×4×4 workgroup in a 64-lane reduction tree of six
or seven `workgroupBarrier()`s: the phi support census
(`writeActiveWorkgroupSummary`), `measure` in the surface-volume correction, and
`uvBalanceMeasure`. Over air every lane contributes the identity, and the tree
still runs: seven barriers where 64 threads rendezvous to add zeros.

Each lane now raises one `var<workgroup>` atomic flag if its contribution is
live (tested on **bits**, so a −0 takes the tree), and a single
`workgroupUniformLoad` of that flag lets a dead workgroup publish the identity
from lane 0 and return. The load is one barrier instead of seven, and it is
uniform control flow, so the tree's own barriers stay legal inside the branch.
Live workgroups run exactly the code they ran before.

| `deadgroups` | mini64 | fig7 128³ | garden |
|---|--:|--:|--:|
| off → on | 18.09 → 18.07 (−0.1%) | 39.16 → 37.92 (−3.2%) | 25.03 → 24.18 (−3.4%) |

Bit-identical on all three scenes; native stages and the layout oracle green.

### 8.5 Donor tiles: the decode only where a row can look

`uvFinishDonorSums` decodes the six-limb exact column sums into one float per
cell. It runs four times a step and was the one transport pass still dense:
six planar limb loads and a store per cell over the whole lattice, 1.25 ms on
fig7, while every other transport pass already runs on the TRANSPORT tile list.

Its output has two readers. `uvFallback` reads a built row's own word;
`uvNormalizeDonors` reads the words of the donors a built row samples. Rows are
built on TRANSPORT (seeds dilated by m tiles) and a donor lies within ⌈D⌉+1
cells of its row, which is the predicate's own m₀ = ⌈(⌈D⌉+1)/4⌉ tiles. So the
separated dilation now carries a fourth bit, DONORS = transport seeds dilated
by m + m₀ + 1, and the decode returns outside it. The extra tile is margin: a
short TRANSPORT set only stalls a front, a short DONORS set would read a stale
sum. A displacement past the transport cap makes DONORS the whole lattice. The
bit travels in the scan planes and lands in the x plane, which is dead once the
y scan has read it, so the class word keeps exactly the three bits the sampler
and the overlay read.

| `donortiles` | mini64 | fig7 128³ | garden |
|---|--:|--:|--:|
| off → on | 18.06 → 18.07 (0) | 38.77 → 37.92 (−2.2%) | 24.70 → 24.18 (−2.1%) |

Bit-identical on all three scenes.

### 8.6 V_face compiled once per geometry change (WP1, exact)

`storeExtrapolationAuthority` writes rho' and the three V_face values of every
cell each step. V_face is three eight-sample face queries over the solid mask,
terrain and rigid bodies; in a step with no body and no edit it is the value
already sitting in `velocityD`, which nothing else writes outside MacCormack
and the stage audit. So the host keeps one bit, "velocityD holds V_face for the
geometry as it stands", and on such a step launches a second entry point that
stores rho' alone: one phi load a cell.

The bit is dropped by `applySceneUniforms` (the live voxel-stroke path and any
scene change), by `applyRuntimeValues`, by any step that carries a rigid body,
and by any store that did not cover the whole lattice (a windowed store leaves
cells outside it holding whatever geometry they last saw).

| `authoritystatic` | mini64 | fig7 128³ | garden |
|---|--:|--:|--:|
| off → on | 17.98 → 17.89 (−0.5%) | 37.92 → 36.51 (−3.7%) | 24.20 → 23.28 (−3.8%) |

Bit-identical on the three probe scenes. Invalidation was checked on
`tests/uniform-live-solid-edit-dawn.test.ts` (a wall drawn ahead of the front,
then undone, 92 steps): passes on both arms, and a scratch copy hashing the
final V texture gives the same hash with the switch on and off.

### 8.7 The diagnostics reduction is owed, not encoded

`Uniform diagnostics reduction` is a dense pass of contended `atomicAdd`s at the
tail of every step. No kernel loads the words it writes; its one reader is
`readStats`, and the app calls that only on pause ("live frames never map
solver state"). The step now records the pass as owed and `readStats` encodes
it, once, ahead of its copies; the bind group is parity-free, so it reads the
same fields the step left. A paused presentation refresh pays the debt first,
because it rewrites the represented surface the reduction reads. A traced step
keeps the pass, since the final phase closes on it.

| `lazystats` | mini64 | fig7 128³ | garden |
|---|--:|--:|--:|
| off → on | 17.89 → 17.86 (−0.2%) | 36.47 → 36.14 (−0.9%) | 23.31 → 23.16 (−0.6%) |

Looked at and left: **`Refresh corrected surface targets`** (1.16 ms on fig7)
is dense and evaluates `uvTarget`'s 64 phi loads per cell over air, where the
answer is zero. `uvGather` already skips it outside TRANSPORT, but the refresh
runs after the volume correction has shifted phi by up to
0.2·band·h·|∇φ|, and nothing bounds |∇φ| of un-redistanced far phi. The skip
is right in practice and not provable, so it needs a min-phi census to be
exact; not taken. **`Advect page vertex phi`** has no closed form over air
either: far phi is a transported value, not a clamp.

**Everything kept, vs HEAD:** mini64 23.52 → 17.86 (−24%), fig7 55.05 → 36.14
(−34%), garden 36.14 → 23.16 (−36%) GPU-bound ms per advance.

`tests/uniform-pressure-layout-dawn.test.ts` changed: its launch-shape equality
between the native and paged-logical plans now exempts `mgSmooth*` entries
(the native plan halves x and fuses visits; the paged plans do neither) and
asserts the native plan contains `mgSmoothVisitInPlace`. The bit-for-bit field
comparison over 12 frames × 2 fixtures that follows it is untouched and is the
oracle that caught the reassociation above.

Gates with everything on: HEAD's known-red set and nothing else (Figure 9 in
`uniform-pressure`, 22 pass; mini32 far-wall conservation, its parent and the
slice overlay in `uniform-volume`, 10 pass); native stages 1/1, boundaries 7/7,
garden 1/1.

Switches added this round: `fusevisit`, `liquidmask`, `rowsweep`, `deadgroups`,
`donortiles`, `authoritystatic`, `lazystats`
(`FLUID_UNIFORM_ROW_SWEEP=force` for measurement, `FLUID_UNIFORM_ROW_SEGMENT`).

Next, in order: the resident coarse sub-hierarchy (L3–L6 are 4.5 ms of pure
launch floor on fig7; one dispatch per V-cycle leg would remove most of it and
is the natural extension of the fused visit); then the WP3 band/tile list,
the only route left into `measure`, the censuses and the dense volume passes;
then a broad phase for rigid bodies in `bodySolidFractionAt`.
