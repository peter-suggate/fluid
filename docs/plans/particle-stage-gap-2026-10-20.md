# Particle stage: the gap to the bench, pass by pass

2026-10-20. Target: `nbflip-figure-7-pour` at 1/60 s, under 17 ms a frame on the M1 Max.

On the pour the particle stage takes 7.30 ms at 0.61M particles (12.0 ns a particle). The same design in the standalone bench (`particle-brick-slabs-2026-10-19.md` §B) ran 2.36–2.62 ms at 818k (2.9–3.2 ns a particle). Its bound is 1.15 ms at 818k (1.4 ns a particle). Today's stage is about 4× off the bench and 8.5× off the bound.

This doc does five things:
- attributes the 4× pass by pass (§2), using knock-outs and a reading of each kernel;
- names the priors behind each pass that is more than 2× off its bound, with a different approach for each (§3);
- costs the move and the snapshot through hardware-filtered textures (§4);
- answers why the pour's particle count keeps growing after the stream ends (§5);
- ranks the structural changes that close the gap (§6).

Spill headroom at peak is recorded in §7.

## 1. Sources and method

- **Pass means: base7.** This is today's tree with UNIFORM_CORE_VELOCITY off.
  - Frames 94–100, profiled and split per entry point.
  - 585k–626k particles, mean 0.61M. The ns-a-particle figures below divide by 0.61M.
- **Knock-outs: ko1.** One pour run, frames 86–111, at 0.50M → 1.05M particles.
  - Each knock-out is a probe-only shader patch (the probe's `PATCH`). A bit in the NB state's free word 31 turns it on.
  - The probe writes that word before chosen frames: two frames per arm, with base frames on either side. Arms A and B were repeated later in the run.
  - Every knock-out is a branch on that word, so the state up to the knock-out frame is the base run's.
  - Deltas are in ns a particle: the knock-out frames minus the mean of the neighbouring base frames.
- **Noise.** Profiler timestamps come in steps of about 65 µs. Short passes swing ±0.15 ns a particle between neighbouring base frames (refine and seed fall in arms that do not touch them). Only deltas above about 0.2 ns a particle are quoted as findings.
- **What could not be knocked out cleanly:**
  - The census prediction (transferRefine). Removing it breaks the residency closure: a particle arrives in an unstored page, which is fatal.
  - The append in the move, and the sort's scatter, because consumers read their output.
  - Those passes are attributed by reading the kernel, and are marked as such.

## 2. Pass by pass

Bench kernels are scaled to 0.61M by particle count: FUSED 1.12 ms, DEPB 0.44 ms, FIN 0.15 ms, velocity atlas 0.15 ms.

| Production pass | ms | ns/p | Bench equivalent (ms at 0.61M) | What makes production slower (knock-out deltas in ns/p) |
|---|---|---|---|---|
| advectParticles (fast move) | 1.947 | 3.19 | FUSED (1.12), which includes the bench's seed search (0.49) | Fat-kernel predicates and paged taps; see §2a |
| advectParticlesDeferred | 0.197 | 0.32 | FUSED's slow lane (about 1100 records in the bench) | About 5k slow records plus 8–13k spill records through the general sampler, at about 13 ns each |
| advectWalk | 0.197 | 0.32 | none (hull-proven in the bench) | Traces the fast launch cannot prove uncut, walked at half a cell a stride |
| transferDeposit | 1.152 | 1.89 | DEPB (0.44) | Scatter −0.92 to −1.03; flush −0.03 to −0.07. About 0.8 is per-tile overhead; see §2b |
| particle sort | 0.618 | 1.01 | none (bench records stay brick-bucketed) | Not knocked out; see §2c |
| seed (3 dispatches) | 0.524 | 0.86 | FUSED's seed search (0.49 of FUSED) | seedCells ≈ −0.2 to −0.37. seed and seedSources append; about 50% of seeds spill (§7) |
| transferSeams | 0.487 | 0.80 | none (no 4h seams in the bench) | Gathering the 4h owner faces from cell runs: −0.35 to −0.69. A floor of about 0.3 remains |
| transferRefine | 0.477 | 0.78 | none | Read, not knocked out: a second full read of every record after DEPB, then about 0.19 ms of direct prediction for seeds and spills (nbRefineDirect) |
| snapshot / cache | 0.421 | 0.69 | velocity atlas (0.15) | The 4h sampling cache: one lane a texel over every stored block's 10³ texels, whether or not it holds particles |
| snapshot / snapshot | 0.393 | 0.64 | folded into FUSED in the design | The velocity sample: −0.32 to −0.43. The rest is a record pass over the extent, holes included |
| transferProperties (FIN) | 0.374 | 0.61 | FIN (0.15) | One 64-lane workgroup per fine tile. A flagged tile stages 125 paged vertex depths (umVertexAt), 27 tile heats and 27 bandPhi samples before blending; an unflagged tile copies. With the scatter knocked out (no flags) the pass falls by 0.40–0.52 |
| snapshot / unitVelocityTaps | 0.234 | 0.38 | velocity atlas | Unit-halo fill from the snapshot slots, for every unit tile |
| particle placement + slabs | 0.234 | 0.38 | the ALLOC in FUSED | Six dispatches: slab mark, table, allocate, place, overflow, prepare |
| transfer + owners | 0.047 | 0.08 | — | — |
| **Total** | **7.302** | **11.97** | **1.76–1.95** (chain), 0.86 bound | |

Summed by role:
- **Move:** 2.34 ms against FUSED's 1.12. Adding seeding, it is 2.86 against FUSED's 1.12, which includes seeding.
- **Transfer:** 2.54 ms against DEPB + FIN's 0.59. The seams and the census have no bench equivalent.
- **Snapshot chain:** 1.05 ms, against nothing in the design.
- **Sort and placement:** 0.85 ms, against nothing in the design.

### 2a. The move (3.19 ns/p; base 3.23 in ko1)

The kernel (`nbMove` → `nbAdvect`) takes one workgroup per 64-record chunk of an input slab. Per record it runs, in order:
1. G2P: a velocity sample, then nbSampleCold, nbSparseAge, particleDepth, nbTargetHeat, the rigid contact, nbParticleBlocked, gridSupported and nbTargetHeat again.
2. Two more velocity samples, for k2 and k3.
3. nbPointNearIn and the hull.
4. The append turn: a workgroup histogram over 27 bricks, two barriers, and up to 27 device atomics.

Knock-outs:

| Arm | Δ ns/p | Reading |
|---|---|---|
| All three velocity samples return 0 (no page lookups, no fetches) | −0.64 / −0.83 | Velocity samples ≈ 0.7 ns/p |
| Only the texture fetches skipped (page lookups kept) | −0.38 / −0.47 | 9 rgba32f trilinear fetches ≈ 0.42; page lookups (umSampleTile → umCachedFineTile → umUnitHaloHome) ≈ 0.3 |
| particleDepth + nbSparseAge + gridSupported skipped | −0.95 | G2P predicates ≈ 0.95 ns/p |
| gridSupported only | −0.24 | Up to 8 × (umOwnerAt + phase load) |
| Implied particleDepth + nbSparseAge | ≈ −0.7 | Paged φ: umVertexRun plus 8 metric loads. Sparse: a cell run plus up to 9 row runs |

Left after both arms: about 1.5 ns/p. That covers:
- the record read and write (48 B each way);
- the append turn;
- the slab job's binary search and its 27-brick setup;
- nbSampleCold, the two target-heat loads, nbParticleBlocked, nbPointNearIn and the hull.

The bench's whole FUSED, seed search included, is 1.84 ns/p. Its samples were brick-local, and it had none of these predicates.

### 2b. The deposit (1.89 ns/p)

transferDeposit runs one 192-lane workgroup per occupied 4³ tile. At this density that is about 50 samples, so about 70% of the lanes sit idle in the sample loop. Per tile it:
- clears 2 × 756 workgroup atomics;
- stages 27 topology words and 216 paged gridLiquidCell lookups;
- classifies each sample (8 corners), writing before.w and four motion words;
- scatters 3 × 27 nodes × 2 workgroup atomics per sample;
- flushes up to 756 slots with two device atomics each.

The scatter knock-out (−0.92 to −1.03 ns/p) puts 162 workgroup atomics at about 0.95 ns, roughly 170 G/s. That is the measured workgroup-atomic ceiling (218 G/s). The flush is cheap (−0.03 to −0.07). The other ~0.8 ns/p is per-tile overhead: clears, staging and classification. The tile window's halo is 252 nodes per axis for 64 owned faces, a ratio of 4. The bench's brick window (DEPB, 11×10×10) has a ratio of about 1.6, with an eighth of the per-tile fixed work.

### 2c. The sort (1.01 ns/p, read only)

`slabSort` takes one 256-lane workgroup per slab. Per slab it:
- counts 512 cells;
- runs an 8-round scan;
- scatters records into the current bank, plus 16 B of compact position into links;
- dead-marks the slack, which is 25% of cap;
- writes the tile coverage, heat, and the surface and theta window marks.

Traffic is about 48 B read + 48 B written + 16 B links per record, plus the slack writes. That streams in about 0.35 ns/p, so the sort is about 3× off streaming. It has no bench equivalent because the bench's DEPB never needs cell order.

Today's consumers of cell order:
- the cell-run gathers: transferSeams (transferFace), nbSparseAge, and the surface splat;
- the crowding rank in the move's retirement;
- the seeding's cell counts.

## 3. Priors, for each pass more than 2× off its bound

The bounds are the design's (`particle-brick-slabs-2026-10-19.md` §B), at 0.61M:
- move 0.20 ms;
- deposit 0.45 ms;
- FIN 0.13 ms;
- atlas 0.06–0.13 ms.

Passes the design removes (sort, seams, the snapshot chain, the separate seed and census passes) have a bound of zero, or the streaming floor of whatever must stay.

| Pass (× bound) | Prior behind today's design | Where the prior breaks | Different approach |
|---|---|---|---|
| Move (≈10×) | G2P's per-sample predicates are cheap beside the trace | Measured: they are a third of the move (0.95 ns/p). Each one is a paged gather chain per particle. The whole slab is in one brick, yet every sample resolves the page table again | Brick context: the workgroup stages the slab's brick plus a 1-cell halo once: φ corners for depth, liquid bits for support, cell counts for sparse age, the target heat of its 8 tiles. Predicates become workgroup reads. One workgroup per slab, not per 64-record chunk, so staging is paid once a brick |
| Move samples | Unit-halo texture taps are the cheap path | The fetches run at the texture ceiling (0.42 ns/p for 9). The page chain per sample adds 0.3 | A brick-addressed velocity atlas (§4): the 27-brick origins sit in workgroup memory, as nbMoveBrick already does for the append |
| Deposit (4.2×) | A tile (4³) is the deposit's natural job | Per-tile fixed work and a halo ratio of 4 cost as much as the scatter itself | Brick (8³) windows, as DEPB. Pre-reduce in subgroups before the workgroup atomic: records are cell-sorted, so neighbouring lanes share nodes |
| Seams (∞; 0.49 ms) | 4h owner faces need a separate gather over cell runs | The deposit already holds every sample's momentum in its window | Flush the brick window into the 4h owners' faces in DEPB: a face beside h cells takes the window's sum over its sub-nodes |
| Census (∞; 0.48 ms) | The census prediction is a separate lean pass | It re-reads every record that DEPB's classify just read, tile by tile in the same order | Fold the tile loop into DEPB's classify loop (same tile, same records). Seeds predict where they are seeded |
| Snapshot chain (∞; 1.05 ms) | u* must be sampled per particle, in its own pass, through the general sampler and its 4h cache | The cache fills every stored block (0.42). The record pass streams the whole extent, holes included. The fold (scratchpad `core-snapshot-fold-2026-10-20.patch`) lost only because its grid-side u* write ran through paged and seam taps (0.26–0.30 ms of seam taps alone) | u* as a second atlas channel over particle bricks only, written from the snapshot slots in brick order, at stream rate (1.7M texels × 8 B ≈ 14 MB ≈ 0.04 ms of traffic). G2P reads pic and u* with one extra fetch each. Snapshot, cache and unit taps go |
| Sort (∞; 0.62 ms) | Every consumer needs cell runs | Only the cell-run gathers need them. Counts and ranks can come from append atomics, and the deposit needs only brick buckets | Keep the sort while the surface splat needs it, but make it a pure stream: no window marks or heat (the move can raise them), and no dead-marking (consumers stride counts) |
| Seeding (∞ beside FUSED; 0.52 ms) | A cell below 8 samples needs seeds, and a seed lands wherever its slab has room | It churns 7% of the particles each frame (§5), and half of all seeds spill into the deferred lane | Hysteresis, seed slack in the slab cap, a tile-level reject (§6) |
| FIN (4.7×) | The finish needs its own depth and heat context, so it is a pass of its own | The context it re-stages (125 paged vertex depths, 27 heats, 27 bandPhi a tile) is what the brick context of rank 2 stages anyway | Finish in DEPB's flush from the brick context, for bricks that own their fine tiles. Only tiles shared across bricks keep a FIN |

## 4. The move and the snapshot through hardware-filtered textures

Measured texture rates are 13.5 G samples/s for rgba32f and 28.4 G for rgba16f, each a full trilinear lookup.

**The move already traces through a filtered texture** (particleSampler texture32 is the default). Each sample is three rgba32f lookups, one per MAC axis, and keeps one channel. The page chain finds the halo block. Measured cost: fetches 0.42 ns/p, page chain 0.3 ns/p.

| Variant | Fetches/particle | Fetch time (ns/p) | Addressing | Projected samples (ms at 0.61M) |
|---|---|---|---|---|
| Today (texture32, MAC, page chain) | 9 rgba32f | 0.42 measured | 0.3 measured | 0.44 |
| MAC atlas, brick-addressed (27 origins in workgroup memory) | 9 rgba32f | 0.42 | ≈0.02 | 0.27 |
| MAC atlas, rgba16f for k2/k3, rgba32f for G2P | 3 × 32f + 6 × 16f | ≈0.22 + 0.21 | ≈0.02 | 0.15 |
| Collocated atlas (one rgba fetch per stage, not MAC) | 3 rgba32f | 0.22 | ≈0.02 | 0.15 |
| Collocated atlas, rgba16f | 3 rgba16f | 0.11 | ≈0.02 | 0.08 |

Precision caveats:
- **G2P's FLIP delta.** It subtracts u − u* of similar size, and rgba16f's 11-bit mantissa would cost it about 1e-3 relative. G2P stays rgba32f. k2 and k3 can be rgba16f.
- **The collocated atlas** changes interpolation from MAC to vertex. It is a physics change and goes behind a toggle.

**The snapshot through the same texture.** Today the slot snapshot's general sampler reads the unit texture with 24 loads per sample, plus the 4h cache. Measured sample cost: 0.32–0.43 ns/p of a 0.64–1.27 ns/p pass.
- **Kept as its own pass:** 3 rgba32f lookups is 0.22 ns/p. That saves about 0.1–0.2 ns/p, or 0.06–0.12 ms.
- **The lever is to fold it.** u* goes in the atlas's second channel, and G2P samples it beside pic at the same point. That adds 3 lookups (0.22 ns/p, 0.13 ms) and removes snapshot, cache and unit taps (1.05 ms).
- **Projected: −0.85 ms**, provided the atlas fill stays at stream rate. That is the condition the fold patch failed.

## 5. Why the pour's particle count grows after the stream ends

The stream ends at f126, yet live particles keep rising. In one run they reach 1.0M at f140 and 1.49M at f242.

**Same curve on the 5467d4a8 baseline** (scratchpad tree, read only; `scratchpad/churn/churn.mts`):

| Frame | 5467d4a8 live / fine tiles | HEAD run A | HEAD run B |
|---|---|---|---|
| 126 | 1.044M / 7947 | 0.88M / 7392 | 0.76M / 6793 |
| 140 | 1.139M / 8393 | 0.99M / 7850 | 0.93M / 7800 |
| 200 | 1.447M / 12011 | 1.36M / 9811 | — (fatal at f154, below) |
| 242 | 1.120M / 10056 (peak 1.46M at f210) | 1.16M / 9312 | — |

**Particles per fine tile hold at 110–140 in both trees**, before and after f126. The count follows the band's area. Fine tiles grow by half after the stream stops, as the liquid spreads and splashes in the glass. The baseline grows the same way, so slab storage did not introduce it.

**Particles per occupied cell, by the cell's mean particle depth** (HEAD run B, record census):

| Layer | f126 particles/cell | f140 particles/cell | Cells over 8 at f140 |
|---|---|---|---|
| [−1, 0] | 11.2 | 10.3 | 8573 of 24930 (2830 over 16) |
| [−2, −1] | 9.8 | 9.5 | 17713 of 33755 |
| [−3, −2] | 9.7 | 9.7 | 13026 of 23219 |
| [−4, −3] | 7.5 | 7.5 | 7707 of 23887 |
| all | 9.2 | 8.8 | |

The excess over the nominal 8 is about 9–15%. Seeding tops a cell up to 8, but crowding retires only above rank 16 and only below depth −1. The top layer has no cap at all, which keeps thin sheets.

**Seeds against deletions per frame** (temporary counters; HEAD run A, after f126):

| Frames | Seeds | Retired, depth < −4 | Retired, crowded | Retired, sparse / heat / blocked |
|---|---|---|---|---|
| f130–f150 | 55–81k | 29–49k | 17–25k | 2–17k |
| f160–f240 | 83–104k | 65–85k | 10–13k | 0–6k |

The counters balance the live count to within a few thousand a frame. No seed lands below −4. The census and erase keep up.

**Verdict: no over-seeding and no under-deletion.** The count is the band's area times about 9 per cell. There is churn, though:
- seeding fills cells to 8 down to depth −4;
- the move retires at depth −4.0;
- with no hysteresis between the two, about 7% of the particles are replaced each frame;
- about half of all seeds spill (§7).

**Particle-count levers** (all change physics, so they go behind NB-FLIP method params):
- **Density 8 in every layer**, with a crowding cap near 8 below −1: about −10% particles.
- **bandWidth 4 → 3:** about −25% particles. It is already a method param.
- **Seed hysteresis**, for example seeding to −3 and retiring at −4. It cuts churn and seed spill more than the count.

**Fatal on HEAD (unresolved, not particle code).** Run B stopped at f154 with "Narrow-band FLIP trajectory longer than the lattice", meaning a velocity blowup. Run A of the same commit ran to f242 cleanly. The tree included the peers' uncommitted pressure-arena and surface edits. It has been reported to the coordinator.

## 6. Ranked structural changes

Projections are in ms at 0.61M, using the measured rates above. "KO" marks a figure from a knock-out delta; "read" marks a figure from reading the kernel and a streaming estimate.

| Rank | Change | Retires or shrinks | Projection | Basis |
|---|---|---|---|---|
| 1 | Fold the snapshot into G2P through a u* atlas channel over particle bricks, filled at stream rate | snapshot, cache, unitVelocityTaps (1.05) | −0.85 | KO (sample cost) + read (cache scope) |
| 2 | Brick context in the move: one workgroup per slab, with φ corners, liquid bits, cell counts and tile heat staged once | G2P predicates (0.95 ns/p measured) | −0.45 | KO |
| 3 | Deposit on brick windows with subgroup pre-reduction (DEPB) | per-tile overhead about 0.8 ns/p; scatter at the atomic ceiling 0.95 ns/p | −0.6 | KO (scatter, flush) + bench DEPB 0.72 ns/p |
| 4 | Seams in the DEPB flush | transferSeams gather (0.35–0.69 ns/p) | −0.3 | KO |
| 5 | Census in DEPB's classify loop, with seeds predicting at seeding | transferRefine's second record pass (about 0.3) | −0.3 | read |
| 6 | Brick-addressed velocity atlas for the move, rgba16f for k2/k3 | page chain 0.3 ns/p + half the fetch time | −0.3 | KO |
| 7 | FIN in the DEPB flush for bricks that own their fine tiles | transferProperties | −0.2 | read + KO (−0.4 to −0.5 ns/p with no flags) |
| 8 | Seed slab slack: cap += last frame's seeds per brick | about 50% seed spill: 10–20k records a frame through the deferred lane at about 13 ns | −0.1 to −0.15 | measured spill + deferred rate |
| 9 | Sort as a pure stream (marks and heat move into the move) | particle sort's excess over streaming | −0.2 to −0.3 | read |
| 10 | Tile-level seed reject: skip band tiles that are saturated, deep or air in one test a tile | seedCells | −0.05 to −0.1 | KO (seedCells ≈ 0.2–0.37 ns/p) |

Without physics changes, the sum is about −3.4 ms, taking 7.3 → about 3.9 ms. That is 1.6–2.2× the bench chain, not the 2.4 ms target. Two things are left:
- the append and record traffic of the move, about 1.5 ns/p not yet attributed;
- the passes the design deletes outright (sort, placement), which survive here because the surface splat reads cell runs.

**Physics levers**, behind toggles, applied after the structural changes:

| Lever | Projection |
|---|---|
| Seed hysteresis (churn 7% a frame → about 2%) | −0.2 |
| Density 8 in every layer (−10% particles) | −0.4 |
| bandWidth 3 (−25% particles) | −1.0 |

Together they bring the structural 3.9 ms to about 2.3–2.9 ms.

**Order:**
1. Rank 1 and rank 6 share one atlas. Build that first; it covers both of the move's sample costs.
2. Ranks 2–5 and 7 are one DEPB + brick-context rewrite of the transfer and the move's G2P.
3. Seeds (8 and 10) and the sort (9) follow.

## 7. Spill headroom at peak

Method: per-frame readback of the NB state, with seed counters temporarily added and since reverted.

The two banks spill separately:
- the move's overflow goes to the scratch bank's reserve (state word 16), and placement re-homes it;
- seed spill goes to the current bank's reserve (word 21), which is cleared just before seeding.

Each bank's reserve is capacity / 16.

| | Letters, 240 frames | Pour, 250 frames |
|---|---|---|
| Capacity / reserve per bank / slab limit | 6.36M / 397k / 5.96M | 4.26M / 266k / 3.99M |
| Peak live | 1.72M (f157) | 1.49M (f242) |
| Peak extent vs slab limit | 2.02M, 2.95× | 1.73M, 2.31× |
| Peak current-bank spill (seeds) | 56k (f128), 7.0× | 86.6k (f237), 3.07× |
| Peak scratch spill (move) | 71k (f138), 5.6× | 62k (f238), 4.3× |
| Peak seeds a frame | 105k | 174k |
| Clip, fatal | 0, 0 | 0, 0 |

The seed cell list's cap (3× capacity) is more than 100× its use. **Every headroom is at least 2.3×, so nothing is resized.**

In both scenes the current-bank spill equals the seed spill exactly: about half of all seeds find no slab room. Rank 8 targets this.
