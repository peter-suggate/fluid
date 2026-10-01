# Volume group: conservative V transport and volume bookkeeping (fig-9)

Scope: everything between the phi stage and the geometry rebuild in
`uniform-mixed-frame.ts:491-497`. That is transport copy, conservative
transport, dust cleanup, then the global surface-volume (SV) solve.
Analysis was READ-ONLY: code reading plus the mock-device WGSL capture
(`$SCRATCH/wgsl/fig9/`, `pipelines.json`, `used.json`). No GPU runs.

## 0. Frame-level numbers (fig-9, `runs/base/fig9.json`, frames 5-130, n=126)

| trace phase (label) | what it contains | mean | median | p90 |
|---|---|---|---|---|
| "Geometric volume coupling" (`V.coupling`) | `transport.encodeCopy` + `encodeTransport` + `cleanup.encode` (dust>0 at fig-9) | 2.146 ms | 2.032 | 3.146 |
| "Surface volume constraint + geometry" (`V.gather`) | SV solve (22 dispatches) + phiResolve + `geometry.encode(pressure)` | 1.731 ms | 1.638 | 2.490 |

So this group owns about 3.9 ms of a 22.4 ms GPU frame, roughly 17%. Per-kernel splits are
not in the JSON; the historical kernel numbers below come from memory notes
(copyVolume 35 µs, floor 58 µs, orphan 434 µs).

Lattice facts used below: 128×128×64 = 1,048,576 cells; 16,384 tiles of 4³.
Peak is about 11,000 h tiles (64 owners each) plus about 5,000 4h tiles (1 owner each), so
about 709k owners. Dispatch overhead is about 12 µs per dependent dispatch (memory note,
Dawn/Metal).

### V storage format (answer to "how many arrays of V")
- `fields.volume` is r32float, 1,048,576 texels × 4 B = 4.19 MB. It is the canonical V.
- `fields.volumeScratch` is r32float, 4.19 MB. It is the transport source (copy target)
  and the ping-pong partner for sharpening (`uniform-mixed-frame.ts:293`).
- There is no third V array. The "copy" is `copyVolume` writing volume into volumeScratch
  (`uniform-mixed-transport.wgsl.ts:328`), and transport then gathers from scratch back into volume
  (`transportGroup`, `uniform-mixed-transport.ts` ~l.80-90).
- Donor sums are exact 64-bit fixed point. Fraction bits
  TP_F = 64 − (ceil(log2 1,048,576)+1) = 64 − 21 = 43 (`.wgsl.ts:16-19`).
  The 44 integer+sign bits are split into ceil(44/22) = 2 limbs of 22 bits for the workgroup
  window. Global planes are low u32, high u32 and flags u32, i.e. 12 B/cell in the arena
  donor slice, plus a decoded f32 `sums` plane (4 B/cell, binding 3).
- f32 V is kept everywhere outside the donor sums. That is fine: f16 canonical fields
  were rejected before, and V needs about 1e-6 resolution for dust.

---

## 1. Why module #9 compiles 70 pipelines and dispatches 30

Module #9 is "Uniform mixed conservative transport". The 70 compiled are
35 distinct (entry, list) pipelines × 2 solid twins. `UniformMixedSolid.compile` builds both
the full and the `umSolidsPresent=0` variant of every pipeline up front, and `select()` picks
one by `present`.

The 35 distinct pipelines (`uniform-mixed-transport.ts:18-20, 95-125`):

| group | pipelines | used at fig-9 |
|---|---|---|
| transport live launches: clear:2, build:1, rowsFallback:3, decode:2, rowsDivide:4, decode:6, rowsDivide:5, decode:7, gather:5. h tier (Fine entry, one workgroup per h tile) and 4h tier (generic; coarse rows via `parallelCoarseRows`) | 9 × 2 tiers = 18 | 18 |
| generic tier-1 `rowsFallback:3`, `rowsDivide:4`, `rowsDivide:5` (superseded by `rowsFallbackCoarse`/`rowsDivideCoarse` because the only constructor call, `uniform-mixed-frame.ts:252`, passes `parallelCoarseRows=true`) | 3 | 0 (dead) |
| live set: liveSeed, liveSeedCoarse, 7-step chain (liveGather/liveScatter), liveCompact | 10 | 10 |
| copyVolume (2 tiers via dispatchAllCounted) | 2 | 2 |
| restrictVolume (2 tiers); `encodeRestriction` (`.ts:165`) has no caller | 2 | 0 (dead) |
| **total** | **35** | **30** |

Of the 30, all are solid-free twins (fig-9 has no solids).

The 40 not dispatched are:
- **30 solid twins** of the used set. They are architectural: a live voxel edit can turn solids on
  without a pipeline compile (memory "Mixed functionality gaps": solids always compiled, override
  twins recover cost). They cannot be deleted. They *can* be compiled lazily or in the background
  after first frame, which saves setup time only and has zero frame-time effect.
- **10 dead pipelines**, which can be deleted along with their code:
  - `restrictVolume` ×2 tiers ×2 twins = 4. Delete `encodeRestriction`, `restrictGroup` and
    `restrictVolume` (`.wgsl.ts:331-337`).
  - generic tier-1 rows ×3 ×2 twins = 6. Delete the `parallelCoarseRows=false` branch, the generic
    `normalizeRow` (`.wgsl.ts:143-160`), the `uniform-volume-normalization.wgsl.ts` import
    (its only other user is the dense path), and the list-0 dense branches in
    `.wgsl.ts:304-327`.
  - These are compile-time savings only, about 14% of module #9 pipelines, and the module's
    WGSL shrinks.

Which of the 30 are needed? All 30 carry work at fig-9. The redundancy is in *dispatch count*,
not in what each pipeline does: copy (2) + clear (2) + decode (6) are bookkeeping passes that can be
folded away (E1, E4, E5). With h/4h tier merging (E3), the 18 transport launches can become 9.

---

## 2. Per-kernel walk in frame order

Notation: `T_h` ≈ 11,000, `T_4` ≈ 5,000, `N` ≈ 709k owners. "Counted" means the counted grid-stride
wrapper in `uniform-mixed-topology.wgsl.ts` ~281-321: lane 0 reads the count, then one
workgroupBarrier per job, COUNTED_GRID = 4096.

### 2.1 copyVolume (`uniform-mixed-transport.ts:163`, `.wgsl.ts:328`) — 2 dispatches
- **Does:** volume → volumeScratch, per owner, both tiers (dispatchAllCounted).
- **Bytes:** 709k × (4 B load + 4 B store) = 5.7 MB. About 35 µs historically, mostly two dispatch
  latencies plus one pass over V.
- **Needed?** Not as a separate pass. liveSeed/liveSeedCoarse already visit every owner of every
  active tile and *already read V* (`liveSeed`, `.wgsl.ts:456`, reads `volume`, which is the scratch
  copy). See E1 (fold the copy into the seed) and E2 (reverse the direction and drop the copy).
- **Key invariant for ping-pong:** a tile not in S has V = 0. Every box includes its own
  tile, so S ⊆ R1. Rows outside R1 are therefore exactly zero after transport. A
  ping-pong without the copy only has to make sure that non-R1 owners of the destination are
  zero, or that they are never read.

### 2.2 encodeTransport header clear + live set — 1 clearBuffer + 10 dispatches
`.ts:168-172`, WGSL `.wgsl.ts:401-533`.
- **liveSeed** (`:456`) runs one workgroup per h tile (counted over fineTiles). It uses 6 shared atomics,
  2 barriers, 64 V loads plus departure reads, and emits the per-tile box (10-bit packed offsets,
  bias 512) and S membership.
- **liveSeedCoarse** (`:471`) runs one lane per 4h tile and loops over 64 `tpSourced` calls (the
  source-active check per cell).
- **Chain** S→R1→D2→Q2→D1→Q1→Q0→DONOR is 7 dispatches of `liveGather`/`liveScatter` (`:487`, `:495`).
  - Each is one lane per tile over **all 16,384 tiles**, with atomic loads over up to 27 box tiles.
  - Bytes per step: 16,384 × (4 B flag + ≤27×4 B neighbour flags + box word), about 1.8 MB worst
    case, L2 resident.
  - The cost is the 7 dependent dispatches (≈ 7 × 12 = 84 µs floor), not the bytes.
- **liveCompact** (`:520`) has 14 counters (7 lists × 2 tiers). The live buffer is
  (20 + 17 × 16,384) × 4 B = 1.114 MB.
- **Access pattern:** the chain is inherently sequential (each list is a dilation of the
  previous one). Collapsing it would need a closed form for the box union, which is possible
  only if the boxes were bounded by a fixed radius. They are not: departure boxes are
  data-dependent, so I leave the chain alone.
- **Suspicion "liveSeed full-tile scan":** liveSeed scans all 64 cells of each h tile, but it
  must, because V and the departure come per cell. liveSeedCoarse's 64 `tpSourced` calls per 4h tile
  are runtime-gated by source uniforms. With no active source the loop is still entered.
  Hoisting `if (!sourcesActive) {return false}` out of the loop is trivial (E10b).

### 2.3 Transport proper — 9 launches × 2 tiers = 18 dispatches
`liveLaunches` (`.ts:18`): clear:2, build:1, rowsFallback:3, decode:2, rowsDivide:4,
decode:6, rowsDivide:5, decode:7, gather:5.

| launch | h tier | 4h tier | what |
|---|---|---|---|
| clear:2 | per owner | per owner | zero the low/high/flags donor planes (12 B/cell) |
| build:1 | `buildFine`, one wg/h tile | `buildAt` (`.wgsl.ts:106-131`) | trace the departure, write 9 raw edge weights per receiver row (arena edges 40 B/cell), tpAdd into donor sums; sampling atomicOr |
| rowsFallback:3 | `rowsFallbackFineAt` | `rowsFallbackCoarse`, one wg per row (`:169-199`) | round 1: rows with no positive weight fall back; normalize |
| decode:2/6/7 | per owner | per owner | `tpTake` (2 atomicExchange) → f32 `sums` plane, and clears for the next round |
| rowsDivide:4/5 | `rowsDivideFine` (`:281`) | `rowsDivideCoarse` | Sinkhorn round: divide edges by the donor sum, re-accumulate |
| gather:5 | `gatherAt` (`:282-301`) | same | V_new(receiver) = Σ edges × V(donor)/sum(donor) |

**h-tier divide (the brief's suspicion list):** `tpFineRows` uses one workgroup per h tile and a
tpWindow of 512 × TP_LIMBS(2) shared atomics (22-bit limbs), with 3 barriers per job. Per job:
1. zero the window (barrier);
2. each of 64 lanes reads its 9 edges from the arena (`rowAt` `:74`: address = t·640 + lane·10 words,
   stride 64 words for unit rows), recomputes weights, and does atomicAdd into the window for
   in-box donors (barrier);
3. `tpCommit` (`:234`) flushes the window to global over ≤8 box tiles with tpAdd (low atomicAdd with
   return, then a conditional high atomicAdd on carry) (barrier).

Costs: edge stores 64 × 9 × 4 B = 2.3 KB/tile/round, so 11,000 × 2.3 KB = 25 MB written per round, and the
same amount is read in the next round. Memory already established that
"h divide cost is edge stores + barriers + window round trip, not atomics", and
that workgroup-memory accumulation was null (so the window itself is not the lever).
The fixed-point 64-bit carry is exact and cheap. I found no atomic waste to remove: one
atomic per (row, donor) edge is the floor for a scatter formulation.

**decode:** 3 × 2 tiers = 6 dispatches whose only job is turning the 64-bit planes into f32 and
clearing them for the next round. Bytes: 709k × (12 B read+clear + 4 B store) = 11 MB each, about
34 MB total, plus about 72 µs of dispatch floor. See E4.

**4h coarse rows:** one 64-lane workgroup per 9-donor row, with 126-slot shared arrays, 3 barriers
and a lane-0 serial sum (`:169-199`). That gives 5,000 workgroups × 3 barriers to process about 9 values each,
with occupancy wasted at about 9/64 lanes. Grain-4 rows (the common case) could go one row per
lane (E10c). This is a one-workgroup-per-row serial reduction, which is the "reductions on one
workgroup" smell at small scale.

**gather:** per donor edge it reads 9 edges + topology + sums + one V texel. For 4h with active sources
it loops over 64 cells. Bytes ≈ 709k × (36 + 4 + 4 + 4 + 4) ≈ 37 MB, a gather with no atomics.

**departure map:** the departures come from `velocityD` (rgba32float, 16.8 MB). `.w` is always 0
(written by `traceCells` at owner origins), so the .w channel (4.2 MB) is dead. The texture is
shared as the forces "advected" input, so it can't be narrowed in isolation. RK2 cell-centre
departures are traced once in phi's `traceCells`, *not* re-traced in transport. That is good, and
there is no duplicate trace to remove here.

**Ping-pong vs copy:** `encodeCopy` is needed only because transport reads `volumeScratch` and
writes `volume` in place, and the next frame's code reads `volume`. See E1/E2.

### 2.4 Dust cleanup (`uniform-mixed-cleanup.ts`, module #11) — 1 + 2 + 1 dispatches (counted)
Runs because `p.dust>0` at fig-9 (`uniform-mixed-frame.ts:494`).
- **floor pass** (`umOrphan=false`, groups[0] volume→scratch, dispatchAllCounted, 2 tiers).
  It is pointwise: load V; if |V| < floor, sample the centre vertex (`umSampleVertex`) and discard into
  `umAccountDust` (2-4 atomics per discard). Bytes 709k × 8 B = 5.7 MB, about 58 µs historically.
- **orphan pass**: `summarize` over fused seam tiles (8 B/tile summary = 131 KB), then `clean`
  groups[1] scratch→volume. It does real work only for 0 < V < 0.01:
  - h candidate: 64 footprint `umVertexValue` loads + 27 `umOwnerAt` + V loads;
  - 4h candidate: 27 tiles of summary or vertices.
  This was about 434 µs historically, the single largest kernel in the group. The candidate count is
  not in fig9.json (no `uniformVolumeOrphanDustCells`), so the split is unknown.
- `umClean` is at lines 66-121. The reductions buffer is 48 B. `uniform-mixed-detached-mass.wgsl.ts` is
  **not** used by cleanup; it is used by pressure-velocity and authority.
- Pipelines: 6 compiled (3 entries × twins), 3 used.
- Note: the floor pass is volume→scratch and the orphan pass is scratch→volume, which is a second
  full V copy per frame folded inside cleanup.

### 2.5 Surface-volume solve (`uniform-mixed-surface-volume.ts`, modules #21/#22/#23) — 22 dispatches
Sequence (`:460`): begin (1 wg), clearBand, seed, 4×(grow, dilate), grow, measureGrow, metric,
resolveScale, 2×(measure, reduce, solve (1 wg)), apply. That is 22 dispatches, about 264 µs of dispatch
floor alone.
- **Arena:** scratchBytes = 4·(2N + V + 4(groups+chunks) + 8 + 3T) ≈ 13.2 MB. It holds the band in two
  f32 parities per owner index (2 × 4.19 MB), scale as f32 per vertex (4.3 MB), and the per-tile flags as f32.
- **seed** writes the band into both parities per owner. **grow** does 27 tile-flag loads per resident
  tile. **dilate** gates on the visit flag. That is 5 grow + 4 dilate = 9 dependent dispatches to build a
  band of fixed width.
- **measure:** 8 raw + 8 scale vertex loads per owner, and `fill()` for cut owners.
- **solve:** one workgroup, a secant/Newton step reading about 176 chunk sums. It is inherently
  serial but tiny. `reduce` is one dispatch before each solve.
- **fill/tetra** (`uniform-surface-volume.wgsl.ts:2-26`) dynamically index the private arrays `n`/`o`,
  which risks a stack spill on Metal (local memory). See E9.
- **Dead:** module #23 `applyCopy` is never dispatched. `createUniformSurfaceVolumeWGSL`
  (`uniform-surface-volume.wgsl.ts:27-288`) is unused. Module #21 compiles 24 pipelines and uses 12
  (twins again).

### 2.6 Files that the mixed path barely touches
- `uniform-volume.wgsl.ts` (1753 lines): mixed uses only `uniformVolumeTargetWGSL` (`:73-98`, through
  surface-geometry and remap) and the volume-correction import (pressure authority).
  `uniformVolumeWGSL` builds module #1 (254 KB WGSL), of which only `uvPublish` and
  `buildDenseExtrapolationAuthority` are dispatched, at t=0 (`webgpu-uniform-reference.ts:1556-1580`).
  Everything else in it is dense-path only. That is another group's module; flagged for deletion
  review.
- `uniform-volume-donor-sum.wgsl.ts`: `uniformDonorSliceWords` = limbCells × 7 ("donorfuse"),
  which sizes the arena slice at 28 B/cell. Mixed uses 16 B/cell (low, high, flags + f32 sums). The
  extra 12 B/cell is unused headroom, but see the storage note.
- `uniform-volume-stencil.ts`: EDGE_BYTES = 4·(8+2) = 40 B/cell (9 weights + spare).
- `uniform-volume-normalization.wgsl.ts`: used only by the dead generic `normalizeRow` and the dense path.
- `uniform-mixed-source.wgsl.ts` / `uniform-source.wgsl.ts`: drop and inflow sources,
  runtime-gated by uniforms. Cheap when inactive, apart from the 4h loop (E10b).
- `uniform-volume-pages.wgsl.ts`, `uniform-volume-correction.wgsl.ts`, `uniform-volume-pipeline.ts`:
  no per-frame mixed dispatch in this group's window (correction is reached via pressure authority).

---

## 3. Proposals, with arithmetic

Dispatch saving is taken as 12 µs per removed dependent dispatch, plus the bytes no longer moved
(at an effective about 200 GB/s L2/DRAM mix on an M-series chip, 1 MB ≈ 5 µs).

**E1. Fold copyVolume into liveSeed/liveSeedCoarse.** Low effort.
- Bind the restrictGroup-style group (input `volume`, output `volumeScratch`) to the seed. The seed
  already reads every V of every live tile, so it also stores it into scratch.
- Non-live tiles have V = 0 and are never read by gather (only R1 donors are read), so they don't need
  copying.
- Saves 2 dispatches (24 µs) + 709k loads (2.8 MB ≈ 14 µs), which is about 35-40 µs, matching the
  historical copyVolume 35 µs. It is layout-general because the seed is counted over both tiers.

**E2. Reverse the transport direction and fuse the dust floor into gather.** Medium effort.
- Transport reads `volume` and writes `volumeScratch`, with no copy at all.
- Gather applies the pointwise floor (needs phi, tuning and reductions bindings, and the umAccountDust
  atomics) as it writes.
- The orphan pass then reads scratch, masked by live R1|S bits (non-R1 owners of scratch are
  stale, so they must be treated as 0), and writes `volume`.
- When dust == 0, a masked copy-back is still needed (1 pass). That makes it a net win only when dust>0,
  which is the fig-9 default.
- Saves the copy (35 µs) + the floor pass (58 µs, 2 dispatches) ≈ 0.1 ms, and up to about 0.15 ms
  counting the bytes.
- Risk: the staleness of non-R1 scratch must be enforced everywhere scratch is read (sharpen also
  ping-pongs on it, but only after the orphan pass has written `volume`).

**E3. Merge h and 4h tiers into single transport launches.** Medium effort.
- Use the job space [h jobs | 4h jobs], as `umAllOwner` already does.
- 18 → 9 dispatches saves 9 × 12 = 108 µs, plus the 4h workgroups fill the h tail. Estimate 0.1-0.15 ms.
- Cost: registers = max of both paths. The h fine path is the heavy one, and 4h rows are small, so it is
  likely fine.
- Applies equally to the live set (liveSeed + liveSeedCoarse → 1: −12 µs).

**E4. Remove the 3 decode passes.** Medium-high effort.
- Keep per-round 64-bit plane sets A/B/C (interleaved low/high).
- Round k+1 reads set k through a *read-only* binding on a non-overlapping range of the same arena
  buffer (WebGPU allows this) and converts to f32 inline (2 loads + 1 multiply-add instead of 1
  load).
- `clear` zeroes 7 words/cell once (A, B, C low/high + flags). This fits exactly in the existing
  28 B/cell donor slice (`uniformDonorSliceWords = limbCells×7`), so it needs no new storage.
- The f32 `sums` plane (4.19 MB) and lists 6/7 go away.
- Saves 6 dispatches (72 µs) + about 3 × 709k × 16 B ≈ 34 MB of decode traffic (≈ 0.1-0.17 ms), so
  ≈ 0.1-0.2 ms. The clear grows from 3 to 7 words/cell (+11 MB, about 55 µs), so the net is
  **≈ 0.05-0.15 ms**. Only the read side of the rows changes; tpAdd is untouched.

**E5. Fold `clear` into liveSeed.** Low effort.
- The seed already visits every owner. Zero the donor planes and reset the per-tile `sampling` word
  there.
- Saves 2 dispatches ≈ 25 µs. It composes with E4 (the seed clears 7 words).

**E6. SV dispatch trims.** Low effort.
- Fuse `begin` (1 wg) into `clearBand`: −1.
- Fold each `grow` into the following `dilate`: dilate computes visit inline from 27 band flags.
  Superset visiting is harmless because dilate is idempotent on already-banded owners, and the last grow
  stays. That is −4.
- Optionally drop `reduce`: `solve` reads the ~11k partials directly in its one workgroup
  (11k / 64 lanes = 172 loads/lane, about 10 µs) instead of 176 chunk sums. That is −2 dispatches, but solve
  gets slower, so it is roughly break-even. Measure it.
- Net: −5 to −7 dispatches ≈ 60-85 µs.

**E7. Orphan pass (≈434 µs historical, the biggest single kernel in the group).** Medium effort.
- First do a CPU-free counting probe: add the orphan candidate count to the existing stats readback
  (non-blocking), and split its per-kernel time.
- Idea: a per-h-tile min |phi| (or "any vertex ≤ 0") summary for *all* h tiles, not only seam tiles.
  A far-air candidate then reads 8 summary words instead of 64 footprint vertex loads plus 27 owner
  lookups.
- If the candidate count is in the thousands, the loads are not the cost; dispatch and occupancy are,
  and the fix is a compacted candidate list (fixed grid-stride over the GPU list, not indirect).
- Also, if cleanup moves to residentAll job order, the SV `seed` can be fused into the orphan pass
  (both visit all owners near phi≈0).
- Estimate 0.1-0.3 ms if the far-air hypothesis holds. This is unverified.

**E8. SV phi-based band taper.** Medium effort, *semantic change, needs Peter's approval*.
- Use b = clamp(5 − |phi|/w, 0, 1)-style weights computed in `measure` directly from phi.
- This replaces seed + 5 grow + 4 dilate + clearBand (about 11 dispatches and the 8.4 MB band
  parities).
- Saves about 11 × 12 = 132 µs + band traffic ≈ 0.15-0.2 ms. The band would become a geometric distance band
  rather than a topological one, which changes which vertices the correction moves.

**E9. Branchless `fill`/`tetra`.** Low-medium effort.
- Rewrite `uniform-surface-volume.wgsl.ts:2-26` as a sorting network over 4 scalars with
  `select()`, so there is no dynamic indexing into private arrays and no stack spill.
- `measure` calls it for every cut owner each round (×2 rounds + measureGrow).
- Estimate 20-50 µs. Verify first with the Metal compiler output on the capture tool's WGSL.

**E10. Small ones.** Low effort.
- (a) `liveCompact` stores the topology word (owner base) next to the tile index in the compacted list,
  saving one dependent load level (count → tile → topology) in every transport lane of 9 launches.
  About 10-30 µs.
- (b) Hoist the source-active test out of `liveSeedCoarse`'s 64-call loop and gather's 4h source loop.
  About 5-15 µs.
- (c) Coarse rows: pack grain-4 rows one row per lane instead of one 64-lane workgroup per 9-donor
  row with 3 barriers and a lane-0 serial sum. That is 5,000 workgroups → about 80. About 20-50 µs.

**Long shot.** A donor-centric gather formulation where each donor computes its own outgoing row
sum, so there are no atomics and no window. It is a big rewrite. The divide's cost is edge stores
and barriers (memory note), so the gain is speculative. Not ranked.

**Deletions (compile-time only, no frame effect):**
- `restrictVolume`, `encodeRestriction`, `restrictGroup`
- generic tier-1 rows, `normalizeRow`, the normalization import
- list-0 dense branches
- the `parallelCoarseRows` flag
- module #23 `applyCopy`
- SV non-resolved branches and `umMeasureCorners`
- `createUniformSurfaceVolumeWGSL`
- and, for whichever group owns module #1, the bulk of `uniform-volume.wgsl.ts` that only feeds the
  t=0 dense module.

---

## 4. RANKED experiment list

| # | experiment | est. gain (fig-9 GPU) | effort | notes |
|---|---|---|---|---|
| 1 | **E7** orphan-pass split + per-h-tile phi summary or compacted candidate list | 0.1-0.3 ms | M | biggest single kernel (≈434 µs historically); measure the candidate count first |
| 2 | **E3** merge the h/4h tier launches (transport 18→9, seed 2→1) | 0.1-0.15 ms | M | job space [h | 4h] like umAllOwner |
| 3 | **E2** reverse the transport direction, no copy, fuse the floor into gather | 0.1-0.15 ms | M | needs R1-mask discipline on stale scratch; masked copy-back when dust=0 |
| 4 | **E8** SV phi-based band taper (−~11 dispatches, −8.4 MB band) | 0.15-0.2 ms | M | semantic: Peter must approve |
| 5 | **E4** per-round 64-bit plane sets, no decode passes | 0.05-0.15 ms net | M-H | fits in the existing 28 B/cell slice; drops the 4.19 MB sums plane |
| 6 | **E6** SV trims (begin→clearBand, grow→dilate, optional reduce) | 60-85 µs | L | pure dispatch removal |
| 7 | **E1** fold copyVolume into liveSeed (if E2 is not taken) | 35-40 µs | L | subsumed by E2 |
| 8 | **E10c** coarse rows one-per-lane for grain-4 | 20-50 µs | L-M | 5,000 wgs × 3 barriers → ~80 wgs |
| 9 | **E9** branchless fill/tetra (no private-array spill) | 20-50 µs | L-M | check the Metal compile first |
| 10 | **E5** fold clear into liveSeed | ~25 µs | L | composes with E4 |
| 11 | **E10a/b** topology word in the live list; hoist source tests | 15-45 µs | L | |

Combined realistic (2, 3, 5, 6, 10, 1-partial): ≈ 0.4-0.7 ms of the 3.9 ms this group owns.
Dispatch arithmetic for module #9:
- today: copy 2 + live set 10 + transport 18 = 30;
- after E1+E5+E4+E3: copy 0 + live set 9 (seeds merged, clear folded in) + transport 5 (build,
  fallback, divide, divide, gather; tiers merged; no clear, no decode) = 14;
- −16 dispatches ≈ 0.19 ms floor alone.

SV: 22 → 15-17 with E6, or about 11 with E8.

## 5. Storage that can be deleted (or shrunk)

| item | bytes at fig-9 | frame effect | condition |
|---|---|---|---|
| f32 `sums` plane (binding 3) | 1,048,576 × 4 = 4.19 MB | arena | with E4. Note: the arena is extension-bound (max(71.4 MB extension FIM, ~65 MB pressure, 70.3 MB edges+donor)), so this does NOT shrink the allocation unless extension shrinks too |
| unused donor slice headroom (28 → 16 B/cell, if E4 is not taken) | 12 × 1,048,576 = 12.6 MB | arena | same caveat (extension-bound) |
| SV band parities f32 → u8 | 2 × 4.19 MB → 2 × 1.05 MB, −6.3 MB | arena | or −8.4 MB with E8 |
| SV per-tile flags stored as f32 | 3 × 16,384 × 4 = 196 KB → 49 KB as u8 | arena | trivial |
| departure `.w` channel | 1,048,576 × 4 = 4.19 MB | texture | only if velocityD stops being shared with forces "advected" (otherwise keep) |
| `sampling` buffer: 32 bits/tile, 2 used | 16,384 × 4 = 64 KB → 4 KB | buffer | negligible |
| live lists sized to all tiles | 14 × 16,384 × 4 = 917 KB (of 1.114 MB) | buffer | could be sized to the live bound; negligible |
| dead pipelines: module #9 restrictVolume + generic rows (10), module #23 applyCopy | compile-time | setup | delete now |
| `createUniformSurfaceVolumeWGSL` (`uniform-surface-volume.wgsl.ts:27-288`) | source only | none | delete now |

Bottom line on storage: nothing in this group shrinks the device allocation by itself, because
the shared scratch arena (≈ 71.4 MB) is sized by the extension's 4 FIM ranges. The arena only drops
if the extension group shrinks its ranges below the transport slice (edges 40 B/cell = 41.9 MB +
donor 28 B/cell = 29.4 MB = 71.3 MB). After that, E4 + the 28→16 B/cell trim would take the
transport slice to 41.9 + 16.8 = 58.7 MB.
