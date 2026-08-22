# Sparse CM12 compiled topology — one structure for every stage

Status: Phase 1 implemented with deviations; reviewed and re-gated 2026-08-22.
Face preparation and sharpening followers Dawn-gated on the B8/P8 target 2026-08-22.
Reference scene `ocean-seiche`
(320×96×80 fine, 500 resident B16 bricks, 537,220 accepted cells, 1,574,783 rows,
273,288 pressure cells). Receipt: HEAD stage-cost probe, 24 warm frames, all
quiescent (0 committed bricks): advance 80.5 ms = pressure 26.4 + non-pressure 53.5.
Lane is bimodal under contention — read every number below as a proportion.

## Implementation review and revised rollout

Phase 1 delivered the important execution substrate: LOD1 ownership, the effective
transport-velocity plane, leaf/rung-major TEI2 packets, packet authority, immutable
IBO1 boundary operators, and a single authorized topology selector flip. Conservative
transport consumes staged packets and no longer reconstructs ownership in each hot
lane. The topology transaction also compiles and validates its candidate IBO/TEI
images before publishing stable fields.

It did not deliver the whole Level 0/3 cutover described below:

- face preparation, sharpening, D4, tracers, and presentation have not all converged
  on the same compiled sampler;
- TPM1 records exact producer ballots, but its production row/VEX compilers still
  walk incidence, row acceptance, and row terms;
- ITR1, VXI1, and DCA1 exist as independently tested exact transforms, but the current
  resident IBO image contains only IRL1 and geometry supplements, so those transforms
  are not production authority yet;
- rigid scenes are explicitly rejected by the Phase-1 AEI construction; the required
  per-frame open-fraction/transport-active plane has not been implemented;
- the symmetric-expansion raw-bit regression and paired ocean-seiche/mini64 behavior
  and timing acceptance runs have not been captured.

The review found one concrete authority defect: construction initialized the leaf
manifest from active leaves but initialized cell and row lists from every resident
template, including dormant receivers. That made the ocean receipt contain 48,640
unexpected cells (95 dormant B8 leaves × 512) and 144,704 unexpected rows. Construction
now derives all three views from the same active leaf/rung set. The stage-cost probe
hard-fails unless cell partition, row requirements, leaf manifest, and topology delta
all agree. A construction-only solver factory now exposes raw Phase-1 transport
receipts without adding a production runtime selector.

The remaining rollout is therefore gated as follows:

1. Close Phase 1 with symmetric expansion as the B8/B16 raw-bit regression oracle.
   Use ocean-seiche and mini64 for paired behavior receipts and at least three
   interleaved timing captures per scene. Ocean measures the stencil-bound large-scene
   path; mini64 measures launch cost and useful dry/bulk skipping. Long-dam is not an
   acceptance lane for this migration. Do not claim a timing gate from an unpaired
   dirty-tree comparison.
2. Integrate ITR1/VXI1/DCA1 into the resident IBO image and replace TPM1's incidence
   walks. Gate row masks, VEX roots, both gamma banks, and rerung publication before
   converting any further stage.
3. Apply the same compiled acceptance/edge authority to face preparation and
   sharpening first, then projection/collocation and activity measurement. Each
   consumer conversion gets its own raw-bit digest gate and retains template order.
4. Convert D4, tracers, and presentation to the shared sampler once the major consumers
   prove the ABI. These are lower-risk followers, not evidence that the edge authority
   is complete.
5. Add the rigid per-frame predicate plane before removing the rigid-scene rejection.
   Pressure-topology folding remains a separate final phase after compiled edges are
   authoritative.

### First follower: face preparation

The first attempt removed the dense face-support cache and resolved TEI ownership at
every RK2 corner. It was raw-bit exact but failed the required wall-clock gate badly:
on two interleaved ocean B8/P8 runs, committed `HEAD` measured 1.64–3.54 ms for dense
publish plus face tracing, while the on-demand TEI path measured 8.59–8.91 ms. Saving
one 32 MB write did not repay repeated ownership resolution. That attempt was rejected,
not propagated.

The accepted design retains dense support as a read-optimized derived cache. Its
publisher now consumes the accepted TEI leaf and persistent effective-velocity plane;
RK2 keeps its contiguous four-float reads. Dirty-brick scheduling, template row order,
interpolation order, scalar span rule, extended flag, and liquid test remain unchanged.
Two B8/P8 runs of the accepted design measured 3.28–3.67 ms for the face stage and
47.9–48.3 ms total non-pressure, inside `HEAD`'s bimodal 1.64–3.54 ms and
47.0–50.2 ms lanes while also carrying exact accepted worklists. This costs the dense
cache's 32 MB; the wall-clock evidence says that cache is currently economically
necessary.

The A/B also found that the original IBO memory guard was a B16-only census masquerading
as a production limit. Ocean B8/P8 needs 935,936 immutable bytes, 573,696 bytes per
slot, and 20,544 bytes of ISA authority. The B8 shipping plan is now explicitly capped
at 1 MiB immutable, 640 KiB per slot, and 32 KiB ISA; B16 retains its former
512/256 KiB and 16,864-byte caps.

### Second follower: surface sharpening

Sharpening does not trace the transport velocity as the earlier plan claimed. Its hot
path samples the conditioned-density gradient, validates each forward step, and builds
the final density scatter stencil. Those three point-to-cell consumers now use the
workgroup-staged accepted TEI. SCA1 enumeration, incidence-based sharpening statistics,
fixed-point scatter order, receiver closure, and finalization are unchanged.

Dawn gates for the two accepted increments:

- B4/B8/B16: 798 resident entry points and four stage-lens programs compiled.
- B16 symmetric expansion, one step: density, velocity, pressure, divergence,
  accepted resolutions, and combined SHA-256 remain bit-for-bit identical to the
  pre-change capture (`b056d69d…212e`).
- B8/P8 symmetric expansion, one step against the detached committed `HEAD`
  worktree: all four field hashes and combined SHA-256 are identical
  (`c2ca2b39…93644`). The raw-bit tool now accepts explicit brick/page ladders so
  B8 is a named gate rather than inferred from a B16 receipt.
- Ocean B8/P8, two 8-warmup/24-sample runs: sharpening fell from a stable 5.964 ms
  baseline to 5.440 and 5.374 ms (8.8–9.9%); exact cell/row representation, frame,
  scalar and pressure authorities passed with no Dawn validation errors.

Taken further (2026-08-23): the stage is a scalar-finalization transaction, and three
quarters of it is dirty-set bookkeeping, not sharpening — sub-scoped on ocean B8/P8:
finalize 1.90 ms (1.77 of it is VEX-root + SIR1 CAS fan-out, arithmetic 0.13), temporal
frontier 1.18, SCA1 prepare+scatter 1.11, SRR1 compare 1.05, publish 0.20. SRR1 appends
17,892 tile events/frame against 14,240 resident tiles, so it certifies nothing there. The
plan to collapse it to one packet-major finalize ballot (CHANGED/NONEXACT/BULK masks) with
mask consumers is `docs/sparse-cm12-scalar-finalization-handoff.md`.

The next high-value follower is not another arbitrary-point velocity sampler. It is
the compiled edge/row authority needed by projection/collocation and activity census;
promoting those consumers before ITR1/VXI1/DCA1 are resident would merely move their
incidence walks rather than remove them.

Reviewed artifact: "CM12 Frame Anatomy" (18 stages, 32³ mini dam, 16.71 ms). It
describes HEAD faithfully (face-preparation's dense velocity support, FPA1/PCM/PTR1
authorities, SRR1/SCA1 tiles, D4 fold). Its *proportions* mislead for a large scene:

| | mini dam (artifact) | ocean HEAD |
|---|---|---|
| pressure solve | 45 % | 33 % |
| transport | 4 % | 19 % |
| VEX + face-prep + sharpening + projection | 21 % | 41 % |
| pressure-topology | 12 % | 5 % |
| presentation + activity | 9 % | 8 % |

On mini the frame is launch-structure bound. On ocean it is **stencil-resolution
bound**: the frame spends most of its time re-deriving, per lane and per pass, who the
neighbours are, whether they are accepted, and which cell a point lies in. That is the
work that can be compiled.

## 1. What every stage pays for today

Four resolutions recur in almost every kernel. Counts are from the WGSL at HEAD
(`webgpu-sparse-cm12-resident.wgsl.ts`, `sparse-cm12-velocity-extension.wgsl.ts`),
counted as loads and as dependent levels (a level = a load whose address needs a
previous load). Every topology read is `ta()` = `atomicLoad` on the arena binding.

| resolution | today | loads | levels | who calls it (ocean) |
|---|---|---|---|---|
| **point → cell** `ownerCellAt` | LOD1 directory → brickSpan → acceptedBrickResolution (activity atomic) → templateBrickCellRange ×2 → brickActive → cellResolution (cellBase, metadata) → cellOpenVolume | 10–12 | 4 | transport trace ×9/sample ×≤7 samples; `transportStencil` ×9; `sirMassTileCell` once per lane **per pass** (×4 passes); sharpening trace ×11/substep; tracers; D4 ×8/cell; presentation coarse pages; retirement ×26/brick; planning ×24/brick |
| **row → accepted?** `rowAccepted` | requirement count → per requirement metadata + brickActive + acceptedBrickResolution | 4 (intra) – 10 (seam) | 3 | face-prep per row; body-forces per row; VEX sweep per edge ×8 sweeps; collocate per incidence; diagnostics per incidence; sharpeningStats per incidence; activity per incidence; record-incidence per root; submerged test per term; MarkCellClosure; tra1Mark; interface seed |
| **cell → stencil** incidence CSR | incidence begin/end → row id → rowAccepted → rowWord terms begin/end → term cell, coefficient | 10–16 per row, 4+ | 4–5 | preparePressure (RHS), collocate, diagnostics seam test, sharpeningStats (two passes), activity census, record-incidence, pressureCellSubmerged (called in prepare **and** collocate), MarkCellClosure, tra1, gamma (pair CSR, 3-word stride + generation stamp), VEX (pressure edge CSR + rowAccepted + cellActive per edge) |
| **worklist → item** | arrival-ordered list load; SRR1 tiles re-derive the cell by `sirMassTileOrigin` + `ownerCellAt` + `scaCellBounds` (8 float `taf` loads) in *each* of 3–4 passes; FPA rows via a 6-level 32-ary rank-tree descent | 2–20 | 1–6 | every dispatchAccepted kernel; transport; sharpening; projection |

And one bookkeeping pattern: **root / dirty recording per cell with CAS**. VEX roots
(collocate), ACT1 marks, SCA invalidation (≥27 tile appends per finalized cell),
temporal scalar lists (classify + 2 dilations + compactions, 7 dispatches),
FPA row marks, TRA1 leaf marks, PCM membership flips — each is a per-cell incidence
walk ending in a CAS per neighbour. On ocean the receipt says 491k of 537k cells are
in the VEX blast and 392k are "temporal scalar" cells: the scene is topologically
static but **not bit-static** — a rolling seiche plus solver churn flips scalar and
velocity bits in ~73 % of accepted cells every frame. Dirty tracking therefore saves
almost no physics; it only adds ~3 M CAS spins of bookkeeping per frame. The win has
to come from making the dense per-cell/per-row work cheap, not from skipping it.

A trilinear velocity sample today is ≈ 8 × (10–12 owner loads + density + stamp +
3–4 velocity + an `atomicExchange` reuse stamp on air corners) ≈ **130–150 loads,
~6 dependent levels**. Ocean transport issues ~390k cells × (≤7 samples + stencil +
clearance gather) of these.

## 2. The unified structure: a tile-major lattice with compiled seams

One structure, rebuilt per topology generation (on ocean: once), read by every stage.
The universal address is **(resident brick slot, local index at the brick's accepted
rung)**; everything a stage needs about a neighbourhood is either arithmetic on that
address or one plain load from a compiled table. No atomics on the read side.

### Level 0 — Logical Brick Descriptor table (LBD)
Dense over the logical B16 lattice (ocean 20×6×5 = 600 entries), 3 words each:
`slot | active | rungLog2`, `cellBase`, `rowBase` (+ valid extent for clipped bricks,
+ origin for macro bricks — exactly what LOD1 already encodes). It is LOD1 widened
with the four things every owner lookup chases after the directory.
- point → cell: `key = q >> 4` → 1 load → `local = (q & 15) >> (4 − rungLog2)` →
  `cell = cellBase + local.x + r·(local.y + r·local.z)`. **1 load, 1 level** (vs 10–12 / 4).
- Each workgroup stages its brick's 27-neighbourhood (27 × 3 words) in shared memory
  once; every sample within ±1 brick is then a shared-memory read. Samples further
  than 16 cells away fall back to the global table (never happens under CFL: ocean
  |v|·dt ≲ 3 cells).
- `cellActive`, `brickActive`, `acceptedBrickResolution`, `brickSpan`,
  `templateBrickCellRange`, `templateRowOwnerRange` all collapse into it.
- Cell geometry (centre, widths, volume, `scaCellBounds`) is derivable from
  (origin, rung, local) with exact dyadic arithmetic — the B5 proof in the packing audit
  already established the 8-word cell record is derivable from word 7 + brick origin/span.
  No `cellBase` record loads.

### Level 1 — Static accepted tile lists and ballot dirty masks
At commit, compact two lists: **cell tiles** (brick, tile of 64 cells at the accepted
rung; ocean ≈ 8.4k tiles) and **row tiles** (brick, 64 owned rows; ≈ 24.6k tiles).
Every dense stage dispatches *one workgroup per tile*, lane = item,
`id = base + 64·tile + lane`: no worklist load, no per-lane re-derivation, unit-stride
for every SoA field, and one brick per workgroup so the LBD neighbourhood preload
amortises. Per-stage selection is a **64-bit tile mask per brick per cause**
(VEX-dirty, scalar-dirty, pressure-dirty, presentation-dirty — FPL1 already defines
this ABI): a kernel that finds an item dirty sets a lane bit, the workgroup ballots, and
one lane does one `atomicOr`. Closure (1-ring dilation) becomes a brick-local mask
operation over the 27 neighbours' masks, not a per-cell incidence walk with CAS.
Over-approximating a root set to its tile is value-safe everywhere it is used: VEX
recomputes to the same fixpoint the cache already holds; SRR1/SCA1/ACT1 "dirty" only
means "do the work", never "change the answer".

### Level 2 — Implicit interior stencil, compiled seam stencil
Per cell tile, one compiled bit: **uniform** (its brick and the six face-neighbour
bricks share the rung, no wall, no clip). Ocean is ≥ 90 % uniform tiles.
- Uniform tile: the six neighbours are `cell ± 1, ± r, ± r²` inside the brick, or the
  LBD neighbour's cell by arithmetic across the brick face. Coefficients are dyadic
  functions of the rung — bit-identical to the template's floats. Row ids need a
  predictable owner order within (brick, rung, axis); if construction does not already
  guarantee it, either sort at construction or store 3 per-(brick,rung) axis offsets
  in the LBD. **0 topology loads per neighbour.**
- Non-uniform tile: a **compiled accepted-edge CSR** — the pressure template's
  directed-edge list filtered at commit by the realized acceptance, preserving
  template order: `(neighbour | axis | kind | seam/wall flags, row)` = 8 B/edge;
  coefficient derived from the two rungs. Ocean ≈ 3.2 M directed edges ≈ 26 MB
  (only the seam/wall tiles are actually read). **2 loads per edge, 1 level** (vs
  10–16 / 4–5).
- The gamma pair catalog is this same edge list (a pair is an undirected edge).
- Per row: a **rowAccepted bitset** (1.57 M bits = 197 KB) for row-tile passes;
  per cell: static flags `touchesSeam`, `touchesWall`, `transportActive` (static
  without rigid bodies; a per-frame plane when bodies exist).

### Level 3 — One collocated transport-velocity plane, one sampler
Today a velocity corner is a 3-way select (liquid → source bank; blast-stamp → VEX
destination bank; else → accepted cache with an `atomicExchange` reuse stamp), and
face-prep additionally rasterises that value to a dense fine lattice
(`FACE_VELOCITY_SUPPORT`, 4 floats per fine point: ocean 2.05 M points, 33 MB written
per frame) so its trace can be arithmetic. Replace both with **one per-cell plane**
(16 B/cell, ocean 8.6 MB): VEX commit writes the blast cells, collocate writes liquid
cells at frame end, so the plane is always complete. `sampleVelocity`,
`sampleFaceVelocitySupport`, `traceFaceDeparture`, the sharpening trace, tracer
advection, D4's 8-corner fold and presentation's coarse-page gather all become the
same function: **8 × (LBD shared lookup + 4 plain loads) = 32 loads, 2 levels** per
trilinear sample. The bits are the same as today: both paths read
`cm12ExtensionTransportVelocity(cell)` into `result += wx*wy*wz*v` in the same
dz→dy→dx corner order with the same span rule (confirmed against `fvrSampleRead`),
provided no bank is written between VEX commit and the last consumer — which is the
case at HEAD (collocate is the next writer).

### Level 4 — Per-frame compiled predicates that are computed twice today
`pressureCellSubmerged` (nested incidence×term walk with rowAccepted and
`pcmCellContains`) runs in `preparePressure` **and** `collocateAndDiagnose` for every
ρ<1 member — in a smeared pool that is everywhere. It depends only on PCM membership,
so it is computed once per PCM generation (per flipped cell + 1-ring in the repair
pass) into a per-cell flag, and read twice. Same for the diagnostics seam test
(static → Level 2 flag) and θ (already once).

### Memory on ocean
LBD 7 KB · tile lists ~0.3 MB · tile masks 8 B × 600 × causes · rowAccepted 0.2 MB ·
edge CSR 26 MB (seam tiles read; uniform tiles never touch it) · velocity plane
8.6 MB (replaces a 33 MB rasterised support). The arena's incidence/row/term words
leave the hot path entirely.

### Build cost and rebuild rule
Compile per changed brick + 26 neighbours at topology commit (LBD full rewrite is
600 lanes). Ocean: 0 commits in 24 frames → built once. The per-frame parts are the
velocity plane (written by stages that already write those values) and the submerged
flag (inside PCM repair, which already visits the flipped cells). Generation-stamped
like the existing VEX/PCM authorities.

### Bit-exactness contract
- Exact by construction: same cell ids, same rows, same coefficients (dyadic), same
  accumulation order (edge CSR preserves template order; implicit enumeration must
  be defined to match it — verify at construction, gate with the digest lane), same
  sampler expression. Moving *integer* reads off the atomic binding was already shown
  bit-identical; derived dyadic floats are the same values, but any expression that
  changes shape around a former `atomicLoad` is a reassociation hazard (the P1 trap) —
  every kernel conversion is one digest gate, not one assumption.
- Order-free by nature: diagnostics max, ballots/ORs, tile-superset root sets.
- Unchanged contracts: β scatter atomics keep whatever order-independence they have
  now; local staging (EXP_MASS_LOCAL_ATOMICS) becomes attractive because a tile's
  departure stencils land mostly in its own brick.

## 3. The test: every stage's access pattern against the structure

Ocean ms are HEAD (bimodal lane). "After" is theory: loads removed per item × items,
assuming the frame stays load-issue bound. Fusions are named where the same traversal
already exists.

| stage | ocean ms | today (per item) | under the structure | after (est.) | exactness |
|---|---|---|---|---|---|
| **transport-velocity-extension** | 5.70 | 28 dispatches. Blast from roots via incidence→row→terms with CAS per root. 8 sweeps over arrival-ordered blast list; per edge: rowAccepted (4–10) + cellActive (5) + stamp + 4 velocity (+ `atomicExchange` on cached reads). Wet cells copy-and-return. | Roots = VEX tile mask (ballot in collocate). Blast = 8 mask dilations (brick-local). Sweeps dispatch over masked tiles; per edge: implicit neighbour + 4 plain loads from the candidate plane, 0 acceptance loads. Commit writes Level-3 plane directly. | ~1.5 | Same BFS set or superset → same fixpoint. Keep 8 depth sweeps (a shared-memory in-tile iteration would change order — not exact; optional later). |
| **face-preparation** | 4.65 | Retired-support clear + support publish over all resident bricks (33 MB) + dirty-brick row trace: per row rowAccepted, 2 support lookups, ≤7-sample trace (8 plain loads/sample — already cheap). | No rasterised support. Row tiles × rowAccepted bit; trace via the Level-3 sampler (same cost per sample as today, but the 33 MB publish is gone). Fold **body-forces** here is *not* safe on dirty-brick-only visits — see body-forces. | ~1.5–2 | Sampler bits unchanged (same expression, same source). |
| **conservative-transport** | 15.20 | 3 SRR1 passes + compare, each lane re-derives cell (`ownerCellAt` + `scaCellBounds`, ~20 loads/pass). traceGammaAndBeta: ≤7 samples × ~140 loads + `transportStencil` 9 × 12 + clearance second gather + 8 β atomics; gather: departure cache + tra1 closure walk + record-incidence (incidence→rows→terms→CAS for ~every wet cell). | Cell tiles (id arithmetic, 0 loads/pass). Sample = 32 loads; stencil corners = 8 LBD lookups; closure/producer recording = tile-mask ballots, no incidence walk, no CAS. β local staging becomes viable per tile. | ~3–4 | Trace & stencil cell ids identical; β contract unchanged; closure set superset-safe. |
| **tracer-advection** | 0 | Same pointer-chased sampler. | Level-3 sampler. | — | exact |
| **gamma-diffusion** | 1.51 | TRA1 leaf packets scatter; two finalize walks over all accepted cells via pair CSR (3-word stride, generation stamp per pair). | Pairs = accepted edges; finalize = cell tiles × implicit/compiled edges, no stamp test (acceptance is compiled); scatter over scalar-dirty tiles. | ~0.8 | Pair order = template order; preserve. |
| **surface-sharpening** | 5.70 | SCA1 prepare: per lane ownerCellAt + bounds; `sharpeningStats` = incidence → rowAccepted → two term passes; trace 11 ownerCellAt/substep ×≤7; scatter atomics; finalize dense + record-incidence + `scaInvalidateCellClosure` (≥27 tile appends/cell); 7 temporal-worklist dispatches (classify, 2 incidence dilations, atomicAdd compactions) + receipt + SRR1 publish. | Stats from compiled edges (neg×pos counts are a static per-row property → one flag); trace via Level-3 sampler + LBD; finalize = cell tiles; invalidation/temporal lists = tile-mask ballots + 2 mask dilations (brick-local), 7 dispatches → 2; receipt unchanged. | ~2 | Stats accumulation order = template order; trace exact; lists superset-safe. |
| **symmetry-authority (D4)** | 0.07 | 8 ownerCellAt per cell. | 8 LBD lookups. | ~0.03 | exact |
| **body-forces** | 0.20 | Per row: rowAccepted + `+= open·dt·g` + mirror to source bank. | Row tiles × rowAccepted bit. Fold into projection/RHS is *possible* bit-exactly (`fl(u + fl(fl(open·dt)·g))` in register) but dry rows outside the pressure set still need the write for the band's reset-to-`0+dt·g` semantics — keep it a cheap row-tile pass. | ~0.1 | exact |
| **pressure-topology** | 4.13 | 55 dispatches / 17 passes (25 singletons, 26 copies). Membership classify over 2-ring dilation of temporal cells (= everything) with `pressureCellSubmerged` nested walk; PTR1/PCF1/PCFA repair; materialize addresses; `classifyPressureRow` over 1.18 M temporal rows (two term walks + 4 atomicAdds); `preparePressure` incidence gather + submerged walk again. | Membership/θ/coefficients as cell-tile & row-tile passes over implicit/compiled edges with no rowAccepted; submerged computed once per PCM generation (Level 4) and read by both consumers; dirty selection by pressure tile masks. The 17-pass authority chain is the remaining cost — it is shaped for sparse change, and on ocean change is dense. Target: fold PCM→PTR→PCF into ≤ 6 tile-major dispatches keyed on the same masks. | ~1.5 | Gathers keep template order; submerged is a pure function of membership — identical. |
| **pressure-rhs** | 0.79 | 14 dispatches incl. 5 capacity-shaped. | Rank→cell is already a flat list; RHS gather from compiled edges (1 level). | ~0.4 | exact (order preserved) |
| **pressure-solve** | 26.4 | out of scope | SpMV walks the same edge CSR; no change proposed here, but the implicit-interior stencil applies to the pipelined image as well. | — | — |
| **velocity-projection** | 6.09 | 16 dispatches. FPA1: pressure-cell seed (atomicExchange), row marks via incidence CAS, dirty-brick marks at brick capacity, leaf repair, 6-level rank-tree descent per row, term walk + `pcmCellContains` per term; collocate: incidence walk with rowAccepted per incidence + `pcmCellContains`, Kahan fold, record-incidence CAS closure on every changed cell, MarkCellClosure, submerged walk for ρ<1. | Execute over row tiles: rowAccepted bit ∧ θ>0 (every pressure row is live whenever the solve ran — on ocean FPA1's 10 marking/repair dispatches select everything). Collocate over cell tiles: implicit/compiled edges, submerged flag (Level 4), VEX roots and ACT1 marks by tile ballot. **Fold projection-diagnostics in** (per-workgroup max partials, one reduce). Writes the Level-3 plane for liquid cells. | ~1.5 | Kahan fold order = template order; max is order-free. |
| **projection-diagnostics** | 0.59 | Dense re-walk of incidence for seam test + divergence max. | Seam flag static (Level 2); max fused into collocate. | ~0.05 | exact |
| **activity-measurement** | 3.41 | `measureBrickActivity` per dirty brick (all 500): per cell incidence walk + rowAccepted + 8-sibling restriction probe (7/8 redundant) + roster scan; 156-B stride marks. | Cell-tile census over implicit edges; sibling restriction by local arithmetic; per-tile summary from ballots; brick score = 64-tile reduction. | ~1 | census integers — exact |
| **resolution-planning** | 0.26 | 24–27 directory probes per brick. | 27 LBD reads (shared). | ~0.1 | exact |
| **candidate-transfer** | 0.07 | — | — | — | — |
| **brick-retirement** | 0.52 | 26 neighbour directory lookups per brick. | 26 LBD reads. | ~0.1 | exact |
| **presentation-publication** | 2.95 | FPL1 plan passes at brick capacity (8 × 600 × 64 lanes), FPP1 execute on dirty pages; coarse pages gather via `compactOwnerCellAt`. | Plan = the same tile masks (presentation cause); coarse gather via LBD + Level-3/density planes. | ~1.5 | exact |
| **launch floor** | ~5–8 (505 launches) | capacity-shaped + compaction + indirect chains per stage | one dispatch per tile-major stage; masks replace compactions | ~2–3 | — |

Sum (theory): **53.5 → ~15–18 ms non-pressure on ocean**, floor set by the transport
trace's ~90 M plain loads, the pressure-topology pass structure, and ~150–200
remaining launches. The mini dam will move far less (it is launch-bound) — do not
gate this program on mini ms; gate on ocean proportions and digests.

## 4. The coalescing map (one traversal where there are several)

1. **One sampler** (Level 3): face-prep trace, transport trace + `transportStencil`,
   sharpening trace, tracers, D4, presentation coarse gather, VEX velocity reads.
2. **One stencil** (Level 2): preparePressure, collocate, diagnostics, sharpeningStats,
   activity census, record-incidence, submerged, MarkCellClosure, tra1, VEX sweep,
   gamma pairs, projection row terms — all read the implicit/compiled edges.
3. **One item enumeration** (Level 1): the 4 SRR1 passes, 2 SCA1 passes, every
   dispatchAccepted kernel, FPA rows — item id is arithmetic on (tile, lane).
4. **One acceptance** (rowAccepted bitset / compiled edge filtering): removes every
   per-visit rowAccepted/cellActive in the census above.
5. **One dirty bookkeeping** (tile masks + ballot + brick-local dilation): VEX roots,
   ACT1, SCA invalidation, temporal scalar cells/rows, FPA marks, TRA1 leaves,
   presentation pages. Replaces ~3 M CAS spins/frame on ocean with ~30 k atomicOr.
6. **submerged once** (Level 4): preparePressure + collocate.
7. **diagnostics into collocate**; seam test precompiled.
8. **VEX cache + source + blast banks → one plane**; face-prep's 33 MB rasterised
   support deleted (Level 3 subsumes it).
9. **gamma pair catalog = edge list**.
10. **body-forces stays a row-tile pass** (fold into RHS/projection is exact but
    dry-band rows need the write) — 0.1 ms, not worth the semantic risk.

## 5. Where the structure does not help, or hurts

- The pressure solve (26 ms, 33 %) is untouched by this plan; the implicit stencil is
  applicable to its SpMV but that is a separate program.
- Dirty tracking on ocean buys little physics: 73 % of cells change bits per frame.
  The plan makes "dense" cheap rather than pretending ocean is sparse. A truly still
  pool (no seiche) would then be bounded by mask dilation + launch floor.
- Memory: the edge CSR (26 MB) and plane (8.6 MB) exceed SLC on ocean — but so does
  today's arena; what changes is unit-stride tile access instead of pointer chases.
- Rigid bodies: `cellTransportActive`/open fractions become per-frame planes again;
  LBD point→cell must still consult the per-cell open volume where bodies exist.
- Macro bricks (span > 1) and clipped domain-edge bricks: LBD carries origin and valid
  extent; the local-index formula must be span- and clip-aware (the B5 trap).
- Run-to-run nondeterminism from arrival-ordered lists goes away with tile-major
  dispatch (items are enumerated, not claimed) — long-dam oracles can become
  cross-run. That is a bonus, not a goal.
- Implicit rows need a predictable row-owner order inside (brick, rung). If the
  template builder does not guarantee it, the edge CSR path (2 loads/edge) is the
  guaranteed baseline; the implicit path is the upgrade.

## 6. Proposed sequence (each step one digest gate, ocean proportions as the yardstick)

1. **LBD + sampler unification** (Levels 0, 3). Widen LOD1; add the per-cell plane
   written by VEX commit and collocate; point `sampleVelocity`/`transportStencil`/
   `sirMassTileCell`/sharpening/D4/presentation at it; delete the rasterised support
   publish. Expected: transport 15 → ~5, face-prep 4.7 → ~2, sharpening −2, VEX
   reads cheaper. Largest win, smallest blast radius, bit-exact by the sampler
   argument in §2.
2. **Tile lists + masks** (Level 1): SRR1/SCA1/dispatchAccepted/FPA execute become
   tile-major; roots/marks/invalidations become ballots; temporal worklists → masks.
   Expected: projection −3, sharpening −2, transport −2, launch floor −2.
3. **Compiled acceptance + edges** (Level 2): rowAccepted bitset, accepted-edge CSR,
   static seam/wall flags; convert collocate, preparePressure, activity, stats,
   VEX sweep, gamma. Expected: VEX −3, activity −2, ptop −1.5, diag → fused.
4. **Implicit interior stencil** (Level 2 upgrade): needs row-order guarantee; pays
   off in every edge walk and sets up the same trick for the solve.
5. **Pressure-topology tile-major fold** (Level 4 + pass restructuring): the 17-pass
   chain → ≤ 6 dispatches over pressure masks.

What to measure: ocean stage ms as proportions across ≥ 3 captures, digests per stage
conversion (bit-exact or a blessed one-time move with the reason named), launch count
per frame, and the receipt's `acceptedCells/temporalScalarCells` so a "static" claim is
never taken on faith again.

## 7. Compile cost under churn (dynamic scenes such as mini dam break)

The compile is cheap by construction, not by hope; four reasons and two design choices.

1. **O(surface), not O(cells).** Inside a brick at rung r everything is template-identical
   to every other brick at r: intra-brick rows are accepted iff the brick is active at r
   (one bit per row range — a range fill), intra-brick neighbours are implicit arithmetic,
   coefficients are dyadic. Only the **face layers** depend on the neighbourhood (which
   neighbour rung sits across each of the 6 faces → seam rows, seam edges, uniform/seam tile
   flag). Per changed brick the compile writes 1 LBD entry + face-row acceptance
   (6 × 256 ≈ 1.5k rows at rung 16, vs 12k owned) + seam-edge slots for the 16 face tiles
   per face, and the same for the ≤ 6 face neighbours whose facing layer changed. The 2.5k
   interior cells of a 16³ brick are never visited.
2. **Bounded above by one traversal, even at 100 % churn.** Each face row's acceptance is
   evaluated once from the same requirement list `rowAccepted` reads today; HEAD evaluates
   it per visit in face-prep, body-forces, 8 VEX sweeps, collocate, diagnostics, stats,
   activity and the closures — ≥ 10 times per frame. Worst case is "one pass instead of
   ten"; it cannot regress.
3. **Mini dam is 8 logical bricks.** Full recompile ≈ 8 × 27 LBD reads + ≤ 8 × 1.5k face
   rows + a handful of seam tiles ≈ 10–20k lane-ops — under one launch's fixed cost. Mini is
   launch-floor bound; the structure helps it through pass fusion (tile-major stages,
   masks replacing compactions, temporal worklists 7 → 2), not compile savings. Large
   dynamic scenes (long-dam, 248-dam) change ~5–10 % of bricks per frame at the front →
   compile ∝ that fraction's surface.
4. **It runs in the frame tail, double-buffered by generation.** Commit already happens at
   the tail and the plan is already sealed-at-tail/executed-at-head (VEX1). Compile is extra
   writes in the passes that already visit changed bricks (commit, PCM repair, presentation
   replan) into generation N+1 while frame N reads N. The L3 velocity plane needs no compile
   (per-frame writers); rung changes invalidate it exactly as the VEX cache stamps are
   invalidated today.

Design choices that keep churn cheap — decide up front:
- **Fixed-capacity seam slots, no prefix sums.** A compacted CSR with global offsets would
  force a world-wide recompaction whenever one brick's edge count changes. Seam edges live
  in per-tile fixed slots (2:1 grading caps a cell at 6 × 4 = 24 edges; only seam tiles
  allocate; uniform tiles allocate nothing) → brick-local, parallel, write-once. The tile
  lists are the one global artefact; recompacting them is one pass over ≤ 64 × bricks lanes
  (ocean 32k, mini 512).
- **Face-tile granularity for uniform↔seam flips.** When a neighbour refines, only the
  facing 16 tiles of each adjacent brick flip from implicit to compiled; the other 48 stay
  implicit. Rung thrash is the only way compile could become visible — and thrash costs the
  physics more than the compile; rely on (and check) planning hysteresis rather than on
  compile speed.
