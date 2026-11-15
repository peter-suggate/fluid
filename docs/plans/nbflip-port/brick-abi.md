# NB-FLIP brick engine: the brick ABI (lane E, M0(e), frozen 2026-11-15)

This is the storage and scheduling contract of the new brick engine in `lib/methods/narrow-band/`
(design: `docs/plans/nbflip-structural-2026-11-15.md` §2, §5). Lanes F (fluid kernels) and P
(pressure and bodies) write kernels against it. **Changes go through lane E**: a lane that needs a
new channel, list, word or bit asks E, who amends this doc and `lib/methods/narrow-band/brick-abi.ts`
in one commit. Every number here is also a constant in `brick-abi.ts`, and `brick-abi.test.ts`
checks the two agree on the invariants below.

Sources:
- the brick arena (`lib/methods/uniform/uniform-brick-arena.ts`): the 8³ brick, the 520 stride,
  the 32-word record, the seam rule, free-stack fail-fast, lists in slot order;
- seep-m1max (`lib/methods/seep/`): brick-major `slot << 9 | local` addressing, the ctrl block,
  per-brick counting sort, gated slots, the per-dispatch `cfg` uniform, two parity variants,
  the two-part submit and encode-ahead;
- the T2o directory of lane X's bench (design §4.1): one page word, then one child word, per tap.

What is deliberately **not** here: the old engine's owners, windows, page pools, fine slots,
textures, hanging taps or certificates. The new engine imports pure WGSL formula functions and
the host contract only.

## 1. Geometry and conventions

| thing | size | notes |
|---|---|---|
| h cell | 1 | the lattice's finest cell (`voxelDomain.finestCellSize_m`) |
| tile | 4³ h | the unit of h / 4h ownership (unchanged from NB-FLIP, §7.1) |
| F brick | 8³ h = 2³ tiles | `local = x \| y<<3 \| z<<6`, x fastest |
| page | 32³ h = 4³ F bricks = 8³ tiles | `brickLocal = bx&3 \| (by&3)<<2 \| (bz&3)<<4` |
| C brick | the page at 4h: 8³ cells, one a tile | `local = tx&7 \| (ty&7)<<3 \| (tz&7)<<6` |
| B record | a BULK16 page: 2³ 16h cells | one page a B slot, `BULK16_PAGE_WORDS` (960) words, NB-FLIP's record (M3) |

**Coordinates.** Cell coordinates are integers in the *window frame*: the lattice's cell (0,0,0) is
at the frame origin. Static scenes have frame origin 0; the moving window (M6) offsets it. A
brick's coordinate is `cell >> 3`, a page's `cell >> 5`, a tile's `cell >> 2`.

**Staggering: a cell owns its lower faces and its lower corner.**
- Face channel `a` of cell `c` is the face between `c - e_a` and `c` (seep's convention; MAC
  index `c - ½e_a`).
- A vertex channel at cell `c` is the vertex at `c`'s `(-,-,-)` corner (seep: "vertex v stored at
  cell v").
- So a brick owns faces and vertices with local index 0..7. Its +side faces and vertices live in
  the + neighbours.

**Skirt bricks.** A field of `N` cells along an axis has `N+1` vertices and `N+1` faces.
- Faces on the domain's walls (index 0 and `N` along their axis) are walls: readers return the
  wall value by the box rule and never load them.
- Vertices on the upper planes (`v.a == N_a`) are stored in *skirt bricks*: ordinary F bricks whose
  origin lies at or beyond `N_a` on some axis, flagged `F_CLASS.SKIRT`. They are made resident by
  the same reach rule as any brick, so they exist only where liquid nears an upper wall. Only their
  vertex channels are meaningful; their cells are not open.
- No reader clamps a vertex index into the box: that would change NB-FLIP's boundary values.

**The box rule.** A cell outside `[0, N)` on any axis is solid (a wall), except where the scene's
container top is open (`container.top == "open"`: above `N_y` is air). Readers apply the box rule
from coordinates *before* the directory lookup, so no lookup sees a negative coordinate.

**Units** (as NB-FLIP stores them, so ported formulas are verbatim):
- positions: h cells in the window frame (cell `i` spans `[i, i+1)`);
- level set φ: h cells, negative in liquid;
- velocity: m/s;
- time: s; `dt = 1/60` (no substeps);
- pressure: lane P's units, stated in its `pressure-*` header (P1's arena units).

## 2. Slots, pools and channels

### 2.1 Slot kinds and capacities

| kind | slot of | capacity (authored per scene) | free stack |
|---|---|---|---|
| F | an F brick | `FCAP` | `F_FREE` |
| C | a resident page | `CCAP` | `C_FREE` |
| B | a BULK16 page | `BCAP` (0 until M3) | `B_FREE` |
| K (cut) | an F brick with a static-solid cut | `KCAP` | `K_FREE` |

Capacities come from `scene.fluid.brickPools` (§9), set from the M0(b) measured peaks. There is no
runtime growth. A pop from an empty stack latches the kind's `FAIL.POOL_*` bit (§8), counts the
refusal and leaves the brick non-resident; it never clamps or evicts.

### 2.2 Data pools

A *pool* is one GPU buffer, declared in WGSL as `array<u32>` (f32 by `bitcast`) or `array<f32>`
by the module that binds it. A kernel binds only the pools it uses.

Inside a pool, storage is **channel-major, then slot, then cell**:

```
word(pool, kind, ch, slot, local) = BASE[pool][kind][ch] + 520 * slot + local      (local < 512)
BASE[pool][F][ch] = ch * FCAP * 520
BASE[pool][C][ch] = NCH[pool] * FCAP * 520 + ch * CCAP * 520
```

- **Stride 520, not 512.** The 27 neighbours of a brick then do not alias the same cache sets.
  The arena measured 2.5x slower staging at 512. Words 512..519 of each slot are padding: never
  read, never written.
- **F and C share a pool's channel list.** The C region holds the same channels at 4h.
- The cut pool has a K region only: `BASE[cut][K][ch] = ch * KCAP * 520`.

### 2.3 Channels (v1)

| pool | ch | name | layout | written by | read by |
|---|---|---|---|---|---|
| `vel` | 0–2 | `U0 U1 U2` | lower face | `p2g`, `uAdvect`, `project`, `uExtend` | everyone |
| `vel` | 3–5 | `UP0 UP1 UP2` | lower face | `p2g` (the transferred velocity); `uDelta` overwrites with the FLIP delta Δu | `pTrace` |
| `vel` | 6–8 | `US0 US1 US2` | lower face | scratch: u* before projection, extension ping-pong | F, P |
| `phi` | 0 | `PHI` | vertex | `phiUnion`, `redistSweep` | everyone |
| `phi` | 1 | `PHI1` | vertex | scratch: φ′ (`phiAdvect`), redistance ping-pong | F |
| `phi` | 2 | `FLAGS` | cell, u32 | `setInit` at entry (OPEN by the box rule, OWNER_H), `solidStamp` (OPEN), `phiUnion` / `redistInit` (LIQUID, BAND), `p2g` (VALID_*) | everyone |
| `pres` | 0–9 | `X R Z D Q RHS DIAG K0 K1 K2` | cell; `K_a` the coupling across the lower face along `a` (on C rows: native–native only) | `operator`, solver | P; `project` |
| `pres` | 10–12 | `KU0 KU1 KU2` | cell, h rows only (unused in C): the seam coupling V/(2.5h²θ) across the upper face along `a` when the upper neighbour is a 4h row, else 0 (amended at M2, lane P) | `operator` | P |
| `pres` | 13 | `PPHI` | cell: the pressure authority's centre φ (`rootPhi`, uniform-mixed-pressure-authority.ts), for the liquid test and ghost θ of native and seam rows (amended at M2) | P's setup | P |
| `pres` | 14 | `B` | cell: the row's start (pInit/P_BASE), for the seam heads in `project` (amended at M2, lane P) | `operator`, P's setup | P |
| `cut` | 0 | `FRAC` | cell | `solidStamp` | `operator`, F's solid paths |
| `cut` | 1–3 | `AP0 AP1 AP2` | lower face | `solidStamp` | as above |

- **The `phi` pool is bound as `array<u32>`.** φ words are read as `bitcast<f32>`, so FLAGS
  words never pass through a float register: a denormal pattern could be flushed.
- **Entering bricks** are initialised by `setInit`:
  - vel and pres: 0;
  - `PHI` and `PHI1`: `PHI_AIR` (6, in h);
  - FLAGS: OPEN inside the lattice, and OWNER_H.
- **The `pres` list is lane P's.** P may replace it (more vectors, MG scratch), as long as it stays
  a pool indexed by F and C slots at stride 520. P announces the final list to E before M1 lands.
- **MG levels above 4h are P's own.** P allocates them sized from `FCAP`, `CCAP` and `BCAP`, keyed
  by the engine's slots or keys, and rebuilt per step from the lists. They are not channels here.
- **A brick with no cut slot** (record word 30 = `NONE`) is fully open: `FRAC` = 1 and every
  aperture is 1, except where the box rule closes a face. It holds no cut channels.
- **Crossing fractions.** The old native rows' recorded crossing fractions (`nTheta`,
  `rootTopology`) exist only on cut tiles. A C cell has no cut in M2 (below), so they get storage
  with M3's cut tiles, not a channel now.
- **A C cell has no cut.** "4h owners never reach the helpers": promotion keeps liquid one tile
  from cut cells, as today.

### 2.4 FLAGS bits (cell, u32)

These are the same on F cells and on C cells (a tile). `setInit` writes OPEN, plus OWNER_H on F only.
Every other bit of an entering slot is 0.

| bit | name | meaning |
|---|---|---|
| 0 | `OPEN` | not static solid (box rule and the cut's `FRAC > 0`) |
| 1 | `LIQUID` | cell-centre φ < 0 |
| 2 | `BAND` | within the particle band R of the surface |
| 3–5 | `VALID0..2` | the lower face along `a` has a known velocity (extension's seed) |
| 6 | `OWNER_H` | the cell's tile is h-owned in this brick (the record mask's bit, per cell) |
| 7 | `WARM` (P) | the cell held a liquid row of its tier last step (the old pInit/warm rule; P's finish kernels write it) |
| 8–10 | P: wall state / release | bit 8 + a, on the lower face along `a`'s storage (planes 0 and N; plane N at the skirt cell or C tile N/4). Between the rows (`rowsF`/`rowsC`) and `project`: the separating walls' solver state (1 = separated). After `projectF`/`projectC` (P, 9e5d2b14): umRelease (uniform-mixed-pressure-velocity.ts 485–492), `!held && ghost <= 0 && -sign·U·dt > 1e-4·width·h`; read by E's release port (`nrWallReach`, `nrReleasedWalls`; levelset-release.wgsl.ts) before the next pressure stage, never during it |
| 11–12 | reserved, P | spare |
| 13–15 | reserved, E | |
| 16–31 | reserved, F | |

### 2.5 Tiers: what is stored where (M2, proposed 2026-11-15)

**Ownership.** A tile is 4³ h. Each tile is h-owned or 4h-owned (§7.1: the 4³ granularity is
kept exactly).
- `CMASK` bit `t` of the page's C slot is the truth.
- The F record's word 28 (bits 0–7) copies the bits of its 2³ tiles.
- `FLAGS.OWNER_H` is not maintained at M2 (`setInit` sets it at entry). Readers take `nbTileOwnerH`.
- All three are written only by `setTransfer`'s commit, so every stage of a step sees one
  ownership.

**Residency** (implemented 2026-11-15 behind `BrickStages.tiered`). With it off (the M1 engine, all-h),
every tile of a resident page is h-owned: `setClaimPages` sets CMASK all-ones, and a brick's claim or
leave leaves CMASK alone. So a tile whose brick isn't resident is an h region with no slot. It reads
`NB_PHI_FAR` and 0 faces, which is air, and censusCoarse stores it as the identity. C φ is never an
authority all-h. Before this rule, CMASK was set per brick, and an absent brick's tiles read bootPhi's
stale C vertices: lane P's phantom 4h liquid, 296 wet tiles on the dam, 2026-11-15.
- *C (a page).* Resident while any F child lives, or while the head's residency (§5.4) marks it
  (`MARK_C`): tiered, a page is requested at page granularity by every stage's records (the old 4h
  storage); all-h, a page is requested only as the parent of a requested brick. A missing page becomes
  a page candidate, key `nbKey(4p) | NB_KEY_PAGE` (bit 30), which `setClaimPages` claims and
  `setClaimBricks` skips. `setMarkPages` (part 2, C_ALL, G1) counts the page's 4h-owned in-lattice
  tiles with a liquid corner (a C vertex with φ < 0) into the C record's `LIQUID_TILES`, which feeds
  `C_LIQ`; it keeps nothing. A non-resident page reads as AIR (P's V rule).
- *F (a brick).* Resident when the head's residency (§5.4) keeps it:
  - tiered, the ownership records (an h tile now, or a band tile) and the particle records, at F
    granularity, plus the 27-brick ring of every band tile (censusDecide: present ring bricks get
    their keep mark at a step's head, absent ones become candidates);
  - all-h, every stage's records at F granularity (the brick is the h storage).
  Bricks entered at a tiered head hold 4h tiles until the census and transfer say otherwise.
  `setMark`'s class scan (SURFACE) reads h-owned tiles only.
  - Bricks claimed in a step start 4h (word 28 tile bits 0, CMASK untouched). The boot's bricks
    (`setClaimBricks` with `cfg.z = 1`) start h, because `bootPhi` writes F; the first census
    coarsens them.
  - `setInit` also sets entering C slots to air: φ AIR, every velocity and pres channel 0, and FLAGS
    OPEN on in-lattice tiles.
  - The census keeps WANT as the old decision. A band tile whose brick or in-range 27-neighbour is
    not resident lists the missing bricks as candidates. The part 1 head claims them right away
    (claims, records, init, lists), before the transfer. `censusWant` then checks every WANT bit's
    ring. A miss, which can only be a refused pool, latches FAIL.TIER and never yields a silently
    coarser tier. The head then clears the churn counts for part 2 (`stepEnd`, cfg.z = 1).
  - Test: brick-set-tiers.test.ts.
  - A ring brick's tiles are all 4h-owned. It exists to hold the interface faces and the
    vertices of the h tiles beside it, so that every h stencil resolves within the 27-neighbour
    record.

**Values, per channel layout:**
- *Cells (`pres`, `FLAGS`, `FRAC`).* An h-owned tile's cells live in its F brick, and a
  4h-owned tile's single cell lives in the C brick (C local = the tile's index in the page). The
  h cells of a 4h-owned tile in a core or ring brick hold no state. Nothing reads them, except
  `setTransfer` as a prolongation target.
- *Vertices (`PHI`, `PHI1`).* NB-FLIP's authority rule, verbatim (`umVertexAuthority`,
  uniform-mixed-vertex-sampling.wgsl.ts l.78): **a coarse incident tile wins**.
  - *h vertices.* A vertex whose incident tiles are all h-owned is stored in the F brick that
    holds it (the lower-vertex convention), and is the truth. Only h passes write it.
  - *Corners.* A 4h-owned tile's 8 corners are C vertices (C vertex `v` is h vertex `4v`). Every
    corner incident to a 4h-owned tile is C's, even when it also bounds h tiles. Only C passes
    write it. Its F copy is not state, and readers take C.
  - *Hanging vertices.* Every other vertex incident to a 4h-owned tile hangs: it is the
    trilinear interpolant of that tile's corners (`umVertexFrom4`, `nbfRemapRefine`), derived and
    not stored. This includes the non-corner vertices of an h|4h plane. Two 4h tiles that share
    the vertex give the same value, because their faces share corners.
  - So no vertex has two writers. The prelude's `nbVertexOwnerH(v)` is true exactly when no
    incident tile is 4h-owned.
- *Faces (`U*`, `UP*`, `US*`, `AP*`).* NB-FLIP's lattice walls separate (lane P, 2026-11-15), so
  wall faces hold a velocity. They use the lower-face convention: plane 0 is stored at cell 0, and
  plane N at the skirt cell (the vertex row), in F or C by the in-lattice cell's owner.
  - A face with an h-owned cell on either side is an h face, stored at h in the F brick of its
    upper cell (the lower-face convention). That brick is a core or ring brick, so it is resident.
  - A face between two 4h-owned tiles is a 4h face, stored in the C brick of its upper tile.
  - So at an h|4h interface the 16 h patches are stored, and the 4h tile's own face holds no
    state: its V-weighted mean (umPressureFaceV) is derived by P's to-pressure transfer (ruling
    of 2026-11-15).

**Readers resolve the owner.** No kernel reads a non-owned copy. The ABI prelude gains
`nbTileOwnerH(cell)`, `nbVertexOwnerH(v)`, `nbVertexPhi(v)` and `nbFaceU(c, a)`, which apply the
rules above through the directory and masks:
- an h tap in an h-owned tile reads F;
- a 4h-owned tile reads C, with the interpolant at the vertices;
- a vertex that no slot stores, or that lies outside the lattice, reads `NB_PHI_FAR`. That is the old
  vertex field's empty value, `fluidLatticeFarPhi`, in h: the lattice's largest extent (`phiFar`).
  It differs from stored air (`NB_PHI_AIR` = 6, which `setInit` writes). This was P's finding,
  2026-11-15.
  - Residency guarantees that an absent page's vertices are at least 7 cells from 4h liquid
    (setMarkPages' margin) and 8 cells from h liquid (the core ring).
  - The old engine's 33³ page boxes store their upper-face vertices; here they belong to page
    p + 1. So the two engines differ only there, and only in arithmetic that far from any liquid.
  - A tile of a non-resident page is no vertex authority (`nbVertexCoarse` skips it): no slot stores
    its corners. Taking it as 4h made a vertex on an absent page's upper face the interpolant of the
    next page's C corners, which all-h hold bootPhi's stale φ. That was lane P's absent-page guard on
    the all-h dam (2026-11-15: a cell at (63,63,62) read φ -1.58 where F's corners give 67). It was
    not a reach defect: in 461 instrumented all-h steps (three runs), no F liquid vertex had an absent
    brick in its 1-ring, and the least slack between liquid and an absent brick was 7 cells.

**Schedule** (`BrickEngine.scheduleFor`, landed 2026-11-15):
- Part 1 opens with the head. This is `setCensus` (`brickCensusSchedule`: censusBegin, censusFine
  on F_ALL's X8 (a workgroup a brick's tile, a lane a cell, the tile's 5^3 vertices and 240 faces staged once),
  then censusCoarse, censusCubes (G1), censusImportance and censusDecide on C_ALL; the C passes
  but censusCubes run on C_ALL's X8, a workgroup a page's 64-tile slab, a lane a tile). When
  tiered, the claims of the census's ring bricks follow, then `censusWant` (C_ALL, X8), then
  `brickTransferSchedule`, then the churn clear. After the head come F's `move` and `rebin`.
- A bring-up stage set reads no TRIG. On a lattice that isn't whole pages (P's synthetic blocks) it
  runs without the census, which refuses such lattices. Every scene's fluid lattice is whole pages
  (`sceneFluidBox`).
- TRIG is one word per F slot. Bit j (0-7) is set when tile j's importance triggers heat it: the
  old need rule (`nbActivityTriggered`, particleSeeding "need"), over the final word 1 bits 16-21.
  Shape, impact or approach heat (strain and rotation too, dead under the dam's policy); thin heats
  only where dt · deformation / h > 0.02 (UNIFORM_SURFACE_TRANSPORT_ERROR_H), the deformation being
  tileErrorOf's max(4h · gradient, span) of the tile's plan words (census DEF). The budget's drop
  (`dropped`) is not ported: no scene but the seeps sets detailBudgetPercent < 100, and on the dam
  the old policy word is 100 (budget off). Bit 8 + j is tile j's thin trigger alone (word 1
  bit 17), which F's thin move path reads (F's request, 2026-11-15). Bits 16-31 are zero.
- The census runs in the all-h engine too, because F reads TRIG. There it neither commits WANT
  (no transfer) nor requests bricks (censusDecide's `cfg.z` = 0).
- The boot runs the census after its lists and before F's boot stages, since F's seeding reads
  TRIG.
  - After F's boot stages comes the first frame's projected half kick, U ← P(U + g dt/2), when
    the stages supply `kick`. The order is F's `kick.force` (US = U + g dt/2, gravity alone), the
    operator, the solver, `project`, then F's `kick.after`. The old host runs this kick before its
    first frame (webgpu-uniform-reference.ts `kickMixedFrame`; uniform-mixed-frame.ts `kick`), so
    the stored velocity leads the positions by half a step. Without it, step 1 carries g dt instead
    of 1.5 g dt (F's D5c finding).
  - With F's `kick.seed` (optional), the boot then runs a TRIG-only census of the kicked state
    (`brickTrigCensusSchedule`: censusBegin through censusImportance, then censusDecide with
    cfg.z = 2, which writes TRIG and CENSUS_TRIGGERED alone: no band, WANT, candidate or keep mark;
    no claims or transfer), and then `kick.seed`. So the first seeding reads the triggers of the
    state it seeds. The census header words after the boot are this census's.
  - A tiered boot then commits with `setCommitMask` alone and runs no remap: `bootPhi` writes one
    analytic φ to F and to the entering pages' C vertices, and U = 0. A remap would be the same
    injection at the corners.
  - A boot remap would also list every tile of the initial liquid, well over TLIST's capacity.
- The census's scratch is the engine buffer `census`, sized `brickCensusBuffers(L)`: 9376 words
  a C slot (with the near words, the moving-shape bits MOV, the residency records, §5.4, and the
  plan words DEF: 4 a tile, the old frame plan's mean extended face velocity xyz and face span,
  localSpeed's, written by censusFine and censusCoarse and read by censusDecide's deformation).

**The tier change, within a step** (§7, amended):
1. `setCensus` (brick-census.wgsl.ts, landed fb17d41e at the part 1 head; this was planned as part
   of `setMark`) decides each tile's next owner (the NB-FLIP census, verbatim) into `WANT`: 512 bits
   a C slot, `16 * CCAP` words in `set`. It also marks the bricks that the new h tiles and their
   ring need (`CAND`, `MARK_F` bit 0). A brick whose tiles change owner gets `MARK_F` bit 1
   (`CHANGED`). The census reads lane F's `TILE_OCC` (FCAP words in `set`): bits 0–7 the
   brick's tile `j` holds particles (`pScan`, last step), bits 8–15 a particle join on tile `j`
   (F's tail sweep, NB-FLIP's join rule: x ± 0.875 cells, the crossing-cell box where the home
   tile has REACH and φ ≤ 0, swept by dt·before/h, pad 0), bits 16–31 reserved zero. `pScan`
   rewrites the whole word after the census, so joins clear each step.
2. `setUpdate` claims entering bricks and rebuilds records, but does **not** free leaving ones
   yet. Their data is restricted first.
3. `setTransfer` (brick-transfer.wgsl.ts, landed 2026-11-15) is NB-FLIP's remap verbatim on this
   storage (F's `nbfRemap*` for the formulas):
   - `setTransferList` lists tiles with `markTile` and `listedFlags` over the tiles of C_ALL (X8, a lane a tile), into
     `TLIST` (`set.tlist`, 2 words an entry: `cs << 9 | tile`, flags; `cap.t` entries, default
     2 FCAP, over it `FAIL.LIST`). It counts into `H.N_TLIST`. `setTransferArgs` moves the count
     to `N_TLIST_RUN` and writes `ARGS.XFER_G1`.
   - `setTransferFaces` (192 lanes a job, channel U then US in the same workgroup; UP is not carried) is
     `remapTileFaces` with `remapBeside`, `remapSlope`, `remapSample` and `remapFace`, the
     low-wall rows included. Old faces are read through CMASK, new placement through WANT. Values
     go to the `xfer` scratch (`2 · 384 · cap.t` floats). `setTransferCommit` then stores them (both channels): an h face in F at its upper
     cell, a 4h face in C at its upper tile. That is the old remap's second texture.
   - `setTransferFaces`' lanes 0..63 then take `remapTileCells`' vertices (surfaceOnly) in place, φ = `PHI`
     (the separate `setTransferCells` pass until 2026-11-15; the face jobs read no φ). A
     vertex the old layout stored keeps its value and moves to its new storage when that
     changes. A refining tile moves its own corners to F, because their new authority tile may
     not be listed.
   - `setCommitMask` sets CMASK = WANT and the F records' word 28 tile bits.
   - Missing storage for a write latches `FAIL.TIER`.
   - V (`umPressureFaceV`, M2 without cuts) is 1 between lattice cells and 0.5 into the
     exterior. The deviations come from the storage: there are no released bits or cut weights
     until solids land (M3), and a face toward a non-resident (deep-air) page has no storage and
     reads 0.
   - Its frozen-state tests are brick-transfer.test.ts and brick-transfer-dam.test.ts. The
     second matches the old engine's real dam step-22 remap bitwise, apart from 3 φ words at
     1 ulp.
   - It is placed at the part 1 head after the census (Schedule, above), as in NB-FLIP, which
     remaps at the frame head before the particle move.
4. The frees run last, after `setTransfer` (a phase of `setLists`). A slot never leaves and
   enters in the same step, so each pool's measured peak includes one step of churn.

The transfer runs only "within one tile of a width change", as NB-FLIP's `markTile` does. BULK16
(16h) keeps NB-FLIP's rules and comes with B slots at M3.

## 3. Directory (T2o)

The directory lives in the `dir` buffer: `array<u32>` for readers, `array<atomic<u32>>` for
`setUpdate`.

| region | words | content |
|---|---|---|
| `PAGE` | `WX * WY * WZ` | one page word per window page |
| `CHILD` | `64 * CCAP` | the F slot of each F brick of a C page, by `brickLocal`; `NONE` where not F-resident |
| `FREC` | `32 * FCAP` | F records (§4.1) |
| `CREC` | `32 * CCAP` | C records (§4.2) |
| `CMASK` | `16 * CCAP` | per C slot, 512 bits: bit `t` (the tile's C local) set when tile `t` is h-owned |

**The window.**
- `WX`, `WY`, `WZ` are powers of two, each at least the lattice's pages along the axis plus one
  (for the skirt). They come from the lattice in pages, never the cell box. The dam's 4³ pages
  give 8³ words, the seep's 16 × 8 × 2 give 32 × 16 × 4.
- `pageIndex(p) = (p.x & (WX-1)) | (p.y & (WY-1)) << log2 WX | (p.z & (WZ-1)) << (log2 WX + log2 WY)`.
- It is toroidal, so the moving window (M6) reuses it by moving the frame origin.
- Window pages above `2^10` an axis are refused at build (`FAIL.WINDOW` at run time for a key
  outside it).

**Page word.**

| value | meaning |
|---|---|
| `0x00000000 \| cslot` (bit 31 clear) | resident page with C slot `cslot` |
| `0xA0000000 \| bslot` | BULK16 page with B slot `bslot` (no C or F slots) |
| `0xFFFFFFFF` (`NONE`) | not resident: air outside the CFL reach of liquid and particles |
| `0xC0000000` (`CLAIM`) | transient, inside `setUpdate` only: a page being entered |

**The lookup of an h cell `c`** (after the box rule):

```
pw   = PAGE[pageIndex(c >> 5)]
if pw >= 0x80000000: not F-resident (B page or air)
f    = CHILD[64 * pw + brickLocal(c >> 3)]
if f == NONE: not F-resident; the page's C brick gives the 4h value (the seam rule, §4.3)
v    = pool[BASE[F][ch] + 520 * f + local(c & 7)]
```

That is two dependent loads a tap: variant T2o at 1.43x dense (design §4.1).

**Residency is the brick set.** A tap that misses (`NONE`) reads air: φ = +`PHI_AIR` (the band's
positive bound, F's constant), and velocity by extension's air rule. By construction a
non-resident brick is outside the CFL reach of liquid and particles, so every reader agrees on it.
There is no second structure to disagree with.

## 4. Records

### 4.1 F record (32 words, `FREC + 32 * fslot`)

| word | content |
|---|---|
| 0–26 | neighbour `n = (dx+1) + 3(dy+1) + 9(dz+1)` (word 13 is the brick itself): a neighbour word (§4.3) |
| 27 | the parent page's C slot (always resident while the brick is) |
| 28 | bits 0–7: the h-owned tile mask (tile `tx \| ty<<1 \| tz<<2` of the brick); bits 8–13: `brickLocal` in the page |
| 29 | key: the brick coordinate, `bx \| by<<10 \| bz<<20` (window frame, 10 bits an axis) |
| 30 | the K (cut) slot, or `NONE` |
| 31 | class bits (`F_CLASS`, below) |

`F_CLASS` (word 31):

| bit | name | set by | meaning |
|---|---|---|---|
| 0 | `LIQUID` | `setMark` | an h-owned vertex has φ < 0 (F_LIQ) |
| 1 | `SURFACE` | `setLists` | some vertex within the band of the surface |
| 2 | `PARTICLES` | `setLists` | the bin is not empty |
| 3 | `NEAR_SOLID` | `setLists` | it or a neighbour has a cut slot or a box wall |
| 4 | `NEAR_BODY` | `bodyStamp` (P) | within the swept reach of a body |
| 5 | `SEAM` | `setLists` | a tile not h-owned, or a neighbour word that is a seam |
| 6 | `SKIRT` | `setUpdate` | origin at or beyond `N` on some axis (vertex storage only) |
| 7 | `COLD` | `setMark` (M3) | in a frozen region |
| 8 | `CHANGED` | `setMark` | a tile's tier changes this step (`setTransfer`'s input) |
| 9–15 | reserved, E | | |
| 16–23 | reserved, F | | |
| 24–29 | reserved, P | | |
| 30 | `ENTERING` | `setUpdate` | entered this step; its channels are not yet initialised |
| 31 | `LEAVING` | `setUpdate` | leaves this step; its slot returns at the end of `setUpdate` |

### 4.2 C record (32 words, `CREC + 32 * cslot`)

| word | content |
|---|---|
| 0–26 | reserved: `NONE` at claim and not maintained in M2. Kernels find neighbour pages through the directory, with a key check (amended at M2) |
| 27 | the number of F children (`CHILD` entries not `NONE`) |
| 28 | the number of 4h-owned liquid tiles |
| 29 | key: the page coordinate, `px \| py<<10 \| pz<<20` |
| 30 | reserved |
| 31 | `C_CLASS`: 0 `LIQUID`, 1 `SURFACE`, 5 `SEAM` (has F children), 7 `COLD`, 30 `ENTERING`, 31 `LEAVING` |

### 4.3 Neighbour words and the seam rule

An F record's neighbour word is:

| value | meaning |
|---|---|
| `< 0x80000000` | a live F slot |
| `0x80000000 \| cslot` (`SEAM`) | the neighbour brick is not F-resident, but its page is C-resident: its values come from the C brick, by the seam rule |
| `0xA0000000 \| bslot` | the neighbour brick's page is a BULK16 page |
| `NONE` | the page is not resident: air |

- A live F neighbour whose tile is not h-owned (its mask bit clear) is a seam too. The reader
  tests the mask (record word 28 of the neighbour) as the arena's stage does.
- **The seam rule is the stage's, not storage's.** The value a reader takes from a seam (the 4h
  value, its prolongation, or the composite operator's coupling) is the ported NB-FLIP formula's.
  Seam values are never written into F channels.
- Records are rebuilt by `setUpdate` for every entering brick and for every live neighbour of an
  entering or leaving brick. A record is stable from the end of `setUpdate` to the next step's
  `setUpdate`.

**Staging (particle kernels).** A workgroup per F bin reads words 0–28 of its record into
workgroup memory, and each neighbour's word 28 for the tile mask. It then stages its brick plus a
one-cell ring (`10³` or `11³` values a channel) through them, as `UniformBrickArena.stageWGSL`
does. Cell kernels address directly through §3 (design §2.1).

## 5. The `set` buffer: control, ring, stacks, marks, lists and bins

`set` is `array<atomic<u32>>` for writers and `array<u32>` for read-only modules. All offsets are
in words from its start. Regions are laid out in this order, each a multiple of 64 words.

### 5.1 Control header (words 0–255)

| word | name | content |
|---|---|---|
| 0 | `FAIL` | latched failure bits (§8), never cleared by the GPU |
| 1 | `FAIL_INFO` | the first failure's detail (a key or a count), by compare-exchange from 0 |
| 2 | `STEP` | steps completed on the GPU (the tail's last kernel increments it) |
| 3 | `STATE` | reported, non-fatal state bits (§8) |
| 4–7 | `TOP_F TOP_C TOP_B TOP_K` | free stack tops (free slots left) |
| 8–11 | `HIGH_F HIGH_C HIGH_B HIGH_K` | high-water marks: every slot ever taken is below its kind's |
| 12–15 | `LIVE_F LIVE_C LIVE_B LIVE_K` | live slots after `setUpdate` |
| 16–19 | `REFUSED_F .. REFUSED_K` | cumulative refused pops |
| 20–21 | `PCOUNT0 PCOUNT1` | particles in bank 0 and bank 1 |
| 22 | `P_OVERFLOW` | particles refused by the bank's capacity (cumulative) |
| 23 | `P_UNBINNED` | particles that landed in a non-resident brick (cumulative; latches `FAIL.REACH`) |
| 24–27 | `N_ENTER_F N_LEAVE_F N_ENTER_C N_LEAVE_C` | this step's churn |
| 28 | `N_CAND` | this step's enter candidates (append counter) |
| 29 | `N_CHANGED` | this step's tier-changed F bricks |
| 30, 31 | `N_TLIST`, `N_TLIST_RUN` | setTransfer's listed tiles: the append count, and the count its jobs run (§2.5) |
| 32–63 | `LIST_COUNT[k]` | list `k`'s entry count (§5.5) |
| 64–127 | F's | lane F's counters and receipts |
| 128–191 | P's | lane P's counters, residuals, cycle records, PCG scalars |
| 192–223 | `PEAK_*` | running maxima by `atomicMax`: 192 live F, 193 live C, 194 live B, 195 live K, 196 particles, 197 enter candidates |
| 224 | `CENSUS_CLAMPED` | band tiles the residency clamp held at 4h this census |
| 225 | `CENSUS_CAND` | candidate bricks the clamp appended (once per brick per page) |
| 226 | `CENSUS_REACH` | largest tile travel, h per step, f32 bits by atomicMax (old census[18]) |
| 227 | `CENSUS_CROSSING` | crossing tiles (old census[0]) |
| 228 | `CENSUS_REQUIRED` | required tiles (old census[12]) |
| 229 | `CENSUS_BAND` | band tiles (old census[2]) |
| 230 | `CENSUS_REFINED` | band tiles that are 4h now (old census[4]) |
| 231 | `CENSUS_TRIGGERED` | tiles with a trigger bit (sum of TRIG bits) |
| 232, 233 | `REL_SPEED0 REL_SPEED1` | the release port's plan speed (umSupport[4T]): the largest \|US\| face an owner's reach reads, f32 bits (m/s) by `atomicMax`, inf where one is non-finite. `nrReach` reduces into copy `cfg.y` and zeroes the other; `nrSpread` takes its radius from it (umCertificateRadius). |
| 234–239, 240–245 | `WALL_REACH0`, `WALL_REACH1` | the released walls' reach (umPlaneReach, UM_WALL_REACH): for each face direction f = 2a + upper, the largest dt·away / h (cells, f32 bits by `atomicMax`) of a released wall face tile, 0 where none. `nrWallReach` reduces into copy `cfg.y` (6 words from 234 + 6·cfg.y) and zeroes the other; `lcAdvect` and `lAdvect` read it (levelset-release.wgsl.ts). |
| 246 | `RES_RADIUS` | the head's residency (§5.4): the largest region radius this census, in tiles (the old pageSeed's closure radius, `atomicMax`). Cleared by `censusBegin`. |
| 247, 248 | `RES_CLAIMED_F RES_CLAIMED_C` | the bricks and pages the head's residency request took (NONE → CLAIM) this census. Cleared by `censusBegin`. |
| 249–255 | reserved, E | |

### 5.2 Param ring (words 256–1279)

`RING = 4` slots of 256 words. **Step `n` uses slot `n mod 4`**, and the host writes that slot,
with one `queue.writeBuffer`, before it submits part 1 of step `n`. No other host write happens in
a step. The slot index reaches kernels as `cfg.x` (§6), never by reading `STEP`.

Slot layout (f32 unless marked):

| words | owner | content |
|---|---|---|
| 0 | E | step `n` (u32) |
| 1 | E | dt (s) |
| 2 | E | time at the step's start (s) |
| 3 | E | flags (u32): bit 0 solid edit pending (gates `solidStamp`), bit 1 bodies present |
| 4–6 | E | gravity (m/s²) |
| 7 | E | h (m) |
| 8–15 | F | volume control: target volume, the secant state's inputs, the rate limit |
| 16–63 | F | sources: drops (spheres, disks, letters) and the inflow plug |
| 64–191 | P | bodies: 8 records of 16 words (pose, velocity, swept bound) |
| 192–223 | E | solid edits: up to 7 dirty tile boxes (lo xyz, hi xyz as i32, then a count at 222) |
| 224–255 | reserved | |

A lane that needs more ring words asks E. Static per-scene records (motor tables, letter
outlines) are not in the ring: they are written once at build into the owning lane's own buffer.

### 5.3 Free stacks

`F_FREE[FCAP]`, `C_FREE[CCAP]`, `B_FREE[BCAP]`, `K_FREE[KCAP]`, each a LIFO of free slots with
its `TOP_*` word.
- At build: `TOP = CAP`, `FREE[i] = CAP - 1 - i`, so the first pops hand out slots 0, 1, 2, …
- A pop is `atomicSub(TOP, 1)`. A pop that underflows adds 1 back, latches `FAIL.POOL_*`, counts
  `REFUSED_*` and puts the brick's key in `FAIL_INFO`.
- Pushes happen only after every pop of the step (`setLeave`, after the claims), so no slot is
  reused in the step that freed it.
- Page-local slot order (design §2.1, 1.15–1.25x) is M1's measurement, not part of the ABI.

### 5.4 Marks, candidates and bounds

| region | words | content |
|---|---|---|
| `MARK_F` | `FCAP` | the head's keep bits for live F slots (`resRequest`, and tiered `censusDecide`'s band rings): bit 0 keep; consumed (exchanged to 0) by the head's `setLeave` |
| `MARK_C` | `CCAP` | the same for C slots |
| `CAND` | `CAND_CAP = FCAP` | enter candidates: brick keys (§4.1 word 29), `NB_KEY_PAGE` (bit 30) for a page key. Duplicates allowed; `setUpdate` dedupes by compare-exchange on the directory. Bit 31 (`KEY_CLAIMED`): the head's residency request already took the key's directory word NONE → CLAIM, so the claim pops without the exchange. Overflow latches `FAIL.LIST`. |
| `BOUND_F` | `FCAP` | the brick's velocity bound, f32 bits by `atomicMax` (non-negative f32 orders as u32): `project` (grid faces) and `pScatter` (particle speeds) raise it; `setMark` clears it. No residency reads it since the head's residency (2026-11-15). |
| `BOUND_C` | `CCAP` | the same for C slots (4h faces) |
| `ACT` | `FCAP` | the active class (§5.5, brick-abi.ts `ACT`): bits 0–7 this step's ACTIVE tiles, bit 8 `NOW` (F_ACT(t)), bit 9 `PREV` (F_ACT(t − 1)). `setMark` raises the tiles and, by `atomicOr`, `NOW` on every resident brick within 3 tiles of one; the head's `setLists` moves `NOW` to `PREV`; `setClaimBricks` zeroes it. |
| `AIR_GATE` | `16 * CCAP` | per C slot, 512 bits: bit `t` (the tile's C local) set where the far-air gate holds (umFarAirKept's plane 4: the 3³ about tile `t` meets the collar, the seeds dilated 4 tiles an axis over live tiles). Written by `lcClear` every step, read by the next step's `lcAdvect`: a C vertex with φ ≥ 16 h whose cell's tile has the bit clear keeps φ untraced. Zeroed at `setClaimPages`, like `WANT`. |
| `REACH0`, `REACH1` | `512 * CCAP` each | per C slot, a word a tile (the tile's C local): the tile's signed reach, umPackReach (5 bits an axis each way, 31 saturated), in tiles. `nrReach` (part 1, after the head, before move: the old frame plan's localSpeed) writes the local word into `REACH0` from the US faces its owners read, F's bricks for h tiles and the page for 4h tiles. `nrSpread`'s three passes (x: 0→1, y: 1→0, z: 0→1) take the separable maximum over ±radius tiles. An AIR tile saturates, the box contributes 0. The final word is `REACH1`, read by umDeepAirKept (`nbrDeepAirKept`). |
| `REACH_STEP` | `CCAP` | the ring step + 1 that the slot's reach words were made for (`nrReach`). A slot claimed after part 1 (setClaimPages in part 2) has no words for this step, and its tiles read saturated, as the spread's AIR. |
| `WALL_TILE` | `384 * CCAP` | per C slot, 64 words a face direction f (umFaceTile: u + 8v of the tile in the face's plane): a wall face tile's largest released dt·away / h (cells, f32 bits; 0 where none). Written by `nrWallReach` (part 2, before `stages.levelSet`: the old wallReach) for each of the page's wall faces, the box across. umReleasedMayChange reads it. The other faces' words are not written or read. |
| `PSUPPORT` | `8 * FCAP` | per F slot (`NB_S_PSUPPORT + 8·slot`), the particle support the head's residency reads (§5.4; the old DEPB refine prediction, uniform-particle-brick.wgsl.ts 33–38, 240–262): word 0 bit j = tile j of the brick (tx \| ty<<1 \| tz<<2) is in some live record's support (the 2³ tiles from floor((x − 2)/4)); words 1–3 and 4–6 the −xyz and +xyz travel maxima of those records (f32 bits, cells, `atomicMax`: ballistic \|v\|·dt/h all six ways, coupled max(∓before, 0)·dt/h + NB_SEPARATION_REACH); word 7 pad. F's `pJoin` accumulates it in part 2 for every live record of bank dst (an absent support brick latches `FAIL.REACH`); the head's `resSeed` reads and zeroes it; `setClaimBricks` zeroes a new slot's. |
| `CAND_MASK` | `8 * CCAP` | per C slot, censusDecide's candidate bricks: bit `e` of the 6³ bricks about the page (from 4·page − 1, `e = x + 6y + 36z`). The X8 slabs `atomicOr` a missing ring brick's bit, and the one that sets it lists the brick in `CAND`, so a page lists each once. Cleared by censusImportance (slab 0) each census. |

**The boot's reach rule** (`bootBricks`, the CPU, M1). The boot's set is every brick that overlaps
a **reach box** of a brick holding a liquid vertex (φ < 0) of the boot boxes. In cells, that box is

```
[origin + lo - R, origin + hi + R],   R = (bound + |g| dt) * dt / h + REACH_MARGIN
```

- `lo` and `hi` bound the brick's liquid vertices; at boot `bound` = 0.
- R is real-valued, and the box's ends are floored to bricks. `REACH_MARGIN` is 7 cells.
- With no travel this is the design's one-brick ring. The first step's head replaces the set with
  the residency rule's.

**The residency rule** (the head's, lane E, 2026-11-15; brick-residency.wgsl.ts). NB-FLIP's local
residency, ported verbatim on the brick storage: the old pageSeedOf (uniform-mixed-dynamic.ts
2097–2148) and lrAsk (uniform-local-residency.ts 140–158).

- *Timing.* Part 1's head, in both engines, after the census and before `stages.move`, so it reads
  the completed step's projected and extended velocity (US), as the old frame-head census does. Every
  head claim lands before `nrReach` (the release port's tile reach words).
- *Records* (`resSeed`, a workgroup a page of C_ALL, G1). Per 16-cell region (4³ tiles, 8 a page) and
  stage (RESIDENCY_STAGES: ownership, transport, pressure, surface, particles), two seed-mask words and
  the −xyz/+xyz radii in tiles (lrRadii), in the page's census scratch (`CENSUS.RES`):
  - a tile is considered when it is near (classify's near, `CENSUS.NEAR`), band (`WANT`) or a sample
    (`PSUPPORT`);
  - seeds: ownership = h now (tiered) or band; transport and pressure = wet or sample; surface =
    crossing or sample; particles = sample;
  - flow: sampledFlow (SAMPLE_REACH 2.5, shift 0) over the region grown by departureTravel
    (`H.CENSUS_REACH`) plus RESIDENCY_MARGIN_TILES (5) tiles; particle travel (`PSUPPORT`'s maxima
    over the region's bricks) joins stages ≥ 2; stage 0's radii are 0;
  - `H.RES_RADIUS` is the largest radius of the census.
- *Requests* (`resRequest`, a workgroup a page of C_ALL, lanes over each region's candidate box):
  a destination is kept when a stage's record meets it (lrStageMeets). The candidates come from each
  record's own box, so the work scales with the sources and their reach, never with the lattice.
  - A present destination gets its keep mark (`MARK_F`, `MARK_C`).
  - An absent one's directory word goes NONE → CLAIM, and the one winner appends its key with
    `KEY_CLAIMED`.
  - Pass 0 (cfg.z = 0) takes pages and the bricks of resident pages; a brick of an absent page takes
    its page. `setClaimPages` follows. Pass 1 (cfg.z = 1) takes the bricks of the pages pass 0 entered.
    `setClaimBricks` follows.
  - A destination already kept or claimed is skipped before the meets (rqSettled).
- *Leaving* follows the old rule: whatever no record reached retires at this head (`setLeave`,
  `setLeavePages`), with no hysteresis (idleSteps 1).
- *Deviations, stated in the kernel header.*
  - All-h has no old counterpart (the old engine has no all-h mode; every tile is h by
    construction there, and a verbatim seed would keep every brick): all-h ownership seeds are the
    band tiles (WANT) alone, and every stage is requested at F granularity.
  - Destinations are F bricks (8 cells) and C pages (32 cells), not 32³ blocks. The old blocks keep
    cells resident beyond the footprints, and old stages may read or propagate into them, while our
    owner readers return far or air there. If a dam gate row moves for that reason, revisit the
    granularity, not the gate.
  - No closure-cap or window failure (there is no window to outgrow). Only resident pages have
    census words, so only they seed; a sample's support in an absent brick is latched by F's hook.
- The checks stay: a particle that lands outside a resident brick (`pScatter`, `pJoin`'s support)
  and a liquid vertex whose 1-ring is not resident (`setMark`'s scan of the φ < 0 vertices on its
  brick's faces against the neighbour words, part 2 after the move) latch `FAIL.REACH` with the
  absent brick's key. They are defects to root-cause, never clamped.
- The step-14 tiered repro (2026-11-15, lane F: FAIL.REACH at brick (3,5,15); step 15 at (4,5,15),
  info 0xf01404) came from part 2's reach box reading a BOUND_F that missed step n's projected
  velocity. The head's residency reads the completed step's US, as the old census does.
- The brick range is `NB_BRICKS` (`floor(n/8) + 1` an axis, skirts included), in the prelude.

### 5.5 Lists and their indirect arguments

Lists live in `set` at `LIST[k]`, each `LIST_CAP[k]` words of slots, with the count at
`LIST_COUNT[k]` (word `32 + k`). `setLists` rebuilds every list each step, after `setUpdate`.
- **Order (M1):** atomic appends, so no order is promised, as with the arena's `ubaLists`
  (which appends too).
- Page-local or slot order is an M1 locality measurement, and no kernel may rely on it.
- A live slot is one whose record key (word 29) is not `NONE`. Leaving and never-taken slots hold
  `NONE` there.
- **M1 (superseded 2026-11-15):** `F_LIQ = F_ALL`.
- **(2026-11-15)** `setMark` writes the real `LIQUID` bit: the brick has an h-owned vertex with φ < 0, so it holds
  pressure rows; `F_LIQ` lists exactly those bricks (no longer `F_ALL`, since lane P's row list, 04298677: P reads
  no `F_LIQ` and no `LIQUID`). `project` runs over `F_ALL` (P's contract: `WARM` and the `WALLS` bits on every F
  brick).
- **`F_ACT` (2026-11-15, seep-techniques.md §1–6: work follows the active set, not the resident set).** The active
  class is a word an F slot, `ACT` in `set` (brick-abi.ts `ACT`):
  - bits 0–7 (`TILES`, bit j tile j): `setMark`'s ACTIVE tiles this step (part 2's head, after the rebin). A tile is
    ACTIVE when it is h-owned (tiered only: all-h every tile is h, so the criterion is dropped there), has a φ < 0
    vertex through the tier authority (F's vertices on an h tile, the C corners on a 4h tile: the trilinear
    interpolant's minimum over the tile is a corner's), holds a particle (`TILE_OCC` bits 0–7, this step's
    `pScan`), or is hot (the stage set's tile heat words, `BrickStages.activity`: F's `FT_H` or `FT_T` nonzero, or
    the census's `TRIG` bit, `lHeatNow`'s max);
  - bit 8 (`NOW`, F_ACT(t)): set by `setMark` on every resident brick within `ACT_DILATION = 3` tiles (12 cells,
    Chebyshev, at tile precision) of an ACTIVE tile, a lane an offset of the 5³ bricks about the source;
  - bit 9 (`PREV`, F_ACT(t − 1)): the head's `setLists` (cfg.z = 1) moves `NOW` down and clears the rest.
  `F_ACT` lists `NOW | PREV`. At the head it also lists the bricks entering this step (no class yet). The boot
  (`setLists` cfg.z = 2) makes every live brick `NOW`. `setClaimBricks` zeroes the word.
  The moved stages (lane F: the level set, the velocity and the extension, all but `lInit`, `pScan` and `pTiles`)
  dispatch over `F_ACT` (G1, G8) or `FA_*` (`F_ACT` then `C_ALL`). The gate is the per-kernel frozen comparison
  of lane E's progress log (2026-11-15): every word each moved kernel writes is bitwise the F_ALL kernel's, and the
  words F_ALL would write on an idle brick are never read.

| k | name | kind | entries |
|---|---|---|---|
| 0 | `F_ALL` | F | every live F slot (skirts included) |
| 1 | `F_LIQ` | F | `LIQUID`: the bricks with pressure rows at h |
| 2 | `F_SURF` | F | `SURFACE` |
| 3 | `F_PART` | F | `PARTICLES` |
| 4 | `F_SEAM` | F | `SEAM` |
| 5 | `F_SOLID` | F | `NEAR_SOLID` |
| 6 | `F_BODY` | F | `NEAR_BODY` |
| 7 | `F_COLD` | F | `COLD` |
| 8 | `F_ENTER` | F | entered this step (written by `setUpdate`, not `setLists`) |
| 9 | `F_LEAVE` | F | left this step (`setUpdate`) |
| 10 | `F_CHANGED` | F | `CHANGED` (`setMark`) |
| 11 | `C_ALL` | C | every live C slot |
| 12 | `C_LIQ` | C | C slots with a 4h-owned liquid tile |
| 13 | `C_SEAM` | C | C slots with F children |
| 14 | `C_ENTER` | C | (`setUpdate`) |
| 15 | `C_LEAVE` | C | (`setUpdate`) |
| 16 | `B_ALL` | B | live B slots (M3) |
| 17 | `F_ACT` | F | `ACT` `NOW | PREV` (and, at the head, entering bricks) |
| 18–23 | reserved, F | | |
| 24–29 | reserved, P | | |
| 30–31 | reserved, E | | |

A list over F and C at once (`phiAdvect`'s "the tier comes from the list entry") is launched over
`F_ALL` then `C_ALL`: entry `e < count(F_ALL)` is `F_ALL[e]`, else `C_ALL[e - count(F_ALL)]`. Its
arguments are the `FC_*` triples below.

**Indirect arguments** (the `args` buffer, `STORAGE | INDIRECT`, written only by `setLists` and
by the owning lane's gated slots). For list `k`, at word `12 * k`:

| words | name | workgroups |
|---|---|---|
| `12k + 0..2` | `G1` | one a list entry |
| `12k + 3..5` | `G8` | one per 8 entries |
| `12k + 6..8` | `G64` | one per 64 entries |
| `12k + 9..11` | `X8` | eight per entry (setCensus's C passes and setTransferList: a workgroup a 64-tile slab of a page; censusFine: a workgroup a brick's tile) |

Then `FC_G1`, `FC_G8` and `FC_G64` at words 384–392, `XFER_G1` (setTransfer's jobs) at 396, and `FA_G1`, `FA_G8`
and `FA_G64` at 400–408 (`F_ACT`'s bricks, then `C_ALL`'s pages: entry `e < count(F_ACT)` is `F_ACT[e]`). A stage or
solver may dispatch from any of words 0–408 (`ARGS_LIST_END`). Words 512 and up belong to the lanes: P from
512 (the gated solver slots), F from 2048.

**The fold rule.** A count `n` of workgroups is written as `x = min(n, 32768)`,
`y = ceil(n / 32768)`, `z = 1`, so no dimension passes the 65535 limit. A kernel's group index is
`g = wid.x + 32768 * wid.y`, and it returns when `g` passes its own count. Every list-indirect
kernel uses this rule, even when `y` is 1.

### 5.6 Particle bins

| region | words | content |
|---|---|---|
| `BIN_START` | `FCAP` | first record of the F slot's bin in the bank `pScatter` wrote |
| `BIN_COUNT` | `FCAP` | its records |
| `BIN_CURSOR` | `FCAP` | `pScatter`'s per-bin append cursor (scratch) |

**Banks.** `parts` holds two banks of `PCAP` records at NB-FLIP's stride: 12 floats (FLIP) or 28
(APIC), the `Particle` struct of `narrowBandParticleWGSL`, unchanged. At step `n` with parity
`q = n & 1`:
- the move reads bank `q`;
- the rebin (`pCount`, `pScan`, `pScatter`) writes bank `q ^ 1`, grouped by bin.

Bin starts are an exclusive scan over `F_ALL` as it stood when the rebin ran (the previous step's
list). Entering bricks have empty bins, and leaving bricks have empty bins by the keep rule, so
`setUpdate` never invalidates a bin. `PCOUNT[q ^ 1]` is the bank's total.

Overflow of `PCAP` latches `FAIL.PARTICLES` and counts `P_OVERFLOW`. Retirement and the cell cap
merge are F's, at `pScatter`. Lane F owns the compact-position, class and motion streams of the
move (design §2.4) as its own buffers.

**Bin order and seeds** (lane F, amended at M2):
- `pCount` zeroes `PCOUNT[q^1]`. `pScan` takes each brick's start by `atomicAdd` on it, after a
  workgroup scan of its 512 cell counts.
- So bins are contiguous per brick and cell-sorted inside it, but their order across bricks is
  not promised. This is not an exclusive scan in F_ALL order.
- `pSeed` appends after bank `q^1`'s binned records, as per-brick segments, and bumps
  `PCOUNT[q^1]`. Seeds are not in `BIN_COUNT` until the next rebin. Nothing reads `BIN_COUNT`
  between `pSeed` and the next part 1.


## 6. Kernels, bindings and the record

**Ownership (amended at M2, coordinator ruling).** `operator` and `project` are lane P's, in
`pressure-operator.wgsl.ts`. Their entry points (`nbOperator`, `project`), the zero-row rule over
F_ALL with its `FAIL.LIST` check (§6.1) and the `UNSUPPORTED` cut check are kept. The old
solve's wall/held/closed active set and its seam heads are ported verbatim, as an outer loop
around the MGPCG. Its per-face state takes FLAGS bits 7–12 (§2.4).

**Bindings.** Group 0 only. A kernel declares a list of bindings by name, in order, and E's
`record()` builds its bind groups once at build:
- `cfg`: a 16-byte uniform `vec4u`, baked per dispatch (seep's pattern). `cfg.x` is the ring slot,
  `cfg.y` the parity, `cfg.z` and `cfg.w` the kernel's own.
- storage: `set`, `dir`, `args`, `vel`, `phi`, `pres`, `cut`, `parts`, `xfer`, and lane-owned buffers by
  name.

At most 8 storage buffers a kernel. A kernel that needs more splits.

**WGSL names (amended at M1).** WGSL reserves some words the ABI uses:
- `set`: the host's buffer `set` is declared `nbSet` in WGSL. `declareBindings()` maps it
  (`WGSL_NAMES` in `brick-kernel.ts`), and a module passes `{ set: { name: "nbSet" } }` to
  `brickAbiWGSL`.
- `operator`: a kernel's `entry` may differ from its `label`. P's `operator` kernel has the entry
  point `nbOperator`.
- `active` is reserved too, so a lane flag needs another name (E uses `live`).

Tint's errors are the ground truth. Every kernel compiles in M1's GPU test.

**Kernel source.** A lane exports `{ label, entry, code, bindings }` objects, as
`seep-kernels.ts`'s `KernelSource`, where `code` starts with `brickAbiWGSL(...)` (the constants
and accessors of `brick-abi.ts`). E's `record()` places them in the schedule.

**Recorded variants.** The step is recorded 4 times, once a ring slot (the parity is
`slot & 1`). Every data-dependent count is an indirect argument, so a recorded step never changes.
Encode-ahead: after submitting step `n`, the host encodes step `n + 1`'s command buffers.

**The schedule** (design §2.11). Each line is one dispatch; `[gated]` means its count is an
indirect argument that can be zero.

| part | stage | kernels (owner) |
|---|---|---|
| 1 | head | `bodyStamp` (P, bodies only), `solidStamp` (E, gated by ring flag bit 0); the census, the residency (`resSeed`, `resRequest` ×2, the claims, `setLeave`, `setLeavePages`, records, init, lists; §5.4) and, tiered, `censusWant` and the transfer (E) |
| 1 | move | `pTrace`, `pBand`, `pSolid` (F) |
| 1 | rebin | `pCount`, `pScan`, `pScatter` (F) |
| 2 | brick set | `setMark` (class words, the 1-ring check), `setMarkPages` (tiered, `LIQUID_TILES`), `setLists` (E) |
| 2 | level set | `phiAdvect`, `phiSolid`, `phiUnion`, `pSeed`, `redistInit`, `redistSweep` ×2 (F) |
| 2 | velocity | `p2g`, `uAdvect`, `uExtendPre` (F) |
| 2 | pressure setup | `operator`, `coarsen`, `coarsenTop`, `pcgGuess` (P) |
| 2 | pressure slots | 8 × (`update`, `check`, `down` ×4, `coarse`, `up` ×4, `precond`, `direction`) [gated] (P) |
| 2 | projection | `project` (P) |
| 2 | tail | `uExtend` ×2, `uDelta` (F), `couple` and the rigid integrator (P), `stepEnd` (E) |

`setUpdate` is several entry points of one module when it needs a grid-wide barrier between its
phases (claim, then records, then frees). Each is one dispatch, and the table above counts them as
`setUpdate`.

**Two compile rules.** No kernel inlines the solid walk (`phiSolid` and `pSolid` carry it). Every
module compiles cold in under 30 s (M1's test).

### 6.1 The pressure slot contract (lane P writes against it)

The schedule's pressure stage is `operator`, then the solver, then `project`. `operator` and
`project` are E's in M1, and P takes them over. The solver is P's. This contract is what sits
between them. Code: `lib/methods/narrow-band/brick-pressure-slot.ts` (`PressureSolver`).

**Units.** `X = p dt / ρ` in m²/s, where p is the physical pressure (`p = ρ X / dt`). RHS is in
m²/s. DIAG and K are dimensionless. h is ring word `H` (m).

**Liquid cells.** A cell is liquid when its centre φ < −θ_min, with φ in h and θ_min = 1e-3
(`uniformMixedPressureLiquidWGSL`, verbatim). In M1 the centre φ is the mean of the cell's 8
vertex φ. If F later supplies a centre channel, E changes that one helper.

**Inputs.** `operator` writes them on every cell of every live (`F_ALL`) brick, before the solver's
first kernel:
- `operator` is a pure function of the channels. It reads `PHI` and `FLAGS` (OPEN, box rule),
  `US0..2`, and the cut's `FRAC` and `AP*` (an aperture of 1 where there is no cut). It writes
  `RHS`, `DIAG`, `K0..2` and `X` (below).
- Its signature is `(cfg, set, dir, vel, phi, pres, cut)`, and it does not change when P extends
  it to C and B rows and to the seam.
- `A_a(c)` is the aperture of c's lower face along a. A face is closed (A = 0) when the box rule
  or the cut closes it.
- `K_a(c) ≥ 0` is the coupling across c's lower face along a, so it sits at the **upper** cell of
  the pair. It is `A_a(c)` when c and `c − e_a` are both liquid, and 0 otherwise. A row's coupling
  to its upper neighbour along a is therefore `K_a(c + e_a)`, read from that neighbour's brick
  when it lies in another one.
- `DIAG(c)` sums over c's six faces: `A` to a liquid neighbour, `A / θ` to an air neighbour, and
  0 for a closed face. θ is the ghost-fluid crossing `uniformMixedSurfaceThetaWGSL`, clamped to
  [θ_min, 1/θ_min], verbatim.
- `RHS(c) = −h (Σ_a (A_a(c + e_a) US_a(c + e_a) − A_a(c) US_a(c)) − rate · h · capacity)`. Without
  the rate term it is −h² times the discrete divergence of u*. (Amended at M2, P 6b06ae15.)
  - `rate` is NB-FLIP's volume control (Kim et al. 2007, the proportional term), in 1/s, as f32
    bits in set header word 64 (F's `FH.RATE`). F's `uAdvect` writes it before the operator.
  - `capacity` is the cell's open fraction, 1 without solids.
  - This is uniform-mixed-pressure-velocity.ts l.360. Rate 0 is the plain divergence.
- On non-liquid cells, DIAG, K and RHS are 0 and X is 0.
- (Amended at M2.) `operator` runs over F_ALL, not F_LIQ, so this holds in every live brick. A
  live brick outside F_LIQ holds zero rows, and a solver that reads a neighbour's `K` or `X`
  through a neighbour word never sees a stale value. A liquid cell in a brick outside F_LIQ would
  have no row in the solve, so `operator` latches `FAIL.LIST` (info: the brick's key).

**Sign convention.** The system is `DIAG(c) X(c) − Σ_nbr K X(nbr) = RHS(c)` over the liquid cells,
with X = 0 at air (the Dirichlet ghost):
- The matrix is symmetric, with a positive diagonal and off-diagonals of `−K`.
- It is positive definite when every connected liquid component has a face to air.
- A sealed component is singular. P handles that case; M1's scenes have a free surface.

**Neighbours across bricks.** These go through the record's neighbour words:
- An `F` slot is read directly.
- `NONE` is air. The head's residency keeps the stages' footprints around liquid (§5.4), so a liquid
  cell never borders `NONE`, and if one does, that is `FAIL.REACH`.
- `SEAM` and `BULK` neighbours, and a C or B row, raise `FAIL.UNSUPPORTED` in M1 (there are no
  tiers). P extends the contract here.

**Output.** X on the liquid cells of `F_LIQ` bricks, converged to P's tolerance. The solver
writes nothing outside `pres` and its own buffers, apart from the gates below.

**Warm start.** X as `operator` leaves it, which is:
- the last step's solution in bricks that stayed live;
- 0 in entering bricks (`setUpdate` zeroes their channels);
- 0 in entering C slots: `setInit` zeroes every `pres` channel of C_ENTER, X included (d769d9d0);
- 0 in this step's non-liquid cells.

At a tier change (M2), `setTransfer` carries no `pres` channel (NB-FLIP's remap carries φ, U and the
extension velocity only):
- A tile that coarsens starts from its C cell's X. That X is either the solution from the last step
  the tile was 4h-owned in a resident page, or 0 if the page entered since.
- A tile that refines starts from its F cells' X. That X is either the solution from the last step
  the cells were h-owned, or 0 if the brick entered since.
- In both cases the value is any valid start; whether P prolongs or restricts X at the change is P's
  choice.

Any starting value is valid. The warm start affects only the iteration count.

**`project`** reads X and the same θ, recomputed from φ:
- A lower face with a liquid side gets `U = US − (X(c) − X(c − e_a)) / h`, with the air side's X
  taken as the ghost `X_liq (1 − 1/θ)`.
- A closed face gets `U = 0`.
- A face with no liquid side gets `U = US`, which F's extension then overwrites.

**Recording.** A `PressureSolver` declares:
- `before` (optional, 2026-11-15, lane P's request): kernels placed once, just before `operator` (a pass the operator
  reads, such as a tile pass once a step), in the step and in the boot's kick;
- `setup`: kernels placed once, after `operator`;
- `slots`: the number of encoded slots, which `record()` allots. P's MGPCG declares 8 (§6's
  schedule). The bring-up solver below declares its own count.
- `slot`: the kernel list of one slot, placed `slots` times with `cfg.z` = the slot index;
- `finish`: kernels placed once, before `project`.

Each placement is either direct (a fixed workgroup count) or indirect from an `args` word.

**Gates.**
- Args words 512–2047 are P's, laid out as P chooses. Gated dispatches read their counts there.
- `setup` opens slot 0 and closes every later slot, by writing 0 workgroups, every step. A gate
  left from the last step is never trusted.
- Slot k ends with a direct, one-workgroup `check`. It opens slot k + 1 (copying the list's
  arguments from words 0–383) or leaves it closed.
- A closed slot's `check` sees that it was closed (a P header word) and writes nothing.
- Header words 128–191 are P's: residuals, PCG scalars, cycle counts.

**State.**
- `setup` clears `STATE.PRESSURE_CAP`.
- The last open slot's `check` sets it when the residual is above tolerance after the final
  slot.
- P also keeps a cumulative count of capped steps in its header, because the host reads receipts
  lagged.
- A solver failure (for example `PIVOT`) latches `FAIL` through `nbFail`.

**The bring-up solver (M1 testing only).** `brick-bringup-cg.ts` is a Jacobi-preconditioned CG,
one iteration per slot, written against this contract, so that M1's tests have a solve:
- It is labelled as such in its code and in `progress-lane-e.md`.
- No M2 gate and no timing claim runs on it.
- P's MGPCG replaces it at the same slot before any scene is gated.

### 6.2 The fluid stage contract and the registration files (amended at M2)

Lane F writes its kernels against `BrickStages` (`brick-stages.ts`), as P writes against §6.1:
- `move` and `rebin` sit in part 1, after the head (the census, and when tiered the transfer).
- `tiered` (default false) says the stages read through the owners and write only owned storage.
  It turns on the transfer and the §2.5 residency.
- `levelSet` and `velocity` sit in part 2, after the brick set and before `operator`.
  `velocity` writes `US`.
- `tail` sits after `project` and before `stepEnd`.
- `boot` runs once, after E's boot phi and lists.

Each entry is a `BrickPlacement` (`brick-kernel.ts`). The rules:
- A gated dispatch reads list arguments (words 0–392) or F's region (2048–4095).
- F's buffers are named in `buffers`, in words, and allocated zeroed. Their names may not shadow
  an engine buffer.
- `validateBrickStages` checks both rules at build.
- F owns header words 64–127 and class bits 16–23.

A stage factory receives a `BrickStageContext`:
- the layout;
- dt, h and gravity, as the ring carries them;
- the method's resolved values;
- the fluid's constants (density, dynamic viscosity, surface tension), which F's forces apply,
  or whose scene F refuses.

**Registration files.** Each lane owns one file, and E imports from it. Its export stays
`undefined` until the lane lands:
- `brick-fluid-stages.ts` exports `brickFluidStages` (F);
- `brick-pressure-solver.ts` exports `brickPressureSolver` (P).

The method (`brick-method.ts`) refuses a scene and names every gap: the scene's own gaps
(`brick-scene.ts`), each unregistered lane, and a registered bring-up solver. A bring-up stage set
(`bringUp: true`, `bringUpStages`, `uGravity`) is for E's tests only.

**The host adapter.** `BrickNBFlipSolver` (`brick-solver.ts`) is the app's `GPUSolverInstance`:
- `advanceTo` runs whole steps of dt, with no substeps.
- The interim presentation (`brick-present.ts`, ruling 7.1(3)) copies the pool into dense
  `vertexPhi`, `volume` and `openFraction` textures:
  - It runs as its own submission after the steps, never inside one.
  - It dispatches one workgroup a pool slot, and clears the bricks that left.
  - `setPresentationEnabled(false)` turns it off for benches.
  - `info.brickEngine.presentations` and `lastPresentationMs` report it.
- Observers read `brickEngine` after a fence. `tools/nbflip-observe-brick.ts` is the box
  observer, and `createNBFlipBoxObserver` routes a brick solver to it.

## 7. The brick-set update in words

1. The head's residency (§5.4; part 1, since 2026-11-15; `setMark` in part 2 keeps only the class
   words and the 1-ring check):
   - `resSeed` writes the records from the census words, `WANT` and `PSUPPORT`;
   - `resRequest` sets `MARK_*` keep bits on live slots its records meet and claims the absent ones
     into `CAND` (`KEY_CLAIMED`); tiered, `censusDecide` keeps and lists its band rings.
2. `setUpdate`:
   - claims each candidate: page word `NONE → CLAIM → cslot` (C pop), then child `NONE → fslot`
     (F pop), each by compare-exchange, so duplicates enter once;
   - lists entering slots (`F_ENTER`, `C_ENTER`) with `ENTERING`, and live slots without a keep
     bit (`F_LEAVE`, `C_LEAVE`) with `LEAVING`. A page leaves only when it has no F children left
     and no keep bit;
   - rebuilds the records of entering bricks and of the live neighbours of entering and leaving
     ones;
   - clears the leaving bricks' directory words and pushes their slots;
   - initialises entering bricks' channels to air (φ = `PHI_AIR`, velocity 0, FLAGS from the box
     rule and `solidStamp`).
3. `setTransfer` (M2): tier changes with NB-FLIP's conservative remap, verbatim.
4. `setLists`: class bits, lists and indirect arguments.

## 8. Failure and state bits

`FAIL` (word 0) latches; the host reads it lagged (receipts ring) and raises the solver error.

| bit | name | meaning |
|---|---|---|
| 0 | `POOL_F` | F free stack exhausted |
| 1 | `POOL_C` | C free stack exhausted |
| 2 | `POOL_B` | B free stack exhausted |
| 3 | `POOL_K` | cut free stack exhausted |
| 4 | `PARTICLES` | particle bank capacity exceeded |
| 5 | `REACH` | a particle (`pScatter`), a particle's support (`pJoin`), or a liquid vertex's 1-ring (`setMark`), outside the resident bricks |
| 6 | `LIST` | a list or the candidate buffer overflowed |
| 7 | `WINDOW` | a key outside the window |
| 8 | `UNSUPPORTED` | a kernel met a feature it does not implement |
| 9 | `PIVOT` | coarsest Gauss–Jordan pivot not positive (P) |
| 10 | `SEGMENTS` | backtrace segment cap (F) |
| 11 | `TIER` | a tier change needs storage no resident brick or page holds (§2.5) |
| 12–15 | reserved, E | |
| 16–23 | reserved, F | |
| 24–31 | reserved, P | |

`STATE` (word 3), reported, not fatal: bit 0 `PRESSURE_CAP` (the step ended above tolerance at the
slot cap, P1 item 13).

`FAIL_POOL` in the brief is the family `POOL_F | POOL_C | POOL_B | POOL_K`.

## 9. Scene authoring

`scene.fluid.brickPools`: `{ f, c, b, k, particles, t? }`, the capacities of §2.1, `PCAP`, and setTransfer's listed tiles a step (`t`, default 2 f). They are
required for `engine: "brick"`. A scene without them fails explicitly at build, naming the field.
M0(b) measures each gate scene's peaks and authors them with a margin (its own section of
`progress-lane-e.md`).

## 10. What M1 implements

M1 implements:
- the pools for `vel`, `phi` and `pres`, and `dir` and `set`, the args, ring and record;
- `setMark` with the liquid, particle and reach criteria;
- `setUpdate` and `setLists`;
- `operator` with P's solver, then `project`;
- `stepEnd`.

Everything else in the schedule is an empty slot until its lane lands it. A scene that needs a
kernel not yet landed fails explicitly (§9 and design §2.9), and the method runs it on the uniform
engine.
