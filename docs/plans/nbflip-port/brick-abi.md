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
| `phi` | 2 | `FLAGS` | cell, u32 | `setLists` (OPEN, OWNER_H), `phiUnion` / `redistInit` (LIQUID, BAND), `p2g` (VALID_*) | everyone |
| `pres` | 0–9 | `X R Z D Q RHS DIAG K0 K1 K2` | cell; `K_a` the coupling across the lower face along `a` | `operator`, solver | P; `project` |
| `cut` | 0 | `FRAC` | cell | `solidStamp` | `operator`, F's solid paths |
| `cut` | 1–3 | `AP0 AP1 AP2` | lower face | `solidStamp` | as above |

- **The `pres` list is lane P's.** P may replace it (more vectors, MG scratch), as long as it stays
  a pool indexed by F and C slots at stride 520. P announces the final list to E before M1 lands.
- **MG levels above 4h are P's own.** P allocates them sized from `FCAP`, `CCAP` and `BCAP`, keyed
  by the engine's slots or keys, and rebuilt per step from the lists. They are not channels here.
- **A brick with no cut slot** (record word 30 = `NONE`) is fully open: `FRAC` = 1 and every
  aperture is 1, except where the box rule closes a face. It holds no cut channels.
- **A C cell has no cut.** "4h owners never reach the helpers": promotion keeps liquid one tile
  from cut cells, as today.

### 2.4 FLAGS bits (cell, u32)

| bit | name | meaning |
|---|---|---|
| 0 | `OPEN` | not static solid (box rule and the cut's `FRAC > 0`) |
| 1 | `LIQUID` | cell-centre φ < 0 |
| 2 | `BAND` | within the particle band R of the surface |
| 3–5 | `VALID0..2` | the lower face along `a` has a known velocity (extension's seed) |
| 6 | `OWNER_H` | the cell's tile is h-owned in this brick (the record mask's bit, per cell) |
| 7–15 | reserved, E | |
| 16–31 | reserved, F | |

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
| 0 | `LIQUID` | `setLists` | some cell has φ < 0 |
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
| 0–26 | neighbour pages: a C slot, `0xA0000000 \| bslot`, or `NONE` |
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
| 32–63 | `LIST_COUNT[k]` | list `k`'s entry count (§5.5) |
| 64–127 | F's | lane F's counters and receipts |
| 128–191 | P's | lane P's counters, residuals, cycle records, PCG scalars |
| 192–223 | `PEAK_*` | running maxima by `atomicMax`: 192 live F, 193 live C, 194 live B, 195 live K, 196 particles, 197 enter candidates |
| 224–255 | reserved, E | |

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
- Pushes happen only at the end of `setUpdate`, after the records are rebuilt, so no slot is
  reused in the step that freed it.
- Page-local slot order (design §2.1, 1.15–1.25x) is M1's measurement, not part of the ABI.

### 5.4 Marks, candidates and bounds

| region | words | content |
|---|---|---|
| `MARK_F` | `FCAP` | `setMark`'s keep bits for live F slots: bit 0 keep, bit 1 h-owned wanted (M2); cleared by `setUpdate` |
| `MARK_C` | `CCAP` | the same for C slots |
| `CAND` | `CAND_CAP = FCAP` | enter candidates: brick keys (§4.1 word 29). Duplicates allowed; `setUpdate` dedupes by compare-exchange on the directory. Overflow latches `FAIL.LIST`. |
| `BOUND_F` | `FCAP` | the brick's velocity bound, f32 bits by `atomicMax` (non-negative f32 orders as u32): `project` (grid faces) and `pScatter` (particle speeds) raise it; `setMark` reads it, then clears it |
| `BOUND_C` | `CCAP` | the same for C slots (4h faces) |

**The reach rule** (`setMark`). A brick or page within `reach` bricks of a liquid cell or of a
non-empty bin is kept or entered, where

```
reach = ceil((bound + |g| dt) * dt / (8 h)) + 1        (bricks; bound from BOUND_F)
```

- The `|g| dt` term covers a step of gravitational acceleration.
- The `+1` is the stencil ring (the design's `ceil(bound·dt / 8h) + 1`).
- A particle that still lands outside a resident brick latches `FAIL.REACH`. It is a defect to
  root-cause, never clamped.

### 5.5 Lists and their indirect arguments

Lists live in `set` at `LIST[k]`, each `LIST_CAP[k]` words of slots in slot order (ascending, as
the arena's `ubaLists`), with the count at `LIST_COUNT[k]` (word `32 + k`). `setLists` rebuilds
every list each step, after `setUpdate`.

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
| 17–23 | reserved, F | | |
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
| `12k + 9..11` | reserved | |

Then `FC_G1`, `FC_G8` and `FC_G64` at words 384–392. Words 512 and up belong to the lanes: P from
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

## 6. Kernels, bindings and the record

**Bindings.** Group 0 only. A kernel declares a list of bindings by name, in order, and E's
`record()` builds its bind groups once at build:
- `cfg`: a 16-byte uniform `vec4u`, baked per dispatch (seep's pattern). `cfg.x` is the ring slot,
  `cfg.y` the parity, `cfg.z` and `cfg.w` the kernel's own.
- storage: `set`, `dir`, `args`, `vel`, `phi`, `pres`, `cut`, `parts`, and lane-owned buffers by
  name.

At most 8 storage buffers a kernel. A kernel that needs more splits.

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
| 1 | head | `bodyStamp` (P, bodies only), `solidStamp` (E, gated by ring flag bit 0) |
| 1 | move | `pTrace`, `pBand`, `pSolid` (F) |
| 1 | rebin | `pCount`, `pScan`, `pScatter` (F) |
| 2 | brick set | `setMark`, `setUpdate`, `setTransfer` [gated], `setLists` (E) |
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

## 7. The brick-set update in words

1. `setMark` (one lane per tile of `F_ALL`, then `C_ALL`; 64 lanes):
   - reads FLAGS, φ, `BIN_COUNT` and `BOUND_*`;
   - sets `MARK_*` keep bits on live slots in reach of liquid or particles;
   - appends the keys of non-resident bricks in reach to `CAND`;
   - M2 adds the tier criteria and sets `CHANGED`.
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
| 5 | `REACH` | a particle or liquid cell outside the resident bricks |
| 6 | `LIST` | a list or the candidate buffer overflowed |
| 7 | `WINDOW` | a key outside the window |
| 8 | `UNSUPPORTED` | a kernel met a feature it does not implement |
| 9 | `PIVOT` | coarsest Gauss–Jordan pivot not positive (P) |
| 10 | `SEGMENTS` | backtrace segment cap (F) |
| 11–15 | reserved, E | |
| 16–23 | reserved, F | |
| 24–31 | reserved, P | |

`STATE` (word 3), reported, not fatal: bit 0 `PRESSURE_CAP` (the step ended above tolerance at the
slot cap, P1 item 13).

`FAIL_POOL` in the brief is the family `POOL_F | POOL_C | POOL_B | POOL_K`.

## 9. Scene authoring

`scene.fluid.brickPools`: `{ f, c, b, k, particles }`, the capacities of §2.1 and `PCAP`. They are
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
