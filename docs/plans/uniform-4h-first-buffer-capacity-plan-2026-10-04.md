# Order step 3: owner-indexed buffers from fine capacity

Source: `docs/plans/uniform-4h-first-storage-design-2026-10-03.md` (Order step 3), handoff `uniform-4h-first-implementation-handoff-2026-10-03.md`.
Read-only analysis, 2026-10-03/04. No GPU run. Byte counts are a CPU mock-device allocation census
(`createCm12Figure7_256` rescaled to n = 64/128/256, `detailPolicy:"requested"`, no regions, `DETAIL=packed`, solids compiled).
Line numbers are a snapshot: other agents are editing the tree. Each is paired with a symbol.

Steps 0-3, 4a and 5 are implemented (2026-10-04). Section 6 records what was built and where this analysis was wrong;
where a line above section 6 disagrees with it, section 6 is the code. Corrected lines carry `[6.n]`.

Paths are under `lib/methods/uniform/` unless given. Abbreviations: FRAME = `uniform-mixed-frame.ts`, REF = `webgpu-uniform-reference.ts`,
OWN = `uniform-mixed-ownership.ts`, ARENA = `uniform-scratch-arena.ts`, BAND = `uniform-pressure-band.ts`, BUILD = `uniform-mixed-layout-builder.ts`,
DET = `uniform-detail-fields.ts`.

## Notation

| Symbol | Meaning | 64³ | 128³ | 256³ |
|---|---|---|---|---|
| N | h cells n³ | 262,144 | 2,097,152 | 16,777,216 |
| T | tiles N/64 | 4,096 | 32,768 | 262,144 |
| F | live fine tiles (GPU count `umCounts.x`) | | | |
| L | live owners `umCounts.x*64u+umCounts.y` = 63F+T | | | |
| C | fine-tile capacity (host-chosen, always host-known) | | | |
| O | owner capacity = 63C+T (≤ 64C+T; exact because F fine tiles leave T−F coarse) | | | |
| c4 | all-4h root words T+6(n/4)² | 5,632 | 38,912 | 286,720 |
| cS | all-fine root words N+6n² | 286,720 | 2,195,456 | 17,170,432 |
| V | h vertices (n+1)³ | | | |
| S | hanging slots | | | |
| Sp, E | detail patch slots (`DET capacity`), patch edge (32) ; R = Sp·E³ resident h cells, Rv = Sp·(E+1)³ | | | |
| CAP | band slots, today `capacityOf(T)=min(T,max(4096,⌈T/2⌉))` | 4,096 | 16,384 | 131,072 |

Totals today (zero detail, packed fields): 54.1 MB / 317.9 MB / 2,504.8 MB.

## Findings that shape the plan

1. Owner indices are already compact: fine first (`packUniformMixedLayout`, `uniform-mixed-layout.ts:193-199`; GPU `BUILD:299,302`). Only capacities, bind sizes and baked bases are all-fine.
2. All ownership metadata is T-scaled (topology 16T, support ≈36T B, speeds 8T, builder, census, remap worklist, transport live set 124T, cleanup, geometry, momentum deferred, surface claims). No change needed.
3. No owner-indexed or arena buffer persists across frames or a relayout. The remap rewrites textures and the 3n² wall planes only (`uniform-mixed-remap.ts`). So growth is reallocate + rebind, with no content copy, except two tables (section 3).
4. Four arena consumers are NOT owner-indexed, they are h-lattice or tile keyed: transport `edges` (640 words per tile), extension slots (192 per tile), sharpening raw/cache (4 per h cell), surface-volume scale and the surface deferred list (per h vertex). These need re-keying, not resizing.
5. The arena is `max` over overlapped consumers plus a dead prefix. It only shrinks when every large consumer shrinks (section 2b).
6. C is always known on the host (it is the host's allocation). Only F is GPU-only under Dynamic. So capacity reaches shaders as host-written uniform words; nothing needs C baked.
7. There is no GPU check of owner count against any capacity today, and a latched relayout fatal does not stop the same frame (`adoptGpu` copies are ungated, `uniform-mixed-remap.ts:134-135`).
8. Dynamic's initial host layout is all-h (`REF:1451 createUniformMixedLayout(lattice,regions,1,forced)`), and `detail.reserveAll()` gives every patch a slot under a GPU relayout (`FRAME:682-691`, `DET:353`). Both force C = T and Sp = all today.
9. Default detail placement is identity (`DET:154 DEFAULT_OPTIONS`); packed arrives only through `setUniformDetailStorageForQA` / `DETAIL=`. Patch-keyed scratch (section 2b) only saves bytes under packed.
10. Latent overflow (code read only): surface volume reserves ⌈N/64⌉ = T partial rows but runs `svJobs()=umCounts.x+umResidentPageCount()` jobs with no guard (`uniform-mixed-surface-volume.ts:116,132,403-407`). All-fine with every page resident writes past the chunk sums into the Newton state. Fix in step 4e (rows = C + pages, guard on index). [6.6: real, fixed outside this plan as rows = T + pages]

---

## 1. Inventory

### 1a. Buffers that are 64T / n³ / (n+1)³ scaled

| # | Label | Allocation (symbol) | Size as written | 64³ | 128³ | 256³ | True scaling | Shader index (accessor, site) |
|---|---|---|---|---|---|---|---|---|
| 1 | Uniform shared stage scratch (arena) | `ARENA:66,86` ← `REF:845` | `max(pressureWords*4, donorOffset+N*16)`; pressureWords = 12·⌈(n+2)³/4⌉·4 + 17·Σ_{k≥1}(n/2ᵏ+2)³ | 16,956,096 | 128,162,016 | 996,430,592 | max of consumers, see 1b | see 1b |
| 2 | Uniform pressure band rows | `BAND:167` | `4*rowFields*CAP*64`, rowFields 21 (15 without solids) | 22,020,096 | 88,080,384 | 704,643,072 | band tiles ≤ F | field-major `rows[f*N+c]`, N=CAP*64, c=`(slot-1u)*64u+bRow(l)` (`BAND:221,314-319,460-466`) |
| 3 | Uniform mixed velocity tap cache (hanging) | `OWN:208-210` | `(2T+173*S)*4`, S=`uniformMixedHangingSlotCapacity(T)`=T | 2,867,200 | 22,937,600 | 183,500,800 | seam tiles ≤ min(T,27F) | `2u*UM_TILES+slot*173u+…`; tables [0,T) tile→slot, [T,2T) slot→tile (`uniform-mixed-velocity-sampling.wgsl.ts:11-22`) |
| 4 | Uniform reference compatibility scratch (conditioning) | `REF:1001,1025`; `ARENA:23-29` | `max(N*4, ⌈(sharpenBaseWords+8+T)/4⌉*16) + balance + pages` | 1,081,480 | 9,044,868 | 72,358,020 | only live use: root phi 4·L | `phi[o.index]` sim owner (`uniform-mixed-pressure-authority.ts:240`, `BAND:292,304,422`), then pressure owner over [0,T) |
| 5 | Uniform mixed sharpening tile list | `FRAME:314`; `uniform-mixed-sharpening.ts:64 workBytes` | `4*(8+3T+64T)` | 1,097,760 | 8,781,856 | 70,254,624 | active regular owners ≤ L | list entries `o.tile*64u+o.lane` |
| 6 | Uniform pressure band aggregates | `BAND:168` | `4*(8*9+23)*CAP` | 1,556,480 | 6,225,920 | 49,807,360 | band tiles | `bM(k,c)=k*M+c`, `bC(k,s)=9u*M+k*CAP+s` (`BAND:374-375`) |
| 7 | Uniform presented pressure phi and stage grids | `FRAME:344-348` | `4*(T + T+64*CAP + 8+7T+⌈T/32⌉)` | 1,196,576 | 5,378,080 | 43,024,416 | 64·CAP section only | `presented[2T+s*64u+lane]` (`BAND:851`); overlay `umStageBase()` from `arrayLength` (`uniform-stage-grids.ts:126-148`) |
| 8 | Uniform pressure band iterate | `BAND:169` | `(CAP*64 + 6n²)*4` | 1,146,880 | 4,587,520 | 35,127,296 | band tiles + fixed wall halo | `solve[c]`, halo at `N+face` (`bHalo`, `BAND:231-237`) |
| 9 | Uniform pressure band tiles (index) | `BAND:166` | `(26+CAP+T)*4` | 32,872 | 196,712 | 1,572,968 | T + band | `bIndex(LIST+s)`, `bIndex(SLOTS+tile)`, `SLOTS=26+CAP` |
| 10 | Uniform mixed solid displacement deposits (lazy, not in census) | `uniform-mixed-solid-displacement.ts:88` | `4*T*64` | 1,048,576 | 8,388,608 | 67,108,864 | L | `deposits[o.index]`, `o=umAllOwner(gid)` |

Not capacity-dependent, listed so nobody re-audits them: presented pressure 4·c4 (`FRAME:348`); 4h sampling cache (n/4+2)³·16; builder/census/remap/transport live set; boundary velocities 3n²·4 ×4; CM11a levels ≥ 2 (in the arena, 22.7 MB at 256³); extension hierarchy textures (n/4 and coarser).
T-scaled but dead in Geometric (optional cleanup, not step 3): `activeScratch` summaries 48 B/tile (12.6 of 15.8 MB at 256³, `REF:974-988`, `activeRegionEnabled=false`); conditioning balance + page tails (5.2 MB at 256³).

### 1b. Arena consumers (all overlap from offset 0; correctness is encode order)

| Consumer | Bind (symbol) | Extent today | 256³ | Sub-range: address — domain | Baked bases |
|---|---|---|---|---|---|
| CM11a L0 + L1 fields | placeholders, `webgpu-uniform-pressure-multigrid.ts:439-466` | 48·(n+2)³ + 68·(n/2+2)³ | 973.7 MB | dead as fields in the continuation (`MIXED_CONTINUATION_LEVEL=2`); only its extent is borrowed | offsets are host table + uniform metadata (`mg.fieldDims[].w`) |
| CM11a L≥2 fields | binding 35 whole buffer | Σ_{k≥2} 68·(n/2ᵏ+2)³ | 22.7 MB | haloed level cell × 17 words — fixed in n | cycles bake `MG_P…MG_P1` (`uniform-mixed-pressure-cycles.ts:126-127`) |
| Pressure root | `uniform-mixed-pressure-memory.ts:13-23` ← `FRAME:285` (called with the SIM layout) | 8 × ⌈4·cS⌉₂₅₆ | 549.5 MB | all-4h owner (= tile key) + wall slot; live words c4 | cycles bake `UM_P,UM_B0,UM_B1,UM_M0,UM_M1,UM_BACKUP,UM_RES` from range spacing (`…-cycles.ts:125`) |
| Pressure authority scratch (= `root.frozen`) | `FRAME:323,353`; `…-authority.ts:64-68,88` | sim `8*(1+2*(groups+chunks))`, groups=⌈cellCount/64⌉; split `…+T` | inside root | `balance[1u+2u*job]`, chunk sums, `UM_CUT_BASE+o.index` (all-4h) | `${groups}u`, `${1+2*groups}u`, `${chunks}u`, `UM_CUT_BASE` |
| Transport edges | `uniform-mixed-transport.ts:90` `{offset:0,size:cells*40}` | 2560·T | 671.1 MB | `rowAt`: `t*640u+select(lane*(maxSide³+2u),lane,unit)`, stride 64 (unit) or 1 (4h) — TILE index (`…transport.wgsl.ts:82-92`) | literal `640u` |
| Transport donors | `…transport.ts:91` `{offset:donorOffset,size:cells*12}` | 12·N | 201.3 MB | `rigidExchange[i]`, `[TP_PLANE+i]`, `[TP_FLAGS+i]` — owner | `TP_PLANE=${cells}u`, `TP_FLAGS=${2*cells}u` (`…wgsl.ts:53,56`) |
| Transport sums | `…transport.ts:92` | 4·N | 67.1 MB | `sums[o.index]` — owner | bind offset `donorOffset+cells*12` |
| Extension slots | `FRAME:300`; `uniform-mixed-extension.ts:39-42,64-65,84` | 2·⌈8·(3N+5T+3n²+3(n/4)²)⌉₂₅₆ | 829.6 MB | `umSlot`: unit `(3u*tile+axis)*64u+local` — TILE + h-local; coarse `UE_COARSE+axis*UE_TILES+tile`; walls; flags `UE_FLAGS+2u*tile` (`:136-141,306-312`) | `UE_COARSE`, `UE_UNIT_WALL`, `UE_COARSE_WALL`, `UE_FLAGS` from UM_T/UM_D |
| Sharpening scratch | `FRAME:315` `{offset:0,size:edgeBytes}`; `uniform-mixed-sharpening.ts:96,150-152` | 16·N + 24·L (bound 40·N) | 671.1 MB bound | `umRawAt`: `3u*hLinear(anchor)+axis` — h anchor; `umBudgetAt`: `${3*n}u+6u*o.index` — owner; `umCacheAt`: `${3*n}u+6u*L+hLinear(anchor)` — h anchor | `${3*n}u`, n = 64T |
| Forces normals | `FRAME:322` `{offset:0,size:16*N}`; `uniform-mixed-forces.ts:67,127,130` | 16·N | 268.4 MB | `normals[owner.index]` — owner | none |
| Surface volume | `FRAME:311`; `uniform-mixed-surface-volume.ts:55-59,76,83` | `4*(2N+V+4*(groups+chunks)+8+3T)` | 209.5 MB | band `scratch[o.index]`, `scratch[N+o.index]` — owner; scale `scratch[2N+umVertexIndex(p)]` — h vertex; partial rows, chunk sums — job; flags — tile | `N`, `P=2N+V`, `R`, `S`, `SV_BAND/VISIT/MEASURE` |
| Surface evidence + deferred | `FRAME:308`; `uniform-mixed-surface.ts:87-98` | ⌈12T+32⌉₂₅₆ + 16 + 4·V | 71.0 MB | `evidence[tile]`, `evidence[UM_SOLID_WORDS(+UM_TILES)+tile]` — tile; `deferred[4u+i]` — list slot, capacity V | `UM_SOLID_WORDS=UM_TILES+8u`; JS bind offset |

Arena today = L0+L1+L≥2 table = 996.4 MB; the largest real need is transport edges+donors 56·N = 939.5 MB.

---

## 2. Proposed capacities

### 2a. Own buffers

| # | Buffer | Proposed bytes | 256³ at C=0 | C=4096 | C=T | Shader change |
|---|---|---|---|---|---|---|
| 2 | band rows | `4*rowFields*64*CAP`, CAP = max(1, min(C, capacityOf(T))) [6.9: wrong for host layouts] | 5 KB | 22.0 MB | 704.6 MB | `const CAP/N/M/SLOTS` (`BAND:216-217`) → words of a 4th vec4 in the band params uniform (48 → 64 B, binding `0:{buffer:f.params,size:48}` `BAND:191`, writer `FRAME:423`). `SHARED=min(capacity,memory)` (`BAND:210-213,666`) → `memory` only (kernel already spills past SHARED, `BAND:671,717`) |
| 6 | band aggregates | `380*CAP` | 0.4 KB | 1.6 MB | 49.8 MB | same runtime `M`, `CAP` |
| 8 | band iterate | `4*(6n² + 64*CAP)`, halo FIRST [6.3: cells first] | 1.6 MB | 2.6 MB | 35.1 MB | halo base `N+face` → `face`; cells at `HALO+c`, HALO = 6n² baked from UM_D |
| 9 | band index | `4*(26 + T + CAP)`, slot map FIRST [6.2: `4*(26 + 2T)`, list first] | 1.0 MB | 1.1 MB | 1.6 MB | `SLOTS=26` (fixed), `LIST=26+T`; `slotMapOffset` (`BAND:148`) becomes `4*26` |
| 7 | presented phi + stage grids | `4*(2T + 64*CAP + 8+7T+⌈T/32⌉)` | 9.5 MB | 10.5 MB | 43.0 MB | none: overlay reads band capacity from header word 4; header writer `FRAME:425` writes CAP |
| 3 | hanging tap cache | `4*(2T + 173*S)`, S = min(T, 27·C) | 2.1 MB | 78.6 MB | 183.5 MB | `UM_HANGING_SLOTS` const (`uniform-mixed-topology.wgsl.ts:89-100`) → `(arrayLength(&hanging)-2u*UM_TILES)/173u` [6.4]; builder already takes `params.hangingCapacity` at run time (`BUILD:156,275`) |
| 5 | sharpening tile list | `4*(8+3T+O)` | 4.2 MB | 5.2 MB | 70.3 MB | none: the active list is the last section (`SH_ACTIVE=${8+3*tiles}u`, `uniform-mixed-sharpening.ts:136`), appended by atomic count (`:407-409`) |
| 4 | root phi | new frame-owned buffer `4*O` ("Uniform mixed pressure phi"); conditioning floor becomes `⌈(sharpenBaseWords+8+T)/4⌉*16` | 1.0 MB (+ conditioning 12.6 MB) | 2.1 MB | 67.1 MB | none (bind sizes only: `…-authority.ts:86`, `BAND:186,193,195`, `…-memory.ts:19`) |
| 10 | solid deposits | `4*O` | 1.0 MB | 2.1 MB | 67.1 MB | none |

S = min(T, 27·C) is a proof bound, not a budget: a seam tile has a mixed-width 3³ stencil, so it lies within one tile of an h tile. It only saves bytes while C < T/27. A tighter S from lagged seam counts (work receipt words seam-h/seam-4h) with the existing `FATAL_HANGING` is a later option.

### 2b. Arena

New layout, low to high: `[stage scratch = max of consumer extents][root: 7×⌈4·c4⌉₂₅₆ + frozen][native L≥2]`. [7.1: the stage scratch is a frame-owned buffer, not an arena prefix; 7.3-7.5: the consumers are keyed by owner index / fine-tile rank, never by patch slot]
L0/L1 labels keep an offset entry (so they stay 1³ placeholders, `uniform-texture-pages.ts:38-41`) but occupy no extent; all point at offset 0.

| Consumer | Key after change | Proposed extent (bytes) | 256³ C=0 | C=4096, Sp=64 | C=T, Sp=512 |
|---|---|---|---|---|---|
| native L≥2 | unchanged | 68·Σ_{k≥2}(n/2ᵏ+2)³ | 22.7 MB | 22.7 MB | 22.7 MB |
| root | plan from the all-4h layout | 7·⌈4·c4⌉₂₅₆ + ⌈frozen⌉₂₅₆ | 10.2 MB | 10.2 MB | 12.3 MB |
| frozen (authority) | chunk sums first (fixed), partials after | max(8·(1+2⌈T/1024⌉+2·(C+⌈T/64⌉)), 8·(1+2(⌈T/64⌉+1)+T)) | in root | | |
| transport edges, A1 | owner rank: unit `10u*base+lane` stride 64; 4h `640u*umCounts.x+128u*(index-64u*umCounts.x)` | 4·(512C+128T) | 134.2 MB | 142.6 MB | 671.1 MB |
| transport edges, A2 | 4h regular row 16 words; grain<4 rows in a wide pool of W = min(T−C, 26C) × 128 words, slot in `sampling[tile]` bits 2..31 | 4·(640C+16(T−C)+128W) | 16.8 MB | 81.5 MB | 671.1 MB |
| transport donors + sums | owner, run-time stride L | 16·O | 4.2 MB | 8.3 MB | 268.4 MB |
| extension slots | unit slots by detail patch slot; coarse/flags/walls unchanged | 16·(3R + 5T + 3n² + 3(n/4)²) | 24.3 MB | 125.0 MB | 829.6 MB |
| sharpening | budgets by owner; raw + cache by detail patch slot | 4·(6·O + 4R) | 6.3 MB | 46.0 MB | 671.1 MB |
| forces normals | owner (bind size only) | 16·O | 4.2 MB | 8.3 MB | 268.4 MB |
| surface volume | fixed/T sections first, then partial rows (C+pages), band 2·L, scale by patch slot (vertex class) | 4·(8+3T+4⌈(T+pages)/64⌉+4(C+pages)+2·O+Rv) | 5.4 MB | 16.7 MB | 215.3 MB |
| surface evidence + deferred | deferred capacity Rv + (n/4+1)³ entries | ⌈12T+32⌉₂₅₆+16+4·(Rv+(n/4+1)³) | 4.2 MB | 13.4 MB | 77.8 MB |
| **arena total, A1 / A2** | | native + root + max(above) | 171.3 / 57.2 MB | 183.8 / 157.9 MB | 974.5 MB |

Shader changes for baked bases:

| Stage | Today | Change |
|---|---|---|
| transport | `TP_PLANE=${cells}u`, `TP_FLAGS=${2*cells}u` | `fn tpOwners()->u32{return umCounts.x*64u+umCounts.y;}`; planes at `tpOwners()` and `2u*tpOwners()` (precedent: `umBoundaryIndex`, `uniform-mixed-pressure-boundary.wgsl.ts:59-66`). Keep `fraction=64-(⌈log2 N⌉+1)` baked from N (a bound, not a size). Bind: rigidExchange `{offset:D,size:12*O}`, sums `{offset:D+12*O,size:4*O}`, D after edges |
| transport `rowAt` | `t*640u+…` | owner-rank address above. `rowAt` already loads `umTopology[t]`; every `edges` access is the owner's own row (`rowBase`, `donorFrom`, `buildAt`, `normalizeCoarseRow`, `rowsFallbackFineAt`, `tpFineJob0/2`, `gatherAt`), so no neighbour lookup is added |
| sharpening | `${3*n}u+6u*o.index`; h-linear raw and cache | budgets at `6u*o.index` (base 0); raw/cache at `6u*L + patchWord(anchor)`. Sign −1 reads (`shLimit`, `shCommit:382`, `shLimitSeam`, `shCommitSeam:296`, `umProposal:342`) address the lower neighbour's anchor; patch keying keeps them position-addressed. This is the "(owner, axis) key" defect named in the storage design |
| extension | `(3u*tile+axis)*64u+local` | `patchSlot(anchor)*3u*E³ + axis*E³ + local(anchor)`; one directory read per address (`udEntry`, `DET:886`). The code comment "The address needs the patch width, never a topology load" (`…extension.ts:119-121`) no longer holds: measure |
| surface volume | `N`, `P`, `R`, `S` baked | section order `[state 8][flags 3T][chunk sums 4⌈(T+pages)/64⌉][partials at fixed base, 4/job][band0 L][band1 L][scale]`; band base = partial base + 4·svJobs() at run time; add `index<jobs` guard in `storePartial` |
| authority | `${groups}u`, `${1+2*groups}u`, `${chunks}u`, `UM_CUT_BASE` | chunk sums at base 1 sized 2⌈T/1024⌉ (jobs ≤ T always); partials after; guard on the run-time job count already computed at `:254`. No C in the shader |
| pressure cycles | `UM_*` from the sim-layout spacing | call `planUniformMixedPressureMemory(coarseLayout, …)`; constants become functions of n only. `MG_*` are re-derived at build from the new offsets (n only) |
| band | `CAP`, `N`, `M`, `SLOTS` | params words (2a) |

Patch-slot helper to add beside `detailAddressWGSL` (`DET:871-908`): `udSlotWord(p:vec3u)->i32` = `(e.x+LX*(e.y+LY*e.z))*E³ + local(p)`, −1 when `udEntry(p>>2).w==0u`. Under identity placement Sp = all patches and the formulas reduce to today's N-scaled sizes: one code path at every occupancy.

Coarse transport rows: a 4h row is side³+2 words, side = 4/grain+1: 10 words at grain 4, 127 at grain 1 (grain 2 unreachable). Grain is decided in `buildAt` (`…transport.wgsl.ts:126-133`) and stored in `sampling[tile]`. A2's W bound assumes a departed box only reaches an h tile from an adjacent 4h tile; check that against the certified reach before relying on it, and latch `transportCapacity` (frame-status cause 8, defined with no setter today) on pool overflow.

---

## 3. Growth transaction

### What exists

| Where | What it decides | Host-known before the frame? |
|---|---|---|
| `REF:1427-1465 updateMixedRegions` → `planDetail` (`uniform-detail-planner.ts:144-232`) + solid/body `forced` | Requested/Full fine set; `frame.updateLayout(layout)` | Yes, F exact |
| `FRAME:653-678 updateLayout` | `detail.reserve(layout)` (grows the atlas, `DET:327-349`) → `remap.apply(layout, encodeAdmit)` → phi resolve → `detail.encodeRetire` | Yes |
| `FRAME:682-691 setRelayout` | `detail.reserveAll()` | n/a: gives up, every patch gets a slot |
| `FRAME:719-754 encodeRelayoutHead` | census → builder → `detail.encodeAdmit` → `remap.applyGpu` (`adoptGpu`) → phi resolve → `detail.encodeRetire` | No: F of frame N is GPU-only |
| `FRAME:464-465, 638` | lagged work receipt (ring 2): exact F of frame N−2, used for launch widths only (`OWN:148-162`: "Never use these estimates as storage/admission bounds") | F(N−2) |
| `uniform-detail-pool.ts` | geometric growth within a byte budget, old+new peak, deferred reclamation | Not wired to the frame |

### Rule

`UniformMixedFrame.reserveFine(need)`, called between frames only:
- `if(need<=C)return; C'=min(T, max(need, 2*C))` (`UNIFORM_DETAIL_POLICY.poolGrowth`).
- Create every buffer of section 2 at C'; rebuild the bind groups that reference them; write C' to the band params and to builder params word 3; destroy the old buffers after the submit of the last frame encoded on them (as `DET grow()` does).
- No content copy, with two exceptions: hanging tables words [0,2T) (`copyBufferToBuffer` old → new, 8T B; a host relayout rewrites them anyway), and the presentation phi buffer (re-created, see hazards).
- It runs before the relayout head is encoded. The remap touches no owner-indexed buffer, so "admit before remap, retire after resolve" is satisfied by ordering alone: grow → admit (detail) → remap → resolve → retire.
- Shrink only at a frame boundary, on lagged evidence (F(N−2) < C/4 for 8 frames), same rebuild. Optional.
- Changing C compiles no shader (section 2: no C-derived constant remains).

| Policy | Capacity before frame N | Overflow |
|---|---|---|
| Requested | `reserveFine(layout.fineTiles.length)` in `updateLayout`, before `detail.reserve` | impossible (host layout); host throw in `OWN update()` if `fineTiles.length>C` |
| Full | first layout reserves C = T | impossible |
| Dynamic | C = min(T, max(M_host, κ·budgetTiles, 2·F(N−2))); M_host = host count of static-fine ∪ solid-coupled ∪ body ∪ join tiles; budget 100% ⇒ C = T | GPU fatal below |

Dynamic prerequisites: the initial host layout must stop being all-h (`REF:1451`; start from `plan.fine ∪ forced` on a 4h background, as Requested does), and `detail.reserveAll()` must become `reserve` for the same budgeted closure. The declared lane default `detailBudgetPercent: 100` keeps C = T, so existing Dawn lanes see today's sizes.

### GPU overflow fatal (Dynamic)

- `BUILD scan` lane 0 (`:241-278`) knows the total `f=grand[0]` before `scatter` writes any tile word. Add beside `:274-276`: `if(f>params.fineCapacity){atomicOr(&status[R_FATAL],FATAL_OWNERS);}` with `FATAL_OWNERS=16` and the spare 4th params word (`BUILD:156`, written 0 today).
- Carrier already exists: builder status word 34 → remap `markListed` (`uniform-mixed-remap.ts:474-481`) → `umLatchFailure(layoutCapacity=6,…)` → frame receipt → host throw at `FRAME check` (≤ 2 frames later). Extend the `markListed` mask so bit 16 maps to cause 6, and name it in `relayoutDetail` (`uniform-mixed-frame-status.ts`). Message: demand from receipt word 21 (`tiers`), capacity C.
- Same frame: stages still run on the over-capacity generation. Every owner-indexed range is its own binding with an explicit size, so out-of-range `buf[o.index]` is clamped inside that binding by WebGPU; projection is withheld and the host throws. Not verified on Dawn/Metal.
- Stronger option, unverified: `scan` skips its counts/header writes and `scatter` returns on the fatal bit, so `adoptGpu` re-copies the previous generation. Needs a check that the staged buffers still hold a complete previous generation. [6.1: they do not; not viable]

### What a hard capacity needs from the importance budget

Today (`uniform-mixed-dynamic.ts`): `importance` drops seeds below `census[cutoffIndex()]` (`:796-799`); `budget` computes that cutoff for the NEXT census (`:838-848`); the saturated bin is always kept. Not capped: closure and travel in `decide` (`:1036-1090`), joins, source tiles, hysteresis holds, solid promotion (`:1100-1117`), the builder's static fine mask (`BUILD:175`). Observed: 927 h tiles against a 460-tile budget.

| Gap | Needed for F ≤ C by construction |
|---|---|
| Cutoff lags one census | apply the cutoff to the census that built the histogram (order: importance → budget → decide) |
| Budget counts seeds, F is the closure | histogram closure tiles, not seeds: each tile inherits the max bin over its 27-neighbourhood seeds (one max in the existing dilate), then cut on cumulative closure count ≤ C − M |
| Mandatory tiles outside the budget | count M first (solid, body, join, source, static fine) in the saturated bin; M > C is the fatal |
| Holds | hysteresis holds either inherit their seed's bin (evictable) or count in M |
| Backstop | the builder `f>C` fatal stays |

Until that lands the budget stays soft, so C must not equal budgetTiles: use κ ≥ 4 (observed ratio 2.0) plus the 2·F(N−2) term, and accept the fatal as the loud failure. After it lands, κ = 1 and the fatal is unreachable except for M > C.

---

## 4. Hazards

| Hazard | Site | Note |
|---|---|---|
| All-fine capacity assertion | `FRAME:257` | replace with an explicit capacity (`UniformMixedCapacity` gains `fineTiles`, `owners`; `OWN:94-102`). Do not derive C from the constructor layout: `tests/uniform-transport-pruning-dawn.test.ts:46,59` builds the stage all-4h then updates to all-h |
| Tile word low 30 bits | `uniform-mixed-layout.ts:45` (`count*64 > MIXED_CELL_MASK`), `BUILD:320` | unchanged: owner indices only get smaller. The check is missing from `createUniformMixedLayoutFromWidths` and `uniformMixedLayoutFromTiles` |
| Constructor reads of the capacity layout | `…transport.ts:59-66`, `…-memory.ts:16-19`, `…-authority.ts:64`, `BAND:154-160,186`, `uniform-pressure-surface-band.ts:20`, `…sharpening.ts:96,106`, `…forces.ts:67`, `…surface-volume.ts:55-59` | each must take capacity, not `layout.cellCount` / `ownership.layout` |
| Owner index < n³ assumed | transport `TP_PLANE/TP_FLAGS`, sharpening `${3*n}u`, surface volume `N`, authority `groups`, deposits, normals bind | section 2b |
| Authority comment relies on groups = T | `…-authority.ts:60-63` | jobs = F+⌈(T−F)/64⌉ ≤ T still holds; partial capacity becomes C+⌈T/64⌉ |
| Renderer/overlay sharing | `REF:1408 mixedSource` built once; overlay `setDenseLevelSetVolumeSource` (`lib/core/webgpu-grid-overlay.ts:2192-2200`) compares the source object, `vertexPhi`, `openFraction`, `mixedOwnership.buffer` only | re-creating presentation phi (b13) needs a NEW `mixedSource` object, or b12/b13/b14 stay bound to the old buffer. Topology and support are not capacity-sized, so the water pipeline is unaffected |
| Stage-grid record must end at `arrayLength` | `uniform-stage-grids.ts:126-148` | the re-created buffer must be exactly sized; header word 4 = CAP |
| Band `present` group and guard | `BAND:199,201` | rebuilt with the band |
| Band index layout change | `FRAME:582` copy, `tools/probe-uniform-stage-scaling-dawn.ts:179,223` (hard-coded `26*4`), `tools/probe-uniform-surface-compare-dawn.ts:74`, `tests/uniform-pressure-phase-dawn.test.ts:46` | follow `slotMapOffset` |
| Explicit-size arena bindings | cycles `mixedRange/nativeRange` (`…-cycles.ts:110`), sharpen, normals, authority, projection, acceptance | fail validation if the arena shrinks under them: change in the same step |
| Cached CM11a dispatch params | `…-multigrid.ts:1498-1524`, keyed by scratch offsets | invalidate when L≥2 offsets move |
| Host all-fine arithmetic | `REF:854` (`N*24` limit check), `REF:929-931` (`edgeBytes`), `REF:864-867` (`twoLevelTileCount`), `REF:1255-1258` (`allocatedBytes` analytic terms), `REF:1267` | arena and conditioning are counted once from formulas; `refreshMixedAllocation` (`REF:1358`) only tracks the frame and detail storage. Count the arena from `buffer.size` |
| `readStats` support copy | `REF:2913-2937` (offset `capacity.tiles*16`) | T-keyed, safe |
| Tests/tools sized all-fine | `tests/uniform-sharpening-seam-dawn.test.ts:33-60`, `uniform-cubic-surface-dawn.test.ts:40-63`, `uniform-coarse-redistance-dawn.test.ts:25-36`, `uniform-transport-pruning-dawn.test.ts:29,46`, `uniform-pressure-phase-dawn.test.ts:23-54`, `tests/helpers/uniform-geometric.ts:79`, `tools/probe-uniform-sharpen-fusion-dawn.ts`, `tests/uniform-scratch-layout.test.ts:14-21` | direct-stage constructors change signature; the scratch-layout unit test asserts arena offsets (delete per policy if it fails) |
| Tools asserting owner counts | `tools/compare-uniform-dam-resolution-dawn.ts:37-132`, `tools/probe-uniform-stage-scaling-dawn.ts:154-155` | fine arms need C = T (Full) |
| Growth vs capture | `tools/uniform-frozen-gpu-frame.ts` asserts no allocation change during capture; `tests/voxel-editor-retained-frontier-dawn.test.ts` asserts `volumeTexture` identity over 90 frames | run with a fixed capacity |
| Dynamic initial all-h layout; `reserveAll` | `REF:1451`; `FRAME:682-691` | section 3 |
| Surface-volume partial overflow | `…surface-volume.ts:116,132,403-407` | finding 10 [6.6: fixed] |

---

## 5. Steps

Each step converts its buffers to capacity-sized + rebuildable and is verified at a fixed C < T through a QA option (same shape as `setUniformDetailStorageForQA`), default C = T until step 6.
Before every Dawn run: `DETAIL=packed node --import tsx tools/capture-uniform-wgsl.mts` (naga parse + census), plus the mock-device allocation census for bytes. One lane per step; none was run.

| Step | Change | Files | 256³ saving at C=0 | Lane |
|---|---|---|---|---|
| 0 | `UniformMixedCapacity.fineTiles/owners`; frame takes capacity, assertion `:257` removed; `OWN update()` throws on F > C; builder `fineCapacity` word + `FATAL_OWNERS` + `markListed`/status naming | FRAME, OWN, BUILD, `uniform-mixed-remap.ts`, `uniform-mixed-frame-status.ts`, REF | 0 | `npm run test:dawn -- detail-policy` |
| 1 | Band at CAP = min(C, capacityOf(T)) [6.9]: rows, aggregates, iterate (halo first), index (slot map first), presentation; CAP in params; new `mixedSource` on re-create | BAND, FRAME, `uniform-stage-grids.ts`, REF, two probe tools | 823 MB | `-- pressure-phase` |
| 2 | Hanging cache S = min(T, 27C); clamp from `arrayLength` | OWN, `uniform-mixed-velocity-sampling.wgsl.ts`, `uniform-mixed-topology.wgsl.ts` | 181 MB | `-- detail-policy` |
| 3 | Own owner-indexed buffers: sharpening list 4(8+3T+O); root phi to its own 4·O buffer and conditioning floor; deposits 4·O | `uniform-mixed-sharpening.ts`, `uniform-mixed-pressure-memory.ts`, FRAME, ARENA, REF, `uniform-mixed-solid-displacement.ts` | 66 + 60 + (67 lazy) MB | `-- sharpening-seam` |
| 4a | Arena layout: root planned from the all-4h layout, authority scratch reordered, L0/L1 extent dropped, scratch region = max of consumer extents | ARENA, `uniform-mixed-pressure-memory.ts`, `uniform-mixed-pressure-authority.ts`, `uniform-mixed-pressure-cycles.ts`, FRAME, REF | 24 MB (arena → 972) | `-- pressure-local-visit` |
| 4b | Transport: edges by owner rank (A1), donors/sums at run-time stride, forces normals bind 16·O [7.1, 7.2] | `uniform-mixed-transport.ts`, `.wgsl.ts`, `uniform-mixed-forces.ts`, FRAME | arena → 862 | `-- transport-pruning` |
| 4c | Extension unit slots by fine-tile rank [7.3] | `uniform-mixed-extension.ts`, DET (helper) | arena → 704 | `-- detail-policy` (no direct mixed-extension lane exists) |
| 4d | Sharpening raw/cache by owner / fine-tile rank, budgets at base 0 [7.4] | `uniform-mixed-sharpening.ts` | arena → 242 | `-- sharpening-seam` |
| 4e | Surface volume re-layout; surface deferred list left lattice-keyed [7.5] | `uniform-mixed-surface-volume.ts`, `uniform-mixed-surface.ts` | arena → 171 | `-- cubic-surface` |
| 4f | Transport coarse rows A2 (16-word rows + wide pool, cause 8) [7.6: not viable, not built] | `uniform-mixed-transport.ts`, `.wgsl.ts`, `uniform-mixed-frame-status.ts` | arena → 57 | `-- transport-pruning` |
| 5 | `reserveFine` growth; Requested/Full reserve from the layout; default C leaves T [6.10, 6.12] | FRAME, REF, every stage's rebuild | enables the above by default | `-- detail-policy` |
| 6 | Dynamic: non-all-h start, budget-derived C, `reserve` instead of `reserveAll`; then hard budget (section 3) | REF, `uniform-mixed-dynamic.ts`, BUILD, DET | Dynamic stops being container-proportional | `-- dynamic-coarsening` (asserts `compiled==[]`) |

Arena figures in 4a-4f are at C = 0, Sp = 0 and are cumulative; the arena is a max, so 4b-4d only pay once all three have landed.
Memory check after step 1 and after 4f: `node --import tsx tools/profile-uniform-geometric-dawn.ts --scene=cm12-figure-7-256 --frames=60 --out=/tmp/profile.json --allocation-audit --max-gpu-bytes=<N>`.

[7: as built, 2,505 MB → 561 MB. The stage scratch is 138 MB (transport's 128-word 4h rows, 7.6) where this estimate assumed 24 MB.] After all steps, 256³ zero detail (packed): 2,505 MB → about 437 MB (arena 57, conditioning 12.6, presentation 9.5, band 2.6, hanging 2.1; the rest is unchanged T-scaled state, of which the four detail face base blocks are 134 MB).

## Not verified

- Nothing was run on a GPU. All "after" bytes are arithmetic from the proposed formulas.
- A2's wide-row bound W = min(T−C, 26C) and the deferred-vertex bound Rv + (n/4+1)³ are derivations, not code facts.
- CM11a L2 `rhs B`, `p-min B` and all `residual A` look unbound in the continuation; not confirmed by a dispatch census (does not affect sizes here).
- Robust-access clamping of an over-capacity frame on Dawn/Metal.
- The frame-time cost of the extra directory read in extension and sharpening addresses.

## 6. Implementation record and corrections (2026-10-04)

Steps 0, 1, 2, 3, 4a and 5 are in the tree (4b-4e: section 7). Step 6 is not started. Every Dawn result below is an in-run A/B on one device
(default capacity against a fixed C through `setUniformMixedFineCapacityForQA`), bit-identical unless stated.

### Corrections to the analysis

1. **"Stronger option" (section 3) is not viable.** The builder's `widths` pass has already overwritten the staged topology when `scan` learns F, so skipping `scan`/`scatter` leaves no complete previous generation for `adoptGpu`. The fatal is the only GPU-side response: builder params word 3 → status bit 16 → remap `markListed` → cause 6 → receipt → host throw. Verified on Dawn/Metal: "layout build N: h tiles over the owner-indexed storage's capacity ... projection withheld", no device error.
2. **Band index is `4*(26 + 2T)`, list first.** The list holds every tile (`listSlots = T`), the slot map follows at `4*(26+T)`. A list of CAP words was only valid while CAP ≤ capacityOf(T); correction 9 makes CAP reach T. Cost against the table: +0 / +65,536 / +524,288 B at 64³/128³/256³.
3. **Band iterate keeps cells first**, halo at the run-time `N = 64*CAP` (a params word). Halo-first would have changed every cell index for no saving.
4. **`UM_HANGING_SLOTS` cannot become `arrayLength` in the topology WGSL**: that module is included by stages that do not bind the hanging buffer. It is `umHangingSlots()` in `uniform-mixed-velocity-sampling.wgsl.ts`, compiled only where `umHanging` is bound. The counted `hanging` kind in the topology WGSL has no callers.
5. **Seed exemption.** The constructor seed is all-h and may exceed C. Ownership records `overCapacity` and `acquireFrame` refuses a frame until `update(layout)` replaces it; no storage is sized for the seed. A live switch to Dynamic continues from the current generation (it does not re-seed all-h).
6. **Finding 10 was real** and is fixed outside this plan (`uniform-mixed-surface-volume.ts`, partial rows = tiles + pages). Step 4e no longer owns the guard.
7. **Root phi is two buffers, not one 4·O buffer.** "Uniform mixed pressure phi" (4·O, simulation owners: simulation authority writes it, band list/prep read it) and "Uniform mixed root pressure phi" (4·T, all-4h owners: split authority, RHS/projection, cycles, band init, presentation copy). The two index domains were sequential uses of one view; apart, growth rebinds only the simulation authority group and the band.
8. **Authority scratch is not reordered** (step 4a). With groups = T its workspace is already C-free; `frozen` is sized by `frozenBytes(dims)`. A reorder would save at most 4 MB at 256³ in another workstream's file.
9. **Band CAP depends on who owns the layout.** The h band holds every h tile with a liquid row, so a host layout that reserves C h tiles can put C tiles in the band: host layouts (Requested, Full) use CAP = max(1, C); only a GPU relayout (Dynamic, C = T) keeps the liquid bound `capacityOf(T, C) = max(1, min(C, T, max(4096, ⌈T/2⌉)))`. Consequence: Full pays band rows at CAP = T — 176,160,768 B at 128³ (88 MB under the liquid bound) and 1,409,286,144 B at 256³ in one buffer, which the admission check refuses on a device whose `maxBufferSize` or `maxStorageBufferBindingSize` is 2^30.
10. **Admission is a check before adoption, not a fatal.** `UniformMixedFrame.fineReservation(need)` is pure: it returns the capacity `reserveFine` would adopt, or a refusal naming the first capacity-sized buffer over `min(maxBufferSize, maxStorageBufferBindingSize)` (or old + new bytes over `byteBudget`). `reserveFine` and `updateLayout` return the refusal with nothing changed. The host (`REF updateMixedRegions`) keeps the last accepted generation and publishes `uniformDetail.rejected`; it throws only when there is no accepted generation or the kept one no longer holds a solid/body-forced tile at h. A refused runtime detail request is withdrawn whole (`this.detail` restored) so advance parameters match the running layout. The GPU fatal stays for the invariant violation (F > C under a GPU relayout). Seams left open: no host passes `byteBudget` (WebGPU reports no device memory); the planner's commit still precedes admission. [7.8: the detail control now shows `rejected`]
11. **REF's `allocatedBytes` needs no formula change**: it already follows the arena size and the frame's own buffers.
12. **Step 5 as built** differs from "Rule" in section 3:
    - Growth C' = min(T, max(need, 2·C)), exact `need` if the doubled size is refused. Shrink is immediate for host layouts when need·growth² < C (no 8-frame lagged evidence: the host knows F exactly). A fixed QA capacity never changes.
    - Requested and Full construct at C = 0 and reserve from the first layout; Dynamic constructs at C = T and `setRelayout(relayout)` reserves T with the liquid band (the "default C leaves T" row).
    - Content copied across a re-create: hanging words [0, 2T); the presented buffer's 4h phi words [0, T) and its stage-grid record (moved to the new offset, header word 4 rewritten to CAP). Everything else is rebuilt by the next frame.
    - Builder params words 1 (hanging slots) and 3 (h-tile capacity) re-sync at `encode()` from `ownership.capacityRevision`; solid displacement re-creates its deposits lazily when 4·O changes; sharpening takes the new list through `setWork`; the band through `resize({phi, presentation, capacity})`.
    - `denseLevelSetVolumeSource` returns a new `mixedSource` object when the presented buffer was re-created, so consumers keyed on identity rebind.
    - No shader compiles on a capacity change (probe: zero → 64 → 256 → 1536 → 0 h-tile capacity on one solver).

### Bytes (mock-device census, packed, Requested, no regions)

| | 64³ | 128³ | 256³ |
|---|---|---|---|
| before (C = T) | 54,104,060 | 317,936,160 | 2,504,835,488 |
| step 1 band at CAP (C = 0) | not taken | not taken | 1,683,282,476 |
| step 2 hanging cache (C = 0) | 25,602,184 | 192,572,076 | 1,501,878,828 |
| step 3 own buffers (C = 0) | 23,668,920 | 177,105,628 | 1,378,146,908 |
| step 4a arena (C = 0) | 22,099,992 | 171,155,004 | 1,356,173,244 |
| step 5, default capacity (zero detail) | 22,099,992 | 171,220,540 | 1,356,697,532 |
| step 5, Full | 81,967,068 | 652,036,032 | refused (band rows, correction 9) |
| step 5, Dynamic (C = T) | 81,967,068 | 549,341,120 | 4,383,541,056 |

Steps 1-4a are at the QA option C = 0; step 5 is the shipped default. The remaining 1,357 MB at 256³ is the arena (974 MB: steps 4b-4f) and T-scaled state.
The detail atlas (DET) does not shrink, so a solver that visited Full and returned to zero detail holds more than a fresh one (64³ probe: 51.3 MB against 22.0 MB).

### Lanes and probes

| Step | Lane | Result | In-run A/B (64³ dam, Requested, 8 frames per phase) |
|---|---|---|---|
| 0 | `-- detail-policy` | red in the Full phase at the time (finding 10, not this change; zero and Fine-region phases passed); fatal assertion added | zero → fine → wide → zero, default against C = 2600: identical; GPU capacity fatal fires |
| 1 | `-- pressure-phase` | pass | same four phases, default against C = 1600: identical |
| 2 | `-- detail-policy` | pass 2/2 | zero → small → fine → wide → zero, default against C = 1600: identical; C = 100 (S = 2700 < T) identical until the host throw at 256 h tiles |
| 3 | `-- sharpening-seam` | pass | same five phases and arms: identical, and identical to step 2's run |
| 4a | `-- pressure-local-visit` | pass | controlled A/B (tree copy with only 4a reversed, back to back): identical in five phases |
| 5 | `-- detail-policy` | pass 2/2, 137 s | five phases, growing default (C: 0 → 64 → 256 → 1536 → 0) against fixed C = 4096: fields and per-frame stats identical; fixed C = 100 identical until the host throw; GPU capacity fatal fires |

Not exercised on a GPU: the solid-displacement deposits re-create (no solids in the probe scene), the refusal path and `byteBudget` (mock device only), the overlay's read of a re-created presented buffer (no browser run).

## 7. Implementation record: the stage scratch, steps 4b-4f (2026-10-04)

Steps 4b, 4c, 4d and 4e are in the tree. 4f is not built (7.6). Step 6 is not started.
The storage patch layout was reopened (storage design, "Review, 4 October 2026"), so nothing here is keyed by a detail patch slot. Lattice-keyed consumers moved to what the tile word already holds: h tiles are packed first, an h tile's word is 64·rank, an h cell's owner is that word plus its local index. Identity and packed field placement run the same addresses.
Every Dawn result is an in-run A/B between two source trees (the live tree and a copy with only that step reversed), each on its own device of the same adapter in one process, 64³ dam, Requested, phases zero → small → fine → wide → zero → dynamic → full, 16 frames a phase.

### Corrections to the analysis

1. **The stage scratch is a frame-owned buffer, not an arena prefix.** `UniformMixedFrame` owns "Uniform mixed stage scratch", ⌈max consumer extent at C⌉₂₅₆ (`stageBytesAt(fineTiles)`), created after every stage is constructed, re-created by `reserveFine` and bound again by `bindStage()` (transport, extension, surface, surface volume, sharpening, forces). It is in `capacityBuffers`, so admission sees it. The arena ("Uniform shared pressure scratch") is `[root][native L≥2]` with `rootOffset = 0` and no longer depends on C: a C-sized prefix would have moved the root and native offsets, and the cached CM11a dispatch params keyed on them, at every growth.
2. **4b transport** is A1 as planned, with one difference: the high-limb and flag planes stride by `arrayLength(&sums)` (the bound owner capacity), not by a live-count function. Edges, donors (12·O) and sums (4·O) are three 256-aligned ranges of the stage scratch (`UniformMixedTransportStage.scratchRanges(tiles, fineTiles)`, `bindScratch`). `rowAt` derives the row from the owner it already holds: no load added. Forces normals bind 16·O (`UniformMixedForces.normalBytes`).
3. **4c extension: 240 unit slots an h tile, by fine-tile rank; the plan's `3R` had no home for 48 of them.** 192 are (component, cell) for the patches anchored in the tile. The other 48 are, per component, the sixteen unit patches of a 4h tile below it on that axis: they are anchored in the 4h tile's top layer and belong to its seam owner, but exist only because the tile above is h, so they are keyed by that tile's rank. Order: `[coarse 3T][unit wall planes][coarse wall planes][flags 2T][one never-written slot][240 per rank]`; fixed sections first, so no base depends on C. The unit negative-wall planes (3n² slots) stay lattice-keyed.
   - Address cost: the anchor tile's word, and for an anchor in a 4h tile's top layer the word of the tile above. An owner's own positive patch needs neither (`ueOwnSlot`, from the owner index); the sweep tile jobs stage the 27 words they already load for the widths and read them from workgroup memory (`ueStagedTiles`); the restriction loads its tile's words once. A/B, median ms a frame, second repeat of each arm (before / plain addressing / as landed): wide 10.82 / 10.88 / 10.96, dynamic 14.01 / 13.56 / 14.11, full 10.44 / 10.46 / 10.51, fine 8.17 / 8.24 / 8.20. Neither variant is distinguishable from the tree before it: the two repeats of the same arm differ by more (dynamic 13.45 and 14.01).
   - Extent 2·⌈8·(240C + 5T + 3n² + 3(n/4)² + 1)⌉₂₅₆. At C = T that is above transport's (3920T against 3584T bytes), so the stage scratch at Full and Dynamic is 10.8% larger than the old arena prefix: +1.59 MB at 64³, +91.4 MB at 256³. Keying the 48 by 4h rank instead (144C + 48T slots) would give those back and cost zero detail 320T bytes (84 MB at 256³); a compacted list of (4h tile, axis) pairs under h tiles needs an index the builder does not write. Rank keying is kept because zero detail is the endpoint.
4. **4d sharpening: by owner and fine-tile rank, not by h anchor or patch slot.** Budgets `6·index` at base 0. A face patch's record is its lower owner's `3·index + component` (an h owner's unit patch, a 4h owner's width-4 patch); the sixteen unit patches of a 4h owner under an h tile are that tile's `48·rank + 16·component + column`, as in the extension. Both incident owners hold the lower owner, the upper owner and the patch, so no load is added (`umRawAt(a, b, f)`, `umRawOf(o, f)` for either side). The admission cache is one word per face patch at the same key (it was one word per anchor with three components packed), which removes the seam geometry's workgroup staging and its barrier. Bases follow the live owner count; every word read is written earlier in the frame. Bytes 4·(12·O + 96·C): 48T at C = 0, 3456T at C = T.
5. **4e surface volume: only the band moved.** The two band parities were already indexed by owner, at stride N; they are now the last section at stride = live owners. Order `[scale V][partials 4·(T + pages)][chunk sums][state 8][flags 3T][band][band]`, bytes 4·(V + 4·(groups + chunks) + 8 + 3T + 2·O). The partial rows keep groups = T + pages (correction 6.6). The corner scale stays a word per lattice vertex and the surface stage's deferred list stays 16 + 4V: 4V is about 268T bytes, under transport's 528T at every C, so neither is the max of the stage scratch while 4f is out; a rank key for a vertex would cost its authority cell's tile word (up to seven lookups for an unowned corner), and the plan's deferred bound Rv + (n/4+1)³ was a derivation with no proof.
6. **4f is not viable.** The doubt in 2b was right. `buildAt` (`uniform-mixed-transport.wgsl.ts:126-137`) sets a 4h row's grain to the minimum tile width over the tiles its *departed* box touches, "regardless of travel"; the reach a tile certifies (`tpOwnerReach`, `:398-406`) is an offset of up to ±511 tiles and the whole lattice for a non-finite departure. A 4h row therefore needs the 126-edge row whenever its departure lands on an h tile at any distance, not only beside one, and any number of 4h rows can depart into the same h tiles. W = min(T − C, 26C) is not a bound, and no other bound in C holds; a wide pool would turn ordinary dynamics into a capacity fatal. The 128-word 4h row stays, and transport's 528T bytes is the stage scratch at zero detail.
7. **`sharpening-seam` now checks the addresses.** Its reference variant runs the lattice-keyed scratch (a raw flux and an admission word per (h anchor, component), then the budgets; `sharpeningLatticeAddresses` in `tests/helpers/uniform-sharpening-reference.ts`, applied to both modules) against the production owner / rank keys, over the lane's layouts, solids and policies. The lane's scratch is `max(stage.scratchBytes, 48n)`, prefilled whole.
8. **Device limits and the refusal.** The admission check reads `device.limits` (`fineReservation`), and the device is requested with the adapter's values (`requiredFluidDeviceLimits`). On this machine (Dawn node, "Metal driver on macOS Version 26.6.2 (Build 25G83)") adapter and device report `maxBufferSize = maxStorageBufferBindingSize = 4,294,967,295`. Full at 256³ needs band rows of 1,409,286,144 B and a stage scratch of 1,030,947,328 B: both fit one buffer, so correction 6.9's refusal was the mock's 2^30, not this device. Its buffers total 5.30 GB (census), which no limit checks. `uniformDetail.rejected` is shown in the detail control (`components/UniformCoarseControl.tsx`): the coverage readout says "refused" instead of "applying" and a note gives the reason.

### Bytes (mock-device census, packed)

Stage scratch, C = 0 / C = T. The pressure arena beside it is 707,104 / 4,770,880 / 34,932,832 B at every C.

| | 64³ | 128³ | 256³ |
|---|---|---|---|
| before: arena with the scratch prefix, any C | 15,387,168 | 122,211,392 | 974,456,928 |
| 7.1 + 4b | 13,119,488 / 14,680,064 | 104,120,320 / 117,440,512 | 829,620,224 / 939,524,096 |
| 4c | 10,485,760 / 16,265,728 | 83,886,080 / 129,286,656 | 671,088,640 / 1,030,947,328 |
| 4d | 3,312,640 / 16,265,728 | 26,298,112 / 129,286,656 | 209,588,480 / 1,030,947,328 |
| 4e | 2,162,688 / 16,265,728 | 17,301,504 / 129,286,656 | 138,412,032 / 1,030,947,328 |

At C = 0 the max is transport (528T); at C = T it is the extension (7.3).

Whole solver, default capacity:

| | 64³ | 128³ | 256³ |
|---|---|---|---|
| zero detail, section 6 | 22,099,992 | 171,220,540 | 1,356,697,532 |
| zero detail, now | 9,667,804 | 71,749,568 | 560,878,272 |
| Full, now | 83,637,920 | 664,550,212 | 5,301,816,388 (limit raised on the mock) |
| Dynamic (C = T), now | 83,637,920 | 561,855,300 | 4,480,257,092 |

### Lanes and probes

| Step | Lane | Result | In-run A/B |
|---|---|---|---|
| 7.1 + 4b | `-- transport-pruning` | pass | before against after, and default against C = 4096: fields and per-frame stats identical in six phases; C = 100 identical until the host throw; wide 11.37 / 11.43 ms, dynamic 14.34 / 14.23 ms |
| 4c | `-- detail-policy` | pass 2/2, 122.5 s | before, plain addressing, as landed, and as landed at C = 4096: identical over 144 frames; times in 7.3 |
| 4d | `-- sharpening-seam` | pass, 9.8 s (7.7) | before against after, default against C = 4096: identical over 112 frames; C = 100 identical until the host throw; no load added, times inside the repeat spread |
| 4e | `-- cubic-surface` | pass, 53.2 s | same arms: identical over 112 frames |

Not done: `tools/probe-uniform-transport-operator-dawn.ts` and `tools/probe-uniform-recovery-dawn.ts` still read the removed `arena.donorOffset` through `any`; the refusal line was not seen in a browser; no run at 128³ or 256³ (bytes are the mock census, times are 64³ only).

---

## 8. Implementation record: capacity follows receipts (2026-10-05)

One rule for Requested, Dynamic and Full. Supersedes the Dynamic row of section 3's policy table and its "shrink is optional" line. Host only: no shader changed.

### Rule (`followMixedCapacity`, `webgpu-uniform-reference.ts`; constants `UNIFORM_DETAIL_POLICY.capacity`)

Storage C, builder admission A ≤ C (count-then-admit: a build whose need is over A is reverted whole and reports its need). N is the need of the last build receipt, P the largest need in the receipt window, E the tiles the host adds this frame with no liquid moving (a body's swept reach, a live edit's join, newly planned static tiles under Dynamic).

| | Rule | Constant |
|---|---|---|
| First frames | the host's count is a hint, used until the first receipt of this request arrives. A census has no host count: C = T | |
| Grow | when ⌈1.5·N⌉ + E > C, to min(T, 2·N + E + 256) | `grow` 0.5, `headroom` 1, `floorTiles` 256 |
| Deferred build | its reported need is reserved before the next build (and grows as above) | |
| Return, stage 1 | after 30 receipts with no growth or deferral, if 1.5·(2·P + E + 256) ≤ C, A is lowered to 2·P + E + 256. Nothing is freed | `windowBuilds` 30, `returnRatio` 1.5 |
| Return, stage 2 | a receipt of a build made under that A, admitted, lets the storage follow A. A deferral under the cap, or a need over it, cancels the cap | |
| Host-exact requests | a request with no liquid-dependent tiles (no solid contact, no bodies) keeps the host's exact count; a smaller request returns through the same two stages | |
| Zero detail | need 0 detaches to the all-4h host layout: C = 0, base blocks only | |

Retired with it: the capacity-return and Dynamic → Requested uses of `hostMixedLayout` (the relayout stays attached, the band is switched in place, the storage returns through the cap). `hostMixedLayout` remains for the first layout, the QA patch layout and zero detail.

First frames with solids: the hint covers contact. The t = 0 host layout holds every forced tile at h (garden 6,992), so no wet cut tile starts at 4h; capacity follows the receipts from there.

Live edits: the join's tiles are in E before the edit's build, so the build is admitted. If the receipt of the build that carried the join shows a deferral, the join and the displacement are made again for the next build (the displacement is idempotent); the join is released only on a receipt of an admitted build.

### Deferral at impact (Dawn, receipts used two frames late as in the app, return window shortened to 4 builds so the capacity is tight before impact)

| Rule | dam 64³ | fig-9 | dam 128³ |
|---|---|---|---|
| (a) grow only after a deferral, doubling | 6 builds, 956 tile-frames | 13 builds, 3,937 | 9 builds, 8,552 |
| (b) headroom 25%, doubling | 0 | 0 | not run |
| (b) headroom 50%, doubling | 0 | 0 | 1 build, 604 |
| as landed (grow at 1.5 N, to 2 N + 256) | 0 | 0 | 0 |
| as landed, window 30 | 0 | 0 | 0 (timing arm) |

Every arm without a deferral is bitwise equal to C = T in V, velocity and phi at every compared frame (dam 64³: 10, 20, 30, 45, 60, 70; fig-9: 20, 40, 55, 70; dam 128³: 20, 40, 60). The arms with deferrals leave the C = T trajectory (occupancy relative L1 0.14 to 0.50). Rule (a) is not usable.

### Bytes (Dawn, `info.allocatedBytes`, MiB) and steady-state time (paired per frame against C held, null pair in brackets)

| Scene | C before → after | MiB before → after | ms |
|---|---|---|---|
| garden, Requested, contact on | 6,992 → 2,824 (runs 1,324 to 1,366 h tiles) | 233.7 → 194.6 | +0.00 ± 0.15 (+0.16 ± 0.26) |
| garden, Dynamic | 20,736 → 3,376 | 307.8 → 199.8 | +0.04 ± 0.36 (+0.25 ± 0.25) |
| dam 64³, Dynamic | 4,096 → 2,682 | 73.4 → 59.1 | −0.03 ± 0.03 (+0.05 ± 0.05) |
| dam 128³, Dynamic | 32,768 → 14,634 | 485.6 → 402.2 | −0.20 ± 0.22 (−0.08 ± 0.23) |
| fig-9, Dynamic, through impact | 16,384 → 5,190 → 13,582 | 243.4 → 181.7 → 231.8 | not timed |
| still pool 64³, Dynamic, need 0 | 4,096 → 256 | 73.4 → 35.9 | |
| Dynamic → Requested, one region | 4,096 → 384 in one frame | 73.4 → 37.1 (Requested from the start: 37.1) | |
| Dynamic → Requested, no region | 4,096 → 0 | 73.4 → 7.8 | |

A capacity change costs 1.2 to 1.5 ms of host time on its frame and no pass time.

256³ Dynamic (mock census, today's tree): C = T 4,065,919,976 B (4,480,257,092 in section 7 was an earlier tree). With the need taken as 4× the 128³ dam's (20,480 at rest, 28,756 at its peak): C = 41,216 → 2,624.8 MB (2,503 MiB); C = 57,768 → 2,787.4 MB (2,658 MiB). The floor once C > 0 is the domain-sized detail fields (1.61 GB) plus the zero-detail base (500.6 MB).

### Not closed

1. **Dynamic still starts at C = T** and holds it for about 32 frames (the receipt window). Frame 1 runs on the all-h t = 0 host layout, so its bytes are the old ones until the first return: at 256³ the start still needs 4.07 GB. Starting from a bounded host layout changes frame 1 of the declared Dynamic trajectory.
2. A need that grows by more than half within the receipt lag (two to three frames in the app) can pass C and defer builds until the next receipt: requested tiles wait at 4h for those frames (under solid contact that includes newly wet cut tiles). Not seen in any scene above at the landed constants.
3. A request whose host count is exact and tight grows once on its second frame (120 → 496 in the edit probe): the hint carries no headroom.
