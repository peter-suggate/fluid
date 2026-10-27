# NB-FLIP 16h bulk tier: progress

Branch `claude/nbflip-16h`, cut from main `ad40be44`. Goal: the 16h bulk tier
saves real memory and work on the paged NB-FLIP solver, so vast scenes run
interactively with memory following tiers.

## 1. Baseline (ad40be44)

Machine: M1 Max 32 GiB, shared GPU. All runs go through `tools/gpu/fluid-gpu-run.sh`.
`S` = the session scratchpad holding `gpu.sh` (a wrapper: `cd` worktree, runner
`MODE`, `env WEBGPU_NODE_MODULE=<main checkout>/node_modules/webgpu/index.js
node22 --import tsx`, log to file). Artifacts (gitignored) are in
`artifacts/nbflip-16h/`, `A` below.

Tier configs: `tiers-4h-b512.json` = `{"start":"4h","bricks":512}`.
Default bricks (256) fails wide seep at step 52 ("a build needs 2120 h tiles;
fluid.tiers.bricks 256 holds 2048"), so 512 is the baseline. Default tiers
(no `start:"4h"`) keep full h stores in every block: 12434 MiB live, over the
6 GiB rule; not used again.

### Wide seep (scale ladder, 4x domain)

```
$S/gpu.sh timing $A/base-w2d2-<arm>-timing.log tools/probe-scale-ladder.ts \
  --cell=0.005 --width=2 --depth=2 --steps=70 --every=10 --timestamps=40 \
  --block-mib=2 --max-gib=6 --budget-s=200 --tiers=$A/tiers-4h-b512.json \
  --values='{"volumeControlSeconds":0,"bulkTier":"16h"}' --out=$A/base-w2d2-<arm>-timing.json
```
(`off` arm: `--values='{"volumeControlSeconds":0}'`.) Lattice 1024x256x128,
pool 834 blocks, about 550 resident.

| timing mode, 70 steps | off | 16h |
|---|---|---|
| median step, all steps (ms) | 50.1 | 45.6 |
| wall, steps 40-70 median (ms) | 70.0 | 68.3 |
| GPU, steps 40-70 median (ms) | 66.1 | 64.8 |
| live / peak requested (MiB) | 2470 / 5542 | 2476 / 5548 |
| h tiles at step 70 | 3115 | 2631 |

Step time follows the jet's h tiles, not the bulk: the two arms are within
noise and 16h saves nothing today. Phase medians (off / 16h, ms): census +
layout + remap 13.0 / 14.3; velocity advection 8.6 / 7.6; grid surface
advection 7.9 / 6.2; V-cycles 6.6 / 7.0; h band solve 4.9 / 4.7; surface
volume 4.1 / 3.7; 4h transport caches 3.4 / 3.1; surface distance 3.0 / 2.6;
particle union 2.5 / 2.1; RHS pyramid 2.5 / 2.4; particle movement 2.3 / 2.8;
projection 2.2 / 2.4. Over the bulk, 16h makes V-cycles and the census slower,
not faster.

Check mode, 90 steps, 16h (`$A/base-w2d2-16h-check.json`): no failure, members
2398-2436 regions, whole-16h pages (all 8 regions members) 296 of 556 resident
(53%), COLD/WARM/HOT regions 3613 / 552 / 299 (COLD 81%). Volume drift
range 6e-6 .. 2.2e-4, -3.0e-5 at step 90 (off arm 1.16e-4 at step 40, default
tiers).

### Per-block bytes

Marginal from two pools after h-store compaction (`--steps=3`, pools 834 vs
634: `$A/audit3-p834.json`, `$A/audit3-p634.json`): 2476.1 vs 2150.8 MiB live,
so **1665.5 KiB per 4h block**, about 1120 MiB fixed (most of it the h stores at
the fine ceiling: b256 to b512 adds 259 MiB of h detail fields). Largest
per-block items: NB-FLIP cell counts / metrics / crossing masks 497.0 KiB,
mixed stage scratch 158.6, nearest-source origins 143.4, shared pressure
scratch 103.8, arena pool B 78.3, all-4h solid record 78.1, reference velocity
A/B/D and unit taps 41.0 each, frame support 36.2, hierarchy up/down 35.8 each,
ownership census 33.1, tile update errors 32.0, 16h surface evidence 29.1.

- h: an h tile costs its h stores within the fixed fine ceiling (b256 to b512,
  +2048 h tiles: +259 MiB h fields, +48 MiB particles, about 150 KiB per h tile).
- 4h: 1665.5 KiB per block (the historical 1570 was before the shared-scratch
  and solid-record growth; the 48 KiB estimate is the 4h fields alone).
- 16h: no own storage. A 16h member keeps its whole 4h block, so 16h saves 0 B.

Stage-0 peak 5548 MiB is the construction transient (full h stores before the
first compaction), not the running set.

### Rest (closed seep, 5 mm base domain, 120 steps, check mode)

```
$S/gpu.sh check $A/base-rest-<arm>-check.log tools/probe-reservoir-bulk.ts --closed \
  --steps=120 --every=30 --values='{"bulkTier":"16h","volumeControlSeconds":0}' --out=$A/base-rest-<arm>-check.json
```

| step 120 | off | 16h |
|---|---|---|
| max speed (m/s) | 0.0131 | 0.0135 |
| drift | 0 | 0 |
| bulk members | - | 548 regions |
| tiers COLD/WARM/HOT | 806/122/0 | 806/122/0 |

## 2. Design: what a real 16h tier is

Today `bulkTier:"16h"` is a constraint on 4h storage: a member region's 64 4h
rows are held to one mean in the level-2 solve, and the projection prolongs
face means. Storage and launches are still per 4h row, per block.

Real 16h: a **whole-16h page** (all 8 regions members, about 53% of the wide
seep's resident pages) gives its block back to the pool and holds a slot in a
fixed-capacity 16h page store instead:

- per page: 8 pressures, 8x3 region velocities as face means (36 interior and
  boundary face values per page, shared faces counted once), 96 seam heads per
  region already in the bulk record, a solid/open mask. About 8 + 36 + 8 + 4
  words of state plus the existing 102 words per region of bulk record: 1.5 to
  4 KiB per page, against 1665.5 KiB per 4h block.
- page word: a third page kind `BULK16 | slot` beside AIR and SOLID sentinels.
  Every lattice predicate (`umPageKind`, `umTileKind`, `umCellPresent`,
  `umLowOpen`, `umKeyStored`, ...) answers "liquid, no h, no surface" for it.
- velocity: the momentum sampler's single entry `umUnitSample` falls back to a
  trilinear 16h face-mean sample from the slot; the 16h page's own velocity
  advects at 16h (semi-Lagrangian over its 8 regions, one trace per face mean).
- pressure: the 8 rows live at arena level 4 (16h, already per page in pool B
  kt channels) with the analytic coarse operator (member-member 16, seam 25.6
  as now); the page leaves the level-2 lists, so neither memory nor launches
  are spent at 4h.
- promote (16h to 4h): claim a block, prolong the face means linearly (the
  current `umBulkRow` rule), pressures piecewise with the hydrostatic offset;
  this conserves face flux exactly (each 16h face's flux is the sum of its
  4x4 4h subfaces). Demote (4h to 16h): the area-weighted mean of the subfaces
  per face, volume-weighted pressure mean; conservative by construction.
  Demote only when the bulk record's error test and hysteresis already pass.
- capacity: the block pool sized from the non-16h census x headroom, the 16h
  slot store sized from the lattice's page count (cheap: 4 KiB x 4096 pages =
  16 MiB). Both fixed at construction; a promote that finds no free block is
  an explicit fatal, never a silent refusal.

Expected (wide seep): pool 834 to about 420 blocks (264 non-16h resident x 1.5
headroom plus margin), -690 MiB live (2476 to about 1790), +16 MiB slot store.
Step time: V-cycle level-2 launches drop 53%; census and transport skip 16h
pages; the jet-bound phases do not change, so about 45 to 40 ms. On vast
scenes where most pages are deep bulk, memory per page drops about 400x.

### Reader surface (why this is not one change)

39 files in `lib/methods/uniform/` decode page words through the lattice
predicates (several hundred call sites); every one must answer for BULK16
pages, plus the census/heat/residency, region transitions, renderer and
readback. That audit plus promote/demote and their GPU gates is the work.

## 3. Status

### Slice 1 (2dd59e34): surface volume corner scales by page (reverted in main, section 4)

The surface volume's corner scales took a word for every vertex key of every
block (35937 words, 140 KiB a block) and were the stage scratch's largest
consumer. They now live on page slots: `clearBand` empties a page table,
`claimPages` gives every page a measured tile's corners reach a slot of
`fluid.tiers.surfacePages` (default: every stored block, so unset scenes
keep their sizing), and `metric`, `resolveScale` and `measure` address the
scales through it. A refused page, or a visited owner's read or a measured
owner's write that finds no slot, is the host's fatal (receipt words
claimed / refused / missed / capacity, `UniformMixedSurfaceVolume.receiptFailure`).

Tiers `tiers-4h-b512-sp256.json` = `{"start":"4h","bricks":512,"surfacePages":256}`.

| wide seep, 16h | baseline | surfacePages 256 |
|---|---|---|
| live / peak requested (MiB) | 2476.1 / 5547.9 | 2434.1 / 5506.0 |
| stage scratch (MiB) | 131.9 | 89.2 |
| pages claimed, steps 1 / 90 | - | 136 / 198 (most 200) |
| check 90 steps: drift at 90, failure | -3.0e-5, none | -5.4e-6, none |
| timing 70 steps: wall / GPU steps 40-70 median (ms) | 68.3 / 64.8 | 71.1 / 67.0 |
| surface volume + geometry phase (ms) | 3.74 | 4.06 |

The timing difference is one sample each on the shared GPU and sits mostly
in phases this slice does not touch (transport 3.08 to 3.93, advection 7.60
to 7.93); not established as a cost or as noise. The stage scratch is now
bound by the extension's slot arrays (`extensionSlots`: 5 words a 4h tile key
plus the low-face keys), so a capacity under about 256 saves nothing more on
this scene: -42 MiB, about 50 KiB a block.

Rest (closed seep, `--tiers=$A/tiers-sp128.json`, 128 pages, 32 claimed):
step 120 max speed 0.013472 m/s, volume 2949119.8728, drift 0: the same as
the baseline 16h arm to every printed digit.

Gate: `tools/gate-repeat-spread.ts surface-pages-splash
--baseline=<ad40be44 archive> --candidate=<2dd59e34 archive> --scene=splash
--steps=70 --repeats=2 --set=volumeControlSeconds:0` passes (worst window
ratio 0.10; `$A/gate-splash/`). It runs the default capacity, which still
takes the slotted path. The registered seep gate (3 repeats, 6 instances of
about 3 GiB) does not fit the 6 GiB rule and was not run.

### Not landed

- NB-FLIP bins (DEPTH_A vertex bank, DEPTH_B cell bank, surface-hot
  vertex bank: 105k words, 409 KiB a block). Sizing them is not
  allocation-only: about 35 address sites in the membership, nearest,
  redistance, splat, surface, particle-brick and surface-brick-arena shaders
  index them by block, and their contents persist across frames under the
  regional crossing reuse (reach bits survive for regions not searched
  again; the particle refine box reads DEPTH_B in the next frame's move). A
  slot table for them must be stable (claim on first reach, release when a
  page's last reach bit clears, clear a claimed slot), so it is a change to
  the crossing search's lifecycle.
- Nearest-source origins (143 KiB a block): the CM11b 3.3 hierarchy textures
  are a block atlas, and liquid pages carry values between blocks at the 32h
  level, so they are not surface-only.
- BULK16 (steps 1-6 of section 2): not started.

## 4. Step time first (main checkout, 2026-10-27)

Priority change from the user: step frame time matters much more than memory.
Everything below was measured in main's working tree (other agents' hunks
included) in `timing` mode. Artifacts are in `artifacts/nbflip-16h-perf/` (`A`).
`S/gpu.sh MODE LOG script args` is the same wrapper, now `cd`-ing to main.

### Tools

- `tools/gpu-kernel-timer.ts` (new): it wraps a device so that every compute
  pass splits into one pass per dispatch, each with its own timestamp pair.
  Labels are `<pass label> / <pipeline label>:<entry>{override constants}`.
  Splitting serialises: the split sum ran 37 ms against a 34 ms clean step
  early, and 68 against 82 timer-on late. Use it for attribution only. The
  timer quantum is 65.5 µs, so most small kernels read as 0.066 to 0.197.
- `tools/probe-scale-ladder.ts`: `--kernels=<step>` (per-dispatch medians in
  `kernelMedians_ms`) and `--water=inviscid` (zero viscosity and surface
  tension).
- `tools/probe-reservoir-bulk.ts`: `--tiers`, `--water=inviscid`,
  `--kernels=<step>`, and a per-step `stepMs` list with a `timing` block
  (median and p95).

### Where the wide-seep step goes

Wide seep: 16h, `tiers-4h-b512`, 70 steps. Clean step time fits a line in
h tiles very well: rms 0.7 to 1.9 ms, slope 10.6 ms per 1000 h tiles. The
intercept follows resident blocks across a width ladder (one repeat set, 2 to
3 repeats each):

| scene (16h, 4h-b512) | resident blocks (late) | whole-16h pages | intercept a (ms) | ms per 1000 h tiles |
|---|---|---|---|---|
| w1 d1 (2 repeats) | ~158 | 66 | 19.3 | 11.4 |
| w1 d2 (2 repeats) | ~290 | 136 | 24.2 | 11.2 |
| w2 d2 (3 repeats) | ~550 | 296 | 35.6 | 10.6 |

So the clean step is about 13 ms fixed, plus 43 µs per resident block, plus
10.6 µs per h tile. Over half the resident blocks of the wide seep (296 of
557) are whole-16h pages that still pay their full 4h per-block cost.

Per dispatch (kernel timer, medians; early: k2, steps 8-24, ~150-540 h tiles;
late: k4, steps 56-70, ~3000 h tiles), grouped by pass, top entries:

| pass | early ms | late ms |
|---|---|---|
| pressure arena solve (V-cycles) | 6.10 | 7.01 |
| surface advect deferred | 1.51 | 4.85 |
| dynamic ownership census | 2.16 | 4.13 |
| global surface volume | 1.97 | 3.93 |
| body forces (capillary normals/curvature) | 1.25 | 3.34 |
| extension | 2.30 | 2.95 |
| surface advect | 1.31 | 2.88 |
| surface pressure authority | 2.10 | 2.75 |
| NB-FLIP canonical metric and publication | 0.53 | 2.62 |
| pressure project | 1.97 | 2.03 |
| sum of all (448 dispatches) | 35.9 | 61.8 |

The largest single kernel late is `advectDeferred{umMergedTiles,umCubicDeferred,
umSolidTwinned}` at 4.85 ms. It is the general solid-aware twin. The fast and
solid-free twins read 0.000, because a scene with any solid skips them
(`umSolidSkip`). The local solid admission that would let them run applies
only with the all-4h coarse solid record. So in a walled scene every deferred
vertex takes the general path, even far from any wall.

### Closed rest (5 mm base domain, 120 steps, clean, 2 interleaved repeats)

| arm | median ms (steps 21-90) | p95 | step 120: max speed, volume, drift |
|---|---|---|---|
| bulkTier off, r1 / r2 | 13.3 / 13.4 | 14.7 / 14.9 | 0.013092, 2949120.0007, 0 |
| bulkTier 16h, r1 / r2 | 14.3 / 14.7 | 16.4 / 16.2 | 0.013472, 2949119.8728, 0 |

The rest gate holds, with the same values as section 1's 16h arm to every
printed digit. However, **16h is 1.0 to 1.3 ms slower than off at rest** (7 to
10%), and it saves 0 B. Rest per dispatch (16h, steps 90-120, sum 14.4 ms):
global surface volume 1.97 (dilate 0.92), pressure solve 1.77, surface advect
deferred 1.31 (the general solid twin again, over a still surface), census 1.05.

### Measured win: water as the realtime model (inviscid, no tension)

The seep scene authors water's μ = 1.002e-3 Pa s and σ = 0.072 N/m. The user's
model for realtime vast water is inviscid with no surface tension. Interleaved
A/B, scene vs `--water=inviscid`, 3 repeats each (ABABAB), wide seep, 16h,
clean timing:

| window (steps) | scene median / p95 ms | inviscid median / p95 ms |
|---|---|---|
| 5-20 | 37.7 / 44.3 (h 264) | 36.0 / 38.8 (h 324) |
| 21-40 | 45.8 / 54.7 (h 778) | 42.1 / 49.1 (h 834) |
| 41-55 | 57.9 / 66.6 (h 2149) | 55.4 / 62.8 (h 2215) |
| 56-70 | 65.7 / 69.8 (h 2975) | 62.4 / 66.2 (h 2891) |

Fitted at equal h tiles, per repeat:

| fit | scene r1 / r2 / r3 | inviscid r1 / r2 / r3 |
|---|---|---|
| at 1000 h tiles (ms) | 47.7 / 44.8 / 46.4 | 43.7 / 42.3 / 42.9 |
| at 3000 h tiles (ms) | 68.4 / 66.0 / 68.2 | 63.8 / 63.2 / 63.2 |

The repeat ranges do not overlap. The saving is about 3.3 ms at 1000 h tiles
and 4.3 ms at 3000 (about 7%). Per dispatch (k5 against k4, late), body forces
fall 3.34 to 0.92 ms: capillarity's cacheNormals, cacheCurvature and
forceSlopes no longer run, which is 38 fewer dispatches (448 to 410). The
pressure solve falls 7.01 to 5.44, partly because there were fewer h tiles
(2587 against 3015). Memory does not change (live 2476 / peak 5548 MiB). This
is a scene choice, not a code change: the seep scene file was left alone,
because another agent is gating seep physics against it right now.

### surfacePages: reverted

The slice saved 42 MiB (1.7%). Its saving is capped by `extensionSlots`, it
adds a knob and a fatal path, and it never showed that its time cost was
noise (section 3: one sample each, +2.8 ms). With time now the priority, it
was reverted in main: `model.ts`, `uniform-mixed-frame.ts` and
`uniform-mixed-surface-volume.ts` are back to HEAD, the frame call in
`webgpu-uniform-reference.ts` is back, the tools' `surfacePages` fields are
removed, and `tests/uniform-surface-volume-pages.test.ts` is deleted. The
branch commit 2dd59e34 keeps it.

### BULK16 audit (why it did not land here)

Two read-only audits mapped the readers.

- **Lattice and physics.** Keep `umPageKind` reading BULK16 as SOLID-like, and
  add `umPageBulk`. Seven choke points:
  1. The `umPageWord` SolidIsBox clamp, which aliases a BULK16 word onto
     block blocks-2.
  2. The `umBlockHeld` and `bnWall` exact `==0xffffffff` tests, which must
     become range tests.
  3. The low-face table, filled with 16h face means.
  4. The wall slot, which must carry the BULK16 pressure with V=1.
  5. The coarse-cache halo (`cacheTexel`).
  6. `umSolidValid`.
  7. `umPointBounds`, `Held` and `SpanHeld`.

  Exact sentinel tests that disagree with the range-based `umPageKind`:
  - lattice-address host twins at 664, 679, 682 and 691;
  - coarse-velocity-cache 72;
  - mixed-dynamic 823;
  - velocity-extrapolation (twin 168, WGSL 303);
  - narrow-band-near 74.

  About 18 to 25 more sites read neighbour pressure or velocity past a page
  edge:
  - pressure-velocity 306, 311, 418-420, 435-446, 483 and 558;
  - arena 609-610, 1151-1160 and 1298-1360;
  - momentum 86, 412, 417 and 453;
  - momentum-cache 349;
  - bulk-tier 128 and 141;
  - root-owners 161-165 and 174;
  - authority 454 and 502;
  - solid-refinement 61.
- **Lifecycle.**
  - The `upKind`/`upPageOf` wrap makes BULK16 requests no-ops or
    `UP_FATAL_ALIAS`.
  - The census residency (`lrAsk`) requests every wet tile's reach, so BULK16
    pages are always re-requested. Requests must become stage-aware.
  - `upSuperSlots` would retire super-blocks that hold only BULK16.
  - The moving window excludes BULK16.
  - Surface-volume measurement would lose BULK16 volume, which feeds back
    into the budget.
  - Phi advection reads the SOLID sentinel's far-air phi across faces, so a
    "bulk sentinel block" holding the deep-liquid image is needed.
  - The renderer and overlay read BULK16 as SOLID or AIR.

It is a multi-subsystem change, not a session's slice.

**Correction (2026-10-28).** An earlier version of this section said that
removing the wide seep's 296 whole-16h pages from the 4h per-block sweeps was
worth up to about 12.7 ms a step (43 µs a resident block). That does not
hold.
- The 43 µs is a timer-on, serialised marginal from a width ladder. In that
  ladder, h tiles and surface grew along with the blocks.
- The depth ladder below measures the per-bulk-page cost directly, under
  clean pipelined timing: about 25 µs a whole-16h page. That includes the
  few non-whole pages that come with each one, and works out to 20-25 µs a
  resident page.
- So 296 whole pages cost about 7.5 ms a step in total. That is the upper
  bound for BULK16 there, before BULK16 pays for its own 16h work.

### Depth ladder: what a bulk page costs (measured 2026-10-28)

**Setup.**
- Scene: the closed reservoir-seep rest (throat filled), 5 mm, `bulkTier:16h`,
  `volumeControlSeconds:0`, inviscid with no tension.
- Only the height changes, 1x to 2.5x (water depth k x 0.96 m). The plan area
  and the surface stay the same, so h tiles stay at 54 and there are no
  particles.
- The scene's `heightScale` and the probe exist only in a scratch snapshot.
  No lib or scene file in main changed. Copies are in
  `artifacts/nbflip-16h-perf/deep-ladder/`.
- Each pool is the build seed plus 4 blocks (164, 224, 284 and 344).
- The 6 GiB requested cap stops the ladder at 2.5x. At 2.75x (384 blocks,
  5.8 GiB) the run fits, but the bulk tier there does not settle (bulk error
  246, whole pages fall from 198 to 154), so that point was dropped.

**Timing.**
- Mode: `timing`, `--pipelined` (as the app). Each value is the median ms
  over steps 62-119, leaving out census steps 90 and 91. By step 60 the bulk
  tier has formed and the page counts are flat.
- Repeats were interleaved: r1 ran shallow to deep, r2 deep to shallow. r2
  skipped 1.5x to keep GPU use light.

| height | whole-16h pages | resident pages | r1 ms | r2 ms | step-120 drift |
|---|---|---|---|---|---|
| 1x | 66 | 116 | 12.17 | 12.18 | 0 |
| 1.5x | 108 | 164 | 13.41 | - | -9.0e-7 |
| 2x | 150 | 212 | 14.26 | 14.29 | -2.28e-5 |
| 2.5x | 192 | 260 | 15.04 | 15.79 | -9.01e-5 |

**Fit.**
- Step time against whole pages:
  - r1: 10.81 ms + 22.5 µs a page.
  - r2: 10.27 ms + 28.1 µs a page.
  - Both repeats together: 10.53 ms + 25.3 µs a page, rms 0.21 ms.
- Against resident pages it is 19.7 to 24.6 µs a page.
- h tiles do not vary in this ladder, so they could not be used as a second
  regressor. Their cost comes from the width ladder above (10.6 µs an h tile).

**Per-pass split.**
- Method: kernel timer from step 61, `timing` mode, at 1x and 2.5x
  (`k-h1`, `k-h2.5`). With the timer on, the dispatch sum is 11.73 ms at 1x
  and 15.66 ms at 2.5x. That is +3.93 ms for 126 more whole pages, about
  31 µs a page timer-on.
- The slope is spread thinly: no pass exceeds about 4 µs a page. The timer's
  65.5 µs quantum makes each pass's figure accurate to about 0.5 µs a page.

| pass | µs per whole page (timer on) |
|---|---|
| pressure project | 4.2 |
| pressure arena solve (whole V-cycle) | 2.6 |
| local speed certificate | 2.6 |
| pressure rhs | 2.1 |
| dynamic ownership census | 2.1 |
| surface advect (advectWalls) | 1.6 |
| ownership residency (pageMark) | 1.6 |
| shared frame plan | 1.6 |
| global surface volume (dilate) | 1.6 |
| bulk faces, momentum tiers, 4h sampling cache, bulk selection, NB-FLIP transfer, layout build, extension | 1.0 each |

- The arena's 2.6 µs matches the earlier V-cycle knock-out (about 3 µs a
  page; `nbflip-step-time.md` section 6).
- So a bulk page is not expensive in any one pass. It costs about 25 µs
  because about 20 per-block 4h sweeps (projection, RHS, the speed
  certificate, census and residency, surface volume, transport and
  extension) each still visit it.
- Only removing whole pages from all of these sweeps together pays. That is
  BULK16, and it cannot be built one pass at a time.

**Projection to a vast scene (100k bulk pages).**
- **Time today:** 100,000 x 25 µs is about 2.5 s a step for the bulk alone.
  Realtime is out of reach without BULK16.
- **Time with BULK16:** a whole page would carry 8 16h rows instead of its
  4h rows, so per-page sweeps should fall about as rows do. Not measured;
  the 25 µs only bounds what is removed.
- **Memory today:** a bulk page keeps its whole 4h block. At section 1's
  running marginal of 1665.5 KiB a block, that is about 159 GiB.
- **Memory with BULK16:** at a few KiB a page (8 regions of pressure, six
  face velocities, flags and the bulk face texels; 2 to 4 KiB), it is about
  0.2 to 0.4 GiB.
- **Construction memory in this ladder:** live memory right after
  construction rose 14.9 MiB per pool block (2547 to 5222 MiB, pools 164 to
  344). Most of it is per-pool-block NB-FLIP buffers: particles A/B at
  1.5 MiB each, cell starts and samples at 1.6 MiB, counts and metrics at
  0.9 MiB. Stage scratch adds 2.0 MiB and the arena pools 1.7 MiB.
  - This construction figure, not the 1665 KiB running marginal, is what
    capped the ladder at 6 GiB.
  - The running set at end of run was not measured here.

**Physics check: rest drift is not 0 deeper than 1x.**
- Step-120 drift goes 0, -9.0e-7, -2.28e-5 and -9.01e-5 from 1x to 2.5x, and
  -1.70e-4 at 2.75x.
- Each run loses it between steps 1 and 60 and is then flat.
- At 2x with `bulkTier:"off"` the drift is the same to every digit
  (-2.2803e-5), so it is not the bulk tier. It is a deep-water rest
  correctness failure in the shared 4h path.
- Reported, not patched; it needs its own investigation.
- The timings above are still a valid measure of cost.

### Recommendations

1. Author the realtime scenes inviscid with no tension (seep and the vast
   presets): -3.3 to -4.3 ms measured on the wide seep. Do it once the
   seep-volume campaign that gates against the current scene has finished.
2. Do not make `bulkTier:"16h"` the default yet. At rest it costs 1.0 to
   1.3 ms more than off, and it saves 0 B until BULK16 storage lands.
3. Next time targets, in order:
   - BULK16: about 25 µs a whole page clean (depth ladder), so up to about
     7.5 ms on the wide seep and about 2.5 s a step at 100k bulk pages. The
     audit above is its reader map.
   - A locally certified solid-free path for `advectDeferred` in walled
     scenes without the coarse solid record (4.85 ms late, 1.31 ms at rest,
     all on the general twin).
   - The pressure arena V-cycles (6-7 ms; already convergence-gated per
     launch, so the cost is launches over rows).
