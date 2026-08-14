# Sparse uniform (CM12) — research and implementation handoff

Date: 2026-08-14. Every repo claim below was re-verified at HEAD `1f98eeb`
("Split the fluid methods into fluid-core plus three method packages") plus the
staged phase-7 sweep (`lib/` → `lib/core/`). Older docs cite pre-split paths
and stale schedules — where this doc contradicts them, this doc is the one that
was checked against code. Known-stale claims are flagged inline.

Goal (Peter, 2026-08-14): a sparse implementation of the uniform
(Chentanez–Müller 2012 mass-conserving) method with

1. **zero work for empty tiles/bricks** — no dispatch, no bandwidth, no memory;
2. **GPU optimal** — compact worklists, indirect dispatch, no mid-frame
   readbacks, coalesced brick layouts, no per-sample pointer chasing;
3. **reduced resolution for bricks with low activity or far from camera.**

Related prior work this doc builds on and supersedes where they overlap:
`docs/papers/wu-2018-gvdb-flip-uniform-applicability.md` (the sparse-residency
design study — still sound, paths stale), `docs/uniform-region-fluid-handoff.md`
(the block-structured-AMR region plan — its pressure-seam analysis and fidelity
policy carry over; its few-large-regions geometry is replaced here by per-brick
granularity), and the b09d868 active-region experiment (in-tree, defaulted off).

---

## 1. Verified state at HEAD

### 1.1 What the uniform lane is

Live files: `lib/methods/uniform/` — `method.ts` (registration), `harness.ts`,
`webgpu-uniform-reference.ts` (1,922 L, advance graph + allocations),
`webgpu-uniform-reference.wgsl.ts` (1,561 L, 25 live entry points),
`webgpu-uniform-velocity-extrapolation.ts/.wgsl.ts` (FIM + CM11b hierarchy),
`webgpu-uniform-pressure-multigrid.ts/.wgsl.ts` (CM11a LCP multigrid),
`uniform-host-allocation.ts`, `uniform-paper.ts`, `uniform-diagnostics.ts`.

Per-advance pass counts (default params, no bodies, semi-Lagrangian):

| Lattice | Sec 3.3 ext. | transport+sharpen+γ | multigrid | misc | **total** |
|---|---|---|---|---|---|
| 64×32×64 | 47 | 10 | 1,422 | 3 | **≈1,482** |
| 128³ | 51 | 10 | 2,641 | 3 | **≈2,705** |
| 256×128×128 | 51 | 10 | 2,641 | 3 | **≈2,705** |

The multigrid is **97% of all passes** (3 Full + 4 V cycles, 6+6 PRBGS sweeps,
one `beginComputePass` per dispatch; `mgSolveCoarsest` is a single resident
`[1,1,1]` workgroup with a `workgroupUniformLoad` convergence break). Stale-doc
corrections: γ diffusion is **3 passes** (1 iteration × x/y/z Jacobi-gather,
`UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS = 1`), not the "42 passes" in the
implementation map; pre/post sweeps are **6**, not 4; the FIM ceiling is
**16** passes, not `max(dims)`.

Memory (dense, all fields):

| | 128³ | 256×128×128 |
|---|---|---|
| velocity (4× rgba32f) + boundary + transport | 205 MB | 409 MB |
| scalars ρ/γ/p/surface (r32f pairs) | 67 MB | 134 MB |
| `conditioningScratch` (β + ρ/γ deficits, atomic i32) | 25 MB | 50 MB |
| Sec 3.3 scratch (6× haloed rgba32f) + hierarchy | 221 MB | 438 MB |
| CM11a pyramid (13 textures/level, 88 B/haloed cell) | 232 MB | 459 MB |
| **total** | **≈751 MB** | **≈1.49 GB** |

≈330 B/cell asymptotically. Everything is a dense 3D texture except six
buffers; `conditioningScratch` is already linear-indexed (the easiest field to
re-map onto bricks). ~69 MB at 128³ is dead in the semi-Lagrangian default
(`velocityC`, `transportB` allocated for MacCormack unconditionally).

Unconditional per-advance overhead regardless of liquid: **≥4 full-volume
`copyTextureToTexture`** (2 per γ iteration at `webgpu-uniform-reference.ts:1189-1192`,
1 volumeB→volumeA, more under rigid coupling) and **3–5 `clearBuffer`s of the
25 MB conditioningScratch**. This is free bandwidth to reclaim before any
kernel changes.

### 1.2 The active-region experiment (b09d868) is the seed, not a rival

It is fully intact at HEAD, defaulted off (`activeRegion: "off"`,
`lib/methods/uniform/method.ts:39`; env kill `FLUID_UNIFORM_ACTIVE_REGION`).
Mechanism: one rolling padded wet AABB, censused by a contention-free two-level
reduction, published as a 176-word buffer whose words are simultaneously the
**indirect dispatch args** for the main grid (offset 13·4) and for **every
multigrid level** (offset (16+10·L+3)·4). ~24 main entry points take an
`activeId(gid) = gid + origin` offset; every multigrid dispatch except
`mgSolveCoarsest` goes indirect; the FIM consumes the same records. Padding is
`ceil(maxSpeed·dt/h) + ceil(D) + 4` with the max wet face speed reduced per frame.
Copy-before-indirect discipline (census writes `activeScratch`, host copies to
`activeRegion` + `activeDispatch`) exists because WebGPU forbids a writable
storage buffer doubling as the same pass's indirect authority.

What it lacks, and what this plan adds: it is a **single box** (a dam plus a
distant droplet costs the union), it gates neither the texture copies nor the
scratch clears, and it cannot reduce memory. Its ABI, `activeId` plumbing,
per-level records, census reduction, padding rule, and diagnostics
(`uniformActiveRegion*` on `GPUEulerianInfo`) all carry forward.

### 1.3 Reach classes (what an apron can and cannot cover)

At paper dt = 1/30 s the lane runs CFL up to ~25. Stage reach in fine cells:

- **Class A — fixed, ≤6:** extrapolation authority (1), γ diffusion (1/axis),
  sharpening compute (1) and mass-return trace (≤5), solid excess (2), forces
  (1), projection (1), per-level multigrid stencils (1 coarse cell), Sec 3.8
  blur (6). A 6-cell apron — or direct neighbor-page reads — covers all.
- **Class B — CFL-unbounded:** density backward trace / forward scatter / γ
  gather (**≈26 cells** at CFL 25), velocity RK2 trace (≈25, hard ceiling 48),
  FIM front (2-cell valid band but 16-cell dependency diameter), CM11b
  hierarchy fill (**global**), inflow swept plug (≈C cells). No apron works;
  these need swept-bounds activation before transport plus scatter-through-
  directory, and the hierarchy fill needs a two-tier replacement (§4.6).

The fixed-point scatters make this tractable: β, ρ/γ deficits, and sharpening
deposits are **order-independent atomic i32 adds into a linear buffer** — a
brick decomposition only changes the *index translation*, not the arithmetic,
so brick-gated transport can be bit-identical to dense.

### 1.4 Renderer, camera, tests

- **Renderer already has a sparse path.** The uniform lane currently publishes
  dense `volumeTexture`/`surfaceFieldTexture` and gets full-volume marching-
  cubes classification; but `GPUSolverInstance.globalFineLevelSetSource`
  (`lib/core/method-contract.ts:220-223`) drives brick-page indirect extraction
  (`lib/core/webgpu-water-pipeline.ts:2333-2344`, ABI
  `lib/core/fine-levelset-brick-abi.ts`). A sparse lane publishes bricks.
- **No camera→sim channel exists** (verified negative). But camera and solver
  are siblings in the render worker (`lib/core/webgpu-renderer.ts:159-165` vs
  `:2432`) — wiring is a local change, no IPC change. Reusable screen-space
  predicate: `lib/svo/svo-screen-space-termination.ts` (angular threshold,
  DPR-safe). Module rule: methods may not import `lib/svo`, so the predicate
  moves/duplicates into core.
- **Zero uniform unit tests at HEAD** — 1f98eeb deleted all ten
  `tests/uniform-*.test.ts` (including both active-region regex tests). The
  surviving oracles are eight Dawn smoke lanes (catalog
  `lib/harness/scene-webgpu-smoke-catalog.ts`; incl.
  `uniform-detail-first-step` on `brick-quad-dam-break`, added by b09d868) and
  `tools/benchmark-symmetric-expansion-comparison.ts` with D4 symmetry
  1e-3/1e-4, stage-mass ledger 0.002 cells / 1e-6 rel, γ∈[0,2.5] gates.

### 1.5 House patterns to copy (all verified at HEAD)

| Pattern | Where | What to take |
|---|---|---|
| 7-word workset header whose words 4–6 ARE the indirect record | `lib/methods/octree-shared/webgpu-octree-worksets.ts:9-30` | worklist ABI; `octreeDispatchForCount` folds a count into a ≤65,535/dim triple, empty ⇒ (0,1,1); RFC-1982 epoch compare |
| Fail-closed double-buffered brick residency | `lib/core/webgpu-fluid-brick-residency.ts` | RESIDENT/CORE/HALO/ACTIVATED flags, `retireAfterFrames` hysteresis, hash page table + per-workgroup claim memo, 3×3×3 tile ring, generation commit that rejects wholesale on pool saturation |
| Direct-map page table + typed missing defaults | `lib/methods/octree-shared/webgpu-octree-fine-levelset-transport.wgsl.ts:468,804` | `pageOf(key)` array lookup (no hash for bounded key spaces), triple validation, missing neighbor = typed virtual boundary, never a fault |
| Control-block A/B publication | `lib/methods/octree-shared/webgpu-octree-owner-pages.ts:47-65` | 16-word arena header, status bits, stable physical page IDs while logically resident |
| Occupancy summary in spare bits | `lib/svo/svo-brick-occupancy.ts:11-25` | packed macrocell mask + min/max bounds per brick, lifecycle bits orthogonal to content bits |
| Coarse min/max classification | `octree-power-coarse-levelset` `minimumPhi/maximumPhi` | exact all-wet/all-dry brick rejection — the model for a per-brick ρ min/max summary (none exists on the uniform lane today) |
| FIM prepare-pass gating | `lib/methods/uniform/webgpu-uniform-velocity-extrapolation.ts:330-395` | 1-thread pass writes indirect args from an atomic counter; termination = args go to zero; parity + executedPasses readback |

197 `dispatchWorkgroupsIndirect` sites in 53 files — this is the house idiom,
not an experiment.

---

## 2. What sparsity can honestly win (and where it can't)

- **Small scenes are launch-bound, not cell-bound.** The 32×16×32 benchmark
  frame is ~55% smoother passes / ~25% FIM at ~12.3 ms with ~700 sub-ms
  dependent passes; brick gating does not reduce pass count and adds census
  passes. Mini-dam 64³ is a **parity control, not a perf target** (cf. the
  occupancy-mask null result: +0.64% on a cache-resident lane).
- **Large and mostly-empty scenes are the target.** Initial wet-page occupancy
  at B=8 (from the GVDB study's audit): garden dam migration **0.1%**, garden
  hose **0.2%**, large-power dam 1.8%, CM12 Figure 9 23%, 128³ dam 36%, ocean
  seiche 75% (deliberate negative control — pressure must keep the wet bulk).
  Per-cell work scales with residency; memory scales with residency; the dense
  1.49 GB at 256×128×128 is what currently blocks bigger worlds at all.
- **Resolution LOD has a break-even.** DCGrid is 2.2× memory and 1.7× *slower*
  than dense until ~50–60% of cells coarsen away, and liquid LOD concentrates
  fine cells at the surface — where a splashy scene's cells already are. Depth
  and distance are where LOD pays (Narita's 54× DoF cut came from collapsing
  the water column, not the surface). Hence the sequencing below: residency
  first (no break-even, no numerics change), resolution second.

---

## 3. Design decisions

**D1 — Two stages, one architecture.** Stage one is *work sparsity in-lane*:
upgrade the active-region machinery from one AABB to compacted brick
worklists while storage stays dense. Bit-identical by construction (same
cells, same atomics; only launch shape changes), A/B-able behind the existing
`activeRegion` param, and it delivers the "zero work for empty bricks"
dispatch/bandwidth half on every bounded scene. Stage two is *storage
sparsity as a separate registered method* (`sparse-uniform`), per the GVDB
study's oracle argument: a missing page must never be able to hide the same
state in both the method and its reference. The dense lane stays the
authority; the worklist ABI, census, and kernels built in stage one carry to
stage two unchanged.

**D2 — Brick geometry: 8³ storage pages, 4³ work tiles.** Storage/residency
granularity 8³ (owner-page precedent, occupancy audit favors 8 over 16/32 on
the target scenes); dispatch granularity stays the existing 4³ workgroup, 8
tiles per page. Two granularities per the GVDB/DCGrid lesson and the existing
brick/tile worklist split in `webgpu-fluid-brick-residency.ts`. Brick width is
a measured parameter, not a theorem — the shadow census (P0) sweeps 8/16.

**D3 — Buffer-backed SoA pools, no persistent aprons.** Field families get
separate pools (transport ρ/γ + deposits; velocity + faces; pressure pyramid
per level; presentation) because their support sets differ. Formats today are
mostly unfilterable and already manually interpolated, atomics need buffers,
and a one-cell apron on 8³ pages costs 1.95× storage on every channel — the
node-mip lesson says don't pay that without a measured win. Neighbor access =
direct page-table read (`pageOf` pattern), resolved **once per workgroup per
neighbor page** into shared memory where a stage is stencil-heavy — never a
per-sample hash walk (Peter's no-indirect-inner-loop lens). Missing page =
typed default: ρ=0, γ=1, FIM distance=∞, p_min per constraint class, velocity
by boundary authority — never a generic zero.

**D4 — MAC face ownership is unchanged.** Positive faces owned by the
lower-coordinate cell, negative domain planes stay in the boundary buffers.
A cross-page face is owned by the lower page; both divergence rows read it;
retirement is blocked while either neighbor needs it. No duplicated faces.

**D5 — Residency is transactional and predictive.** Required set per accepted
step = wet pages ∪ Class-A stencil closure (1 ring at 8³) ∪ Class-B swept
bounds (`ceil(maxSpeed·dt/h)+D+4` — the b09d868 padding rule, now per-brick
directional rather than global) ∪ inflow plug ∪ moving-solid swept bounds ∪
pressure closure at every sparse level. Build candidate → allocate stable IDs
→ initialize from typed defaults → validate capacity/receipts → atomic flip;
saturation rejects the generation wholesale (house pattern). Tripwires from
the region handoff apply verbatim: empty-accept of a non-empty scene is fatal;
bounded-retries-then-fatal; loud capacity receipts.

**D6 — The pressure pyramid goes sparse only at the finest levels.** Levels
0–2 (the bulk of cells and bytes) get per-level brick worklists with parent
closure; deeper levels stay dense (they are small — at 128³ level 3 is 16³
haloed) and `mgSolveCoarsest` is untouched. The per-level indirect ABI already
exists (words 16+10·L); stage one re-points it at per-level brick lists. LCP
semantics (min-clamp in both colors, constraint transfer depth 3, double-single
coarsest) are not modified — this is storage/launch gating, not a new solver.
Red/black ordering across page boundaries is preserved by direct neighbor
reads, NOT by cached halos (a stale halo silently degrades PRBGS to block
Jacobi — an un-blessed numerical change).

**D7 — γ-diffusion ping-pong ABI is reworked, not gated.** The two full-volume
copies per iteration exist only to realign ping-pong textures. Brick pools
make the copy a pointer swap per page; this is the single cheapest bandwidth
win and it must not survive into the sparse lane.

**D8 — Presentation goes through the existing sparse extraction path.**
Publish `globalFineLevelSetSource`-shaped brick pages (interface bricks only)
instead of a dense surface texture; keep a small dense fallback while
migrating. Diagnostics become per-brick partial reductions + one compact
final reduction; missing-page errors are reported separately from numerical
zeros.

**D9 — Resolution LOD is per-brick level with 2:1 grading, policy from the
papers, numerics from this repo.** Adopt: DCGrid's budget machinery (per-level
brick budgets, swap-based k-selection, move limits) with the CPU partial sort
replaced by a GPU 256-bin histogram select over 8-bit scores (no readback);
Ando–Batty Eq. 39 sizing (curvature + velocity-Jacobian-diagonal channels,
γ_φ=4, γ_u scene-scaled) reduced per brick by **max**, never average; advected
exponentially-decaying demand (T1=0.9, T2=0.01 s) as the hysteresis; volume-
weighted-max propagation + brick dilation as the 2:1 closure. Reject: DCGrid's
level-agnostic apron (breaks the donor-sum invariant — mass created at every
seam), injection prolongation (surface stair-steps), and its no-grading
free-for-all. Level transfers are **integer-domain mass** operations: coarsen
= exact 8→1 fixed-point sum; refine = 1→8 divide with a D4-symmetric
deterministic remainder rule. All Δx-dependent thresholds (γ clamp/reset, φ =
−(ρ′−0.5)·min(h), θ floor, volume-correction gain) are re-derived per level,
never copied.

**D10 — LOD scope ladder (the honest part).** The composite LCP row at a 2:1
seam — symmetric, single authoritative subface flux, complementarity set
independent of which side is coarse — is **unsourced in the literature**
(DCGrid never states one; Narita has SPD-variational but no LCP; Ando–Batty is
CPU-octree). It remains the load-bearing experiment, exactly as the region
handoff concluded. So LOD ships in three rungs, each independently valuable:

1. **Rung 1 — whole-component LOD, no seams.** A distant/calm connected
   component runs entirely at one coarser level (its own bricks, its own
   pyramid), optionally at reduced cadence. No composite operator at all.
   This is most of the camera win for multi-body scenes.
2. **Rung 2 — submerged interior coarsening.** Interface bricks stay finest;
   only deep bulk coarsens; 2:1 seams are steered below the surface band
   (region-handoff submerged-seam policy at brick granularity). The seam
   operator is exercised only on smooth submerged pressure — the easiest
   regime — and the hydrostatic-parasitic-current gate is the blocker.
3. **Rung 3 — far-field surface coarsening.** Coarse interface bricks far
   from camera. Requires the full seam + free-surface-complementarity story
   plus the sharpening/transport level transitions. Gated on rung 2's
   operator surviving its gates.

---

## 4. Architecture

### 4.1 Logical space and directory

Global integer lattice unchanged (scene-authored finest `dx`). Page key =
`(i,j,k) >> 3`. Directory = **direct-map array** `key → physical page ID`
sized to the scene's bounded page box (a 256³ scene is 32³ pages = 128 KB of
u32 — dense directory is fine; two-level/hash is a post-MVP escape hatch for
unbounded worlds). Triple validation on read (id < capacity, key match,
generation match) per the fine-level-set pattern.

### 4.2 Worklists

One 7-word-header workset per consumer shape (house ABI): resident pages;
wet pages; interface pages; per-stage tile lists (transport, sharpening,
FIM-front, per-mg-level, presentation); activation/retirement candidates.
Words 4–6 are the indirect record; empty ⇒ (0,1,1). Kernel-side decode:
`tileId = worklist[7+wg.x]` → page origin + local offset replaces
`activeId(gid)`. Per-brick census summaries extend the b09d868 32-byte
workgroup summary with a packed occupancy word (macrocell mask + min/max ρ
class, `svo-brick-occupancy` packing) so the finalize pass classifies
empty / wet / interface / halo per brick in one reduction.

### 4.3 Class-A stages (bounded stencils)

Dispatch over tile worklists; neighbor pages resolved workgroup-uniform;
missing neighbors read typed defaults. Bit-identical target: same cells
processed as dense for every resident cell, and non-resident cells provably
contribute nothing (their dense values are the typed defaults — this is the
invariant the shadow census must certify before stage two retires dense
storage).

### 4.4 Class-B transport (CFL-unbounded)

Traces *read* arbitrary positions — reads go through the directory (or the
still-dense textures in stage one). Scatters *write* — every deposit address
is translated `linearIndex → (page, offset)`; a deposit to a non-resident page
is **a swept-bounds activation bug, counted fatal in debug and ledgered in
release** (mass is never dropped; the deposit lands in a per-page overflow
ledger applied next step, and the counter is a red gate). The activation
pre-pass computes per-brick max outgoing speed (the census already reduces max
face speed) and dilates the resident set directionally before transport clears
the deposit buffers.

### 4.5 Sparse multigrid

Per-level brick lists with parent closure (a fine page's parent page is
resident at every level down to the dense cutover). Smoother/residual/
restrict/prolong dispatch over level worklists; levels below the cutover run
dense exactly as today; `mgSolveCoarsest` unchanged. The pyramid's 13
textures/level become per-level page pools in stage two; in stage one they
stay dense textures and only the *dispatch* is gated. Acceptance is the
existing projected-residual telemetry — unchanged tolerance, plus a new
"sparse rows == dense liquid rows" census gate.

### 4.6 Velocity extension (the one redesign)

FIM front: keep the counter/parity/termination protocol, upgrade the binary
box gate to a compacted front-tile worklist (the front is ≤2 cells wide — the
all-or-nothing dispatch is the known F8 waste). The CM11b hierarchy fill is
the only global stage and has no sparse formulation; replace with two tiers:
brick-local extension for the ≤2-cell accurate band (worklist-driven), plus
the existing small dense mip chain (9.6 MB at 128³ — it can simply stay
dense) for far-field fill. Far-field air velocity feeds only characteristic
tails through mostly-empty space; the re-bless bar is the standard energy/D4
indicators.

### 4.7 Presentation and diagnostics

Interface-brick pages published through the `globalFineLevelSetSource` path;
Sec 3.8 blur runs on interface tiles ±6 (or is folded into extraction).
`reduceDiagnostics` becomes per-page partials + one compact final pass.
Keep the 17-entry `UNIFORM_ADVANCE_PHASE` seam map and `FLUID_GPU_PASS_TIMESTAMPS`
attribution intact — the perf ledger must stay comparable.

### 4.8 LOD machinery (P4+)

Per-brick level field (2 bits), 2:1 graded by construction (dilation closure);
per-level budgets + histogram k-selection swap (D9); sizing channels computed
on interface/wet tiles only, propagated by volume-weighted max ≤5 rings,
demand advected with decay; camera channel = required level such that a cell
projects ≤ p pixels at the brick's nearest point (p≈2/4/8 presets), combined
as an authored *ceiling* (min), frustum/occlusion falls to floor level;
wavelength floor (≥8–12 cells per dominant wavelength) overrides camera
coarsening so an incoming swell is not degraded before it arrives. Level
changes: one rung per epoch, coarsen needs sustained quiet (8 epochs seed),
refine reacts in 1; integer mass transfers with receipts (D9); velocity
refine initializes children to parent flux + uniform correction so
`Σ area·u` matches; pressure warm-starts, never transferred as truth.

---

## 5. Camera channel

Widen the worker frame call, not the IPC: pass `CameraState` (or a derived
`{position, tanHalfFov, viewportHeight, frustum}` record) alongside
`advanceTo`, or add `GPUSolverInstance.setCameraHint(hint)` called from the
draw path (`webgpu-renderer.ts:159-165` already has both in scope). The hint
is advisory and generation-stamped; the solver samples it at regrid epochs
only, so an unset hint (harness/Dawn lanes, benchmarks) means "no camera
channel" and bit-stable behavior. Screen-space predicate ports from
`svo-screen-space-termination.ts` into core (methods cannot import `lib/svo`).

---

## 6. Phases

### P0 — Baselines, shadow census, minimal net (no numerics changes)

- Re-establish a uniform-lane regression net: re-create the two active-region
  regex tests' assertions in the current style, plus one Dawn numeric test
  that runs one advance with `activeRegion:"on"` vs `"off"` and byte-compares
  the stage-audit textures (the machinery exists; the test died in 1f98eeb).
- Capture pass-timestamp + wall baselines: 128³ dam, garden hose, brick-quad,
  ocean seiche (negative control), at `activeRegion` off AND on.
- **Shadow residency census** on the dense lane (the GVDB doc's Phase 0, now
  cheap because the b09d868 census already reduces per-workgroup summaries):
  per advance, per stage-class, publish resident/wet/interface/halo brick
  counts at B=8/16, fill ratios, modeled bytes, workgroups avoided vs dense,
  max characteristic reach. Wire into the diagnostics panel next to the
  existing "Work box %" readout.
- Exit: census numbers on the target scenes confirm (or kill) the ≥4×
  work / ≥2× bytes margins; baselines archived.

### P1 — Work sparsity in-lane (brick worklists, dense storage)

- Replace the single-box words 7–15 with brick worklists + `octreeDispatchForCount`
  folds; `activeId` → tile decode; gate all Class-A kernels + transport +
  finest 3 multigrid levels + FIM front tiles.
- Gate/eliminate the unconditional copies and clears (γ ping-pong pointer
  swap where possible in dense storage; scoped clears of conditioningScratch
  by wet-tile list).
- Keep `activeRegion:"box"` as a param value; add `"bricks"`; dense default.
- Exit: **bit-identical** to dense on the smoke lanes and symmetric-expansion
  (`FLUID_AWAIT_EVERY_STEPS=1` protocol; D4 gates green — the census and any
  padding rules must be D4-symmetric by construction); ≥2× wall on garden
  hose / large-power dam at 128³-class sizes; no regression >2% on mini-dam
  (launch-bound floor accepted and documented).

### P2 — Storage sparsity: `sparse-uniform` method

- New package `lib/methods/sparse-uniform/` registered as a fourth method
  (shared CM12 WGSL bodies factored into a shared zone the boundary checker
  blesses — decide `uniform-shared/` vs core at implementation time).
- Page pools for transport scalars + deposits first (the linear-indexed
  buffer), then velocity/faces, then presentation via the sparse extraction
  path; pressure pyramid still dense-storage (dispatch already gated).
- Transactional residency (D5) with the fail-closed generation commit;
  activation tests from the GVDB doc §14 (slab across seams, diagonal droplet
  through page corners, forced activation at the last trace destination,
  capacity overflow fail-closed).
- Exit: dense-vs-sparse artificial-seam equivalence (insert page seams through
  a dense state, every kernel, byte-compare); mass receipts exact; memory on
  garden hose / large dam ≤ ⅓ of dense; dense `uniform` untouched and green.

### P3 — Sparse pressure storage + big-domain scene

- Per-level page pools with parent closure; dense cutover level chosen by
  measurement; the ocean-seiche negative control must show pressure keeping
  the wet bulk resident (a "win" there is a bug).
- A new authored large-world scene (e.g. 512×128×256-class basin with local
  activity) that dense cannot allocate — the demonstration target.
- Exit: projected-residual parity with dense on shared scenes; the big scene
  runs; bytes scale with residency (report B/resident-cell vs the 330 dense).

### P4 — LOD rungs 1–2

- Rung 1: component labeling over the page graph + whole-component level and
  cadence; sleep/wake with analytic rest state (region-handoff §6 rules).
- Rung 2: per-brick levels, submerged seams only; the composite LCP seam
  operator with the region-handoff §5.5 gate list (symmetry, single subface
  counting, exact-on-linears, hydrostatic no-parasitic-current as the
  blocker); integer mass transfer receipts; sizing channels + budget ranker.
- Exit: seam-sweep front-trajectory collapse (seam position and coarse level
  swept); hydrostatic seam current bounded and non-growing; refine/coarsen
  ×1,000 cycle with exact mass and no energy drift; budget over-demand
  degrades deterministically with published attribution.

### P5 — Camera + rung 3 + presets

- Camera hint channel; far-field surface coarsening (gated on P4 gates);
  presets off/quality/balanced/performance; brick/level/priority overlay in
  the diagnostics panel; freeze-topology debug action.

---

## 7. Acceptance summary

Hard gates (never relaxed): exact fixed-point mass across pages, transfers,
levels, sleep/wake (receipted); typed-missing-default equivalence (absent page
≡ dense default, certified by artificial-seam byte tests); fail-closed
transactions with the region-handoff tripwires; D4 symmetry at parity with the
dense lane's own disclosure (brick partition and remainder rules must be
D4-symmetric); projected-residual and divergence contracts unchanged;
`mgSolveCoarsest` and LCP semantics unchanged until a re-bless says otherwise.

Value gates before claiming success: garden-hose-class scene ≥4× awake-cell
reduction and ≥2× GPU frame at matched plausibility; a scene dense cannot
allocate runs; N-body scene scales with awake bricks only; mini-dam floor
regression bounded and documented.

Known reds not to inherit: the canonical 250-step uniform arm's 48 D4
failures are pre-existing (proven twice); the losasso arm's cutover crash;
the CPU suite (~124 red at HEAD).

---

## 8. Risks

| Risk | Answer |
|---|---|
| Launch-bound floor: gating adds census passes, small scenes regress | accepted + measured in P1 exit; box mode retained; sparse targets are large scenes |
| Scatter into non-resident page loses mass | swept-bounds pre-activation + overflow ledger + red counter; debug-fatal |
| Per-sample directory walks (Peter's lens) | workgroup-uniform page resolution, direct-map array, claim-memo pattern; no hash in inner loops |
| Stale cross-page halos turn PRBGS into block Jacobi | direct neighbor reads for the smoother; halos only where a stage is snapshot-based already |
| Worklist/pool capacity | preallocated, fail-closed generation reject, published receipts; never partial publication |
| D4 breakage from brick census/padding | symmetric construction + the existing 1e-3/1e-4 benchmark as the tripwire; exact-reduction (superaccumulator) pattern for any new reductions |
| γ=…, φ=…, defaults wrong in air | typed per-field missing defaults (γ=1 matches the dry reset; FIM ∞; p_min per class) with a dedicated equivalence test |
| LOD break-even not reached (surface-dominated scenes) | rung ladder: rung 1 (components) and rung 2 (deep bulk) pay without the break-even exposure; rung 3 gated |
| 2:1 LCP seam is unprecedented | it is *the* experiment (rung 2, hydrostatic gate as blocker); nothing earlier depends on it |
| Brick-level chatter pumps dissipation | decaying advected demand + 8-epoch quiet + one rung/epoch + trilinear mass-correct prolongation (never injection) |
| Camera hint destabilizes harness lanes | advisory, generation-stamped, unset ⇒ exactly current behavior |
| Docs/paths rot again | this doc cites HEAD 1f98eeb paths; re-verify any file:line against `lib/methods/uniform/` before acting |

---

## 9. Open decisions for Peter

1. **P1 in-lane vs jumping straight to the new method.** Recommended: in-lane
   first (bit-identical, fast payoff, shared ABI); the counter-argument is
   double migration cost.
2. **Brick width** 8 vs 16 for storage pages (census will say; 8 recommended).
3. **Shared-kernel zone** for CM12 WGSL between `uniform` and `sparse-uniform`
   (`lib/methods/uniform-shared/` vs pushing pure snippets to core) — needs a
   module-boundary ruling.
4. **LOD rung 3 ambition** — whether far-field *surface* coarsening is in
   scope this program, or whether rung 1+2 (components + submerged bulk)
   plus cadence covers the goal.
5. **MacCormack** — the second full extension invocation (~50 passes) exists
   only for MacCormack's predicted field; whether the sparse lane carries
   both transport profiles or semi-Lagrangian only (paper-conformance vs
   cost; WP7's history says this is squarely a re-bless).

## 10. References

Repo: `docs/papers/wu-2018-gvdb-flip-uniform-applicability.md` (residency
design study; paths stale), `docs/uniform-region-fluid-handoff.md` (seam
gates, fidelity policy, tripwires), `docs/uniform-mass-conserving-remaining-handoff.md`
(perf anatomy + verification protocol), `docs/method-decoupling-handoff.md`
(package rules), `lib/methods/octree-shared/webgpu-octree-worksets.ts` +
`lib/core/webgpu-fluid-brick-residency.ts` + fine-level-set ABI (house
patterns). Papers in `docs/papers/`: massConservingLiquids (CM12),
raateland-2022-dcgrid (budget/k-selection — adopt policy, reject seam math),
wu-2018-gvdb-flip (+ critique/challenge docs), ando-batty-2020 (sizing §6),
sg2025narita (depth coarsening, damping warnings), losasso-2004, tallCells.
