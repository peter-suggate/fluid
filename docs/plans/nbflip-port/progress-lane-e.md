# NB-FLIP brick engine, lane E: engine core (progress)

**Brief.** Lane E owns the brick engine's core in `lib/methods/narrow-band/` (design
`docs/plans/nbflip-structural-2026-11-15.md` §5):
- storage, the directory and the `set*` kernels;
- lists, the schedule and record(), the param ring and submission;
- the host, contract and presentation adapters;
- `solidStamp` and the scene-apply bridge;
- the method-id switch;
- the brick ABI doc.

Milestones: M0(e) the ABI, M0(b) peak counts, M1 the skeleton, then M2 tier transfer, M3 solids
and M6. Scratch: `~/code/me/fluid-scratch/lane-e/`.

## 2026-11-15 STARTED

Base a610cde4. Read the design (all of it, with the §7.1 rulings), the brick arena
(`uniform-brick-arena.ts`), the seep's solver host, layout and WGSL (`lib/methods/seep/`), the
BULK16 page record, NB-FLIP's particle record and the window directory header.

## 2026-11-15 M0(e): the brick ABI, frozen

- `docs/plans/nbflip-port/brick-abi.md`: geometry and staggering (a cell owns its lower faces and
  corner; skirt bricks hold the upper vertex planes), the pools and channels at stride 520, the
  T2o directory (a toroidal page window, then 64 child words a C slot), the 32-word F and C
  records, neighbour words and the seam rule, the `set` buffer (control header, param ring, free
  stacks, marks, candidates, bounds, lists, bins), the indirect arguments and fold rule, the
  particle banks and bins, bindings and the recorded schedule, the brick-set update, and the
  failure bits.
- `lib/methods/narrow-band/brick-abi.ts`: the same as code. `BrickLayout` computes every offset
  for a scene's capacities, and `brickAbiWGSL()` gives a module the constants and accessors.
- `brick-abi.test.ts`: 6 CPU tests of the invariants.
- The prelude compiles on the GPU (check slot, rc 0): a probe kernel using every accessor builds
  its pipeline in 193 ms. The adapter reports 10 storage buffers a stage and a 4 GiB binding.

## 2026-11-15 M0(b): per-scene peaks and authored pools

Measured on the old engine (check slot, one run a scene, rc 0, `tools/gpu/fluid-gpu-run.sh`). The
probe is `~/code/me/fluid-scratch/lane-e/peaks.mts`, and its logs are `peaks-<scene>.jsonl`. Each
step it counts:
- the h tiles the brick reach rule would hold (the F ring);
- the tiles holding liquid (the core);
- the 4h pages over the ring;
- the BULK16 pages;
- the particles.

| scene | steps | F ring (step) | F core | liquid ring | C pages | BULK16 | particles (step) |
|---|---|---|---|---|---|---|---|
| high-resolution-dam-break | 300 | 2562 (60) | 1146 | 4022 | 122 | 3 | 997,037 (40) |
| letters | 240 | 6037 (230, still rising) | 2180 | 6991 | 207 | 0 | 977,923 |
| paddleboats | 300 | 6794 (0) | 3480 | 7399 | 238 | 0 | 316,155 (300) |
| reservoir-seep | 1200 | 1451 (1200, slowly rising) | 541 | 3951 | 169 | 72 | 219,671 |

Authored `scene.fluid.brickPools` (`lib/core/model.ts`, validated):
- f and c are 1.5x the peak ring and page counts, rounded up. The seep gets more, because it is
  still rising at step 1200.
- b is the BULK16 peak plus slack.
- particles is 1.25–1.5x the old engine's peak bank.
- k (cut bricks) stays 0 until `solidStamp` lands (M3), where it is measured.

| scene | f | c | b | k | particles |
|---|---|---|---|---|---|
| high-resolution-dam-break | 3968 | 192 | 16 | 0 | 1,250,000 |
| letters | 9088 | 320 | 16 | 0 | 1,500,000 |
| paddleboats | 10240 | 384 | 16 | 0 | 480,000 |
| reservoir-seep | 2560 | 256 | 128 | 0 | 330,000 |

Open items:
- Letters was still rising at step 240, so it is re-measured over the full take before M2 gates
  it.
- The frozen reach formula (two rings at rest) would have needed about 2.6x these F counts. M1's
  cell-precise reach box (brick-abi.md §5.4, amended) is what the table counts.
- During the letters run the runner reported 3 Metal aborts in other processes. `log show` showed
  none from our runs.

## 2026-11-15 M1: the engine skeleton

**BRING-UP SOLVER.** The pressure slot runs `bringUpCG` (`brick-bringup-cg.ts`), a Jacobi-CG for
M1's tests only (coordinator ruling, 2026-11-15). No M2 gate and no timing claim runs on it. P's
MGPCG replaces it at the same slot (brick-abi.md §6.1) before any scene is gated. M1 makes no
timing claims.

What landed (`lib/methods/narrow-band/`):
- `brick-set.wgsl.ts`: the brick-set kernels.
  - `setMark`: the reach box of §5.4, the class bits, and clearing BOUND_F.
  - `setClaimPages` and `setClaimBricks`: CAS claims, free-stack pops, records, child counts and
    the tile mask.
  - `setLeave` and `setLeavePages`: frees, pushed after every pop.
  - `setRecords`: neighbour words of entering bricks and their neighbours.
  - `setInit`: entering bricks are air and zero.
  - `setLists` (atomic appends; F_LIQ = F_ALL in M1) and `setArgs` (the fold rule, live and peak
    counts).
  - `bootPhi` and `stepEnd`.
- `brick-pressure.wgsl.ts`:
  - `operator`, a pure function of the channels `(cfg, set, dir, vel, phi, pres, cut)` (ruling
    3). It writes RHS, DIAG and K0..2 per §6.1 and zeroes X off the liquid.
  - `project`, the ghost-fluid gradient. It raises BOUND_F.
  - `uGravity`, the bring-up body force (US = U + g dt on open faces with a liquid side). It
    stands in for F's velocity stage.
  - The liquid rule and theta are imported verbatim from `uniform-mixed-pressure-surface.wgsl.ts`
    (Gibou et al. 2002).
- `brick-engine.ts`, the host:
  - capacities-only allocation and the CPU boot candidates;
  - the schedule;
  - `record()`: 4 ring slots, bind groups and cfg uniforms built once;
  - one ring write a step, the two-part submit, and the next step encoded ahead;
  - lagged MAP_READ receipts that latch FAIL on the host;
  - per-kernel compile times, with shader errors reported with source context.
- `brick-readback.ts` (snapshots and channels) and `brick-oracle.ts` (the CPU reach oracle, and
  the Float64 assembly and solve of §6.1). Both are for tests only.
- `brick-method.ts` and `uniform-narrow-band-method.ts`:
  - the `engine: "uniform" | "brick"` parameter on NB-FLIP;
  - unknown engines throw;
  - "brick" refuses every scene and names what is missing (F's stages, P's bodies, M3's solids,
    the host adapter).
- ABI amendments (brick-abi.md):
  - §5.4, the cell-precise reach box;
  - §5.3, the push timing;
  - §5.5, list order, live slots and F_LIQ;
  - §2.3, setInit and FLAGS, and φ bound as u32;
  - §6, WGSL names: `set` is `nbSet`, the `operator` kernel's entry is `nbOperator`, and `active`
    is reserved.

Tests (`brick-engine.test.ts`): 7 CPU and 5 GPU. The full file passes 12/12 on the check slot
(rc 0, 2.0 s, no shader warnings, no Metal aborts in `log show`).
- **4x-wider box (CPU).** The direct dispatches are identical for [64,32,64] and [256,32,64], and
  the set, pools and particles are identical in size. **(GPU)** The same bricks and all 17 list
  counts. Bytes differ by exactly the directory's page window.
- **Cold compile.** All 21 kernels build in 135–510 ms each, under the 30 s rule.
- **Enter and leave against the CPU oracle.** φ is rewritten narrower and taller after a step at
  rest, on a [64,32,64] lattice.
  - Entered bricks are exactly brick row 3. Left bricks are exactly x or z brick 3. The resident
    set equals `brickSetOracle`.
  - Every record checks out: 27 neighbour words, parent, mask, class, and C child counts. The free
    stack and the live slots partition the pool. F_ALL, F_LIQ and C_ALL match, and so do
    N_ENTER and N_LEAVE.
  - The enter and leave lists hold the right slots. Entered bricks are air. The next step is
    still.
- **FAIL_POOL latches.** With f set to the boot count, a taller φ latches POOL_F with REFUSED_F > 0.
  The set stays consistent and full. The lagged receipt sets `engine.failure`, and `advance()`
  then throws.
- **Hydrostatic** (32³, h = 1/32, surface at 10.3 cells, one step from rest).
  - The Float64 system (141 Jacobi-CG iterations, residual 3e-17) is the analytic profile
    X = |g| dt (H − y − ½) h to 8.6e-7 relative, which is f32 φ rounding.
  - GPU X matches it to 1.9e-5 relative. The bring-up CG takes 55 iterations from cold, then
    1–2 with the warm start.
  - X is exactly 0 at air. Max |U| after projection is 1.2e-5 m/s, against the 0.16 m/s of
    g dt it cancels.
- `npm run check:types`: only the 3 pre-existing lane-n errors. `npm run test:unit`: 2248 tests,
  0 failed.

For lanes F and P:
- BOUND_F is raised by `project` (and later `pScatter`) and read by the next `setMark`.
- F's p2g and uAdvect replace `uGravity` at its place in part 2.
- `REACH_MARGIN` (7 cells) waits on F's stated band margin.

Not in M1:
- the scene host, contract and presentation adapters (the refusal names them);
- C and B rows in `operator` and the seam (P extends them in place);
- page-local slot order (a locality measurement).
