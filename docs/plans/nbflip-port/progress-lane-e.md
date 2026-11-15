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

## M2, part 1 (2026-11-15): the host, contract and presentation adapters, the stage contract, the observer

**The stage contract** (`brick-stages.ts`, brick-abi.md §6.2):
- Lane F's kernels plug in as `BrickStages`: `move` and `rebin` (part 1); `levelSet` and
  `velocity` (part 2, before `operator`); `tail` (after `project`); `boot`; and F's buffers.
- `validateBrickStages` checks the args regions and buffer names.
- **Registration files.** F owns `brick-fluid-stages.ts`, P owns `brick-pressure-solver.ts`. Both
  export `undefined` until their lane lands.
- The bring-up stage set (`uGravity`) stays for E's tests only.

**The scene adapter** (`brick-scene.ts`). `brickSceneSetup(scene, values)` gives the setup or every
gap. M2 enables high-resolution-dam-break only. For the dam:
- 128³, h = 0.00625 m, dt = 1/60 s, 1936 boot bricks, no scene gaps.
- The box faces on closed walls are continued 2·max(n) cells through the wall, so φ equals
  `damBreakSignedDistanceAtNode` (tested at 6 nodes).
- Density, viscosity (0.001002 Pa·s) and tension pass to F's stage context.

Letters, seep and paddleboats each name their gaps: open top, solids, bodies, sources, initial
volumes.

**The method** (`brick-method.ts`). `engine: "brick"` now calls `createBrickSolver`. It refuses
with the scene gaps plus each unregistered lane, and refuses a bring-up pressure solver.
Today the dam is refused for "lane F's stages" and "lane P's pressure solver" only. Every refusal
says to run the scene with engine "uniform".

**The host adapter** (`brick-solver.ts`, `BrickNBFlipSolver`, a `GPUSolverInstance`):
- whole steps of dt (a partial step returns false), with no substeps;
- `readStats` reads the header (liveF, liveC, peakF, step, PRESSURE_CAP, P's receipt words);
- `assertSimulationHealthy` latches FAIL;
- `info.brickEngine` reports the presentation count and the time of the last copy.

`PressureSolver.receipt` names the header words of iterations and capped counts.

**The presentation** (`brick-present.ts`, ruling 7.1(3)):
- dense `vertexPhi` ((n+1)³, metres), `volume` (six-tet fill) and `openFraction` (1), allocated
  once;
- one workgroup a pool slot, as its own submission after the steps;
- `presentClear` writes the far φ over the bricks a slot showed and no longer holds.

The GPU test steps a [64,32,64] sheet and checks φ, fill (0.3 in the surface cell) and far φ. It
then narrows the sheet and checks that the bricks that left read far φ and fill 0.

**Operator zero rows** (lane P's report, slot contract §6.1, amended). `operator` now runs over
F_ALL, not F_LIQ:
- a live brick outside F_LIQ holds K, DIAG, RHS, X = 0, so the CG's upper-neighbour K read (and
  MGPCG's) never sees a stale value;
- a liquid cell in a brick outside F_LIQ latches FAIL.LIST.

The GPU test seeds 7.5 in every row, steps, and checks 0 at every non-liquid cell (including the
extra vertex row of bricks beyond the lattice).

**The observer** (`tools/nbflip-observe-brick.ts`). The box observer runs over the pool, one
workgroup a slot:
- owners are the h cells of live F bricks;
- liquid is the six-tet fill of the corner φ times the cells;
- u is the mean of the two faces, with walls 0, and a face in a non-resident brick drops out.

`createNBFlipBoxObserver` routes a solver with `brickEngine` to it, a 2-line change to lane G's
file. The bench runs the dam with `--set=engine=brick` once F and P register. The GPU test
matches `boxOracle` over the read-back pool (total, moments, ke, bins, tops, front, wet floor,
column) to fixed-point rounding. `readStats` does not yet give `volumeDrift`/`volumeCellSum`: the
observer's total is the measure.

Tests: brick-engine.test.ts has 9 CPU and 6 GPU tests, and nbflip-observe-brick.test.ts has 1 GPU
test. All 16 GPU tests pass on the check slot (rc 0, no Metal aborts).

**For P (physics parity).** The old engine solves pressure on an all-4h ownership every frame
(`UniformMixedOwnershipTransfer`, uniform-mixed-remap.ts l.319–645). The brick engine solves at h
on F bricks. Matching the old engine's pressure needs either P's coarse level or an explicit
ruling.

**Next (M2, part 2).**
- Tiers: the ABI amendment for h|4h storage, `setMark`'s census criteria (verbatim from
  uniform-mixed-dynamic.ts) and `setTransfer`'s h↔4h remap (verbatim from
  uniform-mixed-remap.ts), with frozen-state parity.
- The dam end to end on F's stages and P's MGPCG.
- Letters' peak re-measured before M4.

## 2026-11-15 — M2 part 2a: the tier ABI (brick-abi.md §2.5)

Coordinator rulings (composite pressure parity, P's review) applied to the layout:
- §2.5 "Tiers: what is stored where". CMASK is the ownership truth (record word 28 and
  FLAGS.OWNER_H copy it, written only by setTransfer's commit). Residency: core + 27-ring + every
  brick the reach box of a non-empty bin overlaps (F). Vertices are coarse-wins, verbatim
  umVertexAuthority; hanging vertices are umVertexFrom4's interpolant. Faces with an h cell on
  either side are h faces in F; 4h|4h plane faces are C faces of the upper tile; the h|4h 4h face is
  derived (umPressureFaceV). New set regions WANT (CMASK_WORDS a C slot), TILE_OCC and TRIG (a word
  an F slot).
- pres channels: KU0..2 (seams whose upper side is a 4h row) and PPHI (the authority's rootPhi);
  FLAGS bits 7–12 reserved for P's active set, 13–15 for E; CREC words 0–26 reserved.
- §6: operator and project move to P (pressure-operator.wgsl.ts); the active set is an outer loop
  around the MGPCG.
- brick-owner.wgsl.ts: nbTileOwnerH, nbVertexCoarse (umCoarseIncident's order), nbVertexPhi,
  nbFacePlane, nbFaceU. brick-owner.test.ts (GPU) checks them against a CPU oracle on a page with a
  4h block: owner and φ exact at every probe (corners, hanging vertices, h|4h planes), stored faces
  exact, interior 4h faces within mix()'s rounding.

Census: the old dam census is resolved (agent report, rule in §3 of that report). Next: setCensus
(part 1 head, WANT + TRIG from the end state), setMark → residency, setTransfer (TLIST, US→U
commit, φ in place, slopes via F's nbfRemap*), with frozen-state parity.

## 2026-11-15 — M2 part 2b: setTransfer (brick-transfer.wgsl.ts)

NB-FLIP's remap on the §2.5 storage:
- `setTransferList` runs markTile and listedFlags verbatim into TLIST.
- `setTransferFaces` and `setTransferCommit` run remapTileFaces, remapBeside, remapSlope,
  remapSample and remapFace on U and US (F: US carries the step), with the low-wall rows. They write
  through an `xfer` scratch, which plays the part of the old second texture.
- `setTransferCells` runs remapTileCells' vertices in place on PHI. Stored vertices move between F
  and C when their storage class changes.
- `setCommitMask` sets CMASK = WANT and the record word 28.

Wall faces hold velocity (lane P: NB-FLIP's walls separate). Planes 0 and N follow the lower-face
convention, so the skirt cell holds plane N. The owner readers now return them instead of 0.

New in the ABI:
- capacity `t` (listed tiles; default 2 f) and the `xfer` buffer;
- header words N_TLIST and N_TLIST_RUN, ARGS.XFER_G1 at 396, FAIL.TIER (bit 11);
- `BrickEngine.run(placements)` for frozen-state work.

brick-transfer.test.ts (GPU) builds a frozen state: a 4h block on the low wall half-refines, and an h
block on the upper wall coarsens. It compares the result against a CPU oracle transcribed from the
old remap:
- 33 tiles listed, 4845 words written (82 tilted slopes, 45 C corners, 64 C faces, 200 wall faces);
- every other word of F and C is unchanged bit for bit;
- CMASK = WANT.

The formulas themselves are F's bitwise nbfRemap* ports. This test checks the assembly, not
old-kernel-vs-new.

Not yet scheduled in the step. NB-FLIP remaps at the frame head before the move, so the transfer's
place comes with setCensus (part 1 head).
