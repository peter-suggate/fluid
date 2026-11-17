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

## 2026-11-15: M2 part 2c, tiered residency (behind BrickStages.tiered)

Lane F's all-h dam runs end to end on fluidStages and P's MGPCG (165 steps). It hits POOL_F at the
authored f = 3968 because all-h keeps every liquid brick at h. Tiers fix that, so residency now
follows §2.5 when a stage set sets `tiered: true`. F will set it once its stages read through the
owners. With the flag off, the engine is the M1 all-h engine, unchanged.

- `setMark` keeps F bricks by core (record word 28) plus the in-range 27-ring, plus the reach of a
  non-empty bin. Its SURFACE scan reads h-owned tiles only.
- `setMarkPages` (new, C_ALL G1):
  - 4h liquid tiles, meaning 4h-owned tiles with a corner φ < 0, keep the pages within setMark's
    reach, using the page's largest 4h |U| as the bound;
  - missing pages become page candidates (`NB_KEY_PAGE`, bit 30);
  - it writes `LIQUID_TILES`.
  `setLeavePages` keeps marked pages.
- Bricks claimed in a step start 4h. The boot's claims (cfg.z = 1) start h, because bootPhi writes F.
- `setInit` sets entering C slots to air.

brick-set-tiers.test.ts (GPU, 64^3) is a frozen state after a commit. Two core blocks, one of whose
rings crosses into page row 1. C φ puts liquid 4h tiles whose reach crosses into row 1. The result
matches the oracle exactly: 75 F bricks (9 entered at 4h, 258 left), 18 C pages (9 entered as air),
the CMASK and LIQUID_TILES counts, and FAIL 0.

Coordinator (2026-11-15): before the M2 gate, check the transfer's listing and tile flow against
real old-engine output. Take a pre-remap state from a real dam step and compare it with the old
engine's post-remap state. Pending, after setCensus.

## 2026-11-15: setTransfer against the old engine's real remap (08a59dac)

This is the coordinator's pre-M2-gate check. tools/nbflip-remap-capture.ts takes a frozen capture of
the old engine's frame-head remap at dam step 22, in a 64^3 window at lo = 0. It wraps the device,
with no hook in the old engine. brick-transfer-dam.test.ts then loads the old state into the brick
storage, runs brickTransferSchedule, and compares the result with the old post-remap state:
- Listing: 262 tiles compared (81 refine, 12 coarsen, 169 neighbour), identical, flags included.
- CMASK and word 28: exact.
- U and US (F, C, plane-0 wall rows): bitwise.
- C φ: bitwise.
- F φ: bitwise except 3 words, 1 ulp each. They are new h vertices interpolated on one refined tile's
  face, from compiler rounding.

The window's top tile layer is excluded, because there the brick engine has an artificial wall. No
setTransfer bug was found.

## 2026-11-15: setCensus (fb17d41e) and the part 1 head

The setCensus sub-agent ported the old Uniform_dynamic_ownership_census verbatim on the brick
storage: classify, classifyCoarse, boundCubes, importance, interfaceBits and decideTile, with the
dam policy baked in. It matches a CPU oracle transcribed from the old census on a GPU frozen state,
and everything is exact except score bytes, which are within 1 (f32 fma). It writes WANT and TRIG,
plus the counters in H 224..231. Deviations:
- deep COLD is taken as D = 0 (a CPU test shows the non-deep decisions are unchanged);
- there are no solids;
- the residency clamp (§2.5) applies.

The part 1 head is wired (BrickEngine.scheduleFor):
- the census runs every step, the all-h engine included, because F reads TRIG;
- when tiered, it is followed by setTransfer;
- then F's move and rebin;
- the boot runs the census before F's boot stages, and a tiered boot commits WANT with
  setCommitMask alone. bootPhi now also writes the boot pages' C vertices.

The readers return NB_PHI_FAR (the old empty value, fluidLatticeFarPhi in h) for vertices that no
slot stores, as P found.

brick-set-tiers.test.ts adds a tiered boot of a 30x40x30 column on a 64^3 lattice:
- 42 h tiles (at the column's edges) and 4054 4h tiles; no deep tile is h;
- C φ equals bootPhi's analytic distance;
- census counts: crossing 214, band 70, clamped 28.
The GPU suite passes 23/23 (engine, census, transfer, transfer-dam, set-tiers, owner).

## 2026-11-15: the residency clamp removed (coordinator's question)

The old clamp held a band tile at 4h when its brick or 27-ring was not resident. That is storage
timing, not a decision: the old engine stores every tile at both widths, so it never needed a clamp.
Now WANT is the old band decision exactly. The part 1 head claims the missing ring bricks before the
transfer. A refused pool latches POOL_F/C, and then censusWant latches FAIL.TIER; the tier never
silently stays coarse. Test: a tiered column boot gives 70 h tiles = the census band of 70, with 0
clamped, and a second head on the same state lists no change.

The real old-census capture (WANT/TRIG at dam collapse and splash steps, plus a deep-COLD count) is
delegated to the remap-capture agent.

F's two requests (the same commit):
- The census no longer refuses an authored B pool (the dam's b:16) on the host. A live B slot still
  latches FAIL.UNSUPPORTED on the GPU (censusBegin).
- TRIG bits 8-15 are each tile's thin trigger alone (word 1 bit 17), beside the OR in bits 0-7
  (brick-abi.md §2.5). The census test's oracle checks both.
- GPU: census, set-tiers and engine pass 20/20, with 0 Metal aborts. Unit tests: 1828 pass.

## 2026-11-15: the boot's half kick (F's D5c step-1 question)

F measured step-1 KE at 1.887 J, matching a Float64 oracle, against the reference's 4.133 J. The
old host kicks before its first frame: kickMixedFrame calls frame.kick(p, dt/2), which computes
U <- P(U + g dt/2) with gravity alone and no transport. So the reference's first step carries
1.5 g dt (2.25 x 1.887 = 4.25 J).
- BrickStages gains an optional `kick: { force, after }`. The boot runs force, then the operator,
  the solver and project, then after, before its stepEnd (brick-abi.md §2.5).
- F supplies the force (US = U + g dt/2) and the particle side.
- Schedule unit test added. GPU tests: engine and set-tiers pass 18/18. Unit tests: 1829 pass.

## 2026-11-15: the census head's cost (dam, all-h, 24 steps, profiled from step 13, split passes)

censusFine 8.2 ms/step, decide 1.3, coarse 0.9, importance 0.7; about 11 ms in all. Inside censusFine,
measured by turning each part off in a scratch copy:
- surface displacement (crossing tiles): 3.2 ms. Of that, the corner Laplacian 1.4, the metric 0.6,
  the crossings 0.5, the curvature 0.4;
- depth sampling: 0.15 ms;
- classifying every tile: about 4.7 ms. Most of these are deep interior tiles, which go 4h under the tiers
  and are skipped by the real D, so per the coordinator I leave them until tiered timing.

Landed (bitwise: census and dam-census GPU tests pass 5/5):
- The tile's corners are read from the staged 5³ raw instead of 8 reads a lane: 8.2 to 7.9 ms.
- The corner Laplacian runs on 24 lanes (corner by axis), summed in axis order by the corner's lane:
  7.9 to 6.8 ms.

## 2026-11-15: all-h CMASK is page-wide (lane P's phantom 4h liquid)

Lane P saw 25,832 tiles in resident pages read 4h on the all-h dam (CMASK clear where a brick was absent or
had left), and 296 of them were wet: bootPhi's analytic C φ, which nothing updates all-h. These made rowsC
build native rows and seams. The fix: when not tiered, setClaimPages sets CMASK all-ones, and brick claims
and leaves don't touch it. An absent brick is now h air (NB_PHI_FAR, 0 faces), and censusCoarse stores its
tiles as the identity without a TIER failure. Test: the all-h column boot gives all-ones CMASK on every
page with absent bricks, and a step keeps it. GPU tests: census, dam census, set-tiers, engine, transfer,
transfer-dam and owner pass 27/27.

## 2026-11-15: T4 exits ruling (coordinator, on 633f2255)

The early identity, cheap affine and Euler exits (umEarlyFarAdvected, umCheapAdvected, umEulerAdvected)
run only with UniformSurfaceWorkControl.surfaceEvidence. That control defaults to false and is turned on
only in its own test, so these exits are not part of the reference NB-FLIP. Deviation 2 in
levelset-coarse.wgsl.ts now records them as "old-engine opt-in experiment, not ported", including at M3.
The BULK16 latch (FAIL.UNSUPPORTED 0x4c430003) stays for a different reason: until M3, nbCSlotOfPage
reads a BULK16 page as absent, so a vertex on such a page would silently get a wrong value. The M3 list
gains the COLD-seed deviation (deviation 6: the previous step's seed for a COLD tile), to land with COLD
(deviation 5).

## 2026-11-15: the all-h absent-page guard hits (reader bug, not reach)

Lane P's absent-page guard fired on 2 of 4 all-h dam runs (step 127, page (1,2,1); step 179, vertex (40,64,31),
page (1,2,0)). The cause was hypothesis 3, a reader bug. `nbVertexCoarse` took a tile of a non-resident page as
the 4h authority, so a vertex on that page's upper face interpolated the next page's C corners. All-h, those
corners still hold bootPhi's stale φ (a cell at (63,63,62) read -1.58 where F's corners give 67): a phantom
liquid row. The other two hypotheses were ruled out by instrumented runs: 461 all-h steps over three runs, no
liquid vertex with an absent brick in its 1-ring, least slack 7 cells, and no FAIL.REACH.
- Fix: `nbVertexCoarse` skips a tile whose page has no slot. The oracles in brick-owner, levelset-coarse,
  brick-census, brick-transfer and pressure-composite tests take the same rule. New test: brick-owner, "a
  non-resident page's tile is no vertex authority".
- Latch: `setMark`'s φ scan checks every face vertex with φ < 0 against its 1-ring's neighbour words, which it
  stages once per workgroup. If one is absent it latches FAIL.REACH with that brick's key. No new pass.
- P's guard is now on in both engines (`pressureOperatorKernels(layout)`).
- Validation: 200-step all-h dam (composite, poolF 4913) runs to the end, drift -1.05%, no guard or latch hit.
  GPU suites 29/29 pass; unit tests pass, 0 failed.
- Open, the tiered reach timing (F's step-14 repro). `setMark` bounds before the project, but the old census
  bounds at the frame head with the extended velocity the next trace samples. The old rule would have covered
  the 7.48 m/s jet (about 27 cells against 20 of travel). The port is a part-1 head bound on max |US|, or F's
  pDelta raising BOUND_F with the post-project |U| (brick-abi.md §5.4 "Open").

## 2026-11-15: kept deep air and the released walls, ported with their producers

Deviations 3 and 4 of levelset-coarse.wgsl.ts are gone. Both exits are verbatim at the old call sites, and
their inputs are made on the GPU at the old frame's points. The new file is levelset-release.wgsl.ts: the
readers (prefix `nbr`), the producers, and the schedule. Each formula cites its source file and line.
- Reach words, per tile. `nrReach` runs on FC_G1, in part 1 after the head and before move, as
  encodeCertificate does (uniform-mixed-frame.ts 1152). It writes each tile's umPackReach from
  umSignedFaceExtent over US: an F group for h tiles with a brick, and a C group for 4h tiles and for h tiles
  whose brick is absent. It also raises the plan speed (REL_SPEED, atomicMax).
- Spread. `nrSpread` z = 0, 1, 2 runs on C_ALL G1 (uniform-frame-plan-spread.ts 12–33), over the
  umCertificateRadius of that speed. An AIR page saturates the word and the box contributes 0.
- Wall reach. `nrWallReach` runs on C_ALL G1, in part 2 before the surface advect (wallReach 1682–1700). It
  writes each wall face tile's released dt·away/h and the six plane maxima.
- New ABI. Header words REL_SPEED0/1 (232–233) and WALL_REACH0/1 (234–245). Set regions REACH0/REACH1 and
  WALL_TILE (512 and 384 words a C slot) and REACH_STEP (a word a C slot). Memory is proportional to the C
  pool, not the lattice.
- Parity and claims. The header copies alternate by ring parity, and each producer zeroes the other copy. A
  slot claimed after part 1 has a stale REACH_STEP, so its tiles read saturated.
- Release bit. FLAGS bit 8 + a at the lower-face storage, agreed with P and landed in 9e5d2b14. It holds
  umRelease after project, and P's solver wall state between the rows and project. §2.4 of brick-abi.md gives
  both meanings.
- lcAdvect (C). Deep air (umDeepAirKept 1398–1406) runs after far air and is skipped for wall vertices
  (umAdvectStore). umReleasedWalls (829–870) runs after the contact (umAdvected 1035).
- F's lAdvect needs the same two calls. The readers are exported for F to include: `levelSetReleaseWGSL()`,
  `nbrDeepAirKept(v,1u,c)` and `nbrReleasedWalls(p,value)`.
- Test levelset-release.test.ts passes. It compares everything against Float64 oracles of the old formulas on
  a frozen tiered state with hashed US blobs and wall U with release bits:
  - every reach word in both halves, the stamp, the speed, the wall tiles and the plane words;
  - every other set word unchanged, and vel/phi/dir/fcell untouched;
  - a GPU probe of the readers over all F and C vertices, covering every path;
  - lcAdvect before and after the producers, covering kept, raised, unchanged, traced-but-not-kept, and
    wall vertices left alone.
  The fixture retires four F bricks under h tiles to cover the absent-brick reduction.
- GPU suites pass: levelset-coarse, brick-engine, brick-set-tiers, brick-census, brick-transfer and
  brick-owner. Unit tests pass.

## 2026-11-15: the old residency rule at the head, both engines (brick-residency.wgsl.ts)

Option (a), verbatim: the old pageSeedOf (uniform-mixed-dynamic.ts 2097–2148) and lrAsk
(uniform-local-residency.ts 140–158), at part 1's head after the census and before move. Part 2's
BOUND_F reach box is gone; no second bound was added. brick-abi.md §2.5 and §5.4 hold the rule.

- Kernels and placement.
  - `resSeed` (C_ALL G1) writes per 16-cell region and stage two seed-mask words and the lrRadii
    radii into `CENSUS.RES`. Seeds come from classify's near (new `CENSUS.NEAR`, stored by
    ncStore), WANT, the census's FIN (NC_WET, NC_CROSSES), tiered CMASK and F's PSUPPORT (925a9454),
    which it reads and zeroes. The flow is sampledFlow over the region grown by departureTravel
    (`CENSUS_REACH`) plus RESIDENCY_MARGIN_TILES = 5. PSUPPORT's travel joins stages ≥ 2, each
    stage with its RESIDENCY_STAGES margin.
  - `resRequest` ×2 (pass 0 pages and the bricks of resident pages, pass 1 the bricks of the pages
    pass 0 entered). Candidates come from each record's own box. Present destinations get MARK_F or
    MARK_C; absent ones go NONE → CLAIM and are appended with KEY_CLAIMED. `rqSettled` skips a
    destination that is already kept or claimed before the meets loop.
  - The claims use the existing kernels with KEY_CLAIMED. `setLeave` and `setLeavePages` move to the
    head: no hysteresis.
  - Tiered `censusDecide` keeps its band rings (cfg.w = 1, at a step's head only).
  - Part 2 keeps `setMark` (class words, the 1-ring FAIL.REACH latch, the BOUND_F clear),
    `setMarkPages` (LIQUID_TILES only) and the lists, with setArgs at z = 1 so the head's churn
    counts stay.
- Stated deviations (kernel header):
  - All-h ownership seeds are WANT alone. The old engine has no all-h mode; every tile is h by
    construction there, and a verbatim seed would keep every brick.
  - Granularity:
    - tiered pages are requested by every stage, and bricks by ownership and particles;
    - all-h, every stage requests at F granularity;
    - condition (a) holds: the old 32³ blocks keep cells resident beyond the footprints, where our
      owner readers return far or air. If a dam row moves for that reason, revisit the
      granularity, not the gate.
  - There is no closure cap or window failure.
  - Only resident pages seed.
  - A bring-up stage set on a partial-page lattice (no census) keeps the boot's set.
- Tests:
  - brick-engine's enter/leave test now checks two consecutive heads against `residencyOracle`
    (brick-oracle.ts):
    - masks are exact against the CPU seeds;
    - radii are bounded by the margin plus the global US travel;
    - the F and C sets, churn counts and lists, RES_RADIUS and RES_CLAIMED_F all match.
  - The tiered set test runs the head on the frozen state against the same oracle. Bricks enter at
    4h, pages enter as air, and LIQUID_TILES is still checked.
  - The tiered boot test checks the first head's set against the oracle.
  - The pool test sizes its pool from the first head's set.
  - The adapter test checks the bricks that actually left.
- Gates (tree = 925a9454 + these files only):
  - Tiered dam to 30 (g1, and g4 after the rqSettled change): no FAIL, past both F repros (step 14
    brick (3,5,15); step 15 (4,5,15)).
  - All-h dam to 200 (g2, g5): no FAIL. Drift −0.051 at step 60 and −0.0105 at step 200, the same
    as the baselines b2 and r6. Tiered drift −0.0655 at step 20, against the baseline's −0.0647.
  - GPU suites engine, set-tiers, census, transfer, owner and levelset-release: 27/27.
  - check:types: only lane n's 3 errors. test:unit: 1832 pass.
- Resident set and churn, mean per step, base (acd355ec/925a9454) → port:

  | run, steps | liveF | liveC | enterF | leaveF | enterC | leaveC |
  |---|---|---|---|---|---|---|
  | tiered, 1–20 | 2070 → 2159 | 71 → 80 | 226 → 248 | 133 → 134 | 5.3 → 4.9 | 1.2 → 0.7 |
  | all-h, 1–60 | 4268 → 4372 | 107 → 110 | 68.7 → 66.9 | 19.1 → 17.3 | 1.9 → 1.7 | 0.5 → 0.2 |

  All-h at steps 61–200: liveF 4176, enterF 25, leaveF 39 per step; 3048 bricks at step 200. The
  old reach box held 4913 (the whole domain) from step 28. Pages enter at step 1 by page request
  (36 → 69).
- Timing, timing mode, all-h, steps 35–45 profiled, steady state at 4913 bricks:
  - set update per step: 0.381 ms → 0.55 ms;
    - resSeed 0.060;
    - resRequest 0.143 for two passes (0.471 before rqSettled);
    - setMark 0.304 → 0.244;
    - claims and leaves under 0.02 each;
  - passes per step in the census and set work: about 16 → 23;
  - step median 114.1 → 115.8 ms, which is noise.
- Outliers, as found:
  - All-h: departureTravel peaked at 81.2 cells at step 23 (the baseline's was 57.9 at step 23). RES_RADIUS
    peaked at 24 tiles, from the surface record +y of region cells (96–112, 32–48, 64–80).
  - Tiered: 47 cells and 15 tiles at step 23. At step 14 the top region is at cells (96–128, 0–16, 0–16),
    surface, +z 15.
  - Isotropic 12–14-tile records are ballistic particle travel through PSUPPORT (about 34 cells).
  - Effect: the all-h set at step 23 is 4817 against the baseline's 4775 (+0.9%). Tiered steps 13–15
    hold 2629–3057 against 2249–2430 (+17–30%), and steps 20 onward match (4273 against 4214).

## 2026-11-15: pool peaks, measured on the brick engine (coordinator: author from the brick engine's own peak)

Letters: the take is 193 steps (NBFLIP_LETTERS_FRAMES at 24 Hz). The old-engine probe's 240-step run therefore
covers the whole take; "still rising" referred to steps past it. Within the take the probe peaks at F ring 5543,
core 2090 and pages 198 (step 180), and particles 944k (step 160). Those numbers are lower bounds only. The scene
pools are re-authored from the brick engine's own peak once letters runs on it (M4 gaps).

The dam, measured with the tiered brick engine at 5233e246 (lane F's dam-300th: composite, poolF 4913, rc 0, no FAIL):

| quantity | peak | step |
|---|---|---|
| F slots, HIGH_F | 4844 | (high-water) |
| liveF | 4811 | 60 |
| pages, HIGH_C | 125 | 30 |
| B, HIGH_B | 0 | |
| particles | 1,010,865 | 45 |

The particle peak is inflated by about 1.1x by lane F's D8 excess, which is still being attributed.

The authored f = 3968 latches POOL_F at step 20. The brick engine's F peak is 1.9x the probe's ring (2562),
because the ported old residency rule keeps 98.6% of the 17^3 window resident. The coordinator has revised the
residency policy for this reason: first (a) active lists that exclude reach-only bricks, then (b) per-source
local reach. The dam pools are re-authored after (b), from the brick engine's peak under the policy that will
ship. Until then runs pass poolF = 4913.

## 2026-11-15: the census's C passes fill the GPU (X8)

censusCoarse, censusImportance, censusDecide and censusWant ran a workgroup a C page (about 90 on the tiered dam),
each lane 8 tiles in turn. They now run on C_ALL's new X8 triple (args words 12k + 9..11, eight workgroups an
entry, filled by setArgs): workgroup r of a page takes its 64-tile slab 64r..64r+63, a lane a tile. Everything
cross-slab is an integer atomic or a disjoint word: REQ and WANT words 2r, 2r + 1 a slab; TRIG children 8r..8r+7;
the page's candidate bricks deduplicated through a new set region `CAND_MASK` (8 words a C slot, `atomicOr`'s old
bit picks the one lane that lists a brick; cleared by censusImportance). Two evaluations are reordered, not changed:
censusCoarse's depth jobs sample each wide vertex of a slab's deep tiles once (the sample is the vertex's alone),
and movingShapeNear reads censusImportance's per-tile MOV bits (census words 6608..6623, a page's 512 bits) as bit
rows instead of FIN and 6 keys a tile. censusCoarse's face parts hold their face job index (u32, 1.5 KB) instead of
a vec4i (6 KB); the smaller workgroup memory alone took the pass from 0.61 to 0.49 ms.

Exactness: brick-census-x8.test.ts runs the frozen G1 kernels (parity/census-g1-frozen.ts, 5233e246) and the X8
kernels on the same state and compares the whole census buffer and the whole set buffer word for word (the
candidate list sorted; CAND_MASK and MOV excluded, the G1 passes have none). Synthetic census scene (31 candidates,
143 4h deep tiles, 51 MOV tiles) and the old engine's dam census frames 14 and 43 (3287 and 8890 WANT tiles, 634
and 1408 4h deep tiles), each tiered, tiered at the head (keep marks) and all-h: 0 differing words.
brick-census, brick-census-dam, brick-engine, brick-set-tiers and brick-abi tests pass unchanged (33/33).

Tiered dam, steps 13-24 (`brick-dam.mts --tiered --solver=composite --steps=24 --profile=13 --poolF=4913`), ms/step:

| kernel | 5233e246 (G1) | X8 |
|---|---|---|
| censusCoarse | 1.387 | 0.492 |
| censusImportance | 1.207 | 0.202 |
| censusDecide | 1.720 | 0.404 |
| censusWant | 0.087 | 0.038 |
| censusCubes (G1, unchanged) | 0.022 | 0.033 |
| censusFine (unchanged) | 3.004 | 2.982 |
| C passes | 4.424 | 1.169 |
| kernel sum | 129.14 | 129.21 |

The kernel sum's non-census rows move by ±0.5 ms a kernel run to run (my three X8 runs summed 126.9, 131.5 and
129.2 ms); the census rows are the change.

## 2026-11-15 — TRIG bits 0..7: the old need trigger

TRIG bit j is now the old `nbActivityTriggered` in need mode (uniform-narrow-band-activity.wgsl.ts l.28, the
dam's particleSeeding): criteria & 0x3c, or shape, or thin with dt · tuDeformation(t) / h > 0.02. Before, any
trigger bit set it. The deformation (tuDeformation, uniform-narrow-band-flip.ts l.940) is the frame plan's
tileErrorOf (uniform-mixed-frame-plan.ts l.357-404): max(4h · gradient, span), where the gradient sums, over each
axis with both neighbours stored, the largest component of max(|above − centre|, |centre − below|) of the tiles'
mean extended face velocity, over 4h; and the span is localSpeed's (l.269-345) sum over axes of the tile's face
velocity range (umSignedFaceExtent l.241: an h owner's own texel, a 4h owner's high face, one patch or sixteen
beside an h tile, plus the low face where low-open; nonfinite saturates). The means and spans are per-tile side
outputs of the census passes that already read those faces (census words DEF, 4 a tile: censusFine for h tiles,
censusCoarse for 4h), and censusDecide's TRIG lanes compute the gradient only for a tile whose criteria are thin
alone. Stated: the plan reads the census's velocity at the head, where the old activity read the certificate its
last frame plan built; the reader-class zero and the error saturation drop (thin needs a crossing tile, which is a
reader; the error saturates only with the span).

Dropped: the old budget (uniform-narrow-band-budget.wgsl.ts activityAdmit, bit 27) runs only when
policy.activity.x < 100 (nbBudgetOn). The dam sets no detailBudgetPercent (default 100); the old engine's probe
(tools/nbflip-census-capture.ts, now printing a dropped count) prints the policy word 24 = 100. So no dam tile is
ever dropped, and the rule is not missing for the dam. The seeps (detailBudgetPercent 5) do run it: a missing rule
there, not ported.

Tests: brick-census.test.ts's oracle transcribes localSpeed, tileErrorOf and the need rule (plan words checked:
span exact, an h tile's 64-term mean within 1e-6, since the compiler reassociates the sum; TRIG exact). The
synthetic scene has no thin-alone tile; brick-census-dam.test.ts transcribes localSpeed on the old engine's own
extension texels and checks every plan word and TRIG of the 32768 tiles of frames 14 and 43 (frame 43: 874
thin-alone tiles, 664 heated, 210 no longer heated). brick-census-x8.test.ts keeps comparing every other word
with the frozen G1 kernels. Tiered dam profile (contended: other GPU processes ran): censusFine 3.10, censusCoarse
0.60, censusDecide 0.45 ms/step (0.12, 0.11 and 0.05 above the X8 run).

## 2026-11-15 — the boot's TRIG census

BrickStages.kick is { force, after, seed? }. With `seed`, the boot runs force, the solve, after, then a TRIG-only
census of the kicked state (brickTrigCensusSchedule: censusDecide at cfg.z = 2 skips the band decision, WANT,
candidates and keep marks; no claims, no transfer), then seed. Without `seed` the boot is unchanged. Tests:
brick-engine.test.ts (the seeded boot is the unseeded one plus the six census passes and the seed before stepEnd,
censusDecide at z = 2, no set/res/transfer kernel among them; all-h and tiered), and brick-census.test.ts (on the
synthetic state the TRIG-only census writes the full census's TRIG words, CENSUS_TRIGGERED, and no WANT word,
candidate or keep mark).

## 2026-11-15 — the active class and F_ACT (ABI; no stage dispatches from it yet)

setMark (part 2, after the rebin) writes the active class (brick-abi.md §5.5, ACT, a set word an F slot): the
ACTIVE tiles (an h-owned tile, tiered; a φ < 0 vertex through the tier authority, a 4h tile's C corners; TILE_OCC;
the census's TRIG or a nonzero heat word of the stage set's `activity` tiles), and NOW on every resident brick within
ACT_DILATION = 3 tiles of one (a lane an offset of the 5^3, at tile precision). The head's setLists lists F_ACT =
NOW | PREV, plus an entering brick, then moves NOW to PREV; the boot lists every live brick. LIST.F_ACT = 17, args
FA_G1/G8/G64 at 400 (F_ACT's bricks, then C_ALL's pages), ARGS_LIST_END = 409 for the stage and solver validators.
setMark's LIQUID bit is now the class (an h-owned φ < 0 vertex); setLists ORs LIQUID onto every live brick until
lane P's row list lands, so F_LIQ = F_ALL still. pres.project dispatches on F_ALL (P's contract);
PressureSolver.before is placed just before the operator. Nothing dispatches from F_ACT yet: lane F's moved
kernels wait on the gate below.

Gate status (per-kernel, frozen F_ALL trajectory, tiered dam, 30 steps): every moved kernel writes bitwise the same
words on its F_ACT bricks. With stale words on the bricks outside F_ACT poisoned (two poisons), lTiles, lDilate,
lNear and lDistance on 4h-only bricks read words that F_ALL refreshes every step from C (TILE FLAGS/DA/DB/DIST,
NEAR, DEPTH; reach 5 tiles in lDilate, beyond the 3-tile dilation). Open: whether the stale values differ from
F_ALL's (a shadow-value trajectory, not poison), before F moves its dispatches.
Checks: check:types (the 3 lane-n errors), test:unit 1833 pass 0 fail, brick-set, brick-set-tiers and
brick-engine GPU suites 20/20 (rc 0, no 00000206).

## 2026-11-15 — F_ACT gate passed; F_LIQ is the real class

Gate (a), the complete-state shadow trajectory (tiered dam, 60 steps, rc 0, no 00000206):
- Method:
  - The reference runs every kernel on F_ALL.
  - Per kernel, a control rerun on F_ALL marks the nondeterministic words (pScan, pMove, pSeed, pScatter atomics; particles are compared as canonical multisets).
  - The variant then runs the moved kernel on F_ACT over the same input.
  - Every word it writes or leaves must match the reference, except stale words that nothing reads later.
- Result: depTotal 0 (no kernel read a stale word) and diffTotal 0 (no written word differs).
- Moved, 15 kernels: uCache, vGauge, lAdvect, lSplat, lUnion, lClear, lNear, lDistance, lPublish, pSeed, uAdvect, pDeposit, uViscous, uTransfer, uExtend.
- Kept on F_ALL: lInit, pScan, pTiles, lTiles and lDilate.
  - lDilate's separable Chebyshev transform reaches R = 5 tiles, beyond ACT_DILATION = 3.
  - When lTiles and lDilate ran on F_ACT, the derived TILE FLAGS/DA/DB/DIST words went stale and were read: a real divergence. Both are light, so they stay.
- Hand-off: the move patch went to lane F (brick-fluid-stages.ts is lane F's). Lane F applies it after its item-1 round.

On the dam, F_ACT ≈ F_ALL:
- F_ACT is 557 at step 0, 1838 at step 10, 4105 at step 20 and 4594 at step 30;
- it equals liveF from about step 28;
- so on the dam the gain comes from (b), the local residency, not from the lists.

F_LIQ:
- setLists no longer ORs LIQUID onto every live brick. F_LIQ is setMark's class (an h-owned φ < 0 vertex), since lane P reads no F_LIQ (04298677, b9315489).
- brick-engine.test.ts checks F_LIQ against the records' LIQUID bit.


## 2026-11-15: the head restructure, round 1 (task 3): censusFine fills the GPU, setTransferList X8

Changes (WANT and TRIG bitwise):
- `censusFine` now runs on F_ALL's X8 triple: eight workgroups a brick, workgroup j taking tile j when it is h-owned, a lane a cell. Before, one workgroup took a brick's h tiles one after another.
  - Each tile's 5^3 raw vertices (ncLoadFine, into ncTileRaw) and its 240 US faces (nbFacePlane on planes 4t..4t+4 of each axis, cfFaces) are staged once.
  - The cells' classify, owner bounds and plan extents (cfClassify, cfOwnerBounds1, cfExtent1) and the surface displacement (ncSmoothTile) read the staged values: the same loads, each made once a tile instead of up to 8x a cell.
- `setTransferList` runs on C_ALL's X8: eight workgroups a page, a lane a tile. Its entries append by atomicAdd, so their order was never the tiles'.
- Checks:
  - brick-census-x8.test.ts compares against the frozen G1 census (which includes the frozen censusFine) on every census and set word (all 3 states, tiered boot, head and all-h). brick-census.test.ts covers the oracle, TRIG and DEF, and brick-census-dam.test.ts the frames 14 and 43 fixtures.
  - brick-transfer.test.ts, brick-transfer-dam.test.ts and brick-engine.test.ts: 27/27 at 84b98c2b.
  - check:types (the 3 lane-n errors only), test:unit 1833 pass 0 fail (84b98c2b plus this change).
- brick-abi.md: 84b98c2b carried this change's doc hunks with the fused transfer's wording; this commit restores the transfer text to the two face jobs (cfg.z = U or US) and the separate setTransferCells.

Tried and reverted: setTransferFaces taking both channels in one workgroup, and setTransferCells folded into it (lanes 0..63).
- That gave 8 → 5 transfer launches, but faces went 1.02 (2 launches) + cells 0.15 → 1.89 ms in one launch: longer workgroups, more registers and workgroup memory.
- Launches cost ~4 µs here, so the fusion lost 0.6 ms.

Timing slot (exclusive, tiered dam composite, profile steps 13–24, ms/step kernel time):
- Before is bbcc5fc2 (tm-head); after is 84b98c2b plus this change (tm-c).

| head kernel | before | after | launches |
|---|---|---|---|
| censusFine | 3.233 | 1.682 | 1 (37.5k wg, was 3.4k) |
| censusCoarse | 0.557 | 0.513 | 1 |
| censusDecide | 0.481 | 0.448 | 1 |
| censusImportance | 0.197 | 0.202 | 1 |
| censusCubes, censusBegin, censusWant | 0.027 | 0.049 | 3 |
| setTransferList | 0.240 | 0.066 | 1 |
| setTransferFaces (U, US) | 1.021 | 1.256 | 2 |
| setTransferCells, Commit (x2), Args, CommitMask | 0.295 | 0.355 | 5 |
| residency (resSeed, resRequest x2, claims, leave) | 0.448 | 0.530 | 7 |
| set tail (setRecords, setInit, setLists, setArgs, stepEnd) | 0.142 | 0.158 | ~7 |
| **head sum** | **6.64** | **5.26** | ~29 |

- The faces and commit moved up with no code change; that is run-to-run spread of about ±0.2 ms.
- Against design §6's brick-set budget (0.6 ms in 4 kernels), the head is 8.8x over.
- What is left: censusFine 1.68 (the per-cell classify and the crossing tiles' shape error), the transfer's face jobs 1.3, and the C passes 1.2.
- The residency rows are a5599629's: resMark replaces resSeed and resRequest.

censusFine's inputs as F's side outputs: spec sent to lane F, with the recommendation not to do it yet.
- Movable bitwise: KEYS and DEF's span words. They are order-free min/max of US faces. uExtend would publish per h tile, per axis, a plane-4t partial and a planes 4t+1..4t+3 partial (12 words a tile), and uExt4 the same for 4h owners. The census combines t's partials with t + e's plane-4t partial.
- Not movable bitwise:
  - DEF's mean (the lane-order sum);
  - every φ-derived word: CROSS, GAP, reach, flags, shape error, depth. Their last writers are lPublish and then lcPublish, the +4 vertex layer is another workgroup's, and the shape and depth stencils reach into neighbour tiles.
- Estimated gain: the face half of the staging, about 0.3–0.5 ms.

Per area (tm-c, 84b98c2b plus this change; kernel sum 100.8 ms, fenced median 124.7 ms; ~270 launches):

| area | ms | launches | §6 dam budget (seep-scaled) |
|---|---|---|---|
| grid velocity (uAdvect, uViscous, uTransfer, uExtend, uExt4Seed, uExt4, vGauge) | 42.5 | 6.4 | 1.6 |
| level set (l*, lc*) | 18.3 | 20 | 2.0 |
| pressure solve (pmg*) | 12.2 | 169 | 3.0 |
| move (uCache, pMove, pScan, pScatter, pTiles, pDelta, pJoin, pArgs) | 10.0 | 6.4 | 1.5 |
| pressure setup and projection (operator, rowsC, setup, walls, project) | 7.9 | 32 | 1.0 |
| brick set (census, residency, set, transfer, nr*, setMark) | 5.8 | 35 | 0.6 |
| transfer and seeding (pDeposit, pSeed) | 4.3 | 1.8 | 1.0 |
| floor (launches at ~3.5 µs, plus fenced minus kernel sum) | ~0.9 + 23.9 | | 1.0 |

## 2026-11-15: the transfer's face jobs (task 4, item 1)

- The list was already restricted: setTransferList is markTile verbatim ("within one tile of a width change"). brick-transfer-dam.test.ts matches the old engine's listing bitwise, and the old dam lists 2648 tiles at frame 22; the brick dam lists about 3.5k a step at steps 13–24.
- So the cost was attributed instead. Before, setTransferFaces cost 180 ns/wg × 3.8k wg × 2 launches (U, US). Each launch repeated the same ownership and address chains. Every face read walked the directory page word → CMASK → child slot → vel, in each of 4 barrier phases (beside, slope, sample, mean).
- The fix (bitwise; brick-transfer.test.ts and brick-transfer-dam.test.ts pass, brick-engine.test.ts 17/17):
  - One launch takes both channels a lane (vec2 values).
  - The job stages its tile's 3^3 tile neighbourhood (page slot, CMASK and WANT owner bits) and its 2^3 F bricks once. A read outside them falls back to the global readers.
  - Every global face read (beside, slope face, sample faces) is issued in a single phase; slopes, samples and means then run on workgroup memory.
  - setTransferCommit also takes both channels in one launch.
- Timing slot (tiered dam composite, steps 13–24, ms/step):

| kernel | before (tm-c) | both channels, staged (tm-x1) | one read phase (tm-x2) |
|---|---|---|---|
| setTransferFaces | 1.256 (2 launches) | 0.754 (1) | 0.650 (1, 187 ns/wg) |
| setTransferCommit | 0.153 (2) | 0.076 (1) | 0.071 (1) |
| head sum | 5.26 | 4.72 | 4.50 |

- Checks: test:unit 1833 pass 0 fail; check:types shows the 3 lane-n errors only.

## 2026-11-15: the real D, scoped (task 4, item 3)

The scoping went to the coordinator, who ruled on three points:
1. No cold freeze. Its terms are identically 0, a stated deviation, and lcAdvect's COLD φ hold stays off.
2. pressureDelta comes from lane P: PDELTA in pmgC, 8 words a C slot, in Pa, computed by the old publishPressure formula after project. regionHeat reads +Inf (HOT) until it lands.
3. A 4h tile with no F slot gets a heat word, updated in censusDecide by the old rule.

**M3 question:** measure the cold freeze against seep's BULK pressure-only rows in the deep interior, and decide by time and gates.

**M3 question (restated 2026-11-16, coordinator ruling on d02947dc):** in the old engine, replay and the cold freeze are one mechanism. `UNIFORM_COLD_FREEZE = true` (lib/methods/uniform/uniform-cold-freeze.ts:52). A frozen face keeps its stored F through project (uniform-mixed-pressure-velocity.ts:453-458), and a frozen owner's texels hold through momentum (uniform-mixed-momentum.ts:146). So a frozen region's velocity is bitwise still, and the replay key check passes. That is why the old dam replays 17/17 regions under gravity while ours replays 0/64: the projection residual moves face bits by up to 8.9e-6 m/s.

The M3 decision is therefore one decision with two options, both judged by time and gates on seep and compared against seep's BULK pressure-only rows:
- (A) freeze plus replay, both verbatim;
- (B) no freeze and no replay.

A tolerant key with no freeze (new design: a certificate bounding each velocity word's effect on the records) is considered only if both A and B lose. Until M3, replay stays on as gated (d02947dc), and regionDeep's 0.07-0.14 ms is paid where it does not fire.

Build plan, with the old frame's order (heat.encode, then the census, then plan.encodeCertificate on the remapped velocity):
- **Certificate:** localSpeed after setTransfer at the part-1 head. It runs on the new ownership, so it cannot be a side output of the census.
- **regionHeat:** C_ALL X8, a workgroup a region (an octant of a C page) and a lane a tile, with tileError computed inline by each lane from the stored tile means.
- **regionSpread**, then **regionDeep:** a lane a region. Each is separated by a grid-wide barrier: spread reads every region's level, deep reads every region's tier.
- **regionDeep again** after the claims.
- **Census:** decide's deep keep of CC_DECIDED and CC_STATIC, regionList and replay, and censusFine/censusCoarse skipping the replayed regions.

## 2026-11-15: the "26 ms floor" attributed (floor task)

**Brief.** tm-x2 gave a profiled fenced median of 126.3 ms/step against a kernel sum of 100.2 ms/step (tiered dam, composite).
Design §6 budgets the floor at about 1 ms. The gap is attributed below.

**Finding: there is no 26 ms floor in the engine.** The gap is three things:
- (a) a window mismatch in the profile tool;
- (b) the profiler's own pass splitting;
- (c) about 1 ms of real host round trip.

The unprofiled step is the GPU's two encoders plus about 1.0 ms.

Runs, all on the timing slot on a HEAD 94577aa0 tree (`~/code/me/fluid-scratch/lane-e/floor/tree`), each rc 0 with no 00000206. The tool is
`floor/tools/brick-dam-floor.mts`, lane F's brick-dam.mts plus the following:
- `--cpu`: performance.now around submit, writeBuffer, advance, encodeAhead, collect, fence and the harness awaits;
- `--spans`;
- `--xcwait`;
- the profile's `window` and `rowsWindow`;
- per-step `profspan` rows.

| run | what | fenced median / mean (ms, steps 18–23) |
|---|---|---|
| `floor/runs/base-cpu` | unprofiled, CPU timing only | 117.2 / 116.7 |
| `floor/xc-att` | unprofiled; Metal System Trace attached at step 17 (`xc-attach.sh`, reduced by `xc-reduce.mts`) | 114.6 / 113.4 |
| `floor/runs/win-prof` | brick-dam's per-dispatch profile (`--profile=13`) | 125.6 / 122.1 |
| `floor/runs/base-spans` | one timestamp pair on each part's single pass | 237.2 / 232.5 |

**Decomposition of tm-x2's 26 ms**, using win-prof as the same measurement (kernel sum 99.9 over all profiled steps; fenced mean 122.1):
- **Window, 12.3 ms.** The profile row averages steps 14–25 (engine ≥ 13). The fenced median covers steps 18–23 only, because steps ≤ 16, 24 and the step after each are samples. The cost grows by about 3.5 ms a step: the kernel sum at steps 18–23 is 102.8 → 119.3, with a mean of 112.2 and a median of 115.5. Over the same window, the kernel sum is 112.2, not 99.9.
- **The profiler's passes, 8.9 ms.** It puts each dispatch in its own timestamped pass, with an args copy before each indirect pass: 293 passes, and about 115 gaps over 50 µs a step. Span minus kernel sum is 121.1 − 112.2.
- **Host, 1.0 ms.** Profiled fenced wall minus GPU span: 122.1 − 121.1.
- These add up to 22.2 = 122.1 − 99.9. The rest of tm-x2's 26 comes from quoting a median (125.6) where the decomposition uses a mean (122.1).

**The unprofiled step** (xc-att, same run, median per step): the part 1 encoder is 13.4 ms, the gap to part 2 is 0.055 ms and the part 2 encoder is 100.2 ms. The time from part 2's end to the next step's part 1 is 1.0 ms: the fence notify, the harness, the ring write, the submit and the 0.25 ms from submit to GPU start. Wall = 114.6. The encoders sum to a mean of 112.4, against a profiled kernel mean of 112.2 (cross-run). So the cost inside the passes, from dispatch boundaries and Dawn's one-thread indirect-validation dispatch before each indirect launch, is within cross-run noise (±4 ms; see the trajectories).

| item | ms/step | source |
|---|---|---|
| 1. CPU encode | encodeAhead 0.90, after the step's submits (overlaps GPU). Encode-ahead is live: step N+1 is encoded before N's fence. Critical-path CPU before part 1 is on the queue: 0.16 (ring writeBuffer 0.03 + part 1 submit). Whole admit 1.40. | base-cpu |
| 2. submit → GPU start | 0.25 (submission start to part 1 encoder start). Both submits' translation 0.43 CPU; part 2's overlaps part 1's 13 ms. No pipeline or bind-group creation on the step path (record() at create). | xc-att, base-cpu |
| 3. GPU idle | part 1 → part 2: 0.055; between steps: 1.0 (fenced round trip; absent when pipelined). A step is 2 compute passes, 2 command buffers and 2 submits, plus 2 Dawn blits (the ring writeBuffer and the receipt copy, 0.003 ms each); 4 Metal encoders. Launches: 34 + 259 = 293. | xc-att, win-prof |
| 4. writes and readbacks | engine: 1 writeBuffer (0.03), 1 receipt copy and 1 mapAsync (lagged, never awaited), no onSubmittedWorkDone. Harness: 1 fence a step (fenced mode by design); presentation off; sample() readbacks only on sample steps, outside the window. | base-cpu |
| 5. profiler inflation | per-dispatch profile: fenced +8.4 median (125.6 vs 117.2), of which 8.9 is pass gaps (span − kernel). One timestamp pair on each recorded part: 2.0x (237 vs 117; the part 2 encoder takes 180–210 ms instead of 100). Never put timestampWrites on the recorded parts. | win-prof, base-spans, base-cpu |

**Fix.** The cause is the measurement, not the host path. The seep levers are already live in brick-engine.ts:
- encode-ahead;
- the two-part submit;
- one ring write;
- lagged receipts with no per-step readback;
- no CPU round trip.

So the engine is unchanged. The tool fix is in `floor/tools/brick-dam-floor.mts`:
- The profile row carries `window`, the kernel sum's median and mean over the fenced median's own steps, and the profiled GPU span.
- It carries `rowsWindow`, the per-kernel rows over those steps.
- `floor/tools/areas-window.py` reduces per area on that window, and takes the fenced figure from an unprofiled run.
- Same-run GPU idle comes from `xc-attach.sh` (attach mode; launch-mode xctrace hung at "ending recording" for 10+ min and was killed) and `xc-reduce.mts`.

Before: 126.3 fenced (profiled) vs 100.2 kernel sum, a "floor" of 26.1. After, same run (xc-att): 113.4 fenced mean vs 112.4 GPU encoders, a floor of 1.0. Cross-run unprofiled vs the window kernel sum: 1.7 (medians) to 4.5 (means).

Per area, win-prof's `rowsWindow` (steps 18–23, ms/step), against design §6's dam budget:

| area | ms | launches | budget |
|---|---|---|---|
| velocity | 49.17 | 7 | 1.6 |
| level set | 21.16 | 22 | 2.0 |
| pressure solve | 13.78 | 184 | 3.0 |
| move | 10.52 | 7 | 1.5 |
| pressure setup and projection (incl. pressureTileRoots) | 8.75 | 35 | 1.0 |
| transfer and seeding | 5.13 | 2 | 1.0 |
| brick set | 3.68 | 36 | 0.6 |
| floor (same-run GPU idle, xc-att) | 1.0 | – | 1.0 |
| **total** | 112.2 kernel + 1.0 | 293 | – |

## 2026-11-15: the real D built (task 4, item 3)

The uniform-region heat, spread and deep tiers run on the brick engine (brick-region-heat.wgsl.ts), tiered only, with
the frame plan's certificate they read. The rules and their arithmetic are the old modules' (035 heat l.1000,
spread l.1139, deep l.1170; 009 seedJob and dilate0..2; uniform-mixed-frame-plan.ts localSpeed and tileErrorOf).
The structure is the brick engine's.

Passes (part 1's head; each separate launch names the grid-wide barrier that forces it, in the kernel header):
- `regionHeat` (C_ALL X8: a workgroup a region, a lane a tile's activity heat, lane 0 the region) after censusBegin;
  `regionSpread` (C_ALL G8, a lane a region): it reads every neighbour's level;
- the head's deep runs inside `censusCoarse` (cfg.w = 1), which already follows the spread's barrier;
- `censusDecide` keeps a deep, unjoined tile's CC_DECIDED and CC_STATIC bits (`DEC`, `STAT`) and updates the
  activity heat `HEAT4` (one word a tile, every tile, by lHeatNow's rule), with cfg.w = 3;
- after the transfer: `certCoarse` (C_ALL X8; its workgroups e < |C_ENTER| also take entering page e's claim: HOT
  promotion of the 4^3 regions about it, its own regions reset as the old free region), `certFine` (F_ALL X8, a
  workgroup an h tile, 125 vertices staged), `certError` (C_ALL X8: the support closure on a 10^3-tile neighbourhood
  in workgroup memory, tileErrorOf, the region reductions, then the post-claims deep);
- `setTransferList` (cfg.w = 1) promotes each listed tile's region WARM.

Rulings applied: no cold freeze (terms identically 0); pressureDelta is lane P's PDELTA (pmgC, `mgLayout(L).PDELTA`,
wired when the solver has pmgC, +Inf otherwise); HEAT4 for tiles with no F slot. Other deviations, stated in the
header: force = the first heat (HEAT_FORCED); no bodies or sources; residual floor 0.05; open = every cell; regions
are the lattice's only (`rgInLattice`: a page beyond the lattice, the reach ring across a closed wall, holds no
lattice tile and takes no heat, tier, deep or count). The census scheduled alone (`brickCensusSchedule(..., heat
false)`, the frozen comparisons) keeps D = 0.

Header words 249–255 (`REGION_H`: HEAT_FORCED, STILL_EXTENT, STILL_HIGH/LOW, STILL_TRAVEL, HEAT_DEEP, HEAT_COLD) and
the census words TC, HEAT4, REG, DEC, STAT (stride 18368) are in brick-abi.md. `BrickEngine.save()` / `restore()`
take and put back every buffer and the clock, so the heat state persists. A state from another layout fails
explicitly.

Not built: **replay** (regionList/replay publishing stored records, censusFine/censusCoarse skipping replayed
regions). Every tile is still classified every census. For a deep region the records are this census's, where the
old replay published the stored ones. Next steps for it:
- a per-region replay mode at the head (deep, stamp == deepEpoch + 1, page generation equal);
- the fresh velocity-key check per region, demoting the whole region on any change (l.1935–2090);
- censusFine and censusCoarse early-out per tile of a mode-1 region, keeping W0/W1/CROSS/GAP/NEAR/KEYS. DEF must
  still be fresh, or stored alongside.
- Then the fixture gate: replaying the old dam's 1088 and 192 tiles at frames 14 and 43 with WANT identical.

Gates (GPU runner, check slot, on a HEAD + lane E tree; rc 0, no 00000206):
- census oracle with D ≠ 0 (brick-census.test.ts): DEEP per the injected tiers, WANT and DEC equal to
  `censusOracle` with the deep tiles' `decided`, more than 50 decidedPrev = 1 tiles, STAT carried;
- the frozen X8 comparisons (brick-census-x8, 3/3) and the dam census and transfer fixtures;
- brick-region-heat.test.ts:
  - tiered dam (composite, PDELTA live): 40 steps with no latch, tiers and heat in range, HEAT_COLD at least the
    tier-0 count, and no deep region beside a non-COLD one;
  - still reservoir (a box at rest, surface at 40 cells): 64 deep regions (H 64), 128/128 deep decided words kept
    across a head, and save/restore with the buffers bitwise and 6 steps after restore identical in tier, deep and
    heat on all 256 regions;
- engine 20/20, tiers 3/3; unit 1834/0.

Timing slot (fixed brick-dam command, steps 13–24). Both trees are at 08dac85d: `realD/base` HEAD, `realD/tree`
HEAD + this work (`fluid-scratch/lane-e/realD/prof/{base,realD}.jsonl`). The real D's kernels:

| kernel | ms/step | launches/step | workgroups/step |
|---|---|---|---|
| regionHeat | 0.055 | 0.92 | 765 |
| regionSpread | 0.027 | 0.92 | 12 |
| certCoarse | 0.175 | 0.92 | 795 |
| certFine | 0.497 | 0.92 | 43869 |
| certError | 0.153 | 0.92 | 795 |
| **sum** | **0.91** | 4.6 | |

Kernel sum 86.7 → 102.8 ms/step; fenced median 115.2 → 128.7. The real D's own kernels are 0.91 of the 16.1. The
rest is F's stages on a different trajectory:
- velocity +8.8, level set +3.3, move +1.7, transfer and seeding +1.4;
- censusFine's F_ALL workgroups are 7% higher (37401 → 40166).

The deep keep acts at the dam's early steps: 48 deep regions at step 10, 2816 tiles, none of them a kept band. None
are deep at step 20, so the trajectories part there. This is one run each, by the policy. A repeat would separate
divergence from noise.

## 2026-11-16: the real D gated (0be9d9f4 against db999a8b)

Both trees in `fluid-scratch/lane-e/realD/`: `base2` is db999a8b, `cand` is 0be9d9f4. Runner, rc 0, no 00000206.

Check slot: dam-300 (lane F's brick-dam, tiered, composite, poolF 4913) judged with tools/nbflip-gates.ts
(`realD/gate/realD-round*.{md,json}`). Worst margin per gate; margin above 1 fails.

| gate | base, ref-2761350a (a–e) | cand, ref-2761350a | base, ref (a, b, d, base-e) | cand, ref |
|---|---|---|---|---|
| D1 | 0.55 | 0.55 | 0.50 | 0.50 |
| D3 | 0.73 | 0.47 | 0.52 | 0.29 |
| D5k | 0.14 | 0.07 | 0.31 | 0.24 |
| D6 | 0.56 | 0.56 | 0.56 | 0.56 |
| D8c | 0.49 | 0.51 | 0.51 | 0.53 |
| **D8** (30–90) | **1.24 FAIL** (7.091e5 vs 6.309e5) | **1.12 FAIL** (7.014e5) | 0.69 | 0.63 |

Every other gate passes in all four columns. D8 does not worsen with the real D. Against the five references, D8 at
frames 30–90 fails at db999a8b as well, 1.24 there and 1.12 with the real D. Lane F's st7 failed by the same margin,
1.25. That failure predates the real D, so no attribution is needed for this commit.

Timing slot, fixed command (`brick-dam-floor.mts --tiered --solver=composite --steps=24 --poolF=4913`), steps
18–23. Two profiled runs (`--profile=13`, reduced with areas-window.py on rowsWindow) and one unprofiled fenced run
per commit (`realD/tm/`):

| | base p1 | base p2 | cand p1 | cand p2 |
|---|---|---|---|---|
| window kernel sum, mean / median | 96.0 / 97.4 | 96.5 / 97.8 | 96.5 / 97.5 | 94.7 / 96.1 |
| velocity | 40.95 | 41.80 | 41.18 | 39.71 |
| level set | 17.23 | 17.28 | 16.94 | 16.93 |
| pressure solve | 12.97 | 12.70 | 13.04 | 12.60 |
| pressure setup and projection | 9.08 | 8.84 | 8.88 | 8.89 |
| move | 8.67 | 8.63 | 8.62 | 8.71 |
| brick set | 3.69 | 3.70 | 3.67 | 3.59 |
| transfer and seeding | 3.43 | 3.59 | 3.51 | 3.55 |
| real D (5 kernels) | – | – | 0.64 | 0.66 |
| profiled fenced median / mean / p95 | 106.7 / 105.5 / 113.0 | 107.1 / 105.9 / 109.3 | 107.7 / 106.0 / 109.4 | 106.2 / 104.7 / 109.7 |
| launches/step | 279 | 279 | 284 | 284 |

Unprofiled fenced, steps 18–23 (median / mean / p95): base 95.2 / 94.6 / 99.9, cand 98.9 / 96.2 / 100.8.

Results:
- The kernel sums across the two runs are base 96.25 ± 0.25 and cand 95.6 ± 0.9. The real D costs its own 0.65 ms and
  5 launches, and no area grows outside the spread between runs.
- The unprofiled figure is +1.6 on the mean and +3.7 on the median. That is one run each, on a cost that rises
  about 2 ms a step over the window (base's thirds 89.0 / 95.1 / 99.7, cand's 90.0 / 98.4 / 100.1), so the median
  moves with the trajectory. The profiled fenced medians are 106.7 / 107.1 for base and 107.7 / 106.2 for cand. The
  difference does not reproduce in the kernel sums.
- The earlier +16 ms (08dac85d, one run each) was trajectory divergence and does not reproduce.

## 2026-11-16: residency (b), seep S2j's local per-axis reach; the far-wall REACH latch's cause

**Rule** (brick-residency.wgsl.ts `resMark`; brick-abi.md §5.4). This retires resSeed's per-stage records (`CENSUS.RES` is now unused), the global departureTravel/sampledFlow envelope and the 5-tile margin.
- A brick source is any of three:
  - `ACT.RES` tiles (setMark's ACTIVE tiles, less those ACTIVE only by their 4h corners);
  - `PSUPPORT` word 0;
  - this head's census `TRIG` bits 0..7.
- The source's tile box is grown per axis and side by max(travel, (BOUND_F + |g| dt) dt / h) + 12 cells. It takes the bricks it covers and, tiered, their pages.
- Tiered page sources (4h liquid tiles) take pages only, grown by flow + 12 cells.
- The run is two `resMark` passes over F_ALL; tiered pass 0 runs FC_G1.
- The boot runs part 2's class pass after F's boot stages, so step 1's head has its ACTIVE tiles.
- The kept latches are setMark's 1-ring, pScatter's and pJoin's. A new one: setLeavePages latches a leaving page that held 4h liquid tiles.

**The far-wall FAIL.REACH latch (lane F's 165-step timing runs, all on the old rule).**
- Instrumented on ddfecfc5, the latching site is pJoin's `pSupport`.
  - Pipelined: 2 of 3 runs latched, at steps 70 and 115. Separately, a 300-step check run of the old rule latched at step 121.
  - The record was a fresh seed: age `SEED_FIRST`, depth −1.76 cells, coupled.
  - It sat in a 4h-owned tile whose brick held no particles at the last setMark. Its 2-cell support reached the absent neighbour brick: (14,0,0) beside the x = 127 wall, (0,0,15) beside the z = 127 wall.
- How such a seed arises:
  - pSeed seeds wherever the tile heat `FT_T` is above 0, deep sites in 4h owners included (seedSite restricts only the outer shell to h owners).
  - `FT_T` is the larger of the tile's heat and half its 27-tile maximum (lDilate).
  - That heat is 2 on this head's triggered tiles (lHeatNow).
- Why the old rule missed it: it seeded bricks only from h ownership (margin 0) and from the previous step's particle support. A seed has no previous support.
- The fix: this head's `TRIG` is a brick source. Every other seed tile is within one tile of a tile with heat at the last setMark, which is in `ACT.RES`. A seed's support tiles then lie within 8 cells of a source box, inside the margin. This is not a margin change.

**Gates** on 0be9d9f4 plus this change. All runs rc 0, with no 00000206.
- Zero latches on:
  - the tiered dam, 300 steps (check slot);
  - all-h, 200 steps;
  - the 165-step fenced and pipelined timing runs.
- The fix was also clean on the ddfecfc5 base, in the same four runs.
- GPU suites, 37/37:
  - brick-engine, brick-set-tiers, brick-census, census-x8, brick-transfer, brick-owner, levelset-release, levelset-coarse, brick-region-heat;
  - headMatchesOracle and the tiers test check `residencyOracle`'s new rule bitwise.
- Unit tests 1836/0; check:types shows only the 3 lane-n errors.
- Drift on the tiered dam: −0.0504 at step 20, −0.0098 at step 300. Before: −0.0555 and −0.0099 (bT300, the old rule on 04298677 + F_LIQ). All-h: −0.0108 at step 200 against −0.0108.

**Residency**, tiered dam over 300 steps:

| | mean liveF | peak liveF | churn (enter+leave)/step |
|---|---|---|---|
| bT300, old rule (04298677 + F_LIQ) | 3170 | 4820 | 138 |
| This rule, 0be9d9f4 base | 3878 | 4851 | 98 |

Peak liveC is 125 in both. The 12-cell margin on every source costs about +22% mean resident F; the per-source attribution and the margin's read-by justification are the next item.

## 2026-11-16: census replay (task 4, item 3, the rest of the real D)

The old regionList and replay (063 census l.1971–2089) are built as one launch, `regionDeep` (C_ALL X8: a workgroup
a region, a lane a tile). It runs after regionSpread and before censusFine. The head's deep pass moved into it from
censusCoarse, since replay needs deep before classification. A region replays (RG MODE 1) when it is deep COLD and
its stamp is unbroken:
- STAMP is the deep epoch + 1, which the old rule also uses;
- GEN is the page key, the old generation;
- two checks the old certificate covered as "widths and masks": OWN, its 64 tiles' h bits, and PAGES, the
  residency of the 27 pages about its page.

Then comes the fresh velocity-key check:
- a 4h tile's face jobs run serially on its lane;
- an h tile's 64 cells run on the workgroup, one tile at a time, until a key differs;
- any change demotes the whole region (MODE 2).

A replayed region's tiles publish their stored words: CROSS and GAP from the new RCROSS and RGAP, FIN from W1, the
crossing count, and the reach from the stored keys at this step's dt. W0, W1, KEYS, NEAR and DEF stand. censusFine
skips a replayed tile, and censusCoarse skips a whole slab when its four regions replay and a tile otherwise.

- RCROSS and RGAP are needed because censusImportance edits CROSS and GAP in place; the old engine stored its
  records apart (CC_RECORD).
- CENSUS.RES is dropped: resMark has not read it since 258aae94. Every word after it moves down 160, and the stride
  is now 19744 with RCROSS and RGAP (brick-abi.md §2.5; x8 compares the frozen words below FROZEN.RES).
- Stated, as the old replay: the records are the last classification's at that step's dt (GAP's speed byte, the
  fast test), and DEF is stored.

Gates (check slot, HEAD 258aae94 + this; rc 0, no 00000206):
- **The fixture gate** (brick-census-dam.test.ts, the real D's head on the old engine's captured states):
  - The old replayed regions go in deep COLD, with stamp, records and decidedPrev. Tiers are COLD on them and their
    neighbours, HOT elsewhere, which reproduces the old deep set.
  - Frame 14: 17 of 17 regions replay and no other does; WANT is the old WANT on all 32768 tiles, the 1088 replayed
    tiles included.
  - Frame 43: 3 of 3 regions (192 tiles), with the same result.
- **The reservoir under gravity** (brick-region-heat.test.ts, 50 steps): none of the 64 deep regions replays. Every
  one holds its stamp, and every one has a moved key: the projection's residual moves its faces' bits each step,
  largest 8.9e-6 m/s. Exact keys never hold in a liquid interior under gravity. The old fixture's replays were
  regions whose keys were bit-identical.
- **The reservoir at g = 0** (velocity exactly at rest):
  - 256 of 400 regions replay, and censusCoarse skips 256 slab workgroups. No h tile is deep, so censusFine skips
    none.
  - The stored words stand.
  - A fresh classification of the same state (every stamp cleared) differs in 0 of 114688 replayed record words,
    with equal keys and WANT.
  - A stored key moved on one 4h tile demotes that region alone, which takes the key afresh; 255 others replay.
  - save/restore across the replay reproduces the modes, the records and WANT.
- census 3/3 (D ≠ 0 and the oracle), x8 3/3, dam fixtures 2/2, engine 17/17, tiers 2/2, region heat 3/3, and the
  replay test, which passed on its own run after the g = 0 split (a test-only change); unit 1836/0; types clean except
  lane N's 3.

Timing slot (`replay/tm/`, one run each; kernel ms/step over the fenced median's window):

| | base 258aae94 | replay |
|---|---|---|
| reservoir, g: censusFine / censusCoarse / regionDeep | 0.000 / 0.408 / – | 0.015 / 0.495 / 0.109 |
| reservoir, g: census total | 0.66 | 1.06 |
| reservoir, g = 0: censusFine / censusCoarse / regionDeep | 0.022 / 0.517 / – | 0.007 / 0.066 / 0.138 |
| reservoir, g = 0: census total | 0.71 | 0.31 |
| dam: censusFine / censusCoarse / regionDeep | 1.223 / 0.306 / – | 1.245 / 0.284 / 0.066 |
| dam: census total, window kernel mean | 1.98, 96.9 | 2.08, 96.2 |
| dam unprofiled fenced median / mean / p95 | 100.0 / 97.1 / 104.1 | 100.8 / 97.9 / 101.8 |

Replay saves 0.45 ms of censusCoarse where it fires and costs regionDeep's launch, 0.07–0.14 ms. On the dam and on a
reservoir under gravity it does not fire. The reservoir's censusDecide moved 0.09 → 0.34 ms in this one run, and
that kernel's code did not change.

**Open (for the coordinator):** exact-key replay only pays where the velocity is bitwise still. A tolerant key
check is the one change that would make it fire in a liquid interior. It would compare each velocity word's effect
on the records (the reach, GAP's speed byte, the fast test, the approach and boundary scores) against the move. That
is a new certificate and a ruling, not a port.

## 2026-11-16: residency margins from a read table (owners 20, support 0, seeds 6, pages 16)

The coordinator's ruling: each source class's margin has to be what some stage reads past that class's tiles. The table gives every row with its file and line, and each margin is the largest row for its class. The 12-cell margin on every source (F_ACT's dilation, 258aae94) is retired.

**Margins** (brick-residency.wgsl.ts `RES_MARGIN`, `resMark` builds one box per class; brick-abi.md §5.4):

| class | tiles | move | margin (cells) |
|---|---|---|---|
| owners `ACT.RES` (bits 10..17) | an h owner (tiered), a φ < 0 vertex, a particle | flow | 20 |
| support `PSUPPORT` word 0 | pJoin's support tiles | max(travel, flow) | 0 |
| seeds `ACT.SEED` (new, bits 18..25) plus this head's `TRIG` | heat at the last setMark, previous head's `TRIG` | none | 6 |
| pages (tiered) | 4h-owned liquid tiles | flow (C plane faces) | 16 |

- flow is `(BOUND_F + |g| dt) dt / h`.
- Heat tiles used to be in `ACT.RES` with the owners' 12. setMark now splits them out into `ACT.SEED`.

**The read table.** Distances are in cells past the class's tiles. Files: L levelset-kernels, C levelset-coarse, P particles-kernels, F particles-formulas, V velocity-kernels, VF velocity-formulas, G fluid-grid; all are .wgsl.ts.

Owners, reads past an h tile:

| reach | stage | file:line |
|---|---|---|
| 20 | lDilate's kept far air (> 5 tiles, lAdvected) | L 137, 344, 368-393 |
| 16 | lDilate's collar (≤ 4 tiles, lProtected) | L 400-407 |
| 12 | lDilate's reach2/ext (≤ 3 tiles) | L 498, 547 |
| 8 | lDilate's reach1 (≤ 2 tiles) | L 498, 547 |
| 9 | lcRetire | C 491-494, 546-553 |
| 7 | the DEPTH chain through lDistance and lNear | G 498-507, P 120-124 |
| 5 | lNear's 11³ box | L 459-470 |
| 5 | lDistance and lPublish | L 490-531 |
| 3 | uExtend | V 695-762 |
| 3 | the sampler's UNIT taps | G 450-453 |
| ≤ 2 | the sampler's fine weight | G 466-479 |
| 1 | the + faces | V 304, 501, 664, 772, 831 |
| 1 | setMark's 1-ring | brick-set 124-135 |
| 1 | tile-word stencils (lUnion, uTransfer, pTiles, lDilate's heat, lClear) | L 250-290, 378-407; V 475-489; P 287-301 |

- lDilate is separable (x, then y, then z), and each pass stores its intermediates per brick.
- An absent brick counts as "no source" (L 370). So anything within R = 5 tiles has to be resident, intermediates included.
- That makes 20 the owners' margin. It equals the retired RESIDENCY_MARGIN_TILES (5 tiles).

Support: every read of the next move lies within 1.5 cells of the particle, inside the ±2 support pJoin records (P 417-426). So the margin past the move is 0.

| reach | read | file:line |
|---|---|---|
| ≤ 1.5 | G2P taps | P 173, G 456-463 |
| ≤ 1.5 | pDeposit, quadratic B-spline | VF 33-48 |
| ≤ 1.5 | pSupported | P 76-82 |
| ≤ 1.5 | separation | P 108-119 |
| ≤ D + 1.5 | Heun samples | P 175-195, F 68-73 |
| e ± 0.25 | split | P 210 |
| ≤ 1 past e | endpoint reads | P 203-208 |
| ≤ 1.25 | surface shift | P 120-124 |

Seeds:

| reach | reason | file:line |
|---|---|---|
| ≤ 4 | pSeed's cell needs FT_T > 0, half the 27-tile max of lHeatNow, which is FT_T's only source; so the cell is within a tile of a seed tile | P 320; L 354-357, 378-393 |
| 0 | the site is inside the cell | F 229-235 |
| 2 | the first pJoin, before any move, records ±2 support | P 417-426 |
| = 6 | total | |

In brick-abi.md the bound had read "within 6 cells", and 258aae94's progress entry said 8. The table confirms 6.

Pages: lcAdvect's band (|φ| < 8) lies within 8 cells of the 4h liquid tiles' box. Its foot stencil and cubic taps read 8 past that, which gives 16.

| reach | stage | file:line |
|---|---|---|
| 16 | lcAdvect foot stencil | C 300-303 |
| 16 | lcAdvect cubic | C 333-340 |
| 16 | lcClear's collar | C 231 |
| 12 | lcRetire | C 548-553 |
| 12 | uExt4Seed writes | V 787-842 |
| 8 | uExt4 | V 868-915 |
| 8 | pressure rows, by storage | pressure-composite 569-590 |
| travel + 4 | vcAdvectPage | V 272-289 |

The old 12 was 4 short.

**Caveats, stated.**
- **BOUND_F.** At the head, BOUND_F holds only project's face maximum. setMark clears pCount's particle speeds (setMark's last store) before project raises the word again. The brick-abi.md and header text that said otherwise are corrected.
- **Travel.** The recorded travel and the flow bound the next move only as far as the extended field stays under the brick's face maximum. Support 0 rests on that.
- **A shortfall fails explicitly.** pJoin's REACH latches, and none fired in the runs below.

**Gates.** The tree c7 is 258aae94 plus these files. Every run is rc 0, with no 00000206.
- Zero latches on all four runs:
  - tiered dam, 300 steps;
  - all-h, 200 steps;
  - the 165-step fenced and pipelined timing runs.
- GPU suites 37/37:
  - The first round failed brick-engine's enter/leave and POOL_F tests. Their 64×32×64 lattice is all inside a 20-cell reach, so no brick could enter.
  - The lattice is now 128×64×128, and brick-engine then passes 17/17.
  - headMatchesOracle and the tiers test check `residencyOracle`'s per-class rule bitwise.
- Unit tests 1836/0; check:types has only the lane-n errors.
- Drift on the tiered dam: −0.0532 at step 20 and −0.0100 at step 300, against −0.0504 and −0.0098 for c6 (258aae94). All-h: −0.0107 at step 200, against −0.0108.
- I rebased onto d02947dc (after lane P's 1bea686f, lane F's 5f2717bc and the census replay) as tree c8 and ran it again:
  - zero latches on all four runs;
  - tiered dam 300: mean F 4408, peak 4913, churn 62/step, drift −0.0537 at 20 and −0.0101 at 300;
  - all-h 200: mean 4531, peak 4913, drift −0.0105;
  - fenced median 72.3 ms, pipelined 71.7 ms a step;
  - suites 36/37. The miss is brick-region-heat.test.ts aborting the process (SIGABRT, no 00000206, after "kept 128 of 128"). It aborts the same way on d02947dc alone (tree h2), so it predates this change.

**Residency** (F bricks; the pool and the lattice are both 4913):

| | tiered 300: mean | peak | churn/step | all-h 200: mean | peak | churn/step |
|---|---|---|---|---|---|---|
| bT300, old rule (resSeed/resRequest on 04298677 + F_LIQ) | 3170 | 4820 | 138 | | | |
| 258aae94, 12 on every source | 3878 | 4851 | 98 | 4220 | 4913 | 59 |
| this commit | 4419 | 4913 | 66 | 4508 | 4913 | 48 |

**Attribution** on the tiered dam. Each resident F brick after head N goes to the first class that covers it:
1. owner: an h-owned tile;
2. ring: censusDecide's 27-ring;
3. reach: a box without the margin;
4. margin: covered only through a margin.

| step | 258aae94: live | owner | ring | reach | margin | this commit: live | owner | ring | reach | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 4194 | 921 | 1178 | 1105 | 990 | 4156 | 934 | 1193 | 1082 | 947 |
| 60 | 4843 | 1573 | 2121 | 644 | 505 | 4909 | 1574 | 2090 | 625 | 620 |
| 150 | 4203 | 831 | 1614 | 460 | 1298 | 4630 | 756 | 1524 | 332 | 2018 |
| 300 | 3279 | 543 | 1286 | 35 | 1415 | 4341 | 546 | 1354 | 12 | 2429 |

- The trig and heat classes are 0 in both: they are inside the owners' and the support boxes.
- The owners' 20-cell margin is the whole increase. Memory does not scale here: the tiered dam reaches the full lattice from step 21 to 60.
- The cheapest cut is an lDilate that resolves an absent brick through its owner (C) instead of "no source". That is lane F's kernel. Each of its intermediates would also need to be covered or read from C. With that, the owners' largest row is lcRetire's 9.

**Head and step** (timing slot, the tiered dam):
- Residency kernels:
  - resMark: 0.19 + 0.14 ms (2 passes), against 0.15 + 0.13 on 258aae94;
  - claims and leaves: about 0.08 ms.
- Window kernel sum, steps 18-23: 97.0 ms against 95.4.
- Fenced step, 165 steps: median 87.7 ms against 85.4.
- Pipelined: 100.2 ms per step against 88.2. The extra resident bricks over steps 21-100 cost about 12 ms per step in F_ALL's stages.

Per area (rowsWindow, steps 18-23), ms per step:

| area | 258aae94 | this commit | §6 budget |
|---|---|---|---|
| velocity | 40.49 | 40.88 | 1.6 |
| level set | 16.83 | 17.03 | 2.0 |
| pressure solve | 13.11 | 13.54 | 3.0 |
| pressure setup and projection | 9.02 | 9.25 | 1.0 |
| move | 8.57 | 8.61 | 1.5 |
| brick set | 3.35 | 3.59 | 0.6 |
| transfer and seeding | 3.39 | 3.47 | 1.0 |
| region D | 0.60 | 0.65 | - |
| total | 95.4 | 97.0 | ~18 |

Letters and seep do not run on the brick engine yet (brick-scene.ts `BRICK_ENGINE_SCENES = ["high-resolution-dam-break"]`), so this commit has no cross-scene check.

## 2026-11-16: the restore abort

brick-region-heat.test.ts aborted (SIGABRT, no 00000206) in the reservoir test's `restore()`, in Dawn's `BufferBase::UploadData` under `queue.writeBuffer`. The disassembly of dawn.node shows a check there: the upload reservation's size must equal the write's size, or the process aborts. One instrumented run showed which write failed. It was pmgF's, the first buffer whose byte size, 34607172, is not a multiple of 8 or 16. Every buffer written before it was a multiple of 16. `restore()` now writes each buffer as its 16-byte multiple, then the tail. After the fix the file passes 4/4 (save/restore: 256 regions, tier/deep 0 and heat 0 differ; replay unchanged), with no 00000206. pressureDelta needs no fix: lane P stores Δ(p/ρ) (08dac85d), so regionHeat's dt²/h² (brick-region-heat.wgsl.ts:110) is the old dt²/(ρh²) on Δp.

## 2026-11-16: the owner margin 20 -> 9 (lDilate reads through C), and the dam's pools

Lane F's lDilateC (9b7761f7) lets lDilate read a tile no F slot holds from its page's DIL4 word. So lDilate no longer
sets a brick reach in the tiered engine, and the owners' brick margin falls to the next row of the read table.

**Read table.** File:line is in the brick-residency.wgsl.ts header.

Owners' brick rows (tiered):
- lcRetire: 9 (C l.537-553). It builds through lcBuildDistance (C l.490-511; fgReach1 and FC_NEAR at the 4h cell) only within lcStencilFine's tile of h. This is the max, so the margin is 9.
- The DEPTH chain through lDistance and lNear: 7 (L l.575-586).
- lcCouple: 7 (C l.400-436; F's item 2). It reads fgReach1, FC_NEAR, SURFACE_TILE, FC_KEY and the bin counts only where θ > 0. θ (C l.389-396) is 0 farther than a tile from an FT_THETA tile, and pTiles writes those within a tile of a particle's tile (P l.285-301): 4 + 2 + 1 cells.
- pTiles' FT_THETA and SURFACE_TILE: 4 (P l.285-301; F's item 1). The 4h F tiles about the particles carry their sample support. The PSUPPORT ±2 support class (margin 0) keeps the deposit support.
- uExtend 3, UNIT taps 3 (fluid-grid l.470-474), and the 1-cell stencils.
- lDilate: 20 (L l.346-410). Tiered, it now reads through C: lDilateC (L l.412-511) writes the sources, DA and DB to DIL4, so this row sets no brick reach.
- F's item 3: an h-owned tile with no F slot latches FAIL.TIER 0x4c440001 (L l.498). An h-owned tile is an owner, so its brick is resident.
- All-h has no DIL4, so the all-h owners keep 20.

Pages' row (tiered): lDilate through C, 20.
- Every source (a φ ≤ 0 corner, or SURFACE_TILE) is an h-owned tile, a particle tile or a 4h liquid tile.
- A tile with DA or DB short of R lies within 20 cells (per axis) of a source.
- So the owners' pages are grown by flow + 20, and the page sources are raised from 16 to 20 (lcAdvect's 16 is inside).
- Every such tile is then in a resident page. A tile in no resident page reads as "no source", which is the saturated value it would have anyway.

`RES_MARGIN = { own: { tiered: 9, allH: 20 }, ownPages: 20, support: 0, seed: 6, page: 20 }`. resMark and the oracle each carry an owners' page box apart from the brick box.

**Gates** (on HEAD 7891be87 plus this change, with git-archive trees; lane F's c16bb49b, read-only fcell binds and bitwise by its message, landed after; every run rc 0 with no 00000206):
- Suites: brick-engine, brick-set-tiers (the residency oracle), census, census-x8, transfer, owner, levelset-release, levelset-coarse, levelset-kernels and region-heat. 41 tests, one fail at first.
- That fail was region-heat's census replay under gravity, which asserted 0 replayed regions. The page sources' 20 cells now keep page row y = 2 (cells 64-95) above the reservoir (surface at 40, box top 44 + 20.4 > 64).
  - Its deep regions are still air with no moving key, so they replay: 64, all at region y 5.
  - 4 "deep" regions that classified with no key moved lay in page (3,2,4) past the lattice (z ≥ 128; rgInLattice false, mode 2).
  - The test now skips out-of-lattice regions, as regions() does, and asserts that every replayed region lies above the surface. Rerun: 4/4.
- Tiered 300 (fenced, check slot): no latch (no FAIL.TIER, REACH or POOL). Drift -0.0101 at step 300, against -0.0099 before.
- All-h 200: no latch. Mean liveF 4556, peak 4913.
- 165-step fenced and pipelined (timing slot): no latch.
- Unit: 1842 pass, 0 fail.
- check:types: the only errors are the committed tools/experiments/lane-n/*.mts imports of a missing scratch tree. They are not this change's.

**Breakdown** (tiered 300; attrib classes; before is HEAD at margin 20/16, after is this commit at 9/20, with --margins as each):

| step | before: live | owner | ring | reach | margin | after: live | owner | ring | reach | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 4199 | 929 | 1173 | 1364 | 733 | 3845 | 947 | 1304 | 1288 | 306 |
| 60 | 4907 | 1577 | 2162 | 667 | 501 | 4750 | 1534 | 2077 | 762 | 377 |
| 150 | 4488 | 772 | 1511 | 332 | 1873 | 3624 | 796 | 1670 | 442 | 716 |
| 300 | 3832 | 510 | 1155 | 16 | 2151 | 2540 | 526 | 1250 | 13 | 751 |

liveF figures (before, then after):

| | before | after |
|---|---|---|
| mean, steps 1-300 | 4224 | 3167 (-25%) |
| mean, steps 201-300 | 3816 | 2458 (-36%) |
| peak live | 4913 | 4750 |
| slot high-water | 4913 | 4780 |
| churn, (enter + leave)/2 per step | 34.9 | 68.1 |

- The trig and heat classes are 0 in both.
- The step 60 peak barely moves. The surge then spans the lattice, and its owners and rings alone are 3611 of 4750.

**Timing** (timing slot, tiered dam, before then after):

| | before | after |
|---|---|---|
| fenced, 165 steps: median | 65.3 ms | 61.5 ms |
| fenced, 165 steps: mean | 71.9 ms | 67.1 ms |
| pipelined, per step | 68.6 ms | 63.7 ms |
| resMark (2 passes) | 0.18 + 0.13 ms | 0.16 + 0.12 ms |
| lDilateC (2 rounds) | 0.10 + 0.11 ms | 0.08 + 0.09 ms |

Per area (rowsWindow, steps 18-23, window mean), ms per step:

| area | before | after | §6 budget |
|---|---|---|---|
| velocity | 26.38 | 25.08 | 1.6 |
| level set | 15.31 | 14.53 | 2.0 |
| pressure solve | 9.69 | 9.40 | 3.0 |
| move | 8.71 | 8.48 | 1.5 |
| pressure setup and projection | 5.16 | 4.96 | 1.0 |
| brick set | 3.54 | 3.45 | 0.6 |
| transfer and seeding | 3.53 | 3.50 | 1.0 |
| region D | 0.72 | 0.69 | - |
| total | 73.0 | 70.1 | ~18 |

The cross-run floor is 0.97 ms before and 1.27 ms after (means). Letters and seep do not run on the brick engine yet, so there is no cross-scene check.

**Pools** (lib/core/scenes.ts, the high-resolution dam):

| run | slot high-water F | live peak | C |
|---|---|---|---|
| tiered 300 | 4780 | 4750 | 125 |
| tiered 300, second run | 4786 | 4754 | 125 |
| all-h 200 | 4913 | - | 125 |

- The lattice holds 17³ = 4913 bricks and 5³ = 125 pages, so no run can take more.
- f stays 4913: the all-h peak exactly, and the tiered peak × 1.027.
- c goes from 192 to 125, which drops 67 page slots no run can use.
- Gate at the authored pools (no --poolF override; tiered 300): no POOL_F or POOL_C, mean liveF 3032, high-water 4786 F and 125 C.

## 2026-11-16: the 4x-wider dam (memory must scale)

The dam's liquid box in a container 4x wider in x (512 x 128 x 128 cells, whole pages), tiered, lane F's stages, the composite solver, HEAD c6516bce (a HEAD snapshot; the working tree's uncommitted pressure edits excluded). Test-only: `lib/methods/narrow-band/brick-wide-dam.test.ts`. Runs: ~/code/me/fluid-scratch/lane-e/wide/ (n1 and w4b: timing slot, fenced, a header read after every step; an1 and aw4: residency attribution, `tools/attrib.mts`, margins 9/0/6).

**Memory follows the pools.** At the same pools the two lattices allocate 839575184 and 839581328 bytes: the difference is 4 x 1536 words, the directory's page window (8x8x8 to 32x8x8 pages). Every engine, stage and solver buffer is pool-sized.

**Residency, lists and work follow the fluid.** Until the front reaches x = 128 (about step 12) both runs hold the same fluid and the same residency:

| step | liveF narrow / wide | F_ACT | F_LIQ | C_ALL | particles |
|---|---|---|---|---|---|
| 1 | 667 / 667 | 667 / 667 | 68 / 68 | 80 / 80 | 71595 / 71595 |
| 10 | 2243 / 2244 | 1906 / 1899 | 366 / 362 | 96 / 96 | 229157 / 228143 |
| 15 | 2621 / 2857 | 2316 / 2434 | 550 / 577 | 88 / 93 | 346264 / 365477 |
| 20 | 3913 / 4125 | 3698 / 3586 | 685 / 812 | 115 / 125 | 487510 / 549029 |
| 30 | 4531 / 7486 | 4412 / 6868 | 915 / 1292 | 125 / 217 | 894021 / 1241358 |
| 40 | 4500 / 12764 | 4499 / 11953 | 1234 / 1797 | 125 / 316 | 1030171 / 1867098 |

After it, the narrow dam piles against its wall and the wide sheet keeps spreading. Its liquid footprint (F_LIQ +46% at step 40) and its band (particles +81%: the band follows the surface) are real fluid. At step 40 no brick lies past the fluid's front plus the residency rule's box (fluid to brick x 53, live F to 54, pages to 13 of 16): the empty far end of the box costs nothing.

Step ms follows liveF in both boxes: 14.5-19.1 ms a thousand live F bricks wide, 15.5-21.6 narrow (fenced, timing slot). No term grows with the box.

**The finding: the reach class's flow box is isotropic.** Attribution (live / owner / ring / reach / margin):

| step | narrow | wide |
|---|---|---|
| 15 | 2661 / 669 / 978 / 231 / 783 | 2913 / 725 / 1013 / 216 / 959 |
| 25 | 4288 / 1172 / 1708 / 1039 / 369 | 5596 / 1404 / 1729 / 1772 / 691 |
| 35 | 4477 / 1392 / 1778 / 840 / 467 | 9550 / 2116 / 2764 / 3761 / 909 |
| 45 | 4574 / 1869 / 1914 / 458 / 333 | 14216 / 3006 / 3913 / 6957 / 340 |

Owners and ring grow with the sheet (x1.6 and x2.0 at step 45). Reach grows x15. The flow term, (max BOUND_F + g dt) dt / h, is one scalar a brick applied on every axis and both signs: 21.8 cells at step 40 (8 m/s spray at the front). A thin, fast, horizontal sheet is therefore held about 3 bricks deep above and below, as well as ahead. A signed per-axis bound (as PSUPPORT's travel already is for particles) would cover the motion without the vertical thickening. It is the lever if the wide case's memory matters; it changes no physics. Its safety argument (every read and write of the step inside the box) needs its own read-by table first.

**Lattice-sized items (findings):**
1. The directory's page window: one word a page of a power-of-two window (6 KB more wide). Bounded at M6 by the moving window.
2. The app's presentation (brick-present.ts; not the engine, and allowed as an interim by structural §2.10):
   - three r32float 3D textures at the lattice's size (vertexPhi (n+1)^3, volume, openFraction): 25.6 MB narrow, 101 MB wide;
   - `shown`, a word a lattice brick;
   - two dispatches a presented frame sized by the lattice's bricks (18x18x18 and 17^3 narrow; 66x18x18 and 65x17x17 wide). Each workgroup exits early off the fluid.
3. Values only, no sizes: NB_LATTICE (clamps), NB_PHI_FAR = max(lattice) (the far φ), RQ_PAGES (resMark's clamp).

## 2026-11-16: the signed per-axis reach bound (parked, coordinator's ruling (b))

**Correction to the entry above.** The x15 "reach" class overstated what the flow term costs. The attribution's classes overlap: a brick in a source's move box but also inside some other source's margin box counted as reach. A new class, moveOnly, isolates the term's true cost: live bricks that are neither owner nor ring and that no source's margin box with zero move covers. On the wide dam at step 45, moveOnly is 958 of 14115 live F bricks (7%). The rest is the spray cloud itself (F_PART 4277 bricks) with its fixed margins (owner 9, the census's 27-ring). Run-to-run noise on the same code is about ±4%: 13557 and 14115.

**What was built (scratch only, not landed).** ~/code/me/fluid-scratch/lane-e/signed/tree, a git repo on 09135ed0:
- 8b572ea, signed BOUND_S:
  - a set region, 6 words an F slot: the largest -u and +u an axis;
  - projectF raises it (6 workgroup atomics; early exits unchanged); setMark and setClaimBricks clear it;
  - resMark takes flow = (u_side + max(+-g_a, 0) dt) dt / h for each axis and side, and the page sources the same from their C faces;
  - the oracle matches; the signed read-by table is in the brick-residency header, with file:line rows;
  - GPU tests 35/35: brick-engine (the residency oracle), brick-set-tiers, composite-mgpcg "live lists", brick-abi.
- 51c694d, experiment: pJoin's ballistic PSUPPORT travel made signed, (v + g dt) dt / h an axis and sign, in place of lrParticleTravel's |v| dt / h all six ways.

**Attribution** (tools/attrib2.mts, with the moveOnly and reachTravel splits; runs in signed/*.log):

Wide 512 x 128^2, step 45:

| run | live F | moveOnly | owner | ring | reach | margin |
|---|---|---|---|---|---|---|
| base 09135ed0 (aw4b) | 13557 | - | 2944 | 3877 | 6330 | 406 |
| base 09135ed0 (aw4b2) | 14115 | 958 | 2964 | 3811 | 7003 | 337 |
| BOUND_S only (aw4s) | 14673 | - | 3034 | 3878 | 7483 | 278 |
| BOUND_S + signed ballistic (aw4sb2) | 12867 | 47 | 2910 | 3880 | 4321 | 1756 |

- FAIL = 0 in every run.
- Drift at step 46: -10.44% base, -10.35% signed.
- Narrow 128^3, BOUND_S only (an1b vs an1s): live F 4394/4463/4580 against 4312/4500/4542 at steps 25/35/45, within ±2%; step 15 was 2614 against 2291.

**Ruling.** Parked:
- the gain is 5-9% on the wide dam and about 0 narrow, against ±4% noise;
- it costs 6 words a slot, the atomics in projectF and a parity deviation (NBF_SUPPORT_TRAVEL_WGSL);
- the latch gates were not run.

The spray cloud's fixed margins are the bigger lever, next.

## 2026-11-16: the spray cloud's owner margin (attribution only, not built)

**The question.** Does a particle-only owner tile need the tiered owner margin of 9? That margin is lcRetire's row, and lcRetire builds only within lcStencilFine's 1 tile of an h-owned tile (levelset-coarse l.537-553). A particle owner's own rows are smaller: lcCouple 7 (4 + 2 + 1 past a particle's tile, C l.389-436) and pTiles 4 (P l.285-301).

**The what-if.** tools/attrib2.mts adds an own7 cover:
- owner tiles neither h-owned nor within a tile of one take margin 7;
- every other box is as the rule has it.

saved7 counts the live bricks only that change would drop: not owner, not ring, outside every remaining box. Runs: base 09135ed0, tiered, composite solver, signed/aw4m and signed/an1m.

| box, step | live F | owner tiles near h / particle-only | saved7 |
|---|---|---|---|
| wide 25 | 5558 | 8329 / 277 | 0 |
| wide 35 | 9160 | 12933 / 824 | 70 (0.8%) |
| wide 45 | 13838 | 18056 / 2314 | 26 (0.2%) |
| narrow 25 | 4320 | 7265 / 461 | 7 |
| narrow 35 | 4379 | 8537 / 672 | 16 |
| narrow 45 | 4581 | 11932 / 1688 | 0 |

**Not built.** The bar was 15% of live bricks; the saving is under 1%. The census makes the spray h-owned, so 89% of the wide owner tiles at step 45 are within a tile of an h-owned tile, and the 9 applies to them. The wide residency is the h-owned spray's own 9-cell shell (owner 21%, ring 28%, the margin boxes the rest; moveOnly 819, 6%). The lever there is the spray's tier, the census's WANT, not the residency's margins.

## 2026-11-16: the move, pSupported's liquid cells once a brick (pLiquid)

The move area was 8.24 ms against §6's 1.5 at 327d4280 (tiered dam, steps 18–23; `fluid-scratch/lane-e/move/tm/head-p{1,2}`).
pMove alone was 5.56 of that. Every figure below is from the timing slot. The frozen numbers come from
`move/tools/frozen.mts` on the tiered dam, frozen at step 20. Logs are in `move/logs/`.

**Attribution of the old pMove** (frozen; knockouts, each one removed alone; `attr1`/`attr2`, `probe-sep*`):
- separation 2.1 ms. Of that, the candidate loads are 0.27 (`noCand`: every candidate reads record i). The 8 runs' fgRun lookups are about 0.2. The rest is the loop and its arithmetic, about 53 candidates per coupled record.
- trace samples 1.3 and the G2P sample 0.45. 97% of samples take the fine interpolant.
- pSupported 0.6 as a knockout. Read from precomputed bits it saves 1.3 (below).
- surface shift 0.77, split block 0.36, depth at q 0.25, merge 0.1, heat about 0. The floor, with everything removed, is 0.63.
- The register cap is 512 threads. Removing the separation and the shift leaves it at 512, and removing the trace lifts it only to 576 (`msl-split/`). Splitting the move by register budget therefore buys no occupancy here.

**Tried and not landed** (the patch is in `move/patches/items-staged.patch`):
- **Bin items.** pScan lists items of up to 256 records of one bin in BIN_ITEMS. pMove takes one workgroup per item, and pMoveTail moves the seeds past the bins. P_SEEDS/P_SEEN check that every binned record is seen.
  - Bitwise equal to the old chain (`chain-dev1`).
  - Slower: pMove 7.35 against 6.02 ms. Records average 184 per 256-record item, so lanes run at 72% fill.
- **Items plus staging of the brick's ring.** The ring staged was the window, the liquid cells of o-1..o+8, the depth vertices o-1..o+9 and the UNIT taps o-1..o+8.
  - pMove 8.77 against 5.93 ms. Staging alone costs 1.96 ms (`chain-dev2s`).
  - Record time saved: liquid 1.43, depth 0.56, UNIT about 0, window 0.1.
  - It differs from the old chain by one ulp in the depth word, on about a third of records (`chain-dev2n`). The cause is fgDepth's sum being contracted differently by the compiler, not the staged values (the cause was not confirmed by a check).
  - At about one particle per cell, a 10^3 box restaged for every item does not pay for itself.
- **Separation loop restructures.** Row-merged runs (S2l), one flattened loop, and both together.
  - All are bitwise equal (fmoved), and all are slower: 6.07, 6.53 and 6.04 against 5.87 ms (`sep-{a,b,c}`).

**Landed: pLiquid.**
- One workgroup per F_ALL brick, placed before pMove. It computes fgLiquidCell of the opening φ for the brick's 512 cells, from a 9^3 vertex ring through the window, and stores them in P_LIQUID (16 words an F slot).
- pSupported reads the bit where a slot holds the cell and fgLiquidCell elsewhere. The bits are valid only within the step's move.
- **Frozen chain** (uCache, pLiquid, pMove, pScan, pScatter, pTiles against HEAD's; `chain-dev4`):
  - Bitwise equal: set, fcell, and the bank dst per-cell multiset (85,470 cells, 473,903 records). The old-vs-old control is also the same.
  - pMove 5.94 → 3.60 ms, pLiquid 0.70 ms; chain 6.45 → 5.87 ms.
- **Gates** (check slot, dam-300, tiered composite, poolF 4913, judged against ref-2761350a; `move/gate/`):
  - FAIL word 0 at every sample, for cand and head.
  - D8 fails at head as well as cand, 1.17 and 1.22. This is the same pre-existing failure as the real-D entry.
  - Every other gate passes on both:

    | | D1 | D2 | D3 | D4 | D4c | D5c | D5ce | D5k | D5p | D5e | D6 | D7t | D7d | D8c | D8 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | head | 0.55 | 0.06 | 0.55 | 0.05 | 0.00 | 0.17 | 0.05 | 0.15 | 0.04 | 0.03 | 0.56 | 0.32 | 0.13 | 0.39 | 1.17 F |
    | cand | 0.55 | 0.06 | 0.38 | 0.06 | 0.00 | 0.17 | 0.05 | 0.23 | 0.06 | 0.06 | 0.56 | 0.32 | 0.13 | 0.34 | 1.22 F |

  - Drift at step 300: head -1.01%, cand -0.99%.
- **Timing slot** (`brick-dam-floor.mts --tiered --solver=composite --steps=24 --poolF=4913`, steps 18–23; two profiled runs and one unprofiled run per tree; `move/tm/`):

  | | head p1 | head p2 | cand p1 | cand p2 |
  |---|---|---|---|---|
  | move area (pLiquid counted in it) | 8.24 | 8.24 | 7.79 | 7.78 |
  | pMove / pLiquid | 5.56 / – | 5.57 / – | 4.39 / 0.69 | 4.46 / 0.70 |
  | window kernel sum, mean | 66.1 | 65.7 | 65.3 | 64.9 |
  | launches/step | 290 | 290 | 291 | 291 |

  - Unprofiled fenced mean: head 66.8, cand 67.6. That is one run each, and the kernel sums do not show the difference.
  - In the step, pMove gains 1.14 ms against 2.34 frozen. The net move-area gain is 0.45 ms. Most of what remains of pLiquid's 0.70 is bricks with no record within one brick, which pLiquid could skip; that is the next step.

## 2026-11-16: tiered against all-h on the dam (measurement; coordinator's ruling: tiered stays)

HEAD 617f9291 (lUnion cc86c9dc and lcCouple 617f9291 in), composite solver, pools as authored (f = 4913, every one of
the 17^3 bricks: all-h cannot latch F; its liveF peak is the whole box). One run each, rc 0, no 00000206 aborts, FAIL
word 0 at every sample. Runner ~/code/me/fluid-scratch/lane-e/floor/tools/brick-dam-floor.mts (`--tiered` or not);
runs, gates and logs in ~/code/me/fluid-scratch/lane-e/tier-vs-h/runs/.

Fenced step (timing slot, unprofiled, 165 steps, steps 13-165, n = 150), ms:

| | tiered | all-h |
|---|---|---|
| median | 56.2 | 58.6 (+4%) |
| mean | 62.0 | 60.5 |
| p95 | 90.4 | 73.2 |
| thirds | 78.4 / 57.9 / 49.8 | 68.9 / 59.1 / 53.5 |

Per area (profiled, rowsWindow steps 19-22, ms (launches); §6 dam budget):

| area | tiered | all-h | budget |
|---|---|---|---|
| velocity | 20.86 (7) | 19.33 (5) | 1.6 |
| level set | 14.09 (24) | 10.50 (17) | 2.0 |
| pressure solve | 9.22 (169) | 14.43 (169) | 3.0 |
| move | 7.42 (9) | 4.54 (8) | 1.5 |
| pressure setup and projection | 3.64 (39) | 6.80 (39) | 1.0 |
| transfer and seeding | 3.60 | 2.41 | 1.0 |
| brick set | 3.42 (35) | 4.59 (27) | 0.6 |
| certificate and region kernels | 0.68 | - | - |
| window kernel mean | 62.9 | 62.6 | ~18 |
| launches per step | 291 | 267 | |

liveF (check run, 36 samples 0-300): tiered mean 2458, peak 4703; all-h mean 3937, peak 4913.

Gates on ref-2761350a (check run, 300 steps), D1/D2/D3/D4/D4c/D5c/D5ce/D6/D7t/D8/D8c:
- tiered: 0.80/0.10/0.49/0.05/0.00/0.17/0.10/0.57/0.35/**1.16**/0.94 (D8 30-90 7.038e5 vs 6.309e5, limit 6.94e5);
- all-h: 0.70/0.01/0.45/0.35/0.85/0.28/0.32/**1.12**/0.50/0.66/**1.17** (D6 wet floor 10162 vs 10767 at 8 and 14154 vs
  14936 at 12; D8c 3.928e5 vs 3.516e5 at 16).

Reading: a speed tie (all-h +4% median, better mean and tail). All-h moves about 8.4 ms into pressure (solve and
setup) and takes 3.6 out of the level set, 2.9 out of the move and 1.2 out of transfer. Tiering's per-sample cost
about cancels all-h's 1.6x live set here. So the gap to the seep is the passes' per-brick cost, not tiering. The
coordinator ruled that tiered stays the engine's mode. A lead for later: tiered's early third is 78.4 ms against
all-h's 68.9, the early collapse where the tiered census and transfer work peaks.

## 2026-11-16: D8c, B (census deformation from the certificate) and the boot φ

The old engine and ours were compared side by side (old-census-dump --phidump/--phitiles against brick-census-e0
--from0, steps 1-9; ~/code/me/fluid-scratch/lane-e/cold/sbs, boot/phi5). D8c's early lag had two causes, and B is
not one of them:
- bootPhi capped air at PHI_AIR (6 h), where the old t = 0 φ is the uncapped analytic distance
  (initialLiquidVertexPhi);
- C vertex 96, tile 23's high corner, lives in page 3. That page was absent at boot and entered at 6, where the old
  engine kept 16 in page 2's high plane.
- Flattened taps give lcFullAdvected's foot gradient g_x = 0.69 against 1.94, so the foot sits deeper in slower flow:
  US read at the foot was 3.8 against 4.6 h/step, and the front lagged about 0.8 h/step. US itself matched within
  3% (census 6, faces 84-96).

Landed:
- **B (item 4, verbatim):**
  - certError writes census DEF (deformation, error, page key + 1, width);
  - ncDeformation saturates on a missing tile, a stale key or a width of 0;
  - ncPlanStore is gone;
  - all-h also runs certFine + certError at w = 1.
- **Boot:**
  - bootPhi is uncapped (NB_PHI_FAR, as initialLiquidVertexPhi);
  - a tiered boot also claims the pages that hold its pages' high corner planes (bootCornerPages, KEY_PAGE only);
  - entering C slots take PHI_AIR_C = 16.
- After the fix the front tracks the old engine (x old/ours): census 5 87.46/87.55, 6 91.78/91.91, 7 96.92/97.47.

Stated deviations (also in brick-abi.md):
1. Entering C takes PHI_AIR_C = 16 h. That is the old stored far-air value (nbAirStore, lcClear) and the least AIR
   an old claim takes (UniformPageInitialiseOptions.near). The old claim's own AIR is the box's far φ. With
   NB_PHI_FAR, the 4x-wider dam failed "residency follows the fluid" (step 3 liveF 958 against 866); 16 passes.
   The magnitude reader is the foot projection in lcFullAdvected (levelset-coarse.wgsl.ts:303-306) and lAdvect
   (levelset-kernels.wgsl.ts:155): offset = -c g/|g|^2 with g a +-4h (+-h) difference whose taps include air
   vertices, so the foot distance scales as 1/(air value). This is verbatim old behaviour, with the old engine's own
   box-dependent AIR.
2. Page retirement: from census 8 our front leads slightly (x 100/104 hold 20/24 against the old engine's 128). The
   old engine retires the corner page's interior at the next head, while the brick residency keeps a page it claims
   at step 1.
3. Absent vertices (nbCVertex on an absent page, nbFVertex on an absent brick, brick-owner.wgsl.ts:78/82) still read
   NB_PHI_FAR at this commit. Converting them to PHI_AIR_C, together with lane F's fsetinit-airc and fgphi-airc and
   the composite mirror (pressure-composite.wgsl.ts:153), was gated as "airc" (~/code/me/fluid-scratch/lane-e/airc):
   both wide tests pass, but D8 30-90 reads 7.01e5 against 6.31e5 (1.11). It is held until the retirement item.
   Out-of-lattice reads (:77, :86, composite :144) stay FAR.

Gates, dam-300 check on ref-2761350a (runs in ~/code/me/fluid-scratch/lane-e/{bgate,land,airc,oldair}/g). D8 and D8c
are upper bounds since 29c8dab8.

| tree | D1 | D6 | D8 | D8c (particles at 16) | wide 2/2 |
|---|---|---|---|---|---|
| HEAD + f-sh + B | 0.30 | 0.10 | 0.87 | low, 281.6k | yes |
| + boot, entering C FAR | 1.00 | 0.11 | 0.80 | 0.11, 347.8k | no (step 3 958/866) |
| + boot, entering C 16 (landed) | 1.00 | 0.11 | 0.86 | low, 303.3k | yes |
| + airc | 1.00 | 0.12 | 1.11 | low, 302.9k | yes |
| old engine, empty 16 h | 1.00 | 0.05 | 0.18 | low, 287.9k | - |

D1's step-6 front is 100 against 96 in every row from the boot on, the old engine at 16 h included (plan §4.6).

The wide-dam flake (dd405e3c): the engine is not bitwise reproducible. The boot's pressure solve differs first:
pressure-mgpcg.ts pRedAdd adds workgroup partials into 64 bins by float CAS in arrival order (RHS and DIAG are
identical; X is about 4e-10 apart). That reaches one splash particle's speed, the max BOUND_F (2.08 or 2.67 m/s at
step 6), and the residency's ceil of the flow then holds one more brick plane or not (step 7 liveF 1575/1579 or
1604). The test compares the residency at an equal ceil(flow) of the step's starting bound. pRedAdd is lane P's.

## 2026-11-16: r2 + R3 landed (R1 retirement, R2 calm demotion, R3 h ring), with the 4h-tile surface rules

The user's direction (via the coordinator): the old method has far too many particles in areas of low interest, so be
aggressive in removing fine particle tiles, with R1 and R2 done together. This lands r2 (airc, R1's two-head
retirement, R2's calm demotion with hysteresis) and R3 (the h residency ring), rebased on c35cbe65 and gated in
~/code/me/fluid-scratch/lane-e/r4/g5.

What a 4h tile's surface is ("no particles is not evidence of air"; the coordinator's rulings 2026-11-16):
- Held records in a 4h-owned cell stamp nothing (lSplat), raise no θ or surface heat (pTiles), and are not merged or
  mass-cut by pMove's crowding merge. They exist only so that a quick reversal keeps them.
- lcCouple's union bound counts a 4h-owned cell by the 4h authority's fill, round(8 clamp(0.5 - φ, 0, 1)), of the
  advected C φ at the cell centre. Skipping the cell instead made C φ depend on the tile's own tier: a limit cycle, 79k
  reversals within 2 steps over 20-120 against the control's 31k. The fill gives 26k.
- The mass rules read pMassDepth: tiered and off the h band, the 4h φ rather than fgDepth's push to ±6 cells.
  - The push made held records at a 4h surface merge as deep. They re-entered heavy and did not split. At step 480,
    dimple columns held 8.7 records in their top 4 cells against 14.9 in steady columns, at mass 1.3-2.0, and the
    nearest record was 0.83 cells away against 0.42.
  - fgDepth itself is unchanged, so every grid reader keeps the push. uTransfer is bitwise HEAD's on frozen states at
    steps 20 and 100 (lane F's frozen2).
- R2's calm verdict is tier-invariant: an h tile takes max(its DEF error, the width-4 error from its own data), TC word
  15 (tileErrorOf's second difference of the mean velocity). A demoted tile is not re-refined by a stricter test.
- A fifth residency class covers the tiles this head refines from 4h. Their re-entry seeds move at the grid velocity
  before any ACT.SEED covers them, which was the cause of the REACH latch at 0x601c09.

Gate (tree against HEAD c35cbe65, same round):
- dam-300 Overall PASS, HEAD PASS too. D4 at 30-90 -0.0438 against -0.0421; D8 at 30-90 6.38e5 against 6.88e5.
- Fenced timing median, steps 13-165: 43.01 against 48.67 ms. Window kernel mean 46.5 against 53.9 ms.
- liveF at 300 is 1435 against 2331; particles 337k against 392k.
- Surface (rough / dimples05 at 240, 360, 480), two runs: 0.2171/6, 0.2191/3, 0.1441/4 and 0.2214/2, 0.2174/2,
  0.1184/4. The mean dimples05 is 3.5 against HEAD's 1.1 over nine samples (max 4). The coordinator accepted it for
  this landing, since 3.5 and the unscoped variant's 2.7 are within the noise.

Late-time energy (dam, 600 steps, two runs each against HEAD c35cbe65; ke_J / KE + PE in J). r4's mechanical
energy is 3-5 J below HEAD at every step in both runs, against HEAD's run-to-run spread of 0.3-0.6 J; volume drift
accounts for under 1 J. The user's ruling: "extra damping is probably good". It is accepted and not a landing gate.

| step | HEAD a | HEAD b | r4 a | r4 b |
|---|---|---|---|---|
| 240 | 21.52 / 279.8 | 21.92 / 280.1 | 19.25 / 274.8 | 20.04 / 276.2 |
| 360 | 6.37 / 274.1 | 6.12 / 274.7 | 2.44 / 270.7 | 3.65 / 271.3 |
| 480 | 10.18 / 274.0 | 10.11 / 273.7 | 13.45 / 270.1 | 12.36 / 270.9 |
| 600 | 9.93 / 270.2 | 9.51 / 269.9 | 3.83 / 267.3 | 5.46 / 267.3 |

Known follow-ups:
1. Seam-adjacent dimples, and the visible pockmarks (about 8-10 shallow pocks under 0.5 h on the calm surface at
   about 5.7 s in the coordinator's visual check; none in HEAD at 5.9 or 8.2 s). The next task.
   The metric's dimples: h tiles with 9-13 of their 26 neighbours at 4h still dimple more often than HEAD. Their
   columns remain somewhat sparse and heavy (mass 1.15-1.3 at 1-2 cells deep against 1.01 steady).
2. The unpushed depth for vcTransferFace (the C-page uTransfer face gather). The unscoped fgDepth variant, which
   included it, averaged 2.7 dimples over nine samples. It needs lane F's check against the old engine's coarse
   transfer and its own surface runs.
3. lcCouple costs 0.32 against 0.15 ms (lcNearKnown's ±5 window walk plus the fill sample), and lcRetire 0.62 against
   0.51 ms. pSeed costs 3.13 against 1.92 ms, from the re-entry seed's second launch.

## 2026-11-16: the seam seed (the visible pockmarks)

The visual check of 06d4ca89 found about 8-10 shallow pocks (under 0.5 h) on the calm surface at about 5.7 s. The
surface probe now counts dimples025 (local minima deeper than 0.25 h below their 8 neighbours' mean).

Attribution (single runs; dimples05 / dimples025 at 240, 300, 342, 360, 400, 480):

| tree | 240 | 300 | 342 | 360 | 400 | 480 |
|---|---|---|---|---|---|---|
| 06d4ca89 | 3/14 | 0/7 | 7/23 | 4/19 | 4/21 | 9/17 |
| HEAD c35cbe65 | 2/7 | 0/2 | 0/0 | 0/2 | 2/3 | 0/4 |
| R1 off, calm demotion on | 3/9 | 0/2 | 0/2 | 2/11 | 0/1 | 3/6 |
| lcCouple's fill in lUnion's bound | 0/16 | 1/7 | 9/24 | 6/22 | 4/34 | 5/15 |

- The pocks sit in h tiles at h|4h seams: 60-85% on a tile's x or z face, with 9-14 of 26 neighbours 4h (2-9 on
  HEAD). Their columns are short of records: top layers 1.6, 1.9, 4.5, 2.9 against 0.5, 6.1, 5.9, 4.6 in steady
  columns at 342, and the nearest record below the surface 0.60 against 0.29.
- R1, not calm demotion, causes them: with R1 off they are HEAD-like at the same liveF. The fine union's bound is not
  the cause (the fill did not help).
- The mechanism is inflow: on HEAD the 4h side's records advect back across the seam; after R1 none arrive, and a
  sparse but not empty seam cell never reseeds above -1 (that needs all 27 neighbours empty).

The fix (`pSeam`, particles-kernels.wgsl.ts): an h cell with a 4h-owned cell among its 27 reseeds as an uncovered
one, to `nbSeedTarget` under the 8-mass cap; the 4h side's empty bins are unknown, not air.

Calm phase (342, 360, 400, 480), two runs each:

| | 06d4ca89 | seam seed | inflow-only seam | HEAD c35cbe65 |
|---|---|---|---|---|
| dimples025 mean | 21.9 | 6.1 | 7.0 | 2.25 |
| dimples05 mean | 6.25 | 2.4 | 2.9 | 0.5 |
| particles at 165 / 300 | 370k / 337k | 404k / 381k | 401k / 366k | 390k / 392k |

Gate against 06d4ca89: units 59/0; dam-300 + judge PASS; re-entry KE ratio 1.49, flicker 2.88 per tile per 100
steps; fenced median 44.84 against 45.20 ms (-0.8%); window kernel mean 47.3 against 47.4 ms. Particles are +13% at
300 (still below c35cbe65's 392k): the coordinator's +5% bar was a cost proxy, and the cost did not show, so the plain
seam seed landed. The inflow-only variant (seam only where u at the cell centre points from a 4h neighbour into it)
held the dimples at +9% particles; it was not gated.

Residual: the sub-0.5 h seam dimples, dimples025 mean 6.1 against HEAD's 2.25. The remaining pocks' columns are not
short of records (top layers 4-7).

## 2026-11-16: step floor, lever 1 (ring bricks out of the fine level set), priced and parked

The level set is 10.7 ms a step (37d57ff4, profile steps 13-36) against the 2.0 ms budget of the structural plan's
§6. In all, 59% of F bricks are ring bricks with no h-owned tile, and the fine level-set passes are dispatched per F
brick.

**The bound.** A frozen-state replay (frozen2 --time=8, state from the correct tree) had bricks without an h-owned
tile return at once. A/A noise is about ±0.03 ms per kernel. Full minus skip, in ms:

| kernel | step 100 | step 300 |
|---|---|---|
| lResolve (×4) | 0.74 | 0.50 |
| lDistance | 0.51 | 0.42 |
| lNear (3 passes) | 0.31 | 0.18 |
| lUnion | 0.28 | 0.20 |
| lClear | 0.21 | 0.20 |
| lTiles | 0.14 | 0.10 |
| lPublish | 0.08 | 0.09 |
| lSplat, lDilate, lInit, lAdvect | ~0 | ~0 |
| total | 2.3 | 1.7 |

lAdvect already skips non-owned vertices per lane, so ring bricks cost it nothing. uExtend's ring share (1.21 ms at
step 100, 0.72 ms at step 300) went to lane F.

**The halo census.** These are non-h-brick tiles near an h tile:

| step | within 1 tile | within 2 tiles | bricks, 1-tile halo | bricks, 2-tile halo |
|---|---|---|---|---|
| 100 | 22% | 62% | 45% | 81% |
| 300 | 26% | 72% | 52% | 93% |

lNear's ±5-cell walk needs the crossing words two tiles out. A split halo would run the producers (lTiles, lClear,
lResolve) on two tiles and the consumers (lNear, lDistance, lUnion, lPublish) on one. Tile masking in a launched brick
still pays the staging, so the estimate is about 0.7-1.1 ms. The risk is a per-kernel reach invariant, where a miss is
a silent stale word at the halo edge.

The lever is parked at 2% of the step, in favour of the particle-defined φ go/no-go (scratch, the user's call).

## 2026-11-16: speed of light for the level set's advects (profile, the far skip, the staged C ring)

**The far-vertex skip** (lAdvect leaves owned vertices with |old φ| > 3 untraced) is worth about nothing, so it is
parked. Frozen lAdvect, full against skip: step 100 2.64 → 2.43 ms, step 300 1.44 → 1.41 ms.

What redistance overwrites: lDistance and lPublish rewrite every vertex that is not an interface vertex.
- In reach2 the value is sign(φ) times the distance to the nearest crossing among the dNear cells, which does not
  depend on |φ|. The exception is the fallback sign·max(|φ|, 5) when no crossing is found.
- Outside reach2 it is nbfFarSentinel(old).

A far vertex's advected value therefore matters only through its sign, whether it becomes a crossing corner, and the
min in lUnion. The skip is exact only when |old φ| > the step's maximum travel + √3. A fixed 3 is not provable:
flowMax is 6.45 cells/step at step 300 and 22 at step 60.

**Hardware counters.** r8/kloop.mts loops one kernel on the frozen step-300 state. xctrace attaches with Metal System
Trace and Metal GPU Counters for 2 s; counters are averaged over busy samples (occupancy > 0.5%).

| | lAdvect | lDistance | lcAdvect |
|---|---|---|---|
| frozen ms | 1.42 | 1.01 | 1.56 |
| compute occupancy, mean (p50) | 13.3% (15.6) | 23.2% (25.0) | 6.7% (5.7) |
| ALU utilisation / F32 utilisation | 35.7 / 1.1 | 48.0 / 7.7 | 18.1 / 1.1 |
| ALU limiter | 37.7 | 54.3 | 18.7 |
| buffer read limiter | 3.7 | 3.0 | 1.5 |
| LLC limiter | 11.0 | 8.9 | 7.3 |
| read bandwidth, GB/s | 2.35 | 8.1 | 1.0 |
| MMU TLB miss rate | 30% | 22% | 19% |

Bandwidth is about 1% of the roofline. The ALU is busy almost entirely with integer and address work (F32 about 1% in
both advects).

**Lane use is not the cause.**
- At step 300, 21% of lAdvect's SIMD groups have an owned lane (5232 of 24448), and those groups are 89% full.
- Packing owned vertices first in the workgroup is bit-identical, but slower: lAdvect 1.42 → 1.92 ms, lcAdvect
  1.63 → 1.69 ms (step 300). It breaks tap locality within a SIMD group.
- A compacted owned-vertex list is therefore no lever.

**Where the time goes** (frozen step 300, one part removed or made free at a time):

| change | lAdvect, ms | lcAdvect, ms |
|---|---|---|
| none | 1.38 | 1.53 |
| lane body empty | 0.18 | 0.06 |
| no trace | 0.67 | 1.29 |
| free velocity taps | −0.74 | −0.22 |
| no cubic | 1.13 | 0.78 |
| free cubic taps | −0.20 | −0.61 |
| no foot stencil | −0.06 | −0.17 |

- In lAdvect, about 54% is the trace's velocity taps (fgSampleVel's owner and face resolution per tap). That sampler
  is lane F's, and the fast path went to F.
- In lcAdvect, about half is the cubic's 64 nbVertexPhi taps. Each one is nbVertexCoarse's 8-tile owner test plus a
  page lookup.

**The owner window is bit-identical to the directory.** lAdvect, staged with the 3³ owner window, gave φ different by
1 ulp (92% of 13843 words; at most a few ulp in small |φ|) and ran slower (1.38 → 1.77 ms). A direct check on frozen
states at steps 100 and 300 found the window and the directory paths identical. Every lAdvect workgroup compared:
- fgWinVertexCoarse and fgWinVertexPhi at its 20³ covered vertices;
- fgTileCSlot and fgTileH at 216 tiles;
- fgBrickSlot at 27 bricks;
- fgSampleVel at 8000 points.

The φ difference is the compiler contracting differently once the lookup code changes.

**The staged C ring.** lcAdvect now runs eight workgroups a page (C_ALL X8), each one z plane of the page's C
vertices with a lane a vertex. Each workgroup stages the resolved φ of the C lattice vertices about its plane
(13 x 13 x 5, two vertices of halo each way) once. The foot stencil, the departure corners and the cubic read
through lcPhiTap, which falls back to nbVertexPhi outside the ring.

Results are exact up to compiler contraction: 1–2 ulp in 964 and 540 φ words. Frozen lcAdvect:

| | step 100, ms | step 300, ms |
|---|---|---|
| HEAD | 1.82 | 1.91 |
| ring, X8 | 1.15 | 0.88 |
| ring, one workgroup a page (15³ ring) | 1.29 | 1.06 |

**Gate** (against HEAD 31630b18):
- Units: 65 of 65 pass. dam-300 judge PASS.
- Fenced 165-step medians, interleaved: 43.89 / 44.36 ms against HEAD's 45.38 / 44.47 ms (−0.8 ms mean).
- Profile, steps 18–23: lcAdvect 1.04 → 0.74 ms. The other areas are within run noise.
- lcAdvect was 6.7% occupancy at 198 workgroups. Its dispatch is now 8 × 64-lane workgroups a page, which follows
  fill-the-gpu.

The fgSampleVel fast path (a one-cell + apron on FC_UNIT, 9³) is lane F's. I reviewed the ABI and it rebases on this.

## 2026-11-16: swing B (fused particle pipeline) is a no-go; the staged pSeed (lever S) lands

**The profile baseline.** Steps 18–23, ms a step:

| kernel | ms |
|---|---|
| pMove | 3.68 |
| pSeed (1.83 launches) | 3.02 |
| uTransfer | 2.12 |
| pDeposit | 0.79 |
| pDelta | 0.39 |
| pJoin | 0.23 |
| pLiquid | 0.21 |
| pScan | 0.14 |
| pScatter | 0.12 |
| pTiles | 0.08 |

**Knockouts** (frozen2, timing slot, steps 20 / 100):
- pMove:
  - 3.66 / 3.96 ms as is;
  - without the separation 1.85 / 1.91;
  - without the G2P sample 2.87 / 3.43.
  - Heun, the merge, depth, heat and split each cost about 0.1–0.4 ms.
- Main pSeed:
  - 2.84 / 2.69 ms as is;
  - without the write pass 1.46 / 1.75;
  - without the per-site depth 1.32 / 1.03.
- pDeposit: 0.99 / 1.00 ms, of which the liquid staging is 0.65.

**Counters at step 100:**
- pMove: occupancy 18%, ALU 41%, LLC limiter 22%.
- pSeed: occupancy 16–18%, ALU 25%, read bandwidth 1.7 GB/s, TLB miss 24%. It is latency-bound on dependent lookups.

**The census and the bound** are in the speed-of-light review, §4.B, with the table:
- Records leave their brick 90% / 39% / 12% of steps (steps 20 / 100 / 300) and travel a median 6.3 cells at step 20.
  The staged move apron therefore misses exactly when the move is expensive.
- Fusion alone bounds at 0.6–1.0 ms. No-go.
- Staged separation in pMove (each workgroup stages its records' cells ±1 and their records):
  - exact, with fmoved bitwise equal;
  - 3.09 → 7.00 ms at step 20 and 3.92 → 8.29 ms at step 100;
  - 28 KB of workgroup memory collapses occupancy. Parked.

**Lever S.** Each pSeed brick that has a tile to seed stages its tile gates and reach (8 tiles), its 4h bits (4³ tiles)
and its 9³ FC_DEPTH vertices once. Nothing is staged in the re-entry launch.
- psDepth is fgDepth's text with the staged vertex reader, and it throws if fgDepth's text changes shape.
- The count pass returns a site mask a cell, and the write pass writes only the masked sites, in the same order and
  at the same offsets.
- uncovered is evaluated only where nbfSeedSiteOk reads it, and the seam test is evaluated near the band only.
- Lane F reviewed it and approved.

Exactness:
- Seed counts and probe words 201–211 are identical. The re-entry seeds' multiset is equal.
- Main seeds differ only by 1 ulp in velocity.w (the site depth) on about 36% of seeds: the compiler contracts the
  same expression differently. An old-against-old run is equal.

Frozen, on 3090d99e: main pSeed 2.62 → 1.65 ms at step 20 and 2.55 → 1.56 ms at step 100.

**Gate** (tree d6ff2690 + S, against d6ff2690):
- Units: 81 of 82 pass. The failure is levelset-release (GPU) "lcAdvect path kept deep air covered". It fails 2 of 3
  runs on d6ff2690 itself, because the no-words lcAdvect writes a different vertex set each run. Lane F is bisecting
  it: uCache's FC_UNIT or the staged lcAdvect.
- dam-300: Overall PASS. D8 at 30-90 6.54e5 against 6.31e5; at 180-300 3.69e5.
- Fenced 165-step runs, interleaved: medians 38.89 / 39.85 ms against 39.60 / 39.67 ms; means 42.82 / 43.23 against
  44.13 / 45.03 ms.
- Profile, steps 18–23: transfer and seeding 6.84 → 5.93 ms; window kernel mean 40.9 → 39.7 ms.
- test:unit 1871/1871.

## 2026-11-16: swing D, the dam's particle band is 2 cells deep (scene.fluid.brickTiers.particleBandDepth)

**Census** (frozen dam at steps 20 / 100 / 300):
- Mass per cell by depth is 10–11 from −2.5 to −1, and about 8 from −1 to 0.
- The merge holds its cap inside the move. The shallow cells (depth ≥ −1) are uncapped, with up to 152 / 125 / 41
  records; that is closed separately.
- At step 100 the deep band's merge cuts 168k records to 136k (28k by the cap, 3.9k by mass).
- Records deeper than −2 make up 20 / 24 / 22% of the total.
- So the band's depth is the count lever. The mass-8 cap is parked.

**Parameter.** scene.fluid.brickTiers.particleBandDepth is brick-only.
- It is validated as finite and within [2, 4] cells; the default is 3.
- It flows through BrickSceneSetup.particleBandDepth to the engine options, then the stage context (particles.bandDepth),
  then particleKernels.
- nbfParticleFormulasAt substitutes the anchored `NB_BAND_DEPTH` and throws if the anchor or the depth is unknown.
- Shader modules are keyed by their code, so the constant is part of the cache key.
- The high-resolution dam sets 2. The old engine (the method value, still pinned at 3 for the brick engine) and every
  other scene are unchanged.

**Thin band** (on d4beeabf, NB_BAND_DEPTH 2.0 and 2.5 against 3.0). Both pass dam-300.

| band | D8 particles 30-90 / 105-165 / 180-300 | fenced 165-step median (ms) |
|---|---|---|
| 3.0 | 6.54e5 / 4.04e5 / 3.69e5 | 39.00 / 39.24 |
| 2.5 | 5.64e5 / 3.61e5 / 3.36e5 | 37.67 / 37.01 |
| 2.0 | 4.39e5 / 3.13e5 / 2.97e5 | 35.75 / 35.53 |

Fenced means are 43.35 / 43.47 ms for 3.0 and 39.11 / 39.11 ms for 2.0.

The 2.0 profile against 3.0 (steps 18–23): move 3.72 → 2.37 ms, transfer and seeding 5.93 → 5.24 ms, window kernel
mean 39.7 → 37.7 ms.

The pressure solve rose 6.80 → 7.27 ms in that window. It comes from the cycle count, not the fill: step 24 ran 7 cycles
against 5. Over dam-300 the mean is 5.03 cycles per step against 5.06 (5.29 against 5.43 in 61–165, 4.56 against 4.67
in 166–300).

Other results at 2.0:
- Energies: D5k at 30-90 is 43.1 against 46.5 (margin 0.73); D5e is 318.9 against 328.0 (margin 0.78). Extra damping
  is accepted.
- Surface roughness / dimples025 at steps 240, 300, 360 and 480: 0.22/11, 0.19/0, 0.25/3, 0.18/10, against 3.0's
  0.26/6, 0.23/2, 0.22/10, 0.17/3. That is within the noise.
- The coordinator's visual check (splash at about 2.2 s, sloshing at about 10–11 s) matches band 3.

## 2026-11-16: the shallow-cell cap is parked (its cut is small)

The first draft extended pMove's held-mass cut (NB_CELL_MASS, 2n) to every h cell. Lane F reviewed it: the cut drops
records past 16 mass, it does not merge them, and the extension now reaches air and ballistic records.

The drop census (r12/cen.out) counts records past the cut per step, by the cell's depth band, with the ballistic
records apart:

| tree | step | below −1 | −1..0 | above 0 (ballistic) |
|---|---|---|---|---|
| band 2 + cap | 20 | 2533 | 3429 | 414 (90) |
| band 2 + cap | 100 | 1607 | 597 | 10 (0) |
| band 2 | 20 | 4134 | 9345 of 85.9k | 642 (216) |
| band 2 | 100 | 2223 | 2434 of 133.6k | 341 (18) |

The coordinator ruled that the cap must conserve mass and exempt air and ballistic records.
- A fold of the excess into the kept records, the crowding merge's way, at depth −1..0 in liquid-mode cells, would
  remove at most the band-2 tree's excess stock.
- That is 9.3k / 2.4k records, or 2.1% / 0.8% of the step's records: about 0.2 / 0.06 ms at swing D's bound of
  2.5 ms per 100k records.
- No ballistic record sits in a −1..0 cell over the cut.
- Parked: the cut is small. The record cap stays as it was. Every record weighs at least 1, so a record cap of 20 never
  bound behind the 16-mass cut; only the depth ≤ −1.5 cap of 6 acts, as before.

## 2026-11-16: the tier machinery's bound (brick set, census, transfer and the C tier) is a no-go against the 1.5 ms bar

**What was measured, on d0f35ebc:**
- Per-dispatch profiles at steps 18–23, 98–103 and 298–303, in r13/tim r13-prof-*.
- Frozen per-kernel timing, with exit-class histograms and knockouts, at steps 20, 100 and 300. The tool is
  fluid-scratch/lane-e/r13/tools/frozenE.mts with `--engine` (the head kernels too), `--subfile`, `--hist` and
  `--flips`.

**Profile:** the machinery is 4.33 / 4.83 / 3.94 ms at steps 20 / 100 / 300. Step 100, by kernel (ms):

| kernel | ms |
|---|---|
| censusFine | 0.91 |
| lcAdvect | 0.86 |
| lcRetire | 0.52 |
| lcCouple | 0.35 |
| certFine | 0.24 |
| setTransferFaces | 0.22 |
| censusDecide | 0.21 |
| censusCoarse | 0.21 |
| resMark | 0.16 |
| censusImportance, nrReach, certCoarse, certError, setMark | 0.11 each |
| the rest | ≤ 0.08 each |

**Lanes with work at step 100:**
- censusFine: 17.1k workgroups launched; 4.1k tiles are h-owned (24%).
- C tier: 103 pages, 52.7k vertex lanes.
  - lcAdvect: 25.8k not C-owned, 6.1k kept far air, and 26.5k on the full path (6.5k at |φ| < 8, 10.4k liquid at
    φ ≤ −8, 9.0k air).
  - lcRetire: 3.9k build a distance and 2.6k are hot; 25.6k are not stencil-fine.
  - lcCouple: 5.1k need the bulk depth or the gather; 27k have θ ≤ 0.

**The C kernels are latency-bound:**
- Each one fits in one wave (about 100 workgroups, or 820 for lcAdvect), so a kernel's time is its slowest lane's chain.
- Holding the 10k deep-liquid lcAdvect vertices (φ ≤ −8, a deviation) saves 0.15 ms at step 20 and nothing at
  steps 100 and 300: the other lanes of each workgroup still trace.
- Knockouts at step 100 (upper bounds):
  - lcAdvect: the trace is 0.25 ms and the released walls 0.1.
  - lcRetire: lcNearKnown's 64-tile window is 0.13 ms, and the build's remaining 0.3.
  - lcCouple: lcNearKnown is 0.03 ms.
  - censusFine: ncMetricCorners + ncCornerLaplacian (lanes < 24) is 0.24 ms, ncSmoothTile 0.1, and the depth
    sample 0.13.
  - Empty workgroups: all of censusFine's are 0.06 ms, and certFine's 0.06.

**Exact options, priced at step 100:**
- **lcRetire and lcCouple at X8**: eight 64-lane workgroups a page, one vertex a lane, as lcAdvect. Measured and
  bit-identical:
  - lcRetire 0.56 → 0.42 ms (0.63 → 0.39 at step 20, 0.53 → 0.34 at step 300);
  - lcCouple 0.35 → 0.27 ms (0.37 → 0.23, 0.41 → 0.25).
- A per-plane staged lcNearKnown: ≤ 0.16 ms.
- censusFine's lane-subset phases spread over the workgroup: ≤ 0.34 ms, realistically about half.
- An h-tile list in place of F_ALL X8 for censusFine and certFine: ≤ 0.12 ms.
- lcAdvect's trace and walls: ≤ 0.35 ms, divergence-bound, with no cheap restructure found.
- Realistic total about 0.6 ms; the upper bounds sum to about 0.9 ms. Both are under the 1.5 ms bar.

**Tier churn (not exact):**
- About 900–1000 tiles change width a step at step 100 (471 refine, 557 coarsen), and about 500 at step 300. Each
  change brings its neighbour entries: the list holds 2.7k / 1.7k entries.
- 10–30% of each step's changes reverse the previous step's, and about 45% reverse a change within six steps.
- Hysteresis is a tier-rule deviation. Its direct share is the transfer (about 0.35 ms) plus claims and leaves, so it
  is under the bar too.

**Verdict:** no-go at 1.5 ms. Not built. The X8 pair (about 0.2 ms at step 100, 0.4 at step 20, bit-identical) is
the one measured exact win, should a smaller bar apply.

## 2026-11-16: lcCouple and lcRetire at X8 (landed, exact)

lcCouple and lcRetire now dispatch as lcAdvect does: eight 64-lane workgroups a C_ALL page (X8), one vertex a lane,
the z plane lcPart. They were one 256-lane workgroup a page with a two-vertex loop (G1). The C kernels are
latency-bound (each fits in one wave), so halving each lane's chain is the win. lcClear and lcPublish stay at G1.
levelset-coarse.wgsl.ts's LC_GROUPS names each kernel's dispatch, and brick-fluid-stages.ts places them by it.

**Gate, on a3cfa458:**
- Frozen dam, the new tree's stages against HEAD's (frozenE, non-engine mode, `--time=8`). Bit-identical in set and
  phi at steps 20 / 100 / 300 (bad 0, aborts 0).

  | ms | step 20 | step 100 | step 300 |
  |---|---|---|---|
  | lcCouple, G1 → X8 | 0.41 → 0.26 | 0.37 → 0.27 | 0.41 → 0.25 |
  | lcRetire, G1 → X8 | 0.57 → 0.38 | 0.63 → 0.40 | 0.56 → 0.33 |
  | the pair's saving | 0.34 | 0.33 | 0.39 |

- test:unit 1847 pass, 0 fail. GPU units 84 / 84. check:types: no narrow-band errors (lane-n's 3 only).
- dam-300 (check slot, aborts 0): PASS. D8 4.32e5 / 3.14e5 / 3.01e5.

The review doc's tier section records the churn: about 45% of width changes reverse within six steps. That is a
quality signal as well as a cost one. Hysteresis stays parked.

## 2026-11-17: the pool sizing rule (rollout task 2), measured

**Rule** (for a scene with no authored `brickPools`). `n` is the lattice in cells and `V` the initial liquid's
cells. `top` is its top in cells. bootF and bootP come from F's `bootBricks` / `bootParticleCount`.
- Window (the hard cap): `WF = Π(n/8 + 1)`, `WC = Π(n/32 + 1)`.
- Energy: no liquid rises above twice its initial top, plus a ring.
  - `EF = floorF · min(n_y/8 + 1, ⌈2·top/8⌉ + 2)`.
  - `EC = floorC · min(n_y/32 + 1, ⌈2·top/32⌉ + 1)`.
- Spread:
  - a film at least 1/16 cell thick over the floor, 4 brick layers: `SF = 4 · min(floorF, ⌈V/4⌉)`;
  - every liquid brick at least 1/16 full: `LF = ⌈V/32⌉`.
- **f** = `min(WF, EF, max(SF, LF, ⌈1.25 bootF⌉, 128))`.
- **c** = `min(WC, EC, f)`.
- **particles** = `3 · bootP`.
- **b** = 16 and **k** = 0 (no cuts until M3).

An underestimate latches and is never grown. Checked on figure-9 with f = 1200, c = 40 and particles = 100k:
- `POOL_F` at step 92, with high = cap;
- `BRICK_POOL_C` at step 30;
- `BRICK_PARTICLES` at step 42.

Each one sets `info.simulationFailure` (SimulationFailureError), and the solver refuses every later step.

**Peaks against the rule.** Runs used generous pools, clean e86757ab, 240–900 steps, sampled every 2–5 steps. F, C
and P are the maxima over every valid run.

| scene | V | f rule | F peak | margin | c rule | C peak | margin | p rule | P peak | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| high-resolution dam | 753664 | 4913 (window) | 4913 | 1.00 (cap) | 125 | 125 | 1.00 (cap) | 1.87M | 997k | 1.87 |
| figure-9 (900 steps) | 245759 | 2601 (window) | 2216 | 1.17 | 75 | 75 | 1.00 (cap) | 720k | 255k | 2.82 |
| minimal-64 | 94208 | 729 (window) | 726 | 1.00 (cap) | 27 | 27 | cap | 525k | 195k | 2.68 |
| minimal-32 | 11776 | 125 (window) | 125 | cap | 8 | 8 | cap | 117k | 37k | 3.13 |
| power-fill-256-6400 (×3) | 6400 | 4356 | 1943 | 2.24 | 162 | 96 | 1.69 | 78.7k | 55.6k | 1.41 |
| power-fill-256-800 (×3) | 800 | 800 | 538 | 1.49 | 162 | 55 | 2.95 | 17.9k | 9.3k | 1.91 |
| power-droplet-256 (×3) | 100 | 128 (floor) | 80 | 1.60 | 128 | 8 | 16 | 3828 | 1127 | 3.40 |
| power-droplet-128 (×3) | 100 | 128 (floor) | 86 | 1.49 | 50 | 10 | 5.0 | 3828 | 1167 | 3.28 |
| power-droplet-64 | 100 | 128 (floor) | 52 | 2.46 | 18 | 8 | 2.25 | 3828 | 1103 | 3.47 |

- **Run to run, the small scenes' F peak varies by up to ~60%.** power-droplet-128 gave 52 / 56 / 86, power-droplet-256
  gave 56 / 56 / 80, and power-fill-800 gave 394 / 502 / 538. The arrival-order lists make this nondeterministic. The
  128-brick floor (8 MB) and the 4-layer film are sized for it. A 3-layer film was already beaten by one droplet run
  (80 > 75).
- **Peaks come early.** Every 900-step peak was before step 290: power-fill-6400 at about 250, figure-9 at about 130.
- **Memory:** the rule asks for 495 / 237 / 223 / 49 / 16 MB, against 412 / 168 / 133 / 36 / 6 MB measured. The window
  rule would have taken 2.25 GB for a 256³ droplet.
- **Untested regime.** For a scene whose spread term `LF` lies between the film and energy terms, the rule rests on
  the 1/16-full brick assumption alone. sparse-cm12-long-dam-break (f = 1280 by `LF`) was meant to test it, but it
  aborts (next item).

**Aborts (not pool related):**
- sparse-cm12-long-dam-break: a Metal command-buffer abort at step 9–10, in 2 of 2 runs, alone on the GPU. "Step 10
  refused", runner rc 77. The lattice is 192×96×32, with generous pools (f = 1625).
- Intermittent rc 77 in 2 of about 16 other runs (the figure-9 c = 40 check and power-droplet-256), both while 2–3
  checks shared the GPU (peak 3–4.4 GB each). The same inputs ran clean on other runs.
- Neither is diagnosed. cm12 blocks enabling that scene.

**Not landed yet:** the rule in brick-scene.ts. Its particle term uses lane F's `bootParticleCount`
(brick-initial.ts, in F's tree), and it goes in with F's initial-liquid interface. The probe is
`fluid-scratch/lane-e/r16/tools/peaks-brick.mts`, and the table is `r16/tools/rule2.py`.

**M5 review (lane P, A1–A8).** I accepted all eight and sent the verdict to P. The changes:
- A2: setMark owns both NEAR_BODY and BODY_WAS. Entering bricks get the body test in setUpdate, which closes a
  one-step open-inside-a-body hole.
- A5: a CUT_STATIC class bit (E, bit 9). K slots are kept for CUT_STATIC, NEAR_BODY or BODY_WAS.
- A6: the stamp stage goes at the end of the head, after the transfer and before the release reach. F_BODY lists
  NEAR_BODY or BODY_WAS.
- A4 is conditional: a per-step prescribed-motion upload must be listed and measured.

## 2026-11-17: M3 stage 1 (padding by the box rule, the five static hooks) and the GPU bound guard (landed)

This lane E agent replaces the one lost to the 08:24 kernel panic. It reviewed the unlanded stage 1 work in main and
finished it.

**Stage 1**
- brick-static.wgsl.ts holds the five hooks lane P's PbStaticHooks call: `nbStaticCellOpen`, `nbStaticFaceOpen`,
  `nbStaticSolidAt`, `nbStaticCellKind` and `nbStaticFaceBoxed`. Here the static solid is the box rule, with an open
  top as air.
- brick-static.test.ts holds the Float64 mirror and a GPU check of the WGSL against it, bit for bit: every cell of the
  box grown by 2 (30 above), plus random world points, with both tops.
- No vector component is written through a dynamic index.
- The lattice is now the container's own cells, and the storage rounds up to pages behind it. `farPhi` keeps
  NB_PHI_FAR at the old engine's page-rounded extent. A container that is not whole bricks is a gap, because it needs
  solidStamp's cuts.
- F's unimplemented forces (nbfVelocitySupported) are now a `brickSceneSetup` gap, so `auto` never picks a scene whose
  stages would throw at creation.
- No scene is enabled yet: BRICK_ENGINE_SCENES is still the dam, and the dam is whole pages, so its lattice and far φ
  are unchanged.

**The bound guard** (after the panic). Every indirect dispatch is listed below with its writer and bound. A writer
that overruns latches FAIL.
- `setArgs` (brick-set): lists 0–17 G1/G8/G64/X8, FC_*, FA_*. Bound: each count is min(count, LIST_CAP[k]) and latches
  LIST. Already so.
- `setTransferArgs`: XFER_G1. Bound: min(n, TCAP), and the append latches LIST. Already so.
- `pArgs` (F, P_ARGS_RECORDS at 2048): (min(PCOUNT, PCAP)+127)/128. The appends latch PARTICLES. Already so.
- P's gates in pressure-mgpcg.ts (new bounds):
  - P_GATE_F and WF are min(T_RBN, FCAP);
  - P_GATE_S is min(T_SBN, 4 FCAP);
  - P_GATE_FC is their sum with the capped C_ALL;
  - P_GATE_C, WC and ONE copy capped list args;
  - pmgListArgs HB/HP/ZL/RL are min(count, FCAP), latch LIST (info 0x5k......), under the fold rule. They were
    written raw, unfolded.
  - rowsList's list writes are guarded to FCAP.
- Loops bounded by buffer words in P's kernels (new):
  - every `wCount = T_RBN / T_SBN` head is min'd to FCAP / 4 FCAP;
  - pSeamN is ≤ 3072, its region;
  - pmgTopCycle's PH_N5/N6/N7 are ≤ CCAP / P_ROWS6 / P_ROWS7.
- bootPhi's host box count is held to the boot binding's boxes.
- An audit of every data-bounded loop (664 loops) left these to their owners:
  - **F:**
    - pJoin (F's fix, the panic);
    - nbfMomentumTrace's loop{}, which never exits on a NaN dt or a NaN span;
    - pScan stores BIN_COUNT unclamped on overflow (pTiles, pDeposit, lSplat);
    - fgRun's FC_RUN ends;
    - lNear's cfg.z.
  - **P:** pbApplyBodyCuts `k<header.y`.
  - **E's own loops:**
    - The census, residency and release boxes are bounded by window or lattice clamps.
    - setClaim* and the list loops are bounded by caps.
    - nbClaim/rqTake CAS loops retry only on a spurious weak-CAS failure.

**Gates** (main, timing slot throughout, one run at a time, rc 0, no 00000206)
- check:types: lane-n's 3 only.
- test:unit: 1875 pass, 0 fail.
- GPU units, serial: 51 of 51 pass. These are brick-static, brick-engine, brick-abi, pressure-mgpcg,
  pressure-composite(-mgpcg), brick-set-tiers and bodies-shapes.
- dam-300, tiered composite, poolF 4913, judged against ref-2761350a: **Overall PASS**. The run was fenced on the
  timing slot. It was judged first with its meta relabelled `check`, then as recorded, after the follow-up commit lets
  nbflip-gates.ts take timing-slot candidates. Both give PASS.
  - D8 at 30–90 is 4.35e5 (an upper bound), against 6.31e5.
- Step time, from the same run, steps 13–165: median **34.23 ms**, mean 39.1, against 34.4–34.7 at 20a233bb. No
  regression.

## 2026-11-17: M3 stage 2a, the static voxel source behind the five hooks, with the AGENTS.md regressions (landed)

**The source.** The hooks now read the static voxels from the shared geometry path's own image, the
SolidOccupancyMask.
- `brickStaticMask(scene)` builds the mask exactly as the old engine does (webgpu-uniform-reference.ts):
  - solidWorldForFluidLattice, with terrain kept separate;
  - half-solid for a procedural solid that collides as voxels;
  - walls by the tank's rule (policy 1, or 2 under an open top), so the shell sits in wall pages and holds no slot;
  - sealed at the pages partly solid now, as many again, plus 64.
- It returns undefined when no cell of the box is solid, so the scene gets no binding and no memory.
- The seep: 16 slots, a 2.4 MiB image (the 2 MiB fixed tables plus one 4 KB unit per partly solid page), built in
  30 ms.
- `brickStaticWGSL({..., mask: {buffer, walls}})` reads the mask through solidOccupancyMaskWGSL's fixed reader, after
  the box rule. A cell outside the box never reads the mask.
- With the mask, the semantics are umCellOpenGeometry and umPressureFaceV for a voxel with no record:
  - VOXEL cells have open fraction 0;
  - a face with one VOXEL side is 0.5·(open + open), so 0.5 beside a voxel or a box wall and 0 between two;
  - any other face is 1.
- Without the mask, stage 1's box faces (0) stay. P's pbStampFace never reads either value at a wall face.
- The mirror reads the same words through solidOccupancyMaskVoxel.
- There is no copy of the geometry. An edit is the mask's `update`, which names the cells it moved.

**New gaps** (brickSceneSetup). The hooks do not read these yet, so they must not run as open water:
- terrain: 15 catalog scenes;
- a procedural boundary sampled as a field (`collision` not "voxels"): 9 catalog scenes.

The dam still has no gaps. The "solid voxels beyond the shell" gap stays until solidStamp binds the mask in the
engine and P and F consume it.

**The AGENTS.md regressions** (brick-static.test.ts, on the seep):
- **Hit testing:** the mask equals voxelReaderForScene (the editor's pick reader) on every cell of the
  512×256×64 box. pickSolidVoxel along +x meets the wall at the first cell the hooks call VOXEL, and solidAt agrees,
  at three heights including the hole's edge.
- **Local edit:** an ellipsoid clear through the wall's face plus a box fill in the floor, via sceneWithSolidStroke.
  - The edited scene keeps every base patch, in order.
  - The rebuilt mask equals the edited reader on every cell.
  - The incremental `update` equals the rebuild on every cell.
  - The moved cells are exactly the cells `update` names (each once), all within the edit boxes.
  - Around the edits, a cell's kind and open fraction change only where its bit moved, and a face only where one of
    its two cells moved.
- **Untouched fidelity:** no cell outside the edits changes, checked on every cell of the box.
- **Save/load:** a JSON round trip builds a bitwise-identical image.
- **Undo:** the live mask returns to the base on every cell, and gives its slots back.
- **Unknown kinds:** an unknown patch kind throws (`Unknown solid patch kind`). Terrain and a field-sampled boundary
  are named gaps.
- **GPU:** the WGSL with the mask matches the mirror bit for bit on about 60k cells (the hole, the edits, and random
  cells in and past the box) plus world points, on both the base and the edited mask.

**Gates**
- check:types: lane-n's 3 only.
- test:unit: 1882 pass, 0 fail.
- brick-static on the GPU: 6 of 6 pass (timing slot, rc 0, no 00000206).
- No engine kernel compiles these hooks yet (only P's tests do), so the dam's step is unchanged and the dam was not
  re-run.

## 2026-11-17: ABI A1/A5/A6 for P's bodyStamp, and solidStamp (M3 stage 2b)

These are the shapes m5-bodies-design-2026-11-17.md §8.1 fixes. brick-abi.md is amended to match.

**A1, the cut pool:** FRAC, AP0–2 and W0–2 per K slot, then one BMASK word per K slot at 7·KCAP·520 (NB_CUT_BMASK).
The pool is 7·KCAP·520 + KCAP words.

**A5, K slots:**
- F_CLASS bit 9 CUT_STATIC is E's: the stamp sets it, and setMark keeps it across the class pass. Bit 24 BODY_WAS is P's.
- K_HOLD (NB_CLASS_K_HOLD) is CUT_STATIC | NEAR_BODY | BODY_WAS.
- nbKTake(fs) is in the exported BRICK_SET_HELPERS_WGSL. It returns the brick's held slot (REC.CUT), or pops one.
  On underflow it latches POOL_K and returns NONE.
- setKRelease (cfgZ 0/1) runs after every stamp of the step:
  - over F_ALL, it pushes a slot whose brick holds no K_HOLD bit;
  - over F_LEAVE, it pushes every held slot.
  - It clears REC.CUT before each push.
- setArgs keeps LIVE_K and PEAK_K. The census's UNSUPPORTED latch while LIVE_K ≠ 0 stays, as intended, until the
  operator and F's solid paths read cuts.
- F_BODY lists NEAR_BODY | BODY_WAS.

**A6, the stamp stage:** BrickStages.stamp (P's bodyStamp) runs after solidStamp, then the two setKRelease passes.
In the head it sits after the tiered transfer and the certificate, before churnClear; at boot, after bootHead. With
k = 0 and no mask the stage is empty, so the dam's schedule is unchanged. Only setMark, setArgs and setLists'
F_BODY test changed in the dam's kernels.

**solidStamp:** BrickEngineOptions.staticMask binds brickStaticMask's words as `solid`. Each F_ENTER brick gets one
workgroup, and solidStamp skips NEAR_BODY bricks (P's):
- A brick is cut when one of its cells, or the lower neighbour of a cell along an axis, is an in-box mask voxel.
- A cut brick takes a K slot and becomes CUT_STATIC. It gets FRAC = cellOpen and AP_a = faceOpen (0 at the box
  faces, the box rule's), with W and BMASK 0.
- FLAGS.OPEN clears on every closed cell.

No scene binds the mask yet. The voxel gap stays until the consumers read cuts.

**Tests**
- CPU, brick-engine.test.ts:
  - the A1/A5 ABI: offsets and words;
  - the A6 schedule: no placement at k = 0. Otherwise the order is stamp, setKRelease, setKRelease:1, then stepEnd,
    and at boot the stamp follows bootPhi and the census.
- GPU, the K lifecycle (A5): a test stamp takes a slot once per brick, POOL_K latches past KCAP, and setKRelease
  returns the slots.
- GPU, solidStamp on [64,32,64] with three patches: a box, an ellipsoid across two bricks, and a column that cuts a
  brick by its faces alone. CUT_STATIC lands exactly on the expected bricks, FRAC/AP/W equal the hooks' mirror,
  FLAGS.OPEN and BMASK are correct, and LIVE_K equals the cut count.

**Gates**
- check:types: lane-n's 3 only.
- test:unit: 1885 pass, 0 fail.
- GPU units, serial, in the timing slot: 66 of 66 pass, rc 0, no 00000206. The set is brick-static, brick-engine,
  brick-census, brick-abi, the three pressure suites, brick-set-tiers and bodies-shapes.
- Dam-300, fenced, timing slot: judged Overall PASS against ref-2761350a.

Step time over steps 13–165, from one paired round:

| tree | median (ms) | mean (ms) |
|---|---|---|
| HEAD 14b1ecc1 | 35.26 | 40.62 |
| HEAD + this diff only | 35.35 | 40.35 |

- The paired per-step difference has a median of +0.24 ms (139 steps, sampled steps excluded). That is inside
  single-run noise.
- Both trees sit above the 34.4–34.7 ms reference today, so the absolute level is the machine's.

## 2026-11-17: the pool rule wired into brick-scene.ts, TCAP, and R2's shape guard (landed)

**The pool rule.** A scene with no authored `scene.fluid.brickPools` is now sized by `brickPoolRule`, the rule
measured above. Its inputs:
- the lattice;
- the initial liquid's extent: `brickInitialExtent` counts the liquid cells and the top liquid row from the boot's
  vertex φ, `bootPhiAt`, which covers both boxes and F's page table;
- the boot's F bricks;
- the boot's particles: F's `bootParticleCount`.

The rule never grows a pool. A scene that outruns its pools latches FAIL.POOL. An authored scene keeps its pools
exactly, so the dam's and the seep's are unchanged.

**TCAP.** t = min(512 c, max(2 f, 4096)).
- 512 c is exact: setTransfer lists a C_ALL tile at most once a step.
- 2 f, the layout default, was too small. falling-water-torus listed 251 tiles against 2·125 and latched FAIL LIST
  0x700000fa.
- 512 c was rejected as the layout default because the xfer scratch is 3 KB a tile, about 197 MB for the dam. The
  dam authors its pools, so its t is unchanged.

**R2's shape guard.** This was the coordinator's ruling, 2026-11-17. Calm means 4h represents both the tile's
velocity and its shape. A band tile whose FIN carries its shape or thin trigger is never calm, so R2 never demotes it.

Before the guard, falling-water-torus lost all of its liquid by step 10. The trace showed the band's h tiles demoting
at their second calm head while 4h could not hold the torus's tube.

Over 60 steps on falling-water-torus (liquid volume = the cell-fraction sum):

| tree | drift at 60 steps | step median (ms) |
|---|---|---|
| R2 as landed | −100% | |
| R2 off | +6.2% | 6.15 |
| R2 with the guard | +5.7% | 6.44 |

The old engine's drift on the same scene is the reference for the +5.7%. It follows in the batch entry.

**Tests**
- brick-engine.test.ts, a CPU test of the rule's terms by hand (window, floor, TCAP) and of the measured table
  through brickSceneSetup:

  | scene | f | c | particles |
  |---|---|---|---|
  | power-droplet-64 | 128 | 18 | 3828 |
  | power-droplet-256 | 128 | 128 | 3828 |
  | power-fill-256-800 | 800 | 162 | 17856 |
  | power-fill-256-6400 | 4356 | 162 | 78672 |
  | minimal-power-dam-break-32 | 125 | 8 | 117096 |
  | minimal-power-dam-break-64 | 729 | 27 | 524568 |
  | mass-conserving-figure-9-dam-break | 2601 | 75 | 720384 |

  The test also checks that the boot fits its pools and that the dam's authored pools are kept.
- The census oracles take the guard:
  - brick-census.test.ts: the oracle's `decide`;
  - brick-census-dam.test.ts: R2's expectation over the old dam's frame-43 census.

  The dam test reaches the guard's branch (3 and 229 calm tiles kept h), and asserts it.

**Gates**
- check:types: lane-n's 3 only.
- test:unit: 1889 pass, 0 fail.
- GPU census units (brick-census, brick-census-dam), timing slot: 5 of 5, rc 0, no 00000206.
- Paired dam-300 against HEAD a8d2d90e, fenced, timing slot: both judged Overall PASS against ref-2761350a, with the
  same D-rows on both trees.

  | row | HEAD | guard |
  |---|---|---|
  | D1 front | 84 | 84 |
  | D6 wet floor | 7645 | 7645 |
  | D4 drift (30–90) | −0.0458 | −0.0463 |
  | D5k (J) | 42.38 | 42.62 |
  | D8 particles | 439297 | 449623 |

  Step time over steps 13–165, with the paired per-step difference over 139 steps:

  | tree | median (ms) | mean (ms) |
  |---|---|---|
  | HEAD | 35.61 | 40.38 |
  | guard | 35.29 | 40.02 |

  The paired median difference is +0.10 ms, inside noise. The dam's census text changes only by the guard's AND.

## 2026-11-17: solidStamp's box-wall face is the old shell voxel's (lane P's A2–A4 review; landed)

**The fix.** solidStamp wrote AP = 0 on a box-wall face, because both sides had to be in the box. With the mask
bound, nbStaticFaceOpen already returns the old engine's shell-voxel face at a wall: 0.5 (open(−1) + open(0)), that
is 0.5 FRAC of the cell inside. That value agrees with:
- the cut-free rows' K_WALL fraction 0.5;
- P's pbStampFace.

solidStamp now writes the hook itself. Two cells past the box give 0.

**Tests.** The GPU solidStamp test's mirror is the hook's own face. The test asserts that some wall faces read 0.5.
- GPU solidStamp and static tests: 8 of 8, rc 0, no 00000206.
- check:types: lane-n's 3 only.
- test:unit: 1889 pass, 0 fail.

**The dam.** No scene binds the mask, so the dam's schedule and kernel text are unchanged.

## 2026-11-17: setup refuses a container that is not whole pages (landed)

**The gap.** brick-scene only checked that a container was whole bricks. A container that was whole bricks but not
whole 32-cell pages passed setup, and then failed at creation:
- brickCensusKernels threw "the lattice is not whole pages";
- levelset-coarse latched 0x4c430001.

This affects 45 of the 69 pools-and-padding batch scenes. Examples are the 16³ ladders, 24×16×24, 32×16×32,
64×48×64, cm12 figures 2 and 3 at 128×128×8, ocean-seiche, and power-droplet-240.

Setup now refuses them with the gap "not whole pages: padding to 32-cell pages needs F's and P's walls at the box
(M3)". This was the coordinator's ruling: setup has to refuse what creation would throw on. A CPU regression checks
ceiling-slab-drop.

**The padding route (approved by the coordinator).** The storage lattice is padded to whole pages, as the old engine
does, and the box rule closes the cells past the container.
- The layout and the static hooks get a `box`, which defaults to the lattice so the dam's text stays byte-identical.
- F's particle clamp and level-set walls, and P's pressure wall faces, switch their wall planes to it.
- A wall appears exactly once, at the box, with no second wall at the storage edge.
- The shared regression is a 24×16×24 dam padded to 32³. Its liquid stays inside the box, and its volume matches
  the unpadded behaviour.

**Gates.** check:types: lane-n's 3 only. test:unit: 1889 pass, 0 fail.

## 2026-11-17: the box hook for padding to whole pages (landed; F's and P's walls to follow)

**What the hook is.** The design is in brick-abi.md §2 (the box rule). `BrickLayout.box` holds the container's
cells and defaults to the lattice. A kernel reaches the box through one of these:
- `boxWGSL(L)`;
- the `NB_WALL_BOX` and `nbInWallBox` tokens, resolved by the builder's `resolveBox`;
- `brickOwnerWGSL({ box })`.

**My sites, switched to the box:**
- the owner readers' walls (nbOnWall, nbFaceInLattice), in the census, set, transfer, residency and presentation
  kernels;
- setInit's FLAGS.OPEN, on F and on C;
- the boot velocity's inner faces;
- the transfer's wall faces (xV's 0.5, xFaceWidth, sFaceWidth);
- the census's cell kind;
- solidStamp's static hooks.

**The engine guard.** BrickEngine.create refuses:
- an unresolved token;
- on a padded lattice, an owner reader whose walls are still at the storage edge.

On a padded dam layout the guard names 25 kernels of F and P, and none of mine. F's other wall planes (the particle
clamp, umWallWithin) are not owner readers, so the guard does not see them. The shared regression covers them.

**The regression to come.** A 24×16×24 dam padded to 32³: its liquid stays inside the box, and its volume matches
the unpadded run. Until F and P switch, brick-scene keeps the "not whole pages" gap.

**Text and tests**
- The dam's 113 kernels (boot, part 1, part 2; P's dump) are byte-identical to b66537fb.
- brick-abi.test.ts, a CPU test: the hook's names and resolution; my padded kernels are all resolved and read the
  box (setInit, censusFine, setTransferFaces); a box outside the lattice throws.
- check:types: lane-n's 3 only.
- test:unit: 1889 pass, 0 fail (before the new test).

## 2026-11-17: the dynamic-write lowering as a diagnostic switch (8f1c47eb), the D2 pool term (a3fed883), BRICK_CELLS (0a81b19e)

**The pass** (brick-lower.ts). It is a diagnostic switch, off by default: NB_BRICK_LOWER=1 or the engine option
lowerDynamicWrites. Off, every module is created from its text exactly as written: the dam's 113 kernel hashes are
identical to a3fed883. On, at every brick compile point (BrickEngine.create before any module is made,
BrickEngine.run, BrickPresentation.create), each module's dynamic vector-component writes are lowered:
- `v[i] op= x` becomes `v=select(v,v op vecN(x),nbDynAxisN(u32(i)))`, for + - * / % & | ^ << >>;
- `v[i]=x` becomes `v=select(v,vecN(x),...)`, and `v[i]++` / `v[i]--` add or subtract vecN(1).

select is componentwise, so the result is exact. N comes from the target's nearest visible declaration:
- the declaration can be a local, a parameter (in the header only, never a call's argument) or a pointer's pointee;
- it can be a struct field, walked from the base's struct, or a constructor or a called function's return type;
- closed blocks and other functions' locals are not visible.

Arrays, atomics, buffers and literal indices are left alone. A wrong width is a WGSL type error at creation, never
a silent change. Two cases stop creation outright:
- brickLowerChecked throws, naming the kernel, if a write survives the lowering;
- the lowering throws on a field whose struct it cannot name.

The old engine is out of scope (it goes at M6).

**Go/no-go** (Metal; bodies-shapes' couple, the hydrostatic sphere, r 0.2). P's verbatim loop,
`var low=id;low[axis]-=1; area[axis]=...`, was run three ways: as written, lowered, and in the module's select form.
All three give 10.8421 N·s against ρgV·dt = 10.9579 (−1.06%, inside the 2% gate). The −6.71 miscompile does not
reproduce on today's driver, so the miscompile may have been a symptom of that morning's GPU hangs. Runs:
fluid-scratch/lane-e/r20/gonogo3.log. The test is sweep-tree's gonogo-lowering.test.ts, which is scratch and not
landed.

**Exact and explicit.**
- brick-lower.test.ts (CPU) covers a corpus of every width (vec2/3/4) and operator, `=`, `++`, `--`, and widths from
  initialisers, copies, parameters, calls and fields. It also covers:
  - arrays, atomics, storage and workgroup buffers, literal indices and comparisons, all left byte for byte;
  - right-hand sides with brackets and calls, and a for-header update;
  - the axis masks;
  - the guard;
  - the dam's whole schedule.
- The scanner on the dam's schedule (113 modules): **1269 dynamic writes before, 0 after** (1267 at 062ae58b; M4 added
  two). The test reports the count on every run, with no upper limit.
- The first GPU suite found one bad width. ncRangeKeys' `var keys=ncIdentityKeys()` returns array<u32,6>, and the
  lowering took it as a vec3. It failed at creation, as designed (censusFine, 40 tests). The fix was to read a called
  function's return type.

**GPU units with the pass on** (sweep2 = 062ae58b + the pass + D2, timing slot, serial): brick-*.test.ts and
bodies-stage, **85 / 85**, rc 0, no 00000206. These include the census oracles:
- setCensus against the old engine's dam census, frames 14 and 43;
- X8 bit-exact against G1;
- the synthetic census.

**Cost, and the ruling.** Creation: 0.7 s of CPU for the dam's 113 modules (memoised module scope and structs).
Step: dam-300 paired A B A B against 062ae58b (tiered composite, poolF 4913), all four Overall PASS. Summary medians:
HEAD 29.81 / 28.91 ms, lowered 30.29 / 30.33 ms; per-step medians from the cpu rows: 31.00 / 29.98 against
30.90 / 31.28 ms. That is **+0.6 to 0.95 ms a step**, over the 0.3 ms bar. A kernel profile of steps 13–24 puts
+0.51 ms on the kernel sum, spread thinly: at most 0.07 ms in any one (pMove, pSeed, censusFine, pmgNativePost1,
censusCoarse, ...). The coordinator ruled for a switch, not a default or an exemption list: step time first, and
the miscompile doesn't reproduce. The dynamic-write rule for new code is now guidance, not a requirement. P's
module and solidStamp keep their select forms. With NB_BRICK_LOWER=1 on main, brick-engine, brick-census and
brick-source gave 32 / 32 on the GPU.

**The D2 pool term** (approved: fix the rule, not the scene). bootF = max(boot brick list, |D2|), where D2 is the
liquid bricks' Chebyshev dilation by two within the window. That is the engine's first residency: the free fall's
step-1 F box, x 1..6, y 2..8, z 0..4, is 210 bricks, exactly D2.
- Only the analytic falls change:
  - free fall: 225 → 263;
  - tilted fall: 225 → 307;
  - translation: D2 is 180, so f stays 225.
- The nine measured scenes keep their pools. brick-engine.test.ts asserts both falls and free fall's |D2| = 210.
- Tilted free fall now passes its pools, highF 238 of 307.
- Free fall gets past step 11 but latches POOL_F at frame 36, after impact. With f = 405 (the window) its peak is
  **289 resident at step 42**: the splash covers the whole floor window (bricks 0..8 × 0..8 × 0..4), 6.4 brick
  layers. The rule's spread term counts a four-layer film (SF = 180). So free fall stays a named fail: a falling
  block's splash peak is over the rule. A splash term from the scene's energy is being measured (below), not landed.

**Batch status** (the coordinator's batch rule, 60 steps against the old engine; tables in r18):
- **Pass (12):** minimal-32, torus, rest, standing-wave, analytic dam-break, mound, droplet-64, dam-64, cm12-fig-5,
  cm12-fig-6, droplet-128, fill-256-100.
- **Named fails:**
  - free fall: POOL_F after impact (above).
  - translation: POOL_F|TIER at frame 42.
  - figure-9-dam-break: peak −6.54% at step 5, against −1.2%. At 1cc3724e, with F's dc195b11 wholly-4h controller fix,
    the first 20 steps are −0.02 … +0.58% and the gauge agrees. So the old failure was the controller bug; it needs a
    60-step re-smoke.
  - cm12-figure-9 and coarse-first-pool-impact: rc 134, Dawn's UploadData abort on boot tables of 7471340 and
    11927924 bytes. A lone 7471340-byte write aborts and a 16-byte multiple does not. F's uploadWords (9e6b9295) fixes
    it; they need a re-smoke.
  - stair-cube (+9.62% against +6.16%), fill-256-800 (+20.61% against +1.17%) and fill-256-6400 (+7.53% against
    −3.02%): the gauge agrees with the presentation, and the budget stays 0. It is a real volume gain on moving
    liquid. The coordinator gave the chase to F, after M4.
  - cm12-fig-7 and fig-7-256: repeat runs of each engine are owed, to set the tolerance.

**BRICK_CELLS** (P's layout rule). The six brick test files read BRICK_CELLS from brick-abi: no literal 512 brick lengths
are left (the TCAP rule's 512 c and a t capacity of 512 are not layout). On the GPU, 37 / 37.

## 2026-11-17: batch 1 enabled (20e1e2e4), P's kick seam fix (1a0c8c24)

**Batch 1.** The 12 that passed, plus mass-conserving-figure-9-dam-break, cm12-figure-9 and coarse-first-pool-impact
(their re-smokes after F's controller fix dc195b11 and uploadWords 9e6b9295), were re-smoked from a frozen tree of
0a81b19e against the old engine (60 steps, two old repeats where noise needs it). All 15 pass the batch rule: brick
|peak| and |final| no worse than the old engine's worst repeat + max(1%, its run noise), with a clean smoke (rc 0,
no 00000206, no latch). They are BRICK_ENGINE_SCENES now; the table is in 20e1e2e4's message and in
fluid-scratch/lane-e/r22/batch1-table.md. Every brick step median is under the old engine's (3.3 to 29.8 ms
against 8.9 to 41.5 ms). The gap message names the list (`not enabled on the brick engine`).

**P's kick seam fix** (P's patch as written): the boot's projected half kick runs 1 + BRICK_KICK_RESOLVES (23) solves
from its own pressure (the old arena's encodeSolve(kick)), then pmgKickScale doubles X into a step's dt units before
F's after. The kick-schedule test expects it. GPU 28 / 28.

## 2026-11-17: R2's volume guard, the re-entry seed at heat 1, and why psSeam stays

**R2's volume guard** (F's criterion, the coordinator's ruling: guard, don't conserve at demotion). censusFine writes
δ for every crossing h tile into CENSUS.VOL (a word a tile, f32):
- δ = (64·fill(the tile's 8 corners) − Σ₆₄ fill(each cell's 8 vertices)) / max(A, 1), with the six-tetrahedron fill
  (lib/core/level-set-fill.wgsl.ts) and A the cells whose 8-vertex mean |φ| < 0.5h; a tile with no crossing writes 0;
- the cell sum is a fixed-point atomic (2^20), so it does not depend on lane order;
- censusDecide's calm demotion of a band h tile now also needs |δ| ≤ UNIFORM_TILE_UPDATE_BUDGET_H (0.08): a tile
  whose 4h corner fill would move its liquid by more than the band's update budget stays h.

The GPU census matches the Float64 oracle (brick-census.test.ts's setCensus) on all 215 h tiles of the synthetic
scene: worst |error| 9.5e-7, δ from −4.19 to +0.33, 42 tiles over the budget. The test asserts δ is exercised off
zero, so a planar scene cannot pass it vacuously.

**The re-entry seed at heat 1** (effect 2, approved as a verbatim port): pSeed's re-entry (cfg.w = 2) seeds at heat
1, the old engine's support without erasure, not max(heat, 2). HEAT4's 2 is unchanged.

**psSeam is needed, not a stand-in** (effect 3). The coordinator asked for the cause: keep the records across R1's
retirement as the old engine does, or show why the seam reseed is needed. I tried both, and the narrow patch, on F's
lone-ball probes (ff.mts --variant=boot --nopool=1; L at rest s30, translating at 2 m/s s12, falling s20; particles at
rest, s30). Every row has the guard and heat 1:

| variant | rest | translating | falling | particles |
|---|---|---|---|---|
| A: psSeam as HEAD | 122.24 | 120.79 | 122.88 | 19.8k |
| B: R1 keeps the seam ring's records (4h tiles with an h tile among their 26), psSeam reverted | 118.98 | 118.96 | 121.53 | 37.4k |
| C: the narrow patch, uncovered = n0 == 0 && (pUncovered ‖ psSeam) | 119.10 | 119.64 | 121.67 | 19.1k |
| D: B's seam ring and psSeam | 122.49 | 120.99 | 122.96 | 45.7k |

References: HEAD at rest 116.57; all-h rest 122.32, translating 121.75; the old engine translating 122.05, falling
122.03.

Keeping the old engine's records at the seam does not bring the seam's h cells back (B loses 3.3 L at rest against A,
and doubles the particles). So the missing inflow across the seam is not the whole cause. The seam's h cells must
reseed even when a record is among their 27 (C, which requires none, loses as much as B). D adds only 0.25 L over A,
for 2.3x the particles. So psSeam stays as HEAD has it: it is the rule the brick engine's retirement needs, not a
patch over a missing cause. The coordinator approved A.

**The gate set** (frozen trees: 1a0c8c24 against 1a0c8c24 + the guard + heat 1; timing slot; every run rc 0, no
00000206):
- the probes, HEAD against A: rest s30 116.27 → 122.58 L, translating s12 119.66 → 120.81, falling s20 122.79 →
  122.86;
- against r18's references, 60 steps, final %. These are the named moving-liquid fails F owns, and the set leaves
  them where they were:
  - stair-cube: 9.62 → 9.66, against the old engine's 6.16;
  - fill-256-800: 20.61 → 19.48, against 1.17;
  - fill-256-6400: 7.53 → 7.42, against −3.02;
- dam-300, paired A B A B, nbflip-gates against ref-2761350a: A passes both rounds. HEAD's second round failed D5e,
  the energy window, at 315.9 against 328.0 ± 11.7, so that is HEAD's run noise. Step medians: HEAD 30.87 and 30.41 ms,
  A 31.18 and 31.32 ms. The paired per-step difference medians are +0.04 and +0.10 ms.

## 2026-11-17 (r22): body rule, splash term, padded runs

**8dce41bf.** P's census body rule and the engine's body wiring, landed as P wrote them.
- Kernel texts are identical to 0b021a46 for all 16 enabled scenes.
- brick-engine, solver-app, bodies-stage and seam-rest: 33/33.

**c8797626, the pool rule's splash term.**
- **What it is.** A scene whose liquid starts at least 16 cells above its rest depth takes
  f ≥ 10·min(floorF, ⌈1.5 V/64⌉), which is eight brick layers with the 1.25 margin, and particles ≥ 10 V.
- **It is empirical.** The term is fitted to five measured peaks (the table is in brick-scene.ts and the rollout
  doc §3). Stair is the weakest point: it is the only V-limited fit, and it sets the 1.5.
- **The audit covers all 142 catalog scenes.**
  - 10 change, none of them enabled.
  - hero-garden-hose-x10 would take 3.2 GB of core buffers. It stays gapped and must author its pools.
  - No pool is more than 2x its measured need. The worst is the tilted fall's 1.70x, at its window.

**84db4da5, padded runs.**
- **The C level set walls at the box.** Both lc owner instances take BOX_TOKEN. The lc functions' holds, wall
  planes and cell and tile domains read the box, through a new nbTileInWallBox token.
- **The presentation closes cells past the box** (volume 0, openFraction 0). Before this, the padded boot showed
  18.6 cells of volume past the box, interpolated from the box-plane vertex row.
- **P's gap patch** lets whole-brick containers set up padded.
- **The regression is tests/brick-padded-dam.test.ts:** a mirror-symmetric slab in 24x16x24 padded to 32^3, 30
  steps.
  - No row and no volume past the box.
  - Up to 1671 box faces, all WALL, and no stray walls.
  - The mirror is at most 6.5e-4 beyond the boot's own 5.2e-3.
  - The drift matches the same slab unpadded (+6.2% against +5.9% at peak), so it is F's floor-sheet gain, not the
    padding.
- **The boot is asymmetric.** brickInitialPhi, the old engine's t = 0 φ of a box volume, gives the slab's low-x face
  column 276 cells against 288 at the high-x face.
- **Open.** F's box observer (tools/nbflip-observe-brick.ts) refuses a padded engine, so padded volume gates wait on
  it.

**Old-engine references await every step.** r18/old-smoke.mts calls advanceTo and then awaitFrameCompletion each step.
The brick solver's advanceTo encodes exactly one step a call, or refuses.

**Parked: lcRetire (P's attribution, 2026-11-17).**
- **The cost.** lcRetire costs about 0.4 ms a step on every tiered scene: 0.42 ms on the 128^3 dam at step 20 and
  0.46 ms on a one-page tank.
- **Where it goes.** The whole cost is the build path's dependent-load chain, one vertex a lane: lcNearKnown's walk
  over 4^3 tiles, the 5-step Newton's 8 corner reads per iteration, the bound's 7 samples, and nbfCellDistance.
- **Two bitwise fixes, if it is picked up** (judge them by hashing PHI1):
  1. Reuse Newton's corners while it stays in one cell.
  2. Stage lcNearKnown's per-tile flag as workgroup bits, one plane plus its halo.
- **Expected gain:** 0.2–0.25 ms, under 1% of the dam.

## 2026-11-17 (r23): the splash finals on the rule's pools

The tree is frozen at 84db4da5. All runs used the timing slot, every one returned rc 0 with no 00000206, and no run
latched. The 60-step peak is judged from the same curves. "Brick" is the mean of two runs; the bar is old worst +
max(1%, old spread).

| scene | peak at 60: old (2 runs) | bar | brick | final at 120: old | bar | brick | result |
|---|---|---|---|---|---|---|---|
| free fall | −24.55, −23.85 | 25.55 | −24.85, −24.94 (24.90) | −0.05, 0.18 | 1.18 | 1.43, 1.54 (1.49) | final FAIL |
| tilted free fall | −26.23, −27.69 | 29.15 | −26.58, −27.00 (26.79) | 1.60, 1.54 | 2.60 | 2.25, 2.81 (2.53) | PASS, enabled |
| cm12-figure-7-256 | rs1 (60 steps) | 10.47 | −9.30, −9.20 (9.25) | see below | | 1.84, 2.26 (2.05) | peak PASS |

- **The dip at step 20.** The −24% at step 20 is the impact frame in both engines, and it is the same measure in each.
- **Peak F** stayed under the pool in every run:

  | scene | peak F | pool f |
  |---|---|---|
  | free fall | 297 | 405 |
  | tilted free fall | 218–222 | 405 |
  | fig-7-256 | 8872–9263 | 10890 |

  fig-7-256's margin is 1.18x. The runs took 2193 MB with the presentation included.
- **Step medians:**

  | scene | brick | old |
  |---|---|---|
  | free fall | 8.5 ms | 15.4 ms |
  | tilted free fall | 8.6 ms | 15.0 ms |
  | fig-7-256 | 66.9 ms | |

- **fig-7-256's old reference fails before 120 steps.** Its own page pool exhausts at frame 79. With 318 blocks it
  still exhausts at frame 84.
- **fig-7-256's old reference at the full box** (514 blocks; old-smoke --page-blocks is a test-only setting): final
  −10.29%, peak −26.24%, V1 240k of 268k, step median 96 ms. The bar is 11.29, and the brick's +2.05 passes. The
  coordinator enabled the scene and asked for it to be re-judged after sym8.

## 2026-11-17: the open top (M3; r24)

**The rule.** The old engine keeps AIR pages above an open box, and its x and z walls rise to its frame's top. Liquid
that reaches the frame's top is its page-edge fatal.

The brick engine does the same with fixed storage:
- brickSceneSetup gives an open top a lattice of `[Nx, whole(Ny) + BRICK_OPEN_TOP_HEADROOM, Nz]`, where the headroom
  is 32 cells (one page).
- The box's y is the lattice's, so every wall consumer (F's, P's, the census, the C level set, the presentation)
  already has the x/z walls rising to the lattice's top. Nothing is a wall at Ny.
- The air above Ny is in the box: it is free surface, with p = 0.
- The lattice's top is a lid the old engine doesn't have. setMark (layout.openTop) latches FAIL.REACH with info bit 31
  plus the brick's key when the last brick layer holds an h vertex with φ < 0, a particle tile, or, tiered, a liquid
  4h corner. describeFail names the headroom, so the lid is never met silently.
- farPhi stays the old engine's, from whole(Ny).
- Initial liquid: the closed form and brickInitialPhi read the container's own height (`containerCells`).
  - The closed form's slab keeps its top face, with no continuation through a lid.
  - The φ table reads the old φ up to whole(Ny), the old engine's AIR column, and is far air above that.
- An unknown `container.top` is a named gap.

**What it unblocks.** It removes the open-top gap from 18 scenes.
- bulk-rest-tank now has no gaps.
- paddleboats and churning-tank are left with P's K-slot pools.
- The seep and the troughs are left with the static voxels.

**Tests.**
- CPU (brick-engine.test.ts):
  - bulk-rest-tank sets up at 256x320x128 with openTop and farPhi 288;
  - paddleboats' slab top is 24;
  - an unknown top is refused;
  - a closed top's setMark has no latch text, and an open one has both latches (tiered: the 4h corners);
  - a layout whose open-top box is short of the lattice throws;
  - describeFail decodes the key.
- GPU (tests/brick-open-top.test.ts): a 32x16x32 open-top dam in a 32x64x32 lattice. Its run-up must stand above
  y = 16, with no liquid in the last layer and no latch. A blob thrown up at 8 m/s must latch REACH, naming the open
  top.
- The enabled scenes' kernel text (P's kernel-hashes.mts, 16 scenes) is identical: the latch is only emitted when the
  top is open.

## 2026-11-17 — the long no-latch run (r27), the droplets' film, and their authored pools

**The rule** (main, 2026-11-17): every batch now includes a long no-latch run, each enabled scene for max(authored, 300)
steps on its own pools. r27 ran the 21 enabled scenes on frozen e001d490 (rigid scenes with their roster on 29d53bbc),
timing slot, rc and 00000206 checked after each run: none invalid.

**Result.** 18 of 21 ran clean. The three 100-cell scenes latched POOL_F on the rule's f = 128: power-droplet-64 at
frame 103, power-droplet-128 and power-fill-256-100 at frame 99.
- Peaks against pools, for the clean runs: high-resolution-dam-break (1200 steps) F 4110/4913, C 125/125 (the window);
  cm12-figure-5 2196/4913; -6 1591/2943; -7-256 8398/10890; -9 2391/2601 with C 75/75 (the window);
  mass-conserving-figure-9 2215/2601; coarse-first-pool-impact 2042/3757 with C 95/100; the analytic set 180-280/360-405;
  the 32^3 / 64^3 dams and the torus at their windows (125, 729, 125). Particles at most 59% of a pool.
- The rigid scenes (smoke2 with initializeRigidBodies' roster; the bare smoke latched LIST|UNSUPPORTED at frame 0 for
  want of the roster): hydrostatic and float F 27/41, K 8/12, drift 0 / +0.14%; sink F 36/54, K 12/18, particles
  5430/8669, final +1.01%.

**The droplets' F growth is a real film, not a residency leak.** A probe (pool f 1024-2048, every 10 steps) split the
resident bricks by class and adjacency:
- Every wet brick (LIQUID, SURFACE or PARTICLES) is on the floor layer (brick y = 0), from step 10 to 300.
- Every other resident brick is within one brick of a wet one; the orphans are 0, with at most 8 for a step (the keep's
  linger).
- At 64^3 the wet footprint grows from 2x2 bricks to the whole 8x8 floor by step 90-150, and F plateaus at 162 (63 wet
  plus their ring) from step 150. At 128^3 and 256^3 it is still spreading at step 300: 16x16 and 20x17 bricks, F 418
  and 452.
- The particles stay about 740 while the footprint grows, so the film thins to 12 a brick (0.2 a cell) at 64^3. At 128^3
  and 256^3 they fall to about 440-480 by step 300, and the φ volume to -87% / -90%.
- The old engine shows the same film at 64^3: +50% at step 55, then -77% at step 300 (its φ measure). The brick engine
  is at -19%.
So the inviscid, tension-free droplet spreads over the floor as both engines model it. The pool rule's spread term
assumes a film 1/16 cell thick, and this one is about 1/46.

**Fix.** The three scenes author brickPools at 1.5x the measured peak (P's convention for the rigid scenes):
power-droplet-64 f 243, power-droplet-128 f 627, power-fill-256-100 f 678. Their c and particles are the rule's, and t
is the layout's 2 f. The pool rule's comment records why: a tension-free small liquid's spread is time-bound, not
closed-form. The rule's table test drops power-droplet-64.

## 2026-11-17 — the census's static need rule (M3), on P's 29d53bbc

**The rule** (brick-static-need.wgsl.ts, after censusDecide, tiered with the static mask only). It is the old engine's
static promotion (uniform-mixed-dynamic.ts solidActive / solidPromote), made brick-granular because the operator needs
every tile of a cut brick at h.
- censusStaticActive: X8 over C_ALL, a lane a tile. A brick is cut when a mask voxel is in its cells or just below its
  lower faces (solidStamp's rule). It is active when one of its tiles holds decide's frozen WANT, its F record is
  LIQUID or PARTICLES, or one of its 4h tiles has a C corner below 0. The bits go to the page's word in `staticNeed`.
- censusStaticPromote: a cut brick that is active, or one of whose 26 neighbours is a cut, active brick, takes all 8 of
  its tiles into WANT, with cdTile's ring (MARK_F at a head, candidates, the clamp counter). It writes no DEC/STAT, so
  there is no ratchet. `staticNeed[4 C]` counts the tiles it promotes.
- With the rule, the census drops its static-K latches: censusBegin's LIVE_K and censusFine's 0x63000004. The
  TRIG-only boot census drops them too (a boot's cut K slots are the rule's) and promotes nothing.
- A body's K slot is still held by P's body rule. A dry cut tile at 4h is allowed by P's narrowing (29d53bbc).

**Tests.**
- CPU: the kernels and the dropped latches follow the mask, tiered; the schedule order is decide, active, promote; the
  TRIG census has neither pass; without the mask the text is unchanged.
- GPU: a still tank (liquid y < 17) under a shelf at y 19..23 and a dry block at y 40..48. Over 30 steps: no latch,
  every wet cut brick fully h, 15-16 tiles promoted a step, and the dry block's cut bricks never resident (so never
  promoted). Without the rule this setup latched 0x63000000 at the boot.
- A column standing in the water latched 0xa2 (P's operator: a liquid row in a static solid cell) within one step.
  That is the next work, phiSolid and pSolid; the regression keeps the solids out of the liquid until then.
- The 21 enabled scenes' schedule kernels are identical (none has a mask). test:unit 1891/0.
