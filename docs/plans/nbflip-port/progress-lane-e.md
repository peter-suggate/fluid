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
