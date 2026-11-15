# Lane F (fluid kernels) progress

Design: `docs/plans/nbflip-structural-2026-11-15.md` (§2.4–2.6, §4.2, §5, §7.1). Scratch: `~/code/me/fluid-scratch/lane-f/`.

## M0(c): the p1b "collapse" root-caused (2026-11-15)

**Verdict: p1b never collapsed. The runs that showed ~7.9k particles ran a different scene.** There is no hidden
dependency between the move and `nbSurfaceShift`; moving the shift out of the move is safe for `pBand`.

**Cause.** P2's p1b attribution runs passed `--scene=letters` to `tools/nbflip-attribution.ts`, whose scene
argument is a raw scene id. `getSceneDefinition` (`lib/core/scenes.ts:2684`) falls back to `SCENE_CATALOG[0]` on an
unknown id, without an error, so "letters" ran `water-box-dam-break` (no NB-FLIP profile, balanced defaults). The base
and p1a runs passed `--scene=nbflip-figure-8-letters`. The logs show it: `attr/L120-base.log` and `L120-p1a.log` print
`"scene":"nbflip-figure-8-letters"` (886k / 885k particles, 7972 / 8026 fine tiles), and both p1b logs print
`"scene":"letters"` (8057 / 7983 particles, 80 / 81 fine tiles, 25 ms a step). The "check on is healthy" run used
`tools/nbflip-move-check.ts`, whose `--scene=letters` is a key mapped to `nbflip-figure-8-letters`. So "check on /
check off" was confounded with "right scene / wrong scene"; the check itself changed nothing.

**Reproduction** (check slot, rc 0 each, f72b3ca6 exports, `tools/p1b-probe.ts`; logs in `lane-f/logs/`):

| tree | `--scene` | resolves to | particles at step 119 | fine tiles |
|---|---|---|---|---|
| base f72b3ca6 (unpatched) | `letters` | water-box-dam-break | 8,020 | 84 |
| p1b, check off | `letters` | water-box-dam-break | 7,961 | 94 |
| p1b, check off | `nbflip-figure-8-letters` | nbflip-figure-8-letters | 884,328 | 7,835 |

The unpatched engine gives the same ~8k on the fallback scene, and p1b with the check off is healthy on letters
(884k against the base's 886k in P2's run; NB-FLIP is nondeterministic). With P2's in-situ check (0 bitwise
mismatches over 120 letters steps, about 94% of corrections taken from the pre-pass), p1b is the same computation as
the base.

**Consequences.**
- §2.4's hazard is closed: `pBand` may form `nbSurfaceShift(q, nbSeparation(...))` from the pre-move bank in a
  separate pass. The surface shift samples the depth field at the record's pre-trace `q`, which the move never
  writes, so a pre-pass and the in-move call are the same function of the same inputs (P2's check confirms it
  bitwise). The only fallback cases are a cell-cap merge that moved the sample or a rigid contact that pushed it.
- P2's p1b timing (`L120-p1b.json`) is of the wrong scene and is void; p1b on letters is untimed. It is moot for the
  new engine, and not worth landing on the old one.
- Tooling: `tools/nbflip-attribution.ts` now uses `findSceneDefinition` and fails on an unknown id (AGENTS.md:
  unknown kinds must fail explicitly). `nbflip-gate-bench.ts` and `nbflip-move-check.ts` map keys and were not
  affected.

## Step 2: the formulas ported, with frozen-state parity (2026-11-15)

**Verdict: every formula on lane F's list is a pure WGSL function, and each matches the old engine bit for bit on
the GPU, except the limited tricubic, which is held to 2 ulps of its taps.** One combined check-slot run (rc 0,
`lane-f/logs/all1.log`, no Metal aborts in `/usr/bin/log`): 16 tests, 0 failures.

**Method.**
- Old text. `tools/nbflip-parity-extract.ts` captures the uniform engine's resolved WGSL for the dam at a22d512d
  (every `createShaderModule`) and writes the 74 functions the formulas port to `parity/uniform-dam.wgsl`. Each
  section names its module.
- Kernels. Each test compiles the old functions whole. Where an old function addresses the old storage, the test
  uses excerpts instead, and `oldExcerpt` checks each one is verbatim. Stubs stand in for the world the functions
  read. A second kernel runs the new `nbf*` functions over the same stubs and inputs, and the test compares the
  two outputs' bits.
- Bind groups. `runParity` fails a kernel whose bind group is invalid. Without that check, a binding the pipeline
  drops leaves both outputs zero and "matching", and two of my early "0 differ" results were void for that reason.
- Inputs. The particle tests use 2,150 records from the dam at step 120: a 10-cell block around a shallow record,
  in `parity/dam-120-block.bin`. They also use a synthetic crowded set of 1,132 records, with 12–20 a cell, every
  mass, and ballistic, seed-first, dead and FLIP-less records, which reaches the merge, the mass bound and the
  split. The other tests use seeded random states that cover each branch.

**Parity table.**

| formula | new (file: functions) | old | cases | result |
|---|---|---|---|---|
| record packing | particles: `nbParticle*`, `nbWithMass`, `nbLive` | same names | 2,150 dam + 1,078 synthetic records | bitwise |
| G2P / FLIP, ballistic kick and stops | particles: `nbfG2P`, `nbfBallistic*` | `nbAdvect` | in the move | bitwise |
| Heun RK3 (the stage loop), drift | particles: `nbfHeun*`, `nbfDrift` | `nbAdvect` | move and parts | bitwise |
| sparse age and expiry | particles: `nbfSparseAge`, `nbfSparseExpired` | `nbSparseAge`, `nbSparseExpired` | in the move | bitwise |
| erasure heat at G2P and on arrival, retirement | particles: `nbfHeatAtG2P`, `nbfArrivalHeat`, `nbfRetireDeep` | `nbAdvect`, `nbFinish` | in the move | bitwise |
| nbSeparation (Sato et al. 2018 pairs) | particles: `nbfSeparation*` | `nbSeparation` | 3,282 records alone, and in the move | bitwise |
| nbSurfaceShift | particles: `nbfSurfaceShift*` | `nbSurfaceShift` | 3,282 records alone, and in the move | bitwise |
| cell cap, mass bound, merge, absorb | particles: `nbfCellCap`, `nbfCellMassAdd`, `nbfMerge*`, `nbfAbsorb*` | `nbCellCap`, `nbMergeRun`, `nbMergeAbsorbed`, `nbAdvect` | in the move (crowded: 293 retired) | bitwise |
| box endpoint, clipped velocity | particles: `nbfBoxEndpoint`, `nbfClippedVelocity` | `nbTraceEndpointIn`, `nbFinish` | in the move | bitwise |
| split (hash direction, halves) | particles: `nbfSplit*` | `nbFinish` | 748 splits in the move, and 3,282 alone | bitwise |
| the whole move, composed | the test's `refMove` | `nbAdvect` + `nbFinish` | 2,150 dam + 1,132 crowded, 26 words each | bitwise |
| seeding (jitter, site gates, cell gate, targets, masses) | particles: `nbfSeed*` | `seedSite`, `seedCell` | 2,744 cells × 8 sites | bitwise |
| sampling (vertex trilinear, far depth, face lerp, composite blend, fine weight) | particles: `nbfVertexTrilinear`, `nbfFarDepth`, `nbfFaceLerp8`, `nbfCompositeBlend`, `nbfFineWeight` | `particleDepthIn`, `umVelocityLerp8`, `umSampleVelocityWeightedIn`, `umVelocitySamplingWeightsAt` | 4,096 | bitwise |
| Ferstl union, bound, erasure, blend | levelset: `nbfUnion*`, `nbfErasure` | `particleSurfaceFinish` | 8,192 vertices | bitwise |
| crossing guard (bulkDepth) | levelset: `nbfBulkDepth`, `nbfCrossingTravel` | `bulkDepth`, `nbCrossingTravel` | 8,192 | bitwise |
| heats: interpolated, couple θ | levelset: `nbfInterpolatedHeat`, `nbfCoupleTheta` | `nbInterpolatedHeat`, `nbCoupleTheta` | 8,192 | bitwise |
| cell distance, far sentinel | levelset: `nbfCellDistance`, `nbfFarSentinel` | `nbCellDistance`, `nbRedistanceVertex` | 8,192 | bitwise |
| surface trace (phiAdvect) and trace helpers | levelset: `nbfSurfaceTrace`, `nbTrace*` | `umSurfaceTrace`, `nbTrace*` | 8,192 | bitwise |
| trilinear φ | levelset: `nbfCellTrilinear` | `umVertexCell` | 8,192 | bitwise |
| limited tricubic with held taps | levelset: `nbfCubicLimited` | `umCubicPhi` | 8,192 (1,932 differ) | ≤ 2^-17 absolute (2 ulps of the taps' range; 2^-18 seen) |
| p2g deposit (nodes, fixed-point words) | velocity: `nbfP2GScatter` | `pbScatter`, `pbAdd` | 2,048 samples × 81 nodes | bitwise |
| p2g face depth, window, expected mass, θ, resolve | velocity: `nbfFaceDepth`, `nbfP2GWindow`, `nbfP2GExpected`, `nbfTransferTheta`, `nbfP2GResolve` | `transferPropertiesJob`, `nbTransferTheta` | 4,096 cells × 3 faces | bitwise |
| momentum trace with the segment cap | velocity: `nbfMomentumTrace` | `umUnitTrace` | 8,192 (dt up to 9 s reaches the cap) | bitwise |
| forces: occupancy, gravity face and kick | velocity: `nbfOccupancy`, `nbfGravityFace`, `nbfGravityKick` | `umOccupancy`, `forcesFineJob` | 8,192 | bitwise |
| extension: fast-sweeping update, seed codes, 4h prolong | velocity: `ufxCombine`, `ufxTerm`, `ufxSeedCode`, `nbfExtProlong` | `ufxCombine`, `ufxProlong` | 16,384 | bitwise |
| uDelta | velocity: `nbfDeltaSample` | `ndSample` | 2,048 | bitwise |
| remap: φ refine, 4h slope, h sample, 4h restrict | velocity: `nbfRemap*` | `remapTileCells`, `remapSlope`, `remapSample`, `remapFace` | 8,192 | bitwise |

**Not ported here.** `nbFinish`'s voxel walk. The dam has no embedded solid, and the old engine also skips the walk
there (`!umSolidEnabled()||uncut`). The walk goes in with solids at M3. The air clear in redistInit
(airSeeds, the ±4-tile spread, the 27-tile gate) is structure, not a formula, so it lands with the M2 kernel.

**What the bitwise work showed (and what M2's kernels must keep).**
- Metal's compiler rounds a sum or product chain according to the code around it. The same verbatim formula
  composed in a different control shape drifted by 1–2 ulps in about 2% of records: 197 words of the dam move.
  `nbfSplitAlong` hashes the position's bits, so that drift amplified to a whole quarter-cell split offset.
  Bitwise came back once the reference matched the old shape:
  - `nbSurfaceShift`'s single loop over k = 0..6;
  - the move's stage loop, which samples the field at one call site.
  The pMove kernel therefore keeps the stage loop. That loop is also what keeps one inlined copy of the field
  sampler (nbAdvect's own comment).
- An analytic stub inlined at several call sites rounds differently at each site. So the level-set world is data:
  φ is a vertex table, and the velocity is constant within each cell, from an exact hash. With that change the
  trace and the trilinear went bitwise.
- The one remaining difference is the tricubic's 64-term sum. The old kernel's tap loop sits inside a width branch,
  beside the solid and buried paths that the dam folds away. Matching the branch and the inner-tap source cut the
  differing threads from 2,651 to 1,932 but did not clear them. The bound is reassociation-sized, so I state it
  rather than chase the compiler further.

**Files.**
- `lib/methods/narrow-band/particles-formulas.wgsl.ts`, `levelset-formulas.wgsl.ts`, `velocity-formulas.wgsl.ts`:
  the formulas. Each module rejects the settings it does not implement (`nbfParticleSupported`,
  `nbfVelocitySupported`).
- Their `.test.ts` files: the parity tests, which skip without `WEBGPU_NODE_MODULE`.
- `lib/methods/narrow-band/parity/`: the harness, the fixture and the frozen dam block.
- `tools/nbflip-parity-extract.ts`: the fixture's generator.

**Next: M2.** The kernels go against lane E's skeleton (ac2fce26) and brick-abi.md §6: `nbSet`, `nbOperator`, and
`active` reserved.

## M2: the kernels on the brick engine (2026-11-15, first part)

**What landed.** Lane F's stage set `fluidStages` (`brick-fluid-stages.ts`). It is not registered yet:
`brickFluidStages` stays undefined until the gates pass.
- `particles-kernels.wgsl.ts`: pMove, pScan, pScatter, pTiles, pSeed, pDelta.
- `levelset-kernels.wgsl.ts`: lInit, lAdvect, lSplat, lUnion, lTiles, lDilate, lClear, lNear, lDistance,
  lPublish.
- `velocity-kernels.wgsl.ts`: uAdvect, pDeposit, uViscous, uTransfer, uExtend, vGauge.
- `fluid-grid.wgsl.ts`: the shared addressing.

All of it is all-h on E's skeleton, and every formula comes from the `*-formulas.wgsl.ts` modules. naga accepts every
kernel. The dam runner (scratch: `fluid-scratch/lane-f/tools/brick-dam.mts`) uses fluidStages and P's poolMGPCG.

**Two formulas added since M1, both bitwise against the old text** (`velocity-formulas.test.ts`):
- The explicit viscosity (`nbfViscousLaplacian`, `nbfViscousKick`; old `umLaplacian`, `umForcedVelocityClear`),
  on 8,192 threads. The dam is μ = 1.002e-3 Pa s with free-slip walls. No-slip viscosity fails at build.
- Volume control, on 16,384 threads (planar, random, zero-vertex, inside, outside and roundoff-sized cells):
  - `nbfCellFill` (old `umSurfaceTarget` and `umTetraFill`): the gauge's occupancy.
  - `nbfVolumeRate` (old `solve`'s gauge branch): Kim et al. 2007's proportional term.
  vGauge sums the cell fills of each brick from the brick's 9³ vertices, staged in workgroup memory. Each brick
  adds its sum with one atomic into a 64-bit fixed-point pair (2^-20 cell), so the atomics' order cannot change
  the total. There is one pair per step parity, and the boot captures the budget. uAdvect turns the gauge into
  the rate in `FH.RATE` (header word 64), ahead of the operator.

**The dam on fluidStages + poolMGPCG, all-h** (check slot, `--poolF=4913`):
- It ran 165 steps with rc 0, no FAIL bits and no Metal aborts. Steps 13+ took about 62–66 ms (n = 132), on a
  shared GPU, so this is not a timing.
- Step 15 against the old reference: KE 275 vs 272 J, drift -4.3% vs -3.5%, the profile within a cell or two.
- The volume loss after that is the missing volume control, not the port. The old engine with
  `--set=volumeControlSeconds=0` loses -19.3% at step 30 and -30% at step 60. Ours loses -19.7% and -32%.
  - With control on, the old engine recovers to -0.7% by step 165.
  - The rate goes live once P's RHS reads it: RHS = −h·(div − rate·h·capacity), as the old
    `uniform-mixed-pressure-velocity.ts`. I asked P for this through E.
  - The gauge tracks the observer to 0.1–1% over 60 steps. It runs one step behind, as in the old engine.

**Stated deviations at this milestone (all-h):**
- No 4h tier: the deep interior is h. So composed state cannot be bitwise with the old mixed engine, whose
  pressure runs on 4h owners there.
- The TRIG stand-in is "the tile holds a crossing cell", until E's census.
- The spread travel is 4·ceil(FC_BOUND·dt/h / 4).
- There is no deep-air keep and no released-wall rule.
- The wall faces hold 0, as the old transfer did. They move to `nbFacePlane` with the owner readers.

**Blockers:**
- E: at all-h the authored f = 3968 runs out (POOL_F at step 20). setMark keeps every liquid brick at h, and the
  splash claims the whole 17³ lattice. Tiers fix this (`tiered: true` once F reads through `brickOwnerWGSL`).
- P: the volume RHS term.

## M2: the dam gate, all-h (2026-11-15)

Run: `dam-300a`, 300 steps on the check slot at 7b0df874 + P's RHS (6b06ae15), with `--poolF=4913`.
- rc 0, with no FAIL bits and no Metal aborts.
- Judged by `tools/nbflip-gates.ts` against the five 2761350a references (`fluid-scratch/lane-f/gates/dam-300a.md`).

**Overall: FAIL.** Every failing row sits at steps 2–16 of the collapse. Every window row passes.

| gate | result | worst margin |
|---|---|---|
| C0 300 steps | pass | 0 |
| D2, D3, D4, D4c, D5k, D5p, D5e, D5ce, D7t, D7d, D8 | pass | ≤ 0.77 |
| D1 front | **FAIL** | 1.30 (step 8: 103 vs 108; step 10: 116 vs 121) |
| D5c collapse KE | **FAIL** | 1.82 (step 4: 29.5 vs 36.1 J; step 16 within) |
| D6 wet floor | **FAIL** | 1.85 (steps 8 and 12: 9,954 vs 10,767; 13,553 vs 14,936) |
| D8c particles at 16 | **FAIL** | 5.12 (531,645 vs 351,600) |

**Cause: the TRIG stand-in.** It marks every tile with a crossing cell, so the whole surface seeds at boot. The
old census triggers by shape, strain, rotation, impact and approach
(`uniform-narrow-band-activity.wgsl.ts`), so at rest only the column's edges seed:

| step | particles (ours) | particles (ref) | front (ours/ref) | wet floor (ours/ref) |
|---|---|---|---|---|
| 0 | 412,608 | 0 | 80 / 80 | 6,400 / 6,400 |
| 1 | 419,500 | 75,687 | 81 / 84 | 6,405 / 6,993 |
| 8 | 535,009 | 146,197 | 103 / 108 | 9,954 / 10,765 |
| 16 | 531,645 | 350,675 | 128 / 128 | 15,831 / 16,144 |

With the whole surface hot, the faces follow the particles' FLIP keep instead of the grid. That slows the foot of
the collapse by about 5 cells, and the gap closes once the old run's surface heats (step 16). E's census
(`setCensus`, TRIG in `set`) replaces the stand-in. F adds the join bits (`TILE_OCC` 8–15) it reads.

Volume control is live and tracks the references. Drift against the reference mean:
- window 30–90: -4.7% vs -4.0%;
- window 105–165: -1.4% vs -1.0%;
- window 180–300: -1.1% vs -0.7%.

### Timing against the 2761350a baseline

Timing slot, 165 steps, clean tree at da593216, MGPCG, `--poolF=4913` (`fluid-scratch/lane-f/timing-7b0df874/`):

| | brick engine (all-h) | 2761350a |
|---|---|---|
| fenced median | 69.44 ms (p95 82.5, mean 68.4) | 45.45 ms |
| fenced thirds | 79.3 / 69.4 / 56.4 | 67.1 / 43.5 / 43.9 |
| pipelined | 67.32 ms/step | 50.61 ms/step |
| construction | 468 ms | 9,045 ms |
| GPU memory | 663 MB | 2,015 MB |

The step is 1.5x slower, while construction is 19x faster and memory is 3x smaller. All-h is the wrong shape to
time. The old engine runs the deep interior at 4h, and here every liquid cell is at h.

Attribution: `prof-60` puts each dispatch in its own timestamped pass, which inflates the step to 86.5 ms; the
kernels sum to 76.0 ms/step.

| group | ms/step | kernels |
|---|---|---|
| MGPCG (P) | 29.2 | 7.8 iterations; pmgUpdate/PostS/PreS about 4.4 each |
| grid transfer (F) | 27.9 | uAdvect 8.6, pMove 7.1, uTransfer 6.4, uExtend 5.7 |
| level set (F) | 10.9 | lAdvect 4.9, lDistance 1.7, lNear 1.1 |
| particles (F) | 5.2 | pSeed 2.2, pDeposit 1.7 |
| rest | 2.8 | project 1.1, operator 0.7, uViscous 0.5, vGauge 0.3, set* 0.15 |

The four transfer kernels and MGPCG run over every interior cell at h, so this is the cost the tiers remove.
uAdvect and uExtend visit every face of every brick. pMove carries the 412k particles that the TRIG stand-in
over-seeds, against the reference's 76k–146k at the same steps. Re-time after E's census and tiers land.

Composed kernels: an ulp-scale comparison against the old engine is not meaningful at all-h. The old engine's
pressure runs on a 4h interior, so the two engines solve different systems. Each formula stays bitwise in isolation
(`velocity-formulas.test.ts`, `levelset-formulas.test.ts`, `particles-formulas.test.ts`), and the gates judge the
composition.

## M2: the transfer cost per item (2026-11-15)

**Volume control is a port, not new physics.** The old frame (`uniform-mixed-frame.ts` 1160-1165) passes
`control = 1 / volumeControlSeconds` to the gauge's `beginStep` and copies the gauge's rate into the pressure
authority's parameters. The gauge's `solve` (`uniform-mixed-surface-volume.ts` 490-500) computes the same Kim et al.
2007 proportional term as `nbfVolumeRate`:
- the same gain, min(control, 1/(3 dt));
- the same 1% dead band (`NB_VOLUME_DEAD_BAND` 0.01);
- the same ±4/s limit (`NB_VOLUME_RATE_LIMIT`);
- the same divisor, max(filled, budget/2), and the same capture gate.

`velocity-formulas.test.ts` checks `nbfVolumeRate` bitwise against that excerpt. The one difference: the old filled sum
counts the 4h bulk cells (`svBulkVolume`) as full, while at all-h every cell is measured by its fill.

**Method.**
- Item counts are means over the profile window (steps 13-60), interpolated from the dam-300a samples. F_ALL holds
  4,517 bricks (6.94M faces, 2.31M cells); there are 762k particles and about 2.2M traced faces (liquid on either
  side, about 3 x the 0.72M liquid cells).
- Times are from the split-pass timestamp profile (`prof-60`, every dispatch in its own pass). The seep figures come
  from its own split-pass counter profile (lane S, S2o, step 1100: 485.5k + 17.2k traced faces, about 305k particles).
  Both profiles are inflated by serialisation in the same way.
- The thread limit is the register limit: maxTotalThreadsPerThreadgroup of the Dawn MSL compiled uncapped
  (`fluid-scratch/lane-f/tools/msl-stat.sh`, lane S's method). 1024 means no register limit.
- I did not take hardware-counter occupancy this round.

| kernel | items | ms | ns/item | seep equivalent | threads/group, register limit | workgroup memory |
|---|---|---|---|---|---|---|
| uAdvect | 2.2M traced faces (6.94M visited) | 8.64 | 3.9 a traced face (1.25 a face) | velocity trace: 1.3 a traced face, about 1 segment each | 256, 704 | 0 |
| pMove | 762k particles | 7.12 | 9.3 (separation 5.1, the rest 4.2) | g2p 1.2 + separate 1.0 = 2.2 | 128, 576 | 0 |
| uTransfer | 6.94M faces | 6.44 | 0.93 | (in seep's velocity pass 1) | 256, 768 | 3.9 KB |
| uExtend | 6.94M faces x 3 axes' windows | 5.69 | 0.82 | extend.2 + .1: 0.33 ms | 512, 1024 | 13.8 KB |

**Causes and fixes.**
- **uTransfer: redundant liveness reads. Fixed.**
  - The cause: every face recomputed fgCentrePhi for its own cell and for the cell below on each axis. That is 32
    corner φ reads a cell, each through two dependent directory loads.
  - The fix: the centre φ of cells o-1..o+7 is staged once (729 floats, 2.9 KB, the same sum in the same order), and
    the deposit words are read and zeroed by one atomicExchange.
  - Result: 6.44 to 2.57 ms (0.37 ns a face). The register limit is unchanged at 768.
- **uExtend: the same redundancy, three times over. Fixed.**
  - The cause: each axis staged 1,728 window cells, and each cell evaluated fgLiquidCell twice (16 corner reads).
  - The fix: the liquid bits of cells o-3..o+9 (13^3 bits, 69 words) are staged once for all three axes. Slot lookups
    go through a workgroup window of the 3^3 bricks (`FG_SLOT_WINDOW_WGSL`), which is bit-identical because the
    directory is read-only in a kernel.
  - Result: 5.69 to 3.10 ms (0.45 ns a face); the window alone is worth about 0.25 ms of that.
- **uAdvect: no per-segment defect. Left as is.**
  - The trace is the old engine's umUnitTrace. A segment is at most a cell of travel, and each one samples all three
    axes at its start and its midpoint.
  - The seep's quiet reservoir traces one segment a face. The collapsing dam moves up to about 5 cells a step, so its
    traced faces take several segments. Per segment the cost is at or below the seep's.
  - What makes uAdvect large is the count: 2.2M traced faces at h, where the old engine and the seep trace the deep
    interior at 4h. The tiers fix this; a per-cell change does not.
  - Tried and rejected (one profile run each):

| variant | ms | register limit |
|---|---|---|
| staged centre φ | 10.76 | 640 |
| slot window and staged centre φ | 9.18 | 640 |
| slot window alone | 9.65 | — |
| original | 8.64 | 704 |

- **pMove: the separation is latency-bound per candidate, not occupancy-bound. Left as is.**
  - Knocking the separation out (scratch only) takes pMove from 7.12 to 3.25 ms, so the separation costs 3.9 ms
    (5.1 ns a particle).
  - The split the design asked for, a `pSeparate` kernel at register limit 1024 ahead of a 576-limit pMove, costs
    4.37 + 3.15 = 7.53 ms against 7.12. The candidate loop runs no faster at full occupancy, and the split repeats the
    merge.
  - Not landed. Lane S found the same for the seep's separate (shared tiles and a flattened loop were both slower).
  - What remains is the candidate count: 8 cells x the particles a cell. The TRIG stand-in over-seeds the whole
    surface (762k particles against the reference's 146-351k at steps 8-16), so E's census cuts this.
  - A slot window in pMove (centred on the middle record's brick) was slower: 8.97 ms.
- **pDeposit:** the slot window made no difference (1.70 ms either way); not landed.
- **lAdvect:** the slot window made no difference (4.63 against 4.60 without it); not landed.

**Landed** (bit-identical by construction: the same values summed in the same order):
- `fluid-grid.wgsl.ts`: `FG_SLOT_WINDOW_WGSL`, built on the `fgBrickSlot` and `fgBrickSlotOfDir` split. `FG_SLOT_WGSL`
  is unchanged in behaviour.
- `velocity-kernels.wgsl.ts`: the staging in uTransfer and uExtend.

The observables stay within run-to-run spread. For example, at step 165, drift is -1.19% against -1.14% and KE is
9.258 against 9.261 J.

**Timing slot, 165 steps** (export of 60309e55 plus this patch, `fluid-scratch/lane-f/timing-staging/`):

| | before (7b0df874) | after | 2761350a |
|---|---|---|---|
| fenced median | 69.44 ms | **63.81 ms** (p95 75.3) | 45.45 ms |
| pipelined | 67.32 ms/step | **62.48 ms/step** | 50.61 ms/step |
| construction | 468 ms | 471 ms | 9,045 ms |

**Open, F:**
- the remaining gap is the all-h count in uAdvect, MGPCG and the level set, plus the over-seeded particles in pMove;
- the next items are E's census (TRIG), then the tiers, then the join bits and owner readers.

Note: at steps 30-60 F_ALL holds 4,907-4,913 bricks, the scratch pool's 4,913. No FAIL bit is raised, and a refusal
would latch FAIL_POOL_F, so nothing was refused. All-h simply makes almost the whole box resident once the splash
spreads, which is why the per-face kernels' item count is the box.
