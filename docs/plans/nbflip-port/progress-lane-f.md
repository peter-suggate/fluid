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

**Cause: the TRIG stand-in.** (Superseded: the census TRIG fixed D8c only; see "M2: the census TRIG and the step-1 oracle" below.) It marks every tile with a crossing cell, so the whole surface seeds at boot. The
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

## M2: the census TRIG and the step-1 oracle (2026-11-15)

**Landed:**
- 1bf829a0: `lHeatNow` takes its trigger from lane E's census (`NB_S_TRIG`, bit tile), replacing the crossing-cell
  stand-in. An absent slot's φ reads `NB_PHI_FAR`, as `brickOwnerWGSL` does.
- c553d1a6: pMove's thin trigger reads TRIG bit 8 + tile (`fgTileThin`, lane E's 5abc3cc4), replacing `false`.
  It moved nothing outside run-to-run spread over 60 steps.

**Re-gate, `dam-300b`** (5abc3cc4 + the thin patch, the authored B pool, `--poolF=4913`, check slot):
- rc 0, no FAIL bits, no Metal aborts. Report: `fluid-scratch/lane-f/gates/dam-300b.md`.
- **Overall: FAIL** on two rows. The census fixed D8c and D1.

| gate | dam-300a (stand-in) | dam-300b (census) |
|---|---|---|
| D1 front | FAIL 1.30 | pass 1.00 (step 8: 104 vs 108) |
| D5c collapse KE | FAIL 1.82 | **FAIL 1.88**, step 4 only (29.3 vs 36.1 J); steps 8–16 pass |
| D6 wet floor | FAIL 1.85 | **FAIL 1.78** (step 8: 9,809 vs 10,767; step 12: 13,743 vs 14,936) |
| D8c particles at 16 | FAIL 5.12 | pass 1.00 (386.7k vs 351.6k) |
| every window row | pass | pass (D8 0.67, D5p 0.40, D4 0.27) |

Boot particles: 31,183. The old run reports 0 at step 0 and 75,687 at step 1.

**Timing slot, 165 steps** (`fluid-scratch/lane-f/timing-thin/`, the same tree):
- fenced median **69.84 ms** (p95 84.3; thirds 81.2 / 70.0 / 60.9);
- pipelined **68.62 ms/step**;
- construction 500 ms.
Against M2's 63.81 / 62.48 ms (60309e55 plus the staging), this is about 6 ms a step slower. The differences are
E's census head, which runs every step and, at all-h, computes WANT without committing it, and the census's
particle population. I have not attributed the split; the next profile run will. The reference is 45.45 / 50.61 ms.

**The step-1 oracle.** The KE deficit is already present at step 1: ours is 1.887 J, the references' is 4.133 J. I
checked step 1 against a Float64 CPU oracle (`fluid-scratch/lane-f/oracle/dam-step1.mjs`, test oracle only). It is
one MAC projection of g·dt on the dam column (80 x 117.76 x 80 cells, free-slip box, ghost-fluid free surface
θ = φ_l/(φ_l − φ_a)), with KE measured as `boxAdd` measures it.
- **Oracle: 1.8868 J** (faces: 1.8918 J); the brick engine gives **1.887 J**, equal to four digits.
- A projection cannot add energy, so from rest the step-1 KE is at most 0.5·M·(g·dt)² = **2.453 J**.
  The references' 4.133 J exceeds that bound by 68%.
- The old engine therefore puts about 1.5 steps' worth of gravity into its first step. That is not a projection of
  g·dt. Our sqrt(KE) climbs at the references' rate (1.36 against 1.34 J^½ a step), but sits a constant 0.40–0.55
  steps behind them from step 1 to step 12.
- The front lags by the same 0.35–0.5 steps at steps 4–10.
- D5c step 4 is that offset. **Ruling needed:** D5c's early rows (and D1 and D6's early rows) are measured against a
  first step that exceeds the projection bound.

**D6 is partly real.** The wet-floor lag grows: 0.6 steps at step 4, 1.15 at 8, 2.05 at 12. The KE offset is
constant, so the extra lag is not the reference's first step.
- Our early volume loss is larger. Drift is −0.63% against −0.14% at step 8, and −2.3% against −0.6% at step 12.
- The loss is in the spreading sheet. At step 12 the x bins 10–15 hold 3.5 L less than the references, and the column
  holds 0.4 L more. The sheet's tops are also lower (bin 15: 1 against 8 cells).
- Volume control is idle there in both engines (inside the 1% dead band).
- So the sheet loses about 0.1% of the volume a step to the level set and particle transport at the thin foot.
- All-h is the main structural difference left: the census tiles every tile h, and E notes the particle population
  matches only with tiers. It is the next thing to rule out.

**Open, F (in order):**
- `tiered: true`. This means reading every F stage through `brickOwnerWGSL` (`nbVertexPhi`, `nbFaceU` and
  `nbFacePlane`, including the walls' velocities) and writing only owned storage.
- The join bits (TILE_OCC 8–15): the old `transferRefine` box (x ± 0.875, or the crossing cell when the home tile
  has REACH and depth ≤ 0, swept by dt·before/h plus the activity padding, then divided into tiles). They matter
  only once the census commits WANT, so they land with tiers.
- pDeposit's workgroup memory, and the D6 sheet loss if tiers don't close it.

## M2: the half kick, and D6 attributed (2026-11-15)

**Correction.** The references' step-1 surplus is not an artifact. The old host kicks U ← P(U + g·dt/2) before
its first frame (webgpu-uniform-reference.ts `kickMixedFrame`, uniform-mixed-frame.ts `kick`), so step 1 carries
1.5·g·dt; lane E found it. My bound argument stands for a single g·dt step, but that is not what the old engine
integrates.

**Landed, e6fcbec4: the kick, through E's `BrickStages.kick` (5600d180).**
- force, `uKick`: UP = U, then US = U = U + g·dt/2 on the faces uTransfer applies gravity to;
- then E's operator, solve and project;
- after: `uExtend`, then the initial seeding (from the kicked, extended US) and its rebin into bank 0.
- Step-1 KE is now 4.22 J against the references' 4.13 J.

**Re-gate, `dam-300c`** (HEAD 4054ad7e + the kick, the authored B pool, `--poolF=4913`, check slot):
- rc 0, no FAIL bits, no Metal aborts. Report: `fluid-scratch/lane-f/gates/dam-300c.md`. **Overall: FAIL.**

| gate | dam-300b | dam-300c (kick) |
|---|---|---|
| D1 front | pass 1.00 | pass 0.70 |
| D5c collapse KE | FAIL 1.88 | **pass 0.23** (step 4: 36.8 vs 36.1 J) |
| D6 wet floor | FAIL 1.78 | **FAIL 1.10** (step 8: 10,178 vs 10,767; step 12: 14,111 vs 14,936) |
| D8c particles at 16 | pass 1.00 | **FAIL 1.19** (393.4k vs 351.6k, tol 35.2k) |
| window rows | pass | pass (D8 0.54, D4 0.23, D5k 0.20) |

**D6 and the early volume loss: the all-h extension's reach.**
- With the kick, the remaining deficit sits only in the nose: the outermost 2–3 bins in x and in z (up to 0.09 L
  a bin at step 12). Everything behind it matches.
- The old h extension is the same two Jacobi sweeps as ours, so it reaches two cells from the liquid. But a face no
  h sweep reached takes the 4h band's extended value (`ufxFinishH` and `ufxProlong` in the old extension pair),
  which reaches far into the air. All-h has no 4h band, so our faces beyond two cells keep the unextended U (about 0).
  The level set and the particles ahead of a nose moving several cells a step then see no velocity.
- **Diagnostic** (`fluid-scratch/lane-f/tree-extx`, scratch only, not for landing). Two more extension passes after
  `uExtend`, each seeded from the last pass's extended faces, reach about six cells:

| step | wet floor: deeper / kick / ref | drift %: deeper / kick / ref | particles: deeper / ref |
|---|---|---|---|
| 8 | 11,171 / 10,199 / 10,759 | −0.11 / −0.75 / −0.14 | 164.7k / 148.3k |
| 12 | 14,920 / 14,088 / 14,936 | −0.36 / −2.54 / −0.61 | 260.8k / 261.8k |
| 16 | 16,155 / 15,917 / 16,145 | −2.88 / −5.73 / −5.00 | 319.2k / 345.4k |

  The deeper reach closes the drift, the wet floor and the particle count together, so D8c's surplus has the same
  cause. The variant is not the old formula, and it races across bricks.
- **The port:** the 4h band extension, prolonged into the h faces no sweep reached (`ufxProlong`), lands with the
  tiers. The interim alternative is a deeper all-h extension, which would be a stated deviation and needs a ruling.

## M2 tiered, T1 and T2: owner readers, owned writes, the join bits, the blended sampler (2026-11-15)

Ruling (coordinator): take the verbatim route and flip `tiered: true`. The scratch 6-cell extension above is
evidence only and is not landed. Plan T1–T6 approved: port the old 4h formulas verbatim, but run both tiers in the
existing passes, adding a kernel only where there is a real dependency barrier. Lane E takes T4, the 4h corner φ, in
its own file `levelset-coarse.wgsl.ts`.

**T1** (`fluid-grid.wgsl.ts`, the kernels):
- `fluidLayout(L, { tiered })` emits `FG_TIERED`, and `fluidStagesAt(tiered)` builds both stage sets
  (`fluidStages` is all h).
- The tier owners come from brickOwnerWGSL's own text, cut into parts:
  - `fo*` is the tile part;
  - φ: `foVertexPhi`, plus `fo1VertexPhi` for PHI1;
  - faces: `foU*`, `foP*` and `foS*`, one set per channel.
- Tiered, `fgPhiWord` reads PHI and PHI1 through the vertex authority, and `fgFace` reads `foFaceU`.
- Writes are gated on ownership:
  - `lSetPhi` writes only vertices F owns;
  - the velocity kernels write only faces with an h-owned cell on either side (`fgFaceOwned`);
  - `pDelta` gives NONE on a face F does not own until the 4h deltas land (T3).
- `pJoin` (tail, before `pDelta`) sets TILE_OCC bits 8–15 with the old transferRefine join rule. The census reads
  these bits for the band decision only, not for TRIG.
- Tiered viscosity fails explicitly, because the 4h Laplacian is not ported.

**T2** `FG_SAMPLE_WGSL`, `fgSampleVel(p, lo, hi)`:
- It is the old umUnitSample (the cached unit interpolant plus umUnitCoarse). The fine weight is:
  - 0 without support bit 0 (an occupied tile within 2 tiles);
  - 1 in an h tile;
  - otherwise the 26-neighbour blend.
- A tap is F's face where the face is F-owned, else `umCompiledVelocityTap`'s trilinear of the 4h cache.
- The 4h cache of a plane is the C face between two 4h tiles, else the mean of its 16 subfaces.
- It is used by the momentum trace, G2P, the particle trace, the seeds and the surface trace (with its wall probe),
  and is exported for E's 4h corner advect.
- Deviations:
  - a wall plane reads 0 (F's closed-face rule);
  - support comes from FT_DIST, so a tile with no F slot has weight 0 (every tile next to an h tile is in the
    census ring).

**All h, unchanged** (16 steps, `fluid-scratch/lane-f/runs/t1id-*`):
- Tiered off, every new branch is folded away.
- The engine is not bit-deterministic run to run. Two base runs already differ at step 0 by 1e-10 in KE.
- T1 stays within the base's own run-to-run spread:

| step | base KE / total L / particles | base rerun | T1 |
|---|---|---|---|
| 8 | 121.023 / 182.629 / 201,676 | 121.024 / 182.623 / 201,392 | 121.031 / 182.628 / 201,655 |
| 16 | 282.006 / 173.342 / 392,228 | 281.849 / 173.415 / 394,204 | 282.640 / 173.405 / 394,738 |

- All 46 kernels (both modes) pass naga validation. The only exception is naga's known false positive on
  loop-only returns, which Tint accepts.

## M2 tiered, T3 and T5: the 4h faces in the h passes, the 4h extension, tiered on the dam (2026-11-15)

**T3** (`velocity-kernels.wgsl.ts`): the old 4h formulas verbatim, run in the h passes over FC_G1 (F_ALL's bricks,
then C_ALL's pages, the split by `workgroupUniformLoad`; the page bodies hold no barrier).
- uAdvect: a C page's faces trace from the face centre (umUnitMomentumValue at width 4), culled where neither
  owner is live. The live test at 4h is vLive4: a corner below 0 or the centre below 0.
- uViscous: the 4h Laplacian (umLaplacian at width 4, spacing 4h) over the 4h cache of UP, into C US. An h tap that
  F does not own reads the 4h cache's trilinear of UP (the resolved h tap).
- uTransfer: the 4h gather (transferFace, umTransferFace's verbatim weights, depth window and blend with
  nbInterpolatedHeat), then forces. These are viscosity where O holds liquid, and gravity where either owner is
  occupied or has coarse mass (umCoarseMass: fill·64 > 0.05).
- uKick: the 4h half kick under the same gravity test.
- The 4h extension: uExt4Seed (ufxRestrict4) and uExt4 (the 12³ 4h window, (U, UP) swept jointly) on C_ALL, before
  uExtend. uExtend is ufxFinishH verbatim, with prolongation (ufxProlong) where the cell below is coarse.
- The 4h FLIP deltas: on a 4h tile's +plane, pDelta reads the C face's UP, or the seam subface F publishes.

**T5:** the outer seed shell seeds only h-owned cells. vGauge adds each 4h tile's fill·64.

**T4 wiring (lane E's kernels):** lcAdvect, lcCouple, lcClear, lcRetire and lcPublish run on C_ALL beside their h twins.

**Level set:** lUnion counts a tile with no F slot with heat 0 (lcCouple's rule).

**Dispatches against all-h:**
- 7 C_ALL dispatches are added: uExt4Seed, uExt4, and lc{Advect, Couple, Clear, Retire, Publish}.
- uAdvect, uViscous, uTransfer, uKick and vGauge moved to FC_G1 and cost no extra dispatch.

**Stated deviations:**
- A tile with no F slot counts as supported where it is occupied, and with heat 0.
- umFarAirOwner, the thaw term and the dust control are not ported.
- Seam h faces take the deposit path rather than the 4h gather; the two agree at the fixed point.
- Every C page is seeded, which is a superset of the old active set.
- lTiles and lDilate see no tile that no F slot holds. Beyond the census ring this changes only distances of 3 or
  more tiles.

**All h unchanged** (`runs/allh-pd`, 16 steps, composite MGPCG, HEAD 9c98ac9b plus this patch):

| step | KE J | total L | particles |
|---|---|---|---|
| 16 | 282.28 | 173.40 | 393,327 |

The T1 base, for comparison, was 282.01 / 173.34 / 392,228, so the run is within run-to-run spread.

**Tiered dam** (`runs/tiered-c16`, `tiered-pd`, `dam-300t`; composite MGPCG; the solver must be composite, since the
pool MGPCG alone does not project C faces):

| step | tiered L / KE | all-h L / KE |
|---|---|---|
| 8 | 182.50 / 119.6 | 182.62 / 121.0 |
| 13 | 180.37 / 248.7 | 178.36 / 250.7 |
| 16 | 175.97 / 272.0 | 173.40 / 282.3 |

**Open:**
- FAIL.REACH, seen once in two runs at step 14. A run-up jet climbs the z = 127 wall, going from 3.4 to 7.5 m/s in
  one step, which is about 20 cells. The reach is the stale bound plus 7 cells, about 16.5. This is lane E's,
  ported from the old reach rule.
- POOL_F at step 21 on the authored f = 3968. The F residency is about 7x the h-owned cells.
- The cost of the tiered sampler. In the profile at steps 11–13, uAdvect takes 176 ms and lAdvect 61 ms;
  pressureRowsC (lane P) takes 543 ms. fgSampleVel resolves every tap through the owners, and the 4h cache of a seam
  plane is a 16-face mean per read. The old engine reads precomputed caches (the unit and 4h velocity textures).

## M2 tiered: the opening caches (uCache), the sampler cost removed (2026-11-15)

The old engine samples through caches it builds once per step from the opening state: the unit velocity
texture, the 4h velocity texture and the vertex phi. The tiered sampler instead resolved every tap through the
owners, and read a seam plane's 4h cache as a 16-face mean on each read. `uCache` (FC_G1, one dispatch) builds
the same caches once per step, after the extension and before pMove and pSeed. It writes to fluid-grid's fcell,
which is sized only when tiered:
- C pages: CACHE4, the 4h cache of each face (`fsCacheRaw`: foPCFace, or the mean of the 16 F subfaces).
- F bricks:
  - PHIU, the phi each vertex opens with (lAdvect's back-trace reads it through `lPhi0`/`lCorners0`).
  - HMASK, 27 bits per tile saying which neighbour tiles are h-owned (the sampler's fine weight).
  - UNIT, the resolved unit tap (`fsTapH`), on near bricks only, where some neighbour tile is h-owned.
- The readers change as follows:
  - `fsCache` is a table read.
  - `fsFine` reads UNIT for all 8 taps.
  - `fsFineWeight` reads HMASK in place of 27 owner lookups.
  - lAdvect skips a vertex it does not own.

Profile, steps 11–13, composite MGPCG, `--poolF=4913` (`logs/prof-t2.log`), ms per step:

| kernel | uncached | cached |
|---|---|---|
| uAdvect | 176 | 7.2 |
| lAdvect | 61 | 1.3 |
| uCache | – | 1.5 |
| pressureRowsC (P) | 543 | 543 |

Physics is unchanged. At step 13 the cached run gives 248.43 J and 180.40 L, against 248.64 J and 180.36 L
uncached; the difference is within the run-to-run spread from summation order.

Dispatches against all-h: the 7 C_ALL dispatches, plus uCache (one FC_G1 dispatch).

## M2 tiered: lAdvect reads E's deep-air keep and released walls; the first tiered dam-300 (2026-11-15)

lAdvect now applies lane E's two verbatim ports from `levelset-release.wgsl.ts` (acd355ec) inside `lAdvected`:
- `nbrDeepAirKept`: the deep-air keep, where a still tile's φ is at least 9. It is skipped for a moving wall vertex, as in the old order.
- `nbrReleasedWalls`: the released-wall rule, from the projection's release bits. It runs after the wall-contact block and before the moved-smoothing term.

Short runs on acd355ec plus this change, composite MGPCG for tiered, `--poolF=4913`, rc 0, no FAIL:

| run | step 15 drift | step 24 drift | step 30 drift | particles at 30 |
|---|---|---|---|---|
| tiered (`rel-t`) | -2.48% | -7.08% | – | – |
| tiered before (`vmax-t`) | – | -7.86% | -7.39% | 877,810 |
| all-h (`rel-h`) | -4.60% | -8.15% | -7.65% | 763,646 |
| all-h before (`dam-300c`) | -4.61% | – | -7.51% | 775,733 |

The first tiered dam-300 (`gates/dam-300tf.md`, 9e5d2b14, real census plus the kick) passes every row except D8:
- D8 is the particle window mean over steps 30–90: 711k against 631k ± 63k, margin 1.27.
- All-h `dam-300c` gives 665k on the same row.
- D2 and D7t had failed in dam-300te. That failure came from the scratch sampler, not the engine: it read raw F φ at hanging vertices on h|4h planes. Resolving those vertices through vertex authority (`nbVertexCoarse`) removes it. At step 4, D2 is 114.65 against 114.75.

Max |U| and max |US| per step on liquid faces, against the old engine's `maxSpeed_m_s`, in m/s. Our column is owner-resolved U, with US in brackets:

| step | old | ours |
|---|---|---|
| 10 | 3.38 | 4.00 (2.95) |
| 13 | 7.25 | 12.78 |
| 14 | 13.87 | 12.04 |
| 20 | 14.88 | 18.18 (20.17) |
| 21 | 13.33 | 19.13 |
| 22 | 16.26 | 16.36 |
| 26 | 11.26 | 8.93 |
| 30 | 5.23 | 6.26 |

Over the first 30 steps the old engine peaks at 16.3 and ours at 19.1 (US 20.2), about 1.2 times higher. The maxima are run-up jets: at the far wall at step 13, and in the far corner at steps 20–21. The gap is not large enough to need a kernel attribution.

A dense=0 tiered run latches REACH at step 15: brick (4,5,15), info 0xf01404. Lane E has this as the known census timing gap, which the residency port of the head-time rule closes.

## M2 tiered: DEPB's particle support in pJoin, for E's head residency (2026-11-15)

pJoin now accumulates the old deposit's support words, `pbHalves` and `pbTravel`, into E's NB_S_PSUPPORT region (b895ea8e):
- It runs for every live record of bank dst: ballistic, deep and seed records included. The hook sits above pJoin's early return.
- Each record supports the 2^3 tiles from floor((x-2)/4).
- Per brick slot, word 0 takes the tile bit `tx | ty<<1 | tz<<2`.
- Words 1..3 and 4..6 take the -/+ travel maxima, as f32 bits with atomicMax. A zero travel is skipped.
- The travel is `lrParticleTravelConditional` with scale dt/h, plus NB_SEPARATION_REACH = 1.
- An absent brick inside the lattice latches FAIL.REACH.
- Atomics are issued only when the loaded word would change.
- Clearing the words is lane E's job: the census reads and zeroes them.

Test coverage:
- `particles-formulas.test.ts` checks that the travel text is the old `particleBrickDepositWGSL`'s, and that the reach is `narrowBandSeparationReach("pairs", true, 8)`.
- On the GPU, the scratch probe `--psupport` (brick-dam.mts) checks every step against a CPU oracle built from bank dst's records, zeroing the words after each check.
  - Dam, 24 steps, tiered composite and all-h: the masks are exact in every slot at every step.
  - Every travel word is at least the oracle's bound: |v|·dt/h for a ballistic record, the reach for a coupled one.
  - No word is set in a slot without a mask, no supported brick is absent, and FAIL stays 0.
  - On tiered at step 24 there are 662k records (7,017 ballistic) in 1,872 supported slots, and the largest travel is 34.13 cells, equal to the oracle's ballistic |v|·dt/h.

## Gate, timing, profile and D8 attribution at 5233e246 (2026-11-15)

**Gate.** Tiered dam-300, composite, with poolF 4913 (runs/dam-300th.jsonl, gates/dam-300th.md). The authored pool (f 3968) latches POOL_F at step 20.
- Every row passes except D8 for the 30–90 window: 7.108e5 against 6.383e5, tolerance 63.8k, margin 1.14.
- D8 for the 105–165 window passes, and so does D8c at step 16.
- Peaks: liveF 4811 (HIGH_F 4844), pages 125 and particles 1.01M. E recorded these in ead56e59.

**Timing.** Tiered composite, timing slot, 165 steps.
- Fenced median 134.6 ms (p95 199.8; by third 179 / 134 / 120); pipelined 135.4 ms/step.
- That is about 3x the old engine's 45.45 / 50.61 ms and about 7.5x design §6's 18 ms dam budget.

**Profile, steps 13–24, ms per step** (tiered 148 summed; all-h 107):

| area | ms | launches |
|---|---|---|
| grid velocity | 50.5 | 8 |
| pressure solve (P) | 31.7 | 181 |
| level set | 20.8 | 23 |
| pressure setup and projection (P) | 18.0 | 5 |
| layout, census and residency (E) | 16.4 | 38 |
| move | 6.0 | 5 |
| transfer and seeding | 4.7 | 2 |

**Kernels.** ns per workgroup, tiered against all-h:
- uAdvect: 4131 against 2927
- uExtend: 2642 against 702
- uTransfer: 2323 against 628
- lDistance: 1397 against 279
- uViscous: 855 against 130
- lTiles: 699 against 50
- uExt4Seed: 52.7 µs per page

The tiered per-brick cost is 2–13x all-h. The cause is owner resolution per sample, through foVertexPhi/nbVertexCoarse and the fgFaceOwned/foCellOwnerH lookups, which happens inside the inner loops.

pMove, pScatter, pDelta and pJoin are direct dispatches over the particle cap.

There are 262 dispatches per step against all-h's 244; 182 of them are pmg.

**D8 attribution.** Two differences from the old engine:

1. **The trigger rule (missing).** E's census TRIG sets heat for any criterion. The old activity uses nbActivityTriggered in "need" mode, where thin triggers only while deforming (dt·tuDeformation/h > 0.02), and it excludes dropped tiles.
   - From step 8 on we have more hot tiles. At step 30: T2 3785 against 3119 (+21%) and T1 5743 against 3193 (+80%).
   - That gives +19% cells, +24% particles and +30% mass.
   - Seeds and deletions per hot cell match.
   - E is porting the rule verbatim, with the update-error deformation certificate.
2. **Boot ordering.** The initial seed reads the boot census at U=0: 351 hot tiles against the old 1177, which the old reads after frame 1's activity on the kicked state.
   - Agreed with E: kick = {force, after, seed}, with a census that sets TRIG only between after and seed.

Both are physics rules, so both are ported verbatim, and D8 gets re-gated once E's hash lands.

## Item 1: owner resolution staged once per brick; record passes over the live count (2026-11-15)

Measured on tree-st7 (b9315489 plus these lane F files; the commit carries the same files onto main).

**Record passes.** pMove, pScatter, pDelta and pJoin now dispatch indirect over the live records. pArgs is a one-lane direct kernel (ARGS.LANE_F) placed at the tail after pSeed, and in the kick around the seed rebin. The dam's record workgroups dropped from 8952 a step (the cap) to about 3150.

**Staging.**
- The owner window (fgSlotWindowWGSL) also stages one bit per window tile: the tile is a 4h authority (foVertexCoarse's tile rule).
- A vertex whose candidate tiles lie in the window reads its authority from those bits (fgWinVertexPhi). This is foVertexPhi cut from the owners' text, with only the tile rule replaced. A vertex beyond the lattice keeps the owners' FAR.
- uViscous stages the 375 vpCache taps once, in a brick whose window holds a lattice tile that is not h-owned.
- Every staging call is followed by a barrier. fluid-staging.test.ts checks every lane F kernel for this, in both modes.
- Two windows measured slower and were reverted: pMove's (7.39 against 6.68 ms) and uExt4Seed's 6^3 page window (+3.2 ms).

**Frozen equivalence.** Dam step 20, poolF 4913, new kernel against b9315489's on the same state:
- Bitwise, tiered and all-h: every lane F kernel except the four below.
- pSeed and pScatter differ from old run against old run as well.
- pMove (fpart) and uViscous (vel) differ only in FP contraction:
  - with the window lookups counted against the directory, there are no mismatches;
  - uViscous with its staging disabled differs from the old kernel in the same 365894 words;
  - the all-h ±Inf words are taps of 3e38 (NB_DELTA_NONE). The frozen state is after the tail, so UP still holds uExtend's deltas, and old and new round 3e38 - 2·3e38 + 3e38 differently. The schedule rewrites UP in uAdvect before uViscous.
- uCache first showed a real defect: the fast path read F's vertex beyond the lattice, where the owners return FAR. That is fixed (fgWinCovers requires v <= N).
- frozen.mts skips pJoin, because the frozen state is after pDelta: `before` holds deltas, so the swept box spans the lattice.

**Profile, timing slot, steps 13–24, ms/step.** Tiered 96.2 against b931's 108.6; all-h 79.0 against 77.9.

| area | tiered | b931 tiered | launches |
|---|---|---|---|
| grid velocity | 38.3 | 48.8 | 7 |
| level set | 16.0 | 19.1 | 21 |
| layout, census and residency (E) | 12.7 | 12.7 | 35 |
| pressure solve (P) | 12.7 | 12.4 | 191 |
| pressure setup and projection (P) | 7.7 | 7.7 | 10 |
| move | 6.3 | 5.6 | 5 |
| transfer and seeding | 2.4 | 2.4 | 3 |

**ns/wg, tiered against all-h:**
- uAdvect: 3700 against 3072
- uExtend: 2371 against 696
- uTransfer: 1202 against 585
- uViscous: 462 against 144
- pDeposit: 365 against 235
- lDistance: 1470 against 357
- lTiles: 197 against 51
- lClear: 318 against 194
- lAdvect: 814 against 1165
- uCache: 507 (tiered only)
- uExt4Seed: 57.1 µs a page

Tiered is at or below all-h only in lAdvect. In all-h the window and rings cost about 1 ms (uAdvect 12.08 against 11.53, lDistance 1.40 against 1.13).

**Timing, timing slot, 165 steps.**
- Fenced median: 87.5 against b931's 102.4 ms.
- Pipelined: 88.5 against 106.1 ms/step.
- Both st7 runs latched FAIL.REACH, at steps 162 and 165, on bricks (0,0,14) and (1,1,15). st3's fenced run latched at step 74 on (1,0,15). b931's two runs did not.
- The 300-step gates of st3 and st7, on the check slot, did not latch.
- Each latch is at the z = 127 wall, which is where lane E's reach gap was seen before. It is still open whether lane F's changes make it more frequent.

**D8, the dam-300 gate on st7.**
- Every row passes except D8.
- Against the default references (now dam-ref-a..e): margin 1.25, 7.095e5 against 6.309e5.
- Against a + b only: margin 1.12, against 6.383e5.
- D8c passes, at 0.46.

## Items 2-3 (first round): traced-work compaction and the per-step liquid cache (2026-11-16)

Measured on tree-s6 (0be9d9f4 plus exactly the committed lane F files) against tree-h1 (0be9d9f4). The commit lands on 258aae94 (lane E residency (b)), which touches none of these files. Every change is
bitwise on the dam's frozen step 20, tiered and all-h (frozen.mts: uAdvect, uTransfer, lDistance, uExt4Seed, uExt4
and uExtend all `same`).

**What changed.**
- **uAdvect (S2p's compaction).** A lane marks its cells' faces: a culled face (closed, or umCullAir's two dead
  cells) takes 0 at once, and a traced face goes to a workgroup list (`l | a << 9`). All 256 lanes then trace from the
  list, so whole SIMD groups run the RK trace. A C page does the same. The C page branch is now workgroup-uniform (the
  C count by workgroupUniformLoad), so a page may barrier.
- **Lanes a brick, frozen ms (tiered / all-h):** 64 lanes 21.8 / 17.5, 128 13.5 / 10.3, 256 11.7 / 10.2, 512
  12.3 / 11.0. 256 is kept.
- **Not done: the separate slab kernel.**
  - Seep's own table puts per-brick compaction (633-655 µs) within 3% of the 64-cell slabs (618-636).
  - Our sampler also reads the staged owner window. Slabs would stage it eight times a brick, or go back to the
    directory.
- **lDistance.** The (vertex, nearest crossing cell) pairs of each half brick go to a workgroup list (`lid | k << 8`,
  k the first of the vertex's 8 cells to name the crossing cell). The lanes run nbfCellDistance from the list into
  the vertex's minimum: atomicMin on the bits, since a distance is a length, at least +0, and the bits order as the
  values.
- **Pairs on the frozen state:**
  - All-h: 1.44M pairs. The tiered count was not probed separately.
  - Before the change the tiered kernel took 5.58 ms against all-h's 1.58: 3.5x for a similar band. The cost was
    the serial per-lane loop of owner-resolved corner reads.
  - A flat-corner probe took the tiered kernel from 5.0 to 2.3 ms. That probe reads F directly; it gives different
    results and is timing only.
- **FC_LIQ, a new fcell channel.**
  - Contents: each cell's liquid phase this step (fgLiquidCell), 16 words a slot.
  - Writer: uTransfer writes it from its staged centres. uKick writes it at boot.
  - Readers: vLiquidOwner, used by uExt4Seed, uExt4 and uExtend in place of the eight-corner recomputation. An h
    cell reads its bit, a 4h cell its tile's phase, and a cell no slot holds computes its corners.
  - No φ writer runs between the writer and the readers in either schedule. The new fluid-staging.test.ts case checks
    this.
  - uExtend drops its 14^3 vertex ring, which only fed the liquid bits.
- **uExt4Seed.**
  - The knockout showed the cost: removing the liquid tests took the kernel from 6.2 ms to 0.16 ms. The kernel ran
    19,077 faces with an h owner, and each made 32 liquid tests, 4.9M directory vertex resolutions in all.
  - Change: those faces are now listed and spread over 16 lanes a face, one subface a lane. The face's lane sums in
    the subface order.
  - With FC_LIQ the liquid tests are a bit read.

**Register limits (mstat: Dawn MSL, `max_total_threads` stripped, Metal maxTotalThreadsPerThreadgroup), before → after.**

| kernel | tiered | all-h |
|---|---|---|
| uAdvect | 576 → 576 | 640 → 704 |
| uExtend | 640 → 640 | 1024 → 1024 |
| lDistance | 768 → 832 | 832 → 1024 |
| uExt4Seed | 576 → 576 | - |

The trace holds the registers, at 576 tiered: the blended sampler. Compaction does not lower them. Seep's split
trace kernel sat at 704-832.

**Frozen step 20 (timing slot, min of 5, ms), h1 → s6.**

| kernel | tiered | all-h |
|---|---|---|
| uAdvect | 16.1 → 12.2 | 12.3 → 10.9 |
| uExtend | 10.2 → 6.6 | 3.17 → 2.64 |
| uExt4Seed | 4.99 → 0.48 | - |
| uExt4 | 1.03 → 0.51 | - |
| lDistance | 5.58 → 3.59 | 1.58 → 1.40 |

**Profile, timing slot, steps 13-24.**
- Tiered: 93.7 → 79.1 ms/step. Grid velocity 38.8 → 26.1, level set 15.9 → 13.9.
- All-h: 73.9 → 72.5.

**Per brick (ns/wg, timing slot profile) against seep-m1max.**
- **Seep's figures, from its own logs (progress-lane-s.md, step 1100, about 2600 active bricks):**
  - Velocity plus trace: 618 µs (S2p kbench), 0.24 µs a brick. The 0.25 estimate holds.
  - Extend: 247 + 81 µs before S2o's sweep skip, at most 0.13 µs a brick.
  - Redistance: init 92 + sweeps 223 + 140 µs after S2o, 0.175 µs a brick.
- **Workloads at the measured step** (traced faces and segments counted by probe):
  - Seep at 1100: 503k traced faces, 513k segments (about 1 a face).
  - Our dam at step 20, all-h: 2.06M faces, 7.61M segments (3.7 a face).
  - Our dam at step 20, tiered: 0.53M faces, 3.22M segments (6.05 a face).
  - So our dam does 4-10x seep's sampling work per brick.

| kernel (seep's equivalent) | tiered h1 | tiered s6 | all-h h1 | all-h s6 | seep a brick |
|---|---|---|---|---|---|
| uAdvect (+ uTransfer, pDeposit: velocity + trace) | 3822 (+1222+362) | 2811 (+1198+357) | 3105 (+587+232) | 2793 (+590+229) | 240 |
| uExtend (+ uExt4Seed, uExt4 over the F bricks: extend) | 2363 (+1860) | 1522 (+195) | 685 | 570 | ≤130 |
| lDistance (redistance) | 1469 | 912 | 360 | 312 | 175 |

- **Per traced sample:**
  - uAdvect tiered: 1.49 ns (6.98M momentum samples on the frozen state).
  - uAdvect all-h: 0.64 ns (17.27M samples, 2.27 a segment).
  - Seep velocity plus trace: about 0.40 ns. This is an upper bound: 618 µs, its pass 1 included, over about 1.53M
    samples. The sample count assumes three samples a segment; seep's log gives only the 513k segments.
  - So all-h is 1.6x seep per sample, and tiered is 3.7x.
- **Remaining gap:** the tiered sampler (fsFineWeight, 24 fine taps, 24 coarse taps) costs 2.3x the all-h sampler
  per sample.
- **Target not met:** tiered within 1.5x of seep per brick. The per-brick gap is mostly the dam's workload, and
  second the tiered sampler.

**Timing, timing slot, 165 steps, tiered.**
- Fenced median: s6 70.7 ms, h1 85.3 ms.
- Pipelined: s6 72.8 ms/step, h1 87.3 ms/step.
- No run latched a failure.

**D8, the dam-300 gate on s6 (check slot), against the default refs a-e.**
- Every row passes except D8, which fails at margin 1.34 (7.152e5 against 6.309e5 in 30-90). D8c passes, at 0.45.
- The frozen checks are bitwise, so the margin change from st7's 1.25 comes from HEAD (lane E's region heat) or
  run-to-run spread.
- **Per sample, against the refs' mean (st7 run):** the excess starts at step 7 (1.22x), where the refs' particles
  rise slower. It holds at 1.1-1.2x afterwards.
- The references record only particles, so the T1/T2 tiles, seeds and deletions per step need an instrumented
  old-engine run.

## Read-by for uAdvect, segments per face, and the D8 accounting (2026-11-16)

All runs were on the check slot. The tree was tree-s6 (0be9d9f4 plus the 5f2717bc files); tree-d8 is tree-s6 with counters in header words 74-79.

**0. Does anything read uAdvect's departure where P2G sets the face?** Yes, on the dam, so no trace can be skipped bitwise.
- **The combine** is uTransfer's nbfP2GResolve, a verbatim port of transferPropertiesJob:
  - `mix(original, momentum/mass, transition)`, with transition = θ · mix(min(1, mass/expected), 1, hot).
  - Outside the P2G window, near = (0, 0, 1), so transition is 0 and the face keeps its departure.
- **At transition 1 the departure is still read twice:**
  - uViscous runs before uTransfer and takes the 7-point Laplacian of the departure UP for every liquid face (nbfViscousLaplacian), the face itself and its six neighbours. The dam is viscous: uViscous is scheduled only when viscosity_Pa_s > 0, and it runs in the profile.
  - Metal evaluates `mix(x,y,a)` as `x+(y-x)*a`, so even at a = 1 the result depends on x by an ulp.
- **Counts on the frozen step 20.** These ran uTransfer after a pDeposit of the frozen particles (frozen.mts `--pre-deposit`); the particles are at step end, so the counts are approximate.

  | | tiered | all-h |
  |---|---|---|
  | traced faces | 534k | 2.06M |
  | particle-weighted h faces (mass ≥ 1e-5) | 454k | 352k |
  | h faces at transition 1 | 114k | 84k |

  - Tiered, no C face carries weight: every C face lies outside the depth window -2..1.5, so every C trace is read.
- **The most an inviscid scene could skip** is the transition-1 faces, 21% of the tiered traces. That needs three things:
  - pDeposit and the heat moved ahead of uAdvect;
  - `select` in place of `mix` at t = 1, a sub-ulp change to a verbatim formula;
  - the viscous read gone, which holds only when the scene is inviscid.

  This needs a ruling, so it is not done.

**1. Segments per face.**
- **The multi-segment trace is verbatim.** The rule is the old engine's `momentumTrace: "substep"`: uniform-cached-momentum.wgsl.ts l.111-133. It is the method default (uniform-narrow-band-method.ts l.94), and every canonical reference's meta records "substep".
- **Histogram on the frozen step 20 (faces):**

  | segments | 1 | 2 | 3 | 4 | 5-6 | 7-8 | 9-16 | >16 |
  |---|---|---|---|---|---|---|---|---|
  | tiered (534k) | 0.5% | 0.9% | 4.6% | 17.7% | 37.7% | 29.2% | 9.4% | 0 |
  | all-h (2.06M) | 3.1% | 17.7% | 32.9% | 23.2% | 16.4% | 4.9% | 1.7% | 0 |

- **It is the whole column, not a few jets.** One segment covers at most one cell, 0.375 m/s at h = 6.25 mm and dt = 1/60. So 5-8 segments is the falling column at 1.9-3 m/s; g·t is 3.3 m/s at step 20. Tiered faces are the h band, which skews fast.
- Seep's flow is slow (about 1 segment a face).
- The rule stays.

**D8: what the references count.** The old engine was run once for 30 steps through a scratch copy of nbflip-gate-bench, with its diagnostics-only `narrowBandRetireCounting` on. That flag adds atomics only. Each sample also read the FLIP state words 0-43 and a direct census of the live bank (readParticleMassCensus). Our engine ran 30 steps on tree-d8 with `--pstat`.
- **The reference row is the old receipt `state[1]`.**
  - That is `stage.count`, the sorted slabs' live records.
  - Our run reproduces the references to 0.5%: step 10 reads 194,907 against 195,195, and step 30 reads 789,922 against 794,055.
  - It leaves out live records outside the sorted slabs (the spill tail and the frame's seeds; seed room counts `state[1] + state[SEEDS_WORD]`).
  - The old engine's own census of live records at the same fence runs 4-25% higher: step 7 reads 111,971 against 139,561, step 10 194,907 against 217,884, and step 30 789,922 against 829,355.
  - Step 0 reads 0 in all five references, before the first receipt.
- **Our `particles` is PCOUNT[bank].**
  - pScan's filed records plus pSeed's appended seeds.
  - `--pstat` finds 0 dead records at every step and no spill reserve or tails.
- **The measurement difference is on the reference side.**

| step | ours / refs (receipt) | ours / old census, records | ours / old census, mass |
|---|---|---|---|
| 5 | 1.035 | 0.916 | 0.923 |
| 7 | 1.220 | 0.976 | 0.991 |
| 9 | 1.232 | 1.120 | 1.167 |
| 10 | 1.250 | 1.118 | 1.146 |
| 13 | 1.129 | 1.051 | 1.089 |
| 16 (D8c) | 1.045 | 0.953 | 0.977 |
| 20 | - | 1.045 | 1.068 |
| 25 | - | 1.060 | 1.090 |
| 30 | 1.114 | 1.068 | 1.086 |

- **What remains like for like.**
  - **Step 9 is a seed burst.** We seed 53.6k there against our usual 20-35k, and the record ratio steps from 0.94 to 1.12.
  - **Steps 14-17 run below the old engine** (0.95).
  - **From step 18 the ratio climbs to 1.07-1.08 (mass 1.09), and we remove less.** Cumulative to step 30, ours against the old engine:
    - deep retirement: 131k against 153k;
    - cell-mass cut and merge: 1.04M against its crowded retirements of 1.13M;
    - splits: 1.09M against 1.13M.
  - **Heat expiry:** we retire 1,304 by heat; the old engine's heat word reads 0.
  - **No seed comparison yet.** The old engine has no seed counter in these words, and a seed count derived by accounting does not close, because the merge and crowded words overlap.
- **Proposed next run:** the old-engine census over the D8 window (90 steps, one check run). It would put the corrected row (ours against the old census) beside the receipt row. The ratio here (receipt/census 1.05 at step 30) suggests that D8's 1.12-1.22 against the receipt becomes about 1.07-1.16.
- **Tools:** ~/code/me/fluid-scratch/lane-f/tools/{d8-old.mts, rb.sh, rb2.sh} and frozen.mts `--pre-deposit`.
- **Results:** d8old/{d8-old-30,d8-new-30}.jsonl and logs/rb*-{t,h}.log.

**D8 over the gated window: the old engine's census, 90 steps (2026-11-16).** One check-slot run of the old engine (d8-old.mts, `narrowBandRetireCounting` on), read at the gate's sample steps. Ours is round-s6/dam-300ti.jsonl, the gated tiered run, unchanged. The references are ref-2761350a/dam-ref-{a..e}, the canonical five, as the referee ruled.
- **Receipt row (D8 as gated, unchanged):** ours against the references' mean receipt count.
- **Corrected row:** ours against the old engine's census of its live records (readParticleMassCensus) at the same fence. That census counts the records the receipt leaves out (the spill tail and the frame's seeds).

| step | refs mean (receipt) | old receipt | old census | old census / receipt | ours | **receipt row** ours / refs | **corrected row** ours / old census |
|---|---|---|---|---|---|---|---|
| 16 | 351,574 | 350,224 | 388,580 | 1.110 | 367,319 | 1.045 | 0.945 |
| 30 | 794,055 | 805,468 | 848,452 | 1.053 | 884,787 | 1.114 | 1.043 |
| 45 | 952,090 | 979,559 | 1,007,699 | 1.029 | 1,027,665 | 1.079 | 1.020 |
| 60 | 585,115 | 583,585 | 615,839 | 1.055 | 664,302 | 1.135 | 1.079 |
| 75 | 414,090 | 432,914 | 454,727 | 1.050 | 505,880 | 1.222 | 1.112 |
| 90 | 408,960 | 422,060 | 436,372 | 1.034 | 493,162 | 1.206 | 1.130 |
| **30-90 mean** | 630,862 | | 672,618 | | 715,159 | **1.134** | **1.063** |

- **This old run counts 2.2% more than the references** on its receipt over steps 30-90 (0.997-1.046 a sample). So its census is a little high for the five.
- **The census runs 3-11% above the receipt at every sample.** So the receipt row overstates our excess by that much. The measurement difference is on the reference side; ours counts every live record.
- **Corrected, D8 is 1.063 over the window** against this run's census, 0.63 of the 10% tolerance.
  - Scaling each reference by this run's census/receipt factor at the step instead gives 1.087 (1.058, 1.049, 1.076, 1.163, 1.166 a sample).
  - Both rest on one old-engine run.
- **It is not yet like for like at the end.** Steps 75-90 read 1.11-1.13 corrected. From step 60 we keep more records as the pool settles. That matches the 30-step accounting: we retire less by deep retirement, by cut and merge, and by splits.
- **The step-9 seed burst** is not settled here; it is below.
- **Both rows are printed; D8 stays gated on the receipt row** until the referee rules on the correction.
- **Results:** ~/code/me/fluid-scratch/lane-f/d8old/d8-old-90.jsonl (with -summary.json, run90.log).

**The step-9 seed burst: where the seeds go (2026-11-16).** One 12-step check run of the tiered dam with pSeed's seeds counted by cause (tree-d9, header words 80-118, d8old/d9-12.jsonl). Seeds a step:

| step | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| seeds | 9.6k | 7.7k | 32.5k | 15.5k | **53.7k** | 21.4k | 21.9k | 18.3k |
| in bricks with no binned particle | 2.2k | 3.8k | 21.4k | 8.1k | **37.6k** | 5.6k | 6.2k | 4.0k |
| uncovered (no particle in the 27 cells) | 6.8k | 5.1k | 28.0k | 11.0k | **49.3k** | 12.3k | 12.6k | 9.4k |

- **At step 9:**
  - 53.6k of the 53.7k seeds are in h cells.
  - 51.0k are in cells with n0 = 0.
  - 49.4k are in tiles at crossing distance 0.
  - 37.6k are in resident bricks with no binned particle.
  - None are in bricks entering this step (F_ENTER: 0 at every step past boot).
- **Where they are:**
  - By height: y 0-15 (the floor) has 30.8k, and y 96-111 (the column top) has 19.6k.
  - By max(x, z): 96-111 (the surge front) has 24.1k, and 32-47 has 11.7k.
  - Step 7's burst is the column top: 26.3k at y 96-111.
- **What this means:**
  - The burst is reseeding of liquid φ (after lAdvect and lUnion) at the surface, in cells and whole resident bricks with no particle. It happens at the surge's front on the floor and at the falling column's top.
  - It is not the residency fill. No brick entering that step takes a seed.
  - pSeed's site rule is the old engine's (nbfSeedSiteOk, verbatim), and so is the stage order: the move and rebin in part 1, then φ, then pSeed on this step's bins.
- **Not yet like for like.** The "20-35k" it was set against is the old engine's net growth a step, which is seeds less removals. The old engine has no seed counter (see the D8 accounting above).
- **The comparison that would settle it:** the old engine's own per-step seed count at steps 7-10 (a counter in its seed kernel, one 12-step run). If the old engine bursts there too, it is physics. If not, compare φ against the particles at the front, which means ours leads.
- **Results:** ~/code/me/fluid-scratch/lane-f/d8old/d9-12.jsonl (counters in `extra.lane`, word 64 + i, cumulative).

**lDilate past the owners' margin: lDilateC (2026-11-16).** The coordinator's M2 item: lDilate resolves a tile no F slot holds through its owner's C page, so the owner margin can drop from 20 to 9.
- **The kernel.** lDilateC (tiered only, one workgroup per C_ALL page, before each round's x pass) writes DIL4 (fcell, 512 words per C slot) for each of the page's absent tiles:
  - its source flags from its eight C corners (bits 24-31);
  - lDilate's DA (bits 0-11) and DB (bits 12-23), from sources staged over 18 x 18 x 8 tiles.
- **How lDilate reads it.** lDilate reads an absent tile's sources, DA or DB from DIL4 in place of skipping it. A tile in no resident page, or outside the lattice, is still no source.
- **Why the corner rule is exact.**
  - In a 4h tile every h vertex is the corners' interpolant (foVertexPhi), so lTiles' and lClear's cell flags follow the corners' signs.
  - levelset-kernels.test.ts checks this over 6,010 cases, in f32 with the interpolant's own order.
  - lClear's clear and lcClear raise only positive values, so no corner changes side.
  - The one exception is SURFACE_TILE (sample support: particles bin only into F slots). An absent tile has none.
- **Gates (frozen tiered dam, the analysis sequence lTiles..lDistance as one submit, every rw buffer compared):**
  - **Against the old code (tree-d0), margin 20:**
    - step 60 (old60) is bitwise;
    - step 20 matched bitwise once (old20). On a later frozen state it differed in 4 words (time20) and then 56 (oldL20). Every one is the air-seed field of DIST at a tile 8 or more tiles from any h tile (dh8 or dh99).
    - These are tiles whose seed sources lie in absent tiles (deep 4h liquid). The old code dropped them; now they count, as an F brick there would count them. set and phi are bitwise in every run.
  - **Pseudo-absence (the new code against itself):**
    - In B, every in-lattice F tile with no h tile within r is read as absent.
    - Its source flags come from the corner rule plus its own SURFACE_TILE seed (keepL).
    - B equals A bitwise apart from DIL4 at r = 2 and r = 1, at steps 20 and 60. That is 4 runs: keepL2-20, keepL2-60, keepL1-20, keepL1-60.
    - Without the SURFACE_TILE seed kept (nokeepL2-60), only the air-seed field differs. That is the sample support.
- **A fix on the way.**
  - dSource first returned no source for any tile outside the lattice, ahead of the F check. A brick on a lattice face holds tiles past it, and lDilate reads those through F.
  - It now reads the F flags first. The first pseudo probe also took those tiles as absent; the gates above use the corrected probe.
- **Timing (timing slot, exclusive, frozen step 20, margin 20, `--time=21`):**
  - the analysis sequence takes 7.15 ms against 7.04 ms (+0.11 ms);
  - lDilateC is 0.135 ms for its 2 runs;
  - lDilate's 6 runs take 0.359 ms against 0.377 ms.
  - At margin 9, F_ALL shrinks and lDilate with it.
- **For lane E, at margin 9:**
  - A 4h F tile carries sample support (SURFACE_TILE from the binned particles). So bricks holding particles, and their deposit support, stay resident.
  - lcCouple and lcBuildDistance read fgReach1 at 4h F tiles.
  - FAIL.TIER 0x4c440001 latches if an absent tile is h-owned.
- **Files:**
  - lib/methods/narrow-band/levelset-kernels.wgsl.ts (lDilateC, lDilate);
  - fluid-grid.wgsl.ts (DIL4);
  - brick-fluid-stages.ts (the placements);
  - levelset-kernels.test.ts (the corner rule, DIL4's layout and the schedule order).
- **Tools:** ~/code/me/fluid-scratch/lane-f/tools/{frozen.mts --seq/--seqsub/--inspect/--time, dil-seq.sh, probe-dil*.sh, dil-time.sh}. Logs are in logs/dil/.

## Item 2, the tiered sampler: attribution, and fcell read-only (2026-11-16)

**Where uAdvect's tiered sampler spends its time.** Frozen tiered dam, step 20, timing slot, `--time=6`. The variants keep the trajectory: each computes one component a second time at p + 0.001 and keeps it only on an impossible value, so the added time is that component's marginal cost in place.
- **Samples (counting variant):**
  - 6.97M in all, 19.86M axis components (2.85 a sample);
  - fine weight 1: 6.21M (89%);
  - fine weight 0: 0.27M (3.8%);
  - blended: 0.50M (7.2%).
- **Marginal cost against 12.4-13.0 ms for the kernel:**
  - fsFineWeight +1.4 ms (0.2 ns a sample);
  - fsFine +8.6 ms (about 0.4 ns for each axis component on the 7.47M samples with fine > 0);
  - fsCoarse +1.6 ms (on the 0.77M samples with coarse > 0).
- **So the fine interpolant is the cost.** Its 8 taps are UNIT reads (fsUnit: fgFaceOpen, the slot through the window or the directory, then the load), and the weight and the coarse path are secondary.
- **A one-brick path for fsFine is a loss.** All 8 taps in one F slot means one slot lookup and direct UNIT loads. It was slower (12.62 against 11.64 ms) and not bitwise (167k vel words; probably the compiler's contraction of the reshaped lerp), so it is dropped. Slot lookups are not the limiter.

**fcell read-only where a kernel writes none of it (bitwise).**
- **The change.** fluidKernel now binds fcell as read-only `array<u32>`, read with plain loads (FG_FCELL_RO_WGSL), in place of `atomicLoad` on `array<atomic<u32>>`. It does this for every kernel whose code has no fcStore and no `&fcell[` atomic.
- **Which kernels.**
  - Tiered: pScatter, lAdvect, lcAdvect, lUnion, lcCouple, lcClear, lcRetire, lPublish, lcPublish, pSeed, uAdvect, uExt4Seed, uExt4, uExtend, pJoin and pDelta.
  - All h: the same less the C kernels.
- **Gates.**
  - Frozen step 20, every placement, tiered and all h: bitwise. The only differences are pMove, pSeed and pScatter, which also differ old against old (atomic order).
  - Test: levelset-kernels.test.ts checks every kernel's binding against its code.
- **Timing (timing slot, ms, new against old, two runs):**

| kernel | run 1 | run 2 |
|---|---|---|
| uAdvect | 11.59 / 12.56 | 11.64 / 12.54 |
| lAdvect | 3.59 / 3.40 | 3.33 / 3.52 |
| lUnion | 1.69 / 1.21 | 1.17 / 1.14 |
| uExt4Seed | 0.64 / 0.79 | |
| uExt4 | 0.33 / 0.37 | |
| uExtend | 6.38 / 6.28 | |
| pSeed | 1.81 / 1.90 | |
| lPublish | 0.49 / 0.60 | |
| lcCouple | 0.38 / 0.46 | |
| lcAdvect | 1.48 / 1.53 | |

- **Reading the table.** uAdvect is down 0.9 ms (7%) in both runs. lAdvect and lUnion moved within run-to-run spread.
- **Next for the sampler:** a GPU profile of uAdvect, tiered against all h (limiter, occupancy), before any further restructuring of fsFine.
- **Tools:** ~/code/me/fluid-scratch/lane-f/tools/{samp-attr.sh, ro-gate.sh, samp-v1.sh}, with frozen.mts `--setwords` in per-kernel mode. Logs are in logs/samp/.

**Item 2, the sampler's rare paths: two deferrals, both no gain (not landed).**
- **Attribution (timing slot, uAdvect alone, frozen tiered dam, step 20; HEAD 11.6-11.9 ms).** These variants change values, so they are timing probes only:
  - t0 (fsUnit's no-slot fallback returns 0): 10.47 ms;
  - w1 (no coarse path): 9.60 ms;
  - w1t0 (both): 7.56 ms.
  - The fallback is never taken on this state (header counter 85 = 0 over 20 steps). The coarse path is taken on 6.7M samples.
- **Register limits (uncapped Metal maxTotalThreadsPerThreadgroup):** HEAD 576, t0 640, w1 640, w1t0 704, all h 704.
- **The lean split (bitwise, slower).** The main uAdvect ran the fine-only sampler and marked faces that needed more; uAdvectRetry traced them with the full sampler.
  - 19% of faces were retried.
  - uAdvect alone 9.23 ms + retry 3.78 ms = 13.1 ms in sequence, against 11.6 ms.
- **The fallback-only hybrid (bitwise on the real state, slower).** uAdvect kept the blended sampler, but a tap with no F slot set a flag (fsUnitAbsent) and the face was marked for uAdvectRetry.
  - Register limit 640, the same as t0.
  - Gates: step 20 and step 60 bitwise in sequence; all h bitwise.
  - Time: uAdvect alone 12.13 ms with per-sample flag checks, and 11.89 ms without them (the failed trace runs on); retry 0.15-0.16 ms (empty).
  - **t0's gain is not occupancy.** At the same 640 limit, t0 is 10.47 ms and the hybrid 11.89 ms. The win came from the compiler simplifying fsUnit once its fallback branch has no side effect. Any flag or latch in that branch loses it, so a residency-invariant FAIL would lose it too.
  - **An open fault, moot since the design is dropped.** When every fifth face was forced to retry, about 39.5k vel words differed. The retry path was never exercised on the real state.
  - Patch: ~/code/me/fluid-scratch/lane-f/patches/hybrid-retry.patch. Tools: tools/{samp-reg.sh, samp-occ.sh, lean-gate.sh, hybrid-gate.sh, hybrid2.sh}. Logs are in logs/samp/.
- **Reading.** The maxThreads limit is a poor proxy here: at 256 lanes a workgroup, both 576 and 640 admit one workgroup each. The rare paths cost through code size and scheduling, not a thread cap. Deferring them to a second kernel pays a second staging and a second pass, which is more than they cost.
- **The remaining lever is the coarse blend (w1, -2 ms).** It is taken on real samples, so it cannot be dropped bitwise. Making it cheaper would mean a different representation (e.g. CACHE4 folded into UNIT for the fringe bricks), which is a design question for the coordinator.

## 2026-11-16: the sampler, by the coordinator's item-2 rulings (t0 landed, w2 landed, staged tiles measured)

**t0: uAdvect's fsUnit drops the 4h fallback (c2f9bffa, bitwise).**
- **The invariant.** Only fsFine reads fsUnit, and only where the fine weight is over 0. That needs p's tile to have an F slot and an h owner of the opening tiers (HMASK, uCache's: the tiers the head's setMark saw) within 2 cells. So every tap lies within 3 cells of an h owner.
  - Lane E's read table has this row ("the sampler's UNIT taps", 3), under the owners' tiered margin of flow + 9.
  - Trace travel does not enter: the condition is local to each sample point.
- **Gate-only count** (tree-t0c, header 74-78, cumulative): 0 fallback hits in each of uAdvect, lAdvect, lcAdvect, pMove and pSeed, over the tiered 300 and the all-h 200 dam (composite, no latch, rc 0).
- **Frozen step 20, every placement, tiered and all h:** bitwise. pMove, pSeed and pScatter differ only as they do old against old.
- **Timing slot.** Dropping the fallback in every kernel: uAdvect 11.22 -> 9.91 and 11.57 -> 10.37 ms, but lAdvect 3.30 -> 3.84 and 3.48 -> 4.04 ms (two runs). So it is per kernel:
  - fsUnit calls fsUnitAbsent. By default that is fsTapH; uAdvect defines it as 0.
  - uAdvect 11.78 -> 10.15 ms; lAdvect 3.44 -> 3.38, lcAdvect 1.92 -> 1.84, pSeed 1.83 -> 1.82.
  - Bitwise: uAdvect, lAdvect and lcAdvect are the same.

**w2: one interpolant a sample (the fine one where the weight is over 0, else the coarse one), not the blend.**
- **Why it is close.** In the fringe, the fine taps away from the h owners are fsTapH, the 4h cache's trilinear at the face. Trilinear interpolation reproduces a trilinear field, so there the fine interpolant equals the coarse one except across 4h cell kinks.
- **Timing slot, against t0:**
  - uAdvect 10.31 -> 8.79 ms;
  - lAdvect 3.69 -> 3.72, lcAdvect 1.89 -> 1.85, pMove 6.17 -> 6.04, pSeed 1.65 -> 1.66.
- **One check run, dam-300 (tiered, composite, poolF 4913).** The baseline is tree-t0k, whose values are bitwise HEAD's. Worst margin a gate, base / w2:

| gate | artifacts/nbflip-port/ref (superseded 33b78ebb set, not a gate) | ref-2761350a (canonical, DEFAULT_REF) |
|---|---|---|
| D1 | 0.50 / 0.50 | 0.55 / 0.55 |
| D2 | 0.07 / 0.07 | 0.06 / 0.06 |
| D3 | 0.38 / 0.27 | 0.50 / 0.45 |
| D4 | 0.06 / 0.03 | 0.07 / 0.04 |
| D4c | 0.00 / 0.00 | 0.00 / 0.00 |
| D5c | 0.17 / 0.17 | 0.17 / 0.17 |
| D5ce | 0.05 / 0.05 | 0.05 / 0.05 |
| D5k | 0.26 / 0.29 | 0.16 / 0.38 |
| D5p | 0.01 / 0.00 | 0.05 / 0.02 |
| D5e | 0.02 / 0.04 | 0.04 / 0.01 |
| D6 | 0.56 / 0.56 | 0.56 / 0.56 |
| D7t | 0.31 / 0.31 | 0.32 / 0.32 |
| D7d | 0.14 / 0.12 | 0.13 / 0.15 |
| D8c | 0.45 / 0.40 | 0.43 / 0.38 |
| D8 | 0.71 / 0.76 | **1.28 / 1.38 FAIL** |

- **Reading the table.**
  - **Correction (coordinator, 2026-11-16).** The canonical set is ref-2761350a alone (nbflip-gates.ts DEFAULT_REF). artifacts/nbflip-port/ref is the superseded 33b78ebb set. Its column is kept only as a record and must not be judged on; an earlier version of this entry, and 88cbf0da's commit message, called it canonical.
  - On the canonical ref-2761350a every row passes for both except D8.
  - D8 at 30-90 already failed at HEAD: 7.118e5 particles against 6.309e5, margin 1.28; lane E measured 1.12-1.24 before the margin change. w2 makes it 7.176e5, margin 1.38: +5.8k particles, 0.09 of the tolerance, inside D8's run-to-run spread on identical trees (1.12-1.34).
  - **Ruling:** w2 is kept provisionally. D8 is the open M2 blocker, and w2 is re-judged there. The receipt-against-census question goes to lane G (note-lane-g-d8-census.md).
- Tools: tools/{t0-count.sh, t0-gate.sh, w2-gate.sh}; trees tree-t0c, tree-t0, tree-t0k, tree-w2. Logs: logs/t0, logs/w2.

**The staged-tile sampler (go/no-go measurement, frozen tiered step 20, uAdvect's F bricks).**
- Of 151.3M open fine (UNIT) taps, the share inside the brick plus a halo:
  - halo 2: 78.1%;
  - halo 3: 86.7%;
  - halo 4: 92.6%.
- Of 12.1M coarse (CACHE4) taps, 87.8% lie in the brick's tiles plus one.
- **Fit.** 32 KB of workgroup memory holds:
  - halo 2 (12³ x 3 x 4 B = 20.7 KB), with uAdvect's 6 KB trace list;
  - not halo 3 (32.9 KB) or halo 4 (48 KB) in f32.
  - Halo 4 fits only at f16 (24.6 KB), which changes the physics' precision.
- At halo 2, 22% of the taps still go to global memory, behind a per-tap inside/outside branch. Tools: tools/samp-halo.sh; log: logs/samp/halo.log.

**The step-9 seed burst: the old engine's own seeds (2026-11-16).** One 12-step check run of the old engine (tools/d9-old.mts, d8old/d9-old-12.jsonl).
- **How seeds were counted.** No kernel edit: at each fence it counts the live records still at the deferred seed age (NB_SEED_FIRST, before.w's age bits 0x7f7ffff8). No moved record reaches that age, so the count is that step's seeds.
- **Ours** is the pSeed counters (d9-12.jsonl).

| step | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| old seeds | 11.8k | 6.4k | 27.6k | 18.1k | **17.1k** | 23.6k | 26.4k | 28.7k |
| our seeds | 9.6k | 7.7k | 32.5k | 15.5k | **53.7k** | 21.4k | 21.9k | 18.3k |

- **Step 9 is ours alone: 3.1x the old engine's seeds.** The other steps are within 0.6-1.3x. Cumulative over 5-12: ours 180.6k, old 159.7k.
- **Where, at step 9 (old / ours):**
  - floor, y 0-15: 13.2k / 30.8k;
  - column top, y 96-111: 1.2k / 19.6k;
  - surge front, max(x, z) 96-111: 6.5k / 24.1k.
  - At step 7 the column top is 8.0k / 26.3k.
- **So the old engine does not burst.** Ours reseeds liquid φ in particle-free cells at the falling column's top and at the front on the floor. There, in the old engine, either the particles still cover the φ < 0 cells, or φ is no longer liquid.
- **Next step for the cause:** compare φ against the particle positions at the column top, steps 7-9, in both engines. Either φ lags the particles (lAdvect, lUnion), or the particles leave φ early (pMove, now lane E's).

**The step-9 seed burst, attributed: the column top's h tiles, not φ and not the move (2026-11-16).** Each engine got one 9-10-step check run. Ours ran on HEAD 9f3f417e (tools/brick-dam-l.mts --layers --line), the old one from the main checkout (tools/d9-old-l.mts, tools/old-line.mts: the observer's own owner traversal). Both runs read the column interior (x and z in 8..63) after each step's fence.
- **φ agrees to 0.1-0.15 cells.** The zero crossing on the centre line x = z = 32 is 4h-owned in both engines:

| step | 7 | 8 | 9 |
|---|---|---|---|
| old | 108.19 | 105.38 | 102.25 |
| ours | 108.33 | 105.50 | 102.35 |

  - The face velocities there agree to 1e-3 (step 9, y 100: -1.2963 against -1.2960).
  - So φ does not lag. lAdvect and lUnion are not the cause.
- **The records move alike.** The top records are about 0.3 cells higher in ours from step 6 on (pTop at step 8: 104.41 against 104.12). They fall at the same rate: 3.03 against 2.98 cells at step 8, 3.33 against 3.37 at step 9. So pMove is not the cause.
- **The h tiles at the column top differ.** Interior cells owned at h, by 4-cell layer:

| step | layer y | old h cells | our h cells |
|---|---|---|---|
| 5 | 108, 112 | 320 | 320 |
| 6 | 108, 112 | 640 | 832 |
| 7 | 104, 108, 112 | 1984, 1984, 0 | 2048, 2048, 384 |
| 8 | 100, 104, 108, 112 | 1664, 2560, 1664, 0 | 3136, 4288, 3904, 768 |
| 9 | 100, 104, 108 | 3200, 3200, 0 | 7936, 7936, 3904 |

  - By step 9 ours owns 2.5x the old engine's h cells at the top.
  - A shallow site seeds only in an h owner's cell (nbfSeedSiteOk's fineOwner).
  - The records' own depth at a given layer is 0.1-0.25 cells deeper in ours (y 100 at step 9: -0.94 against -0.69). That depth is read from the h field the extra h tiles carry. Deep sites then seed whatever their cover: step 9, y 100, deep seeds 1.45 against 0.04 per column.
  - It feeds itself: a step's seeds are the next step's records at the top (y 100 at step 9: 1.60 records against 0.95).
- **So the burst follows the census's h set at the column top.** That is lane E's residency and census. The tiers match at step 5 and part from step 6. The heat inputs (lHeatNow and FT_T, levelset-kernels) are lane F's and are not ruled out yet; the next check is a TRIG/heat census of the top tiles at steps 5-7, both engines.

**Item 3, uExtend (2026-11-16).** Frozen tiered dam, step 20. Times are on the timing slot; bitwise equality was checked on the check slot.
- **Attribution.** Each variant removes one part and times the kernel without it:
  - seed reads: 1.5 ms;
  - wSwept staging: 0.6;
  - xCode: 0.35;
  - tile distance: 0.2;
  - a second round on 9f3f417e: the wLiq staging 1.7, wS4 0.35, xFinish 0.2, the sweeps 0.45.
- **9f3f417e:** wSwept staged one lane per tile (216 lanes). 5.88 -> 5.48 ms, bitwise.
- **This commit:** vLiquidOwner's inputs staged once: the 4^3 tiles' 4h bit and centre phase, and the 27 bricks' FC_LIQ words. A coarse cell had re-read its tile's 8 C corners for each of the tile's 64 cells. 5.32 -> 4.23 ms (HEAD and the change in the same run), bitwise. All h: 4.34 against 4.27.
- **Next in uExtend:** the per-axis seed reads (fgCellCoarse, fgSlot, fgFace, xCode), staged per window cell once for all three axes.

**D8: lUnion's crossing-guard travel, ported verbatim (2026-11-16).**
- **The bisect.** I cut step 5 (the 5th advance, from the frozen step-4 state) after each level-set pass, at brick 7's +x plane (x = 63, z = 60, y 109-115), next to a coarse brick.
  - The head, lInit, lAdvect, lcAdvect and lSplat leave it clean: PHI1 -3.36, -2.34, -1.34, -0.34, 0.66.
  - lUnion writes the exact integers -5, -4, -3, -2 at y 109-112. lPublish keeps the -2.00 at the interface row. x = 62 is untouched.
  - Only x = 63 is coupled: nbfCoupleTheta at local x = 3 weighs the coarse +x tile 0.25.
- **The cause.** nbfBulkDepth returns min(φ, -depth), where depth is the distance to the box of last step's nearest crossing cell, a whole-cell distance. The travel should relax it to max(depth - travel - 2, 0). Our travel was a stand-in, 4·ceil(FC_BOUND·dt/h), the brick's record speed bound. That is 0 in a newly-h brick with no records, so the guard clamped φ.
- **The fix.** The travel is now the old nbCrossingTravel of the tile's spread-reach word, which measureBand copied into the candidate words. That word is lane E's REACH1 at the tile's C slot, saturated where no page holds the tile or REACH_STEP is not this step's (nbrTileReach's rule).
  - Lane E's lcCouple carried the same stand-in. Its fix lands next, gated together with this one.
- **The gate (tree-g1: HEAD 259e2e59 + both halves, check slot).**
  - Both brick wide-dam tests pass. They fail with the C half alone.
  - dam-300 tiered composite against ref-2761350a: every gate passes except D8, and D8 fails by a narrower margin than before:

| D8, window mean | 30-90 | 105-165 | 180-300 | score |
|---|---|---|---|---|
| w2 (before) | 7.176e5 | 4.249e5 | | 1.38 |
| lUnion alone (tree-u1) | 7.069e5 | 4.227e5 | | 1.21 |
| both halves (tree-g1) | 6.992e5 | 4.188e5 | 3.859e5 | 1.08 |
| ref-2761350a | 6.309e5 ± 1.7e4 | 3.838e5 | 3.484e5 | tol 6.31e4 at 30-90 |

  - D8c (particles at step 16) moves from 3.674e5 to 3.206e5, against 3.516e5, and passes.
  - lUnion alone, frozen tiered step 20, timing slot: 1.24 -> 1.03 ms.

## D8 counted as the old receipt; the cold freeze measured; the pre-impact seeding lag (2026-11-16)

**Count.** Per the coordinator's ruling (plan §4.6, "Ruling on D8's count"), the dam runner now counts D8's
`particles` as live records less this step's seeds (age NB_SEED_FIRST). That is the old receipt state[1]. On
the old engine, census records less seed-age records equals the receipt at all 35 samples of a dam-300.

**Cold freeze.** tree-cf0 is the old engine at 2761350a with UNIFORM_COLD_FREEZE false. It reads D8 0.13 and
D8c 0.02 against ref-2761350a, so the freeze is not what separates us.

**Re-scored** (one run each, check slot): D8 g1 0.83, sx 0.85, sh 0.57, all passing. D8c fails low: 1.68, 1.95
and 2.14.

Live records, this step's seeds, and the receipt, per step (old = tree-cf0):

| step | old live / seeds / receipt | g1 | sx | sh |
|---|---|---|---|---|
| 1 | 79135 / 3448 / 75687 | 71578 / 40395 / 31183 | 71578 / 40395 / 31183 | 74624 / 74624 / 0 |
| 5 | 103657 / 12453 / 91204 | 93918 / 9199 / 84719 | 93889 / 9216 / 84673 | 96046 / 8717 / 87329 |
| 7 | 140191 / 27550 / 112641 | 115182 / 14319 / 100863 | 114087 / 12939 / 101148 | 115252 / 12040 / 103212 |
| 8 | 165292 / 18100 / 147192 | 127727 / 10327 / 117400 | 126427 / 10085 / 116342 | 127558 / 9924 / 117634 |
| 9 | 189178 / 17399 / 171779 | 165739 / 35015 / 130724 | 143672 / 14260 / 129412 | 165117 / 34544 / 130573 |
| 10 | 218901 / 22904 / 195997 | 192489 / 14702 / 177787 | 172424 / 23298 / 149126 | 185975 / 8626 / 177349 |
| 12 | 288652 / 28103 / 260549 | 241131 / 18870 / 222261 | 225504 / 20538 / 204966 | 231345 / 19921 / 211424 |
| 14 | 342309 / 33002 / 309307 | 279588 / 23810 / 255778 | 267673 / 29055 / 238618 | 262741 / 26771 / 235970 |
| 16 | 390320 / 37870 / 352450 | 320360 / 27781 / 292579 | 316826 / 33942 / 282884 | 311217 / 34756 / 276461 |
| 30 | 832874 / 37705 / 795169 | 883769 / 37294 / 846475 | 879760 / 39665 / 840095 | 862655 / 45227 / 817428 |
| 60 | 629782 / 33425 / 596357 | 641517 / 32112 / 609405 | 664115 / 30879 / 633236 | 608282 / 31508 / 576774 |
| 90 | 432906 / 14519 / 418387 | 495345 / 10822 / 484523 | 491902 / 11069 / 480833 | 497580 / 11760 / 485820 |

- **The old engine seeds its burst at step 7** (27.5k), then about 18k a step. Ours seeds about 10–14k at 7–8 and
  bursts at 9 (35k in g1 and sh). By 16 we hold 18% fewer records.
- **Before the window the excess flips.** From about 30 on we hold more, as before. Retirement and seeding after
  the impact still differ, but inside the band.
- **Step 1 is a convention.** The old bootstrap files the initial band, while ours seeds it at step 1 as deferred
  seeds. So sh, whose HEAT4 is 0 at step 0, counts 0 at step 0.
- **Next** (the coordinator's rule 1, side-by-side): the per-tile support, heat and seeds at steps 1–20, old
  against g1, sx and sh (td-round.sh). The 7-against-9 burst timing is what to explain.

**Side-by-side per tile, steps 1–20** (td-round.sh; the old engine at 2761350a against g1, sx and sh; every
live tile's support word, heat, target and this step's seeds; td-cmp.py and td-cmp2.py):
- **Support (sx).** Through step 17, old and ours disagree on at most about 1% of supported tiles (for example at
  16: 17,532 agree, 118 old only, 214 ours only). g1's rule (FT_DIST field 2 ≤ 3, F tiles only) supports 2–6%
  more tiles than the old engine. So sx moves support to the old engine's and is verbatim. At 18–20, about
  1,000 column-top tiles are ours only: the column top falls through tiles, and the old engine's head φ_n
  against our post-level-set φ_{n+1} is the stated timing deviation.
- **Heat (sh).** FT_H equals HEAT4 on every F tile at every step. On C-only tiles HEAT4 matches the old heat (at
  most 2.4% of tiles off by more than 0.05, at 20). On F tiles all three trees run below the old heat and the
  gap grows (9% of tiles off at 16).
  - The examples are front tiles the old engine triggers that the census never triggers: at x tile 23 by the
    z = 0 wall, the old heat is 1.93 at step 9, cooling at 4 dt a step, and HEAT4 is 0.
  - So the heat deficit is the census trigger (lane E's item 4, tuDeformation), not the read. sh is verbatim in
    what it changes.
- **D8c's deficit is the floor ahead of the front.** At step 7 the old engine seeds 9.3k and 7.3k in floor tiles
  y 0 and 1; 9.6k of its low-region seeds fall in tiles that are C-only in ours. At 8 it seeds 9.4k in y 2,
  8.4k of them in C-only tiles. Ours seeds that floor at step 9 (9.3k, 10.6k and 9.0k in y 0–2), two steps
  late. The tiles are not h in time, the same census trigger and refinement.

## 2026-11-16: velocity redesign diagnostic and experiment A (packed trace): no-go, not landed

Frozen dam at step 20, poolF 4913, timing slot, one run each, no 00000206 aborts. Scratch:
~/code/me/fluid-scratch/lane-f (trees tree-vc, tree-vn, tree-v1, tree-v1l, tree-vdup, tree-pa, tree-pf, base-h1;
logs logs/vd, va, vb, vdup, pa, kt).

**The trace, per sample** (a sample is one 3-axis trilinear, 24 taps; a face of n segments takes 2n samples plus a
last 1-axis one; fixed part from a no-trace knockout, the per-face and per-sample parts from full against
one-segment traces):

| | seep reservoir (step 1100) | all-h | tiered |
|---|---|---|---|
| traced faces | 503k (17k chamber) | 2.06M | 497k F + 24.5k C |
| segments a face | 1.0 | 3.7 | 6.1 |
| trace kernel, ms | 0.41 | 11.1 | 8.9 |
| fixed (no trace), ms | 0.14 | 0.60 | 1.70 |
| ns a sample | 0.22 overall: about 0.1 in the reservoir, about 2.5 in the chamber | 0.63 | 1.08 |

- SIMD divergence is 92% (tiered) and 95% (all-h) lane-busy in active warps; 0.2% of samples leave the 3^3 slot
  window; the fine weight and the per-tap checks are 25% of the tiered sample.
- **Correction (two claims).** The 2026-11-1x entry above says our per-segment cost is at or below the seep's: wrong
  as stated, since it compared totals at the seep's one segment a face. And the seep's per-sample figure is not a
  like-for-like target: its reservoir is at rest (one segment a face, taps in cache); its chamber's long traces cost
  about 2.5 ns a sample, more than ours.
- **The duplicate test was flattered.** Tracing every face twice raised tiered uAdvect only 63%, which read as idle
  lanes. But each duplicate sat beside its twin and read the same cache lines, so the marginal sample was cheap for
  locality, not for idle capacity.

**Experiment A, packed trace (tree-pa): no-go.** uAdvect classifies and appends its traced faces to a global list
(an atomic range a brick, NB_FAIL_LIST 'TR' on overflow); uTrace, 1024 workgroups of 64 lanes, takes 64 entries at
a time from a cursor, stages the slot window when the chunk's brick changes, and traces with the unchanged sampler.
UP bitwise equal to HEAD (45356bb8) in both modes.

| | HEAD uAdvect | classify + uTrace | classify alone |
|---|---|---|---|
| tiered | 9.37 ms | 8.52 (-0.85) | 1.68 |
| all-h | 10.75 ms | 11.00 (+0.25) | 0.65 |

Under the 2 ms gate; dropped (the coordinator's ruling). With every lane busy the tiered sample still costs about
1.05 ns: idle lanes were about 11% of the trace, not the 40% estimated. The trace is latency-bound on distinct taps,
and the cost grows with the travel (6.1 segments a face tiered against 3.7 all-h).

**Fast taps (tree-pf): no-go.** One slot lookup a 2x2x2 tap block where the block lies in one brick: slower
(tiered 11.7 ms, all-h 16.1) and not bitwise (123k and 460k vel words differ, probably the inlined lerp's fma
contraction, not verified). About a third of the samples take the slow path, so nearly every warp runs both.

**uTransfer knockout (tiered):** 4.48 ms as is; 4.20 without the C pages' body (vcTransferPage, about 0.3); 2.66
staging and LIQ only (per-face loop skipped). Staging is about 2.4 ms of it (0.64 us an F brick), the per-face
loop about 1.8. Next: step 3, the staging skeleton and an apron's price.

## 2026-11-16: step 3, the staging skeleton and an apron's price (no-go, not landed)

Frozen dam step 20, timing slot, `--time=6`, no 00000206. Tree tree-ap (45356bb8 plus the measurement aprons: an
`fapron` buffer and fill kernels at the tail's end). Each staging item is the difference between cumulative
uniform early returns (`if(cfg.w<9999u){return;}`); the F rows run with the C pages' body removed.

**Tiered velocity skeleton (ms).**

| kernel (full) | launch | window | φ ring | other staging | work | C pages' body |
|---|---|---|---|---|---|---|
| uAdvect 8.73 | 0.06 | 0.12 | 1.01 | - | classify 0.80, trace 6.50 | 0.24 |
| uTransfer 4.57 | 0.08 | 0.08 | 1.02 | centre 0.27, heat 0.68, depth 0.18, LIQ 0.05 | faces 1.83 | 0.38 |
| uViscous 1.82 | 0.07 | 0.05 | - | wTap 0.55 | faces 0.81 | 0.34 |
| uExtend 4.18 | 0.08 | 0.22 | - | LIQ words, tiles 0.33; liquid bits 0.44; wV/wD 1.70 | sweeps 1.42 | - |

**φ rings elsewhere (ring part of the kernel's time, ms).**

| kernel | tiered | all-h |
|---|---|---|
| lTiles | 0.50 of 0.76 | 0.06 of 0.25 |
| lClear | 0.44 of 1.10 | 0.06 of 0.71 |
| lDistance (12^3) | 1.22 of 3.34 | 0.24 of 1.38 |
| pLiquid | 0.49 of 0.69 | ~0 of 0.16 |
| pDeposit | 0.33 of 1.37 | 0.06 of 0.93 |

The attribution: the tiered staging floor is owner-resolved φ (foVertexPhi) inside heavy kernels, about 5 ms a step
against 0.5 all-h, not memory traffic.

**Aprons.** Velocity (3 axes) plus vertex φ: radius 1 16 KB a brick (78.6 MB at 4913 slots), radius 2 27.6 KB (136
MB). The fill (one light launch over F_ALL) costs 0.86 / 1.33 ms tiered and 0.32 / 0.48 all-h; filled at the tail
it serves the next step's samplers bitwise (uAdvect, lAdvect, pMove). Flat apron reads in place of the staging:
uAdvect ring 1.19 -> 0.14, uTransfer ring plus depth 2.31 -> 1.29, uExtend -0.33.

**Trace locality** ("in r": all three 2x2x2 blocks of the sample inside the apron).

| sampler | now | r1 | r2 | r2 all-flat bound | samples | in r1 | in r2 |
|---|---|---|---|---|---|---|---|
| uAdvect tiered | 8.73 | 8.58 | 7.83 | 4.33 | 6.70M | 54.5% | 68.7% |
| uAdvect all-h | 10.48 | 8.28 | 6.29 | 2.52 | 17.25M | 70.9% | 84.7% |
| lAdvect tiered | 3.34 | 3.17 | 2.90 | 2.42 | 0.94M | 57.9% | 69.6% |
| lAdvect all-h | 5.48 | 5.21 | 4.73 | 2.69 | 3.49M | 81.9% | 90.2% |
| pMove tiered | 3.42 | 3.36 | 3.12 | 2.60 | 0.98M | 75.0% | 85.7% |
| pMove all-h | 2.61 | 2.37 | 2.19 | 1.87 | 0.80M | 75.7% | 86.6% |

r1 and r2 are bitwise (a sample whose blocks leave the apron takes the current path).

**Ruling: not taken.** Tiered net about 2.5-3 ms (6.6 gross less three φ refills inside the level set and the tail
fill) against a 136 MB buffer; all-h about 4.9. Kept for later: the all-h r2 trace sampler (uAdvect -4.2 ms,
bitwise). Next: the resolved-φ pass (owner-resolved φ written into F slots' non-owned vertices), go/no-go.

## 2026-11-16: resolved φ in the F slots (measurement: go, not landed)

Tiered, a vertex an F slot holds but does not own (an incident tile is 4h) reads through its owner: foVertexPhi walks
up to eight candidate tiles, then interpolates eight C vertices. Every ring and every fgPhi pays that again.
Measurement tree tree-rp (45356bb8 plus the following):
- **lResolve** writes each F slot's non-owned vertices with the window's owner-resolved fgPhi into the slot's own PHI
  word (the 512 vertices a slot already stores). It runs 256 lanes a slot over F_ALL, light, as four placements: n0
  at the move stage's start (after the set update: ownership and the transfer's C writes), n1 after lcCouple, n2
  after lcClear, n3 after lcPublish (before pSeed). Boot takes n1 after lInit (and n2 in the analysis).
- **Reads:** fgPhiWord(PHI) and the ring's staging read a vertex an F slot holds as a plain load; any other vertex
  (no F slot, or past the lattice) keeps the owner path.
- **Freshness:** a non-owned vertex's value depends on C PHI and the tile ownership only.
  - C PHI is written by lcCouple, lcClear and lcPublish (lcAdvect and lcRetire write PHI1), and each is followed by
    a resolve.
  - Ownership changes only at the set update, followed by n0.
  - F kernels write PHI only through lSetPhi, which skips non-owned vertices. The set update's writes
    (setTransferCells xWriteF, setInit, bootPhi) come before n0. setTransferCells writes every vertex it takes over
    from its old layout, so it never consumes a stale non-owned F word.

Frozen dam step 20, timing slot, quiet GPU (one process), no 00000206. Bitwise: 24 placements checked, 0 bad.
"Global" is the plain load in every fgPhi read; "ring" is the plain load in ring staging only.

| kernel | base | global | ring only | best saving |
|---|---|---|---|---|
| uCache | 1.17 | 0.87 | 1.22 | 0.31 |
| pLiquid | 0.66 | 0.26 | 0.29 | 0.39 |
| vGauge | 0.66 | 0.29 | 0.69 | 0.38 |
| lAdvect | 3.25 | 3.07 | 3.08 | 0.18 |
| lcAdvect, lUnion | 2.71 | 2.75 | 2.75 | 0 |
| lTiles | 0.71 | 0.32 | 0.30 | 0.39 |
| lClear | 1.12 | 0.68 | 0.66 | 0.45 |
| lDistance | 3.38 | 1.54 | 2.30 | 1.84 |
| uAdvect | 8.70 | 8.39 | 8.00 | 0.81 (ring only) |
| pDeposit | 1.41 | 1.05 | 1.01 | 0.44 |
| uTransfer | 4.48 | 2.99 | 3.73 | 1.48 |

Passes: n0 0.55, n1 0.64, n2 0.54, n3 0.57 ms, 2.30 in all. Net: 6.67 - 2.30 = **4.4 ms a step** tiered (3.9 with
the global read in uAdvect too). Each pass pays for its own consumers: n0 1.26 against 0.55, n1 0.84 against 0.64,
n2 1.84 against 0.54, n3 2.73 against 0.57.

Not verified bitwise: pMove and pSeed, whose outputs differ old against old (atomic ranks). An earlier run with a
concurrent check-slot job agreed in kind but is not quoted. All-h is unchanged (no resolve, FG_TIERED false).

## 2026-11-16: resolved φ landed (go)

Landed as measured, with these changes:
- **Read selection.** FG_PHI_WGSL plain-loads resolved words (FG_PHI_RESOLVED). FG_PHI_OWNER_WGSL reads through
  the owner outside a staged ring; lResolve and uAdvect use it, since uAdvect pays only in its ring.
- **Placements:**
  - move's head (after the set update), after lcCouple, after lcClear (in the analysis), and after lcPublish
    (before pSeed);
  - the boot's head, before lInit;
  - the kick's force and seed.
- **Boot fix.** The measurement tree resolved after lInit, so its boot read stale words of fresh slots. HEAD never
  did: it read through the owner.
- **n1 kept.** lTiles and lClear can't move after lcClear: lcClear raises far C φ to 16, which changes the
  non-owned values they read.
- **Schedule test** (fluid-staging.test.ts): no resolved read between a C PHI writer, or a schedule's start, and
  the next lResolve. C writers are classified by entry: lcCouple, lcClear and lcPublish write PHI; lcAdvect and
  lcRetire write PHI1 only. Any other C writer fails the test.

**Gate.** Tree-rr is 32d46e7f plus the change. Runs were on the check slot unless noted.
- Both wide-dam tests pass.
- dam-300 on ref-2761350a: every row passes except D8 at 1.23, high (7.082e5 at 30–90). That is the
  pre-existing range on HEAD-equivalent trees, 1.08–1.24.
- 60-step probes against tree-hh:
  - steps 0–1 are identical, and step 2 differs by 3 particles (the atomics);
  - at step 60, the candidate against HEAD: KE 66.5 against 68.2 J, drift −0.063 against −0.060, particles 644k
    against 636k.
- Fenced hold, steps 13–165, timing slot: median 54.59 → 52.03 ms (−2.6). The frozen state's −4.4 is step 20's.

Logs: ~/code/me/fluid-scratch/lane-f/logs/rr (and logs/ld, the pre-rebase round on 8d298dc8).

## 2026-11-16: inviscid dam landed (31311324), and uTransfer re-measured (B is moot)

**Fenced hold**, steps 13–165, timing slot. HEAD is 07f5bf3b (tree-h4); the candidate adds the dam at viscosity 0
(tree-v4). The median falls from 52.11 to 48.83 ms (−3.3). The thirds are 73.6/52.3/43.2 against
70.8/49.2/42.9.

**uTransfer re-measured** on the inviscid dam, with resolved φ, at frozen step 20 (timing slot). The full kernel
takes 3.29 ms. Cumulative knockouts:

| part | ms |
|---|---|
| launch | 0.10 |
| window | 0.07 |
| ring | 0.30 |
| centre | 0.30 |
| depth | 0.15 |
| liquid | 0.07 |
| F faces | 1.86 |
| C pages | 0.44 |

All staging together is about 0.75 ms, which is under the 1 ms bar. Inviscid, the Laplacian taps that B removes
are not read at all. B is not built.

**Same round, other kernels:**

| kernel | ms |
|---|---|
| uExtend | 4.44 (next, item 3) |
| uAdvect | 8.14 |
| uCache | 0.87 |

The baselines were checked bitwise. No 00000206 abort was logged. Logs: logs/v4 and logs/bt.

## 2026-11-16: uExtend's staged tile bits and uAdvect's exact sampler switch landed; lever 1 dropped

**uExtend.** The 5³ tiles' supported (bit 2) and owner (bit 3) bits are staged once in wTile4, beside the coarse
bits. xCode, the seeds' coarse test, xFaceU's owner test and the finish read the staged words.

**The tiered sampler (fluid-grid, fgSampleVel), two exact changes:**
- **The fine switch in O(1).** The fine weight is only ever tested for being over 0, so fsFineAny replaces
  fsFineWeight. It reads HMASK first, then the support, then bit 13. Otherwise it forms a 27-bit reach mask: the AND
  over axes of the offsets whose delta is under 2, using the loop's own float expressions. It then tests that mask
  against HMASK.
- **The zero-weight pair skip.** In fsFine and fsCoarse, a trilinear corner beyond an axis whose fraction is 0 is
  not read: at a face centre, 1 tap of 8 on the face's own axis and 4 of 8 on the others. The result equals the
  old one up to the sign of a zero sum.
- The same sampler runs in uAdvect, lAdvect, lcAdvect, pMove and pSeed.

**Lever 1 (the hoisted classification) is dropped.** Both forms were bitwise and neither saved time. The 0.95 ms
attributed to classification is not in the vLive lookups or in the list atomics.

| lever 1 form | step 100 (ms) | step 20 (ms) |
|---|---|---|
| vLive bits staged in workgroup memory (729 a brick) | 3.87 against 3.43 | flat |
| one list atomic a lane instead of one a face | 3.446 against 3.426 | 8.39 against 8.25 |

**Lever 2 measured on HEAD (uAdvect at frozen steps 100 and 20; base about 3.45 and 7.8–8.0 ms):**

| variant | kind | step 100 (ms) | step 20 (ms) |
|---|---|---|---|
| early-exit loop (k1) | exact | 3.28 | 7.64 |
| k1 + pair skip | exact | 2.79 | 7.08 |
| support check dropped | knockout | 3.15 | 7.27 |
| HMASK != 0 only | knockout | 3.10 | 6.77 |
| O(1) switch, support first, + pair skip (e1) | exact | 2.89 | 6.65 |
| O(1) switch, HMASK first, + pair skip (e2, landed) | exact | 2.96 | 6.66 |

- The support-dropped knockout stayed bitwise at both steps but has no proof of exactness, so it is left out.

**Frozen, the landed batch against HEAD (tiered steps 20 and 100; all-h step 20):**
- uAdvect, uExtend, lAdvect and lcAdvect are bitwise at all three.
- pMove and pSeed differ only as the base differs from itself (their atomic ranks).

| kernel | tiered step 20 (ms) | tiered step 100 (ms) | all-h step 20 (ms) |
|---|---|---|---|
| uAdvect | 6.88 against 8.16 | 2.79 against 3.41 | 10.81 against 10.98 |
| uExtend | 4.32 against 4.50 | 4.42 against 4.64 | 4.06 against 4.25 |

**Gate (tree-xb against tree-h5 = 42567020):**
- Both wide-dam tests pass.
- dam-300 fails only D8, the particle upper bound, at margin 1.31 in the 30–90 window. The earlier trees ran
  1.09–1.23, and the grid kernels here are bitwise, so the difference comes from the atomics.
- Fenced hold, steps 13–165, timing slot:
  - median 48.97 → 49.54 ms (+0.57);
  - mean 54.33 → 53.44;
  - p95 78.4 → 75.9;
  - thirds 70.1/49.3/43.6 → 69.5/49.6/41.2;
  - paired per-step difference: median −0.95 ms, mean −1.32.
- The median sits on the steep part of the curve, and the candidate's trajectory carries more particles (498k
  against 476k at step 90, from the atomics). The paired per-step difference agrees with the frozen saving.
- No 00000206 abort was logged.

Logs: ~/code/me/fluid-scratch/lane-f/logs/{fx,fy,fz,xg}.

## 2026-11-16: uExtend skips the sweep of an axis with no source in its window; nbfP2GResolve selects the deposit at t = 1; the overwritten-trace skip dropped

**uExtend's pre-seed skip (exact).** wReach collects whether the window can seed a source: bit a is set by a reached
4h state of axis a in wS4 (xProlong's only finite seeds), bit 3 by any liquid cell in wLiq (xCode's code 1). An axis
with neither seeds only unreached (1e20) and blocked (−1) faces, and ufxCombine of unreached neighbours keeps old, so
the sweep is the identity: the seed loop, the inset sweep and its write-back are skipped (the barriers stay), and the
finish takes the face's own seed (xSeed).

**nbfP2GResolve's select.** A face the deposit takes whole (transition 1) returns the deposit exactly, instead of
mix(original, deposit, 1). On the frozen dam it is bitwise to HEAD at steps 20 and 100: Metal's mix already returned
the deposit at all 113k (step 20) and 139k (step 100) such faces. The p2g resolve parity test carries the select on the
old side.

**The overwritten-trace skip: no-go.** uAdvect skipped the trace of an inviscid face that the deposit (pDeposit moved
first) takes whole. The dependencies allowed the move:
- uAdvect reads the deposit's weights, which pDeposit writes; pDeposit writes nothing else uAdvect reads (fdep and
  parts[i].before.w);
- uAdvect writes UP0 and the set words FH_VOL_FILLED, FH_RATE and FH_TRACE_CAPPED; pDeposit reads none of them.

It was bitwise against the select alone (tiered steps 20 and 100, all-h step 20), but slower. Frozen velocity stage
(pDeposit + uAdvect + uTransfer, ms):

| variant | step 20 | step 100 |
|---|---|---|
| HEAD | 11.1–11.3 | 7.3–7.5 |
| near inputs loaded per face | 12.2–12.5 | 8.0–8.2 |
| near inputs staged like uTransfer | 12.4–12.8 | — |
| pDeposit first, skip compiled out | 11.16–11.19 | — |
| the fdep load only | neutral | — |

With nothing taken, uAdvect with the skip's code ran 9.0 ms against 6.7: the code inflates the trace even where it
never runs (register pressure). A separate taken-mask pass would net at best about 0.5 ms at step 20 and nothing at
step 100, so it is not pursued, and pDeposit stays after uAdvect.

**Frozen, uExtend against HEAD (bitwise at tiered steps 20 and 100 and all-h step 20):**

| step | uExtend (ms) |
|---|---|
| 20 | 4.37 → 3.64 |
| 100 | 4.73 → 3.66 |

**Gate (tree-yf against c35cbe65):**
- The wide-dam and velocity-formulas tests pass (10/10; the p2g resolve parity 0 words differ).
- dam-300 fails only D8, at 1.34 (HEAD 1.39–1.49).
- Fenced holds, paired c6 yf yf c6, steps 13–165:

| pair | median (ms) | mean (ms) | paired per-step median / mean |
|---|---|---|---|
| 1 | 46.98 → 47.31 | 52.42 → 52.02 | −0.42 / −0.47 |
| 2 | 49.32 → 48.66 | 53.99 → 52.72 | −0.99 / −1.57 |

- No 00000206 abort was logged.
- The gate ran on c35cbe65; lane E's 06d4ca89 landed meanwhile and does not touch these files.

Logs: ~/code/me/fluid-scratch/lane-f/logs/{yb,yt,yf}.

## 2026-11-16: the ring-brick skip (velocity area lever 2) lands

**Ring bricks.** A ring brick (tiered) owns no F face: none of its eight tiles, and none of the twelve tiles below
its three lower planes, is h-owned. The readers take the 4h interpolation wherever neither side of a face is
h-owned, so nothing ever reads those F faces (`vRingBrick`, velocity-kernels.wgsl.ts). On a ring brick:
- uAdvect returns after its head, keeping the group-0 work;
- uTransfer stores the brick's FC_LIQ words (uExtend stages its neighbours' FC_LIQ) and clears its deposit, then returns;
- uExtend stages nothing and sweeps nothing; the flag is workgroup-uniform, so the barriers stay uniform.

All-h never takes the skip.

**Frozen, against 31630b18 (bitwise "same" over uAdvect, uTransfer with pDeposit before it, and uExtend, at
tiered steps 20, 100 and 300 and at all-h step 20):**

| step | uAdvect (ms) | uTransfer (ms) | uExtend (ms) | total saved (ms) |
|---|---|---|---|---|
| 20 | 5.84 → 5.77 | 2.70 → 2.29 | 2.84 → 2.19 | 1.13 |
| 100 | 2.40 → 2.30 | 2.50 → 2.23 | 2.84 → 2.42 | 0.79 |

**Gate (tree-zg against 31630b18):**
- The wide-dam, velocity-formulas and fluid-staging tests pass (14/14).
- dam-300 passes every gated row, D8 included (0.87).
- Fenced holds, paired h8 zg zg h8, steps 13–165:

| pair | median (ms) | mean (ms) |
|---|---|---|
| 1 (holds 1 and 2) | 45.39 → 44.54 | 50.30 → 49.18 |
| 2 (holds 3 and 4) | 45.55 → 44.07 | 50.19 → 48.66 |

- No 00000206 abort was logged.
- Lane E's 3090d99e (lcAdvect's C ring) landed during the gate. It does not touch velocity-kernels.wgsl.ts.

Logs: ~/code/me/fluid-scratch/lane-f/logs/zg.

**Velocity area: the levers priced.** These are from an xctrace profile at step 100, with lever 1 bounds from knockouts:
- *Split uAdvect into F-only and C-only:* no-go. The F-only variant stays at 704 max threads, and the C pages cost only
  0.07–0.37 ms inside the fused kernel.
- *Exact per-stencil fast path with a fallback:* slower. About a third of the stencils cross a brick, so the SIMD groups
  diverge.
- *Rolled tap loops, a slot table per sample, and a 128-lane uAdvect:* all slower, by 0.6–2.7 ms.
- *A slot lookup confined to the window:* not exact, because traces leave the window.
- *The 9³ UNIT apron (one slot lookup per sampled stencil):* bound about 4.9 / 2.8 / 1.6 ms net at steps 20 / 100 /
  300 across uAdvect, lAdvect, lcAdvect, pMove and pSeed. The coordinator approved it and lane E agreed the ABI.
  It is in its gate.
