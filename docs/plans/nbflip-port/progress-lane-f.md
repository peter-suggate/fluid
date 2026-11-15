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
