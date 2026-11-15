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
