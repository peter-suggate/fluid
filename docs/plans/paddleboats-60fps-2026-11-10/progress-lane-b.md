# Lane B progress: fused brick passes for the non-pressure frame (plan item 14)

2026-11-10. Work on main, uncommitted. Plans: `paddleboats-60fps-2026-11-10.md` item 14,
`brick-resident-step-2026-11-08.md`.

## BINDING REQUIREMENT (user, via coordinator, 2026-11-10)

"Never, ever do we want single kernel passes." The fused pass must maximise work capacity on the GPU:
1. No pass that runs as a single workgroup or a single serial lane (the pTail / one-lane verdict /
   lane-0 tail shape). Every launch spreads its work over many workgroups so all cores are busy. A brick
   with little work must not hold a whole launch; a brick with a lot (a splash brick, 10x particles) is
   split across groups, not serialised in one.
2. Occupancy is a first-class target: each pipeline's per-thread register and threadgroup footprint must
   keep Metal's maxTotalThreadsPerThreadgroup at 1024 (or as near as can be proven), measured with the
   compile-only probe for every pipeline added. The move's 384 threads (12.5%) is the shape not to build
   again. If fusing more stages drops the cap, two wide, well-occupied launches beat one starved one:
   launch count is a means, busy cores are the goal.
3. No launch that exists to run one tiny kernel over a handful of items: that work folds into a wide
   pass already visiting those bricks.
4. The ledger gives each existing non-pressure launch its workgroup count and whether it is
   single-workgroup or serial-tailed, and how many of those the fused pass removes. This file states each
   new pipeline's thread cap and lanes per group.

(Ledger, choice, design, measurements and gates follow.)

## 1. Launch ledger: paddleboats, step 30 (GPU run 1)

Command (rc 0; GPU memory 4.96 GB at start, 5.48 GB peak):
`tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --pipelined --dispatches --steps=30 --every=30 --out=~/code/me/fluid-scratch/lane-b/ledger-paddleboats-30.json`.
Step 30: drift -0.0059, max speed 6.11 m/s, 192 resident blocks of a 288-block pool. The ms figures are not a timing gate.
`tools/gpu-dispatch-counter.ts` now attributes every dispatch to its pipeline (`takeKernels`), and the probe writes
`dispatchesByKernel` to its JSON.

**489 launches; pressure 215 (out of scope); non-pressure 274.** The probe's last step adds 13 diagnostic
global-surface-volume launches, so the production frame has about 261.

Key: wg = workgroups a launch. **1wg** = single-workgroup launches. **Tiny** = 2 to 8 workgroups.
B1 = reads only last frame's state. B2/B3/B4 = a same-frame halo chain of that depth. T = topology.

| Family (pass label) | Launches | Workgroups a launch | 1wg | Tiny | Class |
|---|---|---|---|---|---|
| Global surface volume (gauge + 13 probe-only) | 17 | 1..4096 (solve 2 x 1) | 2 | 0 | T/diagnostic |
| NB-FLIP surface pressure authority | 16 | bodyArgs, bodyReset 1; stamps 37 | 2 | 0 | T |
| Frame extension (ufxF/RH/R/4/H) | 15 | 256..3072 | 0 | 0 | B4 chain |
| Dynamic ownership census | 13 | regionList 1, interfaceBlocks 5, pageSeed 37 | 1 | 1 | T |
| Mixed layout build | 12 | scan 1, slotCut 1, slotClaim 5 | 2 | 1 | T |
| Surface advect deferred | 12 | 1024 | 0 | 0 | B1 + B2 (staged trace) |
| Geometric fill | 11 | 128..4096 | 0 | 0 | B1 |
| NB-FLIP seed | 9 | seedCut 3 x 1 | 3 | 0 | T |
| Remap | 9 | 256..1024 | 0 | 0 | T |
| Surface advect | 9 | 1024..4096 | 0 | 0 | B1 (+wall B2) |
| BULK16 page transfer | 8 | btDecideAsked, btMapPromoted 1 | 2 | 0 | T |
| **Local speed certificate** | **8** | localSpeed 4096; tileError, spread0/1/2, coarseBits, certify 256; **coarseBlocks 5** | 0 | **1** | **B3 chain** |
| Page pool commit | 8 | upSlots, upSuperSlots, upWindow 1; upClaim, upClaimSuper, upHost, upPublish 5 | 3 | 4 | T |
| **Shared frame plan** (2 encodes) | **8** | seed 4096; dilate0/1/2 256 | 0 | 0 | **B3 chain** |
| NB-FLIP arena search, reconstruction, transfer | 7 + 7 + 7 | 1024..4096 | 0 | 0 | B1/B2 |
| Live block copy | 7 | 256 | 0 | 0 | T |
| Frame bricks | 6 | ubaLevel1 1 | 1 | 0 | T |
| Region heat / dynamics / freeze | 6 + 3 + 2 + 2 | 37..256 | 0 | 0 | T |
| Phi resolve | 5 | 4096 | 0 | 0 | B1 |
| All-4h solid record, claimed blocks | 5 + 4 | 256..4096 | 0 | 0 | T |
| Block neighbours | 4 | bnRows 5, bnWalls 1 | 2 | 2 | T |
| Body forces, page initialise | 4 + 4 | 1024..4096 | 0 | 0 | B1 |
| Slot init, ring enter, hanging taps, retirement, slabs, placement | 3 each | slabAllocate, slabPlace 1 | 2 | 0 | T |
| 2-launch families (activity, crossing, air, twins, transfers, residency, ...) | 26 | pageCompact 1, claim 5 | 1 | 1 | mixed |
| 1-launch families (BULK16 publish, reclaim, rigid integrate, sort, ...) | 22 | finish, begin, integrate 1 | 3 | 1 | mixed |

Non-pressure single-workgroup launches: **22**, almost all topology bookkeeping (layout scan/cut, pool commit, BULK16
decide/map/publish, NB-FLIP seedCut and slab placement, rigid integrate). Tiny launches (5 workgroups, a lane per block
over 288 blocks): coarseBlocks, interfaceBlocks, slotClaim, upClaim/upClaimSuper/upHost/upPublish, bnRows.

## 2. Choice: the frame plan's support dilation and certificate tail in tile windows

Candidates, by launches removed against compile and engineering risk:

- **B1 surface advect (21 launches).** The worst risk. The inlined umAdvected took 640 s to compile in S1, and
  Lane C had to split it. Its pipelines report 64-thread caps (progress-lane-c.md). Fusing it builds the starved kernel
  the binding requirement forbids.
- **B4 extension (15).** Grid-wide ufxF -> ufxR -> ufx4 -> ufxH chains that Lane E already tuned. A brick window of
  depth 4 at the extension's float payload is (8+8)^3 x 3 words, which is over the stage budget.
- **B2 surface reconstruction and redistance (~40).** Long same-frame halo chains, plus the open redistance-reach
  question (Lane D's area).
- **T topology (~85, including the 22 single-workgroup launches).** These are the real serial tails, but they are
  prefix and claim protocols: scan, cut, claim, publish. Folding them is a protocol redesign across Lanes R, D and F,
  not a brick pass.
- **The frame plan and certificate (16 launches, one tiny).** Chosen. The code is integer-only, apart from tileError's
  few floats. No other lane touches `uniform-mixed-frame-plan.ts`. Every stage is a separable halo chain of a
  per-tile-key word, which is exactly what a staged tile window does.
  - 16 launches a frame become 7, the most removed per unit of risk.
  - It removes the 5-workgroup coarseBlocks tail.
  - Every new launch is a 256-lane group per half page, at min(512, 2 x blocks) groups: 512 on paddleboats.

## 3. Design (method param `framePlanPass`: "chain" default | "window"; unknown values throw)

Item = half a resident page: block `ulResidentBlock(i>>1)`, tile z 4*(i&1) .. +3, a lane per core tile (256).
Fixed grids of `framePlanWindowGroups(blocks)` = min(512, 2 x pool blocks) stride the items. No indirect dispatch.
A staged word sets bit 31 where the key is stored. Each group runs the chain's axes in place in threadgroup memory,
a lane owning a row, column or core column, and writes only its 256 core tiles.

- **supportWindow** replaces dilate0/1/2 and runs once per support encode.
  - Window: 14 x 14 x 10 (halo 3), 1960 words (7.7 KB).
  - Axes: x over 140 rows, y over 80 columns, z over 64 columns.
  - Output: umSupport plane 3 only. Planes 1 and 2 are not read outside the plan (grep), so they are not written.
  - A policy with fineReach + shellReach > 3 takes the chain for that encode (host check).
- **certifySpread** replaces tileError, spread0, spread1, coarseBits and coarseBlocks.
  - Window: 28 x 28 x 4 (halo 10 in x and y, none in z), 3136 words (12.3 KB).
  - The spread runs over the raw localSpeed reach in speeds[0, T). x runs over 112 rows, y over 32 columns.
  - Writes the x/y spread to umSupport[T, 2T). That plane is free under this pass, and using it avoids the ping-pong
    race on speeds.
  - Writes the 16 tile-bit words per block. The slab and neighbour summaries (words 16, 17) are written all-set.
    coarseAny then tests each page of a box on its own words: the same answer, without the shortcut.
  - A certificate radius above 10 saturates every reach. That is conservative: no tile is regular, so every tile takes
    the general sampler.
- **certifyTiles** replaces spread2 and certify. It runs the z spread from umSupport[T, 2T) over global memory, then
  certify's classification and its per-workgroup aggregated list appends. Final reach and class go to speeds[T, 2T).
- **Per frame:** chain 4 + 8 + 4 = 16 launches; window 2 + 3 + 2 = 7 launches.

**Radius margin on paddleboats.** The construction report gives h = 0.025 m. With dt = 1/60 s, 6.11 m/s is 4.1
cells of travel a step, so R = ceil(4.1/4) + 2 = 4 tiles against a halo of 10. The window saturates past R = 10,
which is over 32 cells a step (48 m/s at this h). At a 0.005 m lattice that limit is 9.6 m/s. Saturation sends every
tile to the general sampler: a performance cliff, not an error.

**Deleted when it wins** (once "window" becomes the default and then the only value):
- in `uniform-mixed-frame-plan.ts`: dilate0..2, tileError's entry wrapper, the spread0..2 pipelines and the
  `uniformFramePlanSpreadWGSL` include, coarseBits and coarseBlocks (keep only coarseAny/coarseIn), certify, the
  `SUPPORT_ENTRIES` / `CERTIFICATE_ENTRIES` launch tables;
- `uniform-frame-plan-spread.ts`, unless `tools/probe-sparse-speed-spread.ts` keeps it as its oracle;
- the chain fallback for policies wider than 3 tiles, if no scene uses such a policy.

## 4. Measurements (compile only; no timing runs)

### GPU run 2: compile budget and thread caps

GPU run 2 (`~/code/me/fluid-scratch/lane-b/compile/run.sh` under `fluid-gpu-run.sh check`, rc 0) did three things:
- built the paddleboats solver with `framePlanPass "window"`, 0 steps, using `tools/profile-solver-construction.ts
  --dump`, which now takes `--values=`;
- cold-compiled each frame plan entry alone, using `tools/probe-pipeline-compile.ts` (a4114158);
- read Metal's `maxTotalThreadsPerThreadgroup` from Dawn's MSL with Lane T's probe (`dump-msl.mts` + `mslprobe`,
  64/256-thread cap stripped).

naga validates the dumped full module, and the fragment alone over stubs for both shell rules.

| Pipeline | Cold compile | Thread cap | Lanes a group | Threadgroup memory |
|---|---|---|---|---|
| supportWindow (new) | 0.18 s | **1024** | 256 | 7.7 KB |
| certifySpread (new, as first built) | 0.17 s | 576 | 256 | 12.3 KB |
| certifySpread (new, packed max, as landed) | 0.21 s | **1024** | 256 | 12.3 KB |
| certifyTiles (new) | 0.18 s | **1024** | 256 | 8 B |
| dilate0 / tileError / spread0 / spread2 / coarseBits / coarseBlocks / certify (replaced) | 0.17 / 0.07 / 0.07 / 0.09 / 0.06 / 0.04 / 0.12 s | 1024 each | 64 | none |

The compile budget is 60 s cold per pipeline; every new pipeline compiles in under 0.25 s.

### GPU run 3: certifySpread's 576 cap

GPU run 3 was compile only (`~/code/me/fluid-scratch/lane-b/occ/run.sh`), testing variants of certifySpread:

| Variant | Thread cap |
|---|---|
| tileError removed, tileError moved last, or tileError moved into certifyTiles | 576 |
| spread removed | 1024 |
| spread maximum on packed words: even fields and odd fields shifted, a guard bit a field | 1024 |
| packed maximum and tileError moved into certifyTiles | 1024 / 1024 |

The cause was the spread's two vec3u accumulators, not the fused tileError. The packed maximum (`pwMaxReach`) landed,
and its host twin `framePlanMaxReach` is tested against the unpacked maximum.

So every new pipeline is at 1024 threads with 256 lanes a group, and no new launch is a single workgroup or a serial
lane. On paddleboats each launch is 512 groups striding 384 items. The chain's tiny launch (coarseBlocks, 5 groups
of a lane per block) is gone.

## 5. What is verified, and how

`tests/uniform-frame-plan-window.test.ts` (no GPU) holds a CPU mirror of each chain kernel and of each window
kernel. Each mirror follows its WGSL lane by lane, in barrier order. The fixtures are sparse G^3 page lattices, G = 3
to 5: stored, AIR and SOLID pages, AIR past the lattice, mixed h/4h tiles, whole 4h pages, quiet pages and shuffled
resident lists.
- **supportWindow vs dilate0..2.** Plane 3 is equal on every live tile for both shell rules, for (fine, shell) in
  {(2,1), (1,1), (3,0), (0,3), (1,2)} and mode 0, 2 and 3.
- **certifySpread + certifyTiles vs tileError..certify.**
  - For radii 2 to 10 (travel 0 to 32 cells): equal reach words, class words, regular and general lists as sets, and
    tile-bit words 0..15.
  - For radii 11, 17 and 1e8: every reach is saturated and no tile is regular, which is conservative.
  - Both lists are populated (more than 500 tiles each).
- **coarseAny** gives the same answer from the window's all-set summaries as from coarseBlocks' on 16000 random
  boxes, mixed hits and misses.
- **pwMaxReach** matches the unpacked maximum on 200k random pairs, with and without the stored bit.
- **Launches.** A mock encoder against the real `encode` / `encodeCertificate` code:
  - The chain issues 16 launches a frame, including coarseBlocks at 5 groups.
  - The window issues 7: seedHeld, supportWindow, localSpeed, certifySpread, certifyTiles, seed, supportWindow.
    Each non-counted launch is min(512, 2 x 288) = 512 groups.
  - A policy wider than 3 tiles takes the chain's dilation.
- **framePlanPass parsing.** The value parses, defaults to "chain", and unknown values throw, both directly and
  through `uniformNearSettled`.
- `npm run check:types`: clean for these files. The only errors are in a peer's uncommitted
  `lib/core/webgpu-render-worker.ts`.
- `npm run test:unit` (no GPU): 1732 pass, 1 fail, 374 skip. The failure is `tests/reservoir-particle-cost.test.ts`
  "merge cap protects neither the outer layer...". It is a source-text assertion on the peer-modified
  `uniform-narrow-band-flip.ts`, which I did not touch.

## 6. Functional runs (GPU runs 4 and 5, 30 steps, `framePlanPass "window"`)

`~/code/me/fluid-scratch/lane-b/func/run.sh` under `fluid-gpu-run.sh check`, rc 0. No timing gate.

| Run | Launches at step 30 | Drift | Max speed | Notes |
|---|---|---|---|---|
| Pure tank 8x8x4, drop=2 | 408 | -4.2e-6 | 0.05 m/s (at rest) | frame plan families: seed x2, supportWindow x2 (1024 groups), localSpeed, certifySpread, certifyTiles (512 each) |
| paddleboats, window | **480** | -0.00684 | 3.18 m/s | 1702 fine tiles, 147k particles |
| paddleboats, chain (run 1) | **489** | -0.00594 | 6.11 m/s | 2208 fine tiles, 103k particles |

**Launches.** Paddleboats drops 9 launches a frame (489 -> 480), exactly the mock encoder's 16 -> 7.

**Unverified: the paddleboats trajectories differ at step 30.** Max speed, particles and fine tiles all differ, and
the drift is 0.0009 apart. The certificate lists are equal as sets (CPU mirror), but their append order differs. The
order is also nondeterministic in the chain itself, since workgroup atomics run in any order. Paddleboats has driven
bodies, float atomics and pipelined frames, so some divergence between two chain runs is expected. There is no chain
A/A pair to bound it, and the five-run GPU budget is spent.

**Next gate before a default change:**
- paddleboats chain vs chain, then chain vs window, 30 steps each: drift and max speed within the A/A spread;
- seep 300 and letters 120 drift, chain vs window;
- a GPU equivalence probe if the spread is large: run both passes on one frame state and diff umSupport plane 3,
  speeds[T, 2T) and the two lists as sets.

Gate commands:
```
tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --pipelined --dispatches --steps=30 --every=30 '--values={"framePlanPass":"window"}' --out=...
tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-reservoir-bulk.ts --pure --px=8 --py=8 --pz=4 --drop=2 --steps=30 --every=30 --dispatches '--values={"framePlanPass":"window"}' --out=...
tools/gpu/fluid-gpu-run.sh check bash ~/code/me/fluid-scratch/lane-b/compile/run.sh   # compile seconds and thread caps
```

GPU runs used: 5 of 5.
1. the ledger;
2. compile and caps;
3. the cap bisection (compile only);
4. and 5. the two functional runs.

None was invalid (no rc 76/77/78).

## 7. Files

- `lib/methods/uniform/uniform-frame-plan-window.ts` (new): the `framePlanPass` parser, the window constants, the
  three entries' WGSL, and `framePlanMaxReach`.
- `lib/methods/uniform/uniform-mixed-frame-plan.ts`: the `pass` constructor argument, the window entries compiled
  only under "window" (the default module text is unchanged), the window launch tables, and the policy fallback.
- `lib/methods/uniform/uniform-near-settled.ts`: the `framePlanPass` field.
- `lib/methods/uniform/uniform-mixed-frame.ts`: passes `framePlanPass` to the plan.
- `lib/methods/uniform/uniform-narrow-band-method.ts`: the method param, default "chain".
- `tools/gpu-dispatch-counter.ts`: per-pipeline counts (`takeKernels`).
- `tools/probe-reservoir-bulk.ts`: `dispatchesByKernel`.
- `tools/profile-solver-construction.ts`: `--values=`.
- `tests/uniform-frame-plan-window.test.ts` (new).
