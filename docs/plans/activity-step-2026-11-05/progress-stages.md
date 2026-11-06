# Lane S progress: surface-region stages follow the active regions

One run per configuration (pipelined, `tools/gpu/fluid-gpu-run.sh timing`).
Main at start: cd6b085f, with another lane's uncommitted NB-FLIP merge-cap
hunks in the tree (they move letters drift from about -1.5% to -0.95%).

## Baseline (cd6b085f + tree)

| run | median ms | drift | notes |
|---|---|---|---|
| seep 300 | 29.43 | -0.064% | 421+69 dispatches |
| letters 120 (warm) | 50.87 | -0.950% | fine 7690, particles 882k |
| letters 120 (`coldSurface:"still"`) | 50.42 | -0.937% | fine 7882, particles 883k |

## S1: coldSurface "still" is the default

The A1 blocker (letters frames 31/32, h tiles then particles) does not
reproduce beyond repeat noise on this main. Per-frame check runs
(`probe-tier-budget.ts --every=1`, 44 and 34 steps):

| frame | warm run 1 h / particles | warm run 2 | still run 1 | still run 2 |
|---|---|---|---|---|
| 31 | 947 / 116.5k | 1015 / 116.6k | 1005 / 116.5k | 1007 / 116.5k |
| 32 | 990 / 125.2k | 967 / 140.4k | 967 / 139.2k | 967 / 139.0k |
| 33 | 1004 / 141.5k | 1004 / 140.0k | 1006 / 139.9k | 1013 / 139.9k |

The warm arm's own A/A spans the whole "divergence"; the still arm sits on
warm run 2. Suspects from the handoff (§5 A1), measured with `--heatState`:
- COLD surface regions omit at most 2.25e-5 h a step (frames 10-31), about
  1.6 ulp units of the hydrostatic round-off; no region sits near 7e-4.
- The head is inflated by the airborne letters (extent 8-10 regions against
  a 37-cell pool), so the bound is about 3.5e-3 h a step where a pool-depth
  head gives about 9e-4. Both are 40x or more above every COLD region's
  omission, so the tighter head changes no tier on letters (left open).

S1 checks (default, no override):

| run | median ms | drift | regions C/W/H | dispatches |
|---|---|---|---|---|
| tank 16x8x8 | **8.10** (from 18.9) | -3.7e-7 | 2048/0/0 | 114+37 |
| seep 300 | 29.59 (29.43) | -0.060% (-0.064%) | | 421+69 |
| letters 120 | 50.42 (50.87) | -0.937% (-0.950%) | | 419+69 |

Unit tests: one failure, `merge cap protects neither the outer layer nor
mass-saturated deep runs`, from the other lane's uncommitted NB-FLIP hunks
(it fails without this change too).

## S2: stages over the active region list (measured, not built)

Same disturbance, twice the tank (`--drop=8`, kernel timer, 60 steps):

| tank | tiers C/W/H at 60 | fine | particles | kernel-run median ms | distorted sum |
|---|---|---|---|---|---|
| 8x8x4 | 247/282/15 | 459 | 59.2k | 17.10 | 20.67 |
| 16x8x4 | 756/285/15 | 452 | 57.8k | 21.30 | 23.91 |

The extra 509 COLD regions cost +3.24 ms of distorted kernel time:
- pressure (Lane P) +1.81: arena solve +1.43, project +0.24, rhs +0.08;
- every other stage +1.43, spread over about 15 kernels, none above 0.18:
  classifyCoarse +0.17, the extension's coarse sweep (pair_sweepSeams,
  sweep kind 2 over every regular 4h tile) +0.16, 4h cache +0.10,
  bulkSelect +0.10, census decide +0.09, localSpeed +0.08, pageMark +0.08,
  then 0.02-0.07 each (prolong, layout classify, authority stamp,
  surfaceHeld, wetRegions, fillUnitHalo, momentumAffineFit, transferSeams).

Knockout before building (16x8x4 `--drop=8`, `--every=5`, env-gated, now
reverted): census decide over the active list's 64 tiles a region instead of
every resident block's 512, and the extension's regular-coarse sweep
(scalar and paired) under umActiveCoarse. The every-5 readback breaks
pipelining into a 5-step pattern (about 5, readback, 22, 24, 47 ms), so the
comparison is per phase:

| run | phase 2 | phase 3 | frames 10-25, phases 2+3 | fine / particles at 60 |
|---|---|---|---|---|
| baseline | 23.65 | 24.69 | 22.1 | 544 / 68.8k |
| knockout | 22.78 | 23.32 | 22.4 | 466 / 58.5k |

Where the disturbance matched (frames 10-25) the knockout gains nothing;
the later gap follows the baseline's larger disturbance (fine tiles and
particles), not the knockout. Both stages are under 0.3 ms, so they were
not built. The other non-pressure stages are smaller still, and each holds
its outputs differently (census counters and wet bits are cleared every
frame, the extension's slot arrays alternate parity, the 4h cache reads
the COLD term T). So converting them one by one is about 15 ideas of
0.02-0.17 ms each.

## Round 2 (2026-11-06): the corner disturbance's footprint and the non-quiet floor

All runs come from a clean `git archive HEAD` export (739271cb) with
node_modules symlinked. Each is a single 60-step run through
fluid-gpu-run.sh at 16x8x4 `--pure --cell=0.01`, pipelined with
`--every=60` (every=5 breaks the pipelining). The heat-reason breakdown
comes from a scratch-only probe diagnostic (`--heatReasons`), which reads
the heat state words, the trigger bits, the tiers and the still words back
each frame. It is not committed.

| run | median ms | late-half median | tiers C/W/H | fine tiles | particles | dispatches direct+indirect | kernel sum ms |
|---|---|---|---|---|---|---|---|
| rest | 4.69 | 4.60 | 1024/0/0 | 0 | 0 | 114+37 | 5.50 |
| drop=2 | 17.58 | 17.33 | 890/130/4 | 99 | 7.9k | 390+53 | 18.62 |
| drop=8 | 22.53 | 22.91 | 756/285/15 | 409 | 53.7k | 416+69 | 23.91 |

### (b) The non-quiet floor is fixed per frame, not per active item

Going from drop=2 to drop=8 adds 2.2x the WARM regions, 4x the fine tiles
and 7x the particles, yet costs only +4.9 ms (about 0.03 ms a region).
Extrapolated to zero activity, the floor is about 14-16 ms against the
4.7 ms idle step. The 13.1 ms of extra kernel time from rest to drop=2
splits as:

- pressure arena solve +2.17 (Lane P), which grows a further +2.37 by
  drop=8;
- 201 kernels, +8.0 ms, whose cost barely moves from drop=2 to drop=8
  (d8 <= 1.25 x d2): the fixed floor;
- 51 kernels, +2.9 ms, that grow with the disturbance (advectParticlesDeferred
  0.47 -> 0.94, arenaDistance and arenaPublishCanonical, transferSeams,
  momentumStep, remapFaces).

The workgroup counts per stage are almost identical at drop=2 and drop=8.
Examples: particle sort 25 dispatches / 25600 workgroups in both,
geometric fill 14.3k / 14.8k, extension 13.0k / 14.7k, surface advect
12.8k / 13.3k, remap 8448 / 8448, reconstruction 7168 / 7168. These
launches are direct and sized by capacity or by the domain. They do not
follow the active list.

The fixed 8 ms is mostly launch count:

- about 195 kernels at 0.01-0.06 ms each, about 6.4 ms in all;
- five larger sweeps, about 1.6 ms:
  - pair_sweepSeams 0.47 / 0.49;
  - classifyCoarse 0.43 / 0.49, a direct launch over min(live key groups, tiles/64);
  - geometryRegional 0.24 / 0.28;
  - depthCoarse 0.24 / 0.18;
  - prolong 0.225 / 0.225.

The round-1 knockout already moved two of these to the active list
(census decide and the extension's coarse sweep) and gained nothing. So
narrowing these launches will not remove the floor. The floor is the
number of dependent passes a non-quiet frame runs: about 280 more kernels
than rest. Removing it means fewer passes per frame, an architectural
change, not more active-list launches.

Caveat: with `--kernels=40`, the k-d2 wall time drops to about 10.5 ms
after frame 40 while the kernel sum stays 18.6 ms. Timestamp capture
changes the overlap, so the kernel times show where the time goes, not
the wall time.

### (a) Footprint by reason, frames 10-60

drop=8 (WARM regions by reason):

| frame | WARM | omission (own trigger) | still halo face / diagonal | held | HOT | of which detail |
|---|---|---|---|---|---|---|
| 10 | 76 | 31 | 27 / 17 | 0 | 8 | 4 |
| 30 | 184 | 95 | 74 / 12 | 0 | - | - |
| 60 | 285 | 130 | 106 / 34 | 15 | 15 | 13 |

drop=2 at frame 60: WARM 138 = omission 45 + halo 59 face / 30 diagonal +
held 4. HOT 4.

Reasons that never fire:
- travel reach: travel stays at 0.5-2.3 h, against the 16 h bound;
- fading heat;
- the HOT collar.

Particle heat makes only 1-5 HOT regions. The fine band (127-433 tiles at
drop=8) is mostly particle-heated core tiles (56-152), collar tiles
(37-104) and held tiles. Newly required (shape) tiles number 32-112.

The omission triggers:
- The dominant term is pressure at frame 10 and the surface term from
  frame 20 on.
- Per step, the omission is mostly 1e-3 to 1e-2 h.
- The live projection's floor is about 8e-4 h a step (accepted residual x dt).

| run, frame 60 | at or below floor | 1-3x | 3-10x | above 10x |
|---|---|---|---|---|
| drop=8 | 2 | 33 | 33 | 62 |
| drop=2 | 17 | 11 | 8 | 9 |

COLD surface regions do omit less than their budget:

| run | max omission a step (h) | max summed (h) |
|---|---|---|
| drop=8 | 6e-4, bound 6.7e-4 | 0.008, bound 0.01 |
| drop=2 | 2e-4 | 0.0027 |

Reading:

- At drop=8 the footprint is real motion. Inviscid waves cross the 2.5 m
  tank at about 5 m/s and never decay, and 95 of 130 omission triggers
  are at least 3x above the solver floor.
- The still halo is not wider than the physics it protects. COLD freezes
  every face and vertex incident to a COLD region, and the 4h vertices
  are shared with diagonal neighbours. A face-only halo would leave a
  moving region's edge and corner vertices frozen. The halo reads its
  neighbours' own evidence, not their tiers, so it does not chain within
  a frame. It chains over frames only when the wave itself arrives. Not
  narrowed.
- At drop=2, the still bound (about 4e-4 h a step at head 3) sits below
  the solver's own floor (about 8e-4 h a step). 17 of the 45 omission
  triggers are at or below the floor and 28 are below 3x. Budget mode
  subtracts the floor; the still rule does not. A still rule that
  compared against the floor would cut perhaps 20-30 drop=2 regions plus
  their halo, about 1 ms at 0.03 ms a region, and nothing at drop=8 or at
  rest. Not built: it would admit an omission up to the solver residual
  in still water, and it needs the letters 120 / seep 300 drift gates and
  a COLD-omission regression first. It is small next to the 8 ms fixed
  floor.

## Round 3 (2026-11-06): launch widths by live work, measured before building the rest

The judge's thesis: drop=2's 238,863 extra workgroups over rest are sized by
capacity or by the domain, and at that count they cost milliseconds. All
runs come from clean `git archive HEAD` exports (739271cb code) through
fluid-gpu-run.sh: one 60-step run per configuration, 16x8x4 unless named.

**Item 1, built: the particle sort launches from its producers' counts.**
- slabPlace writes slabSort's width (a workgroup a slab) and the record
  passes' width (256 records a workgroup) into free receipt words 27-28.
- A one-thread slabSortArgs writes the merge rounds' width into word 29,
  zero unless a cell is dense.
- The host copies the three x words into an INDIRECT buffer. slabSort, the
  23 merge rounds and slabSortFinish then launch from it: the merges run
  in a second pass after the copy, and they launch zero workgroups unless
  a cell is dense.
- Results: tests/uniform-particle-slab.test.ts passes on the GPU; drop=2's
  direct workgroups fall 293.8k -> 268.7k; drift -1.6e-6.
- Median 17.89 ms against 17.58. Same tree family, with W 143 vs 118:
  no gain.
- Not committed (time-neutral). The patch lives in the session scratchpad
  (sort-indirect.patch) and applies to both the index and the work tree.

**The cost of an early-exit workgroup, measured in the solver.** A scratch
wrapper multiplies every 1-D launch of 64+ workgroups by 4 (capped at
65535). Every such kernel strides or bounds-checks, so the extra
workgroups exit as the existing excess ones do.

| run | workgroups / frame | median ms | tiers C/W/H at 60 |
|---|---|---|---|
| drop=2, sort indirect | 268.7k | 17.89 | 876/143/5 |
| drop=2, every 1-D launch x4 | 990.9k | 18.53 | 874/149/1 |
| rest | 55.1k | 4.69 | 1024/0/0 |
| rest, x4 | 218.6k | 4.88 | 1024/0/0 |

- +722k workgroups cost +0.64 ms at drop=2, and +164k cost +0.19 ms at
  rest: about 0.9-1.2 ns a workgroup.
- drop=2's ~240k extra workgroups therefore cost about 0.2-0.3 ms of its
  +12.9 ms floor.
- A standalone microbenchmark agrees: 100 dependent launches of 1024 empty
  256-lane workgroups take 0.33-0.72 ms, against 0.20-0.39 ms at width 1.
  That is 1-3 us a launch for the 1023 extra workgroups.
- Sizing every launch by live work meets "5x fewer workgroups" but buys
  at most ~0.3 ms.

**The discriminator (drop=2 on 8x8x4 against 16x8x4).**

| tank | rest ms | drop=2 ms | jump | W at 60 | drop=2 workgroups |
|---|---|---|---|---|---|
| 8x8x4 | 3.42 | 13.72 | 10.3 | 118 | 259.0k |
| 16x8x4 | 4.69 | 17.58 | 12.9 | 118 | 293.8k |

- The step does fall with the tank, by 2.6 ms of jump over 512 extra COLD
  regions.
- Kernel time at drop=2, 8x8x4 -> 16x8x4: 15.00 -> 18.62 ms. Of the
  +3.6 ms, pressure is +1.8 (Lane P: solve +1.45, project +0.26, rhs +0.08).
- The other +1.8 ms is spread thinly:
  - pair_sweepSeams +0.16, classifyCoarse +0.15, bulkSelect +0.08,
    prolong +0.08, pageMark +0.08;
  - clearBand, the 4h cache and localSpeed +0.07 each;
  - layout scan +0.05, then <= 0.05 each.
- This is the same set round 1 found (+1.43 ms over 509 COLD regions):
  sweeps over every coarse tile or resident page. Their workgroup counts
  barely change with the tank (35k), so the cost is in-kernel loops over
  domain lists, not launch width.
- About 10 ms of the jump remains on the 8x8x4 tank with only 512
  regions. It is neither workgroup count nor domain size: it is the real
  work of the ~280 launches that a non-quiet frame switches on, at
  0.05-1.3 ms a stage.

So the build order the judge set (launch widths) moves the floor by about
0.3 ms. The domain-sized part (about 1.8 ms non-pressure per 512 COLD
regions) lives in the loops of the kernels named above. Those loops would
have to stride the active-region list instead of every coarse tile or
page. Round 1's knockout of two of them (census decide, the extension's
coarse sweep) gained nothing measurable, so each is worth <= 0.16 ms.

## Round 4, item 1: NB-FLIP grid kernels stride by their launch (30e038c8)

The round-3 knockout result (every NB grid launch at 64 changed the physics)
had a static cause, so no bisect was needed. Sixteen NB-FLIP kernels looped
over their items with a stride of `65536u`, which is 64 lanes x the fixed
1024-workgroup grid:
- activity, activitySpread, countInitial, seedInitial, diagnoseBefore;
- depthLists, surfaceCells;
- coarseCorners, couple, coarseBuilds, depthCoarse, redistanceCoarse;
- airSeeds, airSpread0-2, airGate, airClearCoarse.

At 1024 they cover every item, so no live scene dropped work. At a
narrower grid they skipped every item past their own lanes. Every other
NB grid entry already strides by `num_workgroups` or is certified
(uniformMixedCertifiedEntriesWGSL; surfaceClear is certified in
narrow-band-surface itself). The region launches loop, and the counted
launch is indirect. The only literal-width dispatch in lib is the NB grid.

The fix strides those sixteen by `64 x num_workgroups`. At 1024 this is the
same stride, so nothing changes in any scene today.

`tests/nbflip-launch-width.test.ts` has two checks:
- a source scan: no NB file strides by a literal grid;
- a GPU dam break whose NB-FLIP direct launches are capped at 3 workgroups
  must match full width within 2% in samples and within 0.5 in band centre.

| tree | full-width samples | 3 workgroups | result |
|---|---|---|---|
| HEAD 0dd7c9b3 | 8170 | 6266 | fails |
| with the fix | within 2% | | passes |

Knockout again, the NB grid at 64 on 8x8x4 drop=2 (60 steps):

| run | tiers C/W/H at 60 | particles | drift |
|---|---|---|---|
| grid 1024 | 370/138/4 | 7.9k | -4.3e-7 |
| grid 64, fixed | 372/135/5 | 9.7k | -6.2e-8 |
| grid 64, before (16x8x4, round 3) | 977/43/4 | 2.6k | -3.1e-7 |

drop=2 particle counts spread 6-10k run to run, so 9.7k is inside it.

Cross-scene gates on the fix. Tree: 0dd7c9b3 + the fix. That holds the
same NB files as 30e038c8 and none of the wet-cloth working-tree hunks.

| run | median ms | drift | particles | notes |
|---|---|---|---|---|
| seep 300, fix | 38.08 | +0.022% | 102.6k | 421+69 dispatches |
| seep 300, 0dd7c9b3 without the fix | 38.06 | +0.030% | 100.6k | 421+69 |
| seep 300, cd6b085f (round 1 baseline) | 29.43 | -0.064% | 70.3k | 421+69 |
| letters 120, fix | 63.37 | -0.916% | 993k | H 486, fine 8128 |
| letters 120, cd6b085f | 50.87 | -0.950% | 882k | H 390, fine 7690 |
| 8x8x4 rest, fix | 4.03 | 0 | 0 | 114+37, as 739271cb's 3.42 |
| 8x8x4 drop=2, fix | 16.14 | -4.3e-7 | 7.9k | 739271cb: 13.72 |

The fix is neutral: seep 38.08 against 38.06 without it.

Main itself has moved between cd6b085f/739271cb and 0dd7c9b3, by
+8.6 ms on seep (+46% particles, drift from -0.064% to +0.03%), +12.5 ms on
letters (+111k particles, +96 H regions), and +0.6 ms on the rest tank
with identical dispatches and workgroups. Not bisected here. Since
739271cb the only code commits are the three wet-cloth checkpoints
(8f09bd9f, a44ed3aa, 0dd7c9b3). They touch webgpu-uniform-reference.ts, the
pressure arena and mixture, uniform-mixed-frame and the solid record. The
particle and tier changes on seep and letters are physics, not timing noise.
The rest tank's +0.6 ms, at an unchanged dispatch count, could partly be the
machine's state.

## Round 4, item 2: the ~10 ms switched-on cost is not capacity-sized

Tree: 0dd7c9b3 + 30e038c8's fix. 8x8x4 drop=2, 60 steps, timing mode.

### Capacity halved

| run | median ms | late-half ms | pool blocks | particle capacity | binSlots | slotCapacity | memory |
|---|---|---|---|---|---|---|---|
| full | 16.14 | 16.49 | 832 | 3.39M | 53.4k | 127 | 3.15 GB |
| reduced (`--pool=416`, bricks 780, 2 per fine cell) | 15.38 | 15.28 | 414 | 0.85M | 6.2k | 95 | 1.57 GB |
| 8x8x4 rest | 4.03 | | | | | | |

The bricks can't be halved further: the coarse start's 6171 h tiles need
about 772. Cutting pool, particle and slot capacity by half to 8x changes
the step by 0.76 ms out of the 12.1 ms jump over rest. drop=2 particle
counts also spread 6-10k run to run, so that 0.76 ms is an upper bound. The
capacity hypothesis is rejected. A kernel-timer A/B of the two was dropped:
0.76 ms spread over stages sits near the timer's 65.5 us quantum.

### Where the jump goes (kernel timer, `--kernels=40`, fk-d2)

Kernel-timer sum is 17.7 ms against a 15.15 ms median, because of the 65.5 us
quantum. Stages that dispatch nothing at rest hold 241 launches and 9.64 ms
of timer time, 40 us a launch. Stages that also run at rest hold 201
launches and 8.06 ms; pressure is 3.58 ms of that and belongs to Lane P.

The live items are 379 coarse, 128 W and 5 H tiles, 113 fine, 9.7k
particles, 100 bricks and 8 slots.

| stage (switched on only) | launches | workgroups | timer ms | us/launch | worst kernel | loops over |
|---|---|---|---|---|---|---|
| mixed extension | 20 | 11856 | 0.95 | 47 | pair_sweepSeams 0.32 | launch counts of listed seam jobs |
| dynamic ownership census | 9 | 2446 | 0.73 | 81 | classifyCoarse 0.28 | coarse tiles (umCounts.y), replay or classify |
| mixed surface advect | 9 | 12288 | 0.66 | 74 | advectWalls 0.12 | 1089 lanes per wall face of resident blocks |
| NB advectParticles | 3 | 3072 | 0.61 | 205 | advectParticlesDeferred 0.53 | slab jobs of live particles |
| mixed remap | 9 | 8448 | 0.44 | 49 | remapFaces 0.18 | listed jobs |
| geometric fill | 7 | 13952 | 0.44 | 62 | geometryRegional 0.32 | region list |
| surface advect deferred | 6 | 6144 | 0.43 | 72 | advectDeferred 0.18 | not checked |
| NB coarse retirement | 3 | 3072 | 0.37 | 122 | depthCoarse 0.29 | listed coarse corners |
| NB transfer | 7 | 7168 | 0.37 | 52 | transferSeams 0.12 | one group per band seam tile |
| NB canonical metric and publication | 3 | 3072 | 0.35 | 118 | arenaPublishCanonical 0.17 | arena slots in use |
| body forces | 4 | 4096 | 0.34 | 84 | forcesRegularCoarse 0.15 | not checked |
| NB surface reconstruction | 7 | 7168 | 0.33 | 47 | couple 0.10 | listed coarse entries |
| local speed certificate | 8 | 2574 | 0.32 | 40 | localSpeed 0.19 | the owners its launch covers |
| phi resolve | 5 | 8192 | 0.27 | 53 | resolve 0.21 | not checked |
| layout build | 11 | 2064 | 0.26 | 24 | scan 0.05 | UM_BLOCKS x categories in one group: capacity, but 0.05 ms |
| NB particle sort | 24 | 24576 | 0.22 | 9 | slabSortMerge 0.12 | live slabs |
| 41 more stages | 123 | | 2.55 | 21 | | |

No stage carries more than a tenth of the jump. Every loop checked runs over
live or resident tiles, owners, lists or particles, not pool capacity. The one capacity loop found, the layout scan, costs 0.05 ms. The serial
single-group stages are small: layout build is 0.26 ms over 11 launches.
The NB particle sort's 24 near-empty launches cost 9 us each. So a launch
with no work is cheap, and the 40-120 us stages are doing real work on
few items: chains of dependent loads (owner -> topology -> page -> field)
per lane.

A chain microbenchmark (dependent random loads through a 16 MB buffer, 100
launches) backs that:

| dependent loads a lane | width 1-16 | width 256 |
|---|---|---|
| 0 | 4.6-5.2 us | 5 us |
| 16 | 15-28 us | |
| 64 | 18 us | 105 us |

The NB grid knockout to 64 workgroups confirms that workgroup count is not
the cost: 267k down to 206k workgroups a step, 16.14 against 16.59 ms.

What would halve it is structural, not a loop bound: fewer stages
in a non-quiet frame, or shorter per-item dependency chains in the
40-120 us stages (census, surface advect, body forces, coarse retirement,
canonical publication). No capacity-loop fix was made, since none has a
measurable share.

## Round 5, step 1 tested before building: address depth is not the bound

The resolved per-frame address table only pays if the switched-on stages
are bound by the owner -> topology -> page -> field load chain. Three
probes measured that before anything was built. The tree is HEAD 438e657f;
the scene is 8x8x4 drop=2, 60 steps, timing mode with `--kernels=40`.
Every run is one run.

| run | what changes | median ms | timer sum | switched-on timer ms |
|---|---|---|---|---|
| k5-base | HEAD | 16.02 | 19.05 | 9.82 |
| k5-deep | every page-table lookup gets one extra dependent load | 16.07 | 18.47 | 10.03 |
| k5-plan | the existing workgroup address plan (`umLoadAddressPlan`) on all 12 candidates | 15.86 | 17.10 | |
| k5-cq8 | classifyCoarse takes 8 coarse tiles a workgroup, not 64 | 15.12 | 18.60 | |

How the deep run works: in `PageTableWord`, the flat-window word `r` feeds a
second load, `buffer[table+(r&1023)]`. That value is masked by a runtime
zero (`arrayLength(&buffer)==0`), so the result is unchanged but waits for
the load. It deepens every page lookup in every kernel by one level.
Results are the same and only the latency changes.

- **Deeper chains cost little.** One extra level on every page lookup
  moves the switched-on stages by +2% (9.82 -> 10.03 ms) and the step by
  +0.05 ms.
  - The most sensitive kernels are classifyCoarse (+0.075 ms, 1.26x),
    arenaPublishCanonical (+0.044, 1.23x) and dilate (+0.066, 1.17x).
  - geometryRegional (1.02x), forcesRegularCoarse (0.97x), localSpeed and
    advectDeferred (1.02-1.07x) barely move.
  - A table that removes one or two levels can save the same few percent.
    It cannot take a third off the jump.
- **The existing staged plan is neutral.** It reads the 27 neighbour page
  words and vertex homes from workgroup memory. On the 12 candidate
  kernels the change is noise:

  | kernel | base ms | plan ms |
  |---|---|---|
  | forcesRegularCoarse | 0.193 | 0.237 |
  | forces | 0.094 | 0.112 |
  | momentumStep | 0.122 | 0.109 |
  | project | 0.758 | 0.736 |
  | advectFine | 0.131 | 0.122 |
  | transferProperties | 0.066 | 0.091 |

  The summed change is about +0.08 ms. This matches the plan's own note that
  it regressed the splash workload.
- **Serial loops inside the census group are not the cost.**
  classifyCoarse's queued error and depth loops run per group. Spreading
  them over 8x more groups made the kernel 0.28 -> 0.67 ms, so the
  per-group reset, barriers and census flush cost more than the serial
  loops saved.
- **A chain microbenchmark gives the cost per load.** A dependent random load
  costs about 110 ns at 64 KB, 240 ns at 1 MB and 210-400 ns at 16 MB
  (marginal, 64 -> 256 loads a lane, 8 or 64 workgroups). Even 256
  dependent loads a lane cost 40-100 us. A 40-120 us stage would need
  hundreds of dependent loads per lane, and the deep run shows page lookups
  are not most of them.

Step 2's per-tile threadgroup working set is not built either. Its own
premise was that step 1 works, and the deep run shows that removing
address levels does not reduce stage latency.

What these probes leave open is where a 40-120 us stage's time does go.
Candidates are low occupancy (a few workgroups on 32 cores), long per-lane
instruction streams, barriers, and global atomics or census flushes.
Telling them apart needs per-stage occupancy and limiter counters
(tools/profile-mini-dam-xctrace.ts with `--counters --isolate-pass-labels`).
Pointed at this frame, that is a long exclusive run, so it is left for the
judge to schedule.

## Round 6 (2026-11-06): profiled from outside the process

The new tools/profile-probe-xctrace.ts (05dd8ee7) recorded two runs with
Metal System Trace: 8x8x4 rest and 8x8x4 drop=2, each 60 steps, pipelined.
These are the skill's clean regime: no counters, no in-process probes,
never `--window`, and intervals joined to the encoder list by encoder-id.
The traced probes ran at a median of 3.50 and 16.29 ms (untraced: 4.0 and
16.1). An advance runs from one anchor encoder to the next; 57 advances of
each run were reduced.

| per advance | rest | drop=2 |
|---|---|---|
| wall, median (mean) | 3.22 (3.76) ms | 15.76 (18.46) ms |
| GPU busy, median (mean) | 3.05 (3.58) ms | 15.28 (14.86) ms |
| idle gaps of 2 us or more | 7.1, 0.18 ms | 9.0, 3.61 ms mean; about 0.5 ms in the median advance |
| GPU intervals (Metal kicks) | 7 | 9 |
| CPU encode work (Dawn-poll spin excluded) | 2.60 ms | 7.72 ms |
| of which Dawn/Tint, JS/V8, Metal driver | 1.18, 1.16, 0.19 | 4.25, 2.18, 1.07 |
| CPU spin in Atomics.load (Dawn event polling) | 1.60 ms | 2.79 ms |
| dispatches (f1 counter, untraced) | — | 442 (389 direct, 53 indirect) |

**Largest GPU tasks.** These are medians per advance. The Metal kick is the
finest unit with a GPU timestamp, and there are only five non-trivial kicks
per advance. Every remaining kick is a blit of 0.012 ms or less.

| drop=2 kick | ms | rest kick | ms |
|---|---|---|---|
| shared frame schedule, 3rd kick | 7.05 | [25 passes] compute copy .. pressure .. live block copy | 2.78 |
| shared frame schedule, 1st kick | 2.65 | [3p] compute copy .. region dynamics classify | 0.13 |
| [35p] mixed all-4h solid record .. mixed remap adopt | 2.00 | [7p] mixed all-4h solid record .. live block copy | 0.12 |
| [24p] mixed remap publish .. shared frame schedule | 1.91 | compute copy | 0.007 |
| shared frame schedule, 2nd kick | 1.65 | blits | < 0.005 |

The trace cannot name the stages inside the "Uniform shared frame schedule"
kicks. The sealed regional schedule puts every stage into one WebGPU pass
(uniform-compute-blit.ts), so 11.35 ms of drop=2 is three kicks of serial
dispatches. Metal System Trace has no per-dispatch timestamps, and the
shader timeline stays empty unless counters run.

**Verdict: real GPU work, latency-limited. Not CPU-bound, and not
gap-bound between kicks.**

- **GPU busy time is the wall time.** In the median drop=2 advance the GPU is
  busy for 15.28 of 15.76 ms. The 12.2 ms jump over rest is all busy time,
  in the switched-on kicks that rest does not run.
- **The CPU is not the bound yet.** Encode work is 7.7 ms per advance, under
  the GPU's 15.3, and in pipelined mode it overlaps the GPU. It costs about
  17 us of host time per dispatch (Dawn validation and tracking, NAPI, AGX
  encode), against 2.6 ms at rest. Once the GPU side halves, it becomes the
  bound.
- **Gaps between kicks are small.** There are about 8 encoder switches per
  advance at about 45 us each, which is about 0.5 ms.
  - One side defect sits on top. In 15 of 57 drop=2 advances, one gap of
    2-33 ms appears just before the next advance's first submission.
  - In those gaps the CPU submits late, yet the main thread is mostly off
    the CPU: 58 ms of running samples across 190 ms of gaps. So it is
    blocked, not spinning or encoding.
  - That causes the mean-over-median excess (18.5 against 15.8 ms) and the
    untraced p95 of about 30 ms.
  - Finding the blocking call needs a thread-state (System Trace) capture.
    It is parked; it is not the 12 ms.
- **One counter capture tells drain from work inside the kicks.** The
  capture was taken because the frame-structure trace cannot split a kick.
  - Setup: Blank template with Metal Application, GPU and Metal GPU
    Counters. It attached after construction to a probe held at a scratch
    pre-advance gate, then released all 60 drop=2 advances inside the
    window (median 16.43 ms under capture).
  - A first ungated attach, taken during construction, caught none of the
    advances. Metal pipeline creation under the counter instrument took
    about 4 min.
  - Results for uncontended kicks:

  | kick | ms/adv | occupancy | placement | ALU | read / write GB/s | top limiters |
  |---|---|---|---|---|---|---|
  | shared schedule, 3rd | 6.04 | 12.1% | 1.12x | 7.7% | 5.0 / 3.4 | ALU 8, LLC 8, MMU 3 |
  | shared schedule, 1st | 2.14 | 11.3% | 1.02x | 10.2% | 3.3 / 4.6 | ALU 12, LLC 8 |
  | [35p] solid record .. remap adopt | 1.81 | 9.2% | 1.03x | 9.5% | 4.6 / 2.8 | ALU 11, LLC 6 |
  | [24p] remap publish .. | 1.77 | 10.9% | 1.02x | 11.0% | 4.3 / 4.6 | ALU 12, LLC 10, MMU 8 |
  | shared schedule, 2nd | 1.56 | 9.0% | 1.04x | 7.7% | 3.8 / 5.3 | LLC 10, ALU 8 |

  - How occupancy samples spread over kick time: under 0.5% for 7.7% of
    the time and 0.5-2% for 6.0%. Between 5% and 40% for 71%.
  - So the barrier and drain between the ~440 serial dispatches is at most
    about 14% of kick time. That matches the ~4 us dependent-launch floor
    measured in uniform-compute-blit.ts (442 x 4 us = 1.8 ms).
  - The remaining ~85% is kernels that are resident but stalled. About an
    eighth of the machine's slots are filled, spread evenly over the four
    partitions, with ALU near 10%. Bandwidth is 1-2% of peak, and no
    limiter is above 12%.
  - Each dispatch has too little parallel work in flight to hide its own
    latency. Dispatches run one after another, so those latencies add up
    along a chain of about 440.

**Matching structural fix.**

The cost is serial latency per dependent dispatch. It is not bandwidth,
ALU or capacity, and no stage is over a tenth (round 4). The fix therefore
has to change how many dependent steps the switched-on frame takes and how
much independent work each step carries. Tuning any one kernel will not
do it. It must also cut dispatch count, because host encode at about
17 us per dispatch binds next.

- **Default proposal: run the switched-on frame as a block-local pipeline.**
  - One workgroup per active block runs that block's whole stage chain, with
    workgroup barriers and its working set in threadgroup memory.
  - The places where a block needs a neighbour's result from the same
    frame become the only global dispatch boundaries. Examples: remap
    publication, the pressure solve, and the face exchange before projection.
  - This replaces ~440 dispatches with a few tens. Different blocks' latency
    chains then overlap on the GPU instead of being serialised by global
    barriers.
- **Smaller step if that is too wide: dispatch by dependency level.**
  - Stages with no data dependency between them go into one dispatch,
    selected by workgroup id, so they fill each other's idle slots.
  - The schedule's existing stage list is the dependency graph to level.
    That gives the expected win before any kernel is rewritten: the depth
    of the leveled graph against the current 442.

Counters taken: one capture, the gated one above. A per-stage split inside
the kicks would need the shader timeline in launch mode. Under counters,
construction alone is about 4 min, so it was not taken. The dependency
leveling gives the stage-level answer offline.

## Round 7, step 0: the switched-on frame's stage graph (design input)

**Method.** The scratch recorder hooks the native WebGPU prototypes,
underneath the compute blit's shared pass.
- For one advance it logs every dispatch: pipeline, entry point,
  workgroups and the bound buffer ranges. It also logs each logical copy
  and clear.
- Each binding is split into reads and writes by what the entry point's
  reachable WGSL does with it: assignments, atomics, and `&` passed to a
  function.
- The command stream is then levelled. A command sits one level above the
  latest earlier command it conflicts with.
- Two run recordings were taken through `fluid-gpu-run.sh check`, at
  advance 29 of 8x8x4 rest and of 8x8x4 drop=2.
- Scratch tools: `tools/scratch-gpu-graph-recorder.ts` and
  `tools/scratch-stage-graph.ts` in the dev7 export.

**(a) Dependency depth.** Bindings count as whole ranges, so these depths
are upper bounds.

| advance | dispatches | copies + clears | depth, all hazards | depth, RAW only |
|---|---|---|---|---|
| rest | 140 | 31 | 119 | 116 |
| drop=2 | 475 | 105 | 390 | 350 |

Even with every write-after-read and write-after-write hazard renamed
away, dependency leveling removes at most 26% of the launches. The frame
is a real chain. Merging independent stages into one dispatch cannot
deliver the win.

**Where the 475 drop=2 dispatches sit.** Command buffers in submission
order:

| submission | dispatches | content |
|---|---|---|
| 0 | 151 | census, page pool, BULK16, layout build, h slots, remap, frame plan, particle bins and slab sort (22 merge passes) |
| 1 | 45 | surface advect, phi resolve, depth/air plans, the redistance arena |
| 2 | 36 | geometry, momentum, P2G transfer, forces |
| 3 | 243 | ownership transfer, authority, then the pressure arena: 193 dispatches, against 104 at rest; then projection, publish, extension |

The 3rd shared-schedule kick (7.05 ms) is submission 3, or submissions 2
and 3 together if the hardware coalesced them.
- About 80% of submission 3 is the pressure arena multigrid.
- drop=2 runs 4 Krylov iterations, each with a V-cycle whose three coarse
  levels repeat three times: (down3, down4, tail, up4, up3) x3. Rest
  runs 2.
- At rest the arena's 104 dispatches take about 2.6 ms, so at drop=2
  pressure costs about 5-5.5 ms of the 7.05.

**(b) Global boundaries.** Each dispatch is classified by stage role.
WGSL was spot-checked; it was not proven per kernel.

| class | dispatches | |
|---|---|---|
| pressure multigrid: smoothers, residuals, restrict/prolong, coarse levels | 161 | neighbour stencil at every level |
| pressure Krylov: reductions, verdict, final | 18 | global reductions |
| pressure setup and tables | 15 | |
| global allocation, scan, sort or claim | 92 | census, page pool, BULK16, layout builder, slots, regional plan, particle bins and sort |
| neighbour stencil, gather or scatter outside pressure | 121 | advect, resolve/twins, remap faces, dilate/spread, redistance arena, momentum, P2G deposit, forces, buildRhs/project, extension sweeps |
| pointwise per block | 68 | authority stamps, ownership transfers, page/slot init, region dynamics, live copies, publishes |

Almost every physics stage reads a neighbour block's output from the same
frame. The pointwise stages form 23 separate runs; the longest is 12
dispatches, and the next are 7, 6, 5, 4 and 4.

**(c) What one-workgroup-per-block chains can and cannot reach.**

- **Pointwise runs only:** 68 dispatches become 23 launches. That removes
  45 launches. These are short kernels, so the gain is below about 1 ms.
- **Stencil chains:** the one-pass design must use halo-overlapped tiling.
  Each workgroup computes k chained stencil steps on its block plus a halo
  of width k, all in threadgroup memory. That replaces k global boundaries
  with one, at the cost of redundant halo work. Candidates and expected
  launches:
  - **Pressure (Lane P's files).**
    - A workgroup-resident coarse solve replaces the 15 coarse-level
      launches in each V-cycle with 1: 60 down to 4.
    - Halo-fused fine smoothing replaces each relax x3 and nativeRelax x2
      group with 1: about 24 fewer.
    - Krylov reductions stay global.
    - Expected: 193 down to about 110.
  - **Extension hierarchy:** restrict/sweep/prolong x3 and the paired
    sweeps, 18 down to about 4.
  - **Surface advect:** 15 down to about 4.
  - **Redistance arena:** about 14 down to about 5.
  - **Momentum, then P2G:** 15 down to about 5.
  - **Particle slab sort:** each long cell is merged inside one workgroup,
    so 24 launches become 2.
- **Total for the design:** 475 dispatches down to about 300 launches (half
  of the 175 removed come from pressure). At the measured ~25-35 us a
  launch, that is about 4-6 ms of drop=2's 15.3.

## Round 7 option 1, knockout before the build (2026-11-06)

The coordinator asked for a timing bound before the full restructure. The
bound comes from removing the coarse launches on drop=2. All runs are 8x8x4
`--pure --drop=2`, 60 steps, from a clean export of d4769214.

**Fixed-cycle timing.** The knockout tree forces 2 V-cycles every frame
(`bMet` false, so no early exit). Every variant then encodes and runs the
same outer work.

| variant (2 forced cycles) | steadyMedian ms | step-60 residual |
|---|---|---|
| tailCycles 3 (HEAD) | 13.14 | 0.038 |
| tailCycles 1 | 12.53 | 0.032 |
| tailCycles 5 | 13.63 | (p95 noisy) |

- One 8h tail cycle (pDown3, pDown4, pTail, pUp4, pUp3: 5-6 launches) costs
  0.15-0.25 ms a frame at 2 outer cycles. That is 75-120 us of wall time
  for each tail cycle in each V-cycle.
- The whole tail (3 cycles) is therefore about 0.5-0.75 ms a frame at 2
  cycles, and less at drop=2's real ~1.8.
- Levels 1 and 2 (pDownA, pDownR, pDownB, pDown2, pBulkLift, pUpB, pUpA: 7
  launches a cycle) add at most about 0.3-0.4 ms by the same per-launch
  scale.
- **Bound:** removing every coarse launch is worth under about 1.1 ms. The
  resident tail can remove only part of that, so the bound for the
  restructure is under 1 ms.
- A direct knockout (tail or all coarse levels skipped) diverged within
  about 15 frames: drift -0.4 to -0.5 by step 60, and the band grew. Its
  medians (18.2 and 17.2 ms) measure the exploded scene, not the solve.

**Cycles per frame do not follow the coarse solve's accuracy.** These are
unpipelined traces (`probe-tank-pressure-trace --drop=2`); the counts are
for frames 21-60.

| variant | cycles | m1 (history[1] geomean) |
|---|---|---|
| HEAD | 71 | 0.073 |
| tailCycles 1 | 85 | 0.174 |
| tailCycles 5 | 60 | 0.055 |
| tailCycles 9, every up4 red-black | 78 | 0.085 |
| tailCycles 15, red-black, 96 32h sweeps | 74 | 0.079 |
| resident 16h+32h (prototype) | 79 | 0.088 |

- A near-exact 8h solve (9 or 15 convergent tail cycles) does not lower the
  frame's cycles. The count is noisy, roughly +/-10, because m1 sits on the
  0.05 target.
- tailCycles 1 saves 0.6 ms at fixed cycles, but costs 14 cycles (+20%), so
  it is a net loss.

**Resident prototype (not landed).** pResident is one workgroup (1024
lanes):
- the 16h rows sweep red-black over the whole pool, with their iterate in
  the pool's Y;
- the 32h rows solve in workgroup memory (20 bytes a page, at most 1280
  pages);
- pDown3 becomes residual-only.

It replaces pDown4, pTail and pUp4. It compiles and converges (drift
-6e-7), but runs 79 cycles. The kernel timer's per-dispatch resolution is
65.5 us, so it cannot rank these sub-tick kernels.

**Conclusion.** The resident coarse solve is bounded below 1 ms of drop=2's
~13 ms, and it buys no cycles. The 2 ms target is not reachable through the
coarse levels.

## Round 8, fine-level contraction on drop=2 (2026-11-06)

**Setup.** Unpipelined traces (`probe-tank-pressure-trace --drop=2
--diagnostics`, 8x8x4, 60 steps), clean exports of d4769214. The tables
summarise the residual after cycle 1 (diagnostics row 1) over frames 21-60.
Each entry is the class's largest residual (geomean over frames) and its mean
count of rows past the 0.05 target.

**Row classes** (pDiagClass, pDiagNative):
- band: 1 seam, 2 wall, 7 free surface (an open or ghost face, or a non-unit
  band face);
- native 4h: 4 interior, 5 seam, 6 wall.

| variant | cycles | m1 | c7 surface | c1 band seam | c5 native seam | c4 native | argmax (frames) |
|---|---|---|---|---|---|---|---|
| HEAD (3 band sweeps, 1 native) | 75 | 0.078 | 0.070/36 | 0.064/20 | 0.047/2.6 | 0.043/2.1 | 7:23 5:10 1:6 |
| band sweeps 2 | 83 | 0.110 | 0.108/130 | 0.076/20 | 0.033/1.9 | 0.030/1.6 | 7:36 |
| band sweeps 5 | 57 | 0.052 | 0.023/0.6 | 0.032/1.6 | 0.048/3.3 | 0.038/2.3 | 5:29 1:9 |
| band sweeps 8 | 55 | 0.045 | 0.008/0 | 0.022/0.9 | 0.042/1.9 | 0.032/1.0 | 5:33 |
| native sweeps 2 | 80 | 0.104 | 0.103/141 | 0.087/64 | 0.018/0.2 | 0.022/0.6 | 7:38 |
| band 5, native 2 | 69 | 0.067 | 0.049/27 | 0.065/38 | 0.024/0.6 | 0.028/1.3 | 1:32 |
| global red-black band sweep (halo red rows) | 83 | 0.081 | 0.063/26 | 0.064/21 | 0.058/4.2 | 0.047/3.1 | 5:16 1:13 7:10 |
| natives between band sweeps (2 + native + 1) | 83 | 0.097 | 0.082/66 | 0.072/27 | 0.065/5.7 | 0.049/4.0 | 7:25 5:15 |
| natives before band sweeps, band 5 | 61 | 0.055 | 0.031/3 | 0.025/1.4 | 0.053/3.0 | 0.028/1.0 | 5:34 |
| coarseScale 0.4 | 78 | 0.085 | 0.078/80 | 0.071/40 | 0.047/2.4 | 0.042/1.9 | 7:32 |
| coarseScale 0.65 | 79 | 0.092 | 0.086/115 | 0.078/52 | 0.043/2.0 | 0.037/1.2 | 7:32 |

**Findings.**
- **Which rows hold the residual:** above the target after one cycle are
  the band's free-surface rows (class 7), then the band and native seam
  rows. The 2h, BULK16 and wall classes stay under it.
- **The tile-block smoother is not the cause:** the band sweep is red-black
  inside a 4h tile and block Jacobi across tiles. A global red-black band
  sweep (each black row recomputes its cross-tile red neighbours from plane
  pSrc, deterministic in one launch) leaves class 7 at 0.063.
- **No flip at the fine levels, partly unmeasured:** there is no
  sign-flipping mode like the 16h one before 1058cc0b, because the global
  sweep would have removed it. The diagnostics record no per-row sign ratio
  for band rows, so a flip there is not directly measured. BULK16 rows
  still flip by more than 1/3 after cycle 1 in 1-8 frames of 40 on drop=2
  (largest 0.5-1.2), but they are never the largest residual.
- **The coarse operator does not move the surface rows:** coarseScale 0.4
  or 0.65 leaves class 7 where it is, and so did a near-exact 8h solve
  (round 7). Only fine smoothing reduces the free-surface rows: 5 band
  sweeps take them from 0.070 to 0.023.
- **The next limit is the h/4h seam:** band and native seam rows trade off.
  - More native sweeps lower class 5 (0.047 to 0.018) but raise the band
    surface and seam rows.
  - More band sweeps leave class 5 at 0.042-0.053.
  - Reordering (natives first, or between the band sweeps) does not
    remove it.
  - The native 4h row's seam couples to 16 h cells, and the alternating
    smoothers leave the interface's residual on whichever side ran first.
    That interface operator is what limits a one-cycle solve.

**Timing.** Pipelined, steadyMedian, one run each:

| band sweeps | steadyMedian ms | live particles at step 60 |
|---|---|---|
| 3 (HEAD) | 14.32 | 8198 |
| 4 | 13.34 | 5748 |
| 5 | 14.08 | 8584 |

The scene state diverges between variants: live particles at step 60 vary by
up to 50%. A single-run difference below about 1 ms is therefore not
resolved. The estimate from cycles is that 5 band sweeps save about 0.35
cycles a frame, minus 7 more band launches (16 us each, net): about 0.5 ms.
That is under the 1 ms gate, and nothing has landed.

## Round 9: the seam block and the coarse free-surface operator (not landed)

**What was tried.** Two changes, each behind a knob in a dev tree. The
patch is `seam-block-surface-series.patch` in the agent scratchpad.

- **Seam block (pNativeRelax, same launch).** Each native 4h seam row
  relaxes together with the h cells on its seam faces. Only cells that
  qualify join: active, exactly one seam face, no wall, no inside clamp.
  - The row solves exactly against those cells' Schur complement, then
    writes them from its new value. No new launch.
  - It is race-free because a joining cell belongs to exactly one native
    row.
- **Coarse free-surface operator (pRebase, level 1).** A fixed-zero face
  that leaves an aggregate is a free-surface ghost/open face, or a band
  link to an inactive row.
  - The aggregate's centre lies half a cell further from that boundary
    than its child's centre. So the Dirichlet coefficient becomes the two
    resistances in series: `c*u/(u+0.5c)`, with u the unit coefficient.
  - This is the rediscretised 2h Dirichlet term. The Galerkin sum `c`
    over-weights it by `(theta+0.5)/theta`, which is 2x at theta = 0.5.
    The interior links are already rediscretised by coarseScale 0.5.
  - This is the inconsistency the round-8 coarseScale test could not see,
    because coarseScale scales the links and not the Dirichlet terms.
  - Scaling all of `part - outward` instead (FLUID_SURFACE_SCALE 0.5) also
    scaled the BULK16 links that gCoarseChild leaves in the diagonal. It
    flipped the BULK16 rows (largest flip 3.95).
- **A level-2 version** (the child's excess `d - sum k`, one more cell in
  series, applied at pDownA only) gains a little more.

**Cycles** (check mode, one run each). Drop=2 counts frames 21-60; m1 is the
geomean of the residual after cycle 1:

| variant | drop=2 cycles | m1 | seep 300 cycles/frame | letters 120 cycles/frame |
|---|---|---|---|---|
| HEAD | 75 | 0.078 | 3.58 | 2.98 |
| seam block | 78 | 0.082 | | |
| surface series (L1) | 60 | 0.062 | | |
| both | **57 (-24%)** | **0.047** | 3.56 (-0.5%) | 2.90 (-2.7%) |
| both + L2 series | 55 | 0.049 | 143 vs 144 (60-step trace) | |

- **Drop=2 with both changes.** After cycle 1 every class is at or below
  0.041: band seam 0.064 -> 0.026, surface 0.070 -> 0.026, native seam
  0.047 -> 0.041.
  - BULK16 flips over 1/3 occur in 1 frame of 40 (largest 0.49).
  - tests/pressure-tail-red-black.test.ts still passes with both knobs on.
  - Drift is unchanged on all three scenes.
- **Seep and letters do not move.** A class split on seep separates band
  rows with a below-unit band face (solid apertures) from the surface rows.
  After cycle 1:
  - aperture rows are largest in 18 of 40 frames (3.5x target), walls in
    10 (1.8x), surface in 6 (2.1x, 173 rows);
  - seep starts each frame at about 100x target and runs 3-4 cycles at
    rho1 0.05 and rho2 0.13;
  - letters runs 4 cycles in 69-78 of 120 frames.

  The seep and letters cycles are therefore set by the solid-aperture
  and wall rows, not by the seam or the surface operator.

**Timing.** Drop=2, pipelined, steadyMedian, one run each: HEAD 14.51 ms,
both changes 14.41 ms. That is within single-run noise (live particles
8.9k vs 9.9k).

**Verdict.** Acceptance needs at least 20% fewer cycles on two scenes; this
reaches only drop=2. Not landed. The regression test
(tests/pressure-seam-cycle.test.ts in the patch) asserts the drop=2 cycle-1
geomean <= 0.05 on an 8x4x4 tank. It was not committed with nothing to
guard.

**Next.** The seep and letters limit is the aperture (below-unit) band
faces and the wall rows. Two questions to answer there:
- how the level-1 aggregate weights a partly open face;
- whether the fine smoother's wall active-set handles it.

### Round 9 landing: a05b6782 (surface-operator fix alone)

- **The change.** pRebase bakes `rFixed(c) = c*u/(u+0.5c)` for every
  fixed-zero face that leaves a level-1 aggregate. The seam block is left
  out.
- **Regression test.** tests/pressure-seam-cycle.test.ts runs an 8x4x4 tank
  with a 2-cell drop for 40 steps and takes the cycle-1 geomean over frames
  21-40:
  - HEAD 0.122 (frames 21-30 in a 30-step run: 0.167);
  - with the fix 0.080 (0.107);
  - asserted <= 0.1.
- **Gates.** One run each:

  | scene | steadyMedian HEAD | with fix | drift |
  |---|---|---|---|
  | seep 300 | 30.04 ms | 29.36 ms | 2.1e-4 / 1.6e-4 |
  | letters 120 | 54.62 ms | 50.95 ms | -9.09e-3 / -9.05e-3 |

  - tests/pressure-tail-red-black.test.ts passes.
  - check:types is clean.
  - test:unit has one failure in the main checkout, reservoir-particle-cost
    "merge cap". It passes in a clean HEAD export both with and without the
    fix, so it comes from uncommitted hunks in the shared checkout.

## Round 10: the seep's slow rows (parked)

**Parked (2026-11-07).** After three rounds on seep convergence, cycles stay
at 144 over frames 21-60 before and after every change tried.
- **Known:** the rows that set seep's 3-4 cycles a frame are fluid rows
  beside a voxel face of the wall or throat. Their neighbour across the face,
  at the CM11a dual-cell weight V = 0.5, is an inside-solid band row. That
  row is active and free (not at its clamp, and it barely toggles), and
  dangling: the half face is its only open link. The face always crosses a
  level-1 aggregate boundary, because the wall planes lie at even cells.
  These rows contract only 5-10x a cycle, against about 46x for the other
  rows on the first cycle.
- **Ruled out:**
  - a representation bug: the 0.5 is the documented CM11a weight, and
    tangential faces beside a wall are fully open;
  - candidate 3, clamped rows hidden from the coarse levels: pRebaseC
    already rebakes level 1 from the clamp flags every cycle, and the
    neighbours are free;
  - operator-dependent prolongation;
  - P_SCALE misapplied to the dangling aggregates: the derived link between
    pressure centres left the contraction unmoved.
- **Remaining:** candidates 1 and 2 (the inner fixed-zero weight, and
  rFixed's aperture) were not reached. Only the surface operator fix,
  a05b6782, landed.

**Per-cycle class breakdown on seep.** Check-mode trace, 60 steps, frames
21-60, geomean largest residual / rows past target. "Aperture" means band
rows with a below-unit band face.

| measure | walls | surface | aperture | seam (band) | largest in |
|---|---|---|---|---|---|
| cycle 0 (start) | 81 / 374 | 69 / 1005 | 28 / 112 | | walls 22 of 40 frames |
| cycle 1 | 1.8 | 2.1 | 3.5 | | aperture 18 of 40 |
| cycle 2 | 0.07 | 0.15 | 0.67 / 34 | 0.11 | aperture 28 of 40 |
| cycle 3 | | | 0.07 / 10 | 0.02 | aperture 31 of 40 |

- Walls and the surface contract about 30x a cycle. The aperture rows
  contract only 5-10x a cycle (0.12, 0.19, 0.10), and a few dozen of them
  hold the stop for cycles 3-4.

**Ruled out:**
- **Operator-dependent prolongation** (pProlong0's far-parent weight scaled
  by the opening of the face toward it, a closed face none): 144 cycles vs
  144.
- **Aggregates straddling the wall:** the wall is 16 cells thick on 16-cell
  boundaries, and the 2x2 throat is aligned to one level-1 column.
- **Level-1 aperture links:** these are consistent with rediscretisation,
  since coarseScale times the summed open area gives the 2h conductance.
- **Wall faces:** these are removed from the level-1 diagonal (Neumann)
  unless clamped.

**Open candidates:**
- Inner fixed-zero faces (a child whose sibling is air) still enter the
  aggregate at their Galerkin weight. A huge 1/theta at a shut face then
  suppresses the whole aggregate's correction.
- rFixed's half-cell resistance ignores the face's aperture (`u`, not
  `u*volume`).
- The inside-solid rows' clamp, which no coarse level sees.

The next step is to locate the slow aperture rows: throat walls, mouth, or
reservoir wall face.

### Round 10: where the slow aperture rows are

Aperture rows were re-binned by position and by aperture ratio (c/u of the
below-unit band face), one seep trace per split (60 steps, frames 21-60). The
classes at each split were:
- **x relative to the wall:** upstream (x < 240) has none. The throat
  (240-255) is at most 0.014 at cycle 0. The wall's downstream face and mouth
  (256-259) are negligible. Every slow row is downstream, at x >= 260.
- **Height:** none at y <= 1, so the floor is not the source.
- **z:** the side-wall layers z <= 1 / z >= 62 hold the slow rows. The
  interior is 1e-4.
- **Aperture ratio and layer:** all of them are in the wall-adjacent cell
  layer (z = 0 or z = 63), with a tangential face open at 0.45-0.55: a half
  aperture. There are none below 0.45, and z = 1 / z = 62 rows are
  negligible.

So the cycles 2-4 limit on seep is about 30 rows in the cell layer against
the container's side walls (z = -1 and z = 64 voxel fills). Each has
tangential faces open at one half. Their residuals after cycles 1, 2 and 3
are 3.4, 0.57 and 0.055 (largest in 27, 28 and 31 of 40 frames).

A wall voxel occupies [-1, 0), so the z = 0 cell's tangential faces span
[0, 1) and are geometrically fully open. A half aperture means the solid
representation places the container wall surface at the first cell's
centre. That is a geometry question as well as a solver one.

**Candidates in that light:**
1. **Inner fixed-zero faces:** these do not apply. The wall face of a
   z = 0 or z = 63 cell is its outward side at level 1 (parity), so rFixed
   already covers it.
2. **rFixed ignoring the aperture:** this applies only if the face toward
   the wall is a half-open link to an inactive row. That changes 0.4u to
   0.33u, which is too small to expect a 20% cycle cut.
3. **Clamped rows:** this is the live one if the z = -1 / z = 64 cells are
   inside-solid band rows (liquid phi continues through walls, 728dd36b).
   Their clamp toggles, and the correction rebase then fixes them at zero.

**Next measurement:** read these rows' six face kinds and coefficients, and
whether the z = -1 neighbour is an active inside row.

### Round 10 correction: the position bins above are invalid

The position-based bins in "where the slow aperture rows are" are wrong.
- **What happened:** the classifier computed a band row's cell as
  `ubaOrigin(ubaBrick(slot)) + local`. In the cycle-measure modules (restrict
  and residual) that does not give lattice coordinates. Every band row came
  out at z > 65 in a 64-deep box, although the box origin is (0,0,0)
  (address.boxPage and pool.box.origin, read from the solver).
- **Why it looked like signal:** every x, y and z split put all aperture
  rows in the same bin. The bins shared class slots with the native classes
  4, 5 and 6, which made the splits look informative.
- **Withdrawn:** downstream, side-wall layer, the z = 0 tangential-face
  aperture, and the solid being half a cell thick.

**What still holds.** These tests use no coordinates:
- the slow rows are band rows whose first below-unit band face has c/u in
  [0.45, 0.55);
- that face is normal to x or y, not z.

**The geometry convention, from umPressureFaceV:**
- A face between a voxel cell and an open cell is `0.5*(open + open)`. That
  is 0.5 against a full voxel: the CM11a dual-cell V, documented as "a wall
  page is voxels: its face holds half the open fraction of the cell this
  side of it".
- A face neither of whose cells is a voxel is 1.0. umSolidFaceClear applies
  when the tile is clear; otherwise it is the dual unit box's open fraction.

So a tangential face beside a voxel wall is fully open. A half-open band
face means the row's neighbour across that face is a voxel cell: a band row
inside the solid, since liquid phi continues through walls.

**Inference, not yet measured:** the slow rows are probably the rows beside
the seep wall's (or the throat's) voxel faces, coupled at V = 0.5 to clamped
inside-solid rows. That is candidate 3. The next measurement is the
neighbour's kinds across the half face (active? inside?). It needs no
coordinates.

### Round 10, candidate 3 measured: the neighbour is inside the solid but free

The classifier now reads the neighbour across a row's first half-open band
face (c/u in [0.45, 0.55)) through pNear0 in pRestrict0, so it uses no
positions. Seep ran for 60 steps; the figures are from frames 21-60, as a
geometric mean of the largest residual per class, then the number of rows
past target per frame.

**Run nb1 (does the neighbour clamp?).** The classes are for rows outside
the solid, by their neighbour across the half face:
- c1: inside, active, at its bound (x + base <= 0);
- c2: inside, active, free;
- c3: inside, inactive;
- c4 and c5: not inside;
- c6: the row is itself inside;
- c0: every other row.

| after cycle | c0 | c1 | c2 | c6 | c2 sets the max |
|---|---|---|---|---|---|
| 0 | 81.9 / 1410 | 11.1 / 9 | 27.4 / 112 | 3.64 / 107 | 14 of 40 |
| 1 | 2.01 / 308 | 0.575 / 5 | 3.45 / 85 | 0.373 / 52 | 26 of 40 |
| 2 | 0.222 / 159 | 0.052 / 3 | 0.645 / 44 | 0.070 / 16 | 33 of 40 |
| 3 | 0.033 / 68 | 0.020 / 2 | 0.069 / 20 | 0.008 / 5 | 34 of 40 |

Inside rows whose clamp state (P_CORRECT) changed: 4.4, 0.7 and 0.5 a frame
after cycles 1-3.

**Run nb2 (is the neighbour dangling?).** Rows with an inside, active and
free neighbour are split by whether the half face lies inside the level-1
aggregate, and whether the neighbour has one face that is neither closed
nor Neumann (a dangling row) or more than one:

| after cycle | c0 | out of aggregate, dangling | out of aggregate, more than one face | inside the aggregate |
|---|---|---|---|---|
| 0 | 105 / 1445 | 25.1 / 102 | 7.16 / 170 | none |
| 1 | 2.26 / 305 | 3.42 / 79 | 1.11 / 12 | none |
| 2 | 0.218 / 155 | 0.651 / 39 | 0.258 / 4 | none |
| 3 | 0.035 / 66 | 0.064 / 18 | 0.024 / 1 | none |

**Reading.**
- **Candidate 3 is not the cause.** The slow rows' neighbour is an inside,
  active row, but its clamp is not active and barely toggles.
- **Every half face crosses the level-1 aggregate boundary.** The wall and
  throat planes lie at even cells.
- **The neighbour is a dangling row.** Its only open link is the half face.
  Its aggregate holds only such rows, so the coarse cell couples only
  outward. Under P_SCALE that cell's diagonal and coupling are both half
  the Galerkin sum. Its coarse jump against the fluid aggregate is then
  twice the fine row's (r/c). Its transmitted flux stays exact.
- **Hypothesis, not tested:** the trilinear pProlong0 then spreads a quarter
  of that overshoot onto the fluid rows beside the wall.
- Per the judge's rule, no fix has been fitted. Candidate 3 is reported as
  ruled out.

### Round 10: the coarse link between the pressure centres does not move the slow rows

**The rule.** This is the general form, not the Galerkin shortcut. In
pRebase, a level-1 link out of an aggregate is the series sum of three
parts, between the two aggregates' pressure centres along the face normal:
- the face's own coefficient c;
- half a cell of open link, 0.5/u, behind each side whose child has an
  active link to its second layer along that axis.

It is stored as g/P_SCALE, the way seamCoarse stores its link, so that
rCouple and rDiag are left unchanged. Under the same rule, a fixed-zero face
out of the aggregate takes rFixed only when that second layer exists, and
the face's own c otherwise. The rule reduces as it should:
- two full aggregates with c = u give u/2, which is P_SCALE c;
- two one-layer aggregates give c, the Galerkin weight;
- the fixed-face form reduces to rFixed.

It costs one extra kinds load per outward active band link.

**Measured on seep, 60 steps, with the nb2 classifier.** Frames 21-40; the
"dangling" class is a fluid row whose half face crosses the aggregate
boundary to a dangling inside row.

| after cycle | dangling class, before | dangling class, with the rule | other rows, before | other rows, with the rule |
|---|---|---|---|---|
| 0 | 25.1 | 25.1 | 105 | 93.3 |
| 1 | 3.42 | 4.03 | 2.26 | 2.32 |
| 2 | 0.651 | 1.10 | 0.218 | 0.362 |
| 3 | 0.064 | 0.089 | 0.035 | 0.037 |

Cycles over frames 21-60 were 144 before and 144 with the rule. The
dangling class's contraction did not move toward the other rows' (it was
6x then 4x a cycle, against 7x then 5x before), and cycle 2 got worse for
every class.

**Reading.** The jump from the halved link on the dangling aggregate does
not set the slow rows' rate. Per the judge's rule, I am reporting and
stopping without stacking knobs. Nothing has landed: the rule exists only
in the scratch tree, under FLUID_SERIES.

## Round 11, step 1-2: the block-local critical path on drop=2 (estimate fails the gate, not built)

**Measurements.** All are from a clean export of d438a6d8, on 8x8x4.
- The dispatch graph is a scratch recorder of one advance, at step 29, in
  check mode. rest has 140 dispatches; drop=2 has 432, at a RAW dependency
  depth of 307.
- The Metal System Trace is one capture of drop=2 with
  tools/profile-probe-xctrace.ts: 30 steps, pipelined, 27 advances reduced.
  - Per advance, the wall median is 13.90 ms (mean 14.79) and GPU busy is
    13.47 ms.
  - The three shared-schedule kicks take 9.61 ms. Gaps of 2 us or more
    total 1.75 ms on average, 1.50 ms of it in 0.7 gaps over 50 us: the
    known late-submit stall.
  - The traced probe's steadyMedian was 12.95 ms.
- As in round 6, the trace cannot split a shared kick by stage. Per-stage
  GPU time below therefore comes from round 4's kernel timer, which has a
  65.5 us quantum.

**Excess dispatches over rest, by stage.** Each row gives the excess over
rest, the workgroups per launch, whether the stage's dependency is
block-local, and the boundaries one workgroup per block could remove.

| stage | excess | workgroups a launch | block-local? | removable, generous |
|---|---|---|---|---|
| pressure arena (150 against 104: one more Krylov step and V-cycle) | 46 | 235 | no. Each band sweep, residual and restrict reads the neighbour tiles' previous step; the coarse levels are the same; Krylov reduces globally | 24 at most, only with halo-overlapped fine smoothing (redundant work); the coarse tail is bounded under 1 ms (round 7) |
| NB-FLIP slabSortMerge | 22 | 1024 | yes, if a long cell's merge fits one workgroup | 20 |
| frame plan (seed, dilate0-2 x2, local speed, tile error) | 16 | 384 | no. Each dilation reads the neighbour's previous one | 4 |
| mixed surface advect | 15 | 1228 | partly | 11 |
| layout builder | 11 | 187 (scan chain) | global scan | 10 |
| ownership census | 10 | 263 | partly | 7 |
| extension (paired 8, hierarchy 9, publish 4) | 21 | 256-1028 | sweeps read the neighbour's sweep | 14 |
| remap, momentum, resolve, forces, geometry and the rest | 151 | | mostly neighbour stencils | 7 |
| **total** | **292** | | | **about 97** |

**Estimate.**
- **Removable chains:** about 97 dependent boundaries at about 5 us each
  (round 4's empty-launch microbenchmark of 4.6-5.2 us; round 6's drain
  share of about 14% of kick time) is **about 0.5 ms**. At 10 us a
  boundary it would be about 1.0 ms.
- **Ceilings:** removing every one of the 292 excess boundaries would save
  1.46 ms, and every one of the frame's 432 would save 2.16 ms. Both are
  under the 3 ms gate.
- **Why:** the 11 ms gap is about 85% in-kernel latency. Kernels are
  resident at 9-12% occupancy and run dependent-load chains on few items
  (rounds 4 and 6). Removing launch boundaries does not shorten that, and
  this frame has no long block-local run: its longest is the 22-pass slab
  sort.

**Per the gate:** the estimate is under 3 ms, so nothing was built.

## Round 12: per-dispatch GPU time on drop=2 against rest (measurement only, nothing built)

**Method.**
- **Tool:** tools/gpu-kernel-timer.ts in timing mode, through
  fluid-gpu-run.sh. It ends the pass after every dispatch and puts a
  timestamp pair around each one, so each in-dispatch duration is taken on
  its own and the gaps between passes don't count.
- **Raw timestamps:** Dawn quantizes timestamps to 65.5 us by default. The
  probe passes `disable-dawn-features=timestamp_quantization` (env
  FLUID_TS_RAW, scratch only) to get raw values.
- **Runs:** 8x8x4, `--pure --steps=30`, one capture for drop=2 and one for
  rest. The table averages steps 29 and 30.
- **Not usable for this:** xctrace can't split the shared-schedule kicks by
  dispatch (round 11).
- **Data:**
  - round12-kernels.tsv: every kernel with stage, entry, launches,
    workgroups, us and excess;
  - round12-dispatches-{drop2,rest}-step30.tsv: every dispatch in
    submission order.

**Totals.**
- drop=2: 15.44 and 15.63 ms over 432 and 445 dispatches.
- rest: 2.57 and 3.09 ms over 140 and 153 dispatches.
- Excess: about 12.7 ms. The timer's own run was slower (steady median
  15.87 ms) because of the extra pass gaps; only the in-dispatch time is
  used.
- Live work at drop=2: tiers [425, 83, 4], fine 123, particles 10266,
  bricks 115, band tiles 59.

**The curve is flat.**
- 261 kernels have positive excess. The top 33 make 50% of it and the
  top 93 make 80%.
- At dispatch level (drop=2 step 30), 78 of 445 dispatches make 50% of the
  GPU time and 204 make 80%.
- The top 10 kernels are about 3.5 ms (27%).

**By family (excess, ms).**

| family | ms |
|---|---|
| pressure arena | 3.71 |
| NB-FLIP | 2.84 |
| census | 0.75 |
| extension | 0.56 |
| surface advect | 0.50 |
| remap | 0.41 |
| surface advect deferred | 0.41 |
| extension hierarchy | 0.37 |
| geometric fill | 0.30 |
| layout | 0.26 |
| local speed | 0.26 |
| resolve | 0.24 |
| forces | 0.21 |
| momentum | 0.18 |

**Pressure by level (drop=2 against rest, us).**

| level | drop=2 | launches | rest |
|---|---|---|---|
| coarse tail (pDown3, pDown4, pTail, pUp4, pUp3) | 1781 | 48 | 366 |
| fine (pRelax, pNativeRelax, pResidual2, pRestrict0, pRebase, pProlong0) | 1072 | 44 | 241 |
| levels 1-2 (pDownA/R/B, pDown2, pBulkLift, pUpB/A) | 790 | 21 | 157 |
| Krylov | 427 | 11 | 77 |

- drop=2 runs 1.5x the rest frame's cycle launches (pTail 9/6, pRelax
  18/12, pUp4 12/8).
- Each launch is also 3-9x slower:

  | kernel | drop=2 us/launch | rest us/launch |
  |---|---|---|
  | pTail | 75 | 8 |
  | pUp4 | 38 | 12 |
  | pRelax | 21 | 7 |
  | pNativeRelax | 29 | 6 |
  | pDown2 | 84 | 17 |

  The latency grows with the live rows, at low occupancy.

**Top 22 by excess (us; launches drop=2/rest; us a launch at drop=2; workgroups).**

| # | stage : entry | excess | launches | us/launch | workgroups |
|---|---|---|---|---|---|
| 1 | pressure : pTail | 627 | 9/6 | 75 | 1 |
| 2 | surface advect deferred : advectDeferred | 383 | 5/0 | 77 | 1024 |
| 3 | pressure : pUp4 | 361 | 12/8 | 38 | indirect (page) |
| 4 | extension : pair_sweepSeams | 318 | 2/0 | 159 | 1024-1063 |
| 5 | pressure : pRelax | 299 | 18/12 | 21 | 512 |
| 6 | pressure : pNativeRelax | 294 | 12/8 | 29 | 80 |
| 7 | census : classifyCoarse | 293 | 1/0 | 293 | 256 |
| 8 | NB-FLIP : advectParticlesDeferred | 236 | 1/0 | 236 | 1024 |
| 9 | NB-FLIP : depthCoarse | 228 | 1/0 | 228 | 1024 |
| 10 | remap : remapFaces | 224 | 4/0 | 56 | 1024 |
| 11 | pressure : pDown2 | 219 | 3/2 | 84 | indirect |
| 12 | geometric fill : geometryRegional | 210 | 2/0 | 105 | 256-4096 |
| 13 | pressure : pDown3 | 208 | 9/6 | 32 | indirect |
| 14 | phi resolve : resolve | 197 | 4/0 | 49 | 1024 |
| 15 | pressure : kApplyNB | 165 | 2/1 | 90 | 328 |
| 16 | pressure : pDownB | 156 | 3/2 | 59 | 80 |
| 17 | extension hierarchy : mixedSweepBlocks | 156 | 3/0 | 52 | 256 |
| 18 | local speed : localSpeed | 142 | 1/0 | 142 | 1024 |
| 19 | extension hierarchy : mixedProlongUnknownVelocity | 141 | 3/0 | 47 | 256 |
| 20 | pressure : pUp3 | 136 | 9/6 | 24 | indirect |
| 21 | NB-FLIP : arenaDistance | 133 | 1/0 | 133 | 1024 |
| 22 | NB-FLIP : slabSortMerge | 133 | 22/0 | 6 | 1024 |

**Top 10: why each takes its time, from the WGSL.**

1. **pTail** (uniform-pressure-arena.ts, pTail).
   - One workgroup of lanesTail lanes, sized at least as large as the
     tail's block capacity, so every row is in a register.
   - TAIL_SWEEPS = 32 sweeps x 2 colours, with a workgroupBarrier after
     each colour: **64 serialized steps**. Each step reads 6
     workgroup-memory neighbours and does one divide.
   - That is about 1.1 us a step on one core, which is pure serial
     latency.
   - The rest frame's launches average 8 us, so they hold far fewer rows
     or return converged. This was not checked per launch.
2. **advectDeferred** (uniform-mixed-surface.ts).
   - Five twin launches each grid-stride over the **whole** deferred list
     (near and deep entries), and each skips the words that aren't its
     own (by code or umRegularFine).
   - Per vertex the chain is: umEarlyFarAdvected, then umCheapAdvected,
     then umVertexFoot, then umEulerAdvected, then the dependent trace.
     That is 4 dependent sampling stages before the trace.
   - The list scan is paid 5 times.
3. **pUp4.**
   - One page a workgroup. Each of the 216 lanes does tAt(slot,4,g), then
     **up to 8 serial tAt(slot,5,q) parent lookups**, each a dependent
     page-table load followed by a value load: a chain of about 10
     dependent global loads.
   - Then schedule.sweeps (3) x 2 colours with only 8 lanes active and a
     barrier per colour: 6 more serialized steps.
4. **pair_sweepSeams** (uniform-mixed-extension.ts).
   - ueSweepKind 1 runs ueSweepSeamFine: 3 patches a lane through
     ueExtend, which loops over 6 neighbours, each a ueCompiledNeighbor
     search (a dependent lookup).
   - Kind 2 runs the regular coarse packs and then the coarse seam packs,
     serially in the same lane.
5. **pRelax.**
   - Two band tiles a workgroup, 2 colours with a storageBarrier.
   - pRelaxRow reads 6 faces. A seam face goes through
     nAcross, then nAt (a page-table lookup), then qF: 3 dependent loads.
   - A wall row then runs **4 active-set passes** over its wall faces,
     each with pLift loads.
   - 18 launches x 21 us.
6. **pNativeRelax.**
   - 80 workgroups grid-stride over the level-2 blocks, with a barrier per
     block.
   - A seam face loops **16 serial fine-row reads**: pool[P_KINDS], then
     pF(P_C), then pF(plane).
   - With several seam faces that is 32-96 dependent loads on one lane,
     plus the same 4-pass wall loop.
7. **classifyCoarse** (uniform-mixed-dynamic.ts).
   - One lane a 4h tile (about 425 tiles over 256 x 64 lanes).
   - umClassifyOwner runs serially per lane: umLoadCorner8, then
     umOwnerAt, then **3 axes x 2 signs x up to 16 umFace parts**, each
     part a texture load. That is up to 96 serial loads.
   - Only umResolutionError (8 lanes) and umDepthSample (64 lanes) are
     spread across lanes.
   - 3 barriers per batch of 64.
8. **advectParticlesDeferred** (NB-FLIP).
   - One lane a deferred particle over 10266 particles. The
     sampleVelocity trace is a dependent sample chain.
   - The same shape as nbMove's nbAdvect: per job, an nbSlabJob binary
     search, then nbWinStage, then 27 brick lookups before the first
     record.
9. **depthCoarse** (uniform-narrow-band-redistance.wgsl.ts).
   - One lane a build. nbBuildDistance is nbSurfaceDistance plus a
     gradient trace of **up to 5 dependent iterations**. Each iteration is
     umCellOpen, then nbPhiGradient, then umSlideStart/nbPointNear, then
     nbCellNear.
   - Then nbBoundCoarseMetric: 7 umSampleVertex calls and 8 umCellOpen.
10. **remapFaces** (uniform-mixed-remap.ts).
    - One 192-lane workgroup a listed tile, grid-strided, with 4
      workgroup barriers per tile and another between tiles.
    - At about 56 us a launch, the per-tile barrier steps dominate.

**Proposals, not built, with estimates.**

1. **The coarse tail's serial sweeps (pTail).**
   - **Change:** replace the 64-step red-black loop with a shorter solve.
     The options are fewer sweeps, or a Chebyshev/CG tail in the same
     workgroup, whose few reductions cost fewer barriers than 32 sweeps.
   - **Estimate:** 32 to 8 sweeps is about 55 us x 9 launches, so
     **about 0.5 ms** on drop=2 (rest about unchanged).
   - **Risk:** round 7 showed that a looser tail (tailCycles 1) cost 14
     cycles, so this may give back cycles.
   - **Gates:**
     - pressure cycles a frame on drop=2, seep, letters and dam unchanged
       within noise;
     - seep and letters drift within 3%;
     - steady median on drop=2 and rest not slower;
     - types and unit green.
2. **Parallel parent and neighbour staging in pUp4, pUp3, pDown3 and
   pDown2.**
   - **Change:** a page's 27 neighbour pages and the parent lookups that
     each lane now chains serially (up to 8 tAt calls) are fetched once
     per page by 27 lanes into workgroup memory, behind one barrier. Each
     lane then reads workgroup memory. This shortens the chain from about
     10 dependent global loads to about 3.
   - **Estimate:** these four make 1.07 ms at drop=2, and about half of it
     is the lookup chain, so **about 0.3-0.5 ms**.
   - **Gates:**
     - corrections and Y bitwise identical to the current path on a
       recorded frame (regression test);
     - cycles unchanged;
     - drift and steady median as above.
3. **Spread classifyCoarse's face-part loop across lanes.**
   - **Change:** one lane per (axis, sign, part) instead of 96 serial
     parts per lane. This is the remap kernel's "footprint loops side by
     side" layout.
   - **Estimate:** 293 us to about 60 us, so **about 0.23 ms**.
   - **Gates:**
     - census output (owners, codes, counts) bitwise identical on a
       recorded frame;
     - drift;
     - steady median.

**Runner-up:** advectDeferred's five list scans could be bucketed by code
once, so each twin scans only its own entries. If the scan dominates, this
is about 0.25 ms. That wasn't measured.

**Summary.**
- The three proposals together are **about 1.0-1.2 ms of 12.7 ms**. The
  rest of the excess is spread across 250 kernels, none over 0.63 ms.
- The pressure arena is the largest family (3.7 ms). Proposals 1 and 2
  are in Lane P's file, so they need coordinating.
