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
