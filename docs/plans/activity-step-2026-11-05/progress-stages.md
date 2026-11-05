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
