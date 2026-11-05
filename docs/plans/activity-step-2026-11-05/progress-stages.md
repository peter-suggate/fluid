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
