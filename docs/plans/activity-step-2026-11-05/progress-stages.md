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
