# Uniform Geometric on `cm12-figure-7` (128³): where the GPU time goes, and how much lattice is live

Measured 2026-09-19 on Dawn/Metal, working tree `codex/main-unpushed-20260918`, **no repo file
modified**. Two deliverables: a per-stage hardware-timestamp trace over 70 advances (Task 1) and a
live-tile occupancy census over the same trajectory (Task 2).

Raw data beside this file:

- `fig7-trace.json` — the stage trace (65 accepted hardware samples, per-sample series).
- `fig7-regimes.json`, `fig7-groups.json` — per-regime rollups derived from it.
- `fig7-census.json` — the tile census (16 sampled steps), **final run**: ladders to k=8, tile
  classes, V thresholds and V sign statistics. All Task 2 tables below are from this file.
- `fig7-census-k5.json` — the previous census run (ladders to k=5 only), kept for the run-to-run
  comparison noted in the caveats.
- `fig7-trace.log`, `fig7-census.log` — run logs.

Scripts (scratch only): `probe-fig7.mts`, `census-fig7.mts`, `analyse-trace.mjs`.

## Commands

```sh
cd /Users/petersuggate/code/me/fluid
S=/private/tmp/claude-501/-Users-petersuggate-code-me-fluid/4dd2ff94-6fba-4892-a1dc-4a2c63f35fd8/scratchpad

# Task 1 — one Dawn run
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_PROBE_ADVANCES=70 \
  FLUID_PROBE_MIN_SAMPLES=20 FLUID_PROBE_OUT=$S/fig7-trace.json \
  node --import tsx $S/probe-fig7.mts > $S/fig7-trace.log 2>&1
node $S/analyse-trace.mjs

# Task 2 — same command each time; the script grew V thresholds, then tile classes,
# then k=8 ladders + V sign statistics. The last run is the one reported.
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_CENSUS_ADVANCES=70 \
  FLUID_CENSUS_EVERY=5 FLUID_CENSUS_OUT=$S/fig7-census.json \
  node --import tsx $S/census-fig7.mts > $S/fig7-census.log 2>&1
```

The GPU lock `/tmp/fluid-webgpu-exclusive.lock` did not exist at the start; each tool takes and
releases it itself. Three Dawn processes ran in total, strictly one at a time, in the foreground.

## Configuration actually measured

`uniformVolumeMethod` (`Uniform Geometric`), quality `balanced`,
`resolveMethodValues(uniformVolumeMethod,"balanced",{timeStep:"scene"})`, scene
`cm12-figure-7` = 128³, h = 50 mm (cubic), dt = 1/30 s, `lastSubsteps = 1` (one dt per advance,
so an advance is exactly one paper step). Resolved values that matter:

| value | this run | mini64 reference runs |
| --- | --- | --- |
| `extensionFrontSweeps` | **8** (working-tree default for this method) | 16 |
| `liquidCapacityBalancing` | off | off |
| `sharpeningWorkMap` | **on** (4h tile map live) | on for the "tiled" column, off for "dense" |
| `redistance` | on | on |
| `velocityTransport` | semi-Lagrangian | semi-Lagrangian |
| multigrid levels / passes | 7 / **2661** (setup 22, full 1926, V 712, finish 1) | 6 / 2008 |
| extension passes per invocation | 34 (35 counted incl. the closing boundary pass) | 48 |

Liquid is **33 464 cells of 2 097 152 = 1.596 %** of the lattice, and the conserved sum moved by
2·10⁻⁵ % over 70 steps — conservation is sane.

## Regime boundaries (chosen from the data)

`maxSpeed_m_s` rises exactly at g·dt = 0.3333 m/s per step for 25 steps (free fall), jumps to
40–50 m/s at steps 26–28 (impact), then decays. Liquid extent (cells holding any V) is flat at
~1.7–3 % through step 30 and then climbs to ~27 % by step 60 (spreading sheet).

| regime | samples | why |
| --- | --- | --- |
| free fall | **2–25** | `maxSpeed` = 0.333·n exactly; ball intact; 5.60 cells/step at step 25 |
| impact | **26–40** | `maxSpeed` spikes 40→50 m/s then decays; liquid extent 2.9 → 13 % |
| *(transition)* | *41–44* | excluded; reported separately in `fig7-regimes.json` |
| spread | **45–70** | sheet on the floor, extent 19 → 27 % of the lattice |

Sample 1 is dropped as warm-up by the probe. Sample id equals advance number.

## Task 1 — per-stage GPU time

GPU totals (hardware timestamps, Dawn tick 65.5 µs):

| regime | n | p10 | **median** | p90 |
| --- | ---: | ---: | ---: | ---: |
| free fall | 24 | 98.50 | **104.53** | 109.90 |
| impact | 15 | 122.95 | **137.56** | 162.59 |
| spread | 22 | 144.11 | **152.96** | 161.15 |
| all 2–70 | 65 | 103.61 | **138.54** | 159.12 |

CPU side (the probe's `physicsCPUTrace`): total **30.90 ms** median (p10 30.30 / p90 32.17), of
which `CM11a Full-Cycles` encoding is 13.85 ms, `V-Cycles` 5.09 ms and
"Capture closure + command submission" 10.90 ms. Every other stage encodes in under 0.21 ms. CPU
encode is 22 % of the GPU span, so the frame is GPU-bound; the CPU cost is essentially all
command encoding for the ~2 650 pressure passes.

### Per-phase medians (ms) and share of the GPU total

| phase | passes | free fall | impact | spread | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sec. 3.3 interface authority | 1 | 0.655 (0.6%) | 0.655 (0.5%) | 0.655 (0.4%) | 0.655 (0.5%) |
| Sec. 3.3 narrow-band FIM front | 19 | 17.039 (16.3%) | 20.054 (14.6%) | 23.200 (15.2%) | 20.775 (15.0%) |
| Sec. 3.3 hierarchy fill + transport shell | 15 | 4.915 (4.7%) | 4.588 (3.3%) | 2.425 (1.6%) | 4.653 (3.4%) |
| Dense vertex phi transport + redistance | 2 | 2.228 (2.1%) | 2.884 (2.1%) | 2.163 (1.4%) | 2.294 (1.7%) |
| Dense geometric volume coupling *(= conservative transport)* | 12 | 22.675 (21.7%) | 25.690 (18.7%) | 28.443 (18.6%) | 26.608 (19.2%) |
| Dense liquid capacity balancing *(here: `uvGather` + γ copy only)* | 1 | 6.423 (6.1%) | 6.357 (4.6%) | 6.357 (4.2%) | 6.357 (4.6%) |
| Dense conservative volume sharpening *(tile map on)* | 33 | 1.835 (1.8%) | 2.425 (1.8%) | 3.080 (2.0%) | 2.621 (1.9%) |
| Velocity advection + body forces | 1 | 5.636 (5.4%) | 7.602 (5.5%) | 3.736 (2.4%) | 4.915 (3.5%) |
| CM11a topology + RHS pyramid | 22 | 2.621 (2.5%) | 2.359 (1.7%) | 2.490 (1.6%) | 2.556 (1.8%) |
| CM11a Full-Cycles | 1926 | 27.197 (26.0%) | 44.106 (32.1%) | 45.548 (29.8%) | 43.516 (31.4%) |
| CM11a V-Cycles | 712 | 9.372 (9.0%) | 20.447 (14.9%) | 21.823 (14.3%) | 15.729 (11.4%) |
| CM11a parity copy + fine residual | 1 | 0.131 (0.1%) | 0.131 (0.1%) | 0.131 (0.1%) | 0.131 (0.1%) |
| Pressure projection | 1 | 1.966 (1.9%) | 2.032 (1.5%) | 2.097 (1.4%) | 2.032 (1.5%) |
| Dense phi surface publication | 1 | 0.197 (0.2%) | 0.197 (0.1%) | 0.197 (0.1%) | 0.197 (0.1%) |
| Diagnostics reduction | 1 | 0.328 (0.3%) | 0.328 (0.2%) | 0.328 (0.2%) | 0.328 (0.2%) |
| **GPU total** | **2748** | **104.53** | **137.56** | **152.96** | **138.54** |

p10/p90 for every cell are in `fig7-regimes.json`. Phases that this scene never encodes
(Sec. 3.4 density advection, gamma diffusion, Sec. 3.5 correction / mass return, Sec. 3.6 solid
excess, rigid coupling, Sec. 3.8 post-process) are absent because the geometric-volume path
replaces them, not because they were missed.

Notes on the phase table:

- **`Dense liquid capacity balancing` is a misleading label here.** Balancing is off, so this seam
  measures only `uvGather` (one dense 128³ pass) plus the 8 MiB γB→γA texture copy. It is one
  dense pass costing 6.36 ms.
- **The V-cycle stage is convergence-gated, not fixed.** Per-sample it takes 6.1, 10, 15, 21, 26
  or 35–40 ms in clear bands (see the series in `fig7-trace.json`), because the residual tolerance
  1e-4 terminates before the 4-cycle budget on many steps. This is the main source of the wide
  p10/p90 on the total, and it is *not* stage-cost bimodality of the kind the repo memory warns
  about — it is a different amount of work.
- **Velocity advection is genuinely displacement-dependent.** One dense pass, but it steps
  3.54 → 5.57 → 7.60 → 9.70 → 13.76 ms across free fall in lockstep with the growing back-trace
  length, then falls back to 3.6–4.7 ms in the spread regime where max speed drops. Semi-Lagrangian
  trace length, not cell count, is what moves it within a scene.
- Small phases quantise to the 65.5 µs tick (0.131 = 2 ticks, 0.197 = 3, 0.328 = 5, 0.655 = 10).
- `closureError_ms` is exactly 0 on all 65 samples: the phase medians account for the whole span.

### Grouped, with the pass floor made explicit

| group | passes | median ms (all) | share | **µs per pass** |
| --- | ---: | ---: | ---: | ---: |
| velocity extension (Sec. 3.3, 8 sweeps) | 35 | 24.18 | 17.5 % | 691 |
| phi transport + redistance | 2 | 2.29 | 1.7 % | 1 147 |
| conservative volume transport (coupling) | 12 | 26.61 | 19.2 % | 2 217 |
| `uvGather` + γ copy | 1 | 6.36 | 4.6 % | 6 357 |
| volume sharpening (tile map on) | 33 | 2.62 | 1.9 % | 79 |
| velocity advection | 1 | 4.92 | 3.5 % | 4 915 |
| **pressure (setup + full + V + finish)** | **2661** | **65.60** | **47.4 %** | **25** |
| projection | 1 | 2.03 | 1.5 % | 2 032 |
| publication | 1 | 0.20 | 0.1 % | 197 |
| diagnostics | 1 | 0.33 | 0.2 % | 328 |

Every non-pressure group runs at 79 µs to 6.4 ms per pass — far above the ~13–25 µs pass floor.
Pressure runs at 25 µs per pass, i.e. **at** the floor.

### Per-million-lattice-cell scaling against 64³

128³ = 2.097152 Mcells; `minimal-power-dam-break-64` = 0.262144 Mcells (8×). 64³ numbers are the
dense stage census in `docs/benchmarks/uniform-geometric-tile-work-2026-09-19.json`
(`denseStageCensus`, `minimal-power-dam-break-64`), plus the tiled-sharpening median from
`docs/benchmarks/uniform-geometric-tile-work-2026-09-19.md` and the pressure figure from
`docs/benchmarks/uniform-pressure-tolerance-2026-09-19.json`.

| stage | fig7 128³ ms | fig7 ms/Mcell | mini64 64³ ms | mini64 ms/Mcell | ratio 128³/64³ per Mcell | reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| conservative transport (coupling + gather) | 32.97 | **15.72** | 4.26 | **16.25** | **0.97** | **per-cell bound** |
| velocity extension, whole stage | 24.18 | 11.53 | 7.73 | 29.50 | 0.39 | sweep counts differ (8 vs 16) |
| — FIM front, per sweep | 20.78 / 8 sw | **1.24** /sw | 5.77 / 16 sw † | **1.38** /sw | **0.90** | **per-cell bound** |
| phi transport + redistance | 2.29 | 1.09 | 0.59 | 2.25 | 0.49 | per-cell, but band occupancy differs |
| volume sharpening (tile map on) | 2.62 | 1.25 | 1.54 | 5.87 | 0.21 | per-*live-tile* bound (occupancy 4–30 % vs 47 %) |
| **pressure** | 65.60 | **31.28** | 32.44 (census) / 53.9 (tolerance) | **123.8 / 205.6** | **0.25 / 0.15** | **flat — pass-floor bound** |
| GPU total / wall | 138.54 (GPU span) | 66.06 | 74.80 (CPU wall) ‡ | 285.3 | 0.23 | — |

† the 5.767 ms dense front is from `docs/benchmarks/uniform-geometric-extension-tile-work-2026-09-19.md`,
a different tool from the 7.733 ms whole-extension census figure; those two 64³ tools disagree on
the extension bracket by ~1.6 ms, which is a known and documented discrepancy, not something this
run resolves.
‡ the 64³ `medianWall_ms` is a CPU wall with a queue fence and is not the same instrument as the
GPU timestamp span, so the last row is indicative only.

**The answer to the scaling question.** Pressure is the only group that gets *cheaper* per cell as
the lattice grows — 8× the cells bought only 2.0× the pressure time, and per pass it barely moved
(16.2 µs at 64³ → 24.7 µs at 128³). It is pass-floor bound and a tile map cannot reach it.
Everything else holds its per-cell cost: the conservative transport bracket is within 3 % of the
64³ per-cell figure, and the FIM front within 10 % per sweep. Those stages *are* per-cell bound at
128³ and a tile map can reach them.

Size of the prize on this scene:

| regime | non-pressure GPU ms | share | tile-map-addressable ms § | share |
| --- | ---: | ---: | ---: | ---: |
| free fall | 64.75 | 61.9 % | 62.72 | **60.0 %** |
| impact | 71.96 | 52.3 % | 71.63 | **52.1 %** |
| spread | 75.17 | 49.1 % | 71.50 | **46.7 %** |
| all | 72.94 | 52.6 % | 70.26 | **50.7 %** |

§ FIM front + hierarchy + phi + coupling + gather + advection + sharpen + projection. Excludes the
authority pass, publication and diagnostics (1.18 ms together), which are also dense.

## Task 2 — live-tile occupancy census

`census-fig7.mts` advances the same scene/method/values and every 5th step reads back
`solver.vertexPhiTexture` (129³ r32float, metres — `uniformVolumeInitialPhi` builds it in metres
and `uvRedistancePhi` never rescales), `solver.volumeTexture` (= `volumeA`, which receives
`volumeB` at `webgpu-uniform-reference.ts:1479`, so it is the current V after an advance) and
`solver.velocityTexture` (= `velocityA`, post-projection, rgba32float).

Tiles are 4³ cells on a 32³ = **32 768** tile grid. A tile owns cells [4t..4t+3] and therefore the
5³ vertex block [4t..4t+4]; the census assigns each vertex to every tile whose block contains it.
Dilation is Chebyshev on the tile grid, done separably.

### Seeds and displacement

| step | V≠0 cells | %lat | V sum | min phi (m) | V≠0 tiles | phi<4h tiles | phi<2h tiles | **seed4h** | seed2h | max\|u\| m/s | **disp cells/step** | k for 2d+4 | k for d+2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 36 528 | 1.74 | 33 464.0 | −1.000 | 2.29 % | 3.80 % | 3.05 % | **3.80 %** | 3.05 % | 0.00 | 0.00 | 1 | 1 |
| 1 | 36 408 | 1.74 | 33 464.0 | −1.000 | 2.29 % | 3.80 % | 3.05 % | **3.80 %** | 3.05 % | 0.33 | 0.22 | 2 | 1 |
| 5 | 37 740 | 1.80 | 33 464.0 | −0.968 | 2.51 % | 3.88 % | 3.20 % | **3.88 %** | 3.21 % | 1.67 | 1.11 | 2 | 1 |
| 10 | 41 804 | 1.99 | 33 464.0 | −0.954 | 2.93 % | 3.85 % | 3.25 % | **3.86 %** | 3.28 % | 3.33 | 2.22 | 3 | 2 |
| 15 | 48 742 | 2.32 | 33 464.0 | −0.940 | 3.41 % | 3.75 % | 3.08 % | **3.86 %** | 3.49 % | 5.00 | 3.33 | 3 | 2 |
| 20 | 56 924 | 2.71 | 33 464.0 | −0.931 | 4.03 % | 3.82 % | 3.15 % | **4.27 %** | 4.04 % | 6.67 | 4.45 | 4 | 2 |
| 25 | 64 251 | 3.06 | 33 464.0 | −0.920 | 4.34 % | 3.75 % | 3.08 % | **4.41 %** | 4.34 % | 8.40 | 5.60 | 4 | 2 |
| 30 | 60 048 | 2.86 | 33 464.0 | −0.898 | 3.84 % | 2.40 % | 2.17 % | **3.84 %** | 3.84 % | 19.15 | **12.77** | **8** | 4 |
| 35 | 161 239 | 7.69 | 33 464.0 | −0.752 | 9.67 % | 4.13 % | 4.10 % | **9.67 %** | 9.67 % | 9.27 | 6.18 | 5 | 3 |
| 40 | 274 019 | 13.07 | 33 464.0 | −0.668 | 15.37 % | 6.18 % | 5.83 % | **15.37 %** | 15.37 % | 13.64 | 9.09 | 6 | 3 |
| 45 | 389 406 | 18.57 | 33 464.0 | −0.668 | 20.72 % | 7.76 % | 5.33 % | **20.72 %** | 20.72 % | 14.06 | 9.37 | 6 | 3 |
| 50 | 489 912 | 23.36 | 33 464.0 | −0.668 | 26.10 % | 9.19 % | 6.75 % | **26.10 %** | 26.10 % | 16.97 | 11.31 | 7 | 4 |
| 55 | 542 578 | 25.87 | 33 464.0 | −0.668 | 28.71 % | 9.90 % | 7.49 % | **28.71 %** | 28.71 % | 4.12 | 2.75 | 3 | 2 |
| 60 | 553 506 | 26.39 | 33 464.0 | −0.668 | 29.73 % | 9.37 % | 7.51 % | **29.73 %** | 29.73 % | 4.20 | 2.80 | 3 | 2 |
| 65 | 521 309 | 24.86 | 33 464.0 | −0.668 | 27.77 % | 7.14 % | 6.89 % | **27.77 %** | 27.77 % | 10.18 | 6.78 | 5 | 3 |
| 70 | 480 410 | 22.91 | 33 464.0 | −0.668 | 26.02 % | 6.29 % | 6.11 % | **26.02 %** | 26.02 % | 7.21 | 4.81 | 4 | 2 |

`disp = dt·max|u component|/h`. The solver's own `maxSpeed_m_s` agrees with the readback max
component to within 10 % (it is a magnitude, not a component).

### Which term dominates the seed

**The V≠0 term dominates, and it is a float smear, not liquid.** From step 30 onward `seed4h`
equals the V≠0 set exactly — the phi band is entirely inside it. But the V sum is constant at
33 464 cells while the V≠0 count reaches 560 210. Admission thresholds show what that support is:

| step | V≠0 | \|V\|>1e−9 | >1e−6 | >1e−4 | >1e−2 | mass kept at 1e−6 | at 1e−4 | at 1e−2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 4.34 % | 2.43 % | 2.34 % | 2.34 % | 2.31 % | 100.0000 % | 100.0000 % | 99.9993 % |
| 40 | 15.37 % | 6.21 % | 4.02 % | 3.42 % | 3.33 % | 100.0000 % | 99.9998 % | 99.9961 % |
| 60 | 29.73 % | 10.77 % | 9.30 % | 8.38 % | 7.60 % | 100.0000 % | 99.9989 % | 99.8683 % |
| 70 | 26.02 % | 11.81 % | 9.67 % | 7.96 % | 6.70 % | 100.0000 % | 99.9986 % | 99.8141 % |

Dropping cells with |V| ≤ 1e−6 loses **no measurable mass** (100.0000 % kept to 7 significant
figures) yet cuts the seed from 29.7 % to 9.3 % of tiles at step 60. A skip on that threshold is
*not* bit-exact and is not proposed as one — but it says the exact V≠0 seed is paying for a
sub-picolitre smear, and that a transport operator which flushed near-zero V would make an exact
map far cheaper.

**The smear is signed float dust, both signs.** The final run counted V by sign:

| step | V≠0 | of which negative | full (V ≥ 1−1e−6) | partial (0 < V < 1) | min V | max V | negative mass | V sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 36 528 | 0 | 30 592 | 5 936 | 0 | 1.000000 | 0.000 | 33 464.00 |
| 25 | 64 251 | 22 458 | 8 315 | 33 478 | −3.0e−8 | 1.188945 | −0.000 | 33 464.00 |
| 40 | 274 019 | 115 218 | 2 412 | 156 389 | −6.0e−8 | **20.22** | −0.000 | 33 464.01 |
| 50 | 489 912 | 247 815 | 4 445 | 237 652 | −1.5e−8 | **55.24** | −0.000 | 33 464.01 |
| 60 | 553 506 | 269 736 | 2 833 | 280 937 | −6.0e−8 | 6.99 | −0.000 | 33 464.02 |
| 70 | 480 410 | 153 649 | 5 718 | 321 043 | −6.0e−8 | 27.01 | −0.000 | 33 464.02 |

About half the V≠0 cells hold *negative* V, with |V| ≤ 6e−8 and total negative mass below
0.001 cells — pure round-off from the float-CAS donor sums. Two further facts fall out of the same
count and matter for the design:

- **V is not capacity-limited.** `liquidCapacityBalancing` is off by default in this method, and
  the conservative gather produces cells with V up to **55×** the open capacity of 1. So
  "0 < V < capacity" is not a useful partial-cell test here; over-full cells are common and the
  partial count above (which is `0 < V < 1`) misses them.
- **The sheet has almost no full cells.** Full cells fall from 30 592 at t = 0 (the ball's core) to
  2 400–6 000 during the spread. Figure 7's sheet really is thinner than a cell nearly everywhere.

### Tile CLASSES for a two-level scheme

Definitions as briefed. SURFACE = any of the tile's 5³ vertices has |phi| < 4h, **or** any cell is
partial (0 < V < capacity = 1), **or** any cell has V ≠ 0 while its cell-centre phi > 0 ("liquid
the level set has lost"; cell-centre phi is the mean of the 8 corner vertices, which is what
`uvPhi` returns at a cell centre). BULK-LIQUID = every vertex phi ≤ −4h. FAR-AIR = every vertex
phi ≥ 4h and every cell V = 0. The four classes are disjoint and **`other` is 0 at every sampled
step** — the partition closes exactly at 32 768.

| step | SURFACE k=0 | SURFACE k=1 | BULK | FAR-AIR | other | lost cells | lost tiles | **lost mass (cells)** | **% of V** | 2-level k=0 | 2-level k=1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.38 % | 6.63 % | 136 | 31 524 | 0 | 2 976 | 388 | 834.0 | 2.49 % | **4.94 %** | **8.19 %** |
| 1 | 3.38 % | 6.63 % | 136 | 31 524 | 0 | 2 856 | 388 | 665.1 | 1.99 % | 4.94 % | 8.19 % |
| 5 | 3.47 % | 6.77 % | 136 | 31 496 | 0 | 4 476 | 463 | 826.4 | 2.47 % | 5.03 % | 8.34 % |
| 10 | 3.49 % | 6.79 % | 120 | 31 504 | 0 | 8 740 | 608 | 980.9 | 2.93 % | 5.05 % | 8.35 % |
| 15 | 3.74 % | 6.90 % | 40 | 31 503 | 0 | 15 846 | 777 | 1 136.0 | 3.39 % | 5.30 % | 8.46 % |
| 20 | 4.27 % | 7.54 % | 1 | 31 369 | 0 | 24 168 | 960 | 1 271.4 | 3.80 % | 5.83 % | 9.10 % |
| 25 | 4.41 % | 7.31 % | **0** | 31 324 | 0 | 31 719 | 1 078 | 1 442.9 | 4.31 % | 5.97 % | 8.87 % |
| 30 | 3.76 % | 6.12 % | 24 | 31 511 | 0 | 35 776 | 1 005 | 934.0 | 2.79 % | 5.33 % | 7.68 % |
| 35 | 9.67 % | 14.43 % | 0 | 29 598 | 0 | 126 467 | 2 830 | 2 599.6 | 7.77 % | 11.24 % | 15.99 % |
| 40 | 15.37 % | 19.24 % | 0 | 27 730 | 0 | 238 632 | 5 038 | 3 027.6 | 9.05 % | 16.94 % | 20.80 % |
| 45 | 20.72 % | 24.55 % | 0 | 25 978 | 0 | 344 008 | 6 774 | 2 362.1 | 7.06 % | 22.28 % | 26.11 % |
| 50 | 26.10 % | 30.37 % | 0 | 24 216 | 0 | 447 659 | 8 523 | 5 920.3 | 17.69 % | 27.66 % | 31.93 % |
| 55 | 28.71 % | 33.18 % | 0 | 23 359 | 0 | 494 092 | 9 337 | 5 873.8 | 17.55 % | 30.28 % | 34.74 % |
| 60 | 29.73 % | 34.81 % | 0 | 23 026 | 0 | 500 415 | 9 534 | 6 347.1 | 18.97 % | 31.29 % | 36.37 % |
| 65 | 27.77 % | 33.15 % | 0 | 23 667 | 0 | 468 621 | 8 780 | 7 076.6 | 21.15 % | 29.34 % | 34.72 % |
| 70 | 26.02 % | 31.37 % | 0 | 24 243 | 0 | 429 698 | 8 167 | 7 420.7 | **22.18 %** | 27.58 % | 32.93 % |

"2-level" = (32 768 coarse cells + 64 × SURFACE tiles) / 2 097 152 dense cells. The coarse grid
alone is a 1.5625 % floor.

Three results from this table:

1. **The "lost liquid" term is not a rounding artefact — it is a fifth of the liquid.** By step 70,
   7 421 of 33 464 cells of conserved volume (**22.2 %**) sit in cells whose centre phi is
   positive. It starts at 2.5 % at t = 0 (discretisation of the ball) and climbs monotonically once
   the sheet forms. A classifier that trusted phi alone would put a fifth of figure 7's mass in
   FAR-AIR. This term must be in the SURFACE test, and it is what makes SURFACE identical to the
   exact `V≠0 OR phi<4h` seed from step 25 onward.
2. **There is no BULK-LIQUID class to exploit on figure 7.** 136 tiles (0.4 %) at t = 0 — the
   ball's core — falling to 0 by step 25 and never recovering. Everything is surface. (The scene
   was chosen for exactly this; a pool scene would look different.)
3. **FAR-AIR stays the overwhelming majority**: 96.2 % of tiles at t = 0, 70.3 % at the worst step
   (60). Even in the fully spread sheet, seven tiles in ten are provably empty air.

#### Two-level work against the exact mask, side by side

| regime | 2-level k=0 | 2-level k=1 | exact mask k=3 | k=4 | k=5 | k=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| free fall (step 15) | **5.30 %** | **8.46 %** | 16.42 % | 23.20 % | 31.51 % | 67.14 % |
| free fall (step 25) | 5.97 % | 8.87 % | 15.23 % | 20.47 % | 26.64 % | 51.41 % |
| impact (step 30) | 5.33 % | 7.68 % | 12.73 % | 17.20 % | 22.55 % | 44.62 % |
| impact (step 40) | 16.94 % | 20.80 % | 26.52 % | 30.00 % | 33.45 % | 43.70 % |
| spread (step 50) | 27.66 % | 31.93 % | 38.31 % | 42.05 % | 45.65 % | 55.88 % |
| spread (step 60) | 31.29 % | 36.37 % | 44.29 % | 48.79 % | 53.15 % | 65.78 % |
| spread (step 70) | 27.58 % | 32.93 % | 41.64 % | 46.59 % | 51.45 % | 65.63 % |

The two-level scheme at k = 1 costs less than the exact mask at k = 3 in every regime, and much
less in free fall (8.5 % vs 16.4 %). The full ladders to k = 8 for the exact seed, the SURFACE
class, the `V>1e−6 OR phi<4h` seed and the phi-only seeds are in `fig7-census.json`
(`tiles.ladder_*`, `tiles.classes.surfaceLadder`, `tiles.volumeThresholds[].ladder*`).

The k needed by the reach argument is now covered: at the impact peak (step 30, 12.77 cells/step)
the exact mask needs k = 8 = **44.6 %** live for reach 2d+4, or k = 4 = 17.2 % for reach d+2. The
two-level scheme is insensitive to that, because its coarse level already covers the whole domain;
only the fine surface band has to be reached, which is what makes k = 0/1 sufficient for it.

### Live tiles vs dilation k (% of 32 768)

Exact seed, `V≠0 OR phi<4h` (identical to the SURFACE class from step 25 on):

| step | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (free fall) | 3.80 | 6.69 | 10.63 | 15.77 | 21.67 | 28.48 | 36.21 | 44.84 | 54.65 |
| 15 | 3.86 | 6.90 | 11.04 | 16.42 | 23.20 | 31.51 | 41.51 | 53.34 | 67.14 |
| 25 | 4.41 | 7.31 | 10.87 | 15.23 | 20.47 | 26.64 | 33.78 | 42.01 | 51.41 |
| 30 (impact) | 3.84 | 6.12 | 9.06 | 12.73 | 17.20 | 22.55 | 28.86 | 36.19 | 44.62 |
| 40 | 15.37 | 19.24 | 23.02 | 26.52 | 30.00 | 33.45 | 36.93 | 40.38 | 43.70 |
| 50 (spread) | 26.10 | 30.37 | 34.42 | 38.31 | 42.05 | 45.65 | 49.15 | 52.55 | 55.88 |
| 60 | 29.73 | 34.81 | 39.65 | 44.29 | 48.79 | 53.15 | 57.42 | 61.63 | 65.78 |
| 70 | 26.02 | 31.37 | 36.58 | 41.64 | 46.59 | 51.45 | 56.22 | 60.94 | 65.63 |

Seed `V>1e−6 OR phi<4h` (not exact; mass-preserving to 7 s.f.):

| step | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.80 | 6.69 | 10.63 | 15.77 | 21.67 | 28.48 | 36.21 | 44.84 | 54.65 |
| 15 | 3.75 | 6.64 | 10.58 | 15.72 | 22.20 | 30.18 | 39.78 | 51.17 | 64.49 |
| 25 | 3.76 | 6.38 | 9.63 | 13.54 | 18.25 | 23.83 | 30.38 | 37.99 | 46.72 |
| 30 | 2.47 | 4.35 | 6.86 | 10.08 | 14.09 | 18.95 | 24.73 | 31.52 | 39.38 |
| 40 | 6.27 | 9.62 | 12.99 | 16.42 | 19.92 | 23.50 | 27.15 | 30.48 | 33.79 |
| 50 | 9.27 | 14.31 | 19.20 | 23.88 | 28.44 | 32.91 | 37.28 | 41.17 | 44.96 |
| 60 | 10.66 | 16.69 | 22.30 | 27.67 | 32.99 | 38.26 | 43.48 | 48.65 | 53.78 |
| 70 | 9.67 | 15.36 | 20.69 | 25.86 | 31.12 | 36.50 | 41.98 | 47.57 | 53.27 |

`phi<4h` alone (the shell a phi-only stage such as redistance or sharpening would need), k=0..5;
the full ladders to k=8 are in the JSON:

| step | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.80 | 6.69 | 10.63 | 15.77 | 21.67 | 28.48 |
| 25 | 3.75 | 6.34 | 9.52 | 13.34 | 17.94 | 23.39 |
| 30 | 2.40 | 4.20 | 6.60 | 9.69 | 13.54 | 18.21 |
| 50 | 9.19 | 14.17 | 18.99 | 23.58 | 28.02 | 32.32 |
| 60 | 9.37 | 15.41 | 20.74 | 25.97 | 31.11 | 36.22 |
| 70 | 6.29 | 10.95 | 15.73 | 20.64 | 25.71 | 30.93 |

`phi<2h` alone and the `V≠0 OR phi<2h` seed, plus the ladders at every threshold, are in
`fig7-census.json` (`tiles.ladder_*`, `tiles.volumeThresholds[].ladder*`).

### What the reach requirement costs

| regime | disp cells/step | k needed for reach 2d+4 | k needed for reach d+2 | live % at that k (exact seed) |
| --- | ---: | ---: | ---: | --- |
| free fall (1–25) | 0.22 → 5.60 | 2 → 4 | 1 → 2 | 10.6 % → 20.5 % (2d+4); 6.7 % → 10.9 % (d+2) |
| impact (26–40) | up to **12.77** | up to **8** | up to 4 | **44.6 %** at k=8 (step 30); 17.2 % at k=4 |
| spread (45–70) | 2.75 → 11.31 | 3 → 7 | 2 → 4 | 41.6–52.6 % (2d+4); 30.4–42.1 % (d+2) |

This is the headline limit on a pure skip mask: the reach that semi-Lagrangian transport needs at
the impact peak already puts 45 % of tiles live, and in the spread regime k = 7 is 52–62 % live.
The two-level scheme avoids that because its coarse level is global.

## Caveats

- **Sampling.** 70 advances produced 70 hardware traces; 65 were collected (sample 1 dropped as
  warm-up, ids 52, 54, 60, 66 were overwritten by a newer `info.physicsTrace` before `readStats`
  picked them up). All 65 have `measurementSource = "gpu-hardware-timestamp"` — no declined or
  queue-wall fallback samples, no validation errors, closure error 0.
- **`solver.info` is only filled by `readStats()`**; the probe and census call it after every
  advance. At census step 0 the reduction buffer has never been written, so the *solver-reported*
  `volumeCellSum` and `maxSpeed_m_s` at step 0 are zeros; the readback-computed V sum at step 0 is
  the real one and is used in the tables.
- **Bimodality.** The V-cycle stage is the one genuinely multi-modal stage and the cause is
  convergence gating (6–40 ms on identical code), not measurement noise. p10/p90 are given for
  every phase in every regime in `fig7-regimes.json`. Small stages are quantised to the 65.5 µs
  Dawn tick.
- **Census run-to-run variation.** Three census runs of the same trajectory differ by a few percent
  of tiles (e.g. step 70 seed4h 8 887 / 8 811 / 8 525 across runs, a 4 % spread), because the donor
  column sums use float CAS and are not deterministic — documented behaviour of this transport.
  **All Task 2 tables come from the final run, `fig7-census.json`**; the k=5 run is kept as
  `fig7-census-k5.json`. Conclusions do not turn on differences of this size, but individual cells
  should not be quoted to better than ~5 %.
- **Open capacity is taken as 1.0**, justified rather than measured: `cellOpenFraction` is
  `(1−solidFraction)·(1−terrainFraction)` and `cm12-figure-7` has no rigid bodies, no obstacles and
  no terrain (verified from the scene document). The "partial" column is therefore `0 < V < 1`, and
  since V regularly exceeds 1 (up to 55) it under-counts cells that are not cleanly full.
- **`lostLiquidMass` is the sum of V over cells with centre phi > 0**, not a drift measurement —
  total V is conserved throughout. It says where the conserved volume sits relative to the level
  set, which is the point.
- **Not measured:** any A/B of an actual tile map or two-level scheme (nothing was implemented or
  changed); whether the FIM front's 8-sweep truncation changes the picture at 16 sweeps
  (`uniformFIMExecutedPasses` is 8 on every one of the 70 steps, i.e. the front hits its ceiling
  every step and never converges on this scene); the cost of building or maintaining a tile map, or
  of a coarse level, both of which the two-level fractions above ignore.
- The 64³ comparison mixes three benchmark tools with different brackets and two different sweep
  budgets; per-cell ratios are quoted for the brackets that match (transport, front-per-sweep) and
  flagged where they do not.
