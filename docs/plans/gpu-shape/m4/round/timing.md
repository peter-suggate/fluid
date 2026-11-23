# NB-FLIP port timing round

Generated 2026-11-23T20:37:46.300Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m4/round/out/base; candidate /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m4/round/out/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 25.97 | 25.96 | -0.01 | -0.0% |
| fenced p95 ms | 27.36 | 27.55 | 0.20 | 0.7% |
| fenced mean ms | 25.25 | 25.42 | 0.18 | 0.7% |
| pipelined ms/step | 25.92 | 24.91 | -1.02 | -3.9% |
| paired: fenced median (base first) | 25.97 | 25.96 | -0.01 | -0.0% |
| paired: pipelined (candidate first) | 25.92 | 24.91 | -1.02 | -3.9% |
| **order-balanced change** (mean of the two pairs) | | | | **-2.0%** |
| fenced thirds ms (first) | 23.21 | 23.31 | 0.10 | 0.4% |
| fenced thirds ms (middle) | 26.26 | 26.30 | 0.04 | 0.1% |
| fenced thirds ms (last) | 26.27 | 26.67 | 0.40 | 1.5% |
| construction ms (fenced run) | 4727.11 | 12884.04 | 8156.93 | 172.6% |
| peak GPU MB requested | 1079.45 | 1079.45 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 33.08 | 32.79 | -0.29 | -0.9% |
| fenced p95 ms | 53.60 | 53.40 | -0.20 | -0.4% |
| fenced mean ms | 37.05 | 36.58 | -0.47 | -1.3% |
| pipelined ms/step | 35.63 | 34.74 | -0.89 | -2.5% |
| paired: fenced median (candidate first) | 33.08 | 32.79 | -0.29 | -0.9% |
| paired: pipelined (base first) | 35.63 | 34.74 | -0.89 | -2.5% |
| **order-balanced change** (mean of the two pairs) | | | | **-1.7%** |
| fenced thirds ms (first) | 47.64 | 47.23 | -0.40 | -0.8% |
| fenced thirds ms (middle) | 33.29 | 32.78 | -0.51 | -1.5% |
| fenced thirds ms (last) | 30.22 | 29.72 | -0.50 | -1.7% |
| construction ms (fenced run) | 2963.20 | 2923.63 | -39.57 | -1.3% |
| peak GPU MB requested | 845.89 | 845.89 | 0.00 | 0.0% |

Timing runs: baseline dam-timing-fenced, dam-timing-pipelined; candidate dam-timing-fenced, dam-timing-pipelined.

## letters (240 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 53.08 | 51.74 | -1.34 | -2.5% |
| fenced p95 ms | 60.49 | 60.45 | -0.04 | -0.1% |
| fenced mean ms | 45.98 | 45.51 | -0.47 | -1.0% |
| pipelined ms/step | 43.53 | 42.96 | -0.57 | -1.3% |
| paired: fenced median (base first) | 53.08 | 51.74 | -1.34 | -2.5% |
| paired: pipelined (candidate first) | 43.53 | 42.96 | -0.57 | -1.3% |
| **order-balanced change** (mean of the two pairs) | | | | **-1.9%** |
| fenced thirds ms (first) | 27.16 | 26.60 | -0.57 | -2.1% |
| fenced thirds ms (middle) | 52.18 | 51.33 | -0.85 | -1.6% |
| fenced thirds ms (last) | 58.59 | 58.61 | 0.02 | 0.0% |
| construction ms (fenced run) | 3025.72 | 11360.09 | 8334.37 | 275.5% |
| peak GPU MB requested | 1490.25 | 1490.25 | 0.00 | 0.0% |

Timing runs: baseline letters-timing-fenced, letters-timing-pipelined; candidate letters-timing-fenced, letters-timing-pipelined.

## paddleboats (300 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 36.38 | 36.00 | -0.38 | -1.0% |
| fenced p95 ms | 37.76 | 37.64 | -0.12 | -0.3% |
| fenced mean ms | 36.28 | 35.92 | -0.36 | -1.0% |
| pipelined ms/step | 34.73 | 34.60 | -0.13 | -0.4% |
| paired: fenced median (candidate first) | 36.38 | 36.00 | -0.38 | -1.0% |
| paired: pipelined (base first) | 34.73 | 34.60 | -0.13 | -0.4% |
| **order-balanced change** (mean of the two pairs) | | | | **-0.7%** |
| fenced thirds ms (first) | 35.53 | 35.08 | -0.45 | -1.3% |
| fenced thirds ms (middle) | 36.51 | 36.24 | -0.28 | -0.8% |
| fenced thirds ms (last) | 36.79 | 36.42 | -0.36 | -1.0% |
| construction ms (fenced run) | 6161.16 | 6133.55 | -27.61 | -0.4% |
| peak GPU MB requested | 1650.14 | 1650.14 | 0.00 | 0.0% |

Timing runs: baseline paddleboats-timing-fenced, paddleboats-timing-pipelined; candidate paddleboats-timing-fenced, paddleboats-timing-pipelined.
