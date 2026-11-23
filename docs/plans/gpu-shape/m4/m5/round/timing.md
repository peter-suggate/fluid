# NB-FLIP port timing round

Generated 2026-11-23T21:38:42.090Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m5/round/base; candidate /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m5/round/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 25.62 | 25.61 | -0.01 | -0.1% |
| fenced p95 ms | 26.95 | 28.94 | 1.99 | 7.4% |
| fenced mean ms | 24.84 | 25.27 | 0.43 | 1.7% |
| pipelined ms/step | 24.70 | 23.99 | -0.72 | -2.9% |
| paired: fenced median (base first) | 25.62 | 25.61 | -0.01 | -0.1% |
| paired: pipelined (candidate first) | 24.70 | 23.99 | -0.72 | -2.9% |
| **order-balanced change** (mean of the two pairs) | | | | **-1.5%** |
| fenced thirds ms (first) | 22.54 | 22.52 | -0.02 | -0.1% |
| fenced thirds ms (middle) | 25.85 | 25.66 | -0.19 | -0.7% |
| fenced thirds ms (last) | 26.14 | 27.63 | 1.49 | 5.7% |
| construction ms (fenced run) | 4708.45 | 4694.94 | -13.51 | -0.3% |
| peak GPU MB requested | 1079.45 | 1079.45 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 32.13 | 32.01 | -0.13 | -0.4% |
| fenced p95 ms | 53.57 | 51.15 | -2.42 | -4.5% |
| fenced mean ms | 36.43 | 35.73 | -0.70 | -1.9% |
| pipelined ms/step | 34.56 | 33.71 | -0.85 | -2.5% |
| paired: fenced median (candidate first) | 32.13 | 32.01 | -0.13 | -0.4% |
| paired: pipelined (base first) | 34.56 | 33.71 | -0.85 | -2.5% |
| **order-balanced change** (mean of the two pairs) | | | | **-1.4%** |
| fenced thirds ms (first) | 47.10 | 45.53 | -1.57 | -3.3% |
| fenced thirds ms (middle) | 32.55 | 32.54 | -0.00 | -0.0% |
| fenced thirds ms (last) | 29.63 | 29.10 | -0.53 | -1.8% |
| construction ms (fenced run) | 2878.47 | 2921.79 | 43.32 | 1.5% |
| peak GPU MB requested | 845.89 | 845.89 | 0.00 | 0.0% |

Timing runs: baseline dam-timing-fenced, dam-timing-pipelined; candidate dam-timing-fenced, dam-timing-pipelined.

## letters (240 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 51.21 | 51.79 | 0.58 | 1.1% |
| fenced p95 ms | 59.49 | 59.23 | -0.27 | -0.4% |
| fenced mean ms | 44.94 | 44.94 | -0.01 | -0.0% |
| pipelined ms/step | 42.95 | 42.41 | -0.54 | -1.3% |
| paired: fenced median (base first) | 51.21 | 51.79 | 0.58 | 1.1% |
| paired: pipelined (candidate first) | 42.95 | 42.41 | -0.54 | -1.3% |
| **order-balanced change** (mean of the two pairs) | | | | **-0.1%** |
| fenced thirds ms (first) | 26.63 | 26.43 | -0.20 | -0.7% |
| fenced thirds ms (middle) | 50.75 | 51.19 | 0.44 | 0.9% |
| fenced thirds ms (last) | 57.44 | 57.19 | -0.26 | -0.4% |
| construction ms (fenced run) | 3022.84 | 3013.46 | -9.38 | -0.3% |
| peak GPU MB requested | 1490.25 | 1490.25 | 0.00 | 0.0% |

Timing runs: baseline letters-timing-fenced, letters-timing-pipelined; candidate letters-timing-fenced, letters-timing-pipelined.

## paddleboats (300 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 35.63 | 35.34 | -0.29 | -0.8% |
| fenced p95 ms | 37.32 | 37.60 | 0.28 | 0.7% |
| fenced mean ms | 35.56 | 35.40 | -0.16 | -0.4% |
| pipelined ms/step | 34.15 | 33.55 | -0.60 | -1.8% |
| paired: fenced median (candidate first) | 35.63 | 35.34 | -0.29 | -0.8% |
| paired: pipelined (base first) | 34.15 | 33.55 | -0.60 | -1.8% |
| **order-balanced change** (mean of the two pairs) | | | | **-1.3%** |
| fenced thirds ms (first) | 34.76 | 34.55 | -0.20 | -0.6% |
| fenced thirds ms (middle) | 35.81 | 35.57 | -0.24 | -0.7% |
| fenced thirds ms (last) | 36.10 | 36.07 | -0.03 | -0.1% |
| construction ms (fenced run) | 6131.81 | 6114.76 | -17.05 | -0.3% |
| peak GPU MB requested | 1650.14 | 1650.14 | 0.00 | 0.0% |

Timing runs: baseline paddleboats-timing-fenced, paddleboats-timing-pipelined; candidate paddleboats-timing-fenced, paddleboats-timing-pipelined.
