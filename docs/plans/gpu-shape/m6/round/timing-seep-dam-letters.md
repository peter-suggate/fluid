# NB-FLIP port timing round

Generated 2026-11-23T22:33:53.648Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m6/round/base; candidate /Users/petersuggate/code/me/fluid-scratch/gpu-shape/m6/round/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 26.04 | 24.73 | -1.31 | -5.0% |
| fenced p95 ms | 27.68 | 26.41 | -1.27 | -4.6% |
| fenced mean ms | 25.56 | 24.36 | -1.20 | -4.7% |
| pipelined ms/step | 24.47 | 23.73 | -0.74 | -3.0% |
| paired: fenced median (base first) | 26.04 | 24.73 | -1.31 | -5.0% |
| paired: pipelined (candidate first) | 24.47 | 23.73 | -0.74 | -3.0% |
| **order-balanced change** (mean of the two pairs) | | | | **-4.0%** |
| fenced thirds ms (first) | 23.49 | 22.43 | -1.06 | -4.5% |
| fenced thirds ms (middle) | 26.37 | 25.09 | -1.28 | -4.9% |
| fenced thirds ms (last) | 26.82 | 25.56 | -1.26 | -4.7% |
| construction ms (fenced run) | 4759.99 | 4938.38 | 178.40 | 3.7% |
| peak GPU MB requested | 1079.45 | 1079.45 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 33.88 | 31.81 | -2.06 | -6.1% |
| fenced p95 ms | 56.45 | 51.63 | -4.82 | -8.5% |
| fenced mean ms | 37.74 | 35.28 | -2.46 | -6.5% |
| pipelined ms/step | 35.37 | 33.73 | -1.64 | -4.6% |
| paired: fenced median (candidate first) | 33.88 | 31.81 | -2.06 | -6.1% |
| paired: pipelined (base first) | 35.37 | 33.73 | -1.64 | -4.6% |
| **order-balanced change** (mean of the two pairs) | | | | **-5.4%** |
| fenced thirds ms (first) | 49.17 | 45.78 | -3.40 | -6.9% |
| fenced thirds ms (middle) | 33.85 | 32.25 | -1.60 | -4.7% |
| fenced thirds ms (last) | 30.19 | 27.80 | -2.39 | -7.9% |
| construction ms (fenced run) | 2881.96 | 2863.24 | -18.72 | -0.6% |
| peak GPU MB requested | 845.89 | 845.89 | 0.00 | 0.0% |

Timing runs: baseline dam-timing-fenced, dam-timing-pipelined; candidate dam-timing-fenced, dam-timing-pipelined.

## letters (240 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 54.10 | 50.45 | -3.65 | -6.7% |
| fenced p95 ms | 62.50 | 57.59 | -4.90 | -7.8% |
| fenced mean ms | 47.11 | 43.86 | -3.25 | -6.9% |
| pipelined ms/step | 44.29 | 42.24 | -2.05 | -4.6% |
| paired: fenced median (base first) | 54.10 | 50.45 | -3.65 | -6.7% |
| paired: pipelined (candidate first) | 44.29 | 42.24 | -2.05 | -4.6% |
| **order-balanced change** (mean of the two pairs) | | | | **-5.7%** |
| fenced thirds ms (first) | 27.66 | 26.26 | -1.40 | -5.1% |
| fenced thirds ms (middle) | 53.47 | 49.74 | -3.73 | -7.0% |
| fenced thirds ms (last) | 60.19 | 55.59 | -4.60 | -7.6% |
| construction ms (fenced run) | 3011.39 | 3063.35 | 51.95 | 1.7% |
| peak GPU MB requested | 1490.25 | 1490.25 | 0.00 | 0.0% |

Timing runs: baseline letters-timing-fenced, letters-timing-pipelined; candidate letters-timing-fenced, letters-timing-pipelined.

## paddleboats (300 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | - | 34.68 | - | - |
| fenced p95 ms | - | 36.45 | - | - |
| fenced mean ms | - | 34.63 | - | - |
| pipelined ms/step | - | 33.03 | - | - |
| fenced thirds ms (first) | - | 34.00 | - | - |
| fenced thirds ms (middle) | - | 34.79 | - | - |
| fenced thirds ms (last) | - | 35.09 | - | - |
| construction ms (fenced run) | - | 6237.82 | - | - |
| peak GPU MB requested | - | 1650.14 | - | - |

Timing runs: baseline -, -; candidate paddleboats-timing-fenced, paddleboats-timing-pipelined.
