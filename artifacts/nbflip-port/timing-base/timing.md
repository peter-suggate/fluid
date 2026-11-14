# NB-FLIP port timing round

Generated 2026-11-14T23:08:38.311Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-base/base; candidate /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-base/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 34.23 | 35.00 | 0.76 | 2.2% |
| fenced p95 ms | 38.21 | 40.97 | 2.76 | 7.2% |
| fenced mean ms | 33.83 | 35.04 | 1.21 | 3.6% |
| pipelined ms/step | 33.12 | 32.86 | -0.26 | -0.8% |
| fenced thirds ms (first) | 30.02 | 30.80 | 0.78 | 2.6% |
| fenced thirds ms (middle) | 34.33 | 35.71 | 1.38 | 4.0% |
| fenced thirds ms (last) | 37.15 | 38.62 | 1.46 | 3.9% |
| construction ms (fenced run) | 10945.17 | 11097.48 | 152.31 | 1.4% |
| peak GPU MB requested | 1186.54 | 1186.54 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | - | - | - | - |
| fenced p95 ms | - | - | - | - |
| fenced mean ms | - | - | - | - |
| pipelined ms/step | 45.37 | 44.71 | -0.66 | -1.5% |
| fenced thirds ms (first) | - | - | - | - |
| fenced thirds ms (middle) | - | - | - | - |
| fenced thirds ms (last) | - | - | - | - |
| construction ms (fenced run) | - | - | - | - |
| peak GPU MB requested | - | - | - | - |

Timing runs: baseline -, dam-timing-pipelined; candidate -, dam-timing-pipelined.

## letters (240 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 83.05 | 85.10 | 2.05 | 2.5% |
| fenced p95 ms | 95.90 | 94.51 | -1.39 | -1.4% |
| fenced mean ms | 71.11 | 71.42 | 0.31 | 0.4% |
| pipelined ms/step | 68.42 | 69.26 | 0.83 | 1.2% |
| fenced thirds ms (first) | 40.34 | 40.16 | -0.18 | -0.4% |
| fenced thirds ms (middle) | 82.27 | 84.47 | 2.19 | 2.7% |
| fenced thirds ms (last) | 90.71 | 89.63 | -1.08 | -1.2% |
| construction ms (fenced run) | 8960.73 | 9088.46 | 127.73 | 1.4% |
| peak GPU MB requested | 2022.78 | 2022.78 | 0.00 | 0.0% |

Timing runs: baseline letters-timing-fenced, letters-timing-pipelined; candidate letters-timing-fenced, letters-timing-pipelined.

## paddleboats (300 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 49.26 | 50.22 | 0.97 | 2.0% |
| fenced p95 ms | 55.58 | 53.05 | -2.54 | -4.6% |
| fenced mean ms | 47.48 | 47.15 | -0.33 | -0.7% |
| pipelined ms/step | 43.66 | 44.46 | 0.80 | 1.8% |
| fenced thirds ms (first) | 39.84 | 39.85 | 0.01 | 0.0% |
| fenced thirds ms (middle) | 49.08 | 50.09 | 1.01 | 2.1% |
| fenced thirds ms (last) | 53.42 | 51.41 | -2.01 | -3.8% |
| construction ms (fenced run) | 9255.32 | 9448.59 | 193.28 | 2.1% |
| peak GPU MB requested | 3228.47 | 3228.47 | 0.00 | 0.0% |

Timing runs: baseline paddleboats-timing-fenced, paddleboats-timing-pipelined; candidate paddleboats-timing-fenced, paddleboats-timing-pipelined.
