# NB-FLIP port timing round

Generated 2026-11-15T01:26:14.939Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-2761350a/base; candidate /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-2761350a/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 38.35 | 35.55 | -2.79 | -7.3% |
| fenced p95 ms | 57.57 | 39.91 | -17.66 | -30.7% |
| fenced mean ms | 40.58 | 34.94 | -5.64 | -13.9% |
| pipelined ms/step | 39.46 | 48.84 | 9.38 | 23.8% |
| fenced thirds ms (first) | 38.16 | 30.63 | -7.53 | -19.7% |
| fenced thirds ms (middle) | 40.10 | 35.80 | -4.29 | -10.7% |
| fenced thirds ms (last) | 43.49 | 38.39 | -5.10 | -11.7% |
| construction ms (fenced run) | 31941.01 | 10938.41 | -21002.60 | -65.8% |
| peak GPU MB requested | 1186.54 | 1186.54 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 44.50 | 45.45 | 0.95 | 2.1% |
| fenced p95 ms | 67.83 | 76.88 | 9.05 | 13.3% |
| fenced mean ms | 49.20 | 51.49 | 2.29 | 4.6% |
| pipelined ms/step | 49.66 | 50.61 | 0.95 | 1.9% |
| fenced thirds ms (first) | 61.80 | 67.06 | 5.26 | 8.5% |
| fenced thirds ms (middle) | 44.44 | 43.54 | -0.90 | -2.0% |
| fenced thirds ms (last) | 41.37 | 43.86 | 2.49 | 6.0% |
| construction ms (fenced run) | 8789.37 | 9045.43 | 256.06 | 2.9% |
| peak GPU MB requested | 1921.80 | 1921.80 | 0.00 | 0.0% |

Timing runs: baseline dam-timing-fenced, dam-timing-pipelined; candidate dam-timing-fenced, dam-timing-pipelined.

## letters (240 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 86.50 | 84.81 | -1.69 | -1.9% |
| fenced p95 ms | 95.82 | 96.70 | 0.88 | 0.9% |
| fenced mean ms | 72.93 | 71.81 | -1.12 | -1.5% |
| pipelined ms/step | 67.82 | 69.57 | 1.74 | 2.6% |
| fenced thirds ms (first) | 41.94 | 40.50 | -1.44 | -3.4% |
| fenced thirds ms (middle) | 84.89 | 83.07 | -1.82 | -2.1% |
| fenced thirds ms (last) | 91.96 | 91.87 | -0.09 | -0.1% |
| construction ms (fenced run) | 8964.61 | 9303.97 | 339.36 | 3.8% |
| peak GPU MB requested | 2022.78 | 2022.78 | 0.00 | 0.0% |

Timing runs: baseline letters-timing-fenced, letters-timing-pipelined; candidate letters-timing-fenced, letters-timing-pipelined.

## paddleboats (300 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 50.37 | 50.50 | 0.12 | 0.2% |
| fenced p95 ms | 53.59 | 55.47 | 1.88 | 3.5% |
| fenced mean ms | 48.15 | 48.78 | 0.63 | 1.3% |
| pipelined ms/step | 44.15 | 45.22 | 1.07 | 2.4% |
| fenced thirds ms (first) | 41.27 | 41.53 | 0.25 | 0.6% |
| fenced thirds ms (middle) | 51.32 | 50.98 | -0.34 | -0.7% |
| fenced thirds ms (last) | 51.77 | 53.74 | 1.97 | 3.8% |
| construction ms (fenced run) | 9094.86 | 9090.59 | -4.27 | -0.0% |
| peak GPU MB requested | 3228.47 | 3228.47 | 0.00 | 0.0% |

Timing runs: baseline paddleboats-timing-fenced, paddleboats-timing-pipelined; candidate paddleboats-timing-fenced, paddleboats-timing-pipelined.
