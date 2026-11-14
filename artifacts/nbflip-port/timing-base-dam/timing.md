# NB-FLIP port timing round

Generated 2026-11-14T23:11:13.303Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-base-dam/base; candidate /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-base-dam/candidate.

## dam (165 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 45.12 | 46.25 | 1.13 | 2.5% |
| fenced p95 ms | 72.76 | 74.53 | 1.77 | 2.4% |
| fenced mean ms | 51.14 | 51.54 | 0.39 | 0.8% |
| pipelined ms/step | 50.29 | 49.18 | -1.11 | -2.2% |
| fenced thirds ms (first) | 65.30 | 65.39 | 0.09 | 0.1% |
| fenced thirds ms (middle) | 44.74 | 45.70 | 0.96 | 2.1% |
| fenced thirds ms (last) | 43.38 | 43.52 | 0.14 | 0.3% |
| construction ms (fenced run) | 8976.05 | 8974.36 | -1.69 | -0.0% |
| peak GPU MB requested | 1921.80 | 1921.80 | 0.00 | 0.0% |

Timing runs: baseline dam-timing-fenced, dam-timing-pipelined; candidate dam-timing-fenced, dam-timing-pipelined.
