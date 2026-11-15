# NB-FLIP port timing round

Generated 2026-11-15T01:37:44.313Z by tools/nbflip-gates.ts --timing-only. Baseline /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-2761350a-seep2/base; candidate /Users/petersuggate/code/me/fluid/artifacts/nbflip-port/timing-2761350a-seep2/candidate.

## seep (1200 steps)

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | 35.19 | 35.09 | -0.10 | -0.3% |
| fenced p95 ms | 39.89 | 39.00 | -0.89 | -2.2% |
| fenced mean ms | 34.70 | 34.45 | -0.25 | -0.7% |
| pipelined ms/step | 32.42 | 32.30 | -0.12 | -0.4% |
| fenced thirds ms (first) | 30.40 | 30.63 | 0.23 | 0.8% |
| fenced thirds ms (middle) | 35.27 | 35.11 | -0.16 | -0.4% |
| fenced thirds ms (last) | 38.44 | 37.61 | -0.83 | -2.1% |
| construction ms (fenced run) | 10969.74 | 11027.50 | 57.76 | 0.5% |
| peak GPU MB requested | 1186.54 | 1186.54 | 0.00 | 0.0% |

Timing runs: baseline seep-timing-fenced, seep-timing-pipelined; candidate seep-timing-fenced, seep-timing-pipelined.
