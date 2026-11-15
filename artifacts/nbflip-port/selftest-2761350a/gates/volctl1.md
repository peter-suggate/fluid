# NB-FLIP port gates: dam-volctl1, letters-volctl1, paddleboats-volctl1

Generated 2026-11-15T00:46:29.262Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-ref-e; dam dam-ref-a + dam-ref-b + dam-ref-c + dam-ref-d + dam-ref-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-ref-e.
**Overall: FAIL** (dam FAIL, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## dam: dam-volctl1: FAIL

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.05 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2000 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.03 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1680 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5004 ± 0.0867, tol 4.0000); 16: 78.6250 vs 78.7625 ± 0.0625, tol 4.0000 (restored 80.8008 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.03 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.6250 vs 78.7625 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.21 | 16: -0.0481 vs 0.0438 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.21 | 16: -0.0481 vs -0.0438 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.06 | 4: 36.0848 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.5058 vs 119.4973 ± 0.0237, tol 11.9497; 12: 228.6386 vs 228.6048 ± 0.4230, tol 22.8605; 16: 273.3709 vs 275.0075 ± 1.1711, tol 27.5008 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.28 | 4: 669.1683 vs 669.1684 ± 0.0008, tol 13.3834 (restored 669.9803 ± 0.0009, tol 13.3996); 8: 670.7956 vs 670.7852 ± 0.0429, tol 13.4157 (restored 671.9703 ± 0.0280, tol 13.4394); 12: 664.0471 vs 663.9892 ± 0.7382, tol 13.2798 (restored 667.4997 ± 0.2311, tol 13.3500); 16: 587.6379 vs 590.9589 ± 0.9253, tol 11.8192 (restored 613.4000 ± 3.2444, tol 12.2680) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.28 | 4: 669.1683 vs 669.1684 ± 0.0008, tol 13.3834; 8: 670.7956 vs 670.7852 ± 0.0429, tol 13.4157; 12: 664.0471 vs 663.9892 ± 0.7382, tol 13.2798; 16: 587.6379 vs 590.9589 ± 0.9253, tol 11.8192 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7623.0000 vs 7623.6000 ± 2.0000, tol 381.1800; 8: 10774.0000 vs 10767.4000 ± 4.0000, tol 538.3700; 12: 14939.0000 vs 14936.2000 ± 10.0000, tol 746.8100 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.14 | 8: 0.0094 vs 0.0094 ± 0.0156, tol 4.0000 (restored 0.0929 ± 0.0158, tol 4.0000); 12: 0.0981 vs 0.0981 ± 0.4381, tol 4.0000 (restored 0.2924 ± 0.4252, tol 4.0000); 16: 0.9972 vs 0.9972 ± 2.2985, tol 6.8954 (restored 2.6499 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.14 | 8: 0.0094 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.0981 vs 0.0000 ± 0.4381, tol 4.0000; 16: 0.9972 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.01 | 16: 3.519e+5 vs 3.516e+5 ± 5756.0000, tol 35157.3800 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | **FAIL** | 2.67 | 30-90: 37.6699 vs 45.3007 ± 0.9511, tol 2.8534 (restored 47.1584 ± 0.9143, tol 2.7430) **x**; 105-165: 44.2460 vs 47.0926 ± 0.7294, tol 2.3546 (restored 47.5419 ± 0.6676, tol 2.3771) **x**; 180-300: 46.4691 vs 46.8320 ± 0.5217, tol 2.3416 (restored 47.1994 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (fail, not gated) | 2.67 | 30-90: 37.6699 vs 45.3007 ± 0.9511, tol 2.8534 **x**; 105-165: 44.2460 vs 47.0926 ± 0.7294, tol 2.3546 **x**; 180-300: 46.4691 vs 46.8320 ± 0.5217, tol 2.3416 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | **FAIL** | 5.20 | 30-90: -0.1808 vs 0.0404 ± 0.0090, tol 0.0270 **x**; 105-165: -0.0926 vs 0.0096 ± 0.0082, tol 0.0246 **x**; 180-300: -0.0362 vs 0.0073 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (fail, not gated) | 5.20 | 30-90: -0.1808 vs -0.0404 ± 0.0090, tol 0.0270 **x**; 105-165: -0.0926 vs -0.0096 ± 0.0082, tol 0.0246 **x**; 180-300: -0.0362 vs -0.0073 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | **FAIL** | 1.62 | 30-90: 38.9957 vs 46.5247 ± 1.2119, tol 4.6525 **x**; 105-165: 13.0095 vs 16.6475 ± 4.3305, tol 12.9914; 180-300: 9.6247 vs 10.3905 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | **FAIL** | 5.46 | 30-90: 208.4422 vs 281.4297 ± 4.4590, tol 13.3770 (restored 301.7705 ± 1.1612, tol 6.0354) **x**; 105-165: 224.5475 vs 268.2136 ± 7.5629, tol 22.6888 (restored 273.2416 ± 4.5779, tol 13.7338) **x**; 180-300: 250.0413 vs 265.5091 ± 10.4288, tol 31.2864 (restored 269.6234 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 5.46 | 30-90: 208.4422 vs 281.4297 ± 4.4590, tol 13.3770 **x**; 105-165: 224.5475 vs 268.2136 ± 7.5629, tol 22.6888 **x**; 180-300: 250.0413 vs 265.5091 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | **FAIL** | 6.91 | 30-90: 247.4378 vs 327.9544 ± 3.8834, tol 11.6502 (restored 348.2952 ± 2.0543, tol 6.9659) **x**; 105-165: 237.5570 vs 284.8611 ± 11.8934, tol 35.6801 (restored 289.8892 ± 8.9084, tol 26.7252) **x**; 180-300: 259.6659 vs 275.8995 ± 14.4039, tol 43.2116 (restored 280.0139 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 6.91 | 30-90: 247.4378 vs 327.9544 ± 3.8834, tol 11.6502 **x**; 105-165: 237.5570 vs 284.8611 ± 11.8934, tol 35.6801 **x**; 180-300: 259.6659 vs 275.8995 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.21 | 30-90: 0.0104 vs 0.0104 ± 0.0059, tol 0.0500 (restored 0.0112 ± 0.0058, tol 0.0500); 105-165: 0.0057 vs 0.0057 ± 0.0114, tol 0.0500 (restored 0.0057 ± 0.0113, tol 0.0500); 180-300: 0.0037 vs 0.0037 ± 0.0094, tol 0.0500 (restored 0.0037 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.21 | 30-90: 0.0104 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0057 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0037 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.59 | 30-90: 5.938e+5 vs 6.309e+5 ± 17318.6000, tol 63086.1960; 105-165: 3.538e+5 vs 3.838e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.314e+5 vs 3.484e+5 ± 30193.3333, tol 90580.0000 |

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | - | - | - | - |
| fenced p95 ms | - | - | - | - |
| fenced mean ms | - | - | - | - |
| pipelined ms/step | - | - | - | - |
| fenced thirds ms (first) | - | - | - | - |
| fenced thirds ms (middle) | - | - | - | - |
| fenced thirds ms (last) | - | - | - | - |
| construction ms (fenced run) | - | - | - | - |
| peak GPU MB requested | - | - | - | - |

Timing runs: baseline -, -; candidate -, -.

Check-slot step time (informational): candidate fenced median 39.77 ms; references 67.60, 45.86, 62.58, 66.20, 60.19 ms.

## letters: letters-volctl1: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.53 | 120: -0.0112 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0118 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0124 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.53 | 120: -0.0112 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0118 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0124 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.20 | 60: 0.0304 vs 0.0304 ± 0.0364, tol 1.0000 (restored 0.2671 ± 0.0363, tol 1.0000); 90: 0.0581 vs 0.0581 ± 0.1444, tol 1.0000 (restored 0.4524 ± 0.1444, tol 1.0000); 120: 0.3075 vs 0.3075 ± 0.7010, tol 2.1030 (restored 0.5274 ± 0.7010, tol 2.1029); 180: 0.5854 vs 0.5854 ± 0.9631, tol 2.8893 (restored 0.7464 ± 0.9631, tol 2.8892); 240: 0.5895 vs 0.5895 ± 1.1402, tol 3.4206 (restored 0.7725 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.20 | 60: 0.0304 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.0581 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.3075 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.5854 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.5895 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.08 | 60: 1946.1938 vs 1948.6120 ± 2.2087, tol 194.8612; 120: 2215.9476 vs 2239.5631 ± 101.6255, tol 304.8766; 180: 3005.4274 vs 3053.5803 ± 219.4018, tol 658.2053; 240: 3063.1594 vs 3041.9801 ± 41.3393, tol 304.1980 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.34 | 60: 19.8901 vs 19.8925 ± 0.0023, tol 0.2500 (restored 20.0174 ± 0.0017, tol 0.2500); 120: 20.4554 vs 20.5480 ± 0.1422, tol 0.4266 (restored 20.7089 ± 0.1374, tol 0.4121); 180: 21.6417 vs 21.7277 ± 0.0407, tol 0.2500 (restored 21.8959 ± 0.0390, tol 0.2500); 240: 20.4505 vs 20.5001 ± 0.0231, tol 0.2500 (restored 20.7118 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.34 | 60: 19.8901 vs 19.8925 ± 0.0023, tol 0.2500; 120: 20.4554 vs 20.5480 ± 0.1422, tol 0.4266; 180: 21.6417 vs 21.7277 ± 0.0407, tol 0.2500; 240: 20.4505 vs 20.5001 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.15 | 60: 5.022e+5 vs 5.007e+5 ± 6307.0000, tol 50071.3200; 120: 8.908e+5 vs 8.908e+5 ± 11988.0000, tol 89082.2400; 180: 8.887e+5 vs 9.020e+5 ± 18947.0000, tol 90198.3600; 240: 8.759e+5 vs 8.779e+5 ± 7187.0000, tol 87786.9600 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.53 | 120: -0.0112 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0118 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0124 vs -0.0107 ± 0.0009, tol 0.0050 |

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | - | - | - | - |
| fenced p95 ms | - | - | - | - |
| fenced mean ms | - | - | - | - |
| pipelined ms/step | - | - | - | - |
| fenced thirds ms (first) | - | - | - | - |
| fenced thirds ms (middle) | - | - | - | - |
| fenced thirds ms (last) | - | - | - | - |
| construction ms (fenced run) | - | - | - | - |
| peak GPU MB requested | - | - | - | - |

Timing runs: baseline -, -; candidate -, -.

Check-slot step time (informational): candidate fenced median 88.48 ms; references 123.19, 96.65, 84.54, 87.50, 94.06 ms.

## paddleboats: paddleboats-volctl1: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.00 | 150: -0.0029 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0094 vs 0.0102 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.14 | 150: -0.0029 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0094 vs -0.0102 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.13 | 150: 0.0090 vs 0.0090 ± 0.0176, tol 0.2500 (restored 0.0708 ± 0.0164, tol 0.2500); 300: 0.0376 vs 0.0376 ± 0.0590, tol 0.2500 (restored 0.2289 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.15 | 150: 0.0090 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.0376 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 60: 95.6106 vs 95.2277 ± 1.7659, tol 9.5228; 120: 108.3849 vs 109.1027 ± 1.1784, tol 10.9103; 180: 141.7200 vs 141.8094 ± 1.5890, tol 14.1809; 240: 142.7132 vs 141.9704 ± 1.9409, tol 14.1970; 300: 167.7237 vs 167.2549 ± 2.4230, tol 16.7255 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.05 | 150: 2.608e+5 vs 2.569e+5 ± 26447.0000, tol 79341.0000; 300: 3.038e+5 vs 2.911e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.14 | 150: -0.0083 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0148 vs -0.0155 ± 0.0017, tol 0.0052 |

| step time (timing slot) | baseline | candidate | change ms | change |
|---|---|---|---|---|
| fenced median ms | - | - | - | - |
| fenced p95 ms | - | - | - | - |
| fenced mean ms | - | - | - | - |
| pipelined ms/step | - | - | - | - |
| fenced thirds ms (first) | - | - | - | - |
| fenced thirds ms (middle) | - | - | - | - |
| fenced thirds ms (last) | - | - | - | - |
| construction ms (fenced run) | - | - | - | - |
| peak GPU MB requested | - | - | - | - |

Timing runs: baseline -, -; candidate -, -.

Check-slot step time (informational): candidate fenced median 52.66 ms; references 51.19, 70.16, 50.72, 54.85, 95.74 ms.
