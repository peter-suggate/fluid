# NB-FLIP port gates: dam-base-e, letters-base-e, paddleboats-base-e, seep-base-e

Generated 2026-11-14T23:10:00.173Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d; dam dam-ref-a + dam-ref-b + dam-ref-d; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).

## seep: seep-base-e: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.00 | mean: 0.2697 vs 0.2698 ± 0.0002, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.08 | 120-180: 0.2710 vs 0.2689 ± 0.0011, tol 0.0269; 180-240: 0.2726 vs 0.2713 ± 0.0021, tol 0.0271; 240-300: 0.2698 vs 0.2710 ± 0.0009, tol 0.0271; 300-360: 0.2725 vs 0.2728 ± 0.0018, tol 0.0273; 360-420: 0.2707 vs 0.2717 ± 0.0021, tol 0.0272; 420-480: 0.2722 vs 0.2724 ± 0.0006, tol 0.0272; 480-540: 0.2712 vs 0.2709 ± 0.0007, tol 0.0271; 540-600: 0.2703 vs 0.2715 ± 0.0010, tol 0.0271; 600-660: 0.2708 vs 0.2703 ± 0.0008, tol 0.0270; 660-720: 0.2695 vs 0.2700 ± 0.0012, tol 0.0270; 720-780: 0.2697 vs 0.2695 ± 0.0012, tol 0.0270; 780-840: 0.2687 vs 0.2688 ± 0.0013, tol 0.0269; 840-900: 0.2682 vs 0.2689 ± 0.0017, tol 0.0269; 900-960: 0.2683 vs 0.2680 ± 0.0005, tol 0.0268; 960-1020: 0.2681 vs 0.2685 ± 0.0010, tol 0.0269; 1020-1080: 0.2682 vs 0.2674 ± 0.0005, tol 0.0267; 1080-1140: 0.2666 vs 0.2672 ± 0.0018, tol 0.0267; 1140-1200: 0.2672 vs 0.2665 ± 0.0022, tol 0.0266 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.13 | 300: 1.3276 vs 1.3192 ± 0.0161, tol 0.0660; 600: 2.6845 vs 2.6786 ± 0.0151, tol 0.1339; 900: 4.0314 vs 4.0261 ± 0.0145, tol 0.2013; 1200: 5.3698 vs 5.3637 ± 0.0141, tol 0.2682 |
| G3 | right tank volume (L) | |c - ref| <= max(5%, 3S) | pass | 0.42 | 300: 1.1558 vs 1.1562 ± 0.0701, tol 0.2103; 600: 2.1640 vs 2.0963 ± 0.0540, tol 0.1621; 900: 2.9053 vs 2.8655 ± 0.1074, tol 0.3223; 1200: 3.4006 vs 3.3740 ± 0.1684, tol 0.5053 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.00 | median: 2.6386 vs 2.6387 ± 0.0043, tol 0.2639 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.03 | 90%: 289.8355 vs 292.3502 ± 3.6068, tol 73.0876 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.02 | 1200: 16368.0000 vs 16348.5000 ± 72.0000, tol 817.4250 |
| G6c | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | pass | 0.05 | 600: 368.1540 vs 367.5113 ± 2.7158, tol 16.0000; 1200: 369.3996 vs 368.4382 ± 6.6246, tol 19.8737 |
| G6 | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | pass | 0.19 | 600: 0.0830 vs 0.0000 ± 0.1428, tol 0.4283; 1200: 0.0509 vs 0.0000 ± 0.0992, tol 0.2977 |
| G7 | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | pass | 0.04 | 1200: -0.0050 vs -0.0051 ± 0.0004, tol 0.0013 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.06 | 300: 381.2621 vs 381.4049 ± 3.4907, tol 10.4722; 600: 381.3747 vs 380.7150 ± 3.5712, tol 10.7137 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.20 | 300: 0.4376 vs 0.0000 ± 1.3113, tol 3.9339; 600: 0.6361 vs 0.0000 ± 1.0738, tol 3.2214 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.40 | 300: 69205.0000 vs 69019.5000 ± 4565.0000, tol 13695.0000; 600: 1.343e+5 vs 1.288e+5 ± 4549.0000, tol 13647.0000; 900: 1.860e+5 vs 1.824e+5 ± 3746.0000, tol 18241.8500; 1200: 2.192e+5 vs 2.171e+5 ± 14011.0000, tol 42033.0000 |

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

Check-slot step time (informational): candidate fenced median 34.40 ms; references 37.24, 37.57, 35.16, 65.34 ms.

## dam: dam-base-e: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 165 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| C0f | completed 300 steps with no solver failure | required | (pass, not gated) | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.08 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 107.6667 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | pass | 0.10 | 16: -0.0427 vs -0.0449 ± 0.0078, tol 0.0234 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.03 | 4: 36.0847 vs 36.0848 ± 0.0001, tol 3.6085; 8: 119.4965 vs 119.5041 ± 0.0185, tol 11.9504; 12: 228.6593 vs 228.6979 ± 0.1055, tol 22.8698; 16: 274.8964 vs 274.0942 ± 2.1527, tol 27.4094 |
| D5ce | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | pass | 0.11 | 4: 669.1682 vs 669.1671 ± 0.0033, tol 13.3833; 8: 670.7881 vs 670.7952 ± 0.0191, tol 13.4159; 12: 664.0774 vs 664.1052 ± 0.1563, tol 13.2821; 16: 591.0498 vs 589.6615 ± 4.1982, tol 12.5946 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7624.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10768.0000 vs 10764.0000 ± 8.0000, tol 538.2000; 12: 14936.0000 vs 14938.3333 ± 5.0000, tol 746.9167 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | pass | 0.10 | 8: 0.0483 vs 0.0000 ± 0.1449, tol 4.0000; 12: 0.4070 vs 0.0000 ± 0.0349, tol 4.0000; 16: 0.2434 vs 0.0000 ± 1.6789, tol 5.0368 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.14 | 16: 3.545e+5 vs 3.497e+5 ± 8570.0000, tol 34972.9000 |
| D3 | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | pass | 0.37 | 30-90: 44.1706 vs 45.0049 ± 0.1992, tol 2.2502; 105-165: 46.1663 vs 46.9517 ± 1.1092, tol 3.3275 |
| D4 | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | pass | 0.22 | 30-90: -0.0366 vs -0.0403 ± 0.0055, tol 0.0164; 105-165: -0.0120 vs -0.0108 ± 0.0004, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.31 | 30-90: 46.1679 vs 46.2259 ± 2.3587, tol 7.0762; 105-165: 13.6550 vs 15.4973 ± 1.9563, tol 5.8688 |
| D5p | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | pass | 0.32 | 30-90: 282.1385 vs 280.3223 ± 0.9893, tol 5.6064; 105-165: 263.7671 vs 266.5018 ± 2.8596, tol 8.5789 |
| D5e | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | pass | 0.32 | 30-90: 328.3065 vs 326.5482 ± 1.8413, tol 6.5310; 105-165: 277.4221 vs 281.9990 ± 4.8159, tol 14.4477 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | pass | 0.10 | 30-90: 0.0049 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0048 vs 0.0000 ± 0.0075, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.09 | 30-90: 6.165e+5 vs 6.255e+5 ± 41605.2000, tol 1.248e+5; 105-165: 3.765e+5 vs 3.847e+5 ± 30040.4000, tol 90121.2000 |
| D3+ | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.39 | 180-300: 45.9651 vs 46.8811 ± 0.2855, tol 2.3441 |
| D4+ | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.15 | 180-300: -0.0110 vs -0.0096 ± 0.0019, tol 0.0100 |
| D5k+ | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | (pass, not gated) | 0.38 | 180-300: 6.9588 vs 9.5106 ± 2.2175, tol 6.6526 |
| D5p+ | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.34 | 180-300: 259.9502 vs 263.5621 ± 3.4967, tol 10.4902 |
| D5e+ | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.36 | 180-300: 266.9089 vs 273.0726 ± 5.7143, tol 17.1428 |
| D7d+ | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 180-300: 0.0059 vs 0.0000 ± 0.0043, tol 0.0500 |
| D8+ | particles, window mean | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.22 | 180-300: 3.321e+5 vs 3.515e+5 ± 29223.4444, tol 87670.3333 |

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

Check-slot step time (informational): candidate fenced median 63.34 ms; references 61.07, 62.16, 38.18 ms.

## letters: letters-base-e: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | pass | 0.06 | 120: -0.0086 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0095 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0112 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | pass | 0.26 | 60: 0.0362 vs 0.0000 ± 0.0604, tol 1.0000; 90: 0.1690 vs 0.0000 ± 0.1069, tol 1.0000; 120: 0.4302 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.7201 vs 0.0000 ± 0.9237, tol 2.7711; 240: 0.8908 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.44 | 60: 1948.6154 vs 1944.4019 ± 13.8771, tol 194.4402; 120: 2311.3293 vs 2267.4229 ± 197.0611, tol 591.1833; 180: 3110.6951 vs 3067.4426 ± 169.7827, tol 509.3480; 240: 2941.1081 vs 3077.0938 ± 95.4209, tol 307.7094 |
| L4 | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | pass | 0.12 | 60: 19.8913 vs 19.8956 ± 0.0099, tol 0.2500; 120: 20.6105 vs 20.5694 ± 0.2329, tol 0.6988; 180: 21.6872 vs 21.7167 ± 0.0432, tol 0.2500; 240: 20.4782 vs 20.4897 ± 0.0208, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.09 | 60: 4.983e+5 vs 5.008e+5 ± 8558.0000, tol 50075.2500; 120: 8.998e+5 vs 8.916e+5 ± 22398.0000, tol 89156.2500; 180: 9.101e+5 vs 9.068e+5 ± 19212.0000, tol 90682.4250; 240: 8.665e+5 vs 8.741e+5 ± 17206.0000, tol 87410.7500 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.06 | 120: -0.0086 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0094 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0112 vs -0.0108 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 84.70 ms; references 124.54, 122.69, 127.79, 84.53 ms.

## paddleboats: paddleboats-base-e: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | pass | 0.16 | 150: -0.0031 vs -0.0029 ± 0.0001, tol 0.0025; 300: -0.0102 vs -0.0098 ± 0.0006, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | pass | 0.11 | 150: 0.0156 vs 0.0000 ± 0.0186, tol 0.2500; 300: 0.0284 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.28 | 60: 92.9558 vs 95.6460 ± 1.5841, tol 9.5646; 120: 109.7347 vs 109.5078 ± 2.0263, tol 10.9508; 180: 142.0169 vs 142.2730 ± 1.2921, tol 14.2273; 240: 142.3579 vs 141.8073 ± 4.0339, tol 14.1807; 300: 169.8768 vs 167.1472 ± 1.9815, tol 16.7147 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.13 | 150: 2.653e+5 vs 2.624e+5 ± 11716.0000, tol 35148.0000; 300: 3.046e+5 vs 2.950e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.17 | 150: -0.0084 vs -0.0083 ± 0.0001, tol 0.0025; 300: -0.0155 vs -0.0151 ± 0.0006, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 50.48 ms; references 111.75, 88.86, 94.75, 97.52 ms.
