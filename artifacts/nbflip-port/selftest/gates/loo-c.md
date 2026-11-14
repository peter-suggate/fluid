# NB-FLIP port gates: letters-ref-c, paddleboats-ref-c, seep-ref-c

Generated 2026-11-14T23:09:59.868Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-d + seep-base-e; dam dam-ref-a + dam-ref-b + dam-ref-d + dam-base-e; letters letters-ref-a + letters-ref-b + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-d + paddleboats-base-e.
**Overall: PASS** (seep pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).

## seep: seep-ref-c: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.00 | mean: 0.2698 vs 0.2698 ± 0.0002, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.05 | 120-180: 0.2695 vs 0.2693 ± 0.0025, tol 0.0269; 180-240: 0.2725 vs 0.2713 ± 0.0022, tol 0.0271; 240-300: 0.2706 vs 0.2708 ± 0.0018, tol 0.0271; 300-360: 0.2725 vs 0.2728 ± 0.0018, tol 0.0273; 360-420: 0.2712 vs 0.2716 ± 0.0024, tol 0.0272; 420-480: 0.2723 vs 0.2724 ± 0.0006, tol 0.0272; 480-540: 0.2707 vs 0.2710 ± 0.0007, tol 0.0271; 540-600: 0.2718 vs 0.2711 ± 0.0013, tol 0.0271; 600-660: 0.2700 vs 0.2705 ± 0.0007, tol 0.0271; 660-720: 0.2696 vs 0.2700 ± 0.0013, tol 0.0270; 720-780: 0.2697 vs 0.2695 ± 0.0012, tol 0.0270; 780-840: 0.2687 vs 0.2688 ± 0.0013, tol 0.0269; 840-900: 0.2697 vs 0.2685 ± 0.0011, tol 0.0268; 900-960: 0.2683 vs 0.2680 ± 0.0005, tol 0.0268; 960-1020: 0.2680 vs 0.2685 ± 0.0009, tol 0.0269; 1020-1080: 0.2675 vs 0.2676 ± 0.0011, tol 0.0268; 1080-1140: 0.2670 vs 0.2671 ± 0.0018, tol 0.0267; 1140-1200: 0.2666 vs 0.2666 ± 0.0022, tol 0.0267 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.14 | 300: 1.3281 vs 1.3191 ± 0.0157, tol 0.0660; 600: 2.6866 vs 2.6780 ± 0.0129, tol 0.1339; 900: 4.0343 vs 4.0254 ± 0.0116, tol 0.2013; 1200: 5.3716 vs 5.3633 ± 0.0123, tol 0.2682 |
| G3 | right tank volume (L) | |c - ref| <= max(5%, 3S) | pass | 0.20 | 300: 1.1694 vs 1.1528 ± 0.0701, tol 0.2103; 600: 2.0792 vs 2.1175 ± 0.0928, tol 0.2784; 900: 2.9188 vs 2.8621 ± 0.0939, tol 0.2817; 1200: 3.4368 vs 3.3650 ± 0.1322, tol 0.3967 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6363 vs 2.6393 ± 0.0028, tol 0.2639 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.03 | 90%: 290.3774 vs 292.2148 ± 4.1487, tol 73.0537 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.03 | 1200: 16375.0000 vs 16346.7500 ± 65.0000, tol 817.3375 |
| G6c | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | pass | 0.04 | 600: 367.6459 vs 367.6384 ± 2.7158, tol 16.0000; 1200: 369.2464 vs 368.4765 ± 6.6246, tol 19.8737 |
| G6 | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | pass | 0.19 | 600: 0.0832 vs 0.0000 ± 0.1428, tol 0.4283; 1200: 0.0324 vs 0.0000 ± 0.0992, tol 0.2977 |
| G7 | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | pass | 0.17 | 1200: -0.0049 vs -0.0051 ± 0.0003, tol 0.0010 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.27 | 300: 383.1081 vs 380.9434 ± 2.6155, tol 8.0000; 600: 380.7702 vs 380.8662 ± 3.5712, tol 10.7137 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.20 | 300: 0.8044 vs 0.0000 ± 1.3104, tol 3.9311; 600: 0.6346 vs 0.0000 ± 1.0738, tol 3.2214 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.31 | 300: 69195.0000 vs 69022.0000 ± 4565.0000, tol 13695.0000; 600: 1.259e+5 vs 1.309e+5 ± 5300.0000, tol 15900.0000; 900: 1.835e+5 vs 1.831e+5 ± 6281.0000, tol 18843.0000; 1200: 2.228e+5 vs 2.162e+5 ± 11232.0000, tol 33696.0000 |

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

Check-slot step time (informational): candidate fenced median 35.16 ms; references 37.24, 37.57, 65.34, 34.40 ms.

## letters: letters-ref-c: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | pass | 0.29 | 120: -0.0073 vs -0.0087 ± 0.0005, tol 0.0050; 180: -0.0096 vs -0.0094 ± 0.0003, tol 0.0050; 240: -0.0107 vs -0.0110 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | pass | 0.31 | 60: 0.0564 vs 0.0000 ± 0.0498, tol 1.0000; 90: 0.1018 vs 0.0000 ± 0.1918, tol 1.0000; 120: 0.6638 vs 0.0000 ± 0.7039, tol 2.1116; 180: 0.6635 vs 0.0000 ± 0.9908, tol 2.9724; 240: 1.0759 vs 0.0000 ± 1.2762, tol 3.8286 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.47 | 60: 1950.4720 vs 1943.9378 ± 12.1874, tol 194.3938; 120: 2398.7479 vs 2245.5682 ± 109.6425, tol 328.9274; 180: 3183.7348 vs 3049.1827 ± 96.7430, tol 304.9183; 240: 3090.7092 vs 3039.6935 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | pass | 0.57 | 60: 19.9022 vs 19.8928 ± 0.0030, tol 0.2500; 120: 20.7309 vs 20.5393 ± 0.1126, tol 0.3377; 180: 21.7282 vs 21.7064 ± 0.0536, tol 0.2500; 240: 20.5032 vs 20.4835 ± 0.0106, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 5.001e+5 vs 5.003e+5 ± 8558.0000, tol 50031.4750; 120: 9.050e+5 vs 8.902e+5 ± 17142.0000, tol 89024.8500; 180: 9.179e+5 vs 9.049e+5 ± 11334.0000, tol 90485.4750; 240: 8.669e+5 vs 8.740e+5 ± 17601.0000, tol 87400.8750 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.29 | 120: -0.0073 vs -0.0087 ± 0.0005, tol 0.0050; 180: -0.0096 vs -0.0094 ± 0.0003, tol 0.0050; 240: -0.0106 vs -0.0110 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 127.79 ms; references 124.54, 122.69, 84.53, 84.70 ms.

## paddleboats: paddleboats-ref-c: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | pass | 0.08 | 150: -0.0030 vs -0.0029 ± 0.0002, tol 0.0025; 300: -0.0100 vs -0.0098 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | pass | 0.11 | 150: 0.0121 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.0270 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.18 | 60: 95.8066 vs 94.9333 ± 3.5559, tol 10.6677; 120: 108.5853 vs 109.7952 ± 1.3791, tol 10.9795; 180: 142.7658 vs 142.0858 ± 1.1977, tol 14.2086; 240: 139.8926 vs 142.4236 ± 3.6080, tol 14.2424; 300: 166.8834 vs 167.8955 ± 3.3795, tol 16.7896 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.25 | 150: 2.575e+5 vs 2.643e+5 ± 9080.0000, tol 27240.0000; 300: 2.984e+5 vs 2.965e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.08 | 150: -0.0083 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0153 vs -0.0151 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 94.75 ms; references 111.75, 88.86, 97.52, 50.48 ms.
