# NB-FLIP port gates: dam-base-f, letters-base-f, paddleboats-base-f, seep-base-f

Generated 2026-11-15T00:46:28.510Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-ref-e; dam dam-ref-a + dam-ref-b + dam-ref-c + dam-ref-d + dam-ref-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-ref-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-base-f: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.07 | mean: 0.2702 vs 0.2692 ± 0.0020, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.42 | 120-180: 0.2700 vs 0.2696 ± 0.0022, tol 0.0270; 180-240: 0.2727 vs 0.2712 ± 0.0038, tol 0.0271; 240-300: 0.2713 vs 0.2711 ± 0.0023, tol 0.0271; 300-360: 0.2729 vs 0.2723 ± 0.0028, tol 0.0272; 360-420: 0.2717 vs 0.2721 ± 0.0035, tol 0.0272; 420-480: 0.2696 vs 0.2716 ± 0.0032, tol 0.0272; 480-540: 0.2702 vs 0.2698 ± 0.0034, tol 0.0270; 540-600: 0.2695 vs 0.2712 ± 0.0044, tol 0.0271; 600-660: 0.2684 vs 0.2692 ± 0.0034, tol 0.0269; 660-720: 0.2672 vs 0.2691 ± 0.0036, tol 0.0269; 720-780: 0.2686 vs 0.2685 ± 0.0029, tol 0.0268; 780-840: 0.2667 vs 0.2684 ± 0.0038, tol 0.0268; 840-900: 0.2669 vs 0.2684 ± 0.0039, tol 0.0268; 900-960: 0.2659 vs 0.2678 ± 0.0038, tol 0.0268; 960-1020: 0.2672 vs 0.2669 ± 0.0041, tol 0.0267; 1020-1080: 0.2711 vs 0.2670 ± 0.0023, tol 0.0267; 1080-1140: 0.2770 vs 0.2666 ± 0.0031, tol 0.0267; 1140-1200: 0.2765 vs 0.2654 ± 0.0036, tol 0.0265 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.12 | 300: 1.3278 vs 1.3196 ± 0.0049, tol 0.0660; 600: 2.6817 vs 2.6766 ± 0.0148, tol 0.1338; 900: 4.0194 vs 4.0202 ± 0.0259, tol 0.2010; 1200: 5.3773 vs 5.3538 ± 0.0355, tol 0.2677 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.24 | 300: 1.1384 vs 1.1514 ± 0.0298, tol 0.0895 (restored 1.3196 ± 0.0050, tol 0.0660); 600: 2.1396 vs 2.1261 ± 0.0754, tol 0.2263 (restored 2.6766 ± 0.0148, tol 0.1338); 900: 2.8542 vs 2.9516 ± 0.1341, tol 0.4024 (restored 4.0202 ± 0.0259, tol 0.2010); 1200: 3.6431 vs 3.4760 ± 0.2046, tol 0.6138 (restored 5.3538 ± 0.0356, tol 0.2677) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.27 | 300: 1.1384 vs 1.1514 ± 0.0298, tol 0.0895; 600: 2.1396 vs 2.1261 ± 0.0754, tol 0.2263; 900: 2.8542 vs 2.9516 ± 0.1341, tol 0.4024; 1200: 3.6431 vs 3.4760 ± 0.2046, tol 0.6138 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6390 vs 2.6420 ± 0.0128, tol 0.2642 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.02 | 90%: 289.6763 vs 288.2801 ± 17.8628, tol 72.0700 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.09 | 1200: 16293.0000 vs 16370.0000 ± 12.0000, tol 818.5000 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.08 | 600: 367.1337 vs 368.4723 ± 3.9858, tol 16.0000 (restored 371.6549 ± 3.4575, tol 16.0000); 1200: 371.3335 vs 370.1927 ± 4.0817, tol 16.0000 (restored 375.0367 ± 2.7714, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.08 | 600: 367.1337 vs 368.4723 ± 3.9858, tol 16.0000; 1200: 371.3335 vs 370.1927 ± 4.0817, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.21 | 600: 0.1026 vs 0.1026 ± 0.1616, tol 0.4848 (restored 0.1488 ± 0.1321, tol 0.3962); 1200: 0.0329 vs 0.0329 ± 0.0756, tol 0.2268 (restored 0.0709 ± 0.0500, tol 0.1501) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.21 | 600: 0.1026 vs 0.0000 ± 0.1616, tol 0.4848; 1200: 0.0329 vs 0.0000 ± 0.0756, tol 0.2268 |
| G7 | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | pass | 0.00 | 1200: -0.0044 vs 0.0048 ± 0.0006, tol 0.0019 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.20 | 1200: -0.0044 vs -0.0048 ± 0.0006, tol 0.0019 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.20 | 300: 379.7062 vs 380.3110 ± 1.6943, tol 8.0000; 600: 383.3579 vs 380.7121 ± 4.3196, tol 12.9587 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.15 | 300: 0.5553 vs 0.0000 ± 1.4579, tol 4.3737; 600: 0.5543 vs 0.0000 ± 1.2470, tol 3.7409 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.11 | 300: 68488.0000 vs 69194.2000 ± 2891.0000, tol 8673.0000; 600: 1.312e+5 vs 1.302e+5 ± 6052.0000, tol 18156.0000; 900: 1.833e+5 vs 1.864e+5 ± 9143.0000, tol 27429.0000; 1200: 2.234e+5 vs 2.218e+5 ± 11539.0000, tol 34617.0000 |

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

Check-slot step time (informational): candidate fenced median 59.56 ms; references 54.13, 68.99, 49.39, 53.44, 74.01 ms.

## dam: dam-base-f: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.05 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2000 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1680 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5004 ± 0.0867, tol 4.0000); 16: 78.7500 vs 78.7625 ± 0.0625, tol 4.0000 (restored 80.8008 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7625 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.00 | 16: -0.0422 vs 0.0438 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.08 | 16: -0.0422 vs -0.0438 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.00 | 4: 36.0847 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.4987 vs 119.4973 ± 0.0237, tol 11.9497; 12: 228.5974 vs 228.6048 ± 0.4230, tol 22.8605; 16: 274.8826 vs 275.0075 ± 1.1711, tol 27.5008 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 4: 669.1683 vs 669.1684 ± 0.0008, tol 13.3834 (restored 669.9803 ± 0.0009, tol 13.3996); 8: 670.7889 vs 670.7852 ± 0.0429, tol 13.4157 (restored 671.9703 ± 0.0280, tol 13.4394); 12: 663.9887 vs 663.9892 ± 0.7382, tol 13.2798 (restored 667.4997 ± 0.2311, tol 13.3500); 16: 591.1939 vs 590.9589 ± 0.9253, tol 11.8192 (restored 613.4000 ± 3.2444, tol 12.2680) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.02 | 4: 669.1683 vs 669.1684 ± 0.0008, tol 13.3834; 8: 670.7889 vs 670.7852 ± 0.0429, tol 13.4157; 12: 663.9887 vs 663.9892 ± 0.7382, tol 13.2798; 16: 591.1939 vs 590.9589 ± 0.9253, tol 11.8192 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.00 | 4: 7624.0000 vs 7623.6000 ± 2.0000, tol 381.1800; 8: 10767.0000 vs 10767.4000 ± 4.0000, tol 538.3700; 12: 14936.0000 vs 14936.2000 ± 10.0000, tol 746.8100 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.08 | 8: 0.0063 vs 0.0063 ± 0.0156, tol 4.0000 (restored 0.0970 ± 0.0158, tol 4.0000); 12: 0.0887 vs 0.0887 ± 0.4381, tol 4.0000 (restored 0.3137 ± 0.4252, tol 4.0000); 16: 0.5232 vs 0.5232 ± 2.2985, tol 6.8954 (restored 2.2948 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.08 | 8: 0.0063 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.0887 vs 0.0000 ± 0.4381, tol 4.0000; 16: 0.5232 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.03 | 16: 3.507e+5 vs 3.516e+5 ± 5756.0000, tol 35157.3800 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.29 | 30-90: 44.4636 vs 45.3007 ± 0.9511, tol 2.8534 (restored 47.1584 ± 0.9143, tol 2.7430); 105-165: 47.8128 vs 47.0926 ± 0.7294, tol 2.3546 (restored 47.5419 ± 0.6676, tol 2.3771); 180-300: 47.0541 vs 46.8320 ± 0.5217, tol 2.3416 (restored 47.1994 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.31 | 30-90: 44.4636 vs 45.3007 ± 0.9511, tol 2.8534; 105-165: 47.8128 vs 47.0926 ± 0.7294, tol 2.3546; 180-300: 47.0541 vs 46.8320 ± 0.5217, tol 2.3416 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.08 | 30-90: -0.0416 vs 0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0108 vs 0.0096 ± 0.0082, tol 0.0246; 180-300: -0.0108 vs 0.0073 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.08 | 30-90: -0.0416 vs -0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0108 vs -0.0096 ± 0.0082, tol 0.0246; 180-300: -0.0108 vs -0.0073 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.18 | 30-90: 46.9247 vs 46.5247 ± 1.2119, tol 4.6525; 105-165: 14.3577 vs 16.6475 ± 4.3305, tol 12.9914; 180-300: 8.8603 vs 10.3905 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.11 | 30-90: 280.8500 vs 281.4297 ± 4.4590, tol 13.3770 (restored 301.7705 ± 1.1612, tol 6.0354); 105-165: 265.7518 vs 268.2136 ± 7.5629, tol 22.6888 (restored 273.2416 ± 4.5779, tol 13.7338); 180-300: 262.5633 vs 265.5091 ± 10.4288, tol 31.2864 (restored 269.6234 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.11 | 30-90: 280.8500 vs 281.4297 ± 4.4590, tol 13.3770; 105-165: 265.7518 vs 268.2136 ± 7.5629, tol 22.6888; 180-300: 262.5633 vs 265.5091 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.13 | 30-90: 327.7747 vs 327.9544 ± 3.8834, tol 11.6502 (restored 348.2952 ± 2.0543, tol 6.9659); 105-165: 280.1095 vs 284.8611 ± 11.8934, tol 35.6801 (restored 289.8892 ± 8.9084, tol 26.7252); 180-300: 271.4236 vs 275.8995 ± 14.4039, tol 43.2116 (restored 280.0139 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.13 | 30-90: 327.7747 vs 327.9544 ± 3.8834, tol 11.6502; 105-165: 280.1095 vs 284.8611 ± 11.8934, tol 35.6801; 180-300: 271.4236 vs 275.8995 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.11 | 30-90: 0.0055 vs 0.0055 ± 0.0059, tol 0.0500 (restored 0.0084 ± 0.0058, tol 0.0500); 105-165: 0.0020 vs 0.0020 ± 0.0114, tol 0.0500 (restored 0.0020 ± 0.0113, tol 0.0500); 180-300: 0.0021 vs 0.0021 ± 0.0094, tol 0.0500 (restored 0.0021 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 30-90: 0.0055 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0020 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0021 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.43 | 30-90: 6.579e+5 vs 6.309e+5 ± 17318.6000, tol 63086.1960; 105-165: 3.842e+5 vs 3.838e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.575e+5 vs 3.484e+5 ± 30193.3333, tol 90580.0000 |

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

Check-slot step time (informational): candidate fenced median 71.06 ms; references 67.60, 45.86, 62.58, 66.20, 60.19 ms.

## letters: letters-base-f: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.15 | 120: -0.0089 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0099 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0102 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.15 | 120: -0.0089 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0099 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0102 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.24 | 60: 0.0231 vs 0.0231 ± 0.0364, tol 1.0000 (restored 0.2636 ± 0.0363, tol 1.0000); 90: 0.0638 vs 0.0638 ± 0.1444, tol 1.0000 (restored 0.4429 ± 0.1444, tol 1.0000); 120: 0.4112 vs 0.4112 ± 0.7010, tol 2.1030 (restored 0.5379 ± 0.7010, tol 2.1029); 180: 0.6570 vs 0.6570 ± 0.9631, tol 2.8893 (restored 0.7678 ± 0.9631, tol 2.8892); 240: 0.8285 vs 0.8285 ± 1.1402, tol 3.4206 (restored 0.9260 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.24 | 60: 0.0231 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.0638 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.4112 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.6570 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.8285 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.06 | 60: 1949.3218 vs 1948.6120 ± 2.2087, tol 194.8612; 120: 2228.3534 vs 2239.5631 ± 101.6255, tol 304.8766; 180: 3016.7364 vs 3053.5803 ± 219.4018, tol 658.2053; 240: 3029.6190 vs 3041.9801 ± 41.3393, tol 304.1980 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.16 | 60: 19.8920 vs 19.8925 ± 0.0023, tol 0.2500 (restored 20.0174 ± 0.0017, tol 0.2500); 120: 20.5506 vs 20.5480 ± 0.1422, tol 0.4266 (restored 20.7089 ± 0.1374, tol 0.4121); 180: 21.6889 vs 21.7277 ± 0.0407, tol 0.2500 (restored 21.8959 ± 0.0390, tol 0.2500); 240: 20.4907 vs 20.5001 ± 0.0231, tol 0.2500 (restored 20.7118 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.16 | 60: 19.8920 vs 19.8925 ± 0.0023, tol 0.2500; 120: 20.5506 vs 20.5480 ± 0.1422, tol 0.4266; 180: 21.6889 vs 21.7277 ± 0.0407, tol 0.2500; 240: 20.4907 vs 20.5001 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.09 | 60: 5.050e+5 vs 5.007e+5 ± 6307.0000, tol 50071.3200; 120: 8.930e+5 vs 8.908e+5 ± 11988.0000, tol 89082.2400; 180: 9.028e+5 vs 9.020e+5 ± 18947.0000, tol 90198.3600; 240: 8.702e+5 vs 8.779e+5 ± 7187.0000, tol 87786.9600 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.15 | 120: -0.0089 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0099 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0102 vs -0.0107 ± 0.0009, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 149.00 ms; references 123.19, 96.65, 84.54, 87.50, 94.06 ms.

## paddleboats: paddleboats-base-f: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.00 | 150: -0.0029 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0096 vs 0.0102 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.11 | 150: -0.0029 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0096 vs -0.0102 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.10 | 150: 0.0096 vs 0.0096 ± 0.0176, tol 0.2500 (restored 0.0705 ± 0.0164, tol 0.2500); 300: 0.0274 vs 0.0274 ± 0.0590, tol 0.2500 (restored 0.2321 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 150: 0.0096 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.0274 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.14 | 60: 93.8566 vs 95.2277 ± 1.7659, tol 9.5228; 120: 109.3954 vs 109.1027 ± 1.1784, tol 10.9103; 180: 142.6862 vs 141.8094 ± 1.5890, tol 14.1809; 240: 141.6861 vs 141.9704 ± 1.9409, tol 14.1970; 300: 167.3429 vs 167.2549 ± 2.4230, tol 16.7255 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.17 | 150: 2.706e+5 vs 2.569e+5 ± 26447.0000, tol 79341.0000; 300: 2.948e+5 vs 2.911e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.11 | 150: -0.0082 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0149 vs -0.0155 ± 0.0017, tol 0.0052 |

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

Check-slot step time (informational): candidate fenced median 107.21 ms; references 51.19, 70.16, 50.72, 54.85, 95.74 ms.
