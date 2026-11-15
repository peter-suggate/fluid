# NB-FLIP port gates: dam-residual0.5, letters-residual0.5, paddleboats-residual0.5, seep-residual0.5

Generated 2026-11-15T01:40:26.005Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-ref-e; dam dam-ref-a + dam-ref-b + dam-ref-c + dam-ref-d + dam-ref-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-ref-e.
**Overall: FAIL** (seep FAIL, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-residual0.5: FAIL

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | **FAIL** | 2.12 | mean: 0.2408 vs 0.2692 ± 0.0020, tol 0.0135 **x** |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | **FAIL** | 3.03 | 120-180: 0.2720 vs 0.2696 ± 0.0022, tol 0.0270; 180-240: 0.2736 vs 0.2712 ± 0.0038, tol 0.0271; 240-300: 0.2720 vs 0.2711 ± 0.0023, tol 0.0271; 300-360: 0.2729 vs 0.2723 ± 0.0028, tol 0.0272; 360-420: 0.2707 vs 0.2721 ± 0.0035, tol 0.0272; 420-480: 0.2706 vs 0.2716 ± 0.0032, tol 0.0272; 480-540: 0.2708 vs 0.2698 ± 0.0034, tol 0.0270; 540-600: 0.2685 vs 0.2712 ± 0.0044, tol 0.0271; 600-660: 0.2640 vs 0.2692 ± 0.0034, tol 0.0269; 660-720: 0.2511 vs 0.2691 ± 0.0036, tol 0.0269; 720-780: 0.2462 vs 0.2685 ± 0.0029, tol 0.0268; 780-840: 0.2261 vs 0.2684 ± 0.0038, tol 0.0268 **x**; 840-900: 0.1995 vs 0.2684 ± 0.0039, tol 0.0268 **x**; 900-960: 0.2014 vs 0.2678 ± 0.0038, tol 0.0268 **x**; 960-1020: 0.2073 vs 0.2669 ± 0.0041, tol 0.0267 **x**; 1020-1080: 0.1908 vs 0.2670 ± 0.0023, tol 0.0267 **x**; 1080-1140: 0.1858 vs 0.2666 ± 0.0031, tol 0.0267 **x**; 1140-1200: 0.1904 vs 0.2654 ± 0.0036, tol 0.0265 **x** |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | **FAIL** | 1.92 | 300: 1.3248 vs 1.3196 ± 0.0049, tol 0.0660; 600: 2.6783 vs 2.6766 ± 0.0148, tol 0.1338; 900: 3.8651 vs 4.0202 ± 0.0259, tol 0.2010; 1200: 4.8408 vs 5.3538 ± 0.0355, tol 0.2677 **x** |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.17 | 300: 1.1497 vs 1.1514 ± 0.0298, tol 0.0895 (restored 1.3196 ± 0.0050, tol 0.0660); 600: 2.1051 vs 2.1261 ± 0.0754, tol 0.2263 (restored 2.6766 ± 0.0148, tol 0.1338); 900: 2.8831 vs 2.9516 ± 0.1341, tol 0.4024 (restored 4.0202 ± 0.0259, tol 0.2010); 1200: 3.3876 vs 3.4760 ± 0.2046, tol 0.6138 (restored 5.3538 ± 0.0356, tol 0.2677) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.17 | 300: 1.1497 vs 1.1514 ± 0.0298, tol 0.0895; 600: 2.1051 vs 2.1261 ± 0.0754, tol 0.2263; 900: 2.8831 vs 2.9516 ± 0.1341, tol 0.4024; 1200: 3.3876 vs 3.4760 ± 0.2046, tol 0.6138 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.02 | median: 2.6373 vs 2.6420 ± 0.0128, tol 0.2642 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.09 | 90%: 294.4801 vs 288.2801 ± 17.8628, tol 72.0700 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.02 | 1200: 16354.0000 vs 16370.0000 ± 12.0000, tol 818.5000 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.11 | 600: 366.7736 vs 368.4723 ± 3.9858, tol 16.0000 (restored 371.6549 ± 3.4575, tol 16.0000); 1200: 372.3651 vs 370.1927 ± 4.0817, tol 16.0000 (restored 375.0367 ± 2.7714, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.14 | 600: 366.7736 vs 368.4723 ± 3.9858, tol 16.0000; 1200: 372.3651 vs 370.1927 ± 4.0817, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.17 | 600: 0.0552 vs 0.0552 ± 0.1616, tol 0.4848 (restored 0.1094 ± 0.1321, tol 0.3962); 1200: 0.0394 vs 0.0394 ± 0.0756, tol 0.2268 (restored 0.0814 ± 0.0500, tol 0.1501) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.17 | 600: 0.0552 vs 0.0000 ± 0.1616, tol 0.4848; 1200: 0.0394 vs 0.0000 ± 0.0756, tol 0.2268 |
| G7 | total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.001 | pass | 0.00 | 1200: -0.0034 vs 0.0051 ± 0.0006, tol 0.0010 |
| G7 (first ruling) | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | (pass, not gated) | 0.00 | 1200: -0.0034 vs 0.0048 ± 0.0006, tol 0.0019 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.71 | 1200: -0.0034 vs -0.0048 ± 0.0006, tol 0.0019 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.09 | 300: 379.9199 vs 380.3110 ± 1.6943, tol 8.0000; 600: 381.8278 vs 380.7121 ± 4.3196, tol 12.9587 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.16 | 300: 0.5160 vs 0.0000 ± 1.4579, tol 4.3737; 600: 0.6020 vs 0.0000 ± 1.2470, tol 3.7409 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.09 | 300: 69984.0000 vs 69194.2000 ± 2891.0000, tol 8673.0000; 600: 1.310e+5 vs 1.302e+5 ± 6052.0000, tol 18156.0000; 900: 1.855e+5 vs 1.864e+5 ± 9143.0000, tol 27429.0000; 1200: 2.203e+5 vs 2.218e+5 ± 11539.0000, tol 34617.0000 |

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

Check-slot step time (informational): candidate fenced median 35.65 ms; references 54.13, 68.99, 49.39, 53.44, 74.01 ms.

## dam: dam-residual0.5: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.25 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 107.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2000 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1680 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5004 ± 0.0867, tol 4.0000); 16: 78.7500 vs 78.7625 ± 0.0625, tol 4.0000 (restored 80.8008 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7625 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0428 vs 0.0491 ± 0.0068, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0428 vs 0.0438 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.05 | 16: -0.0428 vs -0.0438 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0847 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.4987 vs 119.4973 ± 0.0237, tol 11.9497; 12: 228.7309 vs 228.6048 ± 0.4230, tol 22.8605; 16: 274.3882 vs 275.0075 ± 1.1711, tol 27.5008 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.03 | 4: 669.1635 vs 669.1684 ± 0.0008, tol 13.3834 (restored 669.9803 ± 0.0009, tol 13.3996); 8: 670.7873 vs 670.7852 ± 0.0429, tol 13.4157 (restored 671.9703 ± 0.0280, tol 13.4394); 12: 664.1969 vs 663.9892 ± 0.7382, tol 13.2798 (restored 667.4997 ± 0.2311, tol 13.3500); 16: 590.5956 vs 590.9589 ± 0.9253, tol 11.8192 (restored 613.4000 ± 3.2444, tol 12.2680) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.03 | 4: 669.1635 vs 669.1684 ± 0.0008, tol 13.3834; 8: 670.7873 vs 670.7852 ± 0.0429, tol 13.4157; 12: 664.1969 vs 663.9892 ± 0.7382, tol 13.2798; 16: 590.5956 vs 590.9589 ± 0.9253, tol 11.8192 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7624.0000 vs 7623.6000 ± 2.0000, tol 381.1800; 8: 10761.0000 vs 10767.4000 ± 4.0000, tol 538.3700; 12: 14933.0000 vs 14936.2000 ± 10.0000, tol 746.8100 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.13 | 8: 0.1444 vs 0.1444 ± 0.0156, tol 4.0000 (restored 0.1853 ± 0.0158, tol 4.0000); 12: 0.2353 vs 0.2353 ± 0.4381, tol 4.0000 (restored 0.4386 ± 0.4252, tol 4.0000); 16: 0.8811 vs 0.8811 ± 2.2985, tol 6.8954 (restored 2.0668 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.13 | 8: 0.1444 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.2353 vs 0.0000 ± 0.4381, tol 4.0000; 16: 0.8811 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.06 | 16: 3.537e+5 vs 3.516e+5 ± 5756.0000, tol 35157.3800 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.81 | 30-90: 44.9047 vs 45.3007 ± 0.9511, tol 2.8534 (restored 47.1584 ± 0.9143, tol 2.7430); 105-165: 49.4709 vs 47.0926 ± 0.7294, tol 2.3546 (restored 47.5419 ± 0.6676, tol 2.3771); 180-300: 48.7117 vs 46.8320 ± 0.5217, tol 2.3416 (restored 47.1994 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (fail, not gated) | 1.01 | 30-90: 44.9047 vs 45.3007 ± 0.9511, tol 2.8534; 105-165: 49.4709 vs 47.0926 ± 0.7294, tol 2.3546 **x**; 180-300: 48.7117 vs 46.8320 ± 0.5217, tol 2.3416 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 30-90: -0.0354 vs 0.0446 ± 0.0090, tol 0.0100; 105-165: -0.0101 vs 0.0119 ± 0.0082, tol 0.0100; 180-300: -0.0098 vs 0.0114 ± 0.0150, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.06 | 30-90: -0.0354 vs 0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0101 vs 0.0096 ± 0.0082, tol 0.0246; 180-300: -0.0098 vs 0.0073 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.18 | 30-90: -0.0354 vs -0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0101 vs -0.0096 ± 0.0082, tol 0.0246; 180-300: -0.0098 vs -0.0073 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.13 | 30-90: 46.1869 vs 46.5247 ± 1.2119, tol 4.6525; 105-165: 15.0181 vs 16.6475 ± 4.3305, tol 12.9914; 180-300: 9.0055 vs 10.3905 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.09 | 30-90: 282.7440 vs 281.4297 ± 4.4590, tol 13.3770 (restored 301.7705 ± 1.1612, tol 6.0354); 105-165: 266.0761 vs 268.2136 ± 7.5629, tol 22.6888 (restored 273.2416 ± 4.5779, tol 13.7338); 180-300: 262.6150 vs 265.5091 ± 10.4288, tol 31.2864 (restored 269.6234 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.10 | 30-90: 282.7440 vs 281.4297 ± 4.4590, tol 13.3770; 105-165: 266.0761 vs 268.2136 ± 7.5629, tol 22.6888; 180-300: 262.6150 vs 265.5091 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.11 | 30-90: 328.9309 vs 327.9544 ± 3.8834, tol 11.6502 (restored 348.2952 ± 2.0543, tol 6.9659); 105-165: 281.0942 vs 284.8611 ± 11.8934, tol 35.6801 (restored 289.8892 ± 8.9084, tol 26.7252); 180-300: 271.6205 vs 275.8995 ± 14.4039, tol 43.2116 (restored 280.0139 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.11 | 30-90: 328.9309 vs 327.9544 ± 3.8834, tol 11.6502; 105-165: 281.0942 vs 284.8611 ± 11.8934, tol 35.6801; 180-300: 271.6205 vs 275.8995 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.11 | 30-90: 0.0048 vs 0.0048 ± 0.0059, tol 0.0500 (restored 0.0064 ± 0.0058, tol 0.0500); 105-165: 0.0055 vs 0.0055 ± 0.0114, tol 0.0500 (restored 0.0056 ± 0.0113, tol 0.0500); 180-300: 0.0029 vs 0.0029 ± 0.0094, tol 0.0500 (restored 0.0029 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 30-90: 0.0048 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0055 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0029 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.12 | 30-90: 6.269e+5 vs 6.309e+5 ± 17318.6000, tol 63086.1960; 105-165: 3.743e+5 vs 3.838e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.516e+5 vs 3.484e+5 ± 30193.3333, tol 90580.0000 |

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

Check-slot step time (informational): candidate fenced median 44.11 ms; references 67.60, 45.86, 62.58, 66.20, 60.19 ms.

## letters: letters-residual0.5: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.07 | 120: -0.0089 vs 0.0089 ± 0.0004, tol 0.0050; 180: -0.0093 vs 0.0095 ± 0.0014, tol 0.0050; 240: -0.0114 vs 0.0110 ± 0.0009, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.13 | 120: -0.0089 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0093 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0114 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.13 | 120: -0.0089 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0093 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0114 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.23 | 60: 0.1001 vs 0.1001 ± 0.0364, tol 1.0000 (restored 0.2863 ± 0.0363, tol 1.0000); 90: 0.1242 vs 0.1242 ± 0.1444, tol 1.0000 (restored 0.4699 ± 0.1444, tol 1.0000); 120: 0.4771 vs 0.4771 ± 0.7010, tol 2.1030 (restored 0.5904 ± 0.7010, tol 2.1029); 180: 0.5390 vs 0.5390 ± 0.9631, tol 2.8893 (restored 0.6566 ± 0.9631, tol 2.8892); 240: 0.6604 vs 0.6604 ± 1.1402, tol 3.4206 (restored 0.8046 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 60: 0.1001 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.1242 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.4771 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.5390 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.6604 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 1952.3274 vs 1948.6120 ± 2.2087, tol 194.8612; 120: 2191.4120 vs 2239.5631 ± 101.6255, tol 304.8766; 180: 3085.8194 vs 3053.5803 ± 219.4018, tol 658.2053; 240: 2991.3072 vs 3041.9801 ± 41.3393, tol 304.1980 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.08 | 60: 19.8843 vs 19.8925 ± 0.0023, tol 0.2500 (restored 20.0174 ± 0.0017, tol 0.2500); 120: 20.5137 vs 20.5480 ± 0.1422, tol 0.4266 (restored 20.7089 ± 0.1374, tol 0.4121); 180: 21.7086 vs 21.7277 ± 0.0407, tol 0.2500 (restored 21.8959 ± 0.0390, tol 0.2500); 240: 20.4948 vs 20.5001 ± 0.0231, tol 0.2500 (restored 20.7118 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.08 | 60: 19.8843 vs 19.8925 ± 0.0023, tol 0.2500; 120: 20.5137 vs 20.5480 ± 0.1422, tol 0.4266; 180: 21.7086 vs 21.7277 ± 0.0407, tol 0.2500; 240: 20.4948 vs 20.5001 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.36 | 60: 4.829e+5 vs 5.007e+5 ± 6307.0000, tol 50071.3200; 120: 8.911e+5 vs 8.908e+5 ± 11988.0000, tol 89082.2400; 180: 9.042e+5 vs 9.020e+5 ± 18947.0000, tol 90198.3600; 240: 8.807e+5 vs 8.779e+5 ± 7187.0000, tol 87786.9600 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.13 | 120: -0.0089 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0093 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0114 vs -0.0107 ± 0.0009, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 83.49 ms; references 123.19, 96.65, 84.54, 87.50, 94.06 ms.

## paddleboats: paddleboats-residual0.5: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.00 | 150: -0.0027 vs 0.0031 ± 0.0003, tol 0.0025; 300: -0.0058 vs 0.0107 ± 0.0018, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.00 | 150: -0.0027 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0058 vs 0.0102 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.83 | 150: -0.0027 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0058 vs -0.0102 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.22 | 150: 0.0321 vs 0.0321 ± 0.0176, tol 0.2500 (restored 0.0713 ± 0.0164, tol 0.2500); 300: 0.1186 vs 0.1186 ± 0.0590, tol 0.2500 (restored 0.1491 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.47 | 150: 0.0321 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.1186 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.33 | 60: 94.8013 vs 95.2277 ± 1.7659, tol 9.5228; 120: 107.5073 vs 109.1027 ± 1.1784, tol 10.9103; 180: 139.5890 vs 141.8094 ± 1.5890, tol 14.1809; 240: 139.1365 vs 141.9704 ± 1.9409, tol 14.1970; 300: 161.7264 vs 167.2549 ± 2.4230, tol 16.7255 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.06 | 150: 2.615e+5 vs 2.569e+5 ± 26447.0000, tol 79341.0000; 300: 3.018e+5 vs 2.911e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.84 | 150: -0.0080 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0111 vs -0.0155 ± 0.0017, tol 0.0052 |

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

Check-slot step time (informational): candidate fenced median 47.15 ms; references 51.19, 70.16, 50.72, 54.85, 95.74 ms.
