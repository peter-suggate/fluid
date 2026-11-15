# NB-FLIP port gates: dam-ref-e, letters-ref-e, paddleboats-ref-e, seep-ref-e

Generated 2026-11-15T01:40:25.261Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d; dam dam-ref-a + dam-ref-b + dam-ref-c + dam-ref-d; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-e: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.05 | mean: 0.2698 vs 0.2691 ± 0.0020, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.10 | 120-180: 0.2683 vs 0.2700 ± 0.0013, tol 0.0270; 180-240: 0.2717 vs 0.2711 ± 0.0038, tol 0.0271; 240-300: 0.2724 vs 0.2708 ± 0.0010, tol 0.0271; 300-360: 0.2722 vs 0.2723 ± 0.0028, tol 0.0272; 360-420: 0.2740 vs 0.2717 ± 0.0025, tol 0.0272; 420-480: 0.2713 vs 0.2717 ± 0.0032, tol 0.0272; 480-540: 0.2697 vs 0.2698 ± 0.0034, tol 0.0270; 540-600: 0.2719 vs 0.2710 ± 0.0044, tol 0.0271; 600-660: 0.2698 vs 0.2691 ± 0.0034, tol 0.0269; 660-720: 0.2708 vs 0.2686 ± 0.0023, tol 0.0269; 720-780: 0.2685 vs 0.2685 ± 0.0029, tol 0.0268; 780-840: 0.2706 vs 0.2679 ± 0.0017, tol 0.0268; 840-900: 0.2685 vs 0.2684 ± 0.0039, tol 0.0268; 900-960: 0.2667 vs 0.2680 ± 0.0038, tol 0.0268; 960-1020: 0.2689 vs 0.2664 ± 0.0030, tol 0.0266; 1020-1080: 0.2669 vs 0.2670 ± 0.0023, tol 0.0267; 1080-1140: 0.2682 vs 0.2662 ± 0.0017, tol 0.0266; 1140-1200: 0.2660 vs 0.2652 ± 0.0036, tol 0.0265 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.06 | 300: 1.3223 vs 1.3189 ± 0.0033, tol 0.0659; 600: 2.6814 vs 2.6754 ± 0.0131, tol 0.1338; 900: 4.0296 vs 4.0178 ± 0.0243, tol 0.2009; 1200: 5.3663 vs 5.3507 ± 0.0342, tol 0.2675 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.23 | 300: 1.1697 vs 1.1469 ± 0.0163, tol 0.0573 (restored 1.3189 ± 0.0033, tol 0.0659); 600: 2.1102 vs 2.1301 ± 0.0754, tol 0.2263 (restored 2.6754 ± 0.0131, tol 0.1338); 900: 2.8850 vs 2.9683 ± 0.1190, tol 0.3571 (restored 4.0178 ± 0.0243, tol 0.2009); 1200: 3.4399 vs 3.4850 ± 0.2046, tol 0.6138 (restored 5.3507 ± 0.0344, tol 0.2675) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.40 | 300: 1.1697 vs 1.1469 ± 0.0163, tol 0.0573; 600: 2.1102 vs 2.1301 ± 0.0754, tol 0.2263; 900: 2.8850 vs 2.9683 ± 0.1190, tol 0.3571; 1200: 3.4399 vs 3.4850 ± 0.2046, tol 0.6138 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6405 vs 2.6424 ± 0.0128, tol 0.2642 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.10 | 90%: 282.3709 vs 289.7574 ± 16.0842, tol 72.4393 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.00 | 1200: 16371.0000 vs 16369.7500 ± 12.0000, tol 818.4875 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.00 | 600: 370.4899 vs 367.9679 ± 2.9824, tol 16.0000 (restored 371.2266 ± 2.7596, tol 16.0000); 1200: 372.8486 vs 369.5287 ± 1.6732, tol 16.0000 (restored 374.5829 ± 0.8248, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.21 | 600: 370.4899 vs 367.9679 ± 2.9824, tol 16.0000; 1200: 372.8486 vs 369.5287 ± 1.6732, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.44 | 600: 0.1328 vs 0.1328 ± 0.1003, tol 0.3009 (restored 0.1087 ± 0.0827, tol 0.2481); 1200: 0.0648 vs 0.0648 ± 0.0722, tol 0.2165 (restored 0.0621 ± 0.0472, tol 0.1500) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.44 | 600: 0.1328 vs 0.0000 ± 0.1003, tol 0.3009; 1200: 0.0648 vs 0.0000 ± 0.0722, tol 0.2165 |
| G7 | total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.001 | pass | 0.00 | 1200: -0.0049 vs 0.0051 ± 0.0006, tol 0.0010 |
| G7 (first ruling) | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | (pass, not gated) | 0.08 | 1200: -0.0049 vs 0.0047 ± 0.0006, tol 0.0019 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.08 | 1200: -0.0049 vs -0.0047 ± 0.0006, tol 0.0019 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.20 | 300: 379.4550 vs 380.5250 ± 1.2330, tol 8.0000; 600: 378.7748 vs 381.1964 ± 4.1176, tol 12.3527 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.17 | 300: 0.6143 vs 0.0000 ± 1.4579, tol 4.3737; 600: 0.6209 vs 0.0000 ± 1.2470, tol 3.7409 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.30 | 300: 70551.0000 vs 68855.0000 ± 1954.0000, tol 6885.5000; 600: 1.290e+5 vs 1.305e+5 ± 6052.0000, tol 18156.0000; 900: 1.818e+5 vs 1.875e+5 ± 5953.0000, tol 18752.6250; 1200: 2.215e+5 vs 2.218e+5 ± 11539.0000, tol 34617.0000 |

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

Check-slot step time (informational): candidate fenced median 74.01 ms; references 54.13, 68.99, 49.39, 53.44 ms.

## dam: dam-ref-e: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2500 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1682 ± 0.0024, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5051 ± 0.0867, tol 4.0000); 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 (restored 80.8215 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0423 vs 0.0491 ± 0.0068, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0423 vs 0.0441 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.09 | 16: -0.0423 vs -0.0441 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0847 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.5027 vs 119.4960 ± 0.0237, tol 11.9496; 12: 228.7048 vs 228.5797 ± 0.4088, tol 22.8580; 16: 274.5961 vs 275.1104 ± 1.1711, tol 27.5110 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.01 | 4: 669.1683 vs 669.1685 ± 0.0008, tol 13.3834 (restored 669.9802 ± 0.0009, tol 13.3996); 8: 670.7939 vs 670.7830 ± 0.0429, tol 13.4157 (restored 671.9712 ± 0.0280, tol 13.4394); 12: 664.1528 vs 663.9483 ± 0.7382, tol 13.2790 (restored 667.5116 ± 0.2311, tol 13.3502); 16: 590.8505 vs 590.9860 ± 0.9253, tol 11.8197 (restored 613.6162 ± 3.1464, tol 12.2723) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.02 | 4: 669.1683 vs 669.1685 ± 0.0008, tol 13.3834; 8: 670.7939 vs 670.7830 ± 0.0429, tol 13.4157; 12: 664.1528 vs 663.9483 ± 0.7382, tol 13.2790; 16: 590.8505 vs 590.9860 ± 0.9253, tol 11.8197 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.00 | 4: 7623.0000 vs 7623.7500 ± 2.0000, tol 381.1875; 8: 10769.0000 vs 10767.0000 ± 4.0000, tol 538.3500; 12: 14936.0000 vs 14936.2500 ± 10.0000, tol 746.8125 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.23 | 8: 0.0117 vs 0.0117 ± 0.0156, tol 4.0000 (restored 0.0927 ± 0.0158, tol 4.0000); 12: 0.1142 vs 0.1142 ± 0.4381, tol 4.0000 (restored 0.3037 ± 0.4252, tol 4.0000); 16: 1.5996 vs 1.5996 ± 2.2985, tol 6.8954 (restored 2.2563 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 8: 0.0117 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.1142 vs 0.0000 ± 0.4381, tol 4.0000; 16: 1.5996 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.03 | 16: 3.506e+5 vs 3.518e+5 ± 5756.0000, tol 35180.6500 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.25 | 30-90: 45.3528 vs 45.2876 ± 0.9511, tol 2.8534 (restored 47.2008 ± 0.9143, tol 2.7430); 105-165: 46.6198 vs 47.2108 ± 0.3428, tol 2.3605 (restored 47.6511 ± 0.2174, tol 2.3826); 180-300: 46.6556 vs 46.8762 ± 0.4740, tol 2.3438 (restored 47.2301 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.25 | 30-90: 45.3528 vs 45.2876 ± 0.9511, tol 2.8534; 105-165: 46.6198 vs 47.2108 ± 0.3428, tol 2.3605; 180-300: 46.6556 vs 46.8762 ± 0.4740, tol 2.3438 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 30-90: -0.0356 vs 0.0446 ± 0.0043, tol 0.0100; 105-165: -0.0105 vs 0.0119 ± 0.0082, tol 0.0100; 180-300: -0.0092 vs 0.0114 ± 0.0150, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.05 | 30-90: -0.0356 vs 0.0416 ± 0.0043, tol 0.0130; 105-165: -0.0105 vs 0.0093 ± 0.0082, tol 0.0246; 180-300: -0.0092 vs 0.0068 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.46 | 30-90: -0.0356 vs -0.0416 ± 0.0043, tol 0.0130; 105-165: -0.0105 vs -0.0093 ± 0.0082, tol 0.0246; 180-300: -0.0092 vs -0.0068 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.18 | 30-90: 46.2895 vs 46.5835 ± 1.2119, tol 4.6584; 105-165: 15.0370 vs 17.0502 ± 3.7190, tol 11.1569; 180-300: 8.9051 vs 10.7618 ± 3.8451, tol 11.5354 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.14 | 30-90: 283.6120 vs 280.8842 ± 2.9770, tol 8.9310 (restored 301.8220 ± 1.1612, tol 6.0364); 105-165: 265.8698 vs 268.7996 ± 6.9063, tol 20.7189 (restored 273.7264 ± 3.4600, tol 10.3800); 180-300: 262.9254 vs 266.1550 ± 10.4288, tol 31.2864 (restored 270.1179 ± 4.5659, tol 13.6978) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.31 | 30-90: 283.6120 vs 280.8842 ± 2.9770, tol 8.9310; 105-165: 265.8698 vs 268.7996 ± 6.9063, tol 20.7189; 180-300: 262.9254 vs 266.1550 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.16 | 30-90: 329.9014 vs 327.4677 ± 2.7101, tol 8.1304 (restored 348.4055 ± 2.0543, tol 6.9681); 105-165: 280.9067 vs 285.8497 ± 10.5232, tol 31.5696 (restored 290.7766 ± 7.1545, tol 21.4634); 180-300: 271.8305 vs 276.9168 ± 14.2739, tol 42.8218 (restored 280.8797 ± 8.4111, tol 25.2332) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.30 | 30-90: 329.9014 vs 327.4677 ± 2.7101, tol 8.1304; 105-165: 280.9067 vs 285.8497 ± 10.5232, tol 31.5696; 180-300: 271.8305 vs 276.9168 ± 14.2739, tol 42.8218 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.07 | 30-90: 0.0030 vs 0.0030 ± 0.0059, tol 0.0500 (restored 0.0040 ± 0.0058, tol 0.0500); 105-165: 0.0037 vs 0.0037 ± 0.0114, tol 0.0500 (restored 0.0038 ± 0.0113, tol 0.0500); 180-300: 0.0025 vs 0.0025 ± 0.0094, tol 0.0500 (restored 0.0024 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.07 | 30-90: 0.0030 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0037 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0025 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.32 | 30-90: 6.228e+5 vs 6.329e+5 ± 17144.2000, tol 63287.2150; 105-165: 3.712e+5 vs 3.869e+5 ± 16355.6000, tol 49066.8000; 180-300: 3.360e+5 vs 3.516e+5 ± 23240.3333, tol 69721.0000 |

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

Check-slot step time (informational): candidate fenced median 60.19 ms; references 67.60, 45.86, 62.58, 66.20 ms.

## letters: letters-ref-e: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.00 | 120: -0.0088 vs 0.0089 ± 0.0004, tol 0.0050; 180: -0.0095 vs 0.0095 ± 0.0014, tol 0.0050; 240: -0.0109 vs 0.0110 ± 0.0009, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.08 | 120: -0.0088 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0095 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0109 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.08 | 120: -0.0088 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0095 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0109 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.22 | 60: 0.0208 vs 0.0208 ± 0.0364, tol 1.0000 (restored 0.2640 ± 0.0363, tol 1.0000); 90: 0.0888 vs 0.0888 ± 0.1444, tol 1.0000 (restored 0.4498 ± 0.1444, tol 1.0000); 120: 0.2610 vs 0.2610 ± 0.7010, tol 2.1030 (restored 0.4307 ± 0.7010, tol 2.1029); 180: 0.5502 vs 0.5502 ± 0.9631, tol 2.8893 (restored 0.6690 ± 0.9631, tol 2.8892); 240: 0.7321 vs 0.7321 ± 1.0952, tol 3.2856 (restored 0.8544 ± 1.0952, tol 3.2855) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.22 | 60: 0.0208 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.0888 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.2610 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.5502 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.7321 vs 0.0000 ± 1.0952, tol 3.2856 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.13 | 60: 1948.3336 vs 1948.6816 ± 2.2087, tol 194.8682; 120: 2254.8020 vs 2235.7534 ± 101.6255, tol 304.8766; 180: 2983.5616 vs 3071.0849 ± 219.4018, tol 658.2053; 240: 3022.6844 vs 3046.8040 ± 32.8480, tol 304.6804 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.11 | 60: 19.8925 vs 19.8925 ± 0.0023, tol 0.2500 (restored 20.0173 ± 0.0017, tol 0.2500); 120: 20.5478 vs 20.5481 ± 0.1422, tol 0.4266 (restored 20.7090 ± 0.1374, tol 0.4121); 180: 21.7061 vs 21.7331 ± 0.0328, tol 0.2500 (restored 21.8998 ± 0.0294, tol 0.2500); 240: 20.4963 vs 20.5010 ± 0.0231, tol 0.2500 (restored 20.7119 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.11 | 60: 19.8925 vs 19.8925 ± 0.0023, tol 0.2500; 120: 20.5478 vs 20.5481 ± 0.1422, tol 0.4266; 180: 21.7061 vs 21.7331 ± 0.0328, tol 0.2500; 240: 20.4963 vs 20.5010 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.10 | 60: 5.039e+5 vs 4.999e+5 ± 4146.0000, tol 49991.2000; 120: 8.978e+5 vs 8.891e+5 ± 7401.0000, tol 88908.8000; 180: 8.948e+5 vs 9.038e+5 ± 18893.0000, tol 90378.7500; 240: 8.740e+5 vs 8.788e+5 ± 6736.0000, tol 87883.4500 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.08 | 120: -0.0088 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0094 vs -0.0090 ± 0.0014, tol 0.0050; 240: -0.0109 vs -0.0106 ± 0.0009, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 94.06 ms; references 123.19, 96.65, 84.54, 87.50 ms.

## paddleboats: paddleboats-ref-e: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.00 | 150: -0.0029 vs 0.0031 ± 0.0001, tol 0.0025; 300: -0.0090 vs 0.0107 ± 0.0007, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.00 | 150: -0.0029 vs 0.0030 ± 0.0001, tol 0.0025; 300: -0.0090 vs 0.0105 ± 0.0007, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.61 | 150: -0.0029 vs -0.0030 ± 0.0001, tol 0.0025; 300: -0.0090 vs -0.0105 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.12 | 150: 0.0113 vs 0.0113 ± 0.0155, tol 0.2500 (restored 0.0698 ± 0.0153, tol 0.2500); 300: 0.0470 vs 0.0470 ± 0.0384, tol 0.2500 (restored 0.2173 ± 0.0382, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.19 | 150: 0.0113 vs 0.0000 ± 0.0155, tol 0.2500; 300: 0.0470 vs 0.0000 ± 0.0384, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.11 | 60: 94.3833 vs 95.4387 ± 1.3203, tol 9.5439; 120: 109.5510 vs 108.9906 ± 1.0608, tol 10.8991; 180: 141.8148 vs 141.8080 ± 1.5890, tol 14.1808; 240: 141.9499 vs 141.9755 ± 1.9409, tol 14.1976; 300: 168.0907 vs 167.0460 ± 2.3544, tol 16.7046 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.46 | 150: 2.648e+5 vs 2.550e+5 ± 26447.0000, tol 79341.0000; 300: 3.380e+5 vs 2.794e+5 ± 42221.0000, tol 1.267e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.60 | 150: -0.0082 vs -0.0084 ± 0.0001, tol 0.0025; 300: -0.0143 vs -0.0158 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 95.74 ms; references 51.19, 70.16, 50.72, 54.85 ms.
