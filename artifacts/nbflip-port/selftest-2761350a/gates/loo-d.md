# NB-FLIP port gates: dam-ref-d, letters-ref-d, paddleboats-ref-d, seep-ref-d

Generated 2026-11-15T01:40:24.934Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-e; dam dam-ref-a + dam-ref-b + dam-ref-c + dam-ref-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-d: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.12 | mean: 0.2679 vs 0.2696 ± 0.0011, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.12 | 120-180: 0.2697 vs 0.2696 ± 0.0022, tol 0.0270; 180-240: 0.2686 vs 0.2719 ± 0.0008, tol 0.0272; 240-300: 0.2710 vs 0.2711 ± 0.0023, tol 0.0271; 300-360: 0.2721 vs 0.2724 ± 0.0028, tol 0.0272; 360-420: 0.2705 vs 0.2725 ± 0.0027, tol 0.0273; 420-480: 0.2696 vs 0.2721 ± 0.0015, tol 0.0272; 480-540: 0.2679 vs 0.2702 ± 0.0022, tol 0.0270; 540-600: 0.2692 vs 0.2717 ± 0.0033, tol 0.0272; 600-660: 0.2675 vs 0.2697 ± 0.0034, tol 0.0270; 660-720: 0.2671 vs 0.2695 ± 0.0023, tol 0.0270; 720-780: 0.2676 vs 0.2687 ± 0.0029, tol 0.0269; 780-840: 0.2678 vs 0.2686 ± 0.0038, tol 0.0269; 840-900: 0.2670 vs 0.2688 ± 0.0039, tol 0.0269; 900-960: 0.2657 vs 0.2683 ± 0.0028, tol 0.0268; 960-1020: 0.2648 vs 0.2674 ± 0.0037, tol 0.0267; 1020-1080: 0.2667 vs 0.2670 ± 0.0023, tol 0.0267; 1080-1140: 0.2666 vs 0.2666 ± 0.0031, tol 0.0267; 1140-1200: 0.2633 vs 0.2659 ± 0.0023, tol 0.0266 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.11 | 300: 1.3173 vs 1.3202 ± 0.0039, tol 0.0660; 600: 2.6666 vs 2.6791 ± 0.0048, tol 0.1340; 900: 4.0037 vs 4.0243 ± 0.0170, tol 0.2012; 1200: 5.3308 vs 5.3595 ± 0.0223, tol 0.2680 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.40 | 300: 1.1443 vs 1.1532 ± 0.0298, tol 0.0895 (restored 1.3202 ± 0.0039, tol 0.0660); 600: 2.0816 vs 2.1372 ± 0.0469, tol 0.1407 (restored 2.6791 ± 0.0049, tol 0.1340); 900: 2.9740 vs 2.9460 ± 0.1341, tol 0.4024 (restored 4.0243 ± 0.0170, tol 0.2012); 1200: 3.5754 vs 3.4511 ± 0.1308, tol 0.3923 (restored 5.3596 ± 0.0223, tol 0.2680) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.40 | 300: 1.1443 vs 1.1532 ± 0.0298, tol 0.0895; 600: 2.0816 vs 2.1372 ± 0.0469, tol 0.1407; 900: 2.9740 vs 2.9460 ± 0.1341, tol 0.4024; 1200: 3.5754 vs 3.4511 ± 0.1308, tol 0.3923 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.00 | median: 2.6410 vs 2.6423 ± 0.0128, tol 0.2642 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.07 | 90%: 284.1495 vs 289.3127 ± 17.8628, tol 72.3282 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.01 | 1200: 16363.0000 vs 16371.7500 ± 8.0000, tol 818.5875 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.00 | 600: 369.4865 vs 368.2187 ± 3.9858, tol 16.0000 (restored 371.4011 ± 3.4575, tol 16.0000); 1200: 370.4401 vs 370.1308 ± 4.0817, tol 16.0000 (restored 375.0695 ± 2.7714, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.08 | 600: 369.4865 vs 368.2187 ± 3.9858, tol 16.0000; 1200: 370.4401 vs 370.1308 ± 4.0817, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.23 | 600: 0.0628 vs 0.0628 ± 0.1616, tol 0.4848 (restored 0.0647 ± 0.1321, tol 0.3962); 1200: 0.0524 vs 0.0524 ± 0.0756, tol 0.2268 (restored 0.1014 ± 0.0464, tol 0.1500) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 600: 0.0628 vs 0.0000 ± 0.1616, tol 0.4848; 1200: 0.0524 vs 0.0000 ± 0.0756, tol 0.2268 |
| G7 | total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.001 | pass | 0.00 | 1200: -0.0044 vs 0.0051 ± 0.0004, tol 0.0010 |
| G7 (first ruling) | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | (pass, not gated) | 0.00 | 1200: -0.0044 vs 0.0049 ± 0.0004, tol 0.0012 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.34 | 1200: -0.0044 vs -0.0049 ± 0.0004, tol 0.0012 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.17 | 300: 381.0613 vs 380.1235 ± 1.6943, tol 8.0000; 600: 378.9768 vs 381.1459 ± 4.3196, tol 12.9587 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.25 | 300: 0.8539 vs 0.0000 ± 1.4579, tol 4.3737; 600: 0.8478 vs 0.0000 ± 1.1111, tol 3.3332 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.37 | 300: 67660.0000 vs 69577.7500 ± 1530.0000, tol 6957.7750; 600: 1.263e+5 vs 1.311e+5 ± 3382.0000, tol 13111.8750; 900: 1.867e+5 vs 1.863e+5 ± 9143.0000, tol 27429.0000; 1200: 2.237e+5 vs 2.213e+5 ± 11539.0000, tol 34617.0000 |

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

Check-slot step time (informational): candidate fenced median 53.44 ms; references 54.13, 68.99, 49.39, 74.01 ms.

## dam: dam-ref-d: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2500 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1680 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4836 ± 0.0074, tol 4.0000); 16: 78.8125 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.7244 ± 0.0194, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.02 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.8125 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.64 | 16: -0.0491 vs 0.0427 ± 0.0004, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.67 | 16: -0.0491 vs 0.0424 ± 0.0004, tol 0.0100 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.67 | 16: -0.0491 vs -0.0424 ± 0.0004, tol 0.0100 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.03 | 4: 36.0848 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.4995 vs 119.4968 ± 0.0237, tol 11.9497; 12: 228.2819 vs 228.6855 ± 0.0394, tol 22.8685; 16: 275.7542 vs 274.8209 ± 0.6903, tol 27.4821 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.05 | 4: 669.1684 vs 669.1685 ± 0.0008, tol 13.3834 (restored 669.9802 ± 0.0008, tol 13.3996); 8: 670.7891 vs 670.7842 ± 0.0429, tol 13.4157 (restored 671.9697 ± 0.0280, tol 13.4394); 12: 663.4202 vs 664.1315 ± 0.0678, tol 13.2826 (restored 667.4546 ± 0.0151, tol 13.3491); 16: 590.5914 vs 591.0507 ± 0.7740, tol 11.8210 (restored 612.8051 ± 0.7414, tol 12.2561) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.05 | 4: 669.1684 vs 669.1685 ± 0.0008, tol 13.3834; 8: 670.7891 vs 670.7842 ± 0.0429, tol 13.4157; 12: 663.4202 vs 664.1315 ± 0.0678, tol 13.2826; 16: 590.5914 vs 591.0507 ± 0.7740, tol 11.8210 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7623.0000 vs 7623.7500 ± 2.0000, tol 381.1875; 8: 10766.0000 vs 10767.7500 ± 4.0000, tol 538.3875; 12: 14941.0000 vs 14935.0000 ± 6.0000, tol 746.7500 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.28 | 8: 0.0117 vs 0.0117 ± 0.0156, tol 4.0000 (restored 0.0925 ± 0.0155, tol 4.0000); 12: 0.4270 vs 0.4270 ± 0.0716, tol 4.0000 (restored 0.5826 ± 0.0707, tol 4.0000); 16: 1.5553 vs 1.5553 ± 1.8263, tol 5.4790 (restored 2.8533 ± 1.8201, tol 5.4604) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.28 | 8: 0.0117 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.4270 vs 0.0000 ± 0.0716, tol 4.0000; 16: 1.5553 vs 0.0000 ± 1.8263, tol 5.4790 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.14 | 16: 3.555e+5 vs 3.506e+5 ± 1569.0000, tol 35059.2750 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.16 | 30-90: 44.9307 vs 45.3932 ± 0.9476, tol 2.8427 (restored 47.2029 ± 0.9143, tol 2.7430); 105-165: 47.1820 vs 47.0703 ± 0.7294, tol 2.3535 (restored 47.5001 ± 0.6676, tol 2.3750); 180-300: 46.7635 vs 46.8492 ± 0.5217, tol 2.3425 (restored 47.1775 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.16 | 30-90: 44.9307 vs 45.3932 ± 0.9476, tol 2.8427; 105-165: 47.1820 vs 47.0703 ± 0.7294, tol 2.3535; 180-300: 46.7635 vs 46.8492 ± 0.5217, tol 2.3425 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.36 | 30-90: -0.0446 vs 0.0409 ± 0.0054, tol 0.0100; 105-165: -0.0115 vs 0.0119 ± 0.0082, tol 0.0100; 180-300: -0.0114 vs 0.0102 ± 0.0138, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.32 | 30-90: -0.0446 vs 0.0393 ± 0.0054, tol 0.0161; 105-165: -0.0115 vs 0.0091 ± 0.0082, tol 0.0246; 180-300: -0.0114 vs 0.0062 ± 0.0138, tol 0.0415 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.32 | 30-90: -0.0446 vs -0.0393 ± 0.0054, tol 0.0161; 105-165: -0.0115 vs -0.0091 ± 0.0082, tol 0.0246; 180-300: -0.0114 vs -0.0062 ± 0.0138, tol 0.0415 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.10 | 30-90: 46.8651 vs 46.4396 ± 1.2119, tol 4.6440; 105-165: 15.7506 vs 16.8718 ± 4.3305, tol 12.9914; 180-300: 9.3976 vs 10.6387 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.32 | 30-90: 279.1530 vs 281.9989 ± 2.9836, tol 8.9508 (restored 301.8246 ± 1.1612, tol 6.0365); 105-165: 266.5264 vs 268.6354 ± 7.5629, tol 22.6888 (restored 273.4469 ± 4.5779, tol 13.7338); 180-300: 262.5628 vs 266.2456 ± 10.0662, tol 30.1985 (restored 269.9229 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.32 | 30-90: 279.1530 vs 281.9989 ± 2.9836, tol 8.9508; 105-165: 266.5264 vs 268.6354 ± 7.5629, tol 22.6888; 180-300: 262.5628 vs 266.2456 ± 10.0662, tol 30.1985 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.24 | 30-90: 326.0180 vs 328.4385 ± 3.3822, tol 10.1465 (restored 348.2642 ± 2.0543, tol 6.9653); 105-165: 282.2769 vs 285.5072 ± 11.8934, tol 35.6801 (restored 290.3187 ± 8.9084, tol 26.7252); 180-300: 271.9604 vs 276.8843 ± 14.4039, tol 43.2116 (restored 280.5616 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.24 | 30-90: 326.0180 vs 328.4385 ± 3.3822, tol 10.1465; 105-165: 282.2769 vs 285.5072 ± 11.8934, tol 35.6801; 180-300: 271.9604 vs 276.8843 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.10 | 30-90: 0.0049 vs 0.0049 ± 0.0057, tol 0.0500 (restored 0.0073 ± 0.0058, tol 0.0500); 105-165: 0.0037 vs 0.0037 ± 0.0114, tol 0.0500 (restored 0.0034 ± 0.0113, tol 0.0500); 180-300: 0.0022 vs 0.0022 ± 0.0094, tol 0.0500 (restored 0.0023 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.10 | 30-90: 0.0049 vs 0.0000 ± 0.0057, tol 0.0500; 105-165: 0.0037 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0022 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.18 | 30-90: 6.401e+5 vs 6.285e+5 ± 12508.2000, tol 62854.2500; 105-165: 3.871e+5 vs 3.829e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.490e+5 vs 3.483e+5 ± 30193.3333, tol 90580.0000 |

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

Check-slot step time (informational): candidate fenced median 66.20 ms; references 67.60, 45.86, 62.58, 60.19 ms.

## letters: letters-ref-d: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.03 | 120: -0.0089 vs 0.0088 ± 0.0002, tol 0.0050; 180: -0.0092 vs 0.0095 ± 0.0014, tol 0.0050; 240: -0.0106 vs 0.0110 ± 0.0009, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.05 | 120: -0.0089 vs 0.0087 ± 0.0002, tol 0.0050; 180: -0.0092 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0106 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.05 | 120: -0.0089 vs -0.0087 ± 0.0002, tol 0.0050; 180: -0.0092 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0106 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.21 | 60: 0.0223 vs 0.0223 ± 0.0364, tol 1.0000 (restored 0.2638 ± 0.0363, tol 1.0000); 90: 0.0860 vs 0.0860 ± 0.1415, tol 1.0000 (restored 0.4545 ± 0.1415, tol 1.0000); 120: 0.2698 vs 0.2698 ± 0.7010, tol 2.1030 (restored 0.4415 ± 0.7010, tol 2.1029); 180: 0.4284 vs 0.4284 ± 0.9631, tol 2.8893 (restored 0.5656 ± 0.9631, tol 2.8892); 240: 0.7053 vs 0.7053 ± 1.1402, tol 3.4206 (restored 0.8260 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.21 | 60: 0.0223 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.0860 vs 0.0000 ± 0.1415, tol 1.0000; 120: 0.2698 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.4284 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.7053 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 60: 1950.0419 vs 1948.2545 ± 0.9871, tol 194.8255; 120: 2221.6367 vs 2244.0447 ± 101.6255, tol 304.8766; 180: 3049.0552 vs 3054.7115 ± 219.4018, tol 658.2053; 240: 3036.0942 vs 3043.4516 ± 41.3393, tol 304.3452 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.06 | 60: 19.8933 vs 19.8923 ± 0.0023, tol 0.2500 (restored 20.0171 ± 0.0011, tol 0.2500); 120: 20.5289 vs 20.5528 ± 0.1422, tol 0.4266 (restored 20.7128 ± 0.1374, tol 0.4121); 180: 21.7272 vs 21.7278 ± 0.0407, tol 0.2500 (restored 21.8959 ± 0.0390, tol 0.2500); 240: 20.5018 vs 20.4997 ± 0.0231, tol 0.2500 (restored 20.7118 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.06 | 60: 19.8933 vs 19.8923 ± 0.0023, tol 0.2500; 120: 20.5289 vs 20.5528 ± 0.1422, tol 0.4266; 180: 21.7272 vs 21.7278 ± 0.0407, tol 0.2500; 240: 20.5018 vs 20.4997 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.03 | 60: 5.018e+5 vs 5.005e+5 ± 6307.0000, tol 50045.2250; 120: 8.932e+5 vs 8.902e+5 ± 11988.0000, tol 89023.4750; 180: 9.003e+5 vs 9.024e+5 ± 18947.0000, tol 90240.4750; 240: 8.794e+5 vs 8.775e+5 ± 7187.0000, tol 87748.6500 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.05 | 120: -0.0089 vs -0.0087 ± 0.0002, tol 0.0050; 180: -0.0091 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0106 vs -0.0107 ± 0.0009, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 87.50 ms; references 123.19, 96.65, 84.54, 94.06 ms.

## paddleboats: paddleboats-ref-d: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.00 | 150: -0.0030 vs 0.0031 ± 0.0003, tol 0.0025; 300: -0.0100 vs 0.0107 ± 0.0018, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.00 | 150: -0.0030 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0100 vs 0.0102 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.04 | 150: -0.0030 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0100 vs -0.0102 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.11 | 150: 0.0107 vs 0.0107 ± 0.0176, tol 0.2500 (restored 0.0731 ± 0.0164, tol 0.2500); 300: 0.0271 vs 0.0271 ± 0.0590, tol 0.2500 (restored 0.2421 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 150: 0.0107 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.0271 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.12 | 60: 96.1493 vs 94.9973 ± 1.0424, tol 9.4997; 120: 109.1198 vs 109.0984 ± 1.1784, tol 10.9098; 180: 142.4505 vs 141.6491 ± 1.4542, tol 14.1649; 240: 142.6455 vs 141.8016 ± 1.6682, tol 14.1802; 300: 166.7681 vs 167.3766 ± 2.4230, tol 16.7377 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.14 | 150: 2.658e+5 vs 2.547e+5 ± 25492.0000, tol 76476.0000; 300: 2.901e+5 vs 2.914e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.04 | 150: -0.0083 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0153 vs -0.0155 ± 0.0017, tol 0.0052 |

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

Check-slot step time (informational): candidate fenced median 54.85 ms; references 51.19, 70.16, 50.72, 95.74 ms.
