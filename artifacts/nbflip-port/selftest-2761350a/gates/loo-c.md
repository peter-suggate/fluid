# NB-FLIP port gates: dam-ref-c, letters-ref-c, paddleboats-ref-c, seep-ref-c

Generated 2026-11-15T01:40:24.575Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-d + seep-ref-e; dam dam-ref-a + dam-ref-b + dam-ref-d + dam-ref-e; letters letters-ref-a + letters-ref-b + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-d + paddleboats-ref-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-c: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.05 | mean: 0.2698 vs 0.2691 ± 0.0020, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 120-180: 0.2704 vs 0.2694 ± 0.0022, tol 0.0269; 180-240: 0.2718 vs 0.2710 ± 0.0038, tol 0.0271; 240-300: 0.2712 vs 0.2711 ± 0.0023, tol 0.0271; 300-360: 0.2732 vs 0.2721 ± 0.0028, tol 0.0272; 360-420: 0.2713 vs 0.2723 ± 0.0035, tol 0.0272; 420-480: 0.2725 vs 0.2714 ± 0.0032, tol 0.0271; 480-540: 0.2712 vs 0.2694 ± 0.0030, tol 0.0269; 540-600: 0.2708 vs 0.2713 ± 0.0044, tol 0.0271; 600-660: 0.2708 vs 0.2688 ± 0.0031, tol 0.0269; 660-720: 0.2695 vs 0.2690 ± 0.0036, tol 0.0269; 720-780: 0.2697 vs 0.2682 ± 0.0029, tol 0.0268; 780-840: 0.2685 vs 0.2684 ± 0.0038, tol 0.0268; 840-900: 0.2698 vs 0.2680 ± 0.0039, tol 0.0268; 900-960: 0.2681 vs 0.2677 ± 0.0038, tol 0.0268; 960-1020: 0.2679 vs 0.2667 ± 0.0041, tol 0.0267; 1020-1080: 0.2678 vs 0.2668 ± 0.0023, tol 0.0267; 1080-1140: 0.2664 vs 0.2667 ± 0.0031, tol 0.0267; 1140-1200: 0.2646 vs 0.2656 ± 0.0036, tol 0.0266 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.05 | 300: 1.3207 vs 1.3193 ± 0.0049, tol 0.0660; 600: 2.6797 vs 2.6758 ± 0.0148, tol 0.1338; 900: 4.0280 vs 4.0182 ± 0.0259, tol 0.2009; 1200: 5.3628 vs 5.3515 ± 0.0355, tol 0.2676 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.19 | 300: 1.1398 vs 1.1543 ± 0.0253, tol 0.0760 (restored 1.3194 ± 0.0050, tol 0.0660); 600: 2.1321 vs 2.1246 ± 0.0754, tol 0.2263 (restored 2.6758 ± 0.0148, tol 0.1338); 900: 2.9801 vs 2.9445 ± 0.1341, tol 0.4024 (restored 4.0182 ± 0.0259, tol 0.2009); 1200: 3.4922 vs 3.4719 ± 0.2046, tol 0.6138 (restored 5.3516 ± 0.0356, tol 0.2676) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.19 | 300: 1.1398 vs 1.1543 ± 0.0253, tol 0.0760; 600: 2.1321 vs 2.1246 ± 0.0754, tol 0.2263; 900: 2.9801 vs 2.9445 ± 0.1341, tol 0.4024; 1200: 3.4922 vs 3.4719 ± 0.2046, tol 0.6138 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.02 | median: 2.6388 vs 2.6428 ± 0.0128, tol 0.2643 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.02 | 90%: 287.1308 vs 288.5674 ± 17.8628, tol 72.1418 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.01 | 1200: 16375.0000 vs 16368.7500 ± 11.0000, tol 818.4375 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.11 | 600: 367.9000 vs 368.6154 ± 3.9858, tol 16.0000 (restored 371.7712 ± 3.4575, tol 16.0000); 1200: 368.7668 vs 370.5492 ± 3.9751, tol 16.0000 (restored 375.2757 ± 2.3554, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.11 | 600: 367.9000 vs 368.6154 ± 3.9858, tol 16.0000; 1200: 368.7668 vs 370.5492 ± 3.9751, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.20 | 600: 0.0699 vs 0.0699 ± 0.1616, tol 0.4848 (restored 0.1269 ± 0.1321, tol 0.3962); 1200: 0.0460 vs 0.0460 ± 0.0756, tol 0.2268 (restored 0.1083 ± 0.0500, tol 0.1501) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.20 | 600: 0.0699 vs 0.0000 ± 0.1616, tol 0.4848; 1200: 0.0460 vs 0.0000 ± 0.0756, tol 0.2268 |
| G7 | total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.001 | pass | 0.00 | 1200: -0.0048 vs 0.0051 ± 0.0006, tol 0.0010 |
| G7 (first ruling) | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | (pass, not gated) | 0.00 | 1200: -0.0048 vs 0.0048 ± 0.0006, tol 0.0019 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.01 | 1200: -0.0048 vs -0.0048 ± 0.0006, tol 0.0019 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.13 | 300: 379.9163 vs 380.4097 ± 1.6943, tol 8.0000; 600: 382.1079 vs 380.3631 ± 4.3196, tol 12.9587 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.18 | 300: 0.7769 vs 0.0000 ± 1.4579, tol 4.3737; 600: 0.5539 vs 0.0000 ± 1.2470, tol 3.7409 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.31 | 300: 69021.0000 vs 69237.5000 ± 2891.0000, tol 8673.0000; 600: 1.323e+5 vs 1.296e+5 ± 5675.0000, tol 17025.0000; 900: 1.910e+5 vs 1.852e+5 ± 5630.0000, tol 18524.0500; 1200: 2.215e+5 vs 2.218e+5 ± 11539.0000, tol 34617.0000 |

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

Check-slot step time (informational): candidate fenced median 49.39 ms; references 54.13, 68.99, 53.44, 74.01 ms.

## dam: dam-ref-c: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2500 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8222 ± 0.0000, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1675 ± 0.0014, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5034 ± 0.0867, tol 4.0000); 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 (restored 80.8198 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0424 vs 0.0491 ± 0.0068, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0424 vs 0.0441 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.08 | 16: -0.0424 vs -0.0441 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.01 | 4: 36.0850 vs 36.0848 ± 0.0001, tol 3.6085; 8: 119.4794 vs 119.5018 ± 0.0036, tol 11.9502; 12: 228.6654 vs 228.5896 ± 0.4230, tol 22.8590; 16: 275.2734 vs 274.9411 ± 1.1711, tol 27.4941 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 4: 669.1690 vs 669.1683 ± 0.0002, tol 13.3834 (restored 669.9804 ± 0.0005, tol 13.3996); 8: 670.7537 vs 670.7931 ± 0.0075, tol 13.4159 (restored 671.9736 ± 0.0186, tol 13.4395); 12: 664.0906 vs 663.9639 ± 0.7382, tol 13.2793 (restored 667.5086 ± 0.2311, tol 13.3502); 16: 591.5167 vs 590.8194 ± 0.5015, tol 11.8164 (restored 613.4308 ± 3.2444, tol 12.2686) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.06 | 4: 669.1690 vs 669.1683 ± 0.0002, tol 13.3834; 8: 670.7537 vs 670.7931 ± 0.0075, tol 13.4159; 12: 664.0906 vs 663.9639 ± 0.7382, tol 13.2793; 16: 591.5167 vs 590.8194 ± 0.5015, tol 11.8164 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7625.0000 vs 7623.2500 ± 1.0000, tol 381.1625; 8: 10769.0000 vs 10767.0000 ± 4.0000, tol 538.3500; 12: 14931.0000 vs 14937.5000 ± 5.0000, tol 746.8750 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.22 | 8: 0.0078 vs 0.0078 ± 0.0156, tol 4.0000 (restored 0.0971 ± 0.0158, tol 4.0000); 12: 0.1248 vs 0.1248 ± 0.4293, tol 4.0000 (restored 0.3210 ± 0.4117, tol 4.0000); 16: 1.4396 vs 1.4396 ± 2.1413, tol 6.4239 (restored 2.3327 ± 2.0580, tol 6.1741) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.22 | 8: 0.0078 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.1248 vs 0.0000 ± 0.4293, tol 4.0000; 16: 1.4396 vs 0.0000 ± 2.1413, tol 6.4239 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.07 | 16: 3.497e+5 vs 3.520e+5 ± 4855.0000, tol 35203.1750 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.07 | 30-90: 45.4036 vs 45.2749 ± 0.9511, tol 2.8534 (restored 47.1297 ± 0.9143, tol 2.7430); 105-165: 47.3492 vs 47.0285 ± 0.6860, tol 2.3514 (restored 47.5354 ± 0.6676, tol 2.3768); 180-300: 46.7032 vs 46.8642 ± 0.5217, tol 2.3432 (restored 47.3235 ± 0.5712, tol 2.3662) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.14 | 30-90: 45.4036 vs 45.2749 ± 0.9511, tol 2.8534; 105-165: 47.3492 vs 47.0285 ± 0.6860, tol 2.3514; 180-300: 46.7032 vs 46.8642 ± 0.5217, tol 2.3432 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 30-90: -0.0406 vs 0.0446 ± 0.0090, tol 0.0100; 105-165: -0.0037 vs 0.0119 ± 0.0018, tol 0.0100; 180-300: 0.0036 vs 0.0114 ± 0.0022, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.01 | 30-90: -0.0406 vs 0.0403 ± 0.0090, tol 0.0270; 105-165: -0.0037 vs 0.0110 ± 0.0018, tol 0.0100; 180-300: 0.0036 vs 0.0100 ± 0.0022, tol 0.0100 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (fail, not gated) | 1.36 | 30-90: -0.0406 vs -0.0403 ± 0.0090, tol 0.0270; 105-165: -0.0037 vs -0.0110 ± 0.0018, tol 0.0100; 180-300: 0.0036 vs -0.0100 ± 0.0022, tol 0.0100 **x** |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.68 | 30-90: 47.1028 vs 46.3802 ± 0.9741, tol 4.6380; 105-165: 19.3674 vs 15.9675 ± 2.3972, tol 7.1917; 180-300: 13.2428 vs 9.6774 ± 1.7591, tol 5.2774 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.62 | 30-90: 281.6254 vs 281.3808 ± 4.4590, tol 13.3770 (restored 301.6917 ± 1.1612, tol 6.0338); 105-165: 273.4327 vs 266.9088 ± 2.1493, tol 6.4478 (restored 272.5819 ± 2.8570, tol 8.5709); 180-300: 272.9916 vs 263.6384 ± 2.6311, tol 7.8932 (restored 268.7814 ± 2.2650, tol 6.7949) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 1.18 | 30-90: 281.6254 vs 281.3808 ± 4.4590, tol 13.3770; 105-165: 273.4327 vs 266.9088 ± 2.1493, tol 6.4478 **x**; 180-300: 272.9916 vs 263.6384 ± 2.6311, tol 7.8932 **x** |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.64 | 30-90: 328.7282 vs 327.7610 ± 3.8834, tol 11.6502 (restored 348.0718 ± 1.7456, tol 6.9614); 105-165: 292.8001 vs 282.8764 ± 4.5465, tol 13.6395 (restored 288.5495 ± 5.2542, tol 15.7626); 180-300: 286.2344 vs 273.3158 ± 4.0276, tol 12.0828 (restored 278.4588 ± 4.0241, tol 12.0723) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 1.07 | 30-90: 328.7282 vs 327.7610 ± 3.8834, tol 11.6502; 105-165: 292.8001 vs 282.8764 ± 4.5465, tol 13.6395; 180-300: 286.2344 vs 273.3158 ± 4.0276, tol 12.0828 **x** |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.21 | 30-90: 0.0044 vs 0.0044 ± 0.0059, tol 0.0500 (restored 0.0041 ± 0.0058, tol 0.0500); 105-165: 0.0103 vs 0.0103 ± 0.0045, tol 0.0500 (restored 0.0103 ± 0.0045, tol 0.0500); 180-300: 0.0079 vs 0.0079 ± 0.0042, tol 0.0500 (restored 0.0078 ± 0.0042, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.21 | 30-90: 0.0044 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0103 vs 0.0000 ± 0.0045, tol 0.0500; 180-300: 0.0079 vs 0.0000 ± 0.0042, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.57 | 30-90: 6.330e+5 vs 6.303e+5 ± 17318.6000, tol 63032.1500; 105-165: 3.809e+5 vs 3.845e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.662e+5 vs 3.440e+5 ± 13056.6667, tol 39170.0000 |

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

Check-slot step time (informational): candidate fenced median 62.58 ms; references 67.60, 45.86, 66.20, 60.19 ms.

## letters: letters-ref-c: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.03 | 120: -0.0086 vs 0.0089 ± 0.0002, tol 0.0050; 180: -0.0095 vs 0.0095 ± 0.0013, tol 0.0050; 240: -0.0110 vs 0.0109 ± 0.0008, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.10 | 120: -0.0086 vs 0.0088 ± 0.0002, tol 0.0050; 180: -0.0095 vs 0.0090 ± 0.0013, tol 0.0050; 240: -0.0110 vs 0.0106 ± 0.0008, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.10 | 120: -0.0086 vs -0.0088 ± 0.0002, tol 0.0050; 180: -0.0095 vs -0.0090 ± 0.0013, tol 0.0050; 240: -0.0110 vs -0.0106 ± 0.0008, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.60 | 60: 0.0237 vs 0.0237 ± 0.0355, tol 1.0000 (restored 0.2649 ± 0.0354, tol 1.0000); 90: 0.1170 vs 0.1170 ± 0.1232, tol 1.0000 (restored 0.4599 ± 0.1232, tol 1.0000); 120: 0.6355 vs 0.6355 ± 0.3540, tol 1.0621 (restored 0.7185 ± 0.3540, tol 1.0619); 180: 0.6928 vs 0.6928 ± 0.8010, tol 2.4030 (restored 0.7922 ± 0.7995, tol 2.3986); 240: 0.9216 vs 0.9216 ± 1.0435, tol 3.1306 (restored 1.0237 ± 1.0435, tol 3.1305) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.60 | 60: 0.0237 vs 0.0000 ± 0.0355, tol 1.0000; 90: 0.1170 vs 0.0000 ± 0.1232, tol 1.0000; 120: 0.6355 vs 0.0000 ± 0.3540, tol 1.0621; 180: 0.6928 vs 0.0000 ± 0.8010, tol 2.4030; 240: 0.9216 vs 0.0000 ± 1.0435, tol 3.1306 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.36 | 60: 1947.8332 vs 1948.8067 ± 2.0109, tol 194.8807; 120: 2304.2648 vs 2223.3877 ± 52.1628, tol 222.3388; 180: 3170.2201 vs 3024.4203 ± 163.4279, tol 490.2836; 240: 3031.1757 vs 3044.6812 ± 41.3393, tol 304.4681 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.07 | 60: 19.8913 vs 19.8928 ± 0.0014, tol 0.2500 (restored 20.0175 ± 0.0017, tol 0.2500); 120: 20.6460 vs 20.5235 ± 0.0440, tol 0.2500 (restored 20.6855 ± 0.0434, tol 0.2500); 180: 21.7140 vs 21.7311 ± 0.0407, tol 0.2500 (restored 21.8974 ± 0.0390, tol 0.2500); 240: 20.4920 vs 20.5021 ± 0.0199, tol 0.2500 (restored 20.7123 ± 0.0048, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.49 | 60: 19.8913 vs 19.8928 ± 0.0014, tol 0.2500; 120: 20.6460 vs 20.5235 ± 0.0440, tol 0.2500; 180: 21.7140 vs 21.7311 ± 0.0407, tol 0.2500; 240: 20.4920 vs 20.5021 ± 0.0199, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.06 | 60: 4.993e+5 vs 5.011e+5 ± 6307.0000, tol 50105.7750; 120: 8.880e+5 vs 8.915e+5 ± 11988.0000, tol 89152.5750; 180: 9.063e+5 vs 9.009e+5 ± 18947.0000, tol 90090.1000; 240: 8.812e+5 vs 8.770e+5 ± 6268.0000, tol 87703.7750 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.10 | 120: -0.0086 vs -0.0088 ± 0.0002, tol 0.0050; 180: -0.0095 vs -0.0090 ± 0.0013, tol 0.0050; 240: -0.0110 vs -0.0106 ± 0.0008, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 84.54 ms; references 123.19, 96.65, 87.50, 94.06 ms.

## paddleboats: paddleboats-ref-c: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.04 | 150: -0.0031 vs 0.0030 ± 0.0001, tol 0.0025; 300: -0.0107 vs 0.0107 ± 0.0018, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.12 | 150: -0.0031 vs 0.0030 ± 0.0001, tol 0.0025; 300: -0.0107 vs 0.0101 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.12 | 150: -0.0031 vs -0.0030 ± 0.0001, tol 0.0025; 300: -0.0107 vs -0.0101 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.11 | 150: 0.0136 vs 0.0136 ± 0.0135, tol 0.2500 (restored 0.0763 ± 0.0135, tol 0.2500); 300: 0.0272 vs 0.0272 ± 0.0590, tol 0.2500 (restored 0.2571 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 150: 0.0136 vs 0.0000 ± 0.0135, tol 0.2500; 300: 0.0272 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.05 | 60: 94.8290 vs 95.3273 ± 1.7659, tol 9.5327; 120: 109.4333 vs 109.0200 ± 1.1784, tol 10.9020; 180: 142.3156 vs 141.6828 ± 1.5890, tol 14.1683; 240: 142.1792 vs 141.9182 ± 1.9409, tol 14.1918; 300: 167.7261 vs 167.1372 ± 2.4230, tol 16.7137 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.12 | 150: 2.623e+5 vs 2.556e+5 ± 26447.0000, tol 79341.0000; 300: 2.684e+5 vs 2.968e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.12 | 150: -0.0085 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0160 vs -0.0154 ± 0.0017, tol 0.0052 |

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

Check-slot step time (informational): candidate fenced median 50.72 ms; references 51.19, 70.16, 54.85, 95.74 ms.
