# NB-FLIP port gates: dam-ref-b, letters-ref-b, paddleboats-ref-b, seep-ref-b

Generated 2026-11-15T01:40:27.098Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-c + seep-ref-d + seep-base-e; dam dam-ref-a + dam-ref-d + dam-base-e; letters letters-ref-a + letters-ref-c + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-c + paddleboats-ref-d + paddleboats-base-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-b: PASS

References seep-ref-a.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.00 | mean: 0.2697 vs 0.2698 ± 0.0002, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 120-180: 0.2684 vs 0.2696 ± 0.0022, tol 0.0270; 180-240: 0.2704 vs 0.2718 ± 0.0020, tol 0.0272; 240-300: 0.2707 vs 0.2708 ± 0.0018, tol 0.0271; 300-360: 0.2717 vs 0.2730 ± 0.0011, tol 0.0273; 360-420: 0.2731 vs 0.2711 ± 0.0009, tol 0.0271; 420-480: 0.2722 vs 0.2724 ± 0.0006, tol 0.0272; 480-540: 0.2710 vs 0.2710 ± 0.0007, tol 0.0271; 540-600: 0.2716 vs 0.2712 ± 0.0015, tol 0.0271; 600-660: 0.2701 vs 0.2705 ± 0.0008, tol 0.0271; 660-720: 0.2708 vs 0.2697 ± 0.0005, tol 0.0270; 720-780: 0.2687 vs 0.2698 ± 0.0002, tol 0.0270; 780-840: 0.2696 vs 0.2685 ± 0.0004, tol 0.0269; 840-900: 0.2691 vs 0.2686 ± 0.0017, tol 0.0269; 900-960: 0.2678 vs 0.2682 ± 0.0003, tol 0.0268; 960-1020: 0.2691 vs 0.2683 ± 0.0006, tol 0.0268; 1020-1080: 0.2673 vs 0.2677 ± 0.0011, tol 0.0268; 1080-1140: 0.2682 vs 0.2668 ± 0.0009, tol 0.0267; 1140-1200: 0.2654 vs 0.2669 ± 0.0013, tol 0.0267 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.17 | 300: 1.3120 vs 1.3231 ± 0.0124, tol 0.0662; 600: 2.6716 vs 2.6818 ± 0.0110, tol 0.1341; 900: 4.0198 vs 4.0290 ± 0.0120, tol 0.2014; 1200: 5.3575 vs 5.3668 ± 0.0131, tol 0.2683 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.07 | 300: 1.1939 vs 1.1467 ± 0.0455, tol 0.1366 (restored 1.3231 ± 0.0124, tol 0.0662); 600: 2.1252 vs 2.1060 ± 0.0928, tol 0.2784 (restored 2.6818 ± 0.0110, tol 0.1341); 900: 2.8558 vs 2.8779 ± 0.1074, tol 0.3223 (restored 4.0291 ± 0.0120, tol 0.2015); 1200: 3.3930 vs 3.3760 ± 0.1684, tol 0.5053 (restored 5.3669 ± 0.0131, tol 0.2683) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.35 | 300: 1.1939 vs 1.1467 ± 0.0455, tol 0.1366; 600: 2.1252 vs 2.1060 ± 0.0928, tol 0.2784; 900: 2.8558 vs 2.8779 ± 0.1074, tol 0.3223; 1200: 3.3930 vs 3.3760 ± 0.1684, tol 0.5053 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6406 vs 2.6382 ± 0.0038, tol 0.2638 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.01 | 90%: 291.2792 vs 291.9893 ± 4.1487, tol 72.9973 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.01 | 1200: 16358.0000 vs 16351.0000 ± 72.0000, tol 817.5500 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.13 | 600: 365.9785 vs 368.0552 ± 1.0484, tol 16.0000 (restored 371.4787 ± 0.9582, tol 16.0000); 1200: 371.2848 vs 367.9669 ± 4.7394, tol 16.0000 (restored 373.9361 ± 2.5504, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.21 | 600: 365.9785 vs 368.0552 ± 1.0484, tol 16.0000; 1200: 371.2848 vs 367.9669 ± 4.7394, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.21 | 600: 0.0701 vs 0.0701 ± 0.1428, tol 0.4283 (restored 0.1407 ± 0.1145, tol 0.3435); 1200: 0.0620 vs 0.0620 ± 0.0985, tol 0.2955 (restored 0.1084 ± 0.0574, tol 0.1722) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.21 | 600: 0.0701 vs 0.0000 ± 0.1428, tol 0.4283; 1200: 0.0620 vs 0.0000 ± 0.0985, tol 0.2955 |
| G7 | total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.001 | pass | 0.00 | 1200: -0.0050 vs 0.0054 ± 0.0004, tol 0.0010 |
| G7 (first ruling) | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | (pass, not gated) | 0.00 | 1200: -0.0050 vs 0.0051 ± 0.0004, tol 0.0013 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.06 | 1200: -0.0050 vs -0.0051 ± 0.0004, tol 0.0013 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.30 | 300: 379.6174 vs 381.8160 ± 2.4470, tol 8.0000; 600: 378.9285 vs 381.3266 ± 1.8381, tol 8.0000 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.23 | 300: 0.4093 vs 0.0000 ± 1.3113, tol 3.9339; 600: 0.7458 vs 0.0000 ± 1.0738, tol 3.2214 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.44 | 300: 71483.0000 vs 68450.0000 ± 2287.0000, tol 6861.0000; 600: 1.305e+5 vs 1.298e+5 ± 8388.0000, tol 25164.0000; 900: 1.831e+5 vs 1.832e+5 ± 6281.0000, tol 18843.0000; 1200: 2.200e+5 vs 2.169e+5 ± 14011.0000, tol 42033.0000 |

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

Check-slot step time (informational): candidate fenced median 37.57 ms; references 37.24, 35.16, 65.34, 34.40 ms.

## dam: dam-ref-b: PASS

References dam-ref-a.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.08 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 107.6667 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8225 ± 0.0010, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1663 ± 0.0036, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4923 ± 0.0096, tol 4.0000); 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.8412 ± 0.3623, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0426 vs 0.0500 ± 0.0078, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0426 vs 0.0449 ± 0.0078, tol 0.0234 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.10 | 16: -0.0426 vs -0.0449 ± 0.0078, tol 0.0234 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0848 vs 36.0847 ± 0.0001, tol 3.6085; 8: 119.4964 vs 119.5041 ± 0.0183, tol 11.9504; 12: 228.7544 vs 228.6662 ± 0.0415, tol 22.8666; 16: 274.6937 vs 274.1617 ± 2.1784, tol 27.4162 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 4: 669.1682 vs 669.1671 ± 0.0033, tol 13.3833 (restored 669.9827 ± 0.0077, tol 13.3997); 8: 670.7870 vs 670.7956 ± 0.0180, tol 13.4159 (restored 671.9625 ± 0.0225, tol 13.4393); 12: 664.1874 vs 664.0685 ± 0.0659, tol 13.2814 (restored 667.4890 ± 0.0445, tol 13.3498); 16: 590.8653 vs 589.7230 ± 4.1982, tol 12.5946 (restored 612.7315 ± 0.4477, tol 12.2546) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.09 | 4: 669.1682 vs 669.1671 ± 0.0033, tol 13.3833; 8: 670.7870 vs 670.7956 ± 0.0180, tol 13.4159; 12: 664.1874 vs 664.0685 ± 0.0659, tol 13.2814; 16: 590.8653 vs 589.7230 ± 4.1982, tol 12.5946 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.00 | 4: 7624.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10767.0000 vs 10764.3333 ± 9.0000, tol 538.2167; 12: 14938.0000 vs 14937.6667 ± 5.0000, tol 746.8833 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.11 | 8: 0.0505 vs 0.0505 ± 0.1441, tol 4.0000 (restored 0.1058 ± 0.1647, tol 4.0000); 12: 0.1209 vs 0.1209 ± 0.4137, tol 4.0000 (restored 0.3041 ± 0.4138, tol 4.0000); 16: 0.5688 vs 0.5688 ± 1.6789, tol 5.0368 (restored 2.1946 ± 1.5458, tol 4.6375) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 8: 0.0505 vs 0.0000 ± 0.1441, tol 4.0000; 12: 0.1209 vs 0.0000 ± 0.4137, tol 4.0000; 16: 0.5688 vs 0.0000 ± 1.6789, tol 5.0368 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.12 | 16: 3.540e+5 vs 3.499e+5 ± 9091.0000, tol 34990.2667 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.21 | 30-90: 44.9512 vs 44.7447 ± 0.9607, tol 2.8820 (restored 46.5123 ± 1.2089, tol 3.6267); 105-165: 47.5186 vs 46.5010 ± 0.7610, tol 2.3250 (restored 47.0183 ± 0.7160, tol 2.3509); 180-300: 46.7926 vs 46.6053 ± 1.1031, tol 3.3093 (restored 47.0819 ± 1.0824, tol 3.2473) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.44 | 30-90: 44.9512 vs 44.7447 ± 0.9607, tol 2.8820; 105-165: 47.5186 vs 46.5010 ± 0.7610, tol 2.3250; 180-300: 46.7926 vs 46.6053 ± 1.1031, tol 3.3093 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.01 | 30-90: -0.0421 vs 0.0420 ± 0.0054, tol 0.0100; 105-165: -0.0107 vs 0.0120 ± 0.0014, tol 0.0100; 180-300: -0.0087 vs 0.0110 ± 0.0016, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.23 | 30-90: -0.0421 vs 0.0384 ± 0.0054, tol 0.0162; 105-165: -0.0107 vs 0.0112 ± 0.0014, tol 0.0100; 180-300: -0.0087 vs 0.0104 ± 0.0016, tol 0.0100 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.23 | 30-90: -0.0421 vs -0.0384 ± 0.0054, tol 0.0162; 105-165: -0.0107 vs -0.0112 ± 0.0014, tol 0.0100; 180-300: -0.0087 vs -0.0104 ± 0.0016, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.26 | 30-90: 46.6916 vs 46.0514 ± 2.3587, tol 7.0762; 105-165: 16.3185 vs 14.6094 ± 2.1560, tol 6.4680; 180-300: 10.4468 vs 8.3479 ± 2.8969, tol 8.6906 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.22 | 30-90: 279.8352 vs 281.0900 ± 1.8315, tol 5.6218 (restored 300.4662 ± 2.1264, tol 6.3793); 105-165: 267.9398 vs 265.1108 ± 2.7182, tol 8.1547 (restored 270.9000 ± 2.2171, tol 6.6513); 180-300: 265.2905 vs 261.7819 ± 3.6517, tol 10.9550 (restored 267.1194 ± 2.8355, tol 8.5065) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.35 | 30-90: 279.8352 vs 281.0900 ± 1.8315, tol 5.6218; 105-165: 267.9398 vs 265.1108 ± 2.7182, tol 8.1547; 180-300: 265.2905 vs 261.7819 ± 3.6517, tol 10.9550 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.08 | 30-90: 326.5268 vs 327.1414 ± 2.6682, tol 8.0046 (restored 346.5176 ± 4.4852, tol 13.4555); 105-165: 284.2583 vs 279.7203 ± 4.8742, tol 14.6226 (restored 285.5094 ± 4.3731, tol 13.1192); 180-300: 275.7374 vs 270.1298 ± 6.5485, tol 19.6456 (restored 275.4673 ± 5.7324, tol 17.1971) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.31 | 30-90: 326.5268 vs 327.1414 ± 2.6682, tol 8.0046; 105-165: 284.2583 vs 279.7203 ± 4.8742, tol 14.6226; 180-300: 275.7374 vs 270.1298 ± 6.5485, tol 19.6456 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.11 | 30-90: 0.0051 vs 0.0051 ± 0.0058, tol 0.0500 (restored 0.0064 ± 0.0057, tol 0.0500); 105-165: 0.0057 vs 0.0057 ± 0.0051, tol 0.0500 (restored 0.0056 ± 0.0052, tol 0.0500); 180-300: 0.0044 vs 0.0044 ± 0.0055, tol 0.0500 (restored 0.0044 ± 0.0055, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.11 | 30-90: 0.0051 vs 0.0000 ± 0.0058, tol 0.0500; 105-165: 0.0057 vs 0.0000 ± 0.0051, tol 0.0500; 180-300: 0.0044 vs 0.0000 ± 0.0055, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.66 | 30-90: 6.416e+5 vs 6.171e+5 ± 34956.4000, tol 1.049e+5; 105-165: 4.013e+5 vs 3.765e+5 ± 10437.8000, tol 37648.8067; 180-300: 3.651e+5 vs 3.405e+5 ± 21445.6667, tol 64337.0000 |

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

Check-slot step time (informational): candidate fenced median 62.16 ms; references 61.07, 38.18, 63.34 ms.

## letters: letters-ref-b: PASS

References letters-ref-a.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.00 | 120: -0.0087 vs 0.0091 ± 0.0018, tol 0.0050; 180: -0.0095 vs 0.0096 ± 0.0004, tol 0.0050; 240: -0.0107 vs 0.0112 ± 0.0005, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.07 | 120: -0.0087 vs 0.0084 ± 0.0018, tol 0.0054; 180: -0.0095 vs 0.0094 ± 0.0004, tol 0.0050; 240: -0.0107 vs 0.0110 ± 0.0005, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.07 | 120: -0.0087 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0095 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0107 vs -0.0110 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.23 | 60: 0.0386 vs 0.0386 ± 0.0684, tol 1.0000 (restored 0.2699 ± 0.0674, tol 1.0000); 90: 0.0970 vs 0.0970 ± 0.1975, tol 1.0000 (restored 0.4489 ± 0.1974, tol 1.0000); 120: 0.5283 vs 0.5283 ± 0.8755, tol 2.6264 (restored 0.6286 ± 0.8740, tol 2.6221); 180: 0.6400 vs 0.6400 ± 0.9551, tol 2.8654 (restored 0.7468 ± 0.9551, tol 2.8654); 240: 0.9150 vs 0.9150 ± 1.3391, tol 4.0172 (restored 1.0122 ± 1.3389, tol 4.0166) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 60: 0.0386 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.0970 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.5283 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.6400 vs 0.0000 ± 0.9551, tol 2.8654; 240: 0.9150 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 1936.5950 vs 1947.4070 ± 8.7137, tol 194.7407; 120: 2201.6868 vs 2294.8335 ± 193.9061, tol 581.7183; 180: 3013.9521 vs 3091.6284 ± 156.1624, tol 468.4873; 240: 3095.4800 vs 3038.5008 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.14 | 60: 19.8923 vs 19.8953 ± 0.0109, tol 0.2500 (restored 20.0185 ± 0.0056, tol 0.2500); 120: 20.5003 vs 20.5970 ± 0.2329, tol 0.6988 (restored 20.7504 ± 0.2077, tol 0.6231); 180: 21.6976 vs 21.7141 ± 0.0536, tol 0.2500 (restored 21.8873 ± 0.0525, tol 0.2500); 240: 20.4844 vs 20.4882 ± 0.0250, tol 0.2500 (restored 20.7055 ± 0.0165, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.14 | 60: 19.8923 vs 19.8953 ± 0.0109, tol 0.2500; 120: 20.5003 vs 20.5970 ± 0.2329, tol 0.6988; 180: 21.6976 vs 21.7141 ± 0.0536, tol 0.2500; 240: 20.4844 vs 20.4882 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 4.968e+5 vs 5.011e+5 ± 7061.0000, tol 50112.6750; 120: 8.861e+5 vs 8.950e+5 ± 22398.0000, tol 89498.4250; 180: 8.987e+5 vs 9.097e+5 ± 17228.0000, tol 90965.7750; 240: 8.841e+5 vs 8.697e+5 ± 6569.0000, tol 86970.7250 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.07 | 120: -0.0087 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0095 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0107 vs -0.0110 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 122.69 ms; references 124.54, 127.79, 84.53, 84.70 ms.

## paddleboats: paddleboats-ref-b: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.00 | 150: -0.0029 vs 0.0031 ± 0.0002, tol 0.0025; 300: -0.0099 vs 0.0102 ± 0.0007, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.04 | 150: -0.0029 vs 0.0030 ± 0.0002, tol 0.0025; 300: -0.0099 vs 0.0098 ± 0.0007, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.04 | 150: -0.0029 vs -0.0030 ± 0.0002, tol 0.0025; 300: -0.0099 vs -0.0098 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.13 | 150: 0.0122 vs 0.0122 ± 0.0215, tol 0.2500 (restored 0.0717 ± 0.0209, tol 0.2500); 300: 0.0336 vs 0.0336 ± 0.0386, tol 0.2500 (restored 0.2403 ± 0.0360, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.13 | 150: 0.0122 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.0336 vs 0.0000 ± 0.0386, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.19 | 60: 96.5117 vs 94.7570 ± 2.8508, tol 9.4757; 120: 109.6019 vs 109.5410 ± 2.0263, tol 10.9541; 180: 142.1813 vs 142.2319 ± 1.2921, tol 14.2232; 240: 143.9265 vs 141.4151 ± 3.1989, tol 14.1415; 300: 166.7293 vs 167.9341 ± 3.3795, tol 16.7934 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.46 | 150: 2.602e+5 vs 2.637e+5 ± 11716.0000, tol 35148.0000; 300: 2.823e+5 vs 3.005e+5 ± 13257.0000, tol 39771.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.03 | 150: -0.0083 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0152 vs -0.0151 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 88.86 ms; references 111.75, 94.75, 97.52, 50.48 ms.
