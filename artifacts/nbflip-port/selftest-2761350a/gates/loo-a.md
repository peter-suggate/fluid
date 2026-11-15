# NB-FLIP port gates: dam-ref-a, letters-ref-a, paddleboats-ref-a, seep-ref-a

Generated 2026-11-15T00:46:26.652Z by tools/nbflip-gates.ts. References: seep seep-ref-b + seep-ref-c + seep-ref-d + seep-ref-e; dam dam-ref-b + dam-ref-c + dam-ref-d + dam-ref-e; letters letters-ref-b + letters-ref-c + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-ref-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-a: PASS

References seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.04 | mean: 0.2688 vs 0.2693 ± 0.0020, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.11 | 120-180: 0.2705 vs 0.2694 ± 0.0021, tol 0.0269; 180-240: 0.2715 vs 0.2711 ± 0.0038, tol 0.0271; 240-300: 0.2701 vs 0.2713 ± 0.0017, tol 0.0271; 300-360: 0.2706 vs 0.2727 ± 0.0013, tol 0.0273; 360-420: 0.2729 vs 0.2719 ± 0.0035, tol 0.0272; 420-480: 0.2720 vs 0.2715 ± 0.0032, tol 0.0272; 480-540: 0.2690 vs 0.2699 ± 0.0034, tol 0.0270; 540-600: 0.2736 vs 0.2706 ± 0.0026, tol 0.0271; 600-660: 0.2674 vs 0.2697 ± 0.0033, tol 0.0270; 660-720: 0.2685 vs 0.2692 ± 0.0036, tol 0.0269; 720-780: 0.2669 vs 0.2689 ± 0.0022, tol 0.0269; 780-840: 0.2668 vs 0.2688 ± 0.0028, tol 0.0269; 840-900: 0.2664 vs 0.2689 ± 0.0034, tol 0.0269; 900-960: 0.2695 vs 0.2673 ± 0.0030, tol 0.0267; 960-1020: 0.2652 vs 0.2673 ± 0.0041, tol 0.0267; 1020-1080: 0.2656 vs 0.2673 ± 0.0012, tol 0.0267; 1080-1140: 0.2651 vs 0.2670 ± 0.0018, tol 0.0267; 1140-1200: 0.2660 vs 0.2652 ± 0.0036, tol 0.0265 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.05 | 300: 1.3184 vs 1.3199 ± 0.0049, tol 0.0660; 600: 2.6766 vs 2.6766 ± 0.0148, tol 0.1338; 900: 4.0126 vs 4.0220 ± 0.0259, tol 0.2011; 1200: 5.3440 vs 5.3562 ± 0.0355, tol 0.2678 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.06 | 300: 1.1472 vs 1.1525 ± 0.0298, tol 0.0895 (restored 1.3199 ± 0.0050, tol 0.0660); 600: 2.1495 vs 2.1203 ± 0.0754, tol 0.2263 (restored 2.6766 ± 0.0148, tol 0.1338); 900: 3.0191 vs 2.9348 ± 0.0951, tol 0.2854 (restored 4.0221 ± 0.0259, tol 0.2011); 1200: 3.5016 vs 3.4696 ± 0.2046, tol 0.6138 (restored 5.3563 ± 0.0356, tol 0.2678) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.30 | 300: 1.1472 vs 1.1525 ± 0.0298, tol 0.0895; 600: 2.1495 vs 2.1203 ± 0.0754, tol 0.2263; 900: 3.0191 vs 2.9348 ± 0.0951, tol 0.2854; 1200: 3.5016 vs 3.4696 ± 0.2046, tol 0.6138 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.04 | median: 2.6513 vs 2.6397 ± 0.0025, tol 0.2640 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.21 | 90%: 300.2337 vs 285.2917 ± 5.1446, tol 71.3229 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.01 | 1200: 16374.0000 vs 16369.0000 ± 12.0000, tol 818.4500 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.04 | 600: 367.9810 vs 368.5951 ± 3.9858, tol 16.0000 (restored 371.7847 ± 3.4575, tol 16.0000); 1200: 370.0344 vs 370.2323 ± 4.0817, tol 16.0000 (restored 375.0835 ± 2.7714, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.04 | 600: 367.9810 vs 368.5951 ± 3.9858, tol 16.0000; 1200: 370.0344 vs 370.2323 ± 4.0817, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.17 | 600: 0.0773 vs 0.0773 ± 0.1527, tol 0.4582 (restored 0.1388 ± 0.1261, tol 0.3782); 1200: 0.0400 vs 0.0400 ± 0.0756, tol 0.2268 (restored 0.0710 ± 0.0500, tol 0.1501) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.18 | 600: 0.0773 vs 0.0000 ± 0.1527, tol 0.4582; 1200: 0.0400 vs 0.0000 ± 0.0756, tol 0.2268 |
| G7 | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | pass | 0.00 | 1200: -0.0047 vs 0.0048 ± 0.0006, tol 0.0019 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.06 | 1200: -0.0047 vs -0.0048 ± 0.0006, tol 0.0019 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.13 | 300: 381.1493 vs 380.1015 ± 1.6063, tol 8.0000; 600: 380.6066 vs 380.7385 ± 4.3196, tol 12.9587 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.28 | 300: 0.9876 vs 0.0000 ± 1.3160, tol 3.9481; 600: 0.9632 vs 0.0000 ± 1.1591, tol 3.4774 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.25 | 300: 69614.0000 vs 69089.2500 ± 2891.0000, tol 8673.0000; 600: 1.320e+5 vs 1.297e+5 ± 6052.0000, tol 18156.0000; 900: 1.874e+5 vs 1.861e+5 ± 9143.0000, tol 27429.0000; 1200: 2.268e+5 vs 2.205e+5 ± 8470.0000, tol 25410.0000 |

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

Check-slot step time (informational): candidate fenced median 54.13 ms; references 68.99, 49.39, 53.44, 74.01 ms.

## dam: dam-ref-a: PASS

References dam-ref-b.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.2500 ± 1.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1679 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5044 ± 0.0867, tol 4.0000); 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 (restored 80.8216 ± 0.3886, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.00 | 16: -0.0423 vs 0.0441 ± 0.0068, tol 0.0204 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.09 | 16: -0.0423 vs -0.0441 ± 0.0068, tol 0.0204 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.01 | 4: 36.0847 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.5031 vs 119.4959 ± 0.0233, tol 11.9496; 12: 228.6810 vs 228.5857 ± 0.4230, tol 22.8586; 16: 274.8309 vs 275.0517 ± 1.1711, tol 27.5052 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 4: 669.1682 vs 669.1685 ± 0.0007, tol 13.3834 (restored 669.9802 ± 0.0009, tol 13.3996); 8: 670.7966 vs 670.7824 ± 0.0402, tol 13.4156 (restored 671.9665 ± 0.0154, tol 13.4393); 12: 664.1240 vs 663.9555 ± 0.7382, tol 13.2791 (restored 667.5114 ± 0.2311, tol 13.3502); 16: 591.0929 vs 590.9254 ± 0.9253, tol 11.8185 (restored 613.5565 ± 3.2444, tol 12.2711) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.01 | 4: 669.1682 vs 669.1685 ± 0.0007, tol 13.3834; 8: 670.7966 vs 670.7824 ± 0.0402, tol 13.4156; 12: 664.1240 vs 663.9555 ± 0.7382, tol 13.2791; 16: 591.0929 vs 590.9254 ± 0.9253, tol 11.8185 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7623.0000 vs 7623.7500 ± 2.0000, tol 381.1875; 8: 10765.0000 vs 10768.0000 ± 3.0000, tol 538.4000; 12: 14936.0000 vs 14936.2500 ± 10.0000, tol 746.8125 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.12 | 8: 0.0078 vs 0.0078 ± 0.0156, tol 4.0000 (restored 0.0974 ± 0.0158, tol 4.0000); 12: 0.1070 vs 0.1070 ± 0.4381, tol 4.0000 (restored 0.3155 ± 0.4252, tol 4.0000); 16: 0.8144 vs 0.8144 ± 2.2985, tol 6.8954 (restored 2.0346 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 8: 0.0078 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.1070 vs 0.0000 ± 0.4381, tol 4.0000; 16: 0.8144 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.03 | 16: 3.507e+5 vs 3.518e+5 ± 5756.0000, tol 35179.8500 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.05 | 30-90: 45.8819 vs 45.1554 ± 0.4729, tol 2.2578 (restored 47.0150 ± 0.4559, tol 2.3508); 105-165: 47.0064 vs 47.1142 ± 0.7294, tol 2.3557 (restored 47.5385 ± 0.6676, tol 2.3769); 180-300: 46.8605 vs 46.8249 ± 0.5217, tol 2.3412 (restored 47.1789 ± 0.9449, tol 2.8346) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.32 | 30-90: 45.8819 vs 45.1554 ± 0.4729, tol 2.2578; 105-165: 47.0064 vs 47.1142 ± 0.7294, tol 2.3557; 180-300: 46.8605 vs 46.8249 ± 0.5217, tol 2.3412 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.13 | 30-90: -0.0402 vs 0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0119 vs 0.0090 ± 0.0077, tol 0.0232; 180-300: -0.0092 vs 0.0068 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.13 | 30-90: -0.0402 vs -0.0404 ± 0.0090, tol 0.0270; 105-165: -0.0119 vs -0.0090 ± 0.0077, tol 0.0232; 180-300: -0.0092 vs -0.0068 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.08 | 30-90: 46.4753 vs 46.5371 ± 1.2119, tol 4.6537; 105-165: 17.4342 vs 16.4509 ± 4.3305, tol 12.9914; 180-300: 10.6642 vs 10.3220 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.01 | 30-90: 282.1299 vs 281.2547 ± 4.4590, tol 13.3770 (restored 301.6119 ± 0.8424, tol 6.0322); 105-165: 268.0190 vs 268.2622 ± 7.5629, tol 22.6888 (restored 273.0122 ± 4.5779, tol 13.7338); 180-300: 265.1939 vs 265.5879 ± 10.4288, tol 31.2864 (restored 269.5517 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.07 | 30-90: 282.1299 vs 281.2547 ± 4.4590, tol 13.3770; 105-165: 268.0190 vs 268.2622 ± 7.5629, tol 22.6888; 180-300: 265.1939 vs 265.5879 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 30-90: 328.6052 vs 327.7917 ± 3.8834, tol 11.6502 (restored 348.1490 ± 2.0543, tol 6.9630); 105-165: 285.4533 vs 284.7131 ± 11.8934, tol 35.6801 (restored 289.4630 ± 8.9084, tol 26.7252); 180-300: 275.8581 vs 275.9099 ± 14.4039, tol 43.2116 (restored 279.8737 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.07 | 30-90: 328.6052 vs 327.7917 ± 3.8834, tol 11.6502; 105-165: 285.4533 vs 284.7131 ± 11.8934, tol 35.6801; 180-300: 275.8581 vs 275.9099 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.09 | 30-90: 0.0041 vs 0.0041 ± 0.0056, tol 0.0500 (restored 0.0046 ± 0.0054, tol 0.0500); 105-165: 0.0047 vs 0.0047 ± 0.0106, tol 0.0500 (restored 0.0049 ± 0.0105, tol 0.0500); 180-300: 0.0029 vs 0.0029 ± 0.0094, tol 0.0500 (restored 0.0029 ± 0.0093, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.09 | 30-90: 0.0041 vs 0.0000 ± 0.0056, tol 0.0500; 105-165: 0.0047 vs 0.0000 ± 0.0106, tol 0.0500; 180-300: 0.0029 vs 0.0000 ± 0.0094, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.35 | 30-90: 6.230e+5 vs 6.328e+5 ± 17318.6000, tol 63282.8550; 105-165: 3.973e+5 vs 3.804e+5 ± 15932.8000, tol 47798.4000; 180-300: 3.429e+5 vs 3.498e+5 ± 30193.3333, tol 90580.0000 |

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

Check-slot step time (informational): candidate fenced median 67.60 ms; references 45.86, 62.58, 66.20, 60.19 ms.

## letters: letters-ref-a: PASS

References letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.06 | 120: -0.0087 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0094 vs 0.0091 ± 0.0014, tol 0.0050; 240: -0.0109 vs 0.0107 ± 0.0009, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.06 | 120: -0.0087 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0094 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0109 vs -0.0107 ± 0.0009, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.28 | 60: 0.0300 vs 0.0300 ± 0.0299, tol 1.0000 (restored 0.2635 ± 0.0299, tol 1.0000); 90: 0.0799 vs 0.0799 ± 0.1444, tol 1.0000 (restored 0.4527 ± 0.1444, tol 1.0000); 120: 0.2993 vs 0.2993 ± 0.6859, tol 2.0576 (restored 0.4542 ± 0.6858, tol 2.0574); 180: 0.6727 vs 0.6727 ± 0.7942, tol 2.3826 (restored 0.7716 ± 0.7942, tol 2.3825); 240: 0.8334 vs 0.8334 ± 1.1402, tol 3.4206 (restored 0.9427 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.28 | 60: 0.0300 vs 0.0000 ± 0.0299, tol 1.0000; 90: 0.0799 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.2993 vs 0.0000 ± 0.6859, tol 2.0576; 180: 0.6727 vs 0.0000 ± 0.7942, tol 2.3826; 240: 0.8334 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 1948.8203 vs 1948.5599 ± 2.2087, tol 194.8560; 120: 2202.6392 vs 2248.7941 ± 89.7918, tol 269.3753; 180: 3114.2462 vs 3038.4138 ± 219.4018, tol 658.2053; 240: 3055.9225 vs 3038.4945 ± 41.3393, tol 303.8494 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.10 | 60: 19.8921 vs 19.8926 ± 0.0023, tol 0.2500 (restored 20.0176 ± 0.0015, tol 0.2500); 120: 20.5136 vs 20.5566 ± 0.1422, tol 0.4266 (restored 20.7176 ± 0.1374, tol 0.4121); 180: 21.7469 vs 21.7229 ± 0.0381, tol 0.2500 (restored 21.8901 ± 0.0156, tol 0.2500); 240: 20.4952 vs 20.5013 ± 0.0231, tol 0.2500 (restored 20.7121 ± 0.0054, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.10 | 60: 19.8921 vs 19.8926 ± 0.0023, tol 0.2500; 120: 20.5136 vs 20.5566 ± 0.1422, tol 0.4266; 180: 21.7469 vs 21.7229 ± 0.0381, tol 0.2500; 240: 20.4952 vs 20.5013 ± 0.0231, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.16 | 60: 4.976e+5 vs 5.015e+5 ± 4583.0000, tol 50148.8750; 120: 8.858e+5 vs 8.921e+5 ± 9751.0000, tol 89208.5000; 180: 9.137e+5 vs 8.991e+5 ± 11546.0000, tol 89905.0750; 240: 8.803e+5 vs 8.773e+5 ± 7187.0000, tol 87726.7500 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.06 | 120: -0.0087 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0094 vs -0.0091 ± 0.0014, tol 0.0050; 240: -0.0109 vs -0.0106 ± 0.0009, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 123.19 ms; references 96.65, 84.54, 87.50, 94.06 ms.

## paddleboats: paddleboats-ref-a: PASS

References paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.08 | 150: -0.0030 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0105 vs 0.0101 ± 0.0018, tol 0.0053 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.08 | 150: -0.0030 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0105 vs -0.0101 ± 0.0018, tol 0.0053 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.12 | 150: 0.0102 vs 0.0102 ± 0.0176, tol 0.2500 (restored 0.0732 ± 0.0164, tol 0.2500); 300: 0.0297 vs 0.0297 ± 0.0590, tol 0.2500 (restored 0.2543 ± 0.0414, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 150: 0.0102 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.0297 vs 0.0000 ± 0.0590, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.11 | 60: 95.3509 vs 95.1968 ± 1.7659, tol 9.5197; 120: 108.3725 vs 109.2852 ± 0.5143, tol 10.9285; 180: 140.8615 vs 142.0464 ± 0.8459, tol 14.2046; 240: 140.7046 vs 142.2868 ± 0.6956, tol 14.2287; 300: 168.0221 vs 167.0632 ± 2.4230, tol 16.7063 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.07 | 150: 2.524e+5 vs 2.581e+5 ± 26447.0000, tol 79341.0000; 300: 3.006e+5 vs 2.887e+5 ± 79574.0000, tol 2.387e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.08 | 150: -0.0083 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0158 vs -0.0154 ± 0.0017, tol 0.0052 |

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

Check-slot step time (informational): candidate fenced median 51.19 ms; references 70.16, 50.72, 54.85, 95.74 ms.
