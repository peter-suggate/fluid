# NB-FLIP port gates: dam-ref-a, letters-ref-a, paddleboats-ref-a, seep-ref-a

Generated 2026-11-15T00:35:56.961Z by tools/nbflip-gates.ts. References: seep seep-ref-b + seep-ref-c + seep-ref-d + seep-base-e; dam dam-ref-b + dam-ref-d + dam-base-e; letters letters-ref-b + letters-ref-c + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-base-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-a: PASS

References seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.01 | mean: 0.2696 vs 0.2698 ± 0.0002, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.04 | 120-180: 0.2690 vs 0.2694 ± 0.0025, tol 0.0269; 180-240: 0.2706 vs 0.2718 ± 0.0022, tol 0.0272; 240-300: 0.2712 vs 0.2706 ± 0.0018, tol 0.0271; 300-360: 0.2736 vs 0.2725 ± 0.0017, tol 0.0273; 360-420: 0.2716 vs 0.2715 ± 0.0024, tol 0.0272; 420-480: 0.2725 vs 0.2723 ± 0.0006, tol 0.0272; 480-540: 0.2706 vs 0.2711 ± 0.0006, tol 0.0271; 540-600: 0.2717 vs 0.2712 ± 0.0015, tol 0.0271; 600-660: 0.2704 vs 0.2704 ± 0.0008, tol 0.0270; 660-720: 0.2700 vs 0.2699 ± 0.0013, tol 0.0270; 720-780: 0.2699 vs 0.2695 ± 0.0011, tol 0.0269; 780-840: 0.2684 vs 0.2688 ± 0.0013, tol 0.0269; 840-900: 0.2680 vs 0.2689 ± 0.0015, tol 0.0269; 900-960: 0.2680 vs 0.2681 ± 0.0005, tol 0.0268; 960-1020: 0.2683 vs 0.2685 ± 0.0010, tol 0.0268; 1020-1080: 0.2672 vs 0.2677 ± 0.0010, tol 0.0268; 1080-1140: 0.2664 vs 0.2672 ± 0.0016, tol 0.0267; 1140-1200: 0.2663 vs 0.2667 ± 0.0022, tol 0.0267 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.10 | 300: 1.3157 vs 1.3222 ± 0.0161, tol 0.0661; 600: 2.6756 vs 2.6808 ± 0.0151, tol 0.1340; 900: 4.0223 vs 4.0284 ± 0.0145, tol 0.2014; 1200: 5.3585 vs 5.3665 ± 0.0141, tol 0.2683 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.81 | 300: 1.1376 vs 1.1607 ± 0.0701, tol 0.2103 (restored 1.3222 ± 0.0161, tol 0.0661); 600: 2.0712 vs 2.1195 ± 0.0848, tol 0.2543 (restored 2.6808 ± 0.0150, tol 0.1340); 900: 2.8114 vs 2.8890 ± 0.0630, tol 0.1890 (restored 4.0284 ± 0.0146, tol 0.2014); 1200: 3.2684 vs 3.4071 ± 0.0438, tol 0.1704 (restored 5.3666 ± 0.0141, tol 0.2683) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.81 | 300: 1.1376 vs 1.1607 ± 0.0701, tol 0.2103; 600: 2.0712 vs 2.1195 ± 0.0848, tol 0.2543; 900: 2.8114 vs 2.8890 ± 0.0630, tol 0.1890; 1200: 3.2684 vs 3.4071 ± 0.0438, tol 0.1704 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.00 | median: 2.6378 vs 2.6389 ± 0.0043, tol 0.2639 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.04 | 90%: 293.9842 vs 291.3131 ± 3.9247, tol 72.8283 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.08 | 1200: 16303.0000 vs 16364.7500 ± 17.0000, tol 818.2375 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.31 | 600: 368.6943 vs 367.3763 ± 2.1755, tol 16.0000 (restored 370.8568 ± 1.6793, tol 16.0000); 1200: 364.6602 vs 369.6230 ± 2.7236, tol 16.0000 (restored 374.8720 ± 1.7216, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.31 | 600: 368.6943 vs 367.3763 ± 2.1755, tol 16.0000; 1200: 364.6602 vs 369.6230 ± 2.7236, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.35 | 600: 0.0986 vs 0.0986 ± 0.1159, tol 0.3476 (restored 0.0939 ± 0.0876, tol 0.2627); 1200: 0.0830 vs 0.0830 ± 0.0800, tol 0.2400 (restored 0.1597 ± 0.0507, tol 0.1520) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.35 | 600: 0.0986 vs 0.0000 ± 0.1159, tol 0.3476; 1200: 0.0830 vs 0.0000 ± 0.0800, tol 0.2400 |
| G7 | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | pass | 0.36 | 1200: -0.0054 vs 0.0050 ± 0.0001, tol 0.0010 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.36 | 1200: -0.0054 vs -0.0050 ± 0.0001, tol 0.0010 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.10 | 300: 382.2329 vs 381.1622 ± 3.4907, tol 10.4722; 600: 380.6617 vs 380.8933 ± 3.5712, tol 10.7137 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.22 | 300: 0.8166 vs 0.0000 ± 1.3113, tol 3.9339; 600: 0.6825 vs 0.0000 ± 1.0212, tol 3.0635 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.50 | 300: 66918.0000 vs 69591.2500 ± 3001.0000, tol 9003.0000; 600: 1.298e+5 vs 1.299e+5 ± 8388.0000, tol 25164.0000; 900: 1.797e+5 vs 1.840e+5 ± 2973.0000, tol 18398.8750; 1200: 2.088e+5 vs 2.197e+5 ± 5971.0000, tol 21970.0750 |

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

Check-slot step time (informational): candidate fenced median 37.24 ms; references 37.57, 35.16, 65.34, 34.40 ms.

## dam: dam-ref-a: PASS

References dam-ref-b.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.25 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 107.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8222 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1674 ± 0.0007, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4869 ± 0.0081, tol 4.0000); 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.7272 ± 0.0231, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.75 | 16: -0.0500 vs 0.0425 ± 0.0005, tol 0.0100 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.75 | 16: -0.0500 vs -0.0425 ± 0.0005, tol 0.0100 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.08 | 4: 36.0847 vs 36.0848 ± 0.0002, tol 3.6085; 8: 119.5149 vs 119.4980 ± 0.0046, tol 11.9498; 12: 228.6489 vs 228.7014 ± 0.0950, tol 22.8701; 16: 272.7180 vs 274.8203 ± 0.2027, tol 27.4820 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.34 | 4: 669.1649 vs 669.1682 ± 0.0000, tol 13.3834 (restored 669.9803 ± 0.0009, tol 13.3996); 8: 670.8061 vs 670.7892 ± 0.0055, tol 13.4158 (restored 671.9681 ± 0.0058, tol 13.4394); 12: 664.0312 vs 664.1206 ± 0.1101, tol 13.2824 (restored 667.4806 ± 0.0193, tol 13.3496); 16: 586.9605 vs 591.0246 ± 0.2934, tol 11.8205 (restored 612.8093 ± 0.2143, tol 12.2562) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.34 | 4: 669.1649 vs 669.1682 ± 0.0000, tol 13.3834; 8: 670.8061 vs 670.7892 ± 0.0055, tol 13.4158; 12: 664.0312 vs 664.1206 ± 0.1101, tol 13.2824; 16: 586.9605 vs 591.0246 ± 0.2934, tol 11.8205 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7624.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10759.0000 vs 10767.0000 ± 2.0000, tol 538.3500; 12: 14936.0000 vs 14938.3333 ± 5.0000, tol 746.9167 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.23 | 8: 0.1441 vs 0.1441 ± 0.0156, tol 4.0000 (restored 0.1854 ± 0.0157, tol 4.0000); 12: 0.1439 vs 0.1439 ± 0.4137, tol 4.0000 (restored 0.3091 ± 0.4134, tol 4.0000); 16: 0.9319 vs 0.9319 ± 1.3161, tol 4.0000 (restored 2.5691 ± 1.3093, tol 4.0000) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 8: 0.1441 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.1439 vs 0.0000 ± 0.4137, tol 4.0000; 16: 0.9319 vs 0.0000 ± 1.3161, tol 4.0000 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.21 | 16: 3.454e+5 vs 3.528e+5 ± 4689.0000, tol 35275.9333 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.00 | 30-90: 45.1313 vs 44.6846 ± 0.7806, tol 2.3418 (restored 46.4543 ± 1.0349, tol 3.1046); 105-165: 46.9272 vs 46.6981 ± 1.3523, tol 4.0569 (restored 47.2101 ± 1.2912, tol 3.8736); 180-300: 46.7826 vs 46.6086 ± 1.1031, tol 3.3093 (restored 47.0742 ± 1.0824, tol 3.2473) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.19 | 30-90: 45.1313 vs 44.6846 ± 0.7806, tol 2.3418; 105-165: 46.9272 vs 46.6981 ± 1.3523, tol 4.0569; 180-300: 46.7826 vs 46.6086 ± 1.1031, tol 3.3093 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.21 | 30-90: -0.0420 vs 0.0385 ± 0.0055, tol 0.0166; 105-165: -0.0111 vs 0.0111 ± 0.0014, tol 0.0100; 180-300: -0.0095 vs 0.0101 ± 0.0023, tol 0.0100 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.21 | 30-90: -0.0420 vs -0.0385 ± 0.0055, tol 0.0166; 105-165: -0.0111 vs -0.0111 ± 0.0014, tol 0.0100; 180-300: -0.0095 vs -0.0101 ± 0.0023, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.23 | 30-90: 47.1725 vs 45.8911 ± 1.8779, tol 5.6336; 105-165: 15.8110 vs 14.7786 ± 2.6635, tol 7.9905; 180-300: 9.8557 vs 8.5450 ± 3.4880, tol 10.4641 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.09 | 30-90: 280.3071 vs 280.9328 ± 2.3033, tol 6.9099 (restored 300.3337 ± 1.7291, tol 6.0067); 105-165: 266.4853 vs 265.5957 ± 4.1727, tol 12.5182 (restored 271.3247 ± 3.4912, tol 10.4736); 180-300: 263.6018 vs 262.3448 ± 5.3404, tol 16.0212 (restored 267.5589 ± 4.1540, tol 12.4619) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.09 | 30-90: 280.3071 vs 280.9328 ± 2.3033, tol 6.9099; 105-165: 266.4853 vs 265.5957 ± 4.1727, tol 12.5182; 180-300: 263.6018 vs 262.3448 ± 5.3404, tol 16.0212 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 30-90: 327.4795 vs 326.8239 ± 2.6682, tol 8.0046 (restored 346.2248 ± 3.6069, tol 10.8208); 105-165: 282.2963 vs 280.3743 ± 6.8362, tol 20.5087 (restored 286.1033 ± 6.1547, tol 18.4641); 180-300: 273.4575 vs 270.8898 ± 8.8284, tol 26.4853 (restored 276.1038 ± 7.6420, tol 22.9260) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.10 | 30-90: 327.4795 vs 326.8239 ± 2.6682, tol 8.0046; 105-165: 282.2963 vs 280.3743 ± 6.8362, tol 20.5087; 180-300: 273.4575 vs 270.8898 ± 8.8284, tol 26.4853 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.09 | 30-90: 0.0043 vs 0.0043 ± 0.0061, tol 0.0500 (restored 0.0058 ± 0.0060, tol 0.0500); 105-165: 0.0029 vs 0.0029 ± 0.0075, tol 0.0500 (restored 0.0028 ± 0.0074, tol 0.0500); 180-300: 0.0023 vs 0.0023 ± 0.0072, tol 0.0500 (restored 0.0023 ± 0.0072, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.09 | 30-90: 0.0043 vs 0.0000 ± 0.0061, tol 0.0500; 105-165: 0.0029 vs 0.0000 ± 0.0075, tol 0.0500; 180-300: 0.0023 vs 0.0000 ± 0.0072, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.12 | 30-90: 6.349e+5 vs 6.193e+5 ± 41605.2000, tol 1.248e+5; 105-165: 3.817e+5 vs 3.830e+5 ± 30040.4000, tol 90121.2000; 180-300: 3.536e+5 vs 3.444e+5 ± 32964.2222, tol 98892.6667 |

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

Check-slot step time (informational): candidate fenced median 61.07 ms; references 62.16, 38.18, 63.34 ms.

## letters: letters-ref-a: PASS

References letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.07 | 120: -0.0086 vs 0.0084 ± 0.0018, tol 0.0054; 180: -0.0094 vs 0.0094 ± 0.0004, tol 0.0050; 240: -0.0112 vs 0.0109 ± 0.0005, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.07 | 120: -0.0086 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0094 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0112 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.24 | 60: 0.0305 vs 0.0305 ± 0.0684, tol 1.0000 (restored 0.2614 ± 0.0674, tol 1.0000); 90: 0.0790 vs 0.0790 ± 0.1975, tol 1.0000 (restored 0.4553 ± 0.1974, tol 1.0000); 120: 0.5141 vs 0.5141 ± 0.8622, tol 2.5865 (restored 0.6133 ± 0.8603, tol 2.5809); 180: 0.6786 vs 0.6786 ± 0.9908, tol 2.9724 (restored 0.7775 ± 0.9908, tol 2.9724); 240: 0.9112 vs 0.9112 ± 1.2805, tol 3.8416 (restored 1.0179 ± 1.2805, tol 3.8416) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.24 | 60: 0.0305 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.0790 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.5141 vs 0.0000 ± 0.8622, tol 2.5865; 180: 0.6786 vs 0.0000 ± 0.9908, tol 2.9724; 240: 0.9112 vs 0.0000 ± 1.2805, tol 3.8416 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.16 | 60: 1941.7583 vs 1946.1162 ± 13.8771, tol 194.6116; 120: 2204.8418 vs 2294.0448 ± 197.0611, tol 591.1833; 180: 3027.5724 vs 3088.2233 ± 169.7827, tol 509.3480; 240: 3108.8035 vs 3035.1700 ± 154.3719, tol 463.1157 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.14 | 60: 19.8933 vs 19.8950 ± 0.0109, tol 0.2500 (restored 20.0192 ± 0.0055, tol 0.2500); 120: 20.4980 vs 20.5976 ± 0.2306, tol 0.6918 (restored 20.7518 ± 0.2024, tol 0.6071); 180: 21.7408 vs 21.7033 ± 0.0410, tol 0.2500 (restored 21.8772 ± 0.0429, tol 0.2500); 240: 20.4888 vs 20.4871 ± 0.0250, tol 0.2500 (restored 20.7019 ± 0.0181, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.15 | 60: 19.8933 vs 19.8950 ± 0.0109, tol 0.2500; 120: 20.4980 vs 20.5976 ± 0.2306, tol 0.6918; 180: 21.7408 vs 21.7033 ± 0.0410, tol 0.2500; 240: 20.4888 vs 20.4871 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.15 | 60: 5.054e+5 vs 4.990e+5 ± 3980.0000, tol 49898.7250; 120: 8.826e+5 vs 8.958e+5 ± 18943.0000, tol 89584.8000; 180: 9.099e+5 vs 9.069e+5 ± 19212.0000, tol 90686.8500; 240: 8.731e+5 vs 8.725e+5 ± 17601.0000, tol 87246.5250 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.07 | 120: -0.0086 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0094 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0112 vs -0.0108 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 124.54 ms; references 122.69, 127.79, 84.53, 84.70 ms.

## paddleboats: paddleboats-ref-a: PASS

References paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.00 | 150: -0.0029 vs 0.0030 ± 0.0001, tol 0.0025; 300: -0.0097 vs 0.0099 ± 0.0007, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.08 | 150: -0.0029 vs -0.0030 ± 0.0001, tol 0.0025; 300: -0.0097 vs -0.0099 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.11 | 150: 0.0152 vs 0.0152 ± 0.0179, tol 0.2500 (restored 0.0703 ± 0.0177, tol 0.2500); 300: 0.0288 vs 0.0288 ± 0.0449, tol 0.2500 (restored 0.2340 ± 0.0435, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 150: 0.0152 vs 0.0000 ± 0.0179, tol 0.2500; 300: 0.0288 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.12 | 60: 94.9276 vs 95.1530 ± 3.5559, tol 10.6677; 120: 110.6116 vs 109.2886 ± 1.1494, tol 10.9289; 180: 141.4736 vs 142.4088 ± 0.7489, tol 14.2409; 240: 143.0915 vs 141.6239 ± 4.0339, tol 14.1624; 300: 168.4788 vs 167.4967 ± 3.3795, tol 16.7497 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.07 | 150: 2.626e+5 vs 2.631e+5 ± 11716.0000, tol 35148.0000; 300: 2.930e+5 vs 2.979e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.08 | 150: -0.0082 vs -0.0083 ± 0.0001, tol 0.0025; 300: -0.0150 vs -0.0152 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 111.75 ms; references 88.86, 94.75, 97.52, 50.48 ms.
