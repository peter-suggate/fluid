# NB-FLIP port gates: dam-ref-d, letters-ref-d, paddleboats-ref-d, seep-ref-d

Generated 2026-11-15T00:35:57.578Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-base-e; dam dam-ref-a + dam-ref-b + dam-base-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-base-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-d: PASS

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.01 | mean: 0.2699 vs 0.2697 ± 0.0001, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.05 | 120-180: 0.2688 vs 0.2695 ± 0.0025, tol 0.0269; 180-240: 0.2715 vs 0.2715 ± 0.0022, tol 0.0272; 240-300: 0.2715 vs 0.2706 ± 0.0014, tol 0.0271; 300-360: 0.2734 vs 0.2726 ± 0.0018, tol 0.0273; 360-420: 0.2710 vs 0.2717 ± 0.0024, tol 0.0272; 420-480: 0.2728 vs 0.2723 ± 0.0003, tol 0.0272; 480-540: 0.2714 vs 0.2709 ± 0.0005, tol 0.0271; 540-600: 0.2708 vs 0.2714 ± 0.0015, tol 0.0271; 600-660: 0.2708 vs 0.2703 ± 0.0008, tol 0.0270; 660-720: 0.2698 vs 0.2700 ± 0.0013, tol 0.0270; 720-780: 0.2698 vs 0.2695 ± 0.0012, tol 0.0269; 780-840: 0.2683 vs 0.2689 ± 0.0012, tol 0.0269; 840-900: 0.2688 vs 0.2687 ± 0.0017, tol 0.0269; 900-960: 0.2680 vs 0.2681 ± 0.0005, tol 0.0268; 960-1020: 0.2686 vs 0.2684 ± 0.0010, tol 0.0268; 1020-1080: 0.2677 vs 0.2675 ± 0.0011, tol 0.0268; 1080-1140: 0.2673 vs 0.2670 ± 0.0018, tol 0.0267; 1140-1200: 0.2676 vs 0.2664 ± 0.0017, tol 0.0266 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.01 | 300: 1.3210 vs 1.3208 ± 0.0161, tol 0.0660; 600: 2.6805 vs 2.6796 ± 0.0151, tol 0.1340; 900: 4.0280 vs 4.0269 ± 0.0145, tol 0.2013; 1200: 5.3672 vs 5.3644 ± 0.0141, tol 0.2682 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.24 | 300: 1.1238 vs 1.1642 ± 0.0563, tol 0.1690 (restored 1.3209 ± 0.0161, tol 0.0660); 600: 2.1094 vs 2.1099 ± 0.0928, tol 0.2784 (restored 2.6796 ± 0.0150, tol 0.1340); 900: 2.8760 vs 2.8728 ± 0.1074, tol 0.3223 (restored 4.0270 ± 0.0146, tol 0.2014); 1200: 3.3980 vs 3.3747 ± 0.1684, tol 0.5053 (restored 5.3644 ± 0.0141, tol 0.2682) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.24 | 300: 1.1238 vs 1.1642 ± 0.0563, tol 0.1690; 600: 2.1094 vs 2.1099 ± 0.0928, tol 0.2784; 900: 2.8760 vs 2.8728 ± 0.1074, tol 0.3223; 1200: 3.3980 vs 3.3747 ± 0.1684, tol 0.5053 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6401 vs 2.6383 ± 0.0043, tol 0.2638 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.03 | 90%: 293.7602 vs 291.3691 ± 4.1487, tol 72.8423 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.01 | 1200: 16358.0000 vs 16351.0000 ± 72.0000, tol 817.5500 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.00 | 600: 367.7267 vs 367.6182 ± 2.7158, tol 16.0000 (restored 371.0963 ± 2.4878, tol 16.0000); 1200: 368.5612 vs 368.6478 ± 6.6246, tol 19.8737 (restored 374.3665 ± 3.7435, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.01 | 600: 367.7267 vs 367.6182 ± 2.7158, tol 16.0000; 1200: 368.5612 vs 368.6478 ± 6.6246, tol 19.8737 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.22 | 600: 0.0843 vs 0.0843 ± 0.1271, tol 0.3813 (restored 0.1591 ± 0.1047, tol 0.3140); 1200: 0.0569 vs 0.0569 ± 0.0992, tol 0.2977 (restored 0.0832 ± 0.0598, tol 0.1793) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.22 | 600: 0.0843 vs 0.0000 ± 0.1271, tol 0.3813; 1200: 0.0569 vs 0.0000 ± 0.0992, tol 0.2977 |
| G7 | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | pass | 0.00 | 1200: -0.0050 vs 0.0051 ± 0.0004, tol 0.0013 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.04 | 1200: -0.0050 vs -0.0051 ± 0.0004, tol 0.0013 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.26 | 300: 380.6611 vs 381.5551 ± 3.4907, tol 10.4722; 600: 382.4998 vs 380.4338 ± 2.4462, tol 8.0000 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.28 | 300: 0.8372 vs 0.0000 ± 0.9858, tol 3.0000; 600: 0.7063 vs 0.0000 ± 1.0137, tol 3.0410 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.05 | 300: 68482.0000 vs 69200.2500 ± 4565.0000, tol 13695.0000; 600: 1.290e+5 vs 1.301e+5 ± 8388.0000, tol 25164.0000; 900: 1.834e+5 vs 1.831e+5 ± 6281.0000, tol 18843.0000; 1200: 2.168e+5 vs 2.177e+5 ± 14011.0000, tol 42033.0000 |

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

Check-slot step time (informational): candidate fenced median 65.34 ms; references 37.24, 37.57, 35.16, 34.40 ms.

## dam: dam-ref-d: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.08 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 107.6667 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8225 ± 0.0010, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1664 ± 0.0036, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4901 ± 0.0162, tol 4.0000); 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.8480 ± 0.3420, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.00 | 16: -0.0422 vs 0.0451 ± 0.0073, tol 0.0220 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.13 | 16: -0.0422 vs -0.0451 ± 0.0073, tol 0.0220 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.03 | 4: 36.0848 vs 36.0847 ± 0.0002, tol 3.6085; 8: 119.5009 vs 119.5026 ± 0.0185, tol 11.9503; 12: 228.6905 vs 228.6875 ± 0.1055, tol 22.8688; 16: 274.8708 vs 274.1027 ± 2.1784, tol 27.4103 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 4: 669.1682 vs 669.1671 ± 0.0033, tol 13.3833 (restored 669.9825 ± 0.0077, tol 13.3997); 8: 670.7926 vs 670.7937 ± 0.0191, tol 13.4159 (restored 671.9619 ± 0.0225, tol 13.4392); 12: 664.0970 vs 664.0987 ± 0.1563, tol 13.2820 (restored 667.4945 ± 0.0445, tol 13.3499); 16: 591.1587 vs 589.6252 ± 4.0894, tol 12.2681 (restored 612.7068 ± 0.4477, tol 12.2541) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.12 | 4: 669.1682 vs 669.1671 ± 0.0033, tol 13.3833; 8: 670.7926 vs 670.7937 ± 0.0191, tol 13.4159; 12: 664.0970 vs 664.0987 ± 0.1563, tol 13.2820; 16: 591.1587 vs 589.6252 ± 4.0894, tol 12.2681 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7624.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10766.0000 vs 10764.6667 ± 9.0000, tol 538.2333; 12: 14941.0000 vs 14936.6667 ± 2.0000, tol 746.8333 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.31 | 8: 0.0483 vs 0.0483 ± 0.1449, tol 4.0000 (restored 0.0995 ± 0.1653, tol 4.0000); 12: 0.1461 vs 0.1461 ± 0.4122, tol 4.0000 (restored 0.3079 ± 0.4138, tol 4.0000); 16: 1.3695 vs 1.3695 ± 0.7694, tol 4.0000 (restored 1.9708 ± 0.6980, tol 4.0000) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.34 | 8: 0.0483 vs 0.0000 ± 0.1449, tol 4.0000; 12: 0.1461 vs 0.0000 ± 0.4122, tol 4.0000; 16: 1.3695 vs 0.0000 ± 0.7694, tol 4.0000 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.04 | 16: 3.498e+5 vs 3.513e+5 ± 9091.0000, tol 35129.2000 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.11 | 30-90: 44.9321 vs 44.7510 ± 0.9607, tol 2.8820 (restored 46.6024 ± 1.2089, tol 3.6267); 105-165: 46.4094 vs 46.8707 ± 1.3523, tol 4.0569 (restored 47.3892 ± 1.2912, tol 3.8736); 180-300: 47.0681 vs 46.5134 ± 0.8276, tol 2.4828 (restored 46.9615 ± 0.7444, tol 2.3481) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.22 | 30-90: 44.9321 vs 44.7510 ± 0.9607, tol 2.8820; 105-165: 46.4094 vs 46.8707 ± 1.3523, tol 4.0569; 180-300: 47.0681 vs 46.5134 ± 0.8276, tol 2.4828 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.09 | 30-90: -0.0367 vs 0.0402 ± 0.0055, tol 0.0166; 105-165: -0.0106 vs 0.0113 ± 0.0013, tol 0.0100; 180-300: -0.0106 vs 0.0097 ± 0.0023, tol 0.0100 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.22 | 30-90: -0.0367 vs -0.0402 ± 0.0055, tol 0.0166; 105-165: -0.0106 vs -0.0113 ± 0.0013, tol 0.0100; 180-300: -0.0106 vs -0.0097 ± 0.0023, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.40 | 30-90: 44.8138 vs 46.6773 ± 1.0045, tol 4.6677; 105-165: 14.3623 vs 15.2615 ± 2.6635, tol 7.9905; 180-300: 8.2293 vs 9.0871 ± 3.4880, tol 10.4641 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.08 | 30-90: 280.8245 vs 280.7603 ± 2.3033, tol 6.9099 (restored 301.0426 ± 0.8274, tol 6.0209); 105-165: 265.0802 vs 266.0641 ± 4.1727, tol 12.5182 (restored 271.8657 ± 3.4912, tol 10.4736); 180-300: 261.7938 vs 262.9475 ± 5.3404, tol 16.0212 (restored 267.9663 ± 4.1540, tol 12.4619) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.08 | 30-90: 280.8245 vs 280.7603 ± 2.3033, tol 6.9099; 105-165: 265.0802 vs 266.0641 ± 4.1727, tol 12.5182; 180-300: 261.7938 vs 262.9475 ± 5.3404, tol 16.0212 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.27 | 30-90: 325.6383 vs 327.4376 ± 1.7796, tol 6.5488 (restored 347.7199 ± 1.8319, tol 6.9544); 105-165: 279.4424 vs 281.3256 ± 6.8362, tol 20.5087 (restored 287.1273 ± 6.1547, tol 18.4641); 180-300: 270.0231 vs 272.0346 ± 8.8284, tol 26.4853 (restored 277.0533 ± 7.6420, tol 22.9260) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.27 | 30-90: 325.6383 vs 327.4376 ± 1.7796, tol 6.5488; 105-165: 279.4424 vs 281.3256 ± 6.8362, tol 20.5087; 180-300: 270.0231 vs 272.0346 ± 8.8284, tol 26.4853 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.09 | 30-90: 0.0049 vs 0.0049 ± 0.0061, tol 0.0500 (restored 0.0049 ± 0.0060, tol 0.0500); 105-165: 0.0046 vs 0.0046 ± 0.0072, tol 0.0500 (restored 0.0046 ± 0.0072, tol 0.0500); 180-300: 0.0028 vs 0.0028 ± 0.0072, tol 0.0500 (restored 0.0029 ± 0.0072, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.10 | 30-90: 0.0049 vs 0.0000 ± 0.0061, tol 0.0500; 105-165: 0.0046 vs 0.0000 ± 0.0072, tol 0.0500; 180-300: 0.0028 vs 0.0000 ± 0.0072, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.41 | 30-90: 6.000e+5 vs 6.310e+5 ± 25134.4000, tol 75403.2000; 105-165: 3.713e+5 vs 3.865e+5 ± 24785.6000, tol 74356.8000; 180-300: 3.359e+5 vs 3.503e+5 ± 32964.2222, tol 98892.6667 |

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

Check-slot step time (informational): candidate fenced median 38.18 ms; references 61.07, 62.16, 63.34 ms.

## letters: letters-ref-d: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.15 | 120: -0.0091 vs 0.0083 ± 0.0015, tol 0.0050; 180: -0.0092 vs 0.0095 ± 0.0002, tol 0.0050; 240: -0.0109 vs 0.0109 ± 0.0005, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.15 | 120: -0.0091 vs -0.0083 ± 0.0015, tol 0.0050; 180: -0.0092 vs -0.0095 ± 0.0002, tol 0.0050; 240: -0.0109 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.19 | 60: 0.0308 vs 0.0308 ± 0.0684, tol 1.0000 (restored 0.2608 ± 0.0674, tol 1.0000); 90: 0.0677 vs 0.0677 ± 0.1975, tol 1.0000 (restored 0.4439 ± 0.1974, tol 1.0000); 120: 0.3172 vs 0.3172 ± 0.8755, tol 2.6264 (restored 0.4744 ± 0.8740, tol 2.6221); 180: 0.5715 vs 0.5715 ± 0.9908, tol 2.9724 (restored 0.6813 ± 0.9908, tol 2.9724); 240: 0.7664 vs 0.7664 ± 1.3391, tol 4.0172 (restored 0.8840 ± 1.3389, tol 4.0166) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.19 | 60: 0.0308 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.0677 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.3172 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.5715 vs 0.0000 ± 0.9908, tol 2.9724; 240: 0.7664 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.09 | 60: 1948.7823 vs 1944.3602 ± 13.8771, tol 194.4360; 120: 2264.4150 vs 2279.1515 ± 197.0611, tol 591.1833; 180: 3044.5112 vs 3083.9886 ± 169.7827, tol 509.3480; 240: 3013.3825 vs 3059.0252 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.05 | 60: 19.8943 vs 19.8948 ± 0.0109, tol 0.2500 (restored 20.0190 ± 0.0056, tol 0.2500); 120: 20.5485 vs 20.5849 ± 0.2329, tol 0.6988 (restored 20.7370 ± 0.2077, tol 0.6231); 180: 21.7002 vs 21.7134 ± 0.0536, tol 0.2500 (restored 21.8882 ± 0.0525, tol 0.2500); 240: 20.4824 vs 20.4887 ± 0.0250, tol 0.2500 (restored 20.7051 ± 0.0181, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.05 | 60: 19.8943 vs 19.8948 ± 0.0109, tol 0.2500; 120: 20.5485 vs 20.5849 ± 0.2329, tol 0.6988; 180: 21.7002 vs 21.7134 ± 0.0536, tol 0.2500; 240: 20.4824 vs 20.4887 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.09 | 60: 5.008e+5 vs 5.001e+5 ± 8558.0000, tol 50013.1750; 120: 8.925e+5 vs 8.934e+5 ± 22398.0000, tol 89337.9750; 180: 9.007e+5 vs 9.092e+5 ± 19212.0000, tol 90916.1750; 240: 8.724e+5 vs 8.726e+5 ± 17601.0000, tol 87264.0250 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.15 | 120: -0.0091 vs -0.0083 ± 0.0015, tol 0.0050; 180: -0.0092 vs -0.0095 ± 0.0002, tol 0.0050; 240: -0.0109 vs -0.0109 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 84.53 ms; references 124.54, 122.69, 127.79, 84.70 ms.

## paddleboats: paddleboats-ref-d: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.00 | 150: -0.0029 vs 0.0030 ± 0.0002, tol 0.0025; 300: -0.0094 vs 0.0099 ± 0.0005, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.20 | 150: -0.0029 vs -0.0030 ± 0.0002, tol 0.0025; 300: -0.0094 vs -0.0099 ± 0.0005, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.11 | 150: 0.0098 vs 0.0098 ± 0.0215, tol 0.2500 (restored 0.0710 ± 0.0209, tol 0.2500); 300: 0.0310 vs 0.0310 ± 0.0419, tol 0.2500 (restored 0.2284 ± 0.0415, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 150: 0.0098 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.0310 vs 0.0000 ± 0.0419, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.14 | 60: 95.3380 vs 95.0504 ± 3.5559, tol 10.6677; 120: 109.2325 vs 109.6334 ± 2.0263, tol 10.9633; 180: 142.6714 vs 142.1094 ± 1.2921, tol 14.2109; 240: 140.3185 vs 142.3171 ± 4.0339, tol 14.2317; 300: 166.4973 vs 167.9921 ± 3.1475, tol 16.7992 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.30 | 150: 2.692e+5 vs 2.614e+5 ± 7778.0000, tol 26139.8500; 300: 3.062e+5 vs 2.946e+5 ± 22306.0000, tol 66918.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.20 | 150: -0.0083 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0148 vs -0.0153 ± 0.0005, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 97.52 ms; references 111.75, 88.86, 94.75, 50.48 ms.
