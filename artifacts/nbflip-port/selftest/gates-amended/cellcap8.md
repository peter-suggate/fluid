# NB-FLIP port gates: dam-cellcap8, letters-cellcap8, paddleboats-cellcap8

Generated 2026-11-15T01:40:29.257Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-base-e; dam dam-ref-a + dam-ref-b + dam-ref-d + dam-base-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-base-e.
**Overall: PASS** (dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## dam: dam-cellcap8: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 107.7500 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8224 ± 0.0010, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1666 ± 0.0036, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4897 ± 0.0162, tol 4.0000); 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.8142 ± 0.3623, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0421 vs 0.0500 ± 0.0078, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0421 vs 0.0443 ± 0.0078, tol 0.0234 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.10 | 16: -0.0421 vs -0.0443 ± 0.0078, tol 0.0234 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0851 vs 36.0848 ± 0.0002, tol 3.6085; 8: 119.5085 vs 119.5022 ± 0.0185, tol 11.9502; 12: 228.5517 vs 228.6883 ± 0.1055, tol 22.8688; 16: 274.8171 vs 274.2947 ± 2.1784, tol 27.4295 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.01 | 4: 669.1637 vs 669.1674 ± 0.0033, tol 13.3833 (restored 669.9821 ± 0.0077, tol 13.3996); 8: 670.8018 vs 670.7934 ± 0.0191, tol 13.4159 (restored 671.9633 ± 0.0225, tol 13.4393); 12: 663.9477 vs 664.0983 ± 0.1563, tol 13.2820 (restored 667.4899 ± 0.0445, tol 13.3498); 16: 591.1671 vs 590.0086 ± 4.1982, tol 12.5946 (restored 612.7269 ± 0.4477, tol 12.2545) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.09 | 4: 669.1637 vs 669.1674 ± 0.0033, tol 13.3833; 8: 670.8018 vs 670.7934 ± 0.0191, tol 13.4159; 12: 663.9477 vs 664.0983 ± 0.1563, tol 13.2820; 16: 591.1671 vs 590.0086 ± 4.1982, tol 12.5946 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7626.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10766.0000 vs 10765.0000 ± 9.0000, tol 538.2500; 12: 14933.0000 vs 14937.7500 ± 5.0000, tol 746.8875 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.04 | 8: 0.0379 vs 0.0379 ± 0.1449, tol 4.0000 (restored 0.1001 ± 0.1653, tol 4.0000); 12: 0.1252 vs 0.1252 ± 0.4137, tol 4.0000 (restored 0.3018 ± 0.4138, tol 4.0000); 16: 0.2034 vs 0.2034 ± 1.6789, tol 5.0368 (restored 2.1298 ± 1.5458, tol 4.6375) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.04 | 8: 0.0379 vs 0.0000 ± 0.1449, tol 4.0000; 12: 0.1252 vs 0.0000 ± 0.4137, tol 4.0000; 16: 0.2034 vs 0.0000 ± 1.6789, tol 5.0368 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.06 | 16: 3.488e+5 vs 3.509e+5 ± 9091.0000, tol 35092.0750 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.07 | 30-90: 45.0407 vs 44.7963 ± 0.9607, tol 2.8820 (restored 46.6066 ± 1.2089, tol 3.6267); 105-165: 46.9714 vs 46.7554 ± 1.3523, tol 4.0569 (restored 47.2666 ± 1.2912, tol 3.8736); 180-300: 46.4245 vs 46.6521 ± 1.1031, tol 3.3093 (restored 47.1100 ± 1.0824, tol 3.2473) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.08 | 30-90: 45.0407 vs 44.7963 ± 0.9607, tol 2.8820; 105-165: 46.9714 vs 46.7554 ± 1.3523, tol 4.0569; 180-300: 46.4245 vs 46.6521 ± 1.1031, tol 3.3093 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 30-90: -0.0396 vs 0.0421 ± 0.0055, tol 0.0100; 105-165: -0.0119 vs 0.0120 ± 0.0014, tol 0.0100; 180-300: -0.0055 vs 0.0110 ± 0.0023, tol 0.0100 |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.08 | 30-90: -0.0396 vs 0.0394 ± 0.0055, tol 0.0166; 105-165: -0.0119 vs 0.0111 ± 0.0014, tol 0.0100; 180-300: -0.0055 vs 0.0100 ± 0.0023, tol 0.0100 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.44 | 30-90: -0.0396 vs -0.0394 ± 0.0055, tol 0.0166; 105-165: -0.0119 vs -0.0111 ± 0.0014, tol 0.0100; 180-300: -0.0055 vs -0.0100 ± 0.0023, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.15 | 30-90: 46.6450 vs 46.2114 ± 2.3587, tol 7.0762; 105-165: 16.2613 vs 15.0367 ± 2.6635, tol 7.9905; 180-300: 9.4106 vs 8.8726 ± 3.4880, tol 10.4641 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 30-90: 282.3501 vs 280.7763 ± 2.3033, tol 6.9099 (restored 300.6130 ± 2.1264, tol 6.3793); 105-165: 266.1430 vs 265.8181 ± 4.1727, tol 12.5182 (restored 271.5385 ± 3.4912, tol 10.4736); 180-300: 265.0640 vs 262.6591 ± 5.3404, tol 16.0212 (restored 267.7871 ± 4.1540, tol 12.4619) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.23 | 30-90: 282.3501 vs 280.7763 ± 2.3033, tol 6.9099; 105-165: 266.1430 vs 265.8181 ± 4.1727, tol 12.5182; 180-300: 265.0640 vs 262.6591 ± 5.3404, tol 16.0212 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.00 | 30-90: 328.9951 vs 326.9878 ± 2.6682, tol 8.0046 (restored 346.8245 ± 4.4852, tol 13.4555); 105-165: 282.4043 vs 280.8548 ± 6.8362, tol 20.5087 (restored 286.5752 ± 6.1547, tol 18.4641); 180-300: 274.4745 vs 271.5317 ± 8.8284, tol 26.4853 (restored 276.6598 ± 7.6420, tol 22.9260) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.25 | 30-90: 328.9951 vs 326.9878 ± 2.6682, tol 8.0046; 105-165: 282.4043 vs 280.8548 ± 6.8362, tol 20.5087; 180-300: 274.4745 vs 271.5317 ± 8.8284, tol 26.4853 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.09 | 30-90: 0.0040 vs 0.0040 ± 0.0061, tol 0.0500 (restored 0.0059 ± 0.0060, tol 0.0500); 105-165: 0.0044 vs 0.0044 ± 0.0075, tol 0.0500 (restored 0.0045 ± 0.0074, tol 0.0500); 180-300: 0.0019 vs 0.0019 ± 0.0072, tol 0.0500 (restored 0.0019 ± 0.0072, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.09 | 30-90: 0.0040 vs 0.0000 ± 0.0061, tol 0.0500; 105-165: 0.0044 vs 0.0000 ± 0.0075, tol 0.0500; 180-300: 0.0019 vs 0.0000 ± 0.0072, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.13 | 30-90: 6.284e+5 vs 6.232e+5 ± 41605.2000, tol 1.248e+5; 105-165: 3.821e+5 vs 3.827e+5 ± 30040.4000, tol 90121.2000; 180-300: 3.338e+5 vs 3.467e+5 ± 32964.2222, tol 98892.6667 |

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

Check-slot step time (informational): candidate fenced median 39.57 ms; references 61.07, 62.16, 38.18, 63.34 ms.

## letters: letters-cellcap8: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.04 | 120: -0.0091 vs 0.0091 ± 0.0018, tol 0.0050; 180: -0.0090 vs 0.0096 ± 0.0004, tol 0.0050; 240: -0.0114 vs 0.0112 ± 0.0005, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.11 | 120: -0.0091 vs 0.0084 ± 0.0018, tol 0.0054; 180: -0.0090 vs 0.0094 ± 0.0004, tol 0.0050; 240: -0.0114 vs 0.0109 ± 0.0005, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.11 | 120: -0.0091 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0090 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0114 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.19 | 60: 0.0514 vs 0.0514 ± 0.0684, tol 1.0000 (restored 0.2796 ± 0.0674, tol 1.0000); 90: 0.0853 vs 0.0853 ± 0.1975, tol 1.0000 (restored 0.4610 ± 0.1974, tol 1.0000); 120: 0.3140 vs 0.3140 ± 0.8755, tol 2.6264 (restored 0.4725 ± 0.8740, tol 2.6221); 180: 0.5417 vs 0.5417 ± 0.9908, tol 2.9724 (restored 0.6523 ± 0.9908, tol 2.9724); 240: 0.7523 vs 0.7523 ± 1.3391, tol 4.0172 (restored 0.8823 ± 1.3389, tol 4.0166) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.19 | 60: 0.0514 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.0853 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.3140 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.5417 vs 0.0000 ± 0.9908, tol 2.9724; 240: 0.7523 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 60: 1941.3119 vs 1945.2446 ± 13.8771, tol 194.5245; 120: 2236.6587 vs 2276.2042 ± 197.0611, tol 591.1833; 180: 3050.1741 vs 3076.0931 ± 169.7827, tol 509.3480; 240: 3024.2720 vs 3049.8967 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.06 | 60: 19.8821 vs 19.8947 ± 0.0109, tol 0.2500 (restored 20.0187 ± 0.0056, tol 0.2500); 120: 20.5389 vs 20.5776 ± 0.2329, tol 0.6988 (restored 20.7325 ± 0.2077, tol 0.6231); 180: 21.7803 vs 21.7108 ± 0.0536, tol 0.2500 (restored 21.8845 ± 0.0525, tol 0.2500); 240: 20.4971 vs 20.4874 ± 0.0250, tol 0.2500 (restored 20.7036 ± 0.0181, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.28 | 60: 19.8821 vs 19.8947 ± 0.0109, tol 0.2500; 120: 20.5389 vs 20.5776 ± 0.2329, tol 0.6988; 180: 21.7803 vs 21.7108 ± 0.0536, tol 0.2500; 240: 20.4971 vs 20.4874 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.33 | 60: 4.894e+5 vs 5.003e+5 ± 8558.0000, tol 50026.2600; 120: 8.703e+5 vs 8.932e+5 ± 22398.0000, tol 89320.4600; 180: 8.778e+5 vs 9.075e+5 ± 19212.0000, tol 90747.3400; 240: 8.499e+5 vs 8.726e+5 ± 17601.0000, tol 87258.5800 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.11 | 120: -0.0091 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0090 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0114 vs -0.0109 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 81.86 ms; references 124.54, 122.69, 127.79, 84.53, 84.70 ms.

## paddleboats: paddleboats-cellcap8: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.41 | 150: -0.0041 vs 0.0031 ± 0.0002, tol 0.0025; 300: -0.0109 vs 0.0102 ± 0.0007, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.45 | 150: -0.0041 vs 0.0030 ± 0.0002, tol 0.0025; 300: -0.0109 vs 0.0098 ± 0.0007, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.45 | 150: -0.0041 vs -0.0030 ± 0.0002, tol 0.0025; 300: -0.0109 vs -0.0098 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.17 | 150: 0.0347 vs 0.0347 ± 0.0215, tol 0.2500 (restored 0.1003 ± 0.0209, tol 0.2500); 300: 0.0418 vs 0.0418 ± 0.0449, tol 0.2500 (restored 0.2638 ± 0.0435, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.17 | 150: 0.0347 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.0418 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.24 | 60: 95.0923 vs 95.1079 ± 3.5559, tol 10.6677; 120: 109.3518 vs 109.5532 ± 2.0263, tol 10.9553; 180: 142.5903 vs 142.2218 ± 1.2921, tol 14.2222; 240: 138.5609 vs 141.9174 ± 4.0339, tol 14.1917; 300: 163.6216 vs 167.6931 ± 3.3795, tol 16.7693 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.22 | 150: 2.553e+5 vs 2.630e+5 ± 11716.0000, tol 35148.0000; 300: 2.908e+5 vs 2.969e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.43 | 150: -0.0094 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0162 vs -0.0152 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 51.25 ms; references 111.75, 88.86, 94.75, 97.52, 50.48 ms.
