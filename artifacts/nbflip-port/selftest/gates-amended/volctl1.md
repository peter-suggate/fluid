# NB-FLIP port gates: dam-volctl1, letters-volctl1, paddleboats-volctl1

Generated 2026-11-15T01:40:30.568Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-base-e; dam dam-ref-a + dam-ref-b + dam-ref-d + dam-base-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-base-e.
**Overall: FAIL** (dam FAIL, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## dam: dam-volctl1: FAIL

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.06 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 107.7500 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.03 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8224 ± 0.0010, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1666 ± 0.0036, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.4897 ± 0.0162, tol 4.0000); 16: 78.6250 vs 78.7500 ± 0.0000, tol 4.0000 (restored 80.8142 ± 0.3623, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.03 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.6250 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | pass | 0.00 | 16: -0.0443 vs 0.0500 ± 0.0078, tol 0.0100 |
| D4c (first ruling) | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0443 vs 0.0443 ± 0.0078, tol 0.0234 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.00 | 16: -0.0443 vs -0.0443 ± 0.0078, tol 0.0234 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0847 vs 36.0848 ± 0.0002, tol 3.6085; 8: 119.5060 vs 119.5022 ± 0.0185, tol 11.9502; 12: 228.7273 vs 228.6883 ± 0.1055, tol 22.8688; 16: 273.7924 vs 274.2947 ± 2.1784, tol 27.4295 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.08 | 4: 669.1681 vs 669.1674 ± 0.0033, tol 13.3833 (restored 669.9821 ± 0.0077, tol 13.3996); 8: 670.7983 vs 670.7934 ± 0.0191, tol 13.4159 (restored 671.9633 ± 0.0225, tol 13.4393); 12: 664.1954 vs 664.0983 ± 0.1563, tol 13.2820 (restored 667.4899 ± 0.0445, tol 13.3498); 16: 589.0613 vs 590.0086 ± 4.1982, tol 12.5946 (restored 612.7269 ± 0.4477, tol 12.2545) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.08 | 4: 669.1681 vs 669.1674 ± 0.0033, tol 13.3833; 8: 670.7983 vs 670.7934 ± 0.0191, tol 13.4159; 12: 664.1954 vs 664.0983 ± 0.1563, tol 13.2820; 16: 589.0613 vs 590.0086 ± 4.1982, tol 12.5946 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.01 | 4: 7623.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10773.0000 vs 10765.0000 ± 9.0000, tol 538.2500; 12: 14940.0000 vs 14937.7500 ± 5.0000, tol 746.8875 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.18 | 8: 0.0362 vs 0.0362 ± 0.1449, tol 4.0000 (restored 0.0953 ± 0.1653, tol 4.0000); 12: 0.1062 vs 0.1062 ± 0.4137, tol 4.0000 (restored 0.2904 ± 0.4138, tol 4.0000); 16: 0.9154 vs 0.9154 ± 1.6789, tol 5.0368 (restored 2.1463 ± 1.5458, tol 4.6375) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.18 | 8: 0.0362 vs 0.0000 ± 0.1449, tol 4.0000; 12: 0.1062 vs 0.0000 ± 0.4137, tol 4.0000; 16: 0.9154 vs 0.0000 ± 1.6789, tol 5.0368 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.01 | 16: 3.514e+5 vs 3.509e+5 ± 9091.0000, tol 35092.0750 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | **FAIL** | 2.36 | 30-90: 37.9938 vs 44.7963 ± 0.9607, tol 2.8820 (restored 46.6066 ± 1.2089, tol 3.6267) **x**; 105-165: 43.1845 vs 46.7554 ± 1.3523, tol 4.0569 (restored 47.2666 ± 1.2912, tol 3.8736); 180-300: 45.2605 vs 46.6521 ± 1.1031, tol 3.3093 (restored 47.1100 ± 1.0824, tol 3.2473) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (fail, not gated) | 2.36 | 30-90: 37.9938 vs 44.7963 ± 0.9607, tol 2.8820 **x**; 105-165: 43.1845 vs 46.7554 ± 1.3523, tol 4.0569; 180-300: 45.2605 vs 46.6521 ± 1.1031, tol 3.3093 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.01 | **FAIL** | 14.21 | 30-90: -0.1842 vs 0.0421 ± 0.0055, tol 0.0100 **x**; 105-165: -0.0996 vs 0.0120 ± 0.0014, tol 0.0100 **x**; 180-300: -0.0337 vs 0.0110 ± 0.0023, tol 0.0100 **x** |
| D4 (first ruling) | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | (fail, not gated) | 8.85 | 30-90: -0.1842 vs 0.0394 ± 0.0055, tol 0.0166 **x**; 105-165: -0.0996 vs 0.0111 ± 0.0014, tol 0.0100 **x**; 180-300: -0.0337 vs 0.0100 ± 0.0023, tol 0.0100 **x** |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (fail, not gated) | 8.85 | 30-90: -0.1842 vs -0.0394 ± 0.0055, tol 0.0166 **x**; 105-165: -0.0996 vs -0.0111 ± 0.0014, tol 0.0100 **x**; 180-300: -0.0337 vs -0.0100 ± 0.0023, tol 0.0100 **x** |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.98 | 30-90: 39.2494 vs 46.2114 ± 2.3587, tol 7.0762; 105-165: 12.4887 vs 15.0367 ± 2.6635, tol 7.9905; 180-300: 9.8920 vs 8.8726 ± 3.4880, tol 10.4641 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | **FAIL** | 10.55 | 30-90: 207.8963 vs 280.7763 ± 2.3033, tol 6.9099 (restored 300.6130 ± 2.1264, tol 6.3793) **x**; 105-165: 221.0190 vs 265.8181 ± 4.1727, tol 12.5182 (restored 271.5385 ± 3.4912, tol 10.4736) **x**; 180-300: 250.9984 vs 262.6591 ± 5.3404, tol 16.0212 (restored 267.7871 ± 4.1540, tol 12.4619) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 10.55 | 30-90: 207.8963 vs 280.7763 ± 2.3033, tol 6.9099 **x**; 105-165: 221.0190 vs 265.8181 ± 4.1727, tol 12.5182 **x**; 180-300: 250.9984 vs 262.6591 ± 5.3404, tol 16.0212 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | **FAIL** | 7.41 | 30-90: 247.1457 vs 326.9878 ± 2.6682, tol 8.0046 (restored 346.8245 ± 4.4852, tol 13.4555) **x**; 105-165: 233.5077 vs 280.8548 ± 6.8362, tol 20.5087 (restored 286.5752 ± 6.1547, tol 18.4641) **x**; 180-300: 260.8904 vs 271.5317 ± 8.8284, tol 26.4853 (restored 276.6598 ± 7.6420, tol 22.9260) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (fail, not gated) | 9.97 | 30-90: 247.1457 vs 326.9878 ± 2.6682, tol 8.0046 **x**; 105-165: 233.5077 vs 280.8548 ± 6.8362, tol 20.5087 **x**; 180-300: 260.8904 vs 271.5317 ± 8.8284, tol 26.4853 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.12 | 30-90: 0.0045 vs 0.0045 ± 0.0061, tol 0.0500 (restored 0.0074 ± 0.0060, tol 0.0500); 105-165: 0.0060 vs 0.0060 ± 0.0075, tol 0.0500 (restored 0.0060 ± 0.0074, tol 0.0500); 180-300: 0.0048 vs 0.0048 ± 0.0072, tol 0.0500 (restored 0.0048 ± 0.0072, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.12 | 30-90: 0.0045 vs 0.0000 ± 0.0061, tol 0.0500; 105-165: 0.0060 vs 0.0000 ± 0.0075, tol 0.0500; 180-300: 0.0048 vs 0.0000 ± 0.0072, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.12 | 30-90: 6.148e+5 vs 6.232e+5 ± 41605.2000, tol 1.248e+5; 105-165: 3.715e+5 vs 3.827e+5 ± 30040.4000, tol 90121.2000; 180-300: 3.460e+5 vs 3.467e+5 ± 32964.2222, tol 98892.6667 |

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

Check-slot step time (informational): candidate fenced median 41.01 ms; references 61.07, 62.16, 38.18, 63.34 ms.

## letters: letters-volctl1: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.005 | pass | 0.58 | 120: -0.0098 vs 0.0091 ± 0.0018, tol 0.0050; 180: -0.0122 vs 0.0096 ± 0.0004, tol 0.0050; 240: -0.0141 vs 0.0112 ± 0.0005, tol 0.0050 |
| L1 (first ruling) | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | (pass, not gated) | 0.64 | 120: -0.0098 vs 0.0084 ± 0.0018, tol 0.0054; 180: -0.0122 vs 0.0094 ± 0.0004, tol 0.0050; 240: -0.0141 vs 0.0109 ± 0.0005, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.64 | 120: -0.0098 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0122 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0141 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.23 | 60: 0.0269 vs 0.0269 ± 0.0684, tol 1.0000 (restored 0.2619 ± 0.0674, tol 1.0000); 90: 0.0698 vs 0.0698 ± 0.1975, tol 1.0000 (restored 0.4568 ± 0.1974, tol 1.0000); 120: 0.2588 vs 0.2588 ± 0.8755, tol 2.6264 (restored 0.4588 ± 0.8740, tol 2.6221); 180: 0.6950 vs 0.6950 ± 0.9908, tol 2.9724 (restored 0.8449 ± 0.9908, tol 2.9724); 240: 0.7251 vs 0.7251 ± 1.3391, tol 4.0172 (restored 0.9144 ± 1.3389, tol 4.0166) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.23 | 60: 0.0269 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.0698 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.2588 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.6950 vs 0.0000 ± 0.9908, tol 2.9724; 240: 0.7251 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.07 | 60: 1949.3027 vs 1945.2446 ± 13.8771, tol 194.5245; 120: 2318.2600 vs 2276.2042 ± 197.0611, tol 591.1833; 180: 3038.4792 vs 3076.0931 ± 169.7827, tol 509.3480; 240: 3043.4770 vs 3049.8967 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.35 | 60: 19.8931 vs 19.8947 ± 0.0109, tol 0.2500 (restored 20.0187 ± 0.0056, tol 0.2500); 120: 20.5801 vs 20.5776 ± 0.2329, tol 0.6988 (restored 20.7325 ± 0.2077, tol 0.6231); 180: 21.6229 vs 21.7108 ± 0.0536, tol 0.2500 (restored 21.8845 ± 0.0525, tol 0.2500); 240: 20.4099 vs 20.4874 ± 0.0250, tol 0.2500 (restored 20.7036 ± 0.0181, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.35 | 60: 19.8931 vs 19.8947 ± 0.0109, tol 0.2500; 120: 20.5801 vs 20.5776 ± 0.2329, tol 0.6988; 180: 21.6229 vs 21.7108 ± 0.0536, tol 0.2500; 240: 20.4099 vs 20.4874 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.11 | 60: 5.038e+5 vs 5.003e+5 ± 8558.0000, tol 50026.2600; 120: 9.028e+5 vs 8.932e+5 ± 22398.0000, tol 89320.4600; 180: 9.011e+5 vs 9.075e+5 ± 19212.0000, tol 90747.3400; 240: 8.687e+5 vs 8.726e+5 ± 17601.0000, tol 87258.5800 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.64 | 120: -0.0098 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0122 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0141 vs -0.0109 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 113.59 ms; references 124.54, 122.69, 127.79, 84.53, 84.70 ms.

## paddleboats: paddleboats-volctl1: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided: within NB-FLIP's worst reference | |c| <= max |ref_i| + 0.0025 | pass | 0.00 | 150: -0.0030 vs 0.0031 ± 0.0002, tol 0.0025; 300: -0.0096 vs 0.0102 ± 0.0007, tol 0.0025 |
| P1 (first ruling) | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | (pass, not gated) | 0.01 | 150: -0.0030 vs 0.0030 ± 0.0002, tol 0.0025; 300: -0.0096 vs 0.0098 ± 0.0007, tol 0.0025 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.11 | 150: -0.0030 vs -0.0030 ± 0.0002, tol 0.0025; 300: -0.0096 vs -0.0098 ± 0.0007, tol 0.0025 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.08 | 150: 0.0077 vs 0.0077 ± 0.0215, tol 0.2500 (restored 0.0722 ± 0.0209, tol 0.2500); 300: 0.0222 vs 0.0222 ± 0.0449, tol 0.2500 (restored 0.2307 ± 0.0435, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.09 | 150: 0.0077 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.0222 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.11 | 60: 95.8531 vs 95.1079 ± 3.5559, tol 10.6677; 120: 109.1023 vs 109.5532 ± 2.0263, tol 10.9553; 180: 143.8244 vs 142.2218 ± 1.2921, tol 14.2222; 240: 142.4680 vs 141.9174 ± 4.0339, tol 14.1917; 300: 166.9309 vs 167.6931 ± 3.3795, tol 16.7693 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.52 | 150: 2.688e+5 vs 2.630e+5 ± 11716.0000, tol 35148.0000; 300: 3.343e+5 vs 2.969e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.11 | 150: -0.0083 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0149 vs -0.0152 ± 0.0007, tol 0.0025 |

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

Check-slot step time (informational): candidate fenced median 49.65 ms; references 111.75, 88.86, 94.75, 97.52, 50.48 ms.
