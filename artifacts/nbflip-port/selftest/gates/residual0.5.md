# NB-FLIP port gates: seep-ref-p, dam-ref-p, letters-residual0.5, paddleboats-residual0.5

Generated 2026-11-14T23:10:00.487Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-b + seep-ref-c + seep-ref-d + seep-base-e; dam dam-ref-a + dam-ref-b + dam-ref-d + dam-base-e; letters letters-ref-a + letters-ref-b + letters-ref-c + letters-ref-d + letters-base-e; paddleboats paddleboats-ref-a + paddleboats-ref-b + paddleboats-ref-c + paddleboats-ref-d + paddleboats-base-e.
**Overall: FAIL** (seep FAIL, dam pass, letters pass, paddleboats FAIL)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).

## seep: seep-ref-p: FAIL

References seep-ref-a.jsonl, seep-ref-b.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | **FAIL** | 2.35 | mean: 0.2381 vs 0.2698 ± 0.0002, tol 0.0135 **x** |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | **FAIL** | 3.54 | 120-180: 0.2695 vs 0.2693 ± 0.0025, tol 0.0269; 180-240: 0.2717 vs 0.2715 ± 0.0022, tol 0.0272; 240-300: 0.2704 vs 0.2708 ± 0.0018, tol 0.0271; 300-360: 0.2728 vs 0.2727 ± 0.0018, tol 0.0273; 360-420: 0.2732 vs 0.2715 ± 0.0024, tol 0.0272; 420-480: 0.2687 vs 0.2724 ± 0.0006, tol 0.0272; 480-540: 0.2698 vs 0.2710 ± 0.0007, tol 0.0271; 540-600: 0.2692 vs 0.2713 ± 0.0015, tol 0.0271; 600-660: 0.2627 vs 0.2704 ± 0.0008, tol 0.0270; 660-720: 0.2514 vs 0.2699 ± 0.0013, tol 0.0270; 720-780: 0.2427 vs 0.2696 ± 0.0012, tol 0.0270; 780-840: 0.2288 vs 0.2688 ± 0.0013, tol 0.0269 **x**; 840-900: 0.2033 vs 0.2687 ± 0.0017, tol 0.0269 **x**; 900-960: 0.1957 vs 0.2681 ± 0.0005, tol 0.0268 **x**; 960-1020: 0.2067 vs 0.2684 ± 0.0010, tol 0.0268 **x**; 1020-1080: 0.1842 vs 0.2676 ± 0.0011, tol 0.0268 **x**; 1080-1140: 0.1730 vs 0.2671 ± 0.0018, tol 0.0267 **x**; 1140-1200: 0.1722 vs 0.2666 ± 0.0022, tol 0.0267 **x** |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | **FAIL** | 2.15 | 300: 1.3148 vs 1.3209 ± 0.0161, tol 0.0660; 600: 2.6683 vs 2.6797 ± 0.0151, tol 0.1340; 900: 3.8573 vs 4.0271 ± 0.0145, tol 0.2014; 1200: 4.7890 vs 5.3649 ± 0.0141, tol 0.2682 **x** |
| G3 | right tank volume (L) | |c - ref| <= max(5%, 3S) | pass | 0.17 | 300: 1.1463 vs 1.1561 ± 0.0701, tol 0.2103; 600: 2.1277 vs 2.1098 ± 0.0928, tol 0.2784; 900: 2.8188 vs 2.8735 ± 0.1074, tol 0.3223; 1200: 3.4037 vs 3.3794 ± 0.1684, tol 0.5053 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.01 | median: 2.6412 vs 2.6387 ± 0.0043, tol 0.2639 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.15 | 90%: 302.5882 vs 291.8473 ± 4.1487, tol 72.9618 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.00 | 1200: 16356.0000 vs 16352.4000 ± 72.0000, tol 817.6200 |
| G6c | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | pass | 0.03 | 600: 367.4680 vs 367.6399 ± 2.7158, tol 16.0000; 1200: 368.1219 vs 368.6305 ± 6.6246, tol 19.8737 |
| G6 | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | pass | 0.18 | 600: 0.0788 vs 0.0000 ± 0.1428, tol 0.4283; 1200: 0.0488 vs 0.0000 ± 0.0992, tol 0.2977 |
| G7 | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | **FAIL** | 1.56 | 1200: -0.0031 vs -0.0051 ± 0.0004, tol 0.0013 **x** |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.26 | 300: 384.0916 vs 381.3763 ± 3.4907, tol 10.4722; 600: 379.2514 vs 380.8470 ± 3.5712, tol 10.7137 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.26 | 300: 0.3517 vs 0.0000 ± 1.3113, tol 3.9339; 600: 0.8370 vs 0.0000 ± 1.0738, tol 3.2214 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.24 | 300: 70459.0000 vs 69056.6000 ± 4565.0000, tol 13695.0000; 600: 1.298e+5 vs 1.299e+5 ± 8388.0000, tol 25164.0000; 900: 1.787e+5 vs 1.831e+5 ± 6281.0000, tol 18843.0000; 1200: 2.144e+5 vs 2.175e+5 ± 14011.0000, tol 42033.0000 |

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

Check-slot step time (informational): candidate fenced median 36.04 ms; references 37.24, 37.57, 35.16, 65.34, 34.40 ms.

## dam: dam-ref-p: PASS

References dam-ref-a.jsonl, dam-ref-b.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 165 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| C0f | completed 300 steps with no solver failure | required | (pass, not gated) | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.19 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 107.0000 vs 107.7500 ± 1.0000, tol 4.0000; 10: 121.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7500 ± 0.0000, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | pass | 0.08 | 16: -0.0425 vs -0.0443 ± 0.0078, tol 0.0234 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.01 | 4: 36.0847 vs 36.0848 ± 0.0002, tol 3.6085; 8: 119.4932 vs 119.5022 ± 0.0185, tol 11.9502; 12: 228.7291 vs 228.6883 ± 0.1055, tol 22.8688; 16: 274.5491 vs 274.2947 ± 2.1784, tol 27.4295 |
| D5ce | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | pass | 0.07 | 4: 669.1634 vs 669.1674 ± 0.0033, tol 13.3833; 8: 670.7808 vs 670.7934 ± 0.0191, tol 13.4159; 12: 664.2019 vs 664.0983 ± 0.1563, tol 13.2820; 16: 590.8446 vs 590.0086 ± 4.1982, tol 12.5946 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.00 | 4: 7624.0000 vs 7624.0000 ± 0.0000, tol 381.2000; 8: 10763.0000 vs 10765.0000 ± 9.0000, tol 538.2500; 12: 14937.0000 vs 14937.7500 ± 5.0000, tol 746.8875 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | pass | 0.17 | 8: 0.1081 vs 0.0000 ± 0.1449, tol 4.0000; 12: 0.1548 vs 0.0000 ± 0.4137, tol 4.0000; 16: 0.8586 vs 0.0000 ± 1.6789, tol 5.0368 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.16 | 16: 3.565e+5 vs 3.509e+5 ± 9091.0000, tol 35092.0750 |
| D3 | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | pass | 0.13 | 30-90: 44.6480 vs 44.7963 ± 0.9607, tol 2.8820; 105-165: 47.2685 vs 46.7554 ± 1.3523, tol 4.0569 |
| D4 | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | pass | 0.16 | 30-90: -0.0421 vs -0.0394 ± 0.0055, tol 0.0166; 105-165: -0.0123 vs -0.0111 ± 0.0014, tol 0.0100 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.15 | 30-90: 47.2047 vs 46.2114 ± 2.3587, tol 7.0762; 105-165: 16.2303 vs 15.0367 ± 2.6635, tol 7.9905 |
| D5p | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | pass | 0.13 | 30-90: 281.6862 vs 280.7763 ± 2.3033, tol 6.9099; 105-165: 266.5009 vs 265.8181 ± 4.1727, tol 12.5182 |
| D5e | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | pass | 0.24 | 30-90: 328.8909 vs 326.9878 ± 2.6682, tol 8.0046; 105-165: 282.7312 vs 280.8548 ± 6.8362, tol 20.5087 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | pass | 0.15 | 30-90: 0.0033 vs 0.0000 ± 0.0061, tol 0.0500; 105-165: 0.0074 vs 0.0000 ± 0.0075, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.19 | 30-90: 6.468e+5 vs 6.232e+5 ± 41605.2000, tol 1.248e+5; 105-165: 3.857e+5 vs 3.827e+5 ± 30040.4000, tol 90121.2000 |
| D3+ | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.05 | 180-300: 46.8296 vs 46.6521 ± 1.1031, tol 3.3093 |
| D4+ | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.31 | 180-300: -0.0069 vs -0.0100 ± 0.0023, tol 0.0100 |
| D5k+ | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | (pass, not gated) | 0.13 | 180-300: 10.2685 vs 8.8726 ± 3.4880, tol 10.4641 |
| D5p+ | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.19 | 180-300: 265.7719 vs 262.6591 ± 5.3404, tol 16.0212 |
| D5e+ | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.17 | 180-300: 276.0404 vs 271.5317 ± 8.8284, tol 26.4853 |
| D7d+ | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.07 | 180-300: 0.0036 vs 0.0000 ± 0.0072, tol 0.0500 |
| D8+ | particles, window mean | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.05 | 180-300: 3.520e+5 vs 3.467e+5 ± 32964.2222, tol 98892.6667 |

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

Check-slot step time (informational): candidate fenced median 58.23 ms; references 61.07, 62.16, 38.18, 63.34 ms.

## letters: letters-residual0.5: PASS

References letters-ref-a.jsonl, letters-ref-b.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | pass | 0.19 | 120: -0.0088 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0085 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0106 vs -0.0109 ± 0.0005, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | pass | 0.19 | 60: 0.0458 vs 0.0000 ± 0.0684, tol 1.0000; 90: 0.1051 vs 0.0000 ± 0.1975, tol 1.0000; 120: 0.3600 vs 0.0000 ± 0.8755, tol 2.6264; 180: 0.5497 vs 0.0000 ± 0.9908, tol 2.9724; 240: 0.7646 vs 0.0000 ± 1.3391, tol 4.0172 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.17 | 60: 1946.6861 vs 1945.2446 ± 13.8771, tol 194.5245; 120: 2183.1489 vs 2276.2042 ± 197.0611, tol 591.1833; 180: 3164.1983 vs 3076.0931 ± 169.7827, tol 509.3480; 240: 3071.5807 vs 3049.8967 ± 167.6953, tol 503.0859 |
| L4 | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | pass | 0.20 | 60: 19.8923 vs 19.8947 ± 0.0109, tol 0.2500; 120: 20.5203 vs 20.5776 ± 0.2329, tol 0.6988; 180: 21.7600 vs 21.7108 ± 0.0536, tol 0.2500; 240: 20.5004 vs 20.4874 ± 0.0250, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.04 | 60: 4.983e+5 vs 5.003e+5 ± 8558.0000, tol 50026.2600; 120: 8.915e+5 vs 8.932e+5 ± 22398.0000, tol 89320.4600; 180: 9.101e+5 vs 9.075e+5 ± 19212.0000, tol 90747.3400; 240: 8.751e+5 vs 8.726e+5 ± 17601.0000, tol 87258.5800 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.19 | 120: -0.0088 vs -0.0084 ± 0.0018, tol 0.0054; 180: -0.0085 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0106 vs -0.0109 ± 0.0005, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 158.99 ms; references 124.54, 122.69, 127.79, 84.53, 84.70 ms.

## paddleboats: paddleboats-residual0.5: FAIL

References paddleboats-ref-a.jsonl, paddleboats-ref-b.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | **FAIL** | 1.53 | 150: -0.0027 vs -0.0030 ± 0.0002, tol 0.0025; 300: -0.0060 vs -0.0098 ± 0.0007, tol 0.0025 **x** |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | pass | 0.42 | 150: 0.0330 vs 0.0000 ± 0.0215, tol 0.2500; 300: 0.1055 vs 0.0000 ± 0.0449, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.41 | 60: 94.8638 vs 95.1079 ± 3.5559, tol 10.6677; 120: 107.2410 vs 109.5532 ± 2.0263, tol 10.9553; 180: 140.5307 vs 142.2218 ± 1.2921, tol 14.2222; 240: 136.5175 vs 141.9174 ± 4.0339, tol 14.1917; 300: 160.7762 vs 167.6931 ± 3.3795, tol 16.7693 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.02 | 150: 2.632e+5 vs 2.630e+5 ± 11716.0000, tol 35148.0000; 300: 2.953e+5 vs 2.969e+5 ± 23924.0000, tol 71772.0000 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (fail, not gated) | 1.53 | 150: -0.0080 vs -0.0083 ± 0.0002, tol 0.0025; 300: -0.0114 vs -0.0152 ± 0.0007, tol 0.0025 **x** |

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

Check-slot step time (informational): candidate fenced median 45.91 ms; references 111.75, 88.86, 94.75, 97.52, 50.48 ms.
