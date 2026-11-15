# NB-FLIP port gates: dam-ref-b, letters-ref-b, paddleboats-ref-b, seep-ref-b

Generated 2026-11-15T00:46:27.031Z by tools/nbflip-gates.ts. References: seep seep-ref-a + seep-ref-c + seep-ref-d + seep-ref-e; dam dam-ref-a + dam-ref-c + dam-ref-d + dam-ref-e; letters letters-ref-a + letters-ref-c + letters-ref-d + letters-ref-e; paddleboats paddleboats-ref-a + paddleboats-ref-c + paddleboats-ref-d + paddleboats-ref-e.
**Overall: PASS** (seep pass, dam pass, letters pass, paddleboats pass)

Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).
Amended (plan §4.5): drift rows are one-sided, |c| <= |ref| + max(floor, 3S) (margin = excess / tolerance); volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.

## seep: seep-ref-b: PASS

References seep-ref-a.jsonl, seep-ref-c.jsonl, seep-ref-d.jsonl, seep-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 1200 steps with no solver failure | required | pass | 0.00 | 1200: 1200.0000 vs - ± 0.0000, tol 0.0000 |
| G1 | q_left, mean over 60-step windows 120-1200 (L/s) | |c - ref| <= max(5%, 3S) | pass | 0.06 | mean: 0.2699 vs 0.2691 ± 0.0019, tol 0.0135 |
| G1w | q_left per 60-step window (L/s) | |c - ref| <= max(10%, 3S) | pass | 0.09 | 120-180: 0.2692 vs 0.2697 ± 0.0022, tol 0.0270; 180-240: 0.2724 vs 0.2709 ± 0.0032, tol 0.0271; 240-300: 0.2707 vs 0.2712 ± 0.0023, tol 0.0271; 300-360: 0.2734 vs 0.2720 ± 0.0026, tol 0.0272; 360-420: 0.2719 vs 0.2722 ± 0.0035, tol 0.0272; 420-480: 0.2728 vs 0.2713 ± 0.0029, tol 0.0271; 480-540: 0.2709 vs 0.2695 ± 0.0034, tol 0.0269; 540-600: 0.2703 vs 0.2714 ± 0.0044, tol 0.0271; 600-660: 0.2705 vs 0.2689 ± 0.0034, tol 0.0269; 660-720: 0.2694 vs 0.2690 ± 0.0036, tol 0.0269; 720-780: 0.2698 vs 0.2682 ± 0.0028, tol 0.0268; 780-840: 0.2683 vs 0.2684 ± 0.0038, tol 0.0268; 840-900: 0.2703 vs 0.2679 ± 0.0035, tol 0.0268; 900-960: 0.2687 vs 0.2675 ± 0.0038, tol 0.0268; 960-1020: 0.2678 vs 0.2667 ± 0.0041, tol 0.0267; 1020-1080: 0.2679 vs 0.2667 ± 0.0022, tol 0.0267; 1080-1140: 0.2668 vs 0.2666 ± 0.0031, tol 0.0267; 1140-1200: 0.2669 vs 0.2650 ± 0.0028, tol 0.0265 |
| G2 | left tank loss (L) | |c - ref| <= max(5%, 3S) | pass | 0.05 | 300: 1.3193 vs 1.3197 ± 0.0049, tol 0.0660; 600: 2.6786 vs 2.6761 ± 0.0148, tol 0.1338; 900: 4.0269 vs 4.0185 ± 0.0259, tol 0.2009; 1200: 5.3650 vs 5.3510 ± 0.0355, tol 0.2675 |
| G3 | right tank volume (L); band to the restored references | c within max(5%, 3S) of [ref, ref restored] | pass | 0.32 | 300: 1.1561 vs 1.1503 ± 0.0298, tol 0.0895 (restored 1.3197 ± 0.0050, tol 0.0660); 600: 2.1571 vs 2.1184 ± 0.0678, tol 0.2035 (restored 2.6761 ± 0.0148, tol 0.1338); 900: 2.9001 vs 2.9645 ± 0.1341, tol 0.4024 (restored 4.0185 ± 0.0259, tol 0.2009); 1200: 3.3708 vs 3.5023 ± 0.1355, tol 0.4065 (restored 5.3510 ± 0.0356, tol 0.2676) |
| G3 (as written) | right tank volume (L) | |c - ref| <= max(5%, 3S) | (pass, not gated) | 0.32 | 300: 1.1561 vs 1.1503 ± 0.0298, tol 0.0895; 600: 2.1571 vs 2.1184 ± 0.0678, tol 0.2035; 900: 2.9001 vs 2.9645 ± 0.1341, tol 0.4024; 1200: 3.3708 vs 3.5023 ± 0.1355, tol 0.4065 |
| G4 | throat u, median over samples 120-1200 (m/s) | |c - ref| <= max(10%, 3S) | pass | 0.02 | median: 2.6385 vs 2.6429 ± 0.0125, tol 0.2643 |
| G5 | step at which wet floor reaches 90% | |c - ref| <= max(25%, 3S) | pass | 0.01 | 90%: 287.5155 vs 288.4712 ± 17.8628, tol 72.1178 |
| G5f | wet floor cells at 1200 | |c - ref| <= max(5%, 3S) | pass | 0.00 | 1200: 16367.0000 vs 16370.7500 ± 12.0000, tol 818.5375 |
| G6c | right-chamber centroid x (cells); band to the restored references | c within max(16 cells, 3S) of [ref, ref restored] | pass | 0.15 | 600: 366.5041 vs 368.9643 ± 2.5899, tol 16.0000 (restored 372.0910 ± 2.2324, tol 16.0000); 1200: 368.8735 vs 370.5225 ± 4.0817, tol 16.0000 (restored 375.1717 ± 2.7714, tol 16.0000) |
| G6c (as written) | right-chamber centroid x (cells) | |c - ref| <= max(16 cells, 3S) | (pass, not gated) | 0.15 | 600: 366.5041 vs 368.9643 ± 2.5899, tol 16.0000; 1200: 368.8735 vs 370.5225 ± 4.0817, tol 16.0000 |
| G6 | right-chamber 16-bin profile, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.15, 3S) | pass | 0.19 | 600: 0.0646 vs 0.0646 ± 0.1616, tol 0.4848 (restored 0.1389 ± 0.1321, tol 0.3962); 1200: 0.0417 vs 0.0417 ± 0.0722, tol 0.2165 (restored 0.0995 ± 0.0500, tol 0.1501) |
| G6 (as written) | right-chamber 16-bin profile, normalised L1 | d(c, ref) <= max(0.15, 3S), S the largest pairwise d between references | (pass, not gated) | 0.19 | 600: 0.0646 vs 0.0000 ± 0.1616, tol 0.4848; 1200: 0.0417 vs 0.0000 ± 0.0722, tol 0.2165 |
| G7 | total drift at 1200 (fraction), one-sided on |drift| | |c| <= |ref| + max(0.001, 3S) | pass | 0.28 | 1200: -0.0051 vs 0.0047 ± 0.0005, tol 0.0014 |
| G7 (as written) | total drift at 1200 (fraction) | |c - ref| <= max(0.001, 3S) | (pass, not gated) | 0.28 | 1200: -0.0051 vs -0.0047 ± 0.0005, tol 0.0014 |
| G8 | jet impact x (cells) | |c - ref| <= max(8 cells, 3S) | pass | 0.30 | 300: 379.9732 vs 380.3955 ± 1.6943, tol 8.0000; 600: 383.0944 vs 380.1165 ± 3.3331, tol 9.9992 |
| G8h | jet mean height per 8-cell slice over x 256-352, RMS (cells) | d(c, ref) <= max(3 cells, 3S), S the largest pairwise d between references | pass | 0.23 | 300: 0.9058 vs 0.0000 ± 1.3062, tol 3.9186; 600: 0.7489 vs 0.0000 ± 1.2470, tol 3.7409 |
| Gp | particles | |c - ref| <= max(10%, 3S) | (pass, not gated) | 0.36 | 300: 69125.0000 vs 69211.5000 ± 2891.0000, tol 8673.0000; 600: 1.312e+5 vs 1.299e+5 ± 6052.0000, tol 18156.0000; 900: 1.850e+5 vs 1.867e+5 ± 9143.0000, tol 27429.0000; 1200: 2.152e+5 vs 2.234e+5 ± 5282.0000, tol 22338.5000 |

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

Check-slot step time (informational): candidate fenced median 68.99 ms; references 54.13, 49.39, 53.44, 74.01 ms.

## dam: dam-ref-b: PASS

References dam-ref-a.jsonl, dam-ref-c.jsonl, dam-ref-d.jsonl, dam-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| D1 | front x_f, furthest wet floor x (cells; Z = x_f / 80) | |c - ref| <= max(4 cells, 3S) | pass | 0.25 | 2: 84.0000 vs 84.0000 ± 0.0000, tol 4.0000; 4: 88.0000 vs 88.0000 ± 0.0000, tol 4.0000; 6: 96.0000 vs 96.0000 ± 0.0000, tol 4.0000; 8: 108.0000 vs 108.0000 ± 0.0000, tol 4.0000; 10: 122.0000 vs 121.0000 ± 0.0000, tol 4.0000 |
| D2 | residual column height at the back wall, highest wet top (cells; eta = h / 117.8); band to the restored references | c within max(4 cells, 3S) of [ref, ref restored] | pass | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000 (restored 114.8221 ± 0.0001, tol 4.0000); 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000 (restored 106.1681 ± 0.0027, tol 4.0000); 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000 (restored 93.5053 ± 0.0859, tol 4.0000); 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 (restored 80.8168 ± 0.3890, tol 4.0000) |
| D2 (as written) | residual column height at the back wall, highest wet top (cells; eta = h / 117.8) | |c - ref| <= max(4 cells, 3S) | (pass, not gated) | 0.00 | 4: 114.7500 vs 114.7500 ± 0.0000, tol 4.0000; 8: 106.0625 vs 106.0625 ± 0.0000, tol 4.0000; 12: 93.1875 vs 93.1875 ± 0.0000, tol 4.0000; 16: 78.7500 vs 78.7656 ± 0.0625, tol 4.0000 |
| D4c | volume drift at the far-wall impact (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.00 | 16: -0.0427 vs 0.0440 ± 0.0068, tol 0.0205 |
| D4c (as written) | volume drift at the far-wall impact (fraction of t = 0) | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.06 | 16: -0.0427 vs -0.0440 ± 0.0068, tol 0.0205 |
| D5c | collapse kinetic energy (J) | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.02 | 4: 36.0848 vs 36.0848 ± 0.0003, tol 3.6085; 8: 119.5019 vs 119.4962 ± 0.0237, tol 11.9496; 12: 228.6907 vs 228.5833 ± 0.4230, tol 22.8583; 16: 274.5831 vs 275.1137 ± 1.1582, tol 27.5114 |
| D5ce | collapse total mechanical energy (J); band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.02 | 4: 669.1684 vs 669.1684 ± 0.0008, tol 13.3834 (restored 669.9803 ± 0.0009, tol 13.3996); 8: 670.7927 vs 670.7833 ± 0.0429, tol 13.4157 (restored 671.9704 ± 0.0280, tol 13.4394); 12: 664.1584 vs 663.9469 ± 0.7326, tol 13.2789 (restored 667.5123 ± 0.2280, tol 13.3502); 16: 590.7427 vs 591.0129 ± 0.9253, tol 11.8203 (restored 613.5917 ± 3.2444, tol 12.2718) |
| D5ce (as written) | collapse total mechanical energy (J) | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.02 | 4: 669.1684 vs 669.1684 ± 0.0008, tol 13.3834; 8: 670.7927 vs 670.7833 ± 0.0429, tol 13.4157; 12: 664.1584 vs 663.9469 ± 0.7326, tol 13.2789; 16: 590.7427 vs 591.0129 ± 0.9253, tol 11.8203 |
| D6 | wet floor cells | |c - ref| <= max(5%, 3S) | pass | 0.00 | 4: 7624.0000 vs 7623.5000 ± 2.0000, tol 381.1750; 8: 10768.0000 vs 10767.2500 ± 4.0000, tol 538.3625; 12: 14937.0000 vs 14936.0000 ± 10.0000, tol 746.8000 |
| D7t | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(4 cells, 3S) | pass | 0.14 | 8: 0.0078 vs 0.0078 ± 0.0156, tol 4.0000 (restored 0.0977 ± 0.0155, tol 4.0000); 12: 0.1220 vs 0.1220 ± 0.4381, tol 4.0000 (restored 0.2965 ± 0.4252, tol 4.0000); 16: 0.9534 vs 0.9534 ± 2.2985, tol 6.8954 (restored 2.5515 ± 2.2457, tol 6.7371) |
| D7t (as written) | collapse surface profile: highest wet top per 8-cell x bin, RMS (cells) | d(c, ref) <= max(4 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.14 | 8: 0.0078 vs 0.0000 ± 0.0156, tol 4.0000; 12: 0.1220 vs 0.0000 ± 0.4381, tol 4.0000; 16: 0.9534 vs 0.0000 ± 2.2985, tol 6.8954 |
| D8c | particles at 16 | |c - ref| <= max(10%, 3S) | pass | 0.01 | 16: 3.513e+5 vs 3.516e+5 ± 5756.0000, tol 35163.9500 |
| D3 | back-wall column mean depth (cells), window mean; band to the restored references | c within max(5% | 2 cells, 3S) of [ref, ref restored] | pass | 0.16 | 30-90: 44.9343 vs 45.3923 ± 0.9511, tol 2.8534 (restored 47.2436 ± 0.7513, tol 2.3622); 105-165: 47.3057 vs 47.0393 ± 0.7294, tol 2.3520 (restored 47.4842 ± 0.6038, tol 2.3742); 180-300: 47.1773 vs 46.7457 ± 0.2049, tol 2.3373 (restored 47.0873 ± 0.5841, tol 2.3544) |
| D3 (as written) | back-wall column mean depth (cells), window mean | |c - ref| <= max(5% | 2 cells, 3S) | (pass, not gated) | 0.18 | 30-90: 44.9343 vs 45.3923 ± 0.9511, tol 2.8534; 105-165: 47.3057 vs 47.0393 ± 0.7294, tol 2.3520; 180-300: 47.1773 vs 46.7457 ± 0.2049, tol 2.3373 |
| D4 | volume drift (fraction of t = 0), window mean, one-sided on |drift| | |c| <= |ref| + max(0.01, 3S) | pass | 0.08 | 30-90: -0.0409 vs 0.0402 ± 0.0090, tol 0.0270; 105-165: -0.0101 vs 0.0094 ± 0.0082, tol 0.0246; 180-300: -0.0102 vs 0.0065 ± 0.0150, tol 0.0450 |
| D4 (as written) | volume drift (fraction of t = 0), window mean | |c - ref| <= max(0.01, 3S) | (pass, not gated) | 0.08 | 30-90: -0.0409 vs -0.0402 ± 0.0090, tol 0.0270; 105-165: -0.0101 vs -0.0094 ± 0.0082, tol 0.0246; 180-300: -0.0102 vs -0.0065 ± 0.0150, tol 0.0450 |
| D5k | kinetic energy (J), window mean | |c - ref| <= max(10% | 2 J, 3S) | pass | 0.17 | 30-90: 45.8909 vs 46.6832 ± 0.8133, tol 4.6683; 105-165: 15.6485 vs 16.8973 ± 4.3305, tol 12.9914; 180-300: 9.7427 vs 10.5524 ± 4.3377, tol 13.0131 |
| D5p | potential energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.07 | 30-90: 280.6283 vs 281.6301 ± 4.4590, tol 13.3770 (restored 301.9023 ± 0.8506, tol 6.0380); 105-165: 267.2202 vs 268.4620 ± 7.5629, tol 22.6888 (restored 273.4408 ± 4.5779, tol 13.7338); 180-300: 263.8716 vs 265.9184 ± 10.4288, tol 31.2864 (restored 269.7434 ± 5.3460, tol 16.0381) |
| D5p (as written) | potential energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.07 | 30-90: 280.6283 vs 281.6301 ± 4.4590, tol 13.3770; 105-165: 267.2202 vs 268.4620 ± 7.5629, tol 22.6888; 180-300: 263.8716 vs 265.9184 ± 10.4288, tol 31.2864 |
| D5e | total mechanical energy (J), window mean; band to the restored references | c within max(2%, 3S) of [ref, ref restored] | pass | 0.15 | 30-90: 326.5193 vs 328.3132 ± 3.8834, tol 11.6502 (restored 348.5854 ± 1.3347, tol 6.9717); 105-165: 282.8686 vs 285.3593 ± 11.8934, tol 35.6801 (restored 290.3381 ± 8.9084, tol 26.7252); 180-300: 273.6143 vs 276.4709 ± 14.4039, tol 43.2116 (restored 280.2958 ± 9.6837, tol 29.0511) |
| D5e (as written) | total mechanical energy (J), window mean | |c - ref| <= max(2%, 3S) | (pass, not gated) | 0.15 | 30-90: 326.5193 vs 328.3132 ± 3.8834, tol 11.6502; 105-165: 282.8686 vs 285.3593 ± 11.8934, tol 35.6801; 180-300: 273.6143 vs 276.4709 ± 14.4039, tol 43.2116 |
| D7d | surface profile: mean depth per 8-cell x bin, window mean, normalised L1; band to the restored references | d(c, [ref, ref restored]) <= max(0.05, 3S) | pass | 0.09 | 30-90: 0.0040 vs 0.0040 ± 0.0059, tol 0.0500 (restored 0.0067 ± 0.0055, tol 0.0500); 105-165: 0.0045 vs 0.0045 ± 0.0114, tol 0.0500 (restored 0.0043 ± 0.0113, tol 0.0500); 180-300: 0.0047 vs 0.0047 ± 0.0080, tol 0.0500 (restored 0.0048 ± 0.0080, tol 0.0500) |
| D7d (as written) | surface profile: mean depth per 8-cell x bin, window mean, normalised L1 | d(c, ref) <= max(0.05, 3S), S the largest pairwise d between references | (pass, not gated) | 0.09 | 30-90: 0.0040 vs 0.0000 ± 0.0059, tol 0.0500; 105-165: 0.0045 vs 0.0000 ± 0.0114, tol 0.0500; 180-300: 0.0047 vs 0.0000 ± 0.0080, tol 0.0500 |
| D8 | particles, window mean | |c - ref| <= max(10%, 3S) | pass | 0.09 | 30-90: 6.353e+5 vs 6.297e+5 ± 17318.6000, tol 62974.5100; 105-165: 3.823e+5 vs 3.841e+5 ± 26100.6000, tol 78301.8000; 180-300: 3.481e+5 vs 3.485e+5 ± 30193.3333, tol 90580.0000 |

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

Check-slot step time (informational): candidate fenced median 45.86 ms; references 67.60, 62.58, 66.20, 60.19 ms.

## letters: letters-ref-b: PASS

References letters-ref-a.jsonl, letters-ref-c.jsonl, letters-ref-d.jsonl, letters-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 240 steps with no solver failure | required | pass | 0.00 | 240: 240.0000 vs - ± 0.0000, tol 0.0000 |
| L1 | volume drift against the t = 0 pool plus the drops (fraction), one-sided on |drift| | |c| <= |ref| + max(0.005, 3S) | pass | 0.00 | 120: -0.0088 vs 0.0088 ± 0.0004, tol 0.0050; 180: -0.0082 vs 0.0094 ± 0.0004, tol 0.0050; 240: -0.0101 vs 0.0108 ± 0.0004, tol 0.0050 |
| L1 (as written) | volume drift against the t = 0 pool plus the drops (fraction) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.24 | 120: -0.0088 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0082 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0101 vs -0.0108 ± 0.0004, tol 0.0050 |
| L2 | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240); band to the restored references | d(c, [ref, ref restored]) <= max(1 cells, 3S) | pass | 0.20 | 60: 0.0174 vs 0.0174 ± 0.0364, tol 1.0000 (restored 0.2607 ± 0.0363, tol 1.0000); 90: 0.0805 vs 0.0805 ± 0.1444, tol 1.0000 (restored 0.4496 ± 0.1444, tol 1.0000); 120: 0.3047 vs 0.3047 ± 0.7010, tol 2.1030 (restored 0.4589 ± 0.7010, tol 2.1029); 180: 0.5680 vs 0.5680 ± 0.9631, tol 2.8893 (restored 0.6550 ± 0.9631, tol 2.8892); 240: 0.6198 vs 0.6198 ± 1.1402, tol 3.4206 (restored 0.7425 ± 1.1402, tol 3.4206) |
| L2 (as written) | shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; 1106 L dropped by 240) | d(c, ref) <= max(1 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.20 | 60: 0.0174 vs 0.0000 ± 0.0364, tol 1.0000; 90: 0.0805 vs 0.0000 ± 0.1444, tol 1.0000; 120: 0.3047 vs 0.0000 ± 0.7010, tol 2.1030; 180: 0.5680 vs 0.0000 ± 0.9631, tol 2.8893; 240: 0.6198 vs 0.0000 ± 1.1402, tol 3.4206 |
| L3 | kinetic energy (J) | |c - ref| <= max(10%, 3S) | pass | 0.23 | 60: 1948.0310 vs 1948.7573 ± 2.2087, tol 194.8757; 120: 2214.4730 vs 2245.8357 ± 101.6255, tol 304.8766; 180: 2950.8183 vs 3079.2708 ± 186.6585, tol 559.9755; 240: 3064.0237 vs 3036.4692 ± 33.2381, tol 303.6469 |
| L4 | liquid centroid height (cells); band to the restored references | c within max(0.25 cells, 3S) of [ref, ref restored] | pass | 0.14 | 60: 19.8935 vs 19.8923 ± 0.0020, tol 0.2500 (restored 20.0174 ± 0.0017, tol 0.2500); 120: 20.5037 vs 20.5591 ± 0.1323, tol 0.3970 (restored 20.7198 ± 0.1283, tol 0.3848); 180: 21.7442 vs 21.7236 ± 0.0407, tol 0.2500 (restored 21.8962 ± 0.0390, tol 0.2500); 240: 20.5151 vs 20.4963 ± 0.0097, tol 0.2500 (restored 20.7109 ± 0.0018, tol 0.2500) |
| L4 (as written) | liquid centroid height (cells) | |c - ref| <= max(0.25 cells, 3S) | (pass, not gated) | 0.14 | 60: 19.8935 vs 19.8923 ± 0.0020, tol 0.2500; 120: 20.5037 vs 20.5591 ± 0.1323, tol 0.3970; 180: 21.7442 vs 21.7236 ± 0.0407, tol 0.2500; 240: 20.5151 vs 20.4963 ± 0.0097, tol 0.2500 |
| L5 | particles | |c - ref| <= max(10%, 3S) | pass | 0.10 | 60: 5.009e+5 vs 5.007e+5 ± 6307.0000, tol 50065.5250; 120: 8.894e+5 vs 8.912e+5 ± 11988.0000, tol 89117.8500; 180: 8.948e+5 vs 9.038e+5 ± 18947.0000, tol 90377.4000; 240: 8.745e+5 vs 8.787e+5 ± 7187.0000, tol 87872.1750 |
| Ln | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.005, 3S) | (pass, not gated) | 0.24 | 120: -0.0088 vs -0.0088 ± 0.0004, tol 0.0050; 180: -0.0082 vs -0.0094 ± 0.0004, tol 0.0050; 240: -0.0101 vs -0.0108 ± 0.0004, tol 0.0050 |

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

Check-slot step time (informational): candidate fenced median 96.65 ms; references 123.19, 84.54, 87.50, 94.06 ms.

## paddleboats: paddleboats-ref-b: PASS

References paddleboats-ref-a.jsonl, paddleboats-ref-c.jsonl, paddleboats-ref-d.jsonl, paddleboats-ref-e.jsonl.

| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |
|---|---|---|---|---|---|
| C0 | completed 300 steps with no solver failure | required | pass | 0.00 | 300: 300.0000 vs - ± 0.0000, tol 0.0000 |
| P1 | volume drift (fraction of t = 0), one-sided on |drift| | |c| <= |ref| + max(0.0025, 3S) | pass | 0.13 | 150: -0.0030 vs 0.0030 ± 0.0003, tol 0.0025; 300: -0.0107 vs 0.0100 ± 0.0017, tol 0.0051 |
| P1 (as written) | volume drift (fraction of t = 0) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.13 | 150: -0.0030 vs -0.0030 ± 0.0003, tol 0.0025; 300: -0.0107 vs -0.0100 ± 0.0017, tol 0.0051 |
| P2 | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells); band to the restored references | d(c, [ref, ref restored]) <= max(0.25 cells, 3S) | pass | 0.13 | 150: 0.0098 vs 0.0098 ± 0.0176, tol 0.2500 (restored 0.0733 ± 0.0164, tol 0.2500); 300: 0.0328 vs 0.0328 ± 0.0537, tol 0.2500 (restored 0.2589 ± 0.0363, tol 0.2500) |
| P2 (as written) | surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells) | d(c, ref) <= max(0.25 cells, 3S), S the largest pairwise d between references | (pass, not gated) | 0.13 | 150: 0.0098 vs 0.0000 ± 0.0176, tol 0.2500; 300: 0.0328 vs 0.0000 ± 0.0537, tol 0.2500 |
| P3 | kinetic energy, the wheels' wake (J) | |c - ref| <= max(10%, 3S) | pass | 0.12 | 60: 95.4258 vs 95.1781 ± 1.7659, tol 9.5178; 120: 109.0367 vs 109.1192 ± 1.1784, tol 10.9119; 180: 141.6046 vs 141.8606 ± 1.5890, tol 14.1861; 240: 142.3728 vs 141.8698 ± 1.9409, tol 14.1870; 300: 165.6677 vs 167.6518 ± 1.3226, tol 16.7652 |
| P4 | particles | |c - ref| <= max(10%, 3S) | pass | 0.55 | 150: 2.393e+5 vs 2.613e+5 ± 13333.0000, tol 39999.0000; 300: 2.584e+5 vs 2.993e+5 ± 69580.0000, tol 2.087e+5 |
| Pn | NB-FLIP's own volume drift (readStats) | |c - ref| <= max(0.0025, 3S) | (pass, not gated) | 0.13 | 150: -0.0083 vs -0.0083 ± 0.0003, tol 0.0025; 300: -0.0160 vs -0.0154 ± 0.0017, tol 0.0051 |

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

Check-slot step time (informational): candidate fenced median 70.16 ms; references 51.19, 50.72, 54.85, 95.74 ms.
