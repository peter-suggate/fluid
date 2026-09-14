# Level-set-plus-volume native lab results

Date: 2026-09-14. Decision: **no-go for promotion beyond the advance lab**.

The level-set-plus-volume arm conserves physical liquid volume to its reported
floating-point bound, and Figure 7's transport stage is substantially faster
than baseline. It does not satisfy the capacity criterion: excess volume grows
again late in Figure 7 and reaches its run maximum on the last half-pool frame.
The three balancing passes also leave order-one receiver-row residuals. These
failures are sufficient for a no-go without a browser gate.

## Method

All arms used the native release `verify_world` runner, `dt = 1/30`, pressure
budget 256, and the same requested frame count for baseline and experiment.
Level-set-plus-volume used one whole-frame trace and zero material substeps;
baseline used its normal adaptive material microsteps. Runs were sequential,
with no Dawn, browser, other native verifier, or Rust build active.

- Base Git commit: `7f6a2cdc196c0703892400a31fd5aa45a84c08b7`; the measured source also
  includes the working-tree level-set implementation.
- Combined SHA-256 manifest fingerprint for `levelset_volume.rs`, `world.rs`,
  `numerics.rs`, `transfer.rs`, `lifecycle.rs`, and `lib.rs`:
  `7a466cf6a7c96327855f80888de6d9ff96a1b5e762ba4ba968f37bfc261b6d9a`
- Native binary SHA-256: `737078c825fb3f4c7442f6b457f86384b69ccff8b71c73d0dbbf2cd967172802`
- [Figure 7 baseline receipt](../../artifacts/level-set-volume/figure7-baseline-30.json)
- [Figure 7 level-set receipt](../../artifacts/level-set-volume/figure7-level-set-volume-30.json)
- [half-pool baseline receipt](../../artifacts/level-set-volume/half-pool-baseline-10.json)
- [half-pool level-set receipt](../../artifacts/level-set-volume/half-pool-level-set-volume-10.json)

`maximum excess ratio` is `V/C - 1`; add one to obtain the fill ratio. Excess
volume is physical fine-lattice area, summed as `max(V-C, 0)`.

## Comparison

| scene and arm | final relative mass drift | mean frame ms | mean transport ms | peak/final excess volume | peak/final excess ratio | peak/final over-capacity cells | maximum seam offset | final eligible seams | pressure iterations mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Figure 7 baseline, 30 | 1.393e-7 | 69.821 | 22.977 | 0 / 0 | 0 / 0 | 0 / 0 | 9.776 | 54 | 92.7 / 256 |
| Figure 7 level-set, 30 | -7.310e-9 | 63.822 | 6.102 | 419.002 / 419.002 | 11.572 / 11.324 | 452 / 438 | 2.373 | 0 | 89.0 / 256 |
| half-pool baseline, 10 | -1.417e-8 | 15.335 | 1.405 | 0 / 0 | 0 / 0 | 0 / 0 | 2.224 | 23 | 29.6 / 48 |
| half-pool level-set, 10 | 6.849e-10 | 25.168 | 1.581 | 27.390 / 27.390 | 1.341 / 1.341 | 102 / 102 | 1.349 | 16 | 47.4 / 83 |

Figure 7's level-set transport stage is 73% faster than baseline and its whole
frame is 8.6% faster. The half-pool transport stage is 12.5% slower and its
whole frame is 64% slower. Thus the wall-time criterion passes only for Figure
7. The aggregate seam maximum is lower in both level-set runs, but a same-frame
comparison is mixed. Figure 7 improves in 7 of the 13 frames where both arms
have eligible coarse/fine partial-plane pairs; half-pool improves in 4 of 7.
Figure 7 has no eligible pair at frame 30 (160 were skipped), so its final zero
cannot be treated as a perfect seam result.

The experiment reported no invalid phi samples in the half-pool. Figure 7
reported up to 32 invalid samples in a frame, despite returning to zero by the
last frame. Maximum trace Courant was 10.814 for Figure 7 and 2.199 for the
half-pool. Maximum receiver-row residuals were 7.007 and 3.836 respectively;
donor residuals stayed at floating-point roundoff because the last balancing
operation normalizes donors. The `zeroWeightDonors` diagnostic reached 240 in
Figure 7 and 18 in the half-pool. This counter is an uncovered-stencil census
before capacity filtering; it includes dry or zero-capacity graph cells and is
not evidence that positive-volume donor mass was lost.

## Per-frame level-set receipts

The full JSON receipts contain baseline rows for the identical frame numbers,
world timing splits, drift bounds, donor residuals, and seam mean/RMS values.
The compact tables below show the decision-driving fields. `seam` is eligible
comparison count / maximum absolute signed-plane difference.

### Figure 7

| frame | rel drift | over cells | total excess | max excess ratio | row residual | trace ms | V gather ms | phi gather ms | plane ms | RDF ms | pressure | seam | invalid phi |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.088e-9 | 118 | 7.890 | 0.169 | 0.169 | 0.082 | 0.100 | 0.010 | 0.279 | 2.054 | 0 | 42 / 1.139 | 0 |
| 2 | 4.610e-9 | 118 | 24.324 | 0.491 | 0.308 | 0.086 | 0.099 | 0.013 | 0.303 | 2.004 | 0 | 44 / 2.373 | 0 |
| 3 | -2.080e-9 | 170 | 40.351 | 0.415 | 0.465 | 0.076 | 0.093 | 0.009 | 0.304 | 1.923 | 0 | 62 / 2.242 | 0 |
| 4 | 1.468e-9 | 174 | 55.293 | 0.414 | 0.717 | 0.088 | 0.102 | 0.011 | 0.357 | 1.993 | 0 | 60 / 1.494 | 0 |
| 5 | -2.094e-9 | 160 | 70.480 | 0.893 | 1.017 | 0.084 | 0.103 | 0.009 | 0.386 | 2.155 | 0 | 56 / 1.329 | 0 |
| 6 | -3.985e-9 | 216 | 92.072 | 1.722 | 1.068 | 0.115 | 0.159 | 0.013 | 0.342 | 2.212 | 46 | 18 / 1.131 | 0 |
| 7 | -2.642e-9 | 214 | 93.934 | 1.722 | 1.150 | 0.120 | 0.150 | 0.013 | 0.380 | 2.357 | 53 | 18 / 0.684 | 20 |
| 8 | -2.971e-9 | 240 | 96.057 | 1.722 | 1.292 | 0.106 | 0.143 | 0.012 | 0.378 | 2.404 | 53 | 18 / 1.075 | 8 |
| 9 | -3.421e-9 | 260 | 96.015 | 1.722 | 1.000 | 0.117 | 0.131 | 0.014 | 0.383 | 2.360 | 53 | 16 / 0.596 | 16 |
| 10 | -4.521e-9 | 220 | 93.657 | 1.722 | 1.173 | 0.121 | 0.155 | 0.015 | 0.516 | 2.400 | 51 | 38 / 1.250 | 16 |
| 11 | -5.064e-9 | 204 | 87.522 | 1.722 | 1.316 | 0.132 | 0.159 | 0.016 | 0.664 | 2.513 | 51 | 40 / 1.182 | 20 |
| 12 | -2.765e-9 | 210 | 83.622 | 1.507 | 1.491 | 0.128 | 0.160 | 0.013 | 0.628 | 2.433 | 60 | 0 / 0.000 | 32 |
| 13 | -4.103e-9 | 321 | 83.246 | 1.507 | 1.775 | 0.158 | 0.197 | 0.017 | 0.748 | 3.062 | 76 | 0 / 0.000 | 0 |
| 14 | -2.043e-9 | 320 | 82.758 | 1.399 | 1.101 | 0.153 | 0.193 | 0.017 | 0.755 | 2.907 | 76 | 0 / 0.000 | 8 |
| 15 | -1.885e-9 | 317 | 81.872 | 1.389 | 1.273 | 0.164 | 0.202 | 0.017 | 0.749 | 3.259 | 69 | 0 / 0.000 | 2 |
| 16 | -9.160e-10 | 332 | 80.712 | 1.231 | 1.444 | 0.155 | 0.189 | 0.020 | 0.727 | 2.969 | 87 | 8 / 0.373 | 2 |
| 17 | -3.926e-9 | 308 | 80.067 | 1.200 | 1.655 | 0.198 | 0.207 | 0.018 | 0.787 | 3.109 | 86 | 6 / 0.498 | 0 |
| 18 | -3.922e-9 | 308 | 80.067 | 1.200 | 1.000 | 0.151 | 0.164 | 0.018 | 0.737 | 3.051 | 85 | 0 / 0.000 | 0 |
| 19 | -2.061e-9 | 319 | 79.417 | 1.124 | 1.191 | 0.172 | 0.195 | 0.016 | 0.726 | 3.077 | 82 | 0 / 0.000 | 0 |
| 20 | -2.954e-9 | 280 | 78.539 | 1.065 | 1.359 | 0.160 | 0.190 | 0.016 | 0.802 | 3.515 | 97 | 0 / 0.000 | 0 |
| 21 | -3.685e-9 | 338 | 82.677 | 0.999 | 1.544 | 0.168 | 0.198 | 0.017 | 0.724 | 3.292 | 94 | 0 / 0.000 | 0 |
| 22 | -6.260e-9 | 342 | 87.360 | 0.995 | 1.799 | 0.167 | 0.202 | 0.019 | 0.737 | 3.311 | 85 | 0 / 0.000 | 0 |
| 23 | -5.424e-9 | 356 | 84.954 | 0.968 | 1.101 | 0.169 | 0.225 | 0.023 | 0.755 | 3.570 | 87 | 0 / 0.000 | 0 |
| 24 | -6.982e-9 | 326 | 85.500 | 0.947 | 1.275 | 0.168 | 0.209 | 0.019 | 0.791 | 3.229 | 86 | 0 / 0.000 | 0 |
| 25 | -7.130e-9 | 423 | 104.921 | 0.913 | 1.447 | 0.186 | 0.214 | 0.020 | 0.666 | 3.234 | 88 | 0 / 0.000 | 0 |
| 26 | -6.279e-9 | 452 | 190.669 | 2.435 | 3.130 | 0.152 | 0.228 | 0.017 | 0.504 | 3.037 | 256 | 0 / 0.000 | 0 |
| 27 | -6.168e-9 | 433 | 266.275 | 4.074 | 3.692 | 0.137 | 0.209 | 0.013 | 0.460 | 2.810 | 216 | 0 / 0.000 | 4 |
| 28 | -5.180e-9 | 440 | 348.733 | 11.572 | 3.297 | 0.221 | 0.221 | 0.017 | 0.393 | 3.020 | 240 | 0 / 0.000 | 0 |
| 29 | -4.824e-9 | 427 | 400.035 | 8.198 | 7.007 | 0.137 | 0.199 | 0.014 | 0.372 | 2.698 | 238 | 0 / 0.000 | 0 |
| 30 | -7.310e-9 | 438 | 419.002 | 11.324 | 3.169 | 0.128 | 0.212 | 0.015 | 0.361 | 2.833 | 256 | 0 / 0.000 | 0 |

### Coarse-first half-pool

| frame | rel drift | over cells | total excess | max excess ratio | row residual | trace ms | V gather ms | phi gather ms | plane ms | RDF ms | pressure | seam | invalid phi |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.783e-10 | 0 | 0.000 | 0.000 | 0.166 | 0.051 | 0.076 | 0.005 | 0.088 | 0.891 | 28 | 0 / 0.000 | 0 |
| 2 | 3.762e-11 | 0 | 0.000 | 0.000 | 0.302 | 0.051 | 0.058 | 0.005 | 0.090 | 0.799 | 43 | 0 / 0.000 | 0 |
| 3 | 1.289e-10 | 60 | 4.863 | 0.095 | 0.455 | 0.046 | 0.054 | 0.005 | 0.142 | 0.774 | 37 | 16 / 1.209 | 0 |
| 4 | 2.891e-11 | 44 | 6.322 | 0.119 | 0.689 | 0.046 | 0.060 | 0.005 | 0.151 | 0.790 | 34 | 16 / 1.023 | 0 |
| 5 | 7.160e-10 | 42 | 4.285 | 0.093 | 1.013 | 0.042 | 0.053 | 0.005 | 0.153 | 0.800 | 35 | 30 / 1.013 | 0 |
| 6 | 9.598e-10 | 76 | 8.113 | 0.226 | 1.060 | 0.041 | 0.051 | 0.005 | 0.133 | 0.788 | 37 | 10 / 1.349 | 0 |
| 7 | 4.463e-10 | 82 | 14.052 | 0.305 | 1.136 | 0.040 | 0.050 | 0.005 | 0.114 | 0.914 | 37 | 12 / 1.249 | 0 |
| 8 | 9.523e-10 | 82 | 16.124 | 0.331 | 1.223 | 0.044 | 0.067 | 0.004 | 0.112 | 0.729 | 68 | 20 / 1.093 | 0 |
| 9 | 1.326e-9 | 86 | 16.343 | 0.326 | 3.836 | 0.046 | 0.080 | 0.004 | 0.103 | 0.768 | 72 | 10 / 0.931 | 0 |
| 10 | 6.849e-10 | 102 | 27.390 | 1.341 | 2.029 | 0.048 | 0.066 | 0.005 | 0.088 | 0.803 | 83 | 16 / 0.957 | 0 |

## Gate decision

| criterion | Figure 7 | half-pool |
|---|---|---|
| physical volume drift within reported roundoff | pass | pass |
| overcapacity bounded and decaying at end | fail | fail |
| seam offsets below same-frame baseline | fail: 7/13 paired frames improve; no final eligible pairs | fail: 4/7 paired frames improve |
| whole frame below baseline wall time | pass | fail |

The advance-lab option remains useful for visual inspection and further
research, but these measurements do not support making it a production
transport. The first numerical problem to resolve is the large receiver-row
defect and the resulting capacity accumulation. A sharper phi sampler is not
the next step yet: the experiment already fails its volume/capacity contract
before surface smoothing becomes the deciding issue.

## Validation context

- Native report utility: 3/3 tests passed; focused ESLint and diff checks passed.
- Core level-set numerical tests: 7 passed (one module test and six integration
  tests); baseline golden and cutover tests: 3 passed.
- Sustained scalar/SIMD flow tests: 4/4 passed, including Figure 7 frame 30 and
  half-pool frame 10 at the UI pressure budget of 256.
- The required sparse CM12 Dawn run completed with unchanged thresholds: 5 of
  17 lanes passed, 9 timed out, and 3 hit GPU assertions. No Dawn lane was run
  concurrently with these native measurements.
- The repository-wide TypeScript check still reports pre-existing errors in
  unrelated paths. No browser gate was run, as required by the lab plan.
