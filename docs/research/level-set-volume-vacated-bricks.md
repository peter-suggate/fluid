# Figure 7 vacated-brick coarsening

Date: 2026-09-14.

The pre-fix evidence is
`artifacts/level-set-volume/cm12-figure-7-lsv-vacated-fine-bricks-frame0-30.json`.
It distinguishes normal release from a grading lock. Brick `2262` first has no
level-set contour at frame 6, coarsens from resolution 8 to 4 at frame 7, and
retires when its volume becomes exactly zero at frame 9. This is the intended
sequence.

The earliest isolated failure is at frame 15. Bricks `2246` and `2249` are
active at resolution 2 with only about `0.00012009` total fine-cell volume.
Their score is zero and plan reason 16 requests coarsening, and their active
neighbors permit resolution 1. They remain at resolution 2 because inactive,
zero-cell resolution-4 neighbors participate in the existing all-seed 2:1
closure. For `2246`, those pins are `2245` and `2262`; the mirror case is
`2250` and `2265` for `2249`.

Brick `2229` shows the same mechanism later. At frame 17 it is active at
resolution 2, has `1.8664e-6` fine-cell volume, no contour, no surface, thin,
predicted-face, or velocity-floor reason, and score zero. Plan reason 16 wants
resolution 1. Active neighbors `2213` at resolution 2 and `2230` at resolution
1 allow that request, while inactive resolution-4 seeds `2228` and `2245` pin
the closure. The brick remains at resolution 2 through frame 30 with about
`3.37e-8` volume. The failure is therefore not thin-fluid classification or a
surface-proof veto.

Retirement must remain exact: lifecycle validation rejects retiring a brick
with any nonzero conservative volume. A mostly vacated brick should first
coarsen while preserving its residue, remain active at resolution 1 while that
residue persists, and retire only after it reaches exact zero.

The minimal fix preserves the repository's all-seed grading invariant and is
enabled only for level-set-volume World resolution planning. Under that option,
an inactive, unfrozen brick requests `apply_regions(1, ...)` instead of
retaining its previous rung. Baseline planning retains its existing behavior.
Existing 2:1 closure raises that metadata rung only where an active or inactive
neighbor requires it. If transport later demands the page again, the existing
reactivation path overrides its request to resolution 8 before closure. This
changes no liquid thresholds, surface authority, transport pass, or topology
invariant.

## Validation

The post-fix capture is
`artifacts/level-set-volume/cm12-figure-7-lsv-vacated-bricks-coarsening-fixed.json`.
At frame 15, bricks `2246` and `2249` reach resolution 1 instead of remaining
at resolution 2, with the same `0.00012009000784` volume. At frame 17, the
later symmetric pair `2229` and `2234` likewise reaches resolution 1 instead
of resolution 2. The graph contains 1,586 rather than 1,592 cells at frame 15,
1,760 rather than 1,778 at frame 17, and 1,296 rather than 1,314 at frame 30.
The 18-cell frame-30 reduction removes the grading lock; it does not delete
the residual bricks, which correctly remain active at the coarsest resolution
while their conservative volume is nonzero. Normal vacated-fine hysteresis is
unchanged.

Final volume is `1252.000010349` from an initial `1252`, a relative drift of
about `8.27e-9`. Validation passed 31 native tests (14 resolution, 4 world
library, and 13 integration tests, including the unchanged baseline world
golden), all four Wasm flow tests including the 30-frame checkpoints, the Rust
release build, all Wasm builds, and the UI build. Served Wasm hashes matched the
current Rust source.
