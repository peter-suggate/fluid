# Figure 7 level-set-volume coarse-cell loss

Date: 2026-09-14.

The pre-fix evidence is
`artifacts/level-set-volume/cm12-figure-7-lsv-resolution-frame0-6-pre-translation-invariant-fix.json`.
The apparent disappearance is a resolution promotion, not loss of liquid.
Liquid measure remains `1252` initially and `1251.99999929` at frame 6.
In this 2-D harness, width-4 cells belong to resolution-2 bricks. There are
eight such cells initially, 16 at frames 2--5, and none at frame 6 because the
four occupied resolution-2 bricks with keys `2215`, `2216`, `2231`, and `2232`
are all scheduled at resolution 4.

The earliest wrong diagnostic is absolute velocity travel. In
`rust/crates/fluid-core/src/world.rs`, post-transport resolution sets
`translation_invariant_motion_sizing` for cellwise remap only. Level-set-volume
therefore enters the coarse-first fallback in
`rust/crates/fluid-core/src/resolution.rs`: every wet cell contributes
`dt * |velocity|`, including rigid free fall. That value becomes the measured
velocity floor and then the required coarse-first resolution.

For Figure 7, `dt = 1/30 s`, finest-cell size is `0.05 m`, and gravity is
`10 m/s^2`, or `200` fine cells per second squared. The coarse-first travel
thresholds are approximately `0.333`, `0.667`, `1.333`, and `2.667` fine cells.
Free-fall travel grows by about `0.222` fine cells per frame, reaching the
resolution-4 threshold at frame 6. The four affected receipts then report only
`occupied` and `velocity-floor`, score byte `128`, and plan reason
`required-resolution-floor`; each requests and schedules resolution 4. This is
the first cause of the reported topology change. It does not depend on volume
diffusion or on whether volume or level set supplies the visible surface.

The minimal fix is to use the existing geometric-transport policy for both
single-gather methods:

```rust
policy.translation_invariant_motion_sizing = cellwise_remap || level_set_volume;
```

That policy measures interface motion relative to the brick's liquid-weighted
mean, so uniform fall does not refine the whole interior. It also uses the
existing symmetric interface support masks during page planning. The
pre-transport projected-support planner already enables these semantics for
both cellwise remap and level-set-volume, so this change makes the later
resolution decision consistent with the support decision. It requires no new
pass, threshold, or surface authority.

## Validation

The matching post-fix capture is
`artifacts/level-set-volume/cm12-figure-7-lsv-resolution-frame0-6-fixed.json`.
The coarse bulk score at frame 1 is `0`, down from `21` before the fix, and no
affected coarse bulk brick reports `velocity-floor` at frame 3. At frame 6, the
four affected resolution-2 bricks all retain score `0` and remain at resolution
2, preserving all 16 width-4 cells; the pre-fix capture had none. Liquid
measure is `1251.99999904` against the initial `1252`.

Curvature and velocity variation remain active sizing inputs. The change
removes uniform absolute translation only: neighbor velocity differences,
deviation from a brick's liquid-weighted mean, and relative approach between
bricks can still request finer resolution. Absolute motion continues to drive
swept page activation.

Validation passed 17 focused native tests and all four Wasm flow tests,
including the frame-6 width-4 regression and the 30-frame Figure 7 flow. The
Rust release build and scalar, SIMD, threaded, and UI production builds passed.
Served hashes matched their build outputs and the recorded Rust source
fingerprint.
