# Quarter-pool transition energy: root-cause corrections

2026-09-17. Scene `coarse-first-pool-impact-quarter`, native 2D level-set +
volume, dt=1/30, 256 pressure iterations, tolerance 1e-6.

## What is implemented

1. **Correct physical Y boundaries.** Reduced solver bricks retain source Y,
   but `production_scene.rs` had reflected the top/bottom boundary flags as if
   they were canvas coordinates. The floor is now closed and the authored open
   top is open. The constructor golden changes only the pool's row-kind hash.
2. **Reconstruct inserted faces from accepted faces.** Native 2D, native 3D and
   GPU generation transfer now use normal-linear staggered interpolation,
   constant tangentially within each donor cell. Existing coplanar faces retain
   geometric flux restriction. Interpolating cached cell velocities instead
   filtered the field again and made refinement depend on a different authority.
3. **Preserve released wall velocity in native collocation.** A separating
   closed-world face carries its accepted fluid velocity. Treating it as solid
   velocity disagreed with the GPU convention.

There is **no energy guard, momentum budget, coarsening veto based on energy,
velocity rescaling, or modified adaptivity threshold**. The provisional guard
was removed at the user's direction. Frame 5 still executes the original merge.
Missing represented face support rejects transfer. Only explicitly declared new
air can initialize zero before ordinary velocity extension; there is no
cell-mean replacement for missing face data.

## Why frame 5 coarsens

The surface representability and quiet-epoch rules permit the ball's next
coarser rung. Frame 3 merges 352→88 cells; frame 5 merges 88→38. Later transport
support and thin-gap demand refine the representation again. Those decisions
are unchanged by this correction; see `quarter-pool-frame5-coarsening.md`.

## Measured result with no guard

All numbers below bracket a topology transfer with **no gravity, transport or
pressure evolution** between the samples. K is recomputed from current face
velocities, liquid amount and fresh cell collocation; units are fine-grid units
with unit liquid density, not joules.

| Frame | Cells before → after | Original K change | Corrected K change |
| --- | --- | ---: | ---: |
| 3 | 352 → 88 | -3.828% | -0.03824% |
| 5 | 88 → 38 | -15.647% | -2.48849% |
| 7 | 48 → 192 | +5.509% | -0.45691% |
| 12 | 168 → 360 | +0.866% | -0.34322% |

Later frames follow different trajectories, so this table compares the transfer
jumps, not identical physical states. Frame-5 accepted K is now **2474.4036**
(previously 2089.2046). Every diagnostic transfer replays bit-exactly against
production publication. Mass remains conserved to floating-point roundoff.

These are kinetic diagnostics, not a proof of total mechanical-energy loss.
At corrected frame 5, cell-lumped potential energy rises by 86.7664 while K
falls by 63.1468, so their sum rises by 23.6196. Coarsening also changes the
represented liquid centroid. A consistent subcell energy/centroid definition
is needed before calling these representation changes physical dissipation.

## Root cause and the remaining restriction loss

Originally, the cell centered at (18,22) gathered liquid velocity -8.15869 in Y.
Its lower face carried -8.15816, but its upper face at Y=24 was incorrectly
closed and carried zero. Collocation nearly halved the cell speed. This cell
and its mirror explained 84% of the gather-to-publication loss in that event.
Correcting the physical boundary removes that artificial damping.

The remaining 2.49% at frame 5 is **not eliminated** by these changes. For
example, the coarse cell centered at (18,14) contains 11.40988 liquid units.
Its gathered liquid velocity is -8.15294, while its retained lower face at Y=12
averages -7.90787 and -7.18349 to -7.54568. Its upper face carries -8.15274.
Coarse cell-center collocation therefore yields -7.84921. The coarser field
spans falling liquid and the slower extension in the air gap, and no longer
contains the interior face that represented the liquid's faster motion.

This is a restriction/reconstruction error, distinct from the corrected wall
bug and inserted-face filtering. The mass-weighted cell gather preserves
momentum, but publishing velocities from geometrically restricted faces need
not preserve that liquid momentum. It should not be called physical viscosity,
and the entire residual should not be attributed to liquid velocity variance.
Preserving both the original face-flux constraints and the lost interior motion
would require retaining additional velocity information or changing the
representation. An energy acceptance threshold does not solve that problem.

## 2D/3D alignment and verification

The physical open-top convention was already correct in 3D. Native 2D/3D and
GPU generation transfers now share the normal-linear face reconstruction;
GPU in-place transfer already used this convention. Analytic coarse→fine→coarse
fixtures preserve the original face flux, including all three axes and clipped
GPU cells, independently of cached cell velocity. No energy guard remains in
any native or GPU planner.

After removing the guard: 199 native unit tests passed (1 ignored), all 3
new integration tests passed, all 3 GPU generation-transfer tests passed, and
all 7 CPU generation-transfer tests passed. Scalar, SIMD and threaded Wasm
artifacts were rebuilt and passed strict feature validation. The repository
TypeScript check still reports errors outside the changed files.

The final no-guard rerun was stopped when the live browser resumed during
the frame-91 investigation, to respect the exclusive GPU-test requirement.
It is not a completed passing gate.

The broad Dawn suite was run during implementation with its thresholds
unchanged and failed (6/17 lanes passed; seven timeouts and four correctness
failures). Baseline checks with the original GPU files reproduced the exact
mixed-region height error and terrain-connectivity halt, plus an outside-drop
connectivity halt and mini32's performance-ceiling failure. The 3D pool also
fails density symmetry starting at step 1 with the provisional guard both on
and off. Broad end-to-end GPU validation is therefore **not green**; these
changes do not claim to resolve those failures.

The discarded guard also added measurable GPU planning cost. Its measurements
and results do not describe the final implementation, which removes that code.

## Reproduction and artifacts

```sh
cargo run --release --manifest-path rust/Cargo.toml -q -p fluid-core \
  --example investigate_transition_energy -- /tmp/fluid-quarter-pool.json 30
cargo test --release --manifest-path rust/Cargo.toml -p fluid-core --lib \
  --test levelset_volume_transition_energy
```

The scene is checked in at `rust/core/testdata/quarter-pool-transition-seed.json`
(as `{ "scene": ... }`, suitable for the example).

- Original diagnostic: `artifacts/level-set-volume/quarter-pool-transition-energy.json`
- Final no-guard diagnostic: `artifacts/level-set-volume/quarter-pool-transition-energy-fixed.json`
- Implementation-time gate: `artifacts/level-set-volume/transition-motion-dawn-gate.json`

## Frame-91 depth follow-up

Before removing the material floor, a no-guard native replay had 96 cells at frame 91. Its four bottom bricks
(keys 24–27, left to right) are B2/B2/B4/B4. Their velocity-travel measurements
are 0.15284, 0.25054, 0.21031 and 0.21643 fine cells per frame. The B2/B4
threshold is 0.33333; all four therefore require at most B2 from that signal.
The two B4 bricks each have one proof epoch; the policy requires two before
demotion. No capacity fault or scheduling-budget deferral is reported.

The unconditional B2 material floor has now been removed from native 2D
`resolution.rs` and the 3D resident GPU planner. Native 3D already had no
material floor. A regression verifies fully submerged direct-level-set liquid
can demote from B2 to B1. The updated scene replay reaches 36 cells at frame
91, with bottom bricks B2/B1/B2/B2; no energy guard was introduced.
Deep-liquid classification clears ordinary velocity-range demand, but the
subsequent approaching-flow measurement can reintroduce a motion floor.

The user's all-fine depth screenshot could not be matched to this current
replay. The live browser had reset to frame 0 and then showed an unrelated
`menu is not defined` error during concurrent UI edits. An earlier loaded
binary/run is plausible, but the screenshot's exact per-brick receipt was not
available; do not claim its blocker was conclusively identified.

Receipt: `artifacts/level-set-volume/quarter-pool-frame91-no-guard.json`.

Updated B1-enabled receipt: `artifacts/level-set-volume/quarter-pool-frame91-b1.json`.

With the floor removed, the GPU hydrostatic lane reports
`hydrostatic.deepMaximumResolution = 1` and `deepFineSamples = 0`,
confirming deep 3D liquid reaches B1 as well. At frame 5 the 2D scene has
26 cells instead of 38 with identical collocated kinetic energy
(2474.4035927); the mass and energy assertions remain unchanged.

B1-removal validation: 200 native unit tests and 3 transfer/scene integration
tests passed; scalar, SIMD and threaded Wasm rebuilt and validated. The full
Dawn gate completed with 8/17 lanes passing; it is not green.
Receipt: `artifacts/level-set-volume/b1-dawn-gate.json`.
