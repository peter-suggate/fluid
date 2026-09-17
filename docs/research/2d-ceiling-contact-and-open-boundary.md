# Ceiling contact and open-boundary sticking

2026-09-17. Follow-up to the air-extension dissipation correction.

## Reproduction and first divergence

Scene `coarse-first-pool-impact-half`, 2D LevelSetVolume, no enforced regions,
dt 1/30 s, 256 pressure iterations, tolerance 1e-6. The original catalog
document has an **open** top, a 64×48 centre-Z slice, and 0.1 m finest cells.
The native reproduction matches the supplied screenshot at frame 61:
maximum face speed **56.88485**, with four negative top phi vertices and
surface maximum y=48.

The splash first wets the top at frame 42. Flow starts reversing at frame 47.
The first incoming characteristic at the wet vertex (1,48) is:

| Quantity, frozen frame-47 transport input | Value |
| --- | --- |
| Previous phi | −0.523996949 |
| Sampled velocity | (1.136802673, −0.038950324) |
| RK2 departure before domain clamp | (0.962824523, 48.000110626) |
| Actual clamped departure | (0.962824523, 48) |
| Sampled old phi | −0.560779512 |

`StaggeredVelocity2d::trace` clamps the departure into the domain, and
`sample_scalar` also clamps scalar queries. The incoming characteristic
therefore resamples old wet boundary data instead of exterior air. At an
open top, `released_walls` previously generated no scalar boundary data:
it recognized only separating `ClosedWorld` rows. Redistancing preserves
the wrongly retained sign. Closed side-wall liquid continuation also
participates at the corners, so the ambient-air condition must take
precedence there.

This is a scalar boundary-condition defect before sharpening. Pressure is
already producing downward velocity: by frame 61 the first open top face
has velocity −47.02054, yet its nearby boundary phi remains wet. Neither
more pressure iterations nor removal of conservative volume addresses the
first wrong scalar sample.

The defect is broader than this scene: a manufactured wet domain with
incoming velocity at an open face retained phi −4 where it should expose
an air gap. The regression failed on the original code and now passes for
all four 2D boundary orientations at widths 1, 2, and 4.

## Numerical correction

[`levelset_volume.rs`](../../rust/crates/fluid-core/src/levelset_volume.rs)
now constructs `IncomingAirBoundary` data for either a released solid wall
or an actual exterior open face with inward fluid velocity. Open faces are
one-term `SparseAir` rows lying on a domain boundary with positive aperture.
Interior sparse-allocation edges are explicitly excluded.

For an inward speed `s`, the incoming air supplies
`phi_air(x) = dt*s - inward_distance(x, boundary)`. Combining this with the
transported liquid phi using `max` supplies the exterior-air boundary
condition and its interior continuation. It uses the same mechanism already
validated for a released closed wall. Applying it after the tangential wall
continuation ensures air wins at shared corners, before redistancing.

Closed-wall release remains wall-relative. Open domain boundaries are
stationary; their speed is the physical fluid velocity obtained from the
accepted aperture-weighted face field, matching the staggered sampler.
Outflow, stationary faces, zero-dt transport, closed apertures, and explicitly
prescribed liquid inflows do not acquire an ambient-air condition.

The correction changes phi's boundary transport, not liquid pressure
membership or conserved V. It does not implement a new open-boundary mass
outflow ledger; the existing conservative remap still retains material.
Explicit open-domain outflow/re-entry modelling is a separate contract.

## Closed tops requested for all catalog scenes

Following the user's instruction, all **85 catalog factories**, their
variants, and newly created empty documents now specify closed simulation
tops. Existing saved/imported documents retain their authored boundary
settings and benefit from the open-boundary correction when run in 2D.

The changes are in `scenes.ts`, `empty-scene.ts`, `garden-scene.ts`, and
`hero-garden-scene.ts`. Ordinary tank shell generation now creates the top
solid on the final scene lattice. Tests explicitly verify the physical
ceiling for all three pool-impact resolutions and after tank resizing.
Scene tests that previously required an open top now assert the requested
closed-top contract; numerical tolerances were not relaxed. The settled tank's
two drop bodies now start at y=0.6 m inside its 0.8 m ceiling, above its
unchanged waterline. Their old y=1.18/1.34 m positions were above the new lid.

Closed tops change physical trajectories. In particular, the falling splash
now strikes a solid ceiling before reversing, instead of crossing the open
domain boundary. These runs are not numerical before/after controls for the
open-boundary correction. Bodies dropped into closed tanks must start inside
the ceiling rather than above it. The 3D catalog receives these scene settings;
the Rust scalar correction itself is 2D and is not a new GPU 3D port.

## Measurements

Coordinates are finest-cell units, with ceiling y=48. The closed run uses
the existing closed-wall separation algorithm; it requires no new ceiling
pressure adjustment.

| Run | Frame | Wet top vertices | Highest surface y |
| --- | ---: | ---: | ---: |
| Original open top | 47 | 4 | 48.000000 |
| Corrected open top | 47 | 2 | 48.000000 |
| Corrected open top | 48 | 0 | 47.737408 |
| Original open top | 61 | 4 | 48.000000 |
| Corrected open top | 61 | 0 | 34.494400 |
| Closed top | 61 | 0 | 35.906914 |
| Original open top | 90 | 4 | 48.000000 |
| Corrected open top | 90 | 0 | 27.631041 |
| Closed top | 90 | 0 | 31.004095 |
| Corrected open top | 120 | 0 | 34.606041 |
| Closed top | 120 | 0 | 35.159523 |

Maximum absolute conserved-volume drift: 1.68146e-5 in the original and
corrected open runs, and 1.40791e-5 in the closed run. These are area units,
not relative percentages. No volume is deleted by the boundary correction.

## Validation and reproduction

- Core library: **205 passed**, 1 ignored, 2 long CPU tests filtered out.
- Focused integration: **13 passed** across air extension (5), adaptive
  symmetry (2), existing closed-ceiling release (1), hydrostatics (3), the
  new open/closed pool test (1), and split-resolution wall contact (1).
- Scene contract/shell/shape/readiness/extent/hillside tests: **21 passed**,
  1 skipped. The new catalog test includes every registered variant.
- Scalar, SIMD, and threaded WASM builds and artifact checks pass. The
  dedicated probe runs both top modes through frame 61 in each variant,
  verifies contact followed by release, and produces exactly matching
  recorded results across all three variants. Native/WASM closed-run
  surface maximum differs by 0.0001564 finest cells at frame 61; open-run
  maxima agree exactly. Served UI binaries match the rebuilt files.
- The full isolated canonical Dawn gate completed in **365.32 s / 480 s**:
  **7/17 passed**. The five previously recorded timeout lanes remain
  (symmetric expansion, topology page budget, clipped transfer, mini32
  correctness, mini32 performance), as do the four previously recorded
  failing lanes (mixed-region surface, long-dam far wall, hillside far wall,
  outside-tank collapse). The additional rigid-coupling failure identified
  the above-lid body placement; its focused rerun is recorded separately.
  Mini64 performance passed at 53.805 ms versus its 110 ms ceiling. No lane
  limits were changed. The full gate is not green.
- After correcting the settled-tank body positions, the isolated
  `live-rigid-body-coupling` lane **passes** in 23.34 s with its original
  drop-distance, buoyancy-ordering, mass, and finite-field assertions.
  `dawn-rigid-corrected.json` records that rerun; it does not replace the
  full-run receipt or establish a fresh full-suite pass. The new body-placement
  unit test also passes.

From the repository root:

```bash
cargo test --manifest-path rust/Cargo.toml -p fluid-core --release --test levelset_volume_pool_ceiling --test levelset_volume_ceiling_separation
node --import tsx --test lib/core/scene-ceilings.test.ts
node --import tsx tools/wasm/ceiling-levelset-probe.ts /tmp/ceiling-wasm.json
cargo run --manifest-path rust/Cargo.toml -p fluid-core --release --example investigate_boundary_transport -- artifacts/level-set-volume/ceiling-release/frozen-frames46-47.jsonl 47
```

The new pool regression runs both boundary modes for 90 frames, verifies
that contact actually occurred, checks detachment at screenshot frame 61,
and checks finite phi and conservative volume throughout. The analytic
tests separately cover axis/rung independence, exact normal displacement,
partial apertures, source precedence, zero timestep, and interior sparse
edges. The existing closed-ceiling test also checks that pressure does not
project released wall velocity into solids.

Evidence is in the ignored local directory
[`artifacts/level-set-volume/ceiling-release/`](../../artifacts/level-set-volume/ceiling-release/):
`measurements.json`, `original-open-scene.json`, `frozen-frames46-47.jsonl`,
`frozen-traces-before.jsonl`, and validation logs. The checked-in fixture
[`half-pool-ceiling-seed.json`](../../rust/core/testdata/half-pool-ceiling-seed.json)
contains the closed catalog scene; the test removes its roof for the explicit
open control. Preserve the local raw evidence when transferring the work.
