# Raster lighting pipeline trials

## Stage spotlight regression

The `sparse-cm12-ladder-long-dam` stage reproduces a 63–65 ms dry-render frame
at 1600×920 on M1 Max, without fluid simulation or water compositing. Two
forward/reverse batches give these GPU medians:

| Settings | First batch | Second batch |
| --- | ---: | ---: |
| Production Raster + AO | 64.55 ms | 62.91 ms |
| Shadows disabled, AO retained | 1.77 ms | 1.77 ms |
| AO disabled, shadows retained | 63.18 ms | 62.39 ms |
| Visibility disabled | 1.31 ms | 1.31 ms |

Raster AO caches directional sun shadows only. The stage has an additional area
spotlight, which falls back to full-resolution exact SVO visibility (two area
samples per illuminated pixel). It also fails the single-sun shader specialization
guard. This isolates the expensive shadow path; it does not separate traversal
cost from the generic shader's resource pressure. The screen-filling floor makes
the problem visible even without any visible voxel mesh.

The benchmark now supports explicit background-only scenes, verifies valid raster
background pixels and zero traced-primary substitution, and applies the supplied
camera during startup as well as measurement. Existing mesh proofs remain the
default. [Measurements and provenance](ladder-shadow-isolation.json) preserve the
camera, source hashes, settings and batch medians. These are the pre-fix results.

### Cached spotlight implementation

Raster AO now has perspective depth maps for up to four forward-facing
spotlights. The caster and receiver share the projection, and blocker depth is
linearized before estimating the emitter's contact-hardening penumbra. Filtering
is bounded to 12 blocker samples and 12 comparison samples, including a centre
sample for small blockers. Sun visibility and both AO estimators are unchanged.
Rigid bodies retain their analytic per-sample shadow correction.

Maps follow the existing GPU mesh/light invalidation receipt and are independent
of camera position. Each active spot adds one 2048² depth layer (16 MiB); sun-only
scenes keep their previous depth allocation. Adding or removing spots recreates
the array, rebuilds its bindings and invalidates all maps before sampling.

The opaque shader specialization now accepts light sets fully covered by cached
maps. Point/other area lights, additional directional lights, excess spotlights,
and cones too wide for the supported perspective projection keep exact shadows.
The producer and specialization share the same eligibility function, preventing
an unsupported light from silently losing its shadow fallback.

CPU type checking and all 883 unit tests passed. GPU image/cache regression tests
and the 1600×920 benchmark are awaiting the repository's exclusive GPU lease;
no post-change performance or image-quality claim is verified yet.

## Garden optimization trials

The baseline is smooth Raster AO with coarse voxel AO strength 0.6, at 1600×920
in hero-garden-hose-x10. These trials preserve the lighting algorithm and geometry.

Three independent changes:

- A full-resolution opaque single-sun shader removes the exact SVO shadow fallback
  from the compiled lighting closure. Draw-time capability checks retain generic
  shading for other lights, non-opaque scenes, and GI.
- Opacity-only maintenance leaves out radiance reduction, publication of its four
  lobes, and diffuse feedback. Radiance is withdrawn immediately. Returning to a
  consumer of radiance requests a complete ordered rebuild before publication;
  the renderer refreshes bindings when that capability changes. Atlas storage is
  retained, so this trial does not claim a memory saving.
- An opt-in static-scene shadow trial omits both cached depth passes when the host
  mesh receipt and lighting publication are unchanged. It remains disabled in
  production: delayed host receipts are not sufficient evidence for arbitrary
  GPU-only scene edits. Existing GPU invalidation remains the default.

Run independent-process diagnostic trials under the repository GPU lease (the
authoritative interleaved command is below):

```sh
node --import tsx tools/run-webgpu-exclusive.ts --import tsx tools/explore-raster-pipeline.ts
```

Results default to `/tmp/fluid-raster-pipeline`. Every arm runs in a fresh process,
measures whole-frame GPU spans and untimed fence-to-fence wall time, and exports
RGBA16F pixels for exact comparison. Each arm has two batches of 24 frames after
warmup. The baseline also measures four alternating pairs of full derived rebuilds
on the same world; these isolate maintenance from initial scene construction.

The harness uses refinement depth 1 to stay under its per-buffer safety limit.
It excludes the fluid solver and water optical composite. It therefore measures
renderer cost, not the full application frame at production refinement depth 2.
Per-pass timestamps may overlap and must not be summed to claim a frame saving.

GPU tests bound generic/specialized color differences to one FP16 step with exact depth, and cover opacity-only rebuild equality,
radiance restoration, camera movement, sun invalidation, AO switching, and depth
preservation in smooth and voxel-face modes. CPU tests cover capability fallback
and the experimental shadow-pass scheduling rules.

## Measured results

Apple M1 Max, smooth, 1600×920, refinement depth 1. No fluid simulation or water
optical composite. Coarse AO stays at the production default, 0.6.

| Trial | Stationary GPU frame | Reduction |
| --- | ---: | ---: |
| Baseline Raster AO | 4.719 ms | — |
| Single-sun specialization | 4.063 ms | 13.9% |
| Shadow-pass skipping only | 4.653 ms | 1.4% |
| Opacity-only maintenance | 4.719 ms | 0.0% |
| Currently enabled: specialization + opacity-only | 4.063 ms | 13.9% |
| All three, including experimental shadow skipping | 3.998 ms | 15.3% |

During the same scripted camera movement, baseline is 4.751 ms, the currently
enabled pair is 4.129 ms (13.1% reduction), and all three reach 4.063 ms (14.5%).
Stationary fence-to-fence wall time falls from 5.320 to 4.647 ms with the enabled
pair. These are renderer measurements, not full production application timings.

Opacity-only maintenance matters when derived data changes: warmed full derived
rebuilds fall from 12.386 to 7.143 ms (42.3%). This excludes source voxelization;
partial edits have different work counts. Both arms retain their atlas allocations.

The shader changes four of 1,472,000 stationary pixels, each by one FP16 color
step (maximum absolute channel difference 0.0001221). The moving capture differs
at three pixels (maximum 0.0004883). All depth channels match exactly. Shadow
skipping and opacity-only rebuilds preserve the stationary image byte for byte.

Keep specialization and opacity-only maintenance enabled. The 0.066 ms shadow
saving is too small to justify weakening live-edit invalidation; it stays opt-in.

### Measurement controls

The authoritative results alternate all arms in one process, in forward and
reverse order over four batches. Each batch has 48 warmup frames, 36 GPU-timed
frames, and 36 untimed fence-to-fence samples. Reported values are medians of the
four batch medians. The exact scene shader variant is explicitly awaited and
its selection asserted during every frame. Timestamp quantization is 0.065536 ms,
so the shadow gain is only one timing quantum, even though all four batches agree.

Earlier separate-process timings (5.83 ms baseline) and the first paired attempt
are superseded: they did not explicitly await the exact scene shader variant, and
one early paired arm was overwritten by asynchronous activation. They must not be
used to claim the initial apparent 28% gain.

[Measurements and provenance](measurements.json) include every batch median,
image comparisons, source hashes, adapter information, and raw derived rebuild
samples. First rebuild pair is excluded as warmup for both arms.

Reproduce the stationary paired run (hold the repository GPU lease):

```sh
FLUID_PROBE_DEPTH=1 FLUID_EXPLORE_RASTER_AO=1 FLUID_EXPLORE_PIPELINE_TRIALS=1 \
FLUID_EXPLORE_VIEWS=hero FLUID_EXPLORE_CYCLES=36 \
FLUID_EXPLORE_OUT=/tmp/fluid-raster-paired \
node --import tsx tools/run-webgpu-exclusive.ts --expose-gc --import tsx tools/explore-hero-render-budget.ts
```

Add `FLUID_EXPLORE_MOVING=1` for the matched camera trajectory.

## Verification status

- Type checking passed.
- Unit suite previously passed: 882 passed, 63 skipped.
- Targeted Dawn tests: 3 passed across both Raster AO files, after fixing the
  test to await shader activation and allow only one FP16 color step. Depth,
  opacity-only rebuild, and radiance restoration remain exact comparisons.
- Full Dawn suite remains incomplete; its previous run was blocked by another
  job's GPU lease. This measurement turn does not claim a clean-repository gate.

Changes remain uncommitted.
