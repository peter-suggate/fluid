# WebGPU octree Phase 8 device-variant audit

Date: 2026-07-25  
Target: 32 GiB Apple M1 Max (`MacBookPro18,2`), Dawn/Metal

## Reduction width

The lane-width audit alternated 64- and 128-lane variants for 21 samples.
Each sample executed 200 reduction/finalize pairs over 6,912 rows after three
warmups per variant.

| variant | median per reduction/finalize pair |
| --- | ---: |
| 64 lanes | 18.844165 µs |
| 128 lanes | 17.303125 µs |

The measured 128-lane win is 8.18%. Production therefore contains only the
128-lane M1 Max shader. The 64-lane measurement is retained as benchmark
evidence, not as executable shader code.

The target Dawn/Metal adapter advertises `subgroups`; production compiles one
subgroup WGSL module and fails closed if that feature or the 128-lane
workgroup-memory limits are absent. The shader retains f32 authoritative
buffers, compensated partial layout, validation flags, stopping threshold,
indirect-tail zeroing, and fail-closed publication. There is no portable or
64-lane shader fallback.

## f16 decision

`shader-f16` is enabled when advertised, but Phase 8 does not use f16 storage.
The evaluated candidates were rejected:

- Fine phi has no separate far-band, non-authoritative allocation. Packing the
  existing field would change the zero-crossing authority.
- Coarse multigrid residuals affect the preconditioner and therefore PCG
  iteration/stopping behavior. There is no isolated residual, volume, and
  energy differential gate that would justify changing their precision.
- Current fine summaries feed coarse-phi/topology and energy consumers; they
  are not purely observational storage.

Pressure, fine-interface phi, face velocity, diagonals, residuals, cut
coefficients, and all reduction partials consequently remain f32. This is a
rejection of the precision optimization, not a placeholder f16 path.

## Pressure/multigrid page-shape gate

The target M1 Max/Dawn Metal page benchmark alternated 4x4x4 and 8x8x4 across
21 samples after three warmups. Each sample executed 40 four-sweep local
Chebyshev stages over 256 pages with the production five-channel footprint:
one resolved-slot u32 halo plus A/B/RHS/diagonal f32 halos.

| shape | useful cells | one-cell halo | halo amplification | shared bytes | 32 KiB occupancy proxy | median/page | 6,912-row estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4x4 | 64 | 216 | 3.375x | 4,320 | 7 | 0.025600 us | 2.764800 us |
| 8x8x4 | 256 | 600 | 2.34375x | 12,000 | 2 | 0.070400 us | 1.900800 us |

Although 8x8x4 costs more per page, it carries four times the useful work. At
fixed row count it is 31.25% faster in this repeated-local-smoother gate and
loads 30.56% less halo per useful cell. Production therefore contains only
the 8x8x4 pressure/MG shape with a 128-lane workgroup. The 4x4x4 pressure/MG
shape is rejected; 4x4x4 remains only the independently owned fine-phi shape.
