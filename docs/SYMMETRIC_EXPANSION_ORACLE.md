# Symmetric expansion oracle

This scene is a deliberately small, exact horizontal-symmetry test for the
octree liquid solver. It is a red test while any directional bias remains.

## Scene

- Domain: `32 x 16 x 32` cells at `0.05 m` per cell.
- Liquid: an exactly centred `2 x 1 x 2` block of 8-cell solver bricks.
- Boundaries: closed, free-slip, with equal one-brick gaps to the four
  horizontal walls.
- Forcing: gravity only; no inflow, rigid bodies, or surface tension.
- Time step: exactly `0.004 s`.

The initial state is exactly invariant under reflection in x, reflection in z,
and interchange of x and z. Those three generators cover all eight horizontal
symmetries of the square (D4).

At every checkpoint the oracle compares:

- volume fraction as a scalar;
- collocated velocity with the matching vector transformation;
- compact pressure, expanded over each octree leaf;
- octree leaf size/topology;
- first contact with each of the four horizontal walls.

Both one-sided sparse support and unequal values are failures. The report names
the first failing step, transformation, component, and cell, so later errors do
not obscure their origin.

## Dawn lanes

Run the coarse-only diagnostic for the full expansion:

```sh
npm run test:webgpu:symmetric-expansion
```

Run the production fine-level-set path for one step:

```sh
npm run test:webgpu:symmetric-expansion:fine
```

The commands exit non-zero while any enforced field is not bitwise symmetric.
The first lane uses fine factor 1 for 250 steps and also requires all
four walls to be reached on the same step. The second uses fine factor 4 for one
step. Before that step it now hard-gates both the complete fine-phi publication
and the emitted position/normal vertex set at exact D4 symmetry, so subsequent
simulation stages cannot hide a construction defect.

## Frame-zero result

The factor-4 Dawn lane now publishes a bit-exact symmetric frame zero:

| Authority | Samples compared | Support mismatches | Value/set mismatches | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| fine SPGrid phi | 933,852 | 0 | 0 | 0 |
| raster position/normal set | 155,235 | n/a | 0 | 0 |

The raster has no narrow slits or enclosed holes. Its front-view terrace-edge
count fell from hundreds to 15; the remaining screen-depth statistic is
view-dependent, while the underlying geometry and normals pass the exact D4
set comparison.

Four construction defects were responsible:

1. Explicit brick seeds were converted to binary occupancy and then
   redistanced, discarding their exact analytic box SDF.
2. Even-sized leaf planes sampled `floor(centre) + 0.5` instead of the actual
   geometric centre, introducing a positive-axis half-cell bias.
3. The cold fine grid received one affine leaf plane at box corners instead of
   the exact rectangular-union SDF. Exact box bounds are now carried in the
   fine seed ABI and evaluated in fine-lattice coordinates.
4. The closest-point fraction codec used `2^24 - 1`, which cannot represent an
   exact half-cell crossing. It now uses a `2^24` scale with a clamped endpoint.
   The renderer also recognizes exact half-cell box faces, edges, and
   trihedral corners instead of letting its fixed tetrahedral diagonal chamfer
   them asymmetrically.

## Current first dynamic divergences

On the Dawn Metal path used during implementation:

| Lane | First volume | First velocity | First pressure | First topology | Wall-contact steps (-x, +x, -z, +z) |
| --- | --- | --- | --- | --- | --- |
| factor 1 | step 13, `0.052 s` | step 1 | step 1 | step 16, `0.064 s` | `67, 63, 67, 64` |
| factor 4 | step 1 | step 1 | step 1 | exact through step 1 | not evaluated in the one-step lane |

After the frame-zero fixes, the factor-4 first-step maximum discrepancies are
about `8.57e-5` in volume, `3.80e-7 m/s` in velocity, and `0.0103` in pressure;
fine phi first loses symmetry after transport/redistance with a maximum error
of about `0.00448 m`. This is a reduction of more than three orders of
magnitude in the former first-step volume defect (`0.3804`), and isolates the
next failure to the common dynamic operator/transport path rather than cold
surface construction.

With fine factor 1, the initial volume and topology remain exact, but the first
projected state differs by about `3.65e-7 m/s` in velocity and `0.0100 Pa` in
pressure. Tightening the pressure solve tolerance reduces those two values but
does not change the later volume/topology divergence or the four-step wall
spread. That is evidence of a second directional bias in the coarse transport or
operator path, rather than merely insufficient pressure convergence.

## Paper correspondence

Section 5 describes a fine SPGrid narrow band at 4x or 8x the background
resolution, dynamic topology, fast marching, and using valid fine values to
correct the coarse level set. That is the path isolated by the one-step factor-4
lane.

Section 4.3 requires the hybrid multigrid preconditioner to be symmetric and
specifically follows its first-order V-cycle correction with a second boundary
smoothing sweep to ensure symmetry. Pressure/operator work should preserve that
property independently of convergence tolerance.
