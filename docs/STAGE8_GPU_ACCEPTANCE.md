# Stage 8 — WebGPU Eulerian Interactive Path

The verified CPU MAC solver remains the binary64 acceptance oracle. The
interactive GPU path uses WebGPU `f32` storage textures and is not claimed to be
bitwise identical to the staggered MAC discretization.

Each x/z column stores one variable-height bottom cell, represented by bottom
and top samples, plus a band of cubic cells around the free surface. Balanced,
high, and ultra request approximately 2,500, 7,000, and 12,500 surface columns
with minimum bands of 24, 32, and 40 layers. The band expands to the uniform
limit when a vertical interface cannot fit; otherwise horizontal resolution is
independent of full water depth.

One encoded step performs velocity extrapolation, bounded MacCormack velocity
advection, conservative semi-Lagrangian surface-density transport, paired gamma
diffusion, bounded excess-density expansion, gravity and molecular viscosity,
periodic conservative remeshing, VOS rigid-body voxelization and
velocity blending, a solid-aware pressure solve, and projection. The pressure
hierarchy uses ghost-fluid and solid-fraction coefficients on every level, two
damped red-black Gauss-Seidel pre/post sweeps, one full cycle plus two V-cycles,
and a high-precision shared-memory RBGS top solve. Divergence remains
collocated; the pressure gradient uses the physical two-cell interior sample
span instead of the paper's printed one-cell denominator.

A compact atomic reduction reports raw surface-density mass, capacity-clamped
represented volume, front position, liquid and air speed extrema, divergence before and after projection, exact pressure
residual, pressure maximum, component CFL, finite-state count, extrema
locations, and maximum tall-cell height without copying the 3D fields to
JavaScript. Exact packed and cubic-equivalent dimensions, compression,
allocated physics memory, encoded time, and clock lag are shown in the UI.
There is no global volume correction.

Known limitations are persistent surface density rather than advected level set, fine-grid
rather than hierarchical velocity extrapolation, amortized rather than
per-frame remeshing, no in-solid `phi_s`, no resolved pressure traction, no
terrain cut cells, and no particle thickening. The native Metal backend remains
its existing uniform-grid implementation.
