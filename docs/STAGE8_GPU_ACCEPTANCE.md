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
advection, conservative donor-limited VOF transport, gravity and molecular
viscosity, periodic conservative remeshing, VOS rigid-body voxelization and
velocity blending, a solid-aware pressure solve, and projection. The pressure
hierarchy uses ghost-fluid and solid-fraction coefficients on every level, two
damped red-black Gauss-Seidel pre/post sweeps, one full cycle plus one V-cycle,
and the paper's collocated divergence and gradient definitions.

A compact atomic reduction reports raw VOF volume, front position, maximum wet
speed, maximum post-projection divergence, and maximum tall-cell height without
copying the 3D fields to JavaScript. Exact packed and cubic-equivalent
dimensions, compression, and allocated physics memory are shown in the UI.
There is no global volume correction.

Known limitations are persistent VOF rather than advected level set, fine-grid
rather than hierarchical velocity extrapolation, amortized rather than
per-frame remeshing, no in-solid `phi_s`, no resolved pressure traction, no
terrain cut cells, and no particle thickening. The UI reports post-projection
divergence rather than an asynchronously reduced linear residual. The native
Metal backend remains its existing uniform-grid implementation.
