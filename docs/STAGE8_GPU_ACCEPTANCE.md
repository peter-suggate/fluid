# Stage 8 — WebGPU Eulerian Interactive Path

The verified CPU MAC solver remains the binary64 oracle. The interactive GPU
path uses WebGPU `f32` storage textures at substantially higher resolution. It
uses a collocated velocity/pressure/volume-fraction layout rather than claiming
bitwise identity with the MAC oracle. A weighted-Jacobi pressure solve is the
GPU baseline; the CPU PCG solve remains the acceptance reference.

One encoded step performs gravity plus a hydrostatic column-height predictor,
semi-Lagrangian backtrace, conservative volume-fraction advection, 24–40
quality-dependent pressure iterations,
projection, and closed-wall normal velocity enforcement. The height-gradient
predictor supplies the free-surface pressure acceleration that a low-iteration
collocated solve otherwise under-resolves; it is explicitly an interactive
approximation. Physics textures remain on the GPU and are sampled
directly by the renderer; animation does not require a per-frame field
readback.

A compact atomic GPU reduction measures volume, front position, and maximum
speed without copying 3D fields to JavaScript. A bounded global
volume-fraction correction restores the initial VOF integral after each accepted
step; raw transport drift and corrected drift are separate diagnostics.

Stable texture views and bind groups are cached. Advection and projection use
fixed ping-pong roles without full velocity copies. A dedicated 2D reduction
builds the column-height field once per substep. The finite-volume transport is
substepped to a three-dimensional CFL bound. Optional timestamp queries report
the complete GPU simulation step.

Quality presets allocate approximately 110k, 500k, or 1.2m cells for the
default tank. Exact effective dimensions and allocated physics memory are
reported. `f16` is not used.

Known limitations: the collocated Jacobi baseline is more dissipative than the
CPU MAC/PCG oracle, may show pressure checkerboarding, and currently reports a
fixed iteration budget rather than an asynchronously reduced residual. It is
the high-resolution interactive path, not the validation oracle.
