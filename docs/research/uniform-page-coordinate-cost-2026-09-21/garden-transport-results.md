# Garden transport support

Method: **Uniform Geometric**, registered as `uniform-volume` in
`uniform-volume-method.ts`. Its production factory calls
`WebGPUUniformReferenceSolver.createAsync` with `uniformGeometricSolverOptions`,
which enables `geometricVolume: true` and `pageDomain: true`. The benchmark and
new oracle use that same constructor/options path. Sparse CM12 is a separate
method; its repository-required regression gate is not the optimization oracle.

Transport previously inherited every solid/terrain seed from velocity extension.
The seed pass now writes independent FINE/SHELL and TRANSPORT bits. Liquid at or
above the dust floor, swept inflow, drops and one-sided phi-band support seed
both; solid-only cells seed velocity support only. Existing axis dilation passes
preserve those bits. There is no additional dispatch, field copy or coordinate
lookup. Threshold equality now matches the strictly-less-than dust discard rule.

Phi support intentionally remains in transport: gather also evaluates the
geometric target used by surface correction. Removing phi-only support would
invalidate the zero-gamma shortcut outside transport.

On hero-garden-hose (144 × 96 × 96), 44 advances, four warmups, balanced defaults,
1e-3 floor, exclusive Dawn/Metal on Apple M1 Max, rendering excluded.
**These original captures accidentally omitted the five scene rigid bodies.**
They isolate the transport change but do not represent UI frame costs. See
[garden-browser-reproduction.md](garden-browser-reproduction.md) for the corrected reproduction:

| Measurement | Before e49add19 | After |
|---|---:|---:|
| Repeated uninstrumented median | 50.296 ms | 46.744 ms |
| Instrumented whole advance | 51.527 ms | 46.858 ms |
| Volume coupling stage | 7.143 ms | 3.539 ms |
| Gather stage | 3.801 ms | 3.375 ms |
| Final transport workgroups | 9,200 | 3,467 |
| Final volume (cell volumes) | 1,077.284668 | 1,077.284668 |
| Final max speed (m/s) | 1.929569 | 1.929569 |

The repeated uninstrumented pair reduces time by 7.1%; transport workgroups fall
62.3%. An initial pair measured 50.045 / 57.547 ms, so that first after capture
was slower. The stage pair and uninstrumented repeat did not reproduce it;
all captures are retained in garden-transport-measurements.json.

The new GPU oracle compares tile transport with full-domain transport over
12 frames of empty terrain, hose startup, remote insertion and source shutoff.
In the original no-body run, volume, phi and velocity match bit-for-bit.
The test now supplies the scene body list and retains its static voxel stones; corrected results are recorded
in the browser reproduction report. Fine and shell counts match. With
inflow disabled there are zero transport workgroups and zero published active
transport pages; starting the hose activates 210 / 20,736 tiles and 8 / 45 pages.
The page flags are read back and checked against their published count. Run with
`npm run test:dawn:uniform-garden`.

This change reduces numerical work, not allocated residency. All 45 authored
pages remain resident, and solid-only velocity support remains intact. The
existing view therefore still shows other resident pages in faint purple.
Type checking retains 15 pre-existing errors outside the changed files.

The repository-required Sparse CM12 gate completed: 5/17 lanes passed.
Hydrostatic adaptivity passed this time; all other statuses match the preceding
4/17 run. This separate-method change is not attributed to the Uniform Geometric
optimization. No thresholds were changed.
See `garden-transport-sparse-regression.json`.
