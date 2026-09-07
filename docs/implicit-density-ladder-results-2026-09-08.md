# Implicit-density ladder: first implementation results

The [roadmap](adaptive-mass-implicit-density-roadmap.md) is now backed by a
standalone TypeScript implementation and independently evaluated example ladder.
This completes the first local-algebra milestone and the exact-polynomial part
of mean-based reconstruction. It does not complete global-field assembly,
feature inference, transport, pressure, or GPU integration.

Production simulation and presentation are unchanged. The implementation lives
under `tools/implicit-density/`, with no imports from production consumers.
Three Astra agents at medium effort contributed the fixtures, stencil tests,
and roadmap; the integrated ladder and receipts were checked together.

## What is implemented

- A physical-coordinate density record with affine/quadratic coefficients,
  analytic volume averages and gradients. Second moments include native cell
  extent; density means are never treated as centre samples.
- Exact coordinate transformations on subdivision. Children receive integrals
  of the retained field and transformed coefficients. A merge compares those
  coefficients in a common frame and rejects new unrepresentable child detail.
- Explicit min/max of two affine branches for convex/concave edges, with
  integrated branch selection. This preserves the two feature normals; it
  does not infer the branches from density means.
- A clamped affine integration control with a fixed physical transition width.
  The 30%-filled example produces four child means of 0.6 and four of zero,
  keeping its half-density surface at y=0.3.
- A geometry-only reconstruction compiler using constrained least squares and
  reorthogonalized QR. It gathers current donor densities once per application;
  topology lookup and fitting geometry remain outside that loop. Stale,
  deficient and malformed supports are rejected.

## Representation ladder

The ten fixtures contain three oblique/near-axis planes, three rotated shallow
quadratics (bowl, saddle and nearly flat), and four convex/concave edges.
References use separately written physical-coordinate equations and integrals.
They provide 266 off-grid surface samples, including 20 exact crease points.
The retained candidate has coefficient records, not an authored-shape callback.

Every fixture goes through 100 split/merge cycles. The finer half-domain region
changes axes and sides; each cycle contains 260 mixed-width query leaves.
Actual production 2:1 grading and its publication transaction are not exercised
by these synthetic query partitions.

| Measurement over all fixtures and cycles | Maximum absolute error |
| --- | ---: |
| Density on independently evaluated surface points, relative to 0.5 | 1.122e-14 |
| Integrated amount in the full 2×2×2 example box | 8.749e-14 |
| Density at off-grid volume probes | 1.133e-14 |
| Local child mean versus independent integral | 1.133e-14 |
| Gradient component | 7.078e-16 |
| Child-face query versus common retained field | 7.772e-16 |

These are exact-family float64 results. A small surface density residual is
not a measured Hausdorff distance. The scalar/gradient and coefficient identity
checks establish this local algebra; general surface-distance convergence and
independently assembled patch boundaries remain subsequent gates. Small
floating-point drift remains visible across repeated rebasings; it is not
reported as bitwise identity.

![Analytic and represented surface sections](../artifacts/implicit-density/ladder.png)

Dark lines are the independent reference; dashed orange lines are the retained
field after 100 cycles. They overlap at this scale. These are sections of 3D
fields, with section planes chosen to show the edge where appropriate. No
mesher, renderer or fluid timestep participates.

## Reconstructing from means and the sharp-edge control

The stencil tests reproduce all ten polynomial basis modes on 16 support
geometries, covering uniform/mixed 2:1 cells, faces/edges/corners and clipped
anisotropic boxes. Donor means and checks use an independent Gauss integration
rule. Separate nonpolynomial data tests check the home-mean constraint without
assuming exact reconstruction of arbitrary data.

Float32 emulation has maximum coefficient error 1.956e-7 and home-mean error
1.609e-7 in those fixtures. This is not a WGSL or GPU result.

The visual fixture family was additionally fitted from native donor means:

| Mean-fit support | Exact plane/quadratic maximum surface density residual | Sharp edge fitted with the same smooth quadratic |
| --- | ---: | ---: |
| 27 uniform donors | 3.886e-16 | 0.008395 |
| 90 mixed-width donors | 5.552e-16 | 0.009084 |

The latter is a negative control, not a passing edge reconstruction. It shows
that fitting a single smooth quadratic conservatively does not preserve a
crease that the retained two-branch record represents exactly. We still need
to detect and fit feature branches from local data, with residual and ambiguity
checks. An arbitrary corner, thin sheet or disconnected body does not fit
automatically into this two-branch family.

## Performance evidence and limits

Compilation uses native box moments and produces a reusable linear map. The
steady-state work counts are explicit:

| Support | Density reads/apply | Multiply-adds/apply | Expanded f32 weights | Donor ordinals |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 27 | 270 | 1,080 bytes | 108 bytes |
| Mixed | 90 | 900 | 3,600 bytes | 360 bytes |

These bytes exclude frame/header, output, reverse dependencies and overlapping
generations. Storing a separate expanded map for every cell is not the intended
GPU architecture. The next integration design must intern equivalent geometry
patterns, use cheap regular-interior paths and bound exceptional support.

This capture measured CPU compilation at 0.73/0.81 ms and application at
3.41/10.67 microseconds for the uniform/mixed cases. They include JS allocation
and Map reads, are a single observational capture, and cannot predict GPU frame
cost. No speedup is claimed. The existing TEI native geometry and staged donor
directory are the intended adapter authority; this first compiler does not yet
consume the production packed topology.

## Checks and regression cadence

Passed together:

- 34 new field/stencil tests, including 100-cycle examples, the fixed-width
  clamped ramp, independent simplex integrals and rejection of lost detail.
- 9 existing CPU checks from coarse-first scene initialization/control routing,
  stable sparse neighbour discovery and renderer live-edit publication.
- Targeted ESLint and strict TypeScript checking of every added TS module/test.
- The standalone ladder capture.

The existing checks validate construction/topology/editor behavior; they do
not advance native Dawn core scenes. No production code changed, so no new
full Dawn result is claimed. The roadmap schedules a core-scene checkpoint at
global-field/zero-time completion and again after prescribed transport. Any
large production simulation/topology/publication change still requires the
unchanged full `npm run test:dawn:sparse-cm12` gate under the exclusive WebGPU
lease, with Fluid browser simulation unloaded. Record baseline failures rather
than weakening thresholds.

## Next construction step

Build a small retained patch neighbourhood with shared interface constraints
and a production-native geometry adapter. Exercise a broad plane and shallow
curved field across an actual 2:1 neighbourhood, then a crossing crease.
Compare independently reconstructed patches and donor integral residuals with
the common-field oracle; reject unintended scalar/normal jumps. Compile reverse
dependencies and test dirty updates against full recomputation.

The open design questions are shared smooth/feature traces, bounded positivity,
and how to infer a feature without inventing one from ambiguous means. Keep
those explicit. Success of the algebra above does not authorize presenting
independently fitted polynomials as a globally consistent surface.

## Reproduction

```sh
node --import tsx --test tests/implicit-density-field.test.ts tests/implicit-density-stencil.test.ts
node --import tsx tools/probe-implicit-density.ts
python3 tools/plot-implicit-density-ladder.py
```

The optional plot requires NumPy and Matplotlib. The numeric implementation and
tests use the repository's existing TypeScript runtime and no new packages.
[The JSON receipt](../artifacts/implicit-density/ladder.json) records source
SHA256 hashes, Node version, budgets, all fixture measurements and compiled-map
costs. Generated JSON/PNG artifacts follow the repository's ignored-artifact
policy; this report preserves the principal measured results.
