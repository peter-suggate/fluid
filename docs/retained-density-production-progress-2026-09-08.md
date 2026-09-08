# Retained density production acceptance

This is an implementation record, not a claim that arbitrary curvature transport
or every boundary interaction is complete. The authored plane, quadratic height,
box and sphere sources now compile to a physical density field used by both the
production native density initialization and surface publication.

The field is `q = clamp(0.5 - min(phi_i) / w, 0, 1)`, with fixed physical `w`.
Native densities are volume integrals of that field. Neither a region edit nor
native rung selection changes its coefficients or transition width. Fine support
moments are integrated once; native rungs restrict those same moments. Solids
use actual open voxel boxes or the authoritative quantized terrain bottom slab,
not a product of an average density and an average aperture.

## Evidence recorded so far

The independent CPU ladder measures the zero set and normals after 100
split/merge cycles. Its ten examples include planes, subtle quadratic curvature,
spheres and sharp branch intersections. Maximum surface displacement is
`9.40e-14 m`; maximum normal-vector error is `5.08e-15`.

The actual quarter and half pool-impact scenes both passed
`tests/sparse-cm12-pool-impact-analytic-dawn.test.ts` in the exclusive Dawn run
`/tmp/fluid-retained-pool-production-6.log`: two tests, 64.1 seconds total. Each
scene visits thirteen paused partitions, including global widths 1, 2 and 4
and returns to the original local min/max-8 region, then resumes one paper
step of 1/30 s. The literal original region query is
`0_0_0_25_66.6667_100_8_8`; no idealized replacement bounds are used.

| Measurement | Quarter | Half |
| --- | ---: | ---: |
| Analytic crossing count | All 1,184 | All 4,728 |
| Upper/lower sphere crossings included | All 160 | All 632 |
| Paused published scalar change | Exactly zero | Exactly zero |
| Native restriction error | `4.48e-8` density | `8.38e-8` density |
| Difference from independent diffuse amount | `3.57e-9 m³` | `2.43e-8 m³` |
| Published pool-plane height error | `5.56e-17 m` | `1.12e-16 m` |
| Extracted pool-plane height error | `5.97e-9 m` | `1.20e-8 m` |
| Extracted sphere vertex radial error | `3.774 mm` | `1.879 mm` |
| Extracted sphere whole-triangle radial error | `4.780 mm` | `2.737 mm` |
| Extracted sphere normal-vector error | `1.92e-4` | `2.36e-4` |
| Non-manifold / interior open edges | Zero / zero | Zero / zero |
| Resume | Velocity and field evolve | Velocity and field evolve |

Values bound the worst measurement over each scene's saved captures at edits
0, 6 and 12, or all thirteen partitions for field/native measurements. The
half scene's literal 66.6667% height snaps outward to 2.0 m, rather than 1.6 m;
the quarter region reaches 0.8 m. The exact catalog parameters, parsed bounds
and independent error budgets are recorded in
[analytic-pool-impact-acceptance-2026-09-08.md](analytic-pool-impact-acceptance-2026-09-08.md).

These mesh errors are finite polygon and field-sampling errors, not an exact
sphere claim for planar triangles. The initial implicit boundary is the sphere;
the actual shipping mesh is independently measured against it. Its previous
averaged polygon centers shrank the sphere by up to 35.4 mm. Centers now project
onto the sampled field, and polygon fans subdivide when their interior scalar
defect exceeds 1/32 of a finest cell. Normals differentiate a quadratic
interpolant with valid one-sided support at publication boundaries. The final
mesh/oracle change is commit `0cabd42d`. Whole-triangle distance finds the
closest point anywhere on each triangle, including a perpendicular projection
that can lie away from its centroid; a CPU counterexample checks this oracle.

Saved raw production meshes, scalar samples and receipts are under each actual
scene ID in `artifacts/retained-density-production/`. The two-scene comparison
`artifacts/retained-density-production/analytic-mesh-sections.png` intersects
those GPU triangles with the physical z=0 plane against independent plane and
circle references. It does not resample or smooth the captured geometry.

The CPU field/oracle tests pass: nineteen tests across
`tests/implicit-density-field.test.ts` and
`tests/pool-impact-analytic-oracle.test.ts`. These establish independent
analytic queries and the oracle's sensitivity. The CPU algebra ladder alone
does not establish production GPU publication or evolved transport accuracy.

The retained-scene compiler/cache change `e4a4beb7` passed 34 focused CPU tests
and its strict TypeScript check. On the half scene's 196,608 physical supports,
building the cache took 5,989.0 ms; one partial interface-voxel edit took
21.2 ms, reused 196,607 supports and required eight new integrals and one
contiguous GPU upload range. An unchanged query took 17.0 ms and required no
new integrals. GPU arrays remained 12,582,912 bytes; total CPU cache storage,
including quantized solid and error receipts, was 14,352,384 bytes. These are
CPU compiler measurements, not renderer or simulation frame timings.

The CPU catalog audit examined 81 default scenes: all 75 supported fields
passed the isotropic-lattice check, with no anisotropic cases or build errors.
Six existing unsupported source forms remain: CM12 figures 2, 3 and 8,
standing wave, standing wave live, and falling-water torus. An isotropy guard
now rejects distorted noncatalog grids whose rounded/minimum-eight axis
dimensions would disagree with the compiler's uniform physical cell width.

## Acceptance coverage and pending gates

The generic `tests/sparse-cm12-retained-field-partition-dawn.test.ts` covers
flat, quadratic, sphere/pool and sharp-box fixtures through twelve reset-time
and four resumed-time partition changes. It checks actual native widths,
restrictions of fine native means, mass, near-interface scalar samples and
every interior vertical column's crossing count and displacement. Its reset
phase now additionally checks independent authored equations for all four
fixtures through `tools/implicit-density/partition-oracle.ts`, which imports
neither the production compiler nor evaluator. Every near-interface sample and
all vertical columns, including boundary columns, are checked against analytic
references. Root precision accounts for binary16 storage and the local slope.
Six CPU oracle tests pass, including changed-field and missing-sample
counterexamples. All four independent production reset assertions passed in
`/tmp/fluid-retained-ladder-gpu-4.log`:

| Reset fixture | Analytic / observed crossings | Maximum continuous zero-set distance |
| --- | ---: | ---: |
| Flat | 256 / 256 | `3.97e-6 m` |
| Quadratic height | 256 / 256 | `6.07e-6 m` |
| Sphere/pool | 300 / 300 | `2.244 mm` |
| Sharp box/pool | 256 / 256 | `5.000 mm` |

Every expected near-interface sample was present and within the declared
binary16/float32 precision budget. The subsequent first mixed partition failed
the fixture's native-width assertion: forcing width 1 directly beside width 4
has no solution under the production 2:1 face-grading contract. The fixture now
requests width 1 beside width 2, and a CPU counterexample verifies the original
pair is rejected and the replacement is admissible. Global widths 4, 2 and 1
and the cycle count are unchanged. In the subsequent exclusive run
`/tmp/fluid-retained-ladder-clipped-gpu-5.log`, all four fixtures passed their
analytic reset and twelve paused edits. Flat and quadratic height also passed
physical evolution and the four resumed partition changes. Sphere/pool and
sharp box failed the first physical step's retained-integral guard, each by
exactly one CM12 amount quantum (`1/65536`). These are still failed lanes.
The paused native-integral threshold has since been tightened from `2e-5` to
`2e-6`, matching the production guard so one whole quantum cannot hide in the
paused acceptance receipt.

The clipped-domain transfer lane in that same run passed with zero mass error,
eight leaves and final resolution eight. The retained hard support now uses
the admitted `origin + dimensions * f32(cellWidth)` lattice endpoint. This
removes an artificial sliver caused by independently rounding the authored
domain endpoint; authored primitive geometry is unchanged.

The sharp-box L-infinity field can change its active face between sample
centers, so a linear sampled crossing near an edge need not be at the exact
authored top. The oracle separately measures the continuous zero-set distance
under a finest-cell interpolation budget and requires the independently
sampled analytic root. It does not confuse an exact implicit box with finite
linear extraction. Subsequent paused checks retain their accepted-generation
baseline. After resume they check finite velocity, not a separate
momentum-conservation receipt. They do not establish high-order curvature
transport or all exterior support/boundary interactions.

`tests/sparse-cm12-adaptive-mesh-dawn.test.ts` checks components, Euler
characteristic, winding, cracks, degeneracies and triangle reduction across
mesh ratios and synthetic flat, curved, sharp and disconnected surfaces. It
also captures the full catalog pool. Those relative topology checks cannot
exclude a consistently shrunken smooth shape; the independent actual-scene
distance and normal checks above supply that missing acceptance. Its full-pool
checkpoints explicitly use `timeStep: "scene"` and 1/60 s targets, a permitted
shorter scene step, rather than the catalog paper step used by the new tests.

The mesh regression now checks the raw vertex allocator, raw active cube
count and current publication generation before reading the emitted geometry.
Checking the draw count alone was insufficient because the production scan
already clamps it to allocation capacity. No capacity or acceptance threshold
was raised. Its synthetic mixed-resolution test passed in 40.2 seconds in
`/tmp/fluid-retained-ladder-mesh-gpu.log`, and the full pool passed all mesh
ratios at steps 0, 1 and 3. The process then exhausted the default roughly 4 GB
JavaScript heap before the step-30 checkpoint. This is an incomplete full-pool
lane, not a pass; no heap limit, checkpoint or threshold has been changed.

An isolated CPU audit of 456,864 vertices made from four copies of a saved
half-scene GPU mesh took 3.60 s: heap use was 9.78 MB before, 192.69 MB after
the audit and 10.54 MB after collection. Maximum RSS was 315 MiB. This shows
the mesh oracle's large temporary maps release in that reproduction; it does
not identify the production out-of-memory cause by itself.

The follow-up advance-only reproduction
`tools/probe-sparse-cm12-full-pool-advance.ts` also exhausted the default heap,
without any mesh capture or oracle. Its generation phase probe isolated the
dominant allocation: `packResidentTopologyTemplates` retained roughly
2.48–2.67 GB of JavaScript cell, row and term objects per catalog build. Source
overlap geometry accounted for another roughly 69–73 MB. The process completed
steps 1, 2 and 3 in 33.19, 40.39 and 75.82 seconds, then exhausted memory during
the fourth catalog build. The saved destination contains 1,836 mutable leaves,
1,074,060 catalog cells and 3,457,000 rows. These are failing performance
measurements, not acceptable frame timings.

A separate GPU implementation is replacing repeated native records with
interned relative operators and compact placement recipes. The device expands
cell, row and term planes from those recipes. The first CPU word-exact stage
completed the previously failing catalog under the default heap, using 4,553
row archetypes and 7,344 cell ranges, but still took 66.14 seconds; this is not
the finished performance result. Synchronous CPU certification consumers still
need a typed numeric shadow. GPU word equivalence, integrated generation
transfer and the unchanged canonical timing gates remain required before this
replacement is accepted. Increasing the Node heap is not the proposed remedy.

The generic resumed partition ladder, broader mesh regressions, browser visual
acceptance and canonical `npm run test:dawn:sparse-cm12` suite remain required
before declaring the implementation milestone complete. Their latest results
must be recorded separately from the passed quarter/half geometry tests. No
acceptance threshold or performance ceiling is to be relaxed to obtain a pass.

## Evolution and remaining validation

An actual scalar update creates the next retained generation on persistent
physical support. Each support retains `a*q_seed + b`; draining scales density,
filling mixes toward capacity, and compression retains positive excess. Native
integrals are rebuilt from these coefficients and checked against accepted
solver amounts before the generation publishes. Topology edits only integrate
the accepted generation. Replacement residents copy physical support and then
re-integrate it into the new native partition.

This first conservative evolution basis is piecewise affine in the retained
seed density, `a*q_seed + b`; it is not a spatial affine approximation to the
authored sphere. The coefficients are constant within each physical support
box and can be discontinuous between boxes. At t=0 the curved and sharp source
zero geometry is preserved to coefficient precision. As fluid enters an
initially empty support, its new shape is represented by this conservative
support basis, not by transporting a high-order curvature model into it.
Conservation and a moving half-density surface therefore do not establish
accurate long-time spherical or other curvature transport.

The initial support arena retains a dense bounded-domain prefix followed by
sparse growth slots; it is not the final sparse curvature representation.
General partial solid reopening needs further geometric support work and must
not silently re-author source liquid. CPU quadrature reports an estimated
integration error, not certified analytic bounds for arbitrary clipped unions.
The exact independent diffuse sphere/pool amount check above is additional
evidence for those specific unclipped acceptance fixtures.
