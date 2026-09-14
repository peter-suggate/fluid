# Hydrostatic level-set-volume instability

Date: 2026-09-14. Investigation of `hydrostatic-power-large-offset` using the
current native and SIMD production World, `dt=1/30 s`, 256 pressure iterations,
relative tolerance `1e-6`, and the level-set-volume transport option.

## Finding

There is a reproducible feedback loop: an inconsistent initial scalar field,
a discontinuous zero-motion advection branch, and two competing free-surface
definitions in pressure. Excess-pressure release is inactive when the error
starts. The pressure solve subsequently converges to incompatible boundary
data, so a small linear residual does not establish hydrostatic equilibrium.

## First mismatches

The scene's intended water height is 15.25 fine cells, or 0.7625 m. Its exact
initial volume is 488 fine-area units. The initial adaptive slice has eight
width-8 cells, with the upper row filled to 0.90625.

`levelset_surface::initialize_from_volume` assigns the entire upper coarse cell
the proxy `(0.5 - 0.90625) * 2 * 8 = -6.5`; absent air receives `+1`. The shared
vertex at y=16 averages those values to -2.75, and y=17 receives +1. The zero
therefore lies at `16 + 2.75/3.75 = 16.733333`. It is flat but 1.483333 fine
cells, or 0.074167 m, too high. The volume is correct; its conversion to phi is
the first incorrect diagnostic, at frame zero.

| Frame | Observed change |
| --- | --- |
| 0 | Correct V; flat phi zero at 16.733333 instead of 15.25. |
| 1 | Support increases from 8 to 12 cells. Phi is unchanged; projected velocities are roundoff. |
| 2 | Thirty-three bottom-boundary vertices switch from proxy -8 to contour distance -16.733333. The zero contour remains flat. |
| 3 | The contour remains flat; pressure velocities are still roundoff. |
| 4 | Twenty-six vertices switch scalar metric, including near-surface vertices around x=1. The first kink appears, with slopes up to about 0.642. False curvature promotes brick 20 to rung 8 and 2:1 closure expands the grid from 12 to 114 cells. |
| 5 | The refined pressure system produces wet-cell speeds up to 0.0462 m/s at the primary projection, then 0.0531 m/s after the existing post-support projection. Excess first appears after transport. |

At frame 4 the maximum wet-cell speed is only `1.20e-7 m/s`. It is enough to
trigger a different scalar branch, despite cell-centre traces rounding back to
their starting positions and the receipt reporting zero trace Courant.

## Zero-motion discontinuity

`levelset_volume::advect_shared_phi` keeps the stored scalar when
`departure == start`. Otherwise it queries exact geometric signed distance to
the previous contour. These are different functions.

For example, at x=1 the near-surface values switch as follows in frame 4:

| Vertex y | Stored proxy | Geometric distance |
| --- | ---: | ---: |
| 15 | -6.5 | -1.733333 |
| 16 | -2.75 | -0.733333 |
| 17 | +1 | +0.266667 |

Neighboring stationary vertices retain the proxy values. The unequal rescaling
changes the interpolated zero contour and its normals. A microscopic departure
thus causes a finite scalar change instead of a microscopic change.

In symbols, for the accepted contour Gamma:

```
A(0)       = stored_phi(x)
lim A(u)   = signed_distance_to_Gamma(x), as u tends to zero through nonzero departures
```

These need not agree. Correcting initialization alone does not remove this
general defect: after deformation, stored vertex values need not be exact
distances to their newly interpolated contour. Even samples of an analytic
signed-distance function generally differ from distances to its polygonal
interpolant. Simply deleting the identity branch can introduce repeated
zero-motion contour resampling; it needs an explicit representation decision.

## How refinement becomes a gravity-driven current

The initial phi zero lies outside the coarse volume-bearing cell. Its centre
is y=12, so a phi-derived horizontal pressure plane would have offset about
4.733333. `publish_pressure_geometry_from_phi` rejects it because that exceeds
the cell's half-width of 4, leaving its normal zero.

During the frame-4 topology transfer, `transfer_fields_allow_overcapacity`
therefore has no valid normal for its geometric split. It divides volume in
proportion to child capacities. Every child in y=8..16 receives fill 0.90625.
The exact-flat overlap error rises from zero to 15 fine-area units although
total volume remains exactly 488.

Pressure membership still comes from V/capacity. It marks those children as
liquid. Pressure geometry comes from phi, but
`prepare_pressure_topology_impl` rejects it when its signs disagree with that
membership. Rejected rows fall back to
`(0.5 - pressure_density) * cell_width`.

Reconstructing the frame-5 cut-row arithmetic gives:

| Rows | Boundary source | Theta | Implied water height |
| --- | --- | ---: | ---: |
| 194–197, width-2 liquid / width-1 air | Density fallback: phi = (-0.8125, +0.5) | 0.6190476 | 15.928571 |
| 199–200 | Accepted phi geometry: phi = (-2.733333, +0.266667) | 0.911111 | 16.733334 |
| 202, 204 | Accepted phi geometry | Consistent with phi | 16.733334 |

Neighboring columns impose a pressure-head difference of 0.804763 fine cells.
The solver enforces these conflicting heads and produces circulation. Its
frame-5 final residual is about `3.65e-7`; convergence cannot repair the
boundary definitions.

The existing volume-column hydrostatic override is gated on authored
`refinement_region_scale` mixtures; this naturally adaptive scene has scale 1
throughout. The mixed-seam theta correction also follows that override. It is
not a general guarantee of hydrostatic balance on this path.

## Why this is broader than the initializer

There are two independent ways to restart motion near equilibrium:

- Any difference between stored phi and contour distance can be exposed by
  the exact-zero departure branch, including after real deformation.
- Any difference between V-derived membership and phi geometry can make
  neighboring pressure rows choose different surface locations. Transport and
  topology transfer can create that difference after a correct initialization.

The new excess source does not cause the first event: excess is zero through
frame 4 and first becomes positive after frame-5 transport. Later it adds the
intended expansion response to a flow already driven out of equilibrium.

## Required corrections

The implementation should satisfy three invariants within the existing stages:

1. Seed the initial zero set from retained scene geometry at the correct height,
   independent of the chosen adaptive carrier resolution. Coarse occupancy
   alone is insufficient to recover arbitrary subcell geometry.
2. Make the phi transport operator continuous at zero motion, using one
   consistent scalar representation for stationary and moving samples while
   preserving the accepted zero set to the chosen accuracy.
3. Make phi the single free-surface authority for pressure, including membership
   and cut-face boundary locations. V remains the mass field and supplies the
   approved excess expansion term; it must not silently replace the surface
   with a width-dependent density contour.

These are representation and boundary-consistency fixes. Extra pressure
iterations, damping, a motion threshold, or more balancing passes do not repair
the demonstrated invariant violations. No production algorithm was changed
during this investigation.

## Reproduction

- `tools/diagnose-hydrostatic-offset-lsv-surface.ts`
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-surface-first-frame.json`
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-surface-first-five.json`
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-surface.json`
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-native-first-five.json`

The 30-frame SIMD capture retains total mass at 487.9999969, but its surface
height spans errors of roughly 0.0884 to 2.5939 fine cells relative to the
authored flat plane. This is late evidence of the same run, rather than the
argument identifying the first cause.

## Implemented correction and validation

The three representation invariants above are now implemented together:

- `levelset_surface::initialize_from_document` samples the authored initial
  scene geometry on the fine vertex lattice. The initial zero set is therefore
  y=15.25 for both the coarse adaptive atlas and a uniformly fine atlas.
- `levelset_volume::advect_shared_phi` samples the same stored centre-fan scalar
  for every departure, including exact zero motion. Exact contour distance is
  still evaluated at accepted cell centres for derived pressure geometry; it
  is no longer substituted selectively into moving vertex traces.
- The direct and embedded pressure preparations use phi for pressure membership
  and cut-boundary geometry at both projection sites. Conservative V remains
  independent and continues to provide the excess expansion source. Internal
  sparse boundaries require a phi-proven crossing. Physical open container
  faces retain their authored pressure boundary and flux; a nearer phi zero
  takes precedence. This distinction preserves uniform through-flow without
  inventing a surface at missing interior support.

The focused native integration file
`rust/crates/fluid-core/tests/levelset_volume_hydrostatic.rs` covers the initial
plane on coarse and fine atlases, independent phi pressure membership and V
excess sourcing, and 30 frames with a forced asymmetric fine-resolution region.
At the UI pressure settings (256 iterations and relative tolerance `1e-6`), all
three tests pass. The forced mixed-rung run measured these maxima:

| Quantity | Maximum over 30 frames |
| --- | ---: |
| Liquid-touching face speed | 0.000107288361 fine cells/s (5.3644e-6 m/s) |
| Cell-centre-derived surface-height error | 0.000016212463 fine cells (8.1062e-7 m) |
| Absolute conservative-volume drift | 0.000003903034 fine-area units |

The speed excludes dry support faces, which can carry extension or gravity
velocity without representing fluid motion. The regression selects rows that
touch the phi-derived pressure phase, so it tests the hydrostatic liquid rather
than the maximum over the entire support atlas.

The unedited production scene also completed 30 native frames without a fault.
Its maximum reported face speed was `3.815e-6` fine cells/s, conservative-volume
drift and excess were zero, and no false curvature refinement occurred: the
topology remained at 12 cells.

The corresponding rebuilt SIMD Wasm regression passes with the same UI
settings and asymmetric refinement, checking liquid-touching velocities,
the published flat contour, and conservative volume.

A separate 300-frame SIMD capture (10 seconds of simulated time) of the unedited
scene preserves every stored phi value exactly. All contour endpoints stay at
y=15.25, all segment slopes remain zero, and both volume drift and excess remain
zero. Maximum published face speed is `4.768371582e-6` fine cells/s. The initial
eight cells acquire four coarse support cells and remain at 12 cells.

Evidence:
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-surface-fixed-300.json`
- `artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-native-fixed-30.json`

Native release and scalar/SIMD/threaded Wasm plus the site build completed.
The served artifacts match source fingerprint
`f8b98a69b53573dbc69fbc1dbe0944436b160346fbeee766d288bac78c938ecf`;
the UI route and isolation headers passed the serving checks. Frozen native
baseline regressions remain unchanged.

The native integration gate passes 23 tests, and the level-set pressure/World
unit gate passes 10. All five Wasm flow tests pass, including the Figure 7
and pool-impact regressions. Figure 7 exercises 79 newly allocated empty
support bricks over frames 8–30; each starts at accepted resolution 1, before
applying geometric and 2:1 requirements. Its regression checks allocations
throughout the run rather than requiring one at a particular frame.
New receivers' own activity reasons start at zero and do not encode incoming
donor constraints. For example, at frame 11 receivers 2288 and 2289 need rung 8
because donors 2165 and 2170 carry thin liquid features, not because of speed.
Unconstrained allocation coverage therefore uses requested resolution 1, with
nonzero coverage, rather than interpreting zero local activity as no constraint.
