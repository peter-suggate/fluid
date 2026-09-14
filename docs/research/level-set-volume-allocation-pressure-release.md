# Coarse support allocation and excess-volume pressure release

Date: 2026-09-14.

## Allocation root cause

At Figure 7 frame 27, the end-of-frame planner created empty finest-resolution
support pages without a curvature or velocity-variation requirement. The
allocator assigned rung 8 directly. The projected-support planner already used
donor-compatible widths, but reactivation retained stale fine metadata and it
did not receive the full region/static-floor constraints.

The level-set-volume path now uses one donor-width calculation in both existing
planners. The receiver starts at the coarsest rung satisfying both 2:1 donor
spacing and each donor's geometry, relative-velocity and thin-feature demands.
These bounds use physical cell widths, including macrobricks. Region bounds,
known static floors, frozen-page constraints and existing 2:1 closure also
apply. Reactivation uses these requirements instead of inheriting stale fine
metadata. Existing represented pages retain their pre-transport rung and
continue through the existing sizing policy after transport.

An intermediate implementation carrying only 2:1 spacing exposed a second
missing input: B8 donors created B4 receivers, then B2, then B1. The old
curvature measurement sampled adaptive cell normals selected by V. At B1 it
had only one normal and therefore reported zero variation, even though the
shared phi contour remained curved. By frame 23 all encoded curvature floors
had fallen to one. A runtime regression reproduces that failure.

For LSV, the existing per-brick measurement now uses the exact centre-fan
triangles used to publish phi. It collects normalized gradients on triangles
crossed by the zero contour, then applies the existing normal-diameter metric
and curvature tolerance. The result is independent of the adaptive carrier
resolution and of diffuse V. Direct contour crossings also prevent the
density-based enclosed-liquid shortcut from hiding a surface. This changes the
input to the existing sizing stage, not the surface or the curvature rule.

This changes allocation resolution, not support extent. Tiny positive volume
still requires a receiver; no material is deleted. The opt-in is internal to
level-set-volume, preserving the baseline and cellwise-remap golden paths.

Allocation churn also exposed a latent leaf-ID collision: a free ID above the
largest accepted ID could be popped, then generated again by the sequential
fallback. Both 2-D planners now start fallback IDs above the accepted and free
ID sets. A focused regression exercises stack exhaustion in both planners.

## Pressure root cause and implementation

The volume gather conserves donor mass but can leave receivers over capacity.
Previously the next projection requested zero divergence for that excess, so
pressure had no expansion target. The initial experiment used

```
q_i = 0.5 * max(V_i - C_i, 0) / dt
V_i = density_i * cell_measure_i
C_i = capacity_i * cell_measure_i
```

The RHS uses integrated flux units, so `q` is fine-area per second in the 2-D
world, not a dimensionless fill ratio. Its positive sign requests outward flux.
The pressure embedding maps this integrated source by the source/reduced cell
measure ratio, consistent with its existing flux scaling.

The existing source term carries `q` only inside each pressure preparation,
solve and projection scope. The physical source is restored on success and
error, before support planning, transport and source accounting. If projected
support changes topology, the existing second projection recomputes `q` from
the transferred V/C. Enforcing zero divergence there would erase the first
projection's expansion. Enforcing the same target does not double the impulse.

The measurements below used no cap. Half of the current excess per frame corresponds to 87.5%
release over three frames under an ideal realization of the requested flux.
It is not a guarantee that the conservative gather removes exactly that amount:
transport can create new excess, and the discrete transport field differs from
the projected face flux. No additional projection, transport pass, volume clamp,
redistribution, sharpening or direct phi/volume correction was added.

### Tall Cells correction

The Tall Cells hillside run subsequently demonstrated the instability CM12's
published cap is meant to prevent. Its maximum excess ratio reached roughly
`6.426` times open capacity; the uncapped formula therefore requested about
3.213 local open-cell capacities of expansion in one 1/30 s frame. Topology
changes that introduced terrain cut cells made their small open apertures
realize that request as a sharp pressure impulse.

The production target is now

```
q_i = min(0.5 * max(V_i - C_i, 0), C_i) / dt
```

Dividing by the integrated open capacity shows the resolution-independent
bound directly: `q_i dt / C_i <= 1`. Thus a coarse cell and a terrain cut cell
receive the same maximum normalized expansion, while the cut cell's absolute
flux scales with the open volume it can represent. The cap changes only the
temporary pressure constraint; conservative `V` is neither clamped nor
deleted, and unreleased excess remains for later frames.

## Native Figure 7 comparison (pre-cap evidence)

Both runs below use the final allocation and direct-phi curvature fixes, the
same scene, `dt = 1/30 s`, and 256 pressure iterations with relative tolerance
`1e-6`. The native production World is advanced through frame 30. The control
uses a separate source copy with only the pressure relaxation constant set to
zero; the workspace and served UI retain `0.5`.

| Frame | Excess without pressure release | Excess with pressure release |
| --- | ---: | ---: |
| 25 | 28.6322 | 28.4093 |
| 26 | 134.2610 | 122.9283 |
| 27 | 226.8031 | 174.1561 |
| 28 | 283.7630 | 201.1685 |
| 29 | 330.2426 | 171.9429 |
| 30 | 368.7420 | 160.1887 |

Total excess is 56.6% lower at frame 30. The worst individual fill ratio falls
from 18.50 to 9.96. Local compression remains substantial; this does not prove
recovery of the wall splash. Relative mass drift remains below `5.7e-9` in the
release run. The final physical source-rate array is zero and pressure converges.
These values are transport-receipt excess, before the final topology transfer.
The final published frame-30 field has `159.7483` excess fine-area units across
356 cells; native and Wasm agree on the transport receipt and conserved mass.

The first positive excess in these runs appears at frame 14 (`5.77e-5` fine-area
units). The significant impact excess grows during frames 23–25. The release source
acts on the following frame's pressure solve, using the accepted excess.

These artifacts record the historical uncapped run:

- `artifacts/level-set-volume/cm12-figure-7-geometry-bound-without-pressure-release-native.json`
- `artifacts/level-set-volume/cm12-figure-7-geometry-bound-pressure-release-native.json`
- `artifacts/level-set-volume/cm12-figure-7-geometry-bound-pressure-release.json`
- `artifacts/level-set-volume/cm12-figure-7-before-coarse-allocation-pressure-release.json`

Focused regressions cover new/reactivated allocation, multiple donors, physical
widths across macrobricks, constraints, integrated source units, outward flux,
and repeated projection without double expansion. The production pressure
embedding regression exercises a topology-changing second projection and four
frames of decreasing excess while retaining mass and restoring the physical
source. Frozen baseline tests remain unchanged.

Validation passed 42 focused native tests and all four scalar/SIMD Wasm flow
tests. The final Figure 7 capture retains 12 contour bricks with curvature
floor 2 at frame 23. New empty upper support pages at frame 25 use rung 1;
frame 27 makes no new allocations. The earlier vacated pair now reaches rung 1
at frame 16 after the existing two-epoch quiet hysteresis, and its regression
checks that first accepted coarsening frame.

Native release, scalar, SIMD, threaded and UI production builds passed. The
production UI was restarted on port 4173; the scene returns HTTP 200, served
Wasm hashes match all three artifacts, and isolation headers pass on nine
HTML/worker/Wasm responses. All artifacts match Rust source fingerprint
`cb416eba6959a7fd59f42337ef45c5282d105623dfb5dd436d01d1a9b4a7897b`.
