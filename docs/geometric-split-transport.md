# Directional geometric transport integration contract

`geometric-split-transport.wgsl.ts` supplies binding-free building blocks only.
It is not a dispatched production transport scheme, and no boundedness or
performance gate has yet been run against it. It consumes the geometric plane
and accepted geometric subface helpers without changing CM12 execution.

For each synchronized transport subcycle, freeze the binary compression
coefficient `c_i = (V_i >= C_i / 2)` from its initial cell amounts. Keep the
projected row velocities fixed across that cycle. Sweep one axis at a time,
reconstructing the interface from the accepted intermediate amounts before the
next axis. Sweep ordering should rotate between cycles to measure splitting
bias. A state update or topology change must not intervene partway through the
cycle.

A row owns each signed liquid and bulk flux record exactly once. Upwind is the
negative cell for positive row velocity and the positive cell for negative
velocity. Its outgoing prism is the physical tangential subface rectangle swept
back into that donor by `abs(u_row) * dt`. Shift the donor-centred liquid plane
to the prism centre before evaluating its clipped box fraction. Empty and full
authoritative cells bypass the plane calculation. For a partial cell, the helper
also checks that the supplied plane reconstructs its current volume to the
caller's explicit volume tolerance.

Do not evaluate three axes' donor prisms independently from one initial volume:
their edge and corner regions overlap. Within one directional sweep, same-side
rectangles must be a disjoint geometric partition. The supplied donor CFL audit
requires `dt * (maximum outward speed on negative side + maximum outward speed
on positive side) <= cell axis width`. This sufficient prism-disjointness bound
must be computed over every physical subface of that cell. A per-face distance
check alone cannot establish it. This bound is not asserted to prove all
intermediate Weymouth–Yue volume bounds on arbitrary adaptive rows; the gather
checks the actual resulting volume and fails if it leaves the numerical bounds.

Both adjacent cells gather the same stored flux with opposite signs. They must
not independently compute donor and receiver versions. Cell update is

    V_after_axis = V_before_axis + liquid_in - liquid_out
                   + c_frozen * (bulk_out - bulk_in).

For a full cell with full upstream donors, liquid flux equals bulk flux and the
compression term preserves its full volume. For an empty cell with empty donors
and c=0, it preserves zero. Partially filled receivers require the same shared
bulk/pressure row field and actual geometric subface partition. Pairwise liquid
flux cancellation alone is insufficient to establish boundedness. The gather
reports the conservative amount, compression source, final amount and any
lower/upper bound violation separately. No clamp or redistribution changes them.

For one complete cycle, the compression source summed at cell i is
`c_i * sum_axes(bulk_out - bulk_in)_i`. Exact discretely divergence-free row
velocities make this zero. The existing approximate pressure solve generally
leaves a residual, so this update can change total liquid volume by the sum of
these weighted residuals. It must not be described as exactly conservative in
that case. `geometricSplitAuditCycle` exposes the actual cell residual and
compression defect and fails when residual exceeds the supplied tolerance.
Also report the signed global defect, sum of absolute cell defects, accumulated
simulation defect and floating-point gather error. A passing finite tolerance
means a measured nonzero conservation error, not exact conservation. Retrying
with smaller dt can reduce a single-step residual receipt but does not establish
that its accumulated physical-time error converges; improve compatible velocity
projection when needed.

The caller sets tolerances in finest-cell volumes using an explicit floating
point and projection-error budget. A tolerance does not authorize modifying
state. The current reconstruction accepts strictly bounded inputs; an accepted
roundoff excursion still requires a defined representation/validation policy
before a following sweep. Any invalid flux, CFL audit, gather or cycle audit
must prevent publishing the candidate state. Existing state remains intact while
the integrator reports or retries the failed candidate; no CM12 fallback is
permitted.

All current flux helpers require uncut rectangular capacity. A scalar open
fraction does not locate open space inside a prism and is explicitly rejected.
The new six-tetrahedron cut-cell geometry helper can represent a piecewise
linear solid, but transport must clip those *same original tetrahedra* by prism
bounds and liquid/solid planes. Resampling the solid SDF at prism corners and
retetrahedralizing can change the solid across original tetrahedron seams.
Small open capacities require geometry-aware bulk fluxes and a local outgoing
volume/CFL bound relative to that capacity, possibly synchronized subcycling.
Multiplying rectangular fluxes by one scalar aperture or using full-box width
as the sole cut-cell CFL does not supply this guarantee. Moving solids also
require wall flux and swept-volume accounting before the static formula applies.
