# Accepted-row geometric subfaces

The helper in `lib/methods/adaptive-volume/geometric-subfaces.wgsl.ts` is a
preparatory module. It is not yet imported into the resident or dispatched by
production. No new solver behavior or GPU validation is claimed here.

`createGeometricSubfacesWGSL()` uses the existing accepted-row and cell accessors.
The caller owns accepted generation selection, validates term/cell addresses,
and enumerates each accepted row exactly once. Within each row, iterate its term
range for negative and positive terms and call `geometricSubface`. Terms on the
same side produce no subface. An accepted descriptor is identified by the row,
negative term and positive term within that generation; these are not stable IDs
across topology changes.

For opposite-side cells, the helper intersects their tangential rectangles and
requires exact normal contact. Integer and half-integer dyadic bounds are exactly
representable within the current lattice's f32 range. Zero tangential overlap
produces no subface. Duplicate cell terms and noncontacting opposite sides are
malformed. Area is the intersection's geometric area, never a product of
incidence coefficients. A row split into ports that cannot be reconstructed from
these cell intersections will fail the area/marginal audit and requires explicit
row-footprint geometry before use; the helper must not silently normalize it.

All coordinates, widths, distances and areas use finest-lattice units. Resident
row velocities are finest cells per second. `areaFine2 * velocityFinePerSecond *
dtSeconds` is signed swept volume in finest-cell volumes. Convert to cubic metres
by multiplying by the physical finest-cell size cubed. Apply a single transfer
with the negative of that value at the negative cell and the positive at the
positive cell. A later bounded transport scheme still needs donor geometry,
available volume, capacity, time-step control and simultaneous flux accounting.

For an uncut row with area A and distance d, require every term's geometric
marginal to satisfy `sum_j A_ij = abs(c_i) * A * d`. Summing
`A_ij * (p_positive - p_negative) / (A * d)` therefore reconstructs the existing
row pressure gradient. Summing signed `A_ij * u_row` at each cell reconstructs
the corresponding pressure divergence incidence. These are algebraic identities;
f32 accumulation orders need not be bit equal. The helper's audit permits only
small f32 coefficient reconstruction error, without changing geometric areas.
Use the row distance for this operator identity, not each pair's center distance.

The caller must check `geometricSubfaceRowAudit(row).valid` before a geometric
transport stage uses the descriptors. Audit work is intended for topology
preparation and diagnostics; repeated nested enumeration is not a proposed hot
transport implementation. Audit failure is an explicit integration failure, not
permission to fall back to CM12 transport. One-sided exterior/boundary rows need
an explicit boundary descriptor and are not certified by this helper.

The audit requires the row's effective and static areas to agree. It does not
infer a clipped solid polygon from a scalar open fraction. Partly obstructed
faces, moving-wall swept geometry and solid-clipped PLIC remain separate work.
Even an aperture equal to one is only the current row's uncut classification;
sub-resolution solid geometry requires its own geometric authority.
