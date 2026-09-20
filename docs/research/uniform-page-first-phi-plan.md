# First page-driven compute target: vertex phi

Select `uvAdvectPhi` and `uvRedistancePhi` as the first substantive rewrite.
The browser measured this combined stage at approximately 28–49 ms in the
hose scene before the inflow scheduling fix. Re-measure after that fix;
these numbers are not a promised saving.

## Contract

Dispatch from a compact GPU list of fluid/support pages, not the bounding box
between them. Page demand must include conservative V, the signed-distance
surface band, inlet/drop support, characteristic travel, and solid contact
requirements. The redistance and optional agreement stencils currently require
six vertices of support beyond the cell work region. Account separately for
backward advection reads; a six-cell halo alone is not a CFL bound.

Use deterministic vertex ownership at shared page faces to avoid duplicate
writers. Build distinct required-write and required-read sets when the two
passes have different dependencies. Newly active pages must be initialized
before use. Retiring pages must clear stale negative phi and preserve the
positive-air contract; absence must never become an unintended wall or zero
level set. The global surface-volume correction and renderer still consume
phi, so preserve their contract even outside the newly dispatched pages.

The first change may retain dense textures for compatibility, but it must
replace rectangular dispatch with page-list dispatch and limit initialization
and clearing to entering/leaving pages. Persistent storage conversion is a
separate milestone, not something the dispatch rewrite should claim.

## Acceptance

- Match liquid/interface phi, surface geometry and conservative volume against
  the current oracle; demonstrate no page-seam artifacts or stale surfaces.
- Check moving fluid, separated pools, continuous/ramped inlet, off/on inlet,
  live drops and edits, and high-CFL backward traces.
- With fixed fluid/support geometry, enlarge the empty domain: phi workgroup
  counts and steady-state phi time should remain approximately constant.
- Report active page count, halo pages, processed vertices and cleared pages.
- Preserve at least 95% mini64 throughput in matched full-step measurements;
  keep a direct compact-scene path if sparse scheduling costs more there.

This is the selected next rewrite; it is not implemented by the inflow fix.

## Inflow prerequisite, 2026-09-21

Continuous inflow no longer renews the host's three-step dense dispatch and
pressure-lattice fallback on every advance. Its full-strength swept footprint
remains in the source census through ramping. Activation, a larger f32 timestep,
scene edits and live drops still trigger conservative rediscovery. Comparing
f32 timesteps matters: double-precision subtraction jitter must not masquerade
as growing source support.

The source census still scans the domain. Surface-deficit balancing also keeps
canonical domain launch dimensions: changing the partial-sum layout changed
floating-point reduction order, which amplified after a live drop. Isolating
this reduction restored bit-exact volume and phi across 32 advances. Removing
these two capacity costs needs an explicit stable page reduction/census design.

Validation: `npm run test:dawn:uniform-inflow` passes with bit-exact full-texture
V and phi comparisons at steps 1, 12, 24, 25 and 32 against the previous dense
launch policy (identical domain pressure plan). Dense fallback is 3/24 steps
before a live drop and 6/32 afterward, versus 24/24 and 32/32. All three arms
report zero clipped steps. A separate window-pressure arm demonstrates that
pressure can shrink (112×96×96 at step 12), but legitimately returns to the
144×96×96 domain as its support grows; this test does not assert numerical
identity between different pressure lattice plans.

The existing page-storage/visualization Dawn suite passes all three tests.
The garden storage parity test now fixes the pressure plan to `domain` in both
arms to isolate storage equivalence from asynchronously prepared pressure
lattice changes. Production `vinext build` passes. Type checking retains the
15 unrelated pre-existing errors.

A mini64 ABBA comparison of the prior versus fixed balancing dispatch, both
with pages32 and 24 measured full advances per arm, gives 46.18 ms versus
45.79 ms medians (100.9% throughput). This excludes presentation and validates
this incremental fix, not the full future page-first implementation. See
`uniform-inflow-mini64-2026-09-21.json` and
`tools/benchmark-uniform-inflow-dawn.ts` for the reproducible measurement.

Production browser confirmation after the fix: the hose SIM panel shows
`host-sized · 0 clipped · 3 dense`, with 7/45 volume pages demanded and a
62.5% rectangular work box. The sampled advance is 134.22 ms and vertex phi
is 33.1 ms; these are browser elapsed-stage observations, not a matched
before/after benchmark. Pressure has returned to domain capacity at this
sample. The large rectangular phi stage therefore remains the first target.
