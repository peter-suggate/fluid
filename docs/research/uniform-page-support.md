# GPU support requests and the finite-phi blocker

## Implemented support mechanism

`UniformPageSupport` feeds `UniformPageGeneration` in the same command buffer.
It now summarizes each accepted page with one workgroup: half-open occupied bounds,
4³ tile masks, physical volume, maximum full-cell fraction, interface bounds,
phi/V disagreement count and signed velocity intervals. Only accepted slots are
written; unused pool metadata is neither scanned nor cleared by classification.
The reduction order is fixed within each page. Negative/nonfinite V is a fault.

Support starts at actual occupied bounds rather than the complete page. A bounded
local closure includes velocities from every accepted page intersecting the predicted
read neighborhood, including air-only pages. A distant fast body does not contribute
unless that neighborhood reaches it. Forward destination bounds carry role 32;
backward-query/stencil support carries role 8; original seed roles remain 1/2 and
source protection is 16. Roles are merged by the generation planner.

The current conservative read box includes forward destinations followed by backward
sampling, so the *read* extent can be symmetric even with one-way motion. This is
intentional: directional destinations alone do not bound backward queries. Local
page coordinates stay integer until a small within-page offset is needed. Half-open
cell endpoints use `floor(lo / edge)` through `ceil(hi / edge) - 1` inclusive.

Closure uses at most eight GPU iterations and fails rather than silently limiting
reach. The request header reports count, fault, and maximum iterations. Faults are
1=request overflow, 2=signed coordinate overflow, 3=invalid fields, 4=closure limit.
The request emitter and topology planner remain serial. Duplicate requests can still
exhaust the request budget before the unique resident budget; no truncation is used.

V's units must be specified at construction: physical volume, full-cell fraction,
or open-cell fraction with an explicit aperture field. Uniform Geometric's V uses
full-cell fractions (its solid-excess code compares V directly with open fraction),
so its eventual adapter must multiply by cell volume exactly once, without multiplying
by aperture again. Physical sums are diagnostics, not yet cleanup authorization.

The caller supplies actual-stage spacing, substep duration, conservative extra travel
allowances, and the composed operator read reach. The default reach still covers only
the elementary phi advection/redistance calculation. Missing pages use the declared
ambient initialization velocity for prediction; freshly initialized pages must have
zero V and phi outside the seed band, preventing recursive ambient seeding.

**This remains a predictor, not an end-to-end support certificate.** Source commands
must include their swept/interface footprint and source-velocity uncertainty. Pressure
or extension must not change sampled velocity after prediction without revalidation.
Actual operator sample checks, finite-phi validity, contact and correction contracts,
and whole-fluid rollback are still required. Retaining phi below the positive band
remains conservative until that field contract replaces it; disagreement is counted,
not used as permission to discard fluid.

Dawn/Metal checks cover:

- GPU classify → request → allocate → initialize → publish without host count reads.
- A corner seed requiring eight pages, and an interior seed requiring one page/one tile.
- Deep negative phi with zero V retained conservatively.
- A million-page separation between a quiet pond and fast jet without shared dilation.
- Neighboring air velocity, backward reads and an unresolved multi-page velocity chain.
- Signed-coordinate extrema and exact page endpoint semantics.
- All three physical-volume conventions, including anisotropic cell dimensions.
- Request, resident/transition and coordinate overflow, and NaN rejection.
- Empty retirement and remote source initialization.

The focused generation/support/publication gate passes 13 tests. This component is
not wired into production fluid operators. The UI remains on the all-resident
cutover. No production scene speedup or memory reduction is claimed.

## Rejected direct phi cap

A controlled experiment capped initial phi, advection, redistancing and surface-volume
correction to ±16 hMax while leaving the dense backing and all operators otherwise
unchanged. The unbounded solver was the control. The cap is wider than the geometric
read bound for an **exact SDF**, but the evolved field does not satisfy that assumption.

Garden hose fields matched through frame 9. At frame 10, near-interface advected phi
still matched exactly, but final phi differed by 1.238 cells. At frame 12 the volume
field L1 difference was 8.196% of reference mass, and the maximum near-interface phi
difference was 3.503 cells. These are field differences, not measurements of lost
mass. This failed the predeclared 0.5% volume-field / 0.1-cell interface comparison.
The diagnostic run printed all samples with threshold assertions disabled; its TAP
success is **not** an acceptance pass.

The cap was removed from runtime code and was not promoted. Evidence is retained:

- `uniform-finite-phi-diagnostic.txt`: per-frame comparison capture.
- `uniform-finite-phi-rejected.patch`: experimental runtime changes.
- `uniform-finite-phi-probe.ts.txt`: probe source; copy into `tests/` after applying
  the patch to reproduce. `PHI_BAND_DIAGNOSTIC=1` captures the first 12 frames without
  threshold assertions; omit it for the 64-frame acceptance test, which fails.

The first observed discrepancy is between the near-interface advection and final-phi
checkpoints, consistent with the redistance/correction stage depending on changed
far-air values. It does not isolate a specific Newton query or distinguish every
redistance effect from correction. Further stage capture is required for that claim.

The next numerical step is to reconstruct a valid finite distance band from the
interface, including its solid-contact treatment, and explicitly validate support.
Increasing a guessed cap is not a substitute for that contract. Allocation and
support generation are now available independently for that implementation.
