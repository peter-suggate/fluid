# Dynamic topology freeze

The UI toggle retains each accepted brick, its span, and its cell size. A dry
catalogue entry may become resident, and the sparse world may allocate new
support. New bricks use the normal incoming demand and refinement bounds,
with the frozen neighbours acting as fixed constraints on 2:1 grading. Once
accepted, a new brick's size is retained too. Releasing the toggle restores
adaptation and retirement.

The previous implementation disabled frontier allocation, synthesis, and
activation along with resolution changes. Frozen mini32 therefore reached a
missing dry corner at step 5. Planning now separates accepted resolution from
new residency: the existing grading dispatches close the new receiver's
resolution without modifying a frozen neighbour. There is no symmetry pass,
extra averaging, or scene-dependent solver rule.

The GPU's page-local growth graph supports fine/fine seams. When a new page
meets a frozen coarse host, it requests the existing generation compiler's
mixed-ratio graph before becoming active. The transfer preserves the old
accepted cells. A new dry receiver may grow to the smallest dyadic macro
patch compatible with a frozen macro host; it cannot absorb existing accepted
coverage. Only the newly allocated dry extent may introduce zero-density air
in the otherwise strict conservative transfer.

Frozen liquid editing prepares support before applying the interaction.
Queued drops and hose doses are applied after support is ready; a later edit
invalidates the earlier readiness receipt. The freeze policy reaches the
interaction immediately, including when the timeline is paused. Unfrozen
editing retains its existing transaction.

Compiling signed world pages also exposed an independent transport-owner
bug: template-owned leaves were assumed to be clipped to the original tank.
The transport execution image now reads their actual unclipped-geometry flag.

Candidate face transfer had a related coordinate fault. Its donor lookup
clamped a new world face into the original tank. Beyond the positive tank
extent, the donor rectangle ended behind the integration cursor, causing
millions of tiny numerical progress increments. The lookup now uses the
actual signed donor, with a normal-side interior fallback on an accepted
coverage boundary. Missing air returns its own voxel extent. This preserves
normal flux and advances by the actual overlapping rectangles.

Presentation uses the same geometry distinction. A world page and its air
apron keep their signed coordinates; clipped authored boundaries retain their
existing continuation. The shared policy applies to cache fills, fallback
sampling, and vertical reconstruction brackets.

Validation:

- Mini32: twelve frozen steps admit the missing corner, retain every accepted
  span and rung, and place nonzero fluid in the new support; unfreeze then
  allows existing cells to refine. Passed.
- CPU policy and transfer tests cover frozen constraints, macro receiver
  allocation, overlap rejection, and explicit new-air transfer. Passed.
- The final CPU integration batch passes 32 tests across freeze controls,
  generation policy/transfer/planning, source leases, activity masks and
  retirement. Paused edits also receive updated runtime thresholds before
  their next numerical step.
- Two paused coarse-host drops on opposite signed world boundaries create a
  mixed-ratio replacement and preserve the original hosts. The second drop
  supersedes an in-flight preparation. Eight subsequent steps preserve total
  mass at 1205.17338180542 fine-cell volumes and remain stationary to about
  2e-12 fine cells/s. Both external halves retain 30.68919944763 fine-cell
  volumes. Passed after repairing the candidate-face coordinate fault;
  previously a candidate prefix alone took 86.5 seconds.
- Emitted candidate-face WGSL tests both signs of every axis, both accepted
  coverage boundaries, constant normal flux, missing air, and geometric
  progress. Passed. The same test checks clipped-boundary continuation and
  unclipped signed air coordinates.
- The signed drop's published scalar was formerly constant +0.10712 across
  its external page, erasing the surface. It now spans +0.19995 in far air to
  negative values inside the drop, including -0.05310 and -0.07214. The
  regression requires both far air and the liquid/air crossing.
- The closed-tank lid reconstruction, affine reconstruction, and fixed-domain
  frozen mini32 controls pass. The six focused GPU lanes passed in 58.3
  seconds including compilation. The new regression files are included
  in `npm run test:dawn:sparse-cm12:coarse-first`.
- Floor/ordinary continuous column receipts and the uniform B1 column
  publisher also retain their height fields under negative, positive, and
  mixed XZ translations. Their remaining tank clamps now share the querying
  brick's geometry policy. Both emitted-WGSL column tests and the unchanged
  closed-lid control pass (three lanes, 1.4 seconds).
- The native-volume planar-waterline regression passes at every coarse rung
  after adding the shared coordinate helper to its standalone WGSL fixture.
- The final canonical Dawn gate ran serially on Metal on 2026-09-07 after
  these changes. Six lanes pass, including four-second mini32 correctness,
  clipped transfer and mixed-ratio topology. Symmetric expansion still fails
  its horizontal density-symmetry assertion. Page-budget and hydrostatic
  lanes exceed 30 seconds; mini32 performance measures 59.7688 ms against the
  unchanged 40 ms ceiling; mini64 performance exhausts the remaining suite
  time. The six tail lanes do not run before the 180-second budget expires.
  The integration gate is red; focused freeze validation does not establish
  general solver correctness. Receipt:
  `artifacts/minidam32-frozen/dynamic-freeze-final-gate.json`; full log:
  `/tmp/fluid-dynamic-freeze-final-gate.log`.
