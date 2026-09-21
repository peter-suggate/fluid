# Uniform Geometric persistent field pages

Implementation checkpoint, 2026-09-21. This advances the
[page-domain cutover](uniform-page-domain-cutover.md); it does **not** complete the
[GPU residency architecture](uniform-page-domain-design.md).

Uniform Geometric now selects pages unconditionally for all scenes. Saved
`volumeStorage`, liquid-window and host pressure-budget settings cannot select a
production dense/window path. The paper/reference method remains available as a
separate numerical reference.

## Implemented

- V, vertex phi, positive MAC velocity faces, transport, geometry and their ping-pong/history fields
  use physical texture pages. Velocity extension's fine scratch, integer source
  origins and every hierarchy level use the same field-addressing abstraction.
- All CM11a pressure levels and accepted/candidate pressure use physical pages.
  Stencils continue in logical coordinates across seams; this is one coupled solve,
  not separate per-page pressure solves.
- Main cell/vertex dispatch and the extension's finest dispatch share the page-domain
  catalogue. Pressure and coarsened extension levels traverse their own page layouts.
- GPU convergence gates can zero normal/recovery pressure-cycle indirect dispatches.
  There is no host pressure-demand feedback in the page production path.
- Surface-volume correction scratch is allocated during initialization, including
  when the runtime control starts disabled. Enabling it does not allocate during a
  fluid frame.
- Read-only dense publications currently adapt the paged fields to the existing
  renderer and diagnostic interfaces. Their GPU cost is included in the surface
  publication stage, not hidden after timing capture.
- Initial surface publication uses the page entry point's page-shaped launch.
- The accepted GPU catalogue publishes cell/vertex indirect dispatches and the
  pages overlay together. Mini64 uses the same page traversal as larger scenes;
  the host contiguous-lattice shortcut is removed. A GPU test shrinks, empties and
  restores a reordered partial-page catalogue and checks every dispatched cell
  and cleared overlay flag.
- Donor-sum finalization and surface-deficit balancing traverse domain pages.
  The latter previously launched rectangular counts against a page-coordinate
  shader; its partial-sum arena now includes partial-page padding. Empty-domain
  reduction counts are cleared before use.
- The production path no longer allocates the unused liquid-window census arena.

The simulation catalogue and transport records use 32³ pages. Texture fields use
16³ physical tiles, including partial tiles and pressure halo coordinates. Logical
field dimensions remain separate from atlas dimensions. Texture storage is therefore
paged, but its capacity is **still derived from the full authored lattice**.

## Numerical validation policy

The storage regression compares dense and paged backing through identical numerical
operators and shader interfaces. It checks exact V, transported phi, redistanced
phi, projected velocity and published surface through 64 frames, a liquid drop,
source removal/movement and redistancing toggles. The dense backing is a QA-only
constructor option; the production method adapter never forwards it.

The earlier test compared against the old direct-texture shader source. Adding
bounds branches and constant-folding logical dimensions changed floating-point
results, initially by a few ULPs. Those small differences can grow substantially
in the old iterative level-set/hose trajectory. Redundant branches were removed
from already-clamped phi loads, and logical dimensions remain runtime values.
A byte-exact backing test is not evidence that long trajectories match that older
compiled shader. The observational
`tools/probe-uniform-page-legacy-trajectory-dawn.ts` preserves that comparison and
reports volume, centroid and interface differences; its successful execution is
not a physics acceptance gate.

Physical boundary, separation, impermeability, mass-conservation and pressure
recovery fixtures exercise the page production path without relaxing their physical
assertions. The boundary suite's outer timeout allows cold compilation of several
paged shader dimensions; it is not a frame-performance allowance.

## Still required for the full cutover

- Replace the all-resident authored catalogue with the accepted GPU coordinate-to-slot
  directory, liquid/source support requests, growth and retirement.
- Replace identity atlas slot assignment and domain-sized buffer arenas with bounded
  pools independent of empty world extent. Derive coarse residency from the same
  accepted generation.
- Establish finite-band phi/support validity before retiring far-air pages. The
  earlier finite-phi experiments are not production implementations.
- Make field/history/source/time updates and publication atomic with residency
  success; preserve the accepted frame on capacity/support failure.
- Replace dense renderer publications, negative physical-boundary face buffers,
  terrain/solid indexing and remaining global
  reductions with consumers of the accepted page generation.

The UI deliberately reports “all-resident domain.” The pages layer shows that
catalogue, not fluid-only residency. Memory and work do not yet meet the requested
empty-world scaling contract. Performance tuning is deferred; these remaining
items are architectural work, not merely tuning.

## Regression receipts

The initial field-storage cutover passed all 24 Uniform Dawn checks (physical
boundaries, manufactured pressure, pressure-fault recovery and 64-frame storage
comparison) and all eight page unit checks. GPU catalogue publication adds a
separate Dawn ownership/overlay regression. After removing the small-scene shortcut, the physical boundary and manufactured
pressure checks passed. Follow-up donor/reduction routing fixes then passed all
11 checks in the final catalogue-publication, pressure-safety and 64-frame backing
comparison run, including live surface-deficit balancing toggles. The expanded
page unit command passes, and the production build succeeds. Typecheck still
reports 15 unrelated existing Sparse/tool errors.

The required `npm run test:dawn:sparse-cm12` did **not** pass: several lanes timed
out, Mini64 exceeded its unchanged 110 ms ceiling, and topology/transport lanes
failed. No Sparse CM12 solver files were changed. A clean detached checkout of
starting commit `d2580c07` reproduced `topology-page-budget`'s exact assertion
("each authored edit must actually commit its requested rung"). The clean
`outside-tank-symmetric-collapse` rerun timed out before reproducing its observed
missing-face fault; that fault and the other failures remain unattributed. These
receipts are not a passing canonical gate, and no assertions or timing ceilings
were relaxed.

The rebuilt production UI at `127.0.0.1:3001` initialized and advanced the garden
hose through 1.733 seconds, then was paused. The 45-page overlay was visible and
the SIM panel displayed per-stage timings (one observed window: 257.33 ms per
advance). No browser console errors were reported. This is a functional UI
receipt, not a performance benchmark or sparse-residency acceptance.
