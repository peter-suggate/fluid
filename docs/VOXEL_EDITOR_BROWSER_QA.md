# Voxel editor acceptance — 2026-09-08

## Environment and scope

Chrome on the local Mac, WebGPU, isolated development server at
`http://localhost:3001` from `/tmp/fluid-voxel-editor-qa`. The shared checkout was
being modified by other simulation and rendering tasks. Initial browser acceptance
used a stable source snapshot; the final pass uses a refreshed integration.
Results below distinguish these versions.
The browser and native Dawn runs used the repository's exclusive WebGPU lease
and did not run concurrently.

The suite consists of Build, Box, Sphere, Wall, Carve, Cut, Drill and Channel.
Implicit smoothing, flattening and arbitrary transforms are research proposals,
not shipped tools. Terrain was initially unavailable; the hero-garden follow-up
adds a bounded live overlay. Native acceptance passes; browser acceptance remains pending.
The saved 99-patch authored scene is retained at
`tests/fixtures/voxel-editor-live-scene.json` for repeatable native/browser checks.

## Browser evidence

| Check | Observed result |
| --- | --- |
| Build | Continuous raised voxel stroke appeared while fluid advanced. |
| Box | Rectangular extruded solid appeared during its drag. |
| Sphere | Voxel sphere stamp appeared; Mirror X produced two symmetric stamps. |
| Wall | Straight raised barrier appeared between drag endpoints. |
| Carve | A visible recess removed part of the authored ridge. |
| Cut | Rectangular cut removed part of the main block. |
| Drill | Cylindrical opening removed solid along the picked face normal. |
| Channel | Straight groove cut through the right edge of the main block. |
| Live history | Undo restored the carved ridge; Redo removed it again without resetting time. |
| Escape cancellation | Export before/after an active canceled stroke was byte-identical, 99 patches. |
| Command-Z cancellation | Synchronized with the visible pending-stroke status; before/after exports were byte-identical, 111 patches. |
| Pending transactions | Save scene, Export JSON, Undo and Redo were all disabled during an active pending stroke and became available afterward. |
| Numeric bounds | Width 100 clamped to 16; construction height 300 clamped to 128. |
| Stroke budget | Oversized mirrored Build reported “Stroke is full; release and start another stroke.” Accepted work remained undoable and time continued. |
| Exterior construction | Height −2 pillar through the exterior floor became visible; Undo removed it and Redo restored it while time advanced. The exact floor remained unchanged. |
| Fixed live world | Height −32 empty-space stamp was rejected with an explicit bounds notice; history remained unchanged and time continued. Height −16 was valid in the saved fixture's prepared domain. |
| Named save/open | Saved “Voxel editor QA 2026-09-08”; reopened from My scenes with the authored geometry restored and initial water configuration. |
| JSON round trip | Imported the recovered authored scene through the native file chooser; exported document retained all 70 patches exactly. |
| Baked terrain | All eight tools disabled with the reason to start a new voxel scene. |
| New empty scene | In the refreshed contextual editor, Box at depth 8 visibly inserted a sixth patch. Undo restored the five original shell patches; Redo restored the byte-identical six-patch document. Named save “Voxel new room redo” reopened by mouse with the exact document. |

The longest continuous fluid run after the solid-stop correction exceeded
227 simulated seconds. The observed running presentation rate was typically
7–8 FPS on this fixture; this is not a 60 FPS claim. Small native edit host work
measured approximately 2.5 ms, which is not an end-to-end worst-case bound.

Exact cancellation export SHA-256 values:

- Escape pair: `3f03c68c2f238388c8276390b83785d56e9ded48c68cc3c11a752ffa9dd6534a`
- Command-Z pair: `43e99275351dfc2f8cee4efc46114b594b873c834c5f8433a5ef0c966c56b98d`

## Contextual UX review and follow-up acceptance

The persistent panel mixed document actions, tool choice and detailed settings.
The revised host separates these into a compact Scene/Tools/history strip,
a transient chooser, and a selected-tool card. Plugins declare primary versus
advanced controls. Scene/object settings start collapsed and disappear while a
voxel tool is active. Selected oaks expose tree settings in their object context.

Current contextual browser checks passed:

| Check | Observed result |
| --- | --- |
| Resting layout | Compact Scene/Tools/history controls replaced the persistent editor panel. |
| Tool selection | Selecting a tool closed the chooser and entered EDIT. |
| Current solid subset | Build, Box and mirrored Sphere passed before the closed-cell arithmetic failure described below. |
| More and bounds | More exposed advanced settings. Width 100 clamped to 16 and construction height 300 clamped to 128; Mirror remained visibly indicated as a nondefault badge. |
| Outside dismissal | Clicking outside the Scene menu dismissed it; the exported document remained exactly unchanged. |
| New room/history | Box depth 8 was accepted as patch six. Undo restored the original five shell patches; Redo restored the byte-identical six-patch scene. |
| Named mouse reopen | Saved “Voxel new room redo”; reopening with the mouse restored the exact document. |
| Contextual trees | Add tree selected the new oak with collapsed Tree settings and Placement and object sections. |
| Rapid tree edits/history | Fork generations 3 → 2 → Undo 3 → Redo 2 completed without a false halt. The final tree visibly rendered; screenshot: `artifacts/oak-v2/editor/ui.png`. |

Still pending in the current contextual revision:

- Done disarms.
- LOOK hides active editing controls.
- All eight operations with running fluid; limits, cancellation and persistence.
- Current SVO presentation during mesh rebuild, without a blank canvas.

## Defects found and corrected during acceptance

- Sharpening could trace into a solid cell and terminate with an empty stencil.
  Added solid-stop checks for the current and next transport owner. The failure
  remains strict; its diagnostic now decodes fixed mass quanta as integers.
- A coarse planar SVO terminal hid newly inserted sampled descendants. Terminal
  splitting now preserves analytic siblings, clears the parent terminal and
  checks the complete allocation before mutation.
- Sampling the planar floor a second time changed its apparent outline. The
  exact global planar catalogue remains authoritative; sampled children contain
  only residual geometry.
- Fixed world and reserved topology capacity are checked before accepting an
  edit; rejected transactions preserve the accepted scene.
- Worker shutdown/failure now rejects outstanding edit validations so history
  and save controls cannot remain stranded in the pending state.
- Renderer-only scenes adopt uniform edits directly instead of alternating
  initialization keys and rebuilding continuously.
- The capacity ledger counts shared split prefixes once, and changed-cell
  bounds avoid refining an entire solid page for a one-voxel edit. Native verification passed for a tight shared-prefix arena and the exact default
  new-room Box insertion/undo. The refreshed browser also passed Box insertion,
  exact Undo/Redo and named save/reopen.

- Live retained-density preparation now visits changed geometry only and uploads
  compact cell ranges. Preflight checks numerical/work limits before scene
  acceptance and reuses the validated proposal at commit. Immutable sparse cache
  overlays preserve concurrent preparation snapshots. Focused native checks
  passed 3/3 after integration, with a small host edit measured at 4.50 ms.

## Automated evidence

- Integrated editor, geometry, transaction, worker, renderer lifecycle, preflight
  and diagnostic CPU suite, including bounded retained edits and contextual
  presentation changes: 71/71.
- Terminal split CPU suite: 3/3, including exact finite stage-floor clipping.
- Native terminal split: 1/1; descendant traversal, sibling preservation,
  allocation/backlinks and rejection immutability checked by GPU readback.
- Snapshot live boundary/presentation: 2/2; same resident world, open fraction
  `1 → 0.875 → 1`, time advanced to `0.1 s`, finite fields, no GPU validation errors.
- Snapshot recovery fixture: 90 steps / 3 simulated seconds with finite fields.
- Current-source isolated production build passed after the contextual editor
  and presentation fallback changes. The snapshot records 816 source-file hashes in
  `/tmp/fluid-voxel-editor-qa/voxel-editor-qa-source.json`.
- Final canonical run exhausted the unchanged 180 s budget: five passes, six
  timeouts and six unrun lanes. No numerical assertions failed; clipped topology
  transfer measured exactly zero mass error. Full log:
  `/tmp/voxel-editor-post-fallback-canonical.log`.
- Native presentation fallback passed 1/1, exercising the production pipeline
  and checking exact rendered pixels.
- Current live boundary/presentation fixture passed 2/2 after retained-density
  initialization was corrected: same world, open fraction `1 → 0.875 → 1`,
  time `0.1 s`, host edit approximately `4.8 ms`.

The refreshed integration later halted the saved 99-patch scene at frame 28,
before any new edit, with `RETAINED_DENSITY_INTEGRAL`: reconstructed mean zero,
accepted mean `1/65536`. Native readback confirmed that a dynamic leaf outside
the initial grid had received transported mass but had never initialized its
retained support. Ordinary evolution now uses the existing bounded leaf-indexed
update. The exact fixture passed 90 frames / 3 seconds, including the former
failing frame, with strict diagnostics and finite fields. Together with live
boundary/presentation checks, this run passed 3/3 (host edit about 4.44 ms).

Earlier canonical runs failed D4 symmetry and multiple timing lanes.
An earlier current-checkout focused boundary test encountered
`EMPTY_DEFICIT_STENCIL` at the same frame/owner as a no-edit baseline. Subsequent
initialization and frontier fixes passed fresh live-boundary verification; that
earlier failure must not be presumed current. No timing
ceiling or strict failure predicate was weakened. These results do not establish
broad simulation acceptance or displaced-liquid conservation.

## Closed-cell arithmetic correction and remaining browser pass

The refreshed solid sweep halted at 14.4667 simulated seconds, frame 434,
with `INVALID_CONSERVED_VALUE` in `advanceRetainedDensitySupport`. Build, Box
and mirrored Sphere had passed. A deterministic native replay reproduced the
negative retained integral immediately after the mirrored Sphere.

GPU initialization had overwritten an exact closed-cell open measure with
`1 - f32(255) / 255`, whose fused arithmetic produced a tiny negative value.
Initialization now preserves uploaded exact moments; sparse complements subtract
integer q8 values before division. Strict diagnostics were not weakened.
The exact 149-patch live sequence passes 450 frames / 15 simulated seconds.
The GPU proof checks all 256 fractions and exact zero at closed supports.
The 149-patch replay is historical evidence under the earlier accepted-solid
contract. The new wet-overlap acceptance check can reject original wet inserts;
it must not be presented as a current replay pass without an explicit rerun.
Logs: `/tmp/fluid-retained-q8-replay-generation-gpu-1.log` (replay and generation
transfer pass; its isolated proof harness error is superseded by the next log)
and `/tmp/fluid-retained-q8-proof-gpu-3.log` (2/2 pass).
The complete current browser sweep must still be rerun.

## Fluid shapes and hover extension

Three plugins provide Water ball, Water cube and Water torus, each with a Remove
water toggle. Scene-aware defaults place the shape above authored pool water
when space permits; height is visible. Release commits once, while previews and
canceled gestures leave fluid and scene history untouched. A dry scene now has
an explicit Scene → Enable water setup action. Live drops are transient and
are not saved as a fluid checkpoint or reversed by scene Undo.

Focused native acceptance passes 1/1 in 16.6 seconds. Ball mass changed
`271.373 → 0`, cube `512 → 0`, and torus `61.942 → 0` across add/remove pairs.
The torus center remained empty, solid cells stayed dry, rejected bounds and
work-budget edits left the field and generation unchanged, and the same world
advanced from `0.0333` to `0.0667` seconds around a live edit.
Log: `/tmp/live-fluid-shapes-dawn.log`.
An initial test failure used an authored-only diagnostic projection, omitting
GPU-grown pages; reading runtime leaves corrected that oracle without changing
production behavior.

The integrated editor CPU suite passes 86/86 in
`/tmp/fluid-editor-integrated-cpu.log`; fluid backend/worker checks separately
pass 7/7 in `/tmp/live-fluid-backend-final-cpu.log`. Faint dotted EDIT object
hover bounds pass 16 focused checks; LOOK and active gestures suppress them,
while selected outlines and idle tool previews retain their distinct styling.
The hover integration build passes in `/tmp/fluid-editor-hover-build.log`.

Browser checks for fluid shapes, hover outlines and the dry-scene Enable water
shortcut remain pending. Live terrain overlays are being added to remove the
blanket terrain restriction in hero-garden-hose-x10; terrain native and browser
verification are also pending. See [the fluid editor contract](LIVE_FLUID_EDITOR_PLAN.md).

## Hero garden terrain follow-up

The reported `hero-garden-hose-x10` restriction had two causes: this stress scene
intentionally disables its fluid system, and the initial solid-tool suite blocked
all terrain scenes. **Scene → Enable water** provides explicit fluid setup. The
solid-tool fix preserves the renderer's immutable refined terrain heights and
updates a reserved ordered fill/clear overlay instead of rebuilding that field.

CPU tests in `tests/live-terrain-overlay.test.ts` pass 4/4: the actual hero x10
scene offers its eight solid tools, terrain picking edits the canonical solid
surface, overlay packing/Undo preserve operation order, and renderer publication
retains the same source through fill, clear and Undo. Sampled terrain topology
splitting passes 8 CPU checks. Its native fixture passes 1/1 in 1.35 seconds:
fill above terrain, deep carve below the original height, and Undo update actual
GPU occupancy while preserving the SVO source. Log: `/tmp/live-terrain-overlay-dawn.log`.
The hero browser rendered and enabled all eight solid tools, but its first Box
drag timed out dispatching a pointer event after 10 seconds and did not enable
Undo. Input responsiveness is under investigation. This is not a browser pass
for hero fill/carve/Undo or a canonical regression pass.

## Empty-scene fluid browser defects under investigation

New → Enable water followed by ball drops advanced the simulation clock but
showed no visible water. The renderer allocator read an authored record before
handling runtime pages; an empty authored atlas underflowed that address.
Additionally, live topology publication did not allocate renderer pages before
publishing their fields. Both corrections await focused native publication
checks and browser verification.

A subsequent Box through the injected water halted at frame 1883 with
`RETAINED_DENSITY_INTEGRAL`. Live static-boundary refresh omitted runtime-grown
leaves. Refreshing those leaves addresses stale apertures, but static closure
also needs a conservation check: avoiding a halt alone does not establish that
displaced water is preserved. Native before/after mass assertions and a safe
acceptance path are in progress. Solid insertion into occupied water now requires
an atomic rejection; conservative displacement is not implemented. A bounded
asynchronous GPU receipt precedes accepted publication while the simulation
continues. Cancellation of a carve can therefore reject restoration if water
has entered that space; in that case the accepted edit remains undoable and a
notice explains the rejection. Current CPU acceptance/picker checks pass 27/27;
native and browser verification results are recorded below.
Evidence is preserved in
`artifacts/voxel-editor/empty-fluid-retained-failure.json`,
`artifacts/voxel-editor/empty-fluid-box-failure.json`, and
`artifacts/voxel-editor/empty-fluid-browser-failure.png`.

### Revised native acceptance

- Empty-scene presentation passes 2/2. The exact browser ball descriptor
  publishes four active pages with 136 wet and 1,912 air samples at time zero;
  resident identity is unchanged. Strict simulation health and uncaptured GPU
  errors are asserted. `/tmp/empty-fluid-presentation-dawn.log`.
- Shape/mass acceptance passes 1/1. Wet solid insertion rejects with identical
  density, open fields, topology generation and time. Dry insertion into the
  same runtime page preserves represented mass exactly at
  `116.18636655807495` before publication, after publication and after the next
  step. Acceptance plus publication measured 5.43 ms; the clock advances from
  `0.0333` to `0.0667`. `/tmp/live-fluid-shapes-mass-dawn.log`.
- Live boundary acceptance passes 2/2. Wet insertion rejects unchanged; a dry
  native-cell fill and Undo produce open fractions `1 → 0 → 1`, with the same
  world and a continuing clock to `0.1 s`. Edit time was 6.45 ms. Repeated
  renderer-only SVO edits also pass. `/tmp/voxel-editor-live-boundary-mass-dawn.log`.
- Latest integrated CPU run: 78 pass, four native tests intentionally skipped,
  zero failures. `/tmp/fluid-editor-final-integrated-cpu.log`.
- Integrated production build passes. The QA snapshot records 1,361 source-file
  hashes. `/tmp/fluid-editor-integrated-acceptance-build.log`.

These native checks supersede the earlier empty-publication failures. The
corresponding browser checks remain pending.

### Browser follow-up after native integration

- Hero Box fill, visible Undo/Redo, and a deep Cut with visible Undo pass.
  First hover still stalled; an exact CPU benchmark isolated 23.55 seconds
  spent regenerating the scenery catalogue on the UI thread. The worker now
  transfers its already generated exact catalogue. Clone/adoption and all cold
  probes measured 79 ms; 18 CPU checks pass. A fresh browser pass is pending.
- Faint object hover bounds pass for both a stone and a tree. Browser DOM/CSS
  checks confirm 12 edges, a `2px, 4px` dash pattern and 38% opacity. Bounds coexist
  with an idle tool preview; LOOK suppresses them. Done removes the tool card.
  Screenshot: `artifacts/voxel-editor/hero-terrain-hover.png`.
- All three fluid additions are visible in the browser, including the torus
  opening. Wet Box insertion rejects with a useful notice while preserving the
  visible water and timeline. The clock subsequently continues past 24 seconds.
- Fluid removal exposes a remaining display defect: removed ball/cube surfaces
  remain visible even after the clock advances, until a later addition replaces
  the mesh. A native field-and-draw-count regression is in progress. Do not count
  add/remove browser acceptance as complete.
- Hero export succeeds (4,615,727 characters, two authored solid patches), but
  the named save exceeded browser quota. Compact JSON and pooled identical
  save/autosave documents now pass seven persistence regressions, with no eviction
  on quota failure. Browser save/reopen remains pending. Export formatting stays
  unchanged. `/tmp/scene-library-compatibility.log`.

The removal follow-up now passes 2/2 natively. Fluid fields and all four active
pages were already air; production mesh classification incorrectly retained
2,304 old vertices because it recognized only zero-crossing samples as valid
publication. A valid compact air page now certifies the empty surface. The real
classification/scan pipeline produces `2304 → 0` vertices on removal; an invalid
generation still retains the previous mesh. No time advance or world replacement
is required. `/tmp/empty-fluid-removal-dawn.log`. Browser recheck remains pending.

### Latest canonical gate

The exact `npm run test:dawn:sparse-cm12` command finished in 180,025.7 ms:
five passes, six timeouts and six unrun lanes. No completed lane reported a
numerical assertion failure. The gate is **not passing**; no timeout, ceiling or
lane was weakened. `/tmp/fluid-editor-final-canonical.log`.

An optional compilation-profiler preload hung npm before the suite started;
that process was stopped and the plain command above was run. Browser rendering
remained off throughout the native suite. A later worker cache-preservation fix
is covered by `/tmp/worker-terrain-cache-cpu.log`; it preserves the already
compiled terrain across structured clones without changing simulation kernels.

### Fresh browser follow-up: terrain persistence and all tool geometry

- Named `Hero terrain final QA` save succeeded with the compact pooled library;
  reopening completed and the new terrain pillar remained visible.
- Cold first-pointer test still timed out after the catalog change. A separate
  vessel-rim ray query eagerly called both procedural grid generation and a
  terrain ceiling scan. Bounded procedural point sampling now replaces those
  cold authoring calls; fresh CPU vessel pick fell from 23,610 ms to 0.57 ms.
  A second first-pick shell enumeration was replaced with exact constant-space
  slab/sphere predicates (full hero gesture begin 92 ms to 25.6 ms).
  Fresh browser verification of these final two changes follows separately.
- Empty room with enabled water: Box accepted during playback at about 6 s,
  Build accepted on retry around 14.8 s. Some live fill samples rejected stale
  water checks, even with no water; this remains an acceptance race being fixed.
- Paused at 23.5333 s, Sphere and Wall added visible geometry; Carve, Cut, Drill
  and Channel visibly removed the intended sections. All eight solid operations
  have now been individually exercised on the current geometry implementation.
  Saved and exported as `Eight tools final QA`.
- Water ball (6), cube (6), torus (outer 8/tube 2), height 10: each added visibly
  and each Remove action immediately cleared its entire mesh at unchanged
  23.5333 s. The stale-empty-mesh browser defect is resolved.
- A torus dropped during playback fell onto the edited floor/structures as the
  clock advanced to 25.2 s; no simulation halt. Screenshot of all three fluid
  primitives plus eight-tool geometry: `artifacts/voxel-editor/all-tools-fluid-shapes.png`.
- Fast New then Enable water before the dry source was ready stalled initial
  loading beyond 90 s. Waiting for the dry scene then enabling completed in
  roughly 10 s. The Enable action now waits for current dry-scene readiness;
  browser verification of that guard is pending.

### Cold garden input: final browser result

Fresh `hero-garden-hose-x10` loaded with no canvas interaction before the test.
Box depth 8, first drag `(290,650) → (330,670)` completed in the CUA call's
1.005 s (including the drag and AX snapshot), with no dispatch timeout. The next
screenshot showed the new terrain wall. Undo removed it and Redo restored it.
Idle stone hover still showed dotted bounds alongside the tool preview.
This resolves the reproduced >10 s first-pointer UI stall; it is not a claim
about steady rendering rate, which remained roughly 2–6 FPS on this stress scene.
Screenshot: `artifacts/voxel-editor/hero-cold-picker-fixed.png`.

The current UI clamps width 100→16, depth 100→32, and construction height
300→128. Mirror X and height badges remain visible with More closed. An oversized
mirrored stroke reported “Stroke is full; release and start another stroke.”
Pending state settled and history controls returned. Done/LOOK removed the tool
card and hover decoration.

Fresh standard `hero-garden-hose` also passed its first Box drag with no prior
canvas interaction: CUA drag plus AX completed in 1.286 s and the next frame
showed the wall. New scene immediately followed by opening Scene correctly
showed Enable water disabled with “Wait for the scene to finish loading before
enabling water.”

### Atomic GPU acceptance native result

The stale overlap-receipt race has been replaced by GPU inspection, conditional
bounded scatter, and aperture refresh in one command buffer. Mapping its result
does not pause ordinary physics. Postcommit faults retain the accepted document
and the fault rather than reverting only the document.

- `/tmp/atomic-solid-accepted-mass-dawn.log`: 1/1 PASS, all three fluid shapes,
  occupied-water rejection, and dry insertion with another ordinary physics
  step submitted before receipt mapping. Acceptance 10.57 ms; clock
  0.0333→0.0667 during acceptance and 0.1 on a further step. Accepted mass
  116.1863665581→116.1863665581→116.1862944365; unchanged 1e-4 tolerance passes.
- `/tmp/atomic-solid-live-boundary-dawn.log`: 2/2 PASS; aperture 1→0→1,
  same resident/timeline, dry acceptance 9.35 ms.
- Earlier apparent +6 mass gain was a test metric error: the growth diagnostic
  sums max(A,B), not the accepted density bank. The corrected oracle sums the
  authoritative runtime-overlay diagnostic image. No physics threshold changed.
- An earlier native timeout was also a fixture error: advanceTo returns false
  when the requested time is already reached. The test now advances to a later
  real step instead of retrying that completed time forever.
- Postcommit lifecycle CPU 16/16 and bounded scatter CPU 2/2 pass. Final browser
  verification of the atomic path follows below.

### Atomic browser follow-up: history bypass caught

On the saved eight-tool room with moving injected water, Build, Box, Sphere,
and Wall all accepted on the first attempt while the clock advanced from
6.0667 to 21.4 s. Undo Wall succeeded at 24.5333 s. Redo Wall exposed a bypass:
history republished geometry without the required GPU solid acceptance, causing
“Solid insertion needs a current water-overlap check before publication” and a
runtime halt at 27.9333 s. This remains a blocking browser defect until history
restoration is routed through the same acceptance transaction. Direct tool
acceptance passed; the full live history workflow has not yet passed.

### Production-path acceptance follow-up

Undo/Redo now invokes the viewport's registered atomic acceptance callback before
changing the document or history stacks. Compare-pane adoption requires its own
acceptance. The worker now reuses accepted history images across structured clone
and handles shorter history targets as replacements. CPU verification is 101
passed, 5 native tests skipped (no Dawn module in the CPU invocation), zero failed:
`/tmp/fluid-editor-history-final-cpu.log`. Representative native verification is
tracked in `VOXEL_EDITOR_ACCEPTANCE_COVERAGE.md`.

The exact required `npm run test:dawn:sparse-cm12` was rerun after atomic acceptance.
It remains red: 5 lanes passed, 6 timed out, and 6 did not run after the unchanged
180-second suite budget was exhausted. No completed lane reported a numerical
assertion failure. Log: `/tmp/fluid-editor-atomic-canonical.log`. This is not a
passing full simulation regression claim; no ceiling or lane was weakened.

### Final guarded-history browser result

Reopened `Eight tools final QA` on the final history/cache/comparison source.
Injected a ball at height 10, then drew a depth 3 Wall. Its first segment accepted;
a later segment correctly rejected overlap with moving water, preserving the
accepted segment and an undoable stroke. The runtime continued. Undo Wall at
12.6667 s and Redo Wall at 13.9333 s succeeded; a subsequent Carve and Undo Carve
completed by 21.2 s. No runtime halt or device release. This supersedes the blocking
history failure above. Screenshot: `artifacts/voxel-editor/live-history-verified.png`.
Browser was navigated to about:blank before releasing the GPU lease.

The representative Dawn suite passed 6/6 in 34.40 s. Its final strengthened production
lane passed separately, including exact untouched GPU occupancy, baseline mass
restoration, cancellation GPU/history identity, and wet Redo rejection. See
`VOXEL_EDITOR_ACCEPTANCE_COVERAGE.md` for coverage and headless limitations.
Exact cloned-scene comparisons also reduce Hero history comparison from 52.43 ms
to 4.30 ms in a 20-iteration CPU probe; focused comparison/history tests 4/4 pass.

Final consolidated CPU rerun: **103 passed, 5 native tests skipped, zero failed**
in 2.10 s (`/tmp/fluid-editor-history-final-cpu.log`). Final snapshot production
build passed (`/tmp/fluid-editor-history-final-build.log`); `git diff --check` passed.
