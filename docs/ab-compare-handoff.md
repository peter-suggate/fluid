# A/B compare: two panes, one clock

STATUS: LANDED 2026-08-29 (WP0–WP4 in the working tree of this date, all
Opus 5 subagent work; not yet committed). Deviations from the design below:
the diff strip's stats block reads `t` from `simulation.paneClocks()` because
`runtime.simulationTime` is the host *minimum* published into every pane;
ms/step shows `—` unless a performance instrument is open (the FPS meter is a
DOM ref, deliberately not a store); cross-pane pins mirror by normalized aim
while View is linked, not by resolved world cell (TODO at
`lib/core/compare/compare-model.ts`, needs the viewport's pick); the compare
record lives in `shell-store` (page-level) with the four link groups as
`links`; `b.link=` lists the *unlinked* groups. WP5 (wipe / pixel diff) not
started. Gates: `tests/{pane-lease,simulation-lockstep-host,compare-model,
compare-divergence}.test.ts` (47 tests) + in-app checks on the dam break
(uniform lane; the CM12 resident WGSL was mid-edit and not compiling on the
day). Known: an app dev server on the working tree is unusable while a fleet
is editing — rsync the tree into the scratchpad and run `next dev` there.

## The ask

See two differently configured simulations — different solver, different
params, or different scenes — side by side, advancing in lockstep, so an A/B
is judged by eye rather than by a checksum in a terminal. Today every A/B is
headless and sequential (`tools/run-sparse-cm12-long-run-ab.ts`,
`tools/benchmark-power-dam-ab.ts`, …); there is no in-app compare of any kind.

## The organizing principle

**B is A plus a diff.** Opening the second pane does not fork a second
independent world; it declares "the same experiment, except…". The pane-B
config is stored as a set of overrides over pane A's `QueryState`
(`lib/core/url-state.ts:76-84` — `{methodId, quality, overrides, presetId,
scene, ui}`, which is already exactly one pane's config). Consequences that
fall out for free:

- Opening B copies A — the diff starts empty, so B *is* A.
- Editing A while the diff is empty edits both; editing B records a diff.
- "What differs" is a first-class readout (the diff), not a comparison of two
  documents after the fact.
- The URL stays readable: pane B is `b.method=…`, `b.param.*`, `b.scene.*`,
  `b.quality`, a handful of keys, never a second full scene.
- A swap (`b.` ↔ base) and a "promote B to A" are diff operations.

A *different scene* in B is just a diff whose key is `sceneId` (+ whatever
scene-level keys). It's the same mechanism at its coarsest.

**One clock.** There is one target clock and one transport. Lockstep is not a
feature toggled on; it is the only way compare mode runs. The clock advances
to `T+dt` only when both panes report completion of `T`.

**The scene is still the hero.** Compare mode is two `.viewport-shell`s in
one `.lab-shell`, separated by a draggable splitter. Nothing docks. Chrome
stays per the hero-scene plan (`docs/hero-scene-ui-plan.md`): the ring is the
route; the only permanent additions are the splitter and a slim **diff
strip** along the seam.

## Entering and leaving

- **Empty-space ring** gains a **Compare** wedge. Keyboard: `\`
  (backslash — the vertical bar, "split"). Both toggle.
- Opening: the current pane becomes A, B mounts to its right with an empty
  diff, camera linked, clock paused-if-paused/running-if-running. Because the
  diff is empty, B shows the same picture as A on the first frame — that is
  the sanity check that the mode works.
- Closing from the ring (**Close compare** on either pane's empty-space ring)
  keeps A. **Keep this one** on pane B's ring promotes B (A ⊕ diff → A, diff
  cleared) then closes. **Swap** exchanges them.
- URL: presence of any `b.*` key (or a bare `b=1` when the diff is empty)
  means compare mode; a reload restores both panes. `b.*` keys are listed
  in the overrides chip under their own "B" group, never counted as edits to A.

## Configuring B

Right-click in pane B → the same rings, the same flyouts. Every flyout that
writes config writes through the **session** it is mounted under (§ realm
below); in B that write lands in the diff. The FIELD flyout's solver popover
therefore "just works" in B — choose sparse CM12 in B and uniform in A.

The **diff strip** on the seam lists each differing key as `key: A → B`, each
row with an `×` (drop this override — B falls back to A for that key) and a
`⇄` (move the override to A instead). Empty diff shows one muted line:
"identical — edit B to diverge".

## Coupling: what the two panes share

The experiment is about the *solver*; everything about how you look at the
water should be identical across the panes unless you say otherwise, or the
eye compares two viewpoints instead of two solvers. So the whole `ui` half of
`QueryState` (`url-state.ts:88-135`) is **linked by default**, and the rule
is the same diff rule: a linked key has no `b.` entry and both panes read A;
an unlinked key may carry a `b.` override. Linking is per **group**, each
with a padlock on the diff strip:

| Group | Keys | Linked means |
|---|---|---|
| **View** | `camera.*` | one `CameraState` fed to both draws; orbit/pan/zoom/keyboard framing in *either* pane moves both |
| **Cut** | `gridOverlayAxis`, `gridOverlaySlice`, `gridOverlayMode`, `gridOverlayLensPhase` | the same slice of the same axis in both panes — the whole point when comparing brick/tile structure, pressure, dirty state, or a stage lens across two methods |
| **Instrument** | `sceneOverlay` (pipeline, diagnostics, fields) | opening an instrument on one pane opens it on both, each reading its own session's data, so the two pipeline overlays sit side by side stage-for-stage |
| **Look** | `svoStageView`, shadows/AO/silhouette, cone tracing, primary traversal, `svoRenderTuning`, `quality` | render settings identical, so a pixel difference is a physics difference |

Unlinking a group is the deliberate act of comparing *that* thing instead:
unlink Look to A/B two render tunings on one solver; unlink Cut to show the
Y slice beside the Z slice of the same field. When any group is unlinked the
strip names it, so a reader never mistakes a view diff for a solver diff.

Two things cross the seam by world position rather than by key, because they
are about *a place*, not a pane:

- **Pins and hover.** A pinned cell/pixel trace (`FluidCellTraceHud`,
  `PixelTraceHud`) is pinned by world cell / by ray, so pinning in A pins
  the same cell in B — two trace HUDs, one per pane, same cell. Hovering a
  cell in one pane highlights it in the other.
- **Selection and edits.** Selecting a body, carrying it, painting water or
  placing a region is a *scene* edit. Under the diff rule it writes to A and
  therefore to B unless B's diff already forks the scene (a different
  `sceneId`, a body override) — then the edit stays in the pane it was made
  in. Selection itself is linked while the scene is: the same entity is
  outlined in both.

Overlay modes that only exist for one method (`isSparseCM12DirtyOverlayMode`,
`isOctreeTechniqueOverlayMode`, …) are legal in a linked Cut even when one
pane's method cannot draw them: that pane draws no overlay and the strip
shows "n/a in B". Do not silently drop the key.

What is **never** linked: the diagnostics, stats, timings and readiness a pane
reports about itself (`diagnostics-store`, `runtime-store`) — these are the
*output* of the A/B and are per-session by construction.

## Lockstep

Today: `controller.tick` (`lib/core/simulation/controller.ts:210-274`) moves a
target clock; each worker renderer queues advances toward it up to two deep
(`webgpu-renderer.ts:144,2729-2742`); completion returns via `advance-completed`
into the one `gpuCompletedTime` (`controller.ts:160,970-976`).

Compare mode: the controller becomes a **host** over N pane sessions.

- Per-pane `gpuCompletedTime`. The host's `completedTime = min(panes)`.
- The target clock advances by one `dt` only when `completedTime == target`
  (a barrier), so neither pane can be more than one step ahead of the other.
  In-flight depth is pinned to 1 in compare mode; the throughput lost is the
  price of the mode and is stated in the diff strip's rate readout.
- `dt` is the **effective simulation step** (`lib/core/simulation-step.ts`)
  of pane A. If B's diff changes `fixedDt_s` or pins a method dt, the host
  steps at the smaller and lets the larger-dt pane skip steps it does not
  need — but this is flagged in the diff strip as "dt differs: not lockstep"
  since visually it is no longer a paired step. (The lane where dt differs is
  a real use — uniform only holds at dt=1/30 per memory — but it's a
  comparison of rates, not of steps.)
- STEP steps both. Pause drains both. Reset resets both to their own t=0.
- Divergence oracle (cheap, optional): per step the host reads each pane's
  existing stats (`solver.info` after `readStats()`) and the diff strip shows
  `t`, volume, pressure iterations, and ms/step for A and B. Two identical
  configs that disagree on volume at step k is the "non-deterministic lane"
  tell, and is worth seeing.

## The session realm (the load-bearing refactor)

Everything binds to module singletons today (audit 2026-08-29):

| Singleton | Where | Two-pane fate |
|---|---|---|
| zustand stores `scene, method, ui, runtime, diagnostics, scene-draft, history, recording` | `lib/core/stores/*` | become **per-session** instances from factories, reached through `useSession()` context |
| `shell, theme, performance-*` | same dir | stay global (page-level) |
| `window.__fluidLabSimulationController` | `controller.ts:1319-1340` | becomes the **host**; each pane holds a `PaneSession` with its own clock fields |
| `window.__fluidLabGPUViewportLifecycle` | `WebGPUViewport.tsx:1078-1095` | keyed by pane id (`Map`); the retained-on-canvas-identity rule survives per pane |
| Web Lock `fluid-lab:webgpu-exclusive` | `gpu-startup.ts:88,106-140` | stays **page-exclusive across tabs**; within the page, the lock holder hands out ≤2 pane leases. Cross-tab safety unchanged. |
| `GPU_MANUAL_START/STOP` window events | `gpu-startup.ts:185-194` | carry a pane id; `undefined` = all |
| URL writer `replaceQueryStateUrl` | `url-state.ts:774` | one writer, serializes A from session A and the diff from session B |
| recording | `recording.ts:111,313` | per pane; the REC wedge records the pane it was opened on (side-by-side capture is a stretch) |
| autosave | `scene-autosave.ts` | saves A only; B is a diff, never a library entry unless promoted |
| keyboard shortcuts | `use-editor-shortcuts.ts` | route to the **focused pane** (last pointer-down / hover); transport keys go to the host |

Mechanism: `createSessionStores()` returns the per-session store set;
`<SessionProvider value={session}>` wraps each pane; every chrome component
swaps `useSceneStore(...)` for `useSession().scene(...)` (same selector
signature — a mechanical edit across ~40 files, the bulk of the work). Single-
pane mode is compare mode with one session; there is no second code path.

`WebGPUViewport.tsx` (3012 lines) splits at its natural seam
(`:1058-1400` lifecycle/worker/render loop vs. the pointer/overlay half) into
`<SimPane session>` and the interaction layer it hosts.

Worker side needs nothing: each pane is its own worker + device
(`webgpu-render-worker-client.ts:138`, no factory singleton; device requested
inside the renderer; compilation manager is device-keyed). Cost: two devices
at advertised maxima, two per-frame structured clones of the scene document
(the `docs/worker-boundary-memo-handoff.md` hazard, now ×2), two
presentation/G-buffer sets reallocated on every splitter drag (debounce the
resize to the drag end).

## Layout

`.lab-shell` (`app/globals.css:313-320`) becomes a two-column grid when the
host has two panes: `minmax(0,1fr) 6px minmax(0,1fr)`, the middle column the
splitter. Each pane is a `.viewport-shell` exactly as today — flyouts are
already container-relative (`components/anchored-flyout.ts`), so they anchor
correctly per pane. CSS that assumes one shell (`globals.css:396,991`) is
scoped to `.viewport-shell` instead of the page. The diff strip is absolutely
positioned over the splitter's top, translucent, in the overlay style of
`scene-instrument`. Each pane shows a small **A** / **B** tag at its top
corner where the scene chip lives; the focused pane's tag is bright.

Stretch (WP5): a **wipe** mode — one pane, a draggable vertical wipe line,
A on the left of it, B on the right. Needs both renderers to draw at full
size into offscreen targets and a composite; defer until the two-pane mode
has proven the realm. A pixel-difference view could reuse
`tools/compare-svo-screen-space-images.ts`'s rgba16f diff.

## Work packages (Opus 5 subagents)

Each WP is a PR-sized unit with its own gate; WP0–WP2 have no visible UI.

- **WP0 — Session realm.** Store factories + `SessionProvider` + mechanical
  migration of chrome to `useSession()`. Gate: app identical in single-pane
  mode; `tsc` clean; every existing browser test green. This is the largest
  and most mechanical WP; split by directory across two agents if needed, but
  the factory + provider must land first.
- **WP1 — Pane-keyed lifecycle and lock leases.** Viewport lifecycle map,
  pane-id on start/stop events, lock-holder leases. Gate: mount two
  `<SimPane>`s on a hidden test route with the *same* config and both render
  (no "another tab owns the lock", no cleanup of pane 1).
- **WP2 — Host clock and lockstep.** Controller → host + `PaneSession`
  clocks, min-completion barrier, depth 1 in compare, per-pane completion
  reporting. Gate: two identical CM12 panes for 300 steps never differ in
  `t` by more than one `dt` at any observed instant, and their volume digests
  match step-for-step (bit-exactness is not expected across two devices; the
  long-dam lane is already non-deterministic — same-run oracles only).
- **WP3 — Compare mode UI.** Ring wedge + `\`, splitter grid, diff model
  (`b.*` URL keys, overrides chip group), A/B tags, focused pane, diff strip
  with `×`/`⇄`, the four group padlocks (View/Cut/Instrument/Look), cross-pane pins, Close/Keep/Swap. Gate: open compare,
  first frame of B pixel-identical to A; change B's solver in the FIELD
  flyout; reload restores both.
- **WP4 — Divergence readout.** Per-step stats rows in the diff strip; the
  dt-differs flag. Gate: the two stats rows update every step and agree for
  identical configs.
- **WP5 — Wipe / pixel diff (stretch).**

## Decisions already made (do not re-open)

- B is a diff over A, not a second document.
- Lockstep is the only mode; no free-running compare.
- One transport, one clock; view/cut/instrument/look linked by default, per-group padlocks.
- Cross-tab GPU exclusivity is preserved; only the in-page lease count grows.
- Two workers and two devices; no attempt to share a device across panes.
- The ring is the route; no docked compare panel.

## Per-pane scenes (landed 2026-08-30)

The coarsest diff the mode was designed for is now sayable from inside the
studio: **each pane chooses its own scene.**

- **The chooser** is `components/SceneSelector.tsx`, a popover mounted *inside*
  a pane (`ScenePane`) and reading the session it is mounted under, so the same
  component under pane B's provider chooses pane B's scene. Ranking lives in the
  pure `lib/core/scene-search.ts` (name/word/shelf/id/blurb tiers, best first;
  `tests/scene-search.test.ts`). Its search text is component state, never the
  shell store's `librarySearch`, which two panes would share.
- **Four ways in**, all pane-aware: the pane's own **A/B tag** (it is now a
  button), the hover-reveal **scene chip** in `SceneOverlay` (which no longer
  routes to the library — the library is a *Browse library…* row inside the
  popover), a **Scene…** wedge on the empty-space ring, and **`o`** on the
  focused pane. `sceneSelectorOpen` is a per-session `ui-store` flag and is
  deliberately not serialized.
- **URL shape.** Nothing new. A pane's scene *is* the `scene` config key plus
  whatever `scene.*` / layer keys the writer emits, so choosing a scene in B
  records `b.scene=<presetId>` (plus `b.scene.*` where the document diverges
  from that preset's own baseline) and a reload restores both panes through the
  existing `parseCompareQuery` → `applyLayers` path. Choosing in A while the
  diff is empty mirrors into B by `exactScene` as any other document edit does;
  choosing in A while B has forked leaves B alone. `×`, `⇄`, Keep and Swap all
  work on the `scene` row, and the strip prints scene *names* rather than ids.
- **Configuration is retained.** `simulation.openSceneCard(card, paneId,
  { retainConfiguration: true })` swaps the document without resetting the
  solver, its parameters, the quality, the raised instrument or the field slice.
  The library front door keeps its clean-entry reset; this is the other gesture
  — swapping the document *under a standing experiment*. Without it, choosing a
  scene in B would reset B's solver to the product default (recording a `method`
  diff nobody asked for) and push B's fresh Cut/Instrument/Look values onto A
  through the padlocked groups.
- **The camera is the deliberate exception**: a scene open adopts the authored
  framing of the scene it opened, and with the View group linked that reframes
  the other pane too — which is right, because a linked View means one camera.
- **The renderer waits for the rebuilt world.** Swapping a scene under a live
  viewport is the first thing in the product that changes the whole document
  without a route change, and it exposed a real seam:
  `publishRenderScene` staged the new scene onto the sparse world still
  attached — the one voxelized for the *previous* document — whose planar
  terminals refuse (`webgpu-svo-sparse-bricks.ts` "requires a sparse-world
  rebuild"), and the refusal escaped `draw` and released the device. The stage
  call is now attempted inside a guard: while a rebuild is in flight
  (`gpuFluidPendingKey`), a refusal defers the publication to the world being
  built for this scene, and the previous frame stands until it lands. A refusal
  with no rebuild coming still travels, because that is a live-edit fault.
  **Seen in the app, fixed on the CPU:** the fault was reproduced in a browser
  (pane B, `bounded-pool-transfer`, `method=uniform`) and the guard was written
  against that trace, but browser testing was withdrawn before the fixed build
  could be re-run. The guard is unexercised by any test — it only fires when a
  live world refuses a staged scene, which needs a device.

Residuals, none of them new and all of them shared with pane A:

- **Starter cards do not survive a reload, in either pane.** `starter:<id>` is
  not a registered preset, so `serializeQueryState` writes `scene=starter:blank`
  and the reload falls back to the default preset; the emitted `scene.*` deltas
  then fail `validateScene` and the whole document falls back with them. This is
  pane A's existing behaviour via the library's *Start from empty* cards — B was
  made to follow A's rule rather than given a scheme of its own.
- **Saved cards round-trip exactly as far as their deltas reach.** A
  `saved:<id>` card opens with the *origin* catalog preset id, so the URL is
  `scene=<origin>` plus the `scene.*` paths that differ (verified: a saved
  `fillFraction` survives a reload). What the query has no path for — a sculpted
  terrain grid, the scenery graph — is lost on reload, again for both panes.
- `⇄` moves one key. A saved scene forked into B is a `scene` row *and* its
  `scene.*` rows; moving only the `scene` row gives A the origin preset without
  the saved edits. **Keep this one** promotes them all, which is the operation
  that means "adopt B".
- Opening a scene is a `reset`, and `reset` is a host operation
  (`PaneClockHost.reset`) — so choosing a scene for one pane returns *both*
  panes' clocks to t=0. Pre-existing for every mirrored re-seed; noted here
  because per-pane scenes make it easy to trigger.
- **Verified on the CPU only.** `tests/compare-model.test.ts` carries the whole
  address round-trip — pane A's query plus the `b.*` block, back through
  `parseCompareQuery` and the mirror's opening pass — for two panes on two
  scenes, for a scene chosen in A with B unforked, and for a saved scene's
  `scene.*` deltas riding beside the key. What no CPU test reaches: the popover
  drawn over a running viewport, the four entry points as *gestures*, and the
  renderer deferral above.

WP3c landed after the WP0–WP4 gate: document mirrors into a pane now reach the controller through a new `adoptSceneEdit(previous, paneId)` (commitEdit's tail with no history entry; `sceneEditRequiresReset` picks reset vs adoptRigidBodies; the receiving pane's run state is preserved across a reset so a mirrored re-seed cannot stall the lockstep clock). Wired as `COMPARE_ADOPTIONS` in compare-mode.ts. 49 tests. 

## Transport is the host's (landed 2026-08-30)

Play/pause, STEP and RESET are statements about the **experiment**, not about a
pane, so they reach every attached pane atomically. The readout beside them —
`t`, the run state — reads the host, which is pane A. With one pane attached the
fan-out is empty and this is the transport that shipped.

- `controller.setRunState(state)`, `controller.singleStep()` and
  `controller.resetAll()` are the host-level gestures `TransportBar` calls. The
  fan-out is an explicit host method; `reset(source, presetId, paneId)` keeps its
  pane argument and its per-pane meaning. The rules themselves are pure and live
  in `lib/core/simulation/pane-transport.ts` so they can be tested without
  importing the controller or `session.ts` (`tests/compare-transport-host.test.ts`).
- **Two failures this fixes.** A pause that reached only pane A left B's renderer
  never told the clock had stopped, so B never took the pause-boundary
  `readStats()` and the diff strip reported a volume divergence between two
  *identical* panes. Worse, B never called `gpuSchedulingPaused`, so the host
  rewound the target on A's drain alone and could land below an advance B had
  already encoded — and a solver never re-encodes a time it has submitted, so the
  completion the min-completion barrier waits for never arrives and the clock
  dies for both panes.
- **`schedulingPaused` therefore defers the rewind until every pane has reported
  its drain** (`pane-clock.ts`). The floor is the maximum over the panes and that
  maximum is unknown until the last report; debt is dropped on the first, only
  the rewind waits. Fanning the pause out is not sufficient on its own — the two
  changes are one fix.
- **RESET re-seeds every pane, because `PaneClockHost.reset` zeroes every pane.**
  A pane left un-re-seeded is stranded above a target that rewound past it — the
  same orphan. `reset(paneId)` now re-seeds the others as *resynchronizing*: same
  clock, same document runtime and renderer epoch, but they keep the run state
  they had, keep their selection, and do not each post a reset notice. That is
  what makes a per-pane scene open safe, and it retires the last residual in the
  section above. The transport's `resetAll` marks nobody resynchronizing: one
  gesture, one outcome — the whole experiment at zero and stopped.
