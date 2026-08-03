# Exploration UX — design and implementation plan

Status: design, 2026-08-03. Branch `perf/structured-cutover`.
Companions: `LIVE_EMPTY_SCENE_ACCEPTANCE_PLAN.md` (the runtime contract this UX
depends on), `EDITOR_ENTITY_ARCHITECTURE.md` (the manipulation model it keeps),
`RESOURCE_PLUGIN_ARCHITECTURE.md` (the readiness model it exposes).

## 0. Status

| phase | state |
| --- | --- |
| 0 · One catalog, one open path | **landed** — `lib/scene-definition.ts`, `SCENE_CATALOG` in `lib/scenes.ts`, `lib/scene-cards.ts`; the eleven duplicate smoke factories are collapsed onto named `SceneVariant` deltas |
| 1 · The library shell | **landed** — `components/SceneLibrary.tsx`, `lib/stores/shell-store.ts`, `lib/scene-glyph.ts`, `lib/empty-scene.ts`, `lib/scene-autosave.ts`, `components/UrlStateSync.tsx` |
| 2 · Stop the editor dying | **landed** — `resourceInteractionGates` split into camera/picking; the activity tray honours `ResourcePluginDefinition.blocks` |
| 3a · Inflow out of the seed tier | **landed** — `inflowBudgetKey` / `inflowAimKey`, and `applySceneUniforms` rederives the inflow grid boundary |
| 3b · Bodies out of the seed tier | **partial** — a body's pose, shape and material are hot (`adoptRigidBodyRoster`); the roster *length* is not, and cannot be until owner numbering is decoupled (below) |
| 3c · Painted water as an injection | not started — the largest piece, and the last |
| 3d · Terrain | blocked, as recorded against `terrain-raise`/`terrain-lower` |
| 4 · Paint hygiene | **landed** — `fluidBrushSample` + the draft path; a stroke writes the document once |
| 5 · Polish | partial — glyphs, URL `view` state and autosave landed; posters not |

The load-bearing change behind 3a and 3b is `sceneEditRequiresReset`: an editor
commit used to reset unconditionally, so every gesture threw the simulation away
in order to apply something the running solver could have adopted. `commitEdit`
now asks the tier boundary first, and `reseed` is a request rather than an
instruction.

**The one decoupling this document now owes.** Every publication that numbers
environment proxies does it as `rigidBodies.length + ownerIndex` —
`svo-scene-primitives`, `svo-scene-thick-glass`, the sparse brick world's
`stageSceneUpdate`, and the dry-scene `ownerBase`. So the roster *length* is
part of the render source ABI, not just the solver's allocation, and adding a
body renumbers every scenery object in the scene. Pinning the environment base
at the fixed rigid capacity (`GPU_RIGID_BODY_CAPACITY`, 12) removes the count
from the seed tier outright; the cost is resizing every per-owner arena from
`bodies.length + proxies` to `12 + proxies`, which is a GPU-side change and not
one to make blind. Until it lands, "drop a body into running water" is warm for
a body that is already there and cold for a new one.

Phases 0–2 changed no scene document: every one of the twenty-five authored
scenes is byte-identical under `canonicalScene` before and after. The smoke
collapse in Phase 0 *did* change eleven GPU-lane documents, each one listed in
`tests/scene-smoke-catalog-parity.test.ts`.

Known gaps: posters, and a `scene=<card id>` URL key (only preset ids are
addressable today — starters and saved scenes are not).

## 1. The goal

Two changes to what the product *is*:

1. **The front door is the scene library.** Opening the app shows what there is
   to explore, categorised, not a dam break in a box with a `<select>` pinned to
   the corner.
2. **A fresh scene is the second thing you can do, and it works.** Start empty,
   drop in terrain, bodies and water, and never wait — no rebuild modal, no
   clock reset, no dead viewport.

The second one is not a UI change. Most of the work is removing the reasons the
runtime currently has to stop.

## 2. What blocks it today

The renderer is already live. `publishRenderScene` republishes analytic,
material and lighting arenas on any document change, `advanceLiveSceneAnimation`
refits continuously-authored motion through the same arena, and
`stageSceneUpdate` / `stageLivePrimitiveUpdates` push into the solver's sparse
scene without allocation. Scene *presentation* does not rebuild.

Everything below is on the other side of that line.

### A. The seed key conflates four unrelated classes of input

`gpuSceneSeedKey` is one string over:

| what | fields | what it actually is |
| --- | --- | --- |
| initial condition | `initialCondition`, `initialDamBreakDimensions_m`, `initialDamBreakOrigin_m`, `initialBrickSeeds_m`, `initialBrickSeedsAdditive`, `container.fillFraction` | t=0 liquid. Genuinely a re-seed. |
| boundary geometry | `rigidBodies`, `terrain` | solids the solve reads every step |
| forcing | `fluid.inflow` | a source term, already packed into the params buffer |
| domain metric | `container.width_m/height_m/depth_m` | metres per cell, already recomputed by `reseed` |

Any change to any of them changes the key, so `currentGPUFluid` calls
`tryReseedGPUFluid`, which calls `WebGPUOctreeProjection.reseed`, which
overwrites the resident level set with `initialOctreeLevelSet(scene, …)` and
re-earns the fenced t=0 raster publication.

**Dropping a sphere into two seconds of settled water throws the water away.**
That single fact is the difference between "an editor with a simulation in it"
and "a simulation you can edit". It is the highest-value fix in this document.

### B. Commit means reset

`commitEdit({ reseed: true })` calls `SimulationController.reset()`, which zeroes
the clock, re-initialises rigid bodies, clears the diagnostics store, pauses
transport, and clears the selection. Tank resize, fluid-body reshape, entity
removal and every paint stroke take that path. The comment in `commitEdit`
already says this is meant to become a warm re-seed; §5 Phase 3 is that work.

### C. The water brush writes the document per brick

`paintFluidAt` calls `sceneStore.patchScene` each time a stroke crosses into a
new brick, and `pointerUp` then calls `commitEdit({ reseed: true })`. So a
twenty-brick stroke is twenty seed-key changes — twenty attempted re-seeds,
serialised behind `reseedInFlight` — followed by a full reset. Every other
gesture in the editor already goes through `beginDraft`/`updateDraft`/
`commitDraft` and writes once. The brush is the one that does not.

### D. A rebuild kills the editor, not just the physics — *fixed, Phase 2*

`effectiveRendererStatus.state` drops to `pending` during any SVO pipeline
compile or while awaiting a live scene publication. The viewport reads it as
`scenePresentationAvailable` and, when false, passes `undefined` for
`onPointerDown`, `onPointerMove`, `onPointerUp` and `onWheel` — **including
camera orbit and zoom**. `editorEntityContext` propagates the same flag, and
`surfacedEntities` / `entityAtRay` / `findEntity` all return nothing, so handles
disappear and the selection becomes unreachable.

`resourceInteractionGates` says the shell is never GPU-gated
(`shellInteractive: true`) and it is right, but `viewportInteractive` is used as
though it meant "can the user touch anything", when what it actually gates is
"can a ray hit published geometry". Orbiting a camera needs neither.

### E. Presets are code, saved scenes are JSON, and they load differently

`ScenePreset.create()` is a constructor; `SceneLibraryEntry.scene` is a
serialized document. `loadPreset` applies `methodProfile`, sets the camera from
`cameraForPreset`, and starts the clock; `loadNamedScene` does none of those. A
library UI that shows both in one grid cannot have two behaviours behind one
card.

### F. There is no empty document

`defaultScene` is a dam break. Nothing constructs a bounded scene with no
content, even though `planSceneRuntime` already reads `systems.fluid === false`
and `WebGPULiveSvoScene` already attaches for exactly that case with
`initialSparseAuthorityReady = true` and no fluid authority at all. The runtime
for "fresh empty scene, renders immediately" exists and is unreachable from the
UI. `LIVE_EMPTY_SCENE_ACCEPTANCE_PLAN.md` Phase A already specifies
`createEmptyScene` as the only new-scene constructor.

### G. The chooser is a 200 px `<select>`

`SceneOverlay` renders every preset as `<option>` grouped by `ScenePreset.group`
— four groups, of which "Comparisons" is fourteen numerical oracles (ceiling
drop, mid-air corner control, 20× hydrostatic) that a person arriving to look at
water should never be shown first. Descriptions exist and are only reachable as
a `title` tooltip.

## 3. The experience

### 3.1 One shell, two views

`view: "library" | "studio"` in a shell store — **not two routes**. The WebGPU
device, the compiled pipelines and the retained viewport survive Fast Refresh on
purpose (`__fluidLabSimulationController`); tearing them down to show a menu
would be the most expensive navigation in the product. The library is a layer
over a live viewport.

That buys the best property of this whole design for free: **the library is what
you look at while the platform lane boots.** `resourceReadiness.platform` is
`preparing` for the first second or two of every cold load. Today that is a
progress card on an empty canvas. Tomorrow it is browsing time, and the studio
is entered when the user picks, by which point the device is usually ready.

### 3.2 The library

```
┌──────────────────────────────────────────────────────────────┐
│  Fluid Lab                                    [search  /]    │
│                                                              │
│  ┌──────────────┐  ┌──────────────────────────────────────┐  │
│  │  + NEW SCENE │  │  Continue — "Pond experiment"        │  │
│  │  start empty │  │  edited 4m ago · 3 bodies · water     │  │
│  └──────────────┘  └──────────────────────────────────────┘  │
│                                                              │
│  EXPLORE                                                     │
│  [glyph] Garden pond      [glyph] Twin dams   [glyph] Ocean  │
│  still water, cork ball   corner collision    rolling wave   │
│                                                              │
│  STUDY                                                       │
│  [glyph] Hose-filled tank [glyph] Dam + boxes [glyph] Jet    │
│                                                              │
│  MY SCENES                                    3 saved        │
│  [glyph] Pond experiment  [glyph] Big tank                   │
│                                                              │
│  ▸ Research & validation (17)                                │
└──────────────────────────────────────────────────────────────┘
```

- **New scene is first and largest.** It is a peer of the presets, not a menu
  item hidden in a config popover.
- **Continue** restores the autosaved working document, so closing the tab is
  not destructive and the library is never a wall between a user and their work.
- **Audience, not implementation group.** `ScenePreset.group` describes what a
  scene is *for us*; the library needs what it is *for them*. Add
  `audience: "explore" | "study" | "validation"`. The fourteen oracles collapse
  into one collapsed "Research & validation" disclosure that remembers its state.
- **Cards carry the description that already exists** — every preset has a good
  one and today it is a tooltip — plus chips derived from the document by
  `planSceneRuntime` and `sceneLatticeDimensions`: fluid on/off, lattice, body
  count, method profile if it pins one.
- **Glyphs, not screenshots, in v1.** A deterministic SVG drawn from the
  document — tank aspect, waterline, body silhouettes, terrain profile — is
  free, never stale, and needs no cache or build step. Posters captured from the
  live renderer and stored in IndexedDB are a later enhancement, not a
  prerequisite.
- **Keyboard**: `/` search, arrows, Enter opens, Esc returns to the studio (or
  does nothing on first run, when there is nothing to return to).

### 3.3 New scene

**No wizard.** "New scene" creates a bounded, empty, fluid-disabled document and
enters the studio immediately. `systems.fluid: false` means `WebGPULiveSvoScene`
attaches: no fluid pipelines, no t=0 fence, no transport gate. First frame is
immediate because there is nothing to solve.

A starter strip offers documents, not code paths. It offers *sizes* — *Room*,
*Small room*, *Hall* — because the lattice is the structural tier of the solver
key and therefore the one decision a starter can honestly make on someone's
behalf; everything else is placed afterwards and costs nothing to change.

**The room is a room.** Two nodes: a uniformly white `room-shell` and one
overhead softbox. There is no grid, value wedge, ground light, prop, or other set
dressing for the author to identify and delete before their own work reads.

For the same reason a fresh document sets `container.vessel: "none"`. The
container is two things at once: the solver's boundary, and — in every
environment except the garden — a tank you can see. Starting a scene should hand
over the first without the second; the tank is then something to add, from
CONFIGURE › Container. That generalises what used to be a renderer branch on one
environment id into a property of the document, which is what let the garden's
"no vessel here" rule and the fresh room's share one path.

Then the build bar, bottom-centre beside transport:

```
  TERRAIN   WATER   BODY   PROP   HOSE          ⏮  ▶  ⏭      t = 0.000 s
```

Dropping terrain, bodies and props changes nothing about the runtime: bodies are
rigid-only until water exists, props never enter the solve, and all three are
already live-published.

**WATER is the one expensive transition, and it is explicit.** Painting the
first brick flips `systems.fluid` true, which requests the fluid lane. That is
the whole trick of this design: the costly step becomes a single, named,
user-initiated, once-per-scene event with its own progress affordance, instead
of an invisible one that fires on every edit. After it, §5 Phase 3 keeps every
subsequent edit warm.

### 3.4 The studio

Mostly what exists. Changes:

- The `SceneOverlay` `<select>` becomes a **scene chip**: name, and a click (or
  ⌘O) that opens the library. `CONFIGURE` stays.
- ~~The **build bar** joins the left-edge tool strip.~~ **Withdrawn.** This
  proposed a second bar on the grounds that `EDITOR_TOOLS` are pointer *modes*
  and a build bar would be the *content palette*. Reading the catalog, the
  distinction does not survive contact: SELECT is the only mode, and BODY, PROP,
  WATER, ERASE, RAISE, LOWER and HOSE are each already "place this kind of
  thing". The strip **is** the palette. Building a second one carrying the same
  eight entries would be the preset-shell mistake in UI form.
- One always-visible world line: `24 × 18 × 16 cells · 0.05 m · water · 3
  bodies`.
- The `resource-activity-tray` stops being full-width cards for everything.
  `ResourcePluginDefinition.blocks` already distinguishes the cases; honour it:
  `blocks: "viewport"` (platform startup) keeps the card, `blocks: "transport"`
  gets a transport-bar inline state, `blocks: "nothing"` gets a status pill.

### 3.5 The non-blocking contract

Four rules. Every phase in §5 exists to make one of them true.

1. **The editor is never dead.** Camera, selection, handles, palettes and panels
   respond during every GPU transition, without exception.
2. **An edit is visible in the next presented frame.** The draft/display-scene
   split already delivers this for terrain and scenery; it extends to everything.
3. **Only simulation may be behind**, and when it is, it says what it is doing
   and where it will resume — inline, never modally.
4. **Nothing silently resets the clock.** An edit that genuinely cannot be
   applied live says so before it happens and offers *apply at t=0* against
   *keep simulating*.

## 4. Naming and taxonomy

| today | becomes | why |
| --- | --- | --- |
| `ScenePreset.group` | `audience` + `group` | "Comparisons" is a shelf label for us; "Research & validation" is one for them |
| "My scenes" section of a config popover | Library rail | saving a scene is not a configuration setting |
| `loadPreset` / `loadNamedScene` | `openSceneCard` | one card, one behaviour |
| left tool strip only | tool strip + build bar | mode versus content |

## 5. Implementation plan

Phases 0–2 are UI and gating and land independently. Phase 3 is runtime work and
is where the real cost is; sequence it by payoff.

### Phase 0 — One catalog, one open path

- `lib/scene-catalog.ts`: a `SceneCard` over presets, saved entries and starters.

  ```ts
  interface SceneCard {
    readonly id: string;
    readonly source: "preset" | "saved" | "starter";
    readonly name: string;
    readonly blurb: string;
    readonly audience: "explore" | "study" | "validation";
    readonly group: string;
    open(): SceneOpening;   // { scene, presetId?, camera?, methodProfile? }
  }
  ```

  Composed statically, like `EDITOR_ENTITIES` and `resource-plugin-catalog.ts` —
  no mutable `register()`.
- `SimulationController.openSceneCard(card)`: the single load path. `loadPreset`
  and `loadNamedScene` become thin wrappers over it so nothing else moves yet.
- Add `audience` to `ScenePreset`; classify the existing twenty-five. The four
  Interactive + Garden entries and the three paper figures are `explore` /
  `study`; every `Octree ·` oracle and `Brick quad` is `validation`.
- Tests: `tests/scene-catalog.test.ts` — every preset yields a card, every card
  round-trips to a document that `validateScene` accepts, ids are unique across
  sources, `audience` covers every preset.

### Phase 1 — The library shell

- `lib/stores/shell-store.ts`: `view`, `setView`, `libraryFilter`,
  `researchExpanded`.
- `components/SceneLibrary.tsx` + `components/SceneCard.tsx` +
  `lib/scene-glyph.ts` (pure document → SVG).
- `FluidLab` renders the library as a layer over `viewport-shell`; the viewport
  keeps drawing.
- First-run: library open, no Esc-to-studio until a scene has been opened.
- Autosave the working document so *Continue* is real — **landed**,
  `lib/scene-autosave.ts`. A debounced write to the same storage under the
  reserved id `autosave`, named after the scene it was opened from. It is the
  Continue affordance rather than a saved scene: `savedSceneEntries` keeps it off
  the *My scenes* shelf, and an explicit save of the same name is a separate
  entry that the autosave never lands on.
- `url-state.ts`: `view=library`, `scene=<card id>` alongside the existing
  `sceneQueryPaths`.
- Retire the `library` section from `SceneConfigPopover`; keep SAVE there.
- Tests: catalog→grid rendering, keyboard navigation, glyph determinism,
  URL round-trip.

### Phase 2 — Stop the editor dying (rule 1)

- Split `resourceInteractionGates`:

  | gate | requires | used by |
  | --- | --- | --- |
  | `shellInteractive` | nothing | panels, library, file actions |
  | `cameraInteractive` | `renderer` | orbit, pan, zoom, framing |
  | `pickingInteractive` | `renderer` + `sparse-voxel-presentation` | `entityAtRay`, hover, surface-drop placement |
  | `transportInteractive` | as today | play/step |

- `WebGPUViewport`: pointer handlers are always attached. `pointerDown`
  branches on `pickingInteractive` for the ray-dependent actions and falls
  through to orbit otherwise.
- `EditorEntityContext.scenePresentationAvailable` splits into
  `pickingAvailable`. `surfacedEntities` and `findEntity` draw from the display
  scene regardless — handles are document geometry, not GPU geometry, per
  `EDITOR_ENTITY_ARCHITECTURE.md`. Only `entityAtRay` keeps the gate.
- Activity tray honours `blocks` as in §3.4.
- Tests: gate unit tests; an editor test that a selection and its handles survive
  a `preparing` fluid lane; a viewport test that orbit works while
  `effectiveRendererStatus` is `pending`.

### Phase 3 — Split the seed key (rules 2 and 4)

This is the work. `gpuSceneSeedKey` becomes four keys with four responses.

**3a — Inflow geometry out of the seed tier.** `writeParams` already packs
centre, radius, direction and length into the params buffer, so aiming the hose
is a uniform write. Careful: inflow *presence* is structural —
`fluidFootprint`'s `inflowLiquidCells`, `includeWholeDomainPressureSupport`, and
the inflow brick budget all read `scene.fluid.inflow !== undefined`. So the split
is `hasInflow` (structural, allocation) versus everything else (hot). Cheapest
real win in the document: dragging the hose stops restarting the simulation.

**3b — Rigid bodies out of the seed tier.** Poses already reach the solver every
step through `advanceTo(time_s, bodies)`. What does not is a change to the
*roster* or a body's shape/size. Add `stageSolidUpdate(bodies)` writing the
dynamic-solid lane against existing allocations (`webgpu-octree-solid-vertex-sdf`
is the existing producer), with a declared body capacity — `MAX_BODIES` is
already 12. **This is what makes "drop a body into running water" work**, and it
is the assertion in §6 that proves the whole design.

**3c — Painted water as an additive injection.** `stageFluidInjection(bricks)`
writes new liquid into the resident surface state instead of recomputing
`initialOctreeLevelSet`. Erase is the same path with the opposite sign. This is
the largest piece and the last: it is the difference between a water brush that
restarts the scene and one that adds water to it.

**3d — Terrain** stays blocked behind the heights-texture work already recorded
against `terrain-raise`/`terrain-lower` in `EDITOR_TOOLS`. Leave them `planned`;
the build bar shows terrain as placement-only until then.

Each of 3a–3c is independently shippable, independently testable, and each one
converts a class of edit from "restarts the simulation" to "changes it".

### Phase 4 — Paint hygiene (independent of 3c, ship first)

Route the brush through `beginDraft("fluid-body")` / `updateDraft` /
`commitDraft` like every other gesture: preview the painted bricks as an overlay
during the stroke, write the document once on release. Removes the mid-stroke
re-seed storm immediately, without waiting for 3c, and collapses a stroke to one
undo entry.

### Phase 5 — Polish

Posters replacing glyphs; scene rename/duplicate/delete from the card;
`docs/EDITOR_ENTITY_ARCHITECTURE.md` and `RESOURCE_PLUGIN_ARCHITECTURE.md`
updates for the new gates.

## 6. Acceptance

UX gates, each one a test rather than a judgement:

1. **Cold load** — the library is interactive before `resourceReadiness.platform`
   reaches `ready`.
2. **Fresh scene to first water** — new scene → place terrain → place two bodies
   → paint water → run, with no full-screen block at any point and no frame where
   the canvas rejects pointer input.
3. **Drop into running water** — with the clock at t ≈ 2.0 s, placing a body
   leaves `simulationTime` monotonic. One assertion, and it is the whole of
   Phase 3b.
4. **Stroke cost** — a twenty-brick paint stroke produces exactly one document
   write, one undo entry, and (post-3c) zero re-seeds.
5. **Hose aim** — dragging the inflow arrow leaves `simulationTime` monotonic
   (Phase 3a).
6. **Scripted authoring run** — the empty → terrain → bodies → fluid sequence
   from `LIVE_EMPTY_SCENE_ACCEPTANCE_PLAN.md` §F, asserting GPU buffer identities
   are unchanged throughout.

## 7. What this does not do

- It does not change the manipulation model. SELECT stays the one editing mode
  and every entity keeps its colocated `EditorEntityDefinition`.
- It does not make terrain sculpting live; that is gated on rendering work that
  is already scoped elsewhere.
- It does not remove the validation scenes. They move behind a disclosure —
  they are how the physics is trusted, and a workbench that hides its oracles is
  worse at both jobs.
- It does not add a server, accounts, or scene sharing. The library reads
  `localStorage` through the existing `SceneLibraryStorage` seam.
