# Editor interaction: modes, targets, gestures

The scene editor is being rebuilt around two ideas: **one modal axis** (look vs.
edit) and **plugin catalogs** for what the cursor can reach and what a gesture
can mean. This is the tracking document; each phase lands on its own and is
green on its own.

Decisions taken with Peter, 2026-08-26:

- **Interact absorbs the tools.** One axis, `camera` vs. `interact`. Inside
  INTERACT there is no armed tool at all — every verb is an action on the
  highlighted target. `EditorTool` is deleted rather than hidden.
- **CAMERA keeps the empty-space ring only**, so the instruments stay reachable
  per `hero-scene-ui-plan.md` while nothing in the scene can be touched. Library
  and Import were on that ring and came off (2026-08-27): they are about leaving
  the document rather than about anything in it, and a ring opened on the scene
  is the wrong place to be offered the exit.
- **A solid voxel belongs to something, and a click lands on it.** Only a *drag*
  makes a voxel region. A click selects the owner — the region it stands in, the
  terrain feature it was baked from, or the vessel whose solid world it is a cell
  of. The hover still names the cell; the three grains are the target, the
  highlight and the selection, exactly as for a tank wall.
- **The vessel is glass.** The container shell is authored into
  `scene.solidVoxels` like any other patch, so the solid world cannot tell it
  from a dam — but the editor must, or the near pane answers every pixel of the
  tank and neither the tank nor the water can be clicked at all. See
  `containerShellContains`.
- **Everything solid sweeps.** A left-drag that starts on a voxel — authored or
  sculpted — or on a bare tank wall sweeps a selection region. The room is what
  still orbits, so the camera keeps the empty pixels and the scene keeps the
  solid ones.

## Why

Interaction was always-on and object-granular. `SELECT` was the default
`EditorTool`, so every left-click in the viewport was a pick and there was no way
to just look at a scene. What a click could reach was limited to whole
*entities* — the seven `EditorEntityDefinition` plugins in
`lib/core/editor-entity-catalog.ts`. A voxel, a tank wall, a fluid cell — the
things a scene is actually made of — were not addressable at all:
`pickSolidVoxel` existed but was reachable only while the `solid-voxel-clear`
tool was armed, and the fluid cell existed only as a GPU instrument behind `C`.

Below the entity plugins the pointer machine had not been factored:
`pointerDown`/`pointerMove`/`pointerUp` in `WebGPUViewport.tsx` were one long
if-chain switching on `activeTool`, and adding an interaction meant adding a
branch. Four near-identical projected-box SVG overlays each re-derived the same
corner projection with their own idea of when an edge is behind the camera.

## Architecture

Three static catalogs, following the convention `editor-entity-catalog.ts` and
`resource-plugin.ts` already set: a frozen array composed at module scope, **no
mutable `register()`**, so import order and hot reload cannot change what the
editor offers. Distance is the primary key; array order is the tie-break.

```
pointer ray
   │
   ▼
┌──────────────────────┐   probes answer "what is here"
│ EDITOR_PROBES        │   nearest wins; array order breaks ties;
│  tank-wall           │   the room probe is the declared fallback,
│  solid-voxel         │   so the result is NEVER undefined
│  entity              │
│  fluid-cell          │
│  terrain             │
│  room      (fallback)│
└──────────┬───────────┘
           ▼  EditorTarget { kind, id, label, point_m, normal,
           │                 selection?, highlight, detail? }
     ┌─────┴───────┬──────────────┬─────────────────┐
     ▼             ▼              ▼                 ▼
  highlight    left-click     right-click         drag
  (one SVG     select         targetActionsAt   ┌──────────────────┐
   layer draws  target.        = probe.actions  │ EDITOR_GESTURES  │
   any          selection      + entity.actions │ first claimer    │
   EditorHigh-                 + general        │ wins; orbit is   │
   light)                                       │ the last entry   │
                                                └──────────────────┘
```

### The fallback rule

**Exactly one probe is a fallback**, and it always answers. A fallback never
competes on distance — it is consulted only when nothing else replied — which is
what makes `targetAtRay` total. "Nothing under the cursor" is then not a state
the interface has to draw, explain or guard against, and the guiding principle of
INTERACT ("there is always something highlighted") holds by construction rather
than by a checklist of covered cases. `tests/editor-probe-catalog.test.ts`
asserts it as a property over a fan of rays from several camera poses.

### Drag-select produces a selection, not an edit

CLEAR SOLIDS used to be an armed tool whose drag destroyed geometry on release,
so choosing the wrong extent meant undo, re-arm, re-aim, re-drag. A sweep now
produces a **`voxel-region` selection** (`lib/core/editor-voxel-region.ts`) and
the verbs live on it — Clear on the ring, Delete on the key and the flyout. This
separates aiming from acting, which is what every other selection in this editor
already does.

The region is the one entity that is not in the document, and must not be: a
selection is not scene data, so it belongs neither to the saved file nor to the
undo history. It lives in `ui-store` and reaches entity code through
`EditorEntityContext`, the same route live body poses take.

A region has no surface of its own, so it declares no `pick` — claiming clicks
inside it would make the voxels it selected unreachable while it stood. Instead
the **voxels inside it name it**: `solidVoxelProbe` reports
`selection: VOXEL_REGION_SELECTION` for any cell within the standing region, so
right-clicking one opens the region's ring.

## Phases

| # | Phase | State |
|---|---|---|
| 0 | The mode gate — `viewportMode`, the toggle, CAMERA short-circuit | **landed** |
| 1 | `editor-target.ts`, `editor-probe-catalog.ts`, `EditorHighlightLayer` | **landed** |
| 2 | `solidVoxelProbe`, `tankWallProbe`, `roomProbe`; `targetAtRay` total | **landed** |
| 3 | `targetActionsAt` — probe verbs + entity verbs + general | **landed** |
| 4 | `editor-gesture-catalog.ts`; delete `EditorTool` and `activeTool` | **landed** |
| 5 | Drag-select regions; CLEAR SOLIDS retired as a tool | **landed** |
| 6 | `fluidCellProbe` — lift `fluidCellTrace` into a store | **landed** |

### Phase 4 — the gesture catalog

The rule the catalog encodes, and the reason `EditorTool` could go away at all:

> **A mode survives only if it changes what a _drag_ means. A single click at a
> point is always an action.**

Eleven tools met that test five times. `BALL`, `WATER`, `ERASE`, `REGION` and
`DRAG` each reinterpret a stroke, so they stayed as **armable** gestures with
their keys intact. `body-place`, `prop-place` and `inflow` were single clicks at
a point the ring already carries — asking "where" twice — so they became the
ring verbs `place`, `place-prop` and `place-inflow`. `solid-voxel-clear` became
the sweep of Phase 5. `terrain-raise`/`terrain-lower` were never implemented and
were deleted rather than carried.

Everything else a press can mean is **implicit**: resolved from the target, never
armed, never on a key — `entity-handle`, `terrain-handle`, `slice-grab`,
`fill-level`, `voxel-sweep`, `body-throw`, and `orbit`/`pan` as catalog entries
of their own.

```ts
export interface EditorGestureDefinition {
  readonly id: EditorGestureId;
  readonly label: string;
  readonly hint: string;
  readonly shortcut?: string;     // armable gestures only
  readonly armable?: boolean;
  readonly claims: (target: EditorTarget, modifiers: GestureModifiers) => boolean;
  readonly needsPresentation?: boolean;
}

export function gestureForPress(
  armed: EditorGestureId | undefined, target: EditorTarget,
  modifiers: GestureModifiers, presented: boolean,
): EditorGestureId;
```

Press resolution, in order: shift or the middle button is always the camera → the
armed gesture if one is pinned and claims → otherwise the first implicit entry
that claims → `orbit` is the last entry and claims everything, so there is no
fallthrough special case. One rule is worth stating because it is not obvious:
**an armed gesture that cannot run yields to the camera, never to an implicit
one.** A reader who armed WATER and pressed while the renderer was still
rebuilding asked for water; answering with a voxel selection instead would be the
interface doing something else in their name.

The load-bearing rules survived the port: handles outrank scene picks (the three
of them are tried before the target is even resolved, and `slice-grab` is hoisted
above the sweep because its band rides the container rim where `tankWallProbe`
answers), the release decides click-vs-drag via `pointerStayedWithinClickSlop`,
and every gesture commits on release through the draft store — **never per
pointer-move**, because each scene write invalidates the solver's seed key.

**What the catalog deliberately does not own.** The plan above sketched an
`EditorGestureSession` (`begin`/`update`/`commit`/`cancel`) living in the
catalog. It does not, and should not: every session closes over React refs, the
renderer handle and the draft store, and a "core" module taking all of those as a
services bag would be the viewport with extra indirection. The catalog decides;
the viewport performs. The seam that mattered — *what does this press mean* — is
the one that moved, and it is the one a new interaction has to extend.

`surfacedBy(tool, selection)` is gone from `EditorEntityDefinition` and all eight
entity files: with no tools it had collapsed to `tool === "select"`, i.e. `true`.
`ui-store`'s `propShape` went with it — the last state a deleted tool was reading
its next click's shape from.

### Phase 6 — the fluid cell probe

The last target, and the only one nothing in the document describes. A leaf's
extent is decided by the solver's topology — how far the octree refined here,
this frame — so there is no analytic answer to compute and no scene field to
read. It comes from the GPU or not at all, which makes it the probe the plugin
boundary was really designed for: it answers a ray without raycasting anything.

Three things had to move for it.

**The trace left component state.** It was `useState` in `WebGPUViewport`, and a
probe is called from `editorEntityContext()` with no React around it — so a trace
only that component could see was a trace the editor could not point at. It is in
`diagnostics-store` now, with the rest of the run's published output.

**The frame came with it.** Sparse CM12 publishes its own frame as *index space*
(`domainOrigin: [0,0,0]`, `fineCellWidth: 1`), so `leafOrigin` counts finest
cells and `leafSize` counts cells across. Reading either as metres puts the cell
at the world origin at one metre per cell — a box in the wrong place at the wrong
scale that still looks like a box, the same trap the renderer's SVO coverage
source calls out. `fluidCellTraceLattice` resolves it: the solver's own
`fluidDomain` when it publishes one (sparse flow can publish a fluid world larger
than the tank), and the container divided by the trace's own `dimensions`
otherwise. The domain crosses the worker boundary as a new snapshot field;
`pinned` deliberately does not, because it is the reader's intent rather than the
frame's output and would be stale exactly when it mattered.

**The probe re-tests the leaf against the live ray.** The published leaf is one
frame behind the pointer, and a slab test is what turns that lag into a decline
instead of a wrong answer: if the pointer has moved off, something else answers.
It is silent while the `C` instrument is off (no gather runs, and a hover must
not start a readback) and while the trace is pinned (that leaf is a pixel the
reader chose earlier and has since moved away from).

It carries no `selection` — a cell exists in the solve, not in the document — and
ranks behind the entities for the same reason, so it never takes a click away
from the water body it sits inside. It surfaces where a leaf is genuinely the
nearest thing: over water that has moved away from the box it was authored in,
and under a selected fluid body, which `exclude` already makes transparent.

## Verification

- `npm run check:types` is the primary gate. The closed unions
  (`EditorHighlight`, `EditorActionIcon`, `satisfies Record<…>`) mean a missing
  case is a compile error, not a blank wedge.
- `npm run test:unit` — the CPU suite has pre-existing failures. Verify by
  diffing the failing *test names* against an archive captured before the phase
  starts, not by expecting green.
- New tests live beside the existing `tests/editor-*.test.ts`:
  `editor-probe-catalog.test.ts` (resolution order and the fallback property),
  `editor-voxel-region.test.ts` (the sweep, and its Clear verb regressed
  directly against `editor-solid-voxel.test.ts`'s patch), and — for Phase 4 —
  `editor-gesture-catalog.test.ts`.
- End to end in the browser, not in headless Dawn: `npm run dev`, hero garden
  scene. CAMERA orbits and clicks nothing; the toggle enters INTERACT; sweeping
  the pointer lights a target on every pixel with no gaps; right-click on each
  opens a ring naming that thing; dragging across the tank floor leaves a region
  outline that stays selected and whose ring clears it.
