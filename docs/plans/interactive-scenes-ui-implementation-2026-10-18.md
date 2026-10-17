# Interactive scenes UI: implementation notes

What the UI half of [the handoff](interactive-scenes-ui-handoff-2026-10-18.md) is,
where it lives, and what it assumes of the host. Everything is in
`lib/features/interactive/` plus small hooks in shared files.

## Bindings

The UI talks only to `InteractionHost` and `InteractionCatalog`. A pane's
binding (`pane.ts`) supplies them:

- **Default (live).** `live-binding.ts` wraps `simulation.interaction(paneId)`
  and `interactionCatalog`. The service is resolved lazily: a scene with no
  interactive entities loads no Box3D until the first command, which waits for
  `host.ready` and is then stamped with the host's real generation and revision.
  The proxy re-resolves the service on every read, so a re-attached pane session
  moves the subscription to the new host.
- **`?interactive=demo`.** `simulated-host.ts` is the local test double, run with
  the demo declarations in `demo/`. The HUD carries a "Simulated" badge, and the
  host's readiness says it has no physics engine, collision queries or fluid.
- **`?interactive=off`.** No interactive UI at all.

`registerInteractionBinding(paneId, binding)` replaces a pane's binding.

## Surfaces

| Surface | Where |
| --- | --- |
| Placement row (kind menu grouped by catalog `group`, "Keep placing") | `InteractiveRows` in `MakeRows.tsx` |
| Ring: "Interactive" wedge on surfaces; Inspect / Duplicate / Control / Delete on an entity | `ring.ts`, `openRadialMenuAt`, the layer's `contextmenu` capture |
| Overlay: outlines, ghost, fallback note, refusal note, spawn ghost while paused mid-run, missing-plugin markers | `InteractionLayer` (SVG) in `ScenePane.tsx` |
| Play HUD: host badge, Edit/Observe/Play chip, readiness and reasons, Play/Pause, Reset, Control/Release, Lock pointer, history count, Export/Import | `PlayHud` |
| Inspector: XYZ transforms, kind and behavior controls with impact tags, local drafts kept on refusal, add/remove behaviors, connect/disconnect outputs, Duplicate, Control, Delete | `Inspector` |
| Prompts: target offers with bound key and refusal reason | `Prompts` |

When the binding's renderer already draws the bodies (`presentsBodies`), a
running scene outlines only the selected, targeted or unavailable entities. At
rest every entity is outlined, which is where sensor volumes are visible.

## Input ownership

All routing is capture-phase on the pane's shell (pointer) and on the window
(keys, focused pane only), so a claimed event never reaches the viewport or
`useEditorShortcuts`.

1. Text entry, an open ring, the scene chooser, a carry or any `aria-modal`
   dialog: nothing is claimed, and play input is dropped.
2. Play: every unmodified key is claimed. Bound ones go to the router, and
   Escape releases control and pointer lock. Drag or pointer lock turns the
   view; wheel sets the follow distance.
3. Placement draft: click commits, drag orbits, Q/E rotate 15°, Esc or
   right-click cancels.
4. EDIT with a hit on an entity: select, drag to move on its base plane (one
   `transform` on release), Cmd/Ctrl+D duplicates, Delete removes, Esc deselects.
   Placement/dragging use the selected grid spacing; inspector Apply commits
   position and rotation together. Upright kinds expose only Y rotation.
5. Everything else falls through to the editor.

`ViewportInputRouter` (`input-router.ts`) is host-agnostic. It binds the
controlled entity's actions plus the target's available offers. Buttons combine
with OR, keyboard axes are summed and clamped, and pointer-motion deltas are
scaled, sent once and summed by the host between ticks. Repeats change nothing.
`clear()` returns releases for anything still held. It runs on blur, hidden tab,
pointer-lock loss, pane unfocus, pause, reset, release and unmount.

## Host assumptions worth knowing

- Control waits for `ready`, possesses, then sends Play if the host is not
  already running.
- The play aim ray starts at the character's eye, not the camera, because the
  production host's reach is measured from the ray origin (3 m) and the follow
  camera sits about 4 m back.
- When the binding has `cameraPosition`, the follow camera is pulled in front of
  walls.
- The double refuses structural edits while running. The production host
  accepts structural edits and resets the world, but configuration honors the
  strongest declared live/rebuild/reset impact. Live/rebuild retain time and
  body motion. The UI shows whichever reason comes back.
- Import uses the production host and scene history. Export serializes authored
  records, including unavailable plugins; it never serializes current poses.

## Checks

Scratch scripts (not unit tests) drove the pane controller through both hosts.
Against the double: stale and duplicate envelopes, one history entry per
commit, the draft kept with its reason on refusal, connection cleanup on
remove, no held key after blur, reset to spawn, missing plugins retained and
blocking, and pointer deltas consumed once. Against `SceneInteractionHost`
(dry, real Box3D): placement, the host's overlap refusal, connect, Control,
WASD walk, blur stop, E on the aimed switch raising the connected platform,
release, reset and remove. The lazy live binding also committed a placement
through the controller into `scene.interaction`. Browser checks now cover inspector transforms, snapping, duplication, undo/redo,
interaction import, control/release, displayed-pose picking after a crate falls,
and a density rebuild retaining simulation time. Saving and reopening a wet
procedural fixture also passed. Pointer-lock/focus acceptance and the complete
live sculpt/undo visual check remain unverified; browser work was stopped at
the user’s request. See the scoped verification record in the architecture plan.
