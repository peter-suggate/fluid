# Editor entities

The editor does not have one editable object. It has a tank, a body of water,
rigid bodies, props and a hose, which differ in what they are made of, in which
degrees of freedom their schema can express, in what a drag costs the solver,
and in whether they exist at all in a given scene.

They do not differ in what direct manipulation means.

So the editor follows the runtime resource-plugin architecture, one level up:

- an entity declares how it is edited beside its own implementation;
- a static catalog composes those declarations in deterministic order;
- the viewport drives a narrow protocol and never switches on entity names;
- a new editable thing is a new declaration, not a new branch in the pointer
  state machine.

## One mode

There is no second mode for the things that happen to be large. SELECT is the
only editing tool: clicking anything selects it, its handles appear because it is
selected, X/Y/Z constrain whatever is held, and Escape or a click that reaches
nothing deselects. The tank is clicked on the inside of its walls and floor, the
water body on its seed box — the same box the handles move, which is the only
target an edit can actually reach.

One consequence is worth naming: the tank is behind everything, so a click on the
floor selects the tank rather than falling through to the background. Escape
still deselects, and so does a click that leaves the tank entirely.

## Colocated declaration

Each entity exports an `EditorEntityDefinition` from the module that already owns
it:

| Owner | Export | Selection kind |
| --- | --- | --- |
| `editor-rigid-body.ts` | `rigidBodyEntity` | `body` |
| `editor-inflow.ts` | `inflowEntity` | `inflow` |
| `editor-props.ts` | `propEntity` | `prop` |
| `editor-fluid-body.ts` | `fluidBodyEntity` | `fluid-body` |
| `editor-tank.ts` | `tankEntity` | `tank` |

`editor-entity-catalog.ts` only composes these exports. It does not know how any
of them are drawn, dragged or written back, and there is deliberately no mutable
`register()` so import order and hot reload cannot change what the editor offers.

Order is the tie-break for *handles*: the water body is listed before the tank
because the tank encloses it, so a pointer within tolerance of both is over the
thing actually visible there. Clicks on the scene are ordered by distance
instead, because there the geometry already says which is in front.

## The load-bearing part

`EditorHandle.drag` closes over the entity's own write-back. The viewport
resolves a pointer ray to a point, calls `drag`, and writes whatever patch comes
back. Nothing downstream of that call can tell a tank from a sphere:

```
press  → entityHandleAtPointer(selection's handles) → capture {entity, handle}
         from the COMMITTED scene, plus the grab offset
move   → frameRayToLocal → 1-DOF? closestPointOnAxis : planeHit
         → handle.drag(point, constraint) → updateDraft(patch)
release→ commitDraft()
```

Two invariants keep this stateless:

- Handles are **redrawn from the display scene**, so a live drag needs no
  separate preview path — they are simply the handles of the entity the scene now
  describes.
- The drag **resolves against the committed scene and the entity the gesture
  opened on**, so the handle cannot walk away from the pointer.

## Frames

Handles are local to the entity. A rigid body tumbles, so grabbing the face
currently facing you resizes along the body's own axis and an X lock constrains
along the body's X — Blender's local constraint. The hose's frame carries its
aim, which is what lets a nozzle be an ordinary box entity: local x and z are its
radius, local y its length. The tank and the water body are the degenerate case
of the same rule, with an identity frame.

Move handles are the exception and say so: `space: "world"`. They act on the
frame itself, and dragging a body along world X is what the arrow visibly
promises.

## Linked axes

`dimensions_m` means something different for each shape — a sphere stores one
radius, a cylinder a radius and a height, a box three full extents. The box gizmo
must never propose a state the document cannot hold, so a shape declares the axes
that share a number and `resizeBox` keeps them equal:

| shape | local half-extent | linked |
| --- | --- | --- |
| rigid sphere | `(r, r, r)` | x·y·z |
| rigid box | `dimensions_m / 2` | — |
| rigid cylinder | `(r, h/2, r)` | x·z |
| rigid capsule | `(r, h/2 + r, r)` | x·z |
| prop box / ellipsoid | `halfSize_m` | — |
| prop cylinder | `(r, hh, r)` | x·z |
| hose | `(radius, length/2, radius)` | x·z |

A linked group resizes symmetrically about the box centre, because that one
number is a half-extent about the centre and there is no side to hold still. Free
axes keep the opposite side pinned, which is what makes a face drag read as
moving that face — and, since a body's centre *is* its position, the position
travels with the size.

Where a linked group owns more than one dragged axis — a cylinder grabbed by a
corner — the largest requested half-extent wins. Reading it radially would be
unstable at rest: the corner of a square cross section stands at r·√2 from the
axis, so the radius would jump outward the instant the handle was touched.

## What a gesture costs

Three tiers, and each entity states which it is in rather than the viewport
deciding for it.

**Authored, presented live.** A prop is render-only and outside the solver's keys
entirely, so its draft is drawn immediately — `PRESENTED_DRAFT_SUBJECTS` in the
viewport. Terrain is here too: it cannot move the lattice.

**Authored, overlay-only.** The tank and the water body own geometry the solver
allocated for, and drawing the fluid at a size it was not allocated for would
tear. The wireframe box is the preview, and the document is written once, on
release.

**Simulated.** A rigid body is drawn from live solver state, and `rigidBodies` is
part of the seed key — so a document write per pointer-move would re-seed the run
dozens of times a second and still not move what is on screen. `simulatedBodyId`
says so, and the gesture previews as a runtime pose.

## Adding an entity

1. Export an `EditorEntityDefinition` beside the code that already owns the thing.
2. Give it `instances`, `find`, and a `pick` — an entity whose handles appear only
   once it is selected, and which no click can select, is unreachable.
3. Build its handles with `boxHandles` (plus `boxResizeDrag` unless it has its own
   symmetry, as the tank does) and `moveHandles`.
4. Declare its resize policy: the snap step, the room it has, its minimum, and the
   axes its schema keeps equal.
5. Add the export to `EDITOR_ENTITIES`, in the position its pick ties want.
6. Add a catalog test and an entity test proving the shapes it cannot express are
   the shapes it never proposes.
