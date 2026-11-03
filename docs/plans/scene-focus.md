# Scene focus: one place where the hand acts

Status: implemented (see "Milestones" at the end). Owner of the focus module,
the gesture hosts and placement; the voxel tool plugins belong to the tool
consolidation work and only read `context.focus`.

## Problem

Every placement interaction answered "where is the hand?" in its own way:

| Interaction | Before | Where |
| --- | --- | --- |
| Pen (voxel tool) | nearest visible surface (`visibleToolSurface`) | `voxel-editor/plugins/pen.ts` |
| Water ball / cube / torus (voxel tools) | a horizontal plane at the *Height* control | `voxel-editor/fluid-geometry.ts` |
| BALL gesture (`b`) | rest on the analytic hover surface (`restFluidInWorld`) | `WebGPUViewport.tsx` |
| DRAG gesture (`d`), new body | rest in the container (`restInContainer`) | `WebGPUViewport.tsx` |
| Throwing / sweeping a body | camera-facing plane through the grab point | `WebGPUViewport.tsx` |
| Selected entity handles | handle axis/plane, narrowed by `axisConstraint` (X/Y/Z, Shift = plane) | `editor-axis-constraint.ts` |

None of them could hold the hand at a fixed height, lift it straight off a
surface, or draw a straight line. Only handles had a constraint at all, and
only while something was selected.

## Concept

A **focus** is the point the hand acts at: `{ point, normal, distance_m?,
locked? }`. It is resolved from the pointer ray by one module,
`lib/core/editor-focus/`, in one of two ways:

- **Free** (no lock): the consumer's own *unlocked pick*, untouched. For the Pen
  that is exactly `visibleToolSurface` (the shared geometry path — no ground
  plane is ever invented); for a water drop it is its drop plane; for BALL and
  DRAG it is where the object rests on the hover surface. Unlocked behaviour is
  byte-for-byte today's: the resolver calls the pick and returns its answer.
- **Locked**: the ray projected onto a world **axis line** or **plane**
  through an **anchor**. The lock is the existing `axisConstraint` (the set of
  allowed axes, `["y"]` = along Y, `["x","z"]` = the XZ plane), generalised from
  "selected handles" to "anything that places".

### Anchor

- At the press: the unlocked pick under the press (the press point).
- When the lock is engaged or switched mid-drag: the current focus, so the
  hand does not jump.
- A press that the unlocked pick misses (open air) reuses the last stroke's
  anchor — so repeated drops or strokes at one height work without re-aiming.
- While hovering, the anchor previews the press: the guide follows the pointer
  over surfaces and freezes on the last anchor over open air.

The lock itself stays up (latched) across strokes until released.

### Locked geometry

- Plane: ray ∩ plane through the anchor, normal = the locked-out axis. The
  locked coordinate is copied from the anchor exactly (no drift). Rays parallel
  to the plane, hits behind the eye, and hits beyond the reach limit (eight
  container extents) resolve to nothing — the press then goes to the camera, as
  a press on nothing always has.
- Axis: closest approach of the ray to the axis line (`closestPointOnAxis`),
  rejected when nearly parallel or behind the eye; the two other coordinates
  are copied from the anchor.
- Normal of a locked focus: the plane normal (or the axis-perpendicular
  direction) turned to face the viewer.

## API (`lib/core/editor-focus`)

```ts
type FocusAxes = readonly EditorAxis[];              // = AxisConstraint, 1 or 2 entries
interface FocusPoint { point: Vec3; normal: Vec3; distance_m?: number; locked?: boolean }
type FocusPick = (ray: EditorRay) => FocusPoint | undefined;

projectFocus(ray, axes, anchor, reach_m?): FocusPoint | undefined   // pure
resolveFocus(ray, lock | undefined, pick, reach_m?)                  // pure, one sample
class FocusSession implements ToolFocus                              // a stroke's anchor bookkeeping
  resolve(ray, pick): FocusPoint | undefined
focusGuide(axes, anchor, extent_m): EditorHighlight                  // the guide lines
focusLockLabel / focusLockHint                                       // caption text
FocusKeyHold                                                         // tap vs hold on the lock keys
```

`ToolContext.focus?: ToolFocus` (additive, optional). A tool resolves every
position as `context.focus ? context.focus.resolve(ray, pick) : pick(ray)`. A
scripted caller that omits it gets today's behaviour. The gesture host builds a
`FocusSession` per hover sample and per stroke; the session reads the live lock
from the UI store, so tapping a key mid-drag takes effect on the next sample.

## Key map

| Keys | Effect |
| --- | --- |
| `X` / `Y` / `Z` | Lock the focus to that world **axis** through the anchor (`Y` = straight up/down). |
| `Shift+X` / `Shift+Y` / `Shift+Z` | Lock to the **plane** that holds that axis fixed: `Shift+Y` = XZ (fixed height), `Shift+X` = YZ, `Shift+Z` = XY. |
| Same keys again | Release (the key that entered the lock leaves it). A different lock key switches, keeping the hand where it is. |
| Hold the key (≥ 350 ms) and let go | Spring-loaded: the lock lasts only while held. While a lock key is held, Shift belongs to the chord, so `Shift+Y`-held + press still strokes rather than pans. |
| `Esc` | Releases the lock first (then the stroke/tool, as before). |

Active in EDIT whenever a selection stands (handles, unchanged) **or** a
placing tool is armed: any voxel tool (Pen, water shapes, …), BALL (`b`) or
DRAG (`d`). Key repeats are ignored. Nothing else changed: Shift / Ctrl /
middle button / Space still hand the press to the camera, Alt is still "do the
opposite", brackets and Alt+wheel still size. The water tools' `X` = "remove
water" toggle is retired (Alt already removes, and the rail toggle remains);
no voxel tool may declare X/Y/Z as a shortcut (unit-tested).

## Per consumer

- **Pen**: unlocked = nearest visible surface (unchanged). Locked = the stamp
  centre is the focus. Draw on a lock builds in open air (a straight bead with
  an axis lock; a level slab or wall course with a plane lock). Carve on a lock
  removes only where the stamp meets solid — in air it is a no-op — which is
  what makes level cuts and straight grooves. The axis lock is Pen's straight
  line (the Line tool can retire). A press in open air with a lock up uses the
  last anchor; with no lock it still goes to the camera.
- **Water ball / cube / torus tools**: unlocked = their drop plane at the
  Height control (unchanged). Locked: the drop centre is the focus (bottom
  snapped to the voxel grid, x/z snapped as before); `Y` lifts the drop
  straight up/down while dragging, plane locks slide it in a vertical plane.
  The tank-bounds check is unchanged.
- **BALL (`b`)**: unlocked = rest on the hover surface (unchanged). Locked:
  press anchors at the rest centre; dragging moves the ball along the lock
  (instead of sizing it); release drops at the focus. The cursor preview
  follows the lock.
- **DRAG (`d`), placing a body**: as BALL, with the body kept inside the
  container (same clamp as `restOnHover`).
- **Throwing/sweeping an existing body** (press on a body with DRAG armed or
  with nothing armed): unlocked = the camera-facing plane through the grab
  point (unchanged). With a lock up, the body moves on the lock through the
  grab point.
- **Selected entity handles**: unchanged (they already consume
  `axisConstraint` through `constrainHandleAxes`).
- **Carry**: not changed; it owns the keyboard while active (Q/E tilt, wheel
  depth) and already rides a vertical plane.

## Feedback

- **Guide**: drawn in `EditorHighlightLayer` (a `guide` layer, dashed, coloured
  by axis): an axis lock draws the axis line through the anchor; a plane lock
  draws the plane's two axis lines and a square outline around the anchor. It
  is a drawing only — hit testing never sees it.
- **Caption**: the existing lock chip at the top of the viewport names the lock
  and how to release it: `ALONG Y · Y or Esc releases`, `ON XZ PLANE ·
  Shift+Y or Esc releases`. With a held handle the existing handle wording is
  kept.

## Tests

`tests/editor-focus.test.ts` (resolver, session anchoring, key hold, guide) and
`tests/editor-focus-consumers.test.ts` (Pen unlocked identical to the plain
pick, Pen straight-line/level strokes and air strokes, water tools lifted on
`Y`, placement rest points, no tool shortcut on X/Y/Z).

## Milestones

1. Design + resolver (this doc, `lib/core/editor-focus/`).
2. Host integration (`useVoxelToolGesture`, keys, guide, caption) + Pen.
3. Water tools, BALL/DRAG placement, body throw.
