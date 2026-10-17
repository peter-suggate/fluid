# Interactive scenes UI handoff

The UI developer owns placement surfaces, inspectors, DOM keyboard/mouse wiring, camera presentation and play controls. The production runtime is now implemented. Use the pane service below; do not instantiate another physics world or advance it from a render callback.

```ts
import { simulation } from "@/lib/core/simulation/controller";
import { interactionCatalog, InteractionInputRouter } from "@/lib/interaction";

const host = simulation.interaction(paneId); // stable per-pane service
const input = new InteractionInputRouter(host);
const unsubscribe = host.subscribe(() => render(host.getView()));
await host.ready;
```

`simulation.interaction()` connects accepted edits to scene history and reset, publishes native body poses into the existing viewport diagnostics, and participates in the shared pane clock. The viewport already attaches its fluid worker driver. No additional physics setup is needed by the UI.

The product design remains in [the architecture plan](interactive-scenes-2026-10-18.md). The declaration/catalog APIs stay pure; importing them does not load Wasm until a host is constructed.

## Available implementation

| API | Implemented behavior |
| --- | --- |
| `createInteractionCatalog({ entities, behaviors })` | Snapshots and validates colocated declarations. No mutable registration or import side effects. |
| `catalog.entities` / `catalog.behaviors` | Immutable metadata; entity placement order sorts by declared order then ID. |
| `catalog.entityKind(id)` / `behaviorType(id)` | Looks up definitions, including previews, configuration controls and logical actions. |
| `catalog.createEntity({ id, kind, transform, name?, config? })` | Creates a complete current-version record from a recipe, validates preview/configuration/composition, and returns an immutable entity. Does not add it to a scene. |
| `catalog.inspectDocument(raw)` | Validates the JSON envelope, migrates supported configs, validates references, and preserves unavailable plugin records for round trip. Returns document plus unavailable reasons. |
| `catalog.planDocument(raw, hostCapabilities)` | Requires all plugins and capabilities, checks connection compatibility and exclusive owners, and returns a deterministic phase plan. Allocates no runtime resources. |
| `catalog.validateAction(document, sample)` | Validates a logical input target, value, sequence shape and button edge against declarations. Use a document already inspected by the catalog. |

Imports come from the single public entry point:

```ts
import {
  createInteractionCatalog,
  type InteractionHost,
  type EntityKindPlugin,
  type BehaviorPlugin,
  type CommandEnvelope,
} from "@/lib/interaction";
```

`lib/interaction/verification/fixtures.ts` contains working character/motor/camera
declaration examples. They are contract fixtures, not production entities or a
playable character. The accompanying tests exercise creation, migrations,
unknown plugins, ownership conflicts, scheduling, wiring and input validation.
Do not import verification fixtures into production UI.

## Placement and configuration

Build the Interactive ring group and placement surface from `catalog.entities`.
Use `editor.label`, `hint`, `icon` (SVG path), `group` and `order`; do not branch
on kind IDs. Selecting a kind starts a local placement draft using
`schema.defaults`. Call `preview(config)` for its origin-centered box, sphere or
upright capsule. Capsule height includes both hemispheres. Placement policy
specifies surface alignment and whether overlap is allowed; collision queries
and final placement admission are host responsibilities.

Position the draft with the existing ray/gesture infrastructure, show a ghost,
and commit once on click. Escape cancels without creating history. In repeat
mode each committed entity gets a fresh stable ID. Block unresolved collision
readiness and show the host's reason instead of pretending overlap checks ran.

```ts
const kind = catalog.entityKind(selectedKindId)!;
const entity = catalog.createEntity({
  id: newStableId,
  kind: kind.id,
  transform: placementTransform,
});
const view = host.getView();
const result = await host.dispatch({
  id: newCommandId,
  generation: view.generation,
  documentRevision: view.documentRevision,
  command: { kind: "place", entity },
});
```

IDs use letters, digits, dots, underscores, colons and hyphens, beginning with a
letter or digit. Runtime engine handles must never be used as document IDs.
`createEntity` takes a complete config when one is supplied; merge a draft with
defaults in the editor before validation if only some fields were changed.

Each configuration control names a top-level `setting`. Number, choice, toggle
and readout controls reuse the framework metadata vocabulary. Editable controls
declare `live`, `rebuild` or `reset`; expose that consequence before an edit that
will reset a running scene. Runtime action buttons are separate `actions` on
behavior declarations. Shape previews may depend on configuration; refresh the
ghost after a validated change.

Use `configure` with a full `{ version, config }` value, addressing either the
entity or a behavior instance. `set-behaviors` replaces the complete authored
behavior list. The host must validate the resulting document before committing
it. The schema supplies config validation; the UI must not duplicate numeric or
migration policy in plugin-specific branches. Preserve a draft if validation
fails and display its error.

## Document rules

`InteractionDocument` is an embeddable version-1 fragment with `entities` and
`connections`. It is not yet a field of `SceneDescription`. The integration owner
must add the scene schema migration, import/export/library handling and store
adapter before live scene persistence is enabled.

Kind recipes run on creation only. Saved behavior instances are the complete
authored composition, so loading a file never silently reruns a changed recipe.
Kinds and behaviors version their configs independently. Migrations return the
current config and must pass its validator. Unknown or newer plugins retain
their data; show unavailable placeholders and preserve them on export. Use
`inspectDocument` for editing/import, and `planDocument` to gate execution.

Entity IDs are document-wide; behavior IDs are local to an entity. Connections
reference entity ID, behavior instance ID and port ID at both ends. Ports use a
canonical channel ID and must match ID, representation, lifetime and unit. For
example, a switch output and platform input can both declare the `active`
boolean channel. Port display labels can be added by the owning UI adapter;
never change a port ID solely to change its label.

A deletion must remove affected connections in the same history transaction or
be refused. Duplication must remap copied entity and behavior references
consistently; these operations belong to the scene adapter and are not provided
by the catalog. Runtime poses, velocities, engine handles and input state never
enter autosave. Reset returns to the authored configuration.

## Host boundary and commands

Obtain one `InteractionHost` per pane through `simulation.interaction(paneId)`.
The interface provides `getView`, `subscribe` and asynchronous `dispatch`.
Use a local test double only in UI unit tests; live integration uses the concrete
service already attached to the application clock.

The view supplies document and revision, generation, readiness and reasons,
running state, edit/observe/play context, controlled entity ID and an optional
completed world snapshot. `controlCandidates` supplies per-entity Control
availability and refusal reasons. `target` supplies the currently aimed-at
entity and available action offers; use those labels and behavior/action IDs
for prompts instead of recognizing character or switch kinds. Each body snapshot includes authority, pose, velocity,
shape reference, bounds and body generation. Render interpolation must not write
back to the document.

| Command | Host admission and effect required by the contract |
| --- | --- |
| `place`, `remove`, `transform` | Validate document and geometry; one authoring/history transaction. |
| `duplicate` | Copy selected authored entities with fresh entity/connection IDs and an explicit offset; remap copied connection endpoints. Preserve external links by default; `externalConnections: false` copies internal links only. |
| `import` | Validate and replace the interaction document atomically; one history entry. Preserve unavailable plugin records and block play until resolved. |
| `configure`, `set-behaviors` | Validate schemas and composition, then apply the strongest declared impact. |
| `connect`, `disconnect` | Validate endpoints and port compatibility; one history transaction. |
| `possess`, `release-control` | Acquire/release the eligible entity's input context; do not change the document. |
| `aim` | Update the view ray; resolve range and occlusion into target offers without authoring changes. Direction must be a unit vector. |
| `action` | Validate target/value, then queue input for an admitted fixed tick; no history entry. |
| `transport` | Play, pause or reset through the existing clock owner. |

Every envelope contains a unique command ID, world generation and expected
document revision. The host rejects stale envelopes and duplicate submissions,
validates before mutation, and returns `{ accepted: false, reason }` on refusal.
Accepted commands return the resulting document revision and actual update
impact. Runtime-only commands retain the document revision and use `live` unless
they reset the world. Reset increments generation. A reply is admission, not
proof that all asynchronous resources are ready: use the subscribed readiness
state to enable subsequent actions. A UI gesture must not fire again just
because a view update arrived. Generation/revision enforcement and duplicate
tracking are implemented by the concrete host, not by the catalog.

## Input and play experience

Keep transport separate from input context. Preserve LOOK/EDIT and add a Play
context; the current internal `interact` name already means editing. Offer an
explicit Control action on the character. From paused state it performs Play,
waits for ready state, then acquires control. Pointer lock requires a direct user
gesture and must fall back to drag-to-look if that gesture cannot be retained.

Logical actions belong to a behavior instance. Keyboard bindings use physical
`KeyboardEvent.code` and declared contributions; pointer buttons use their
numeric button and contribution; pointer motion declares a two-axis scale.
Clamp axis contributions to [-1, 1] per axis. The character motor normalizes a
diagonal movement vector as needed; the input router does not assume every
axis2 action is movement. Button bindings combine with OR. Sum deltas once per
sample and clear them after consumption; do not repeat a render-frame mouse
movement across every fixed tick.

The initial product defaults are WASD movement, Space jump, Shift run, E use,
and mouse camera control. The core catalog does not hardcode these defaults.
Use `actions` to display hints and build routing. `ActionSample` carries an
entity, behavior and action ID, normalized value, sequence and optional button
edge. Sequence is monotonic within a pane generation. The host consumes edges
once; `validateAction` checks shape and target but does not deduplicate samples.

Route input through a single owner: text/modal UI, active gesture, possessed
entity, then observer/editor. Prevent editor shortcut handlers from also seeing
consumed gameplay events. Clear held keys and deltas on blur, hidden tab, pane
change, pause or pointer-lock loss. Only the focused pane owns input. Explicit
A/B mirroring can be added later.

Escape closes the innermost modal/gesture, otherwise releases control and pointer
lock while simulation continues. Pausing releases pointer lock and suspends
inputs; reacquisition needs a new gesture. Keep visible Control, Release and
Pause controls. The camera begins in third person with drag-to-look; continuous
pointer lock is optional. Use current view/range/occlusion for action prompts,
then let the host revalidate the target on execution.

## Production runtime integration

The production contract and backend are ready for UI integration. Native Wasm tests, repository type/unit checks, the production bundle, and the real Metal fluid/Box3D smoke check pass. See the [implementation verification](interactive-scenes-2026-10-18.md#implementation-verification-18-october-2026) for measured timings and the existing cold fluid-shader compilation limitation.

Five production packages are installed: `fluid.character`, `fluid.crate`, `fluid.switch`, `fluid.platform`, `fluid.trigger`. Enumerate `interactionCatalog.entities` for placement. Each supplies its schema, preview, placement rules, editor controls, behavior composition, actions and typed ports. Register new declarations and factories together in `lib/interaction/installed.ts`. Switch logic is independent of its static body; it can also be attached to a crate or another physical entity.

Create an entity through `interactionCatalog.createEntity`, then dispatch `place`. Placement, removal, transforms, duplication, import, composition and connection edits validate before committing, reset runtime poses and record one history entry. `configure` applies the strongest changed control impact: `live` updates runtime parameters; `rebuild` replaces only the affected native body at its current pose and velocity; `reset` rebuilds the world. Undeclared settings default to reset. Live/rebuild preserve the world clock and generation; a body replacement increments its own generation for fluid feedback. Geometry rebuilds retain behavior state, including an activated platform. Rejected edits create no history entry. IDs must be unique. Reset restores authored spawn transforms. Save/load persists only `scene.interaction`, never current physics poses. The production binding exposes Import and actual scene-store history; undo/redo restores the authored document and resets its runtime.

Each command envelope uses the latest `getView().generation` and `documentRevision`, plus a unique command ID. Accepted structural changes invalidate older envelopes. `transport`, `possess`, `release-control`, `aim` and logical `action` commands are live commands. `aim` uses a normalized world ray and a three-metre range; the host resolves occlusion and rechecks ordinary use actions on the execution tick. Observe/play contexts remain separate.

The input helper is DOM-independent. Call `input.key(event.code, down, blocked)` or `input.button(button, down, blocked)` only for the focused viewport; `blocked` covers text entry, modal UI and editor gestures. Prevent defaults only when it returns true. Call `input.flush()` after target changes. Call `input.reset()` on blur, hidden document, pane change and pointer-lock loss, and `input.dispose()` on unmount. Pausing must also clear the router's local keys (`input.reset(false)`); the host already clears admitted input and releases grab constraints. Keyboard repeats do not repeat pressed edges. UI owns pointer-lock requests and Escape/modal priority.

Default bindings: WASD move relative to the aim direction, Shift run, Space jump, E toggle a targeted switch, and G hold a targeted crate with a bounded native motor joint. The UI can also invoke logical actions directly for visible buttons. `host.cameraPosition(focus, desired, clearance)` supplies conservative ray-fan camera clearance; UI owns camera interpolation and orbit behavior. Trigger previews are editor overlays; triggers are invisible during simulation.

`BodySnapshot` includes pose, velocity, bounds, authority (`static`, `dynamic`, `kinematic`, `character`), grounded and submerged observations. Body render IDs are `interactive:<entity-id>`. Use those IDs to associate viewport picking with entity records. Existing gizmo body methods edit legacy bodies; interactive transforms go through the host. Do not write generated bodies into `scene.rigidBodies`.

The runtime owns ordering, input-edge consumption, next-tick connection events, native collision, generation checks and feedback. Fluid is fenced before Box3D advances; a failed, missing or still-preparing fluid adapter blocks wet play. Attached production drivers publish readiness changes so the HUD cannot admit Play/Control during compilation. Dry scenes allocate no fluid solver. The current ABI supports 12 non-sensor physical bodies total, including legacy bodies. Unsupported wet methods/coupling settings fail explicitly. Compound legacy cups preserve their opening.

The initial collision representation is merged boxes plus convex tetrahedral clips of continuous cells. Solid-edit preflight rejects newly occupied body interiors at fixed sampling resolution; it is not an exact arbitrary-shape overlap oracle. Physics collision stays independent of camera-dependent SVO detail. The scene's approved solid edits enter the native world at a serialized tick boundary.

## UI acceptance checklist

- Placement metadata, previews and inspector fields come from the catalog.
- Placement cancellation authors nothing; commit creates one history entry.
- Failed commands retain useful drafts and display the supplied reason.
- Missing plugins survive save/load and visibly block unsupported simulation.
- Play, pause, Control and Release have distinct visible states.
- Text entry, modal dismissal and play input never trigger competing handlers.
- Blur, pointer-lock loss and pane switching cannot leave a held movement key.
- Reset restores authored spawn state; playing cannot mutate autosave.
- Host integration tests verify stale command rejection, input-edge consumption,
  connection cleanup and readiness transitions with the concrete host.

UI mock testing does not validate physics or fluid coupling. The native runtime and real-GPU check are supplied in the repository; UI focus and pointer-lock acceptance remains part of the receiving UI work.

## Authoring and solid verification follow-up

The inspector now offers XYZ position and rotation (Y rotation only for upright kinds), a single Apply/Cancel transform transaction, and Duplicate. Cmd/Ctrl+D and the entity ring use the same duplication command. The placement menu offers free placement and 0.05/0.1/0.25/0.5 m snapping; snapped surface placement stays on the hit tangent plane. Picking uses the completed native pose, checks solid occlusion, and derives normals in the rotated primitive's frame.

See [the scoped verification record](interactive-scenes-2026-10-18.md#authoring-and-solids-follow-up-steps-3-and-4) for passed checks and the browser work still outstanding. Do not treat headless fluid/collision checks as completion of the live browser sculpt/undo acceptance check.
