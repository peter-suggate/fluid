# Runtime resource plugins

The product does not have one loading state. It has independently usable
capabilities whose implementations may allocate buffers, compile shaders,
preprocess authored data, upload publications, and wait for GPU fences.

The architecture follows the render-pass visualization framework:

- a resource owner declares its lifecycle metadata beside its initializer;
- a static catalog composes those declarations in deterministic order;
- the runtime aggregates a small protocol and never switches on owner names;
- UI gates actions by capabilities, not by whether any work is in progress.

## Colocated plugin contract

Each resource owner exports a `ResourcePluginDefinition` containing:

- a stable plugin ID;
- a coarse display lane (`platform`, `fluid`, `svo`, or `optional`);
- capabilities it provides;
- the narrow action it blocks when no usable generation exists;
- phase explanations owned by the code that performs those phases.

Current plugins are declared with their owners:

| Owner | Plugin | Capability |
| --- | --- | --- |
| `FluidLabRenderer` | `platform.webgpu-renderer` | minimum WebGPU presentation |
| `WebGPUStaticSvoScene` | `scene.static-svo-source` | preprocessed static world |
| power-octree method | `fluid.power-octree` | fluid authority and water presentation |
| `SparseVoxelDrySceneRenderer` | `presentation.svo-global` | GLOBAL sparse presentation |

`resource-plugin-catalog.ts` only composes these exports. It does not contain
their progress copy or know how they initialize. A mutable `register()` API is
deliberately avoided so import order and hot reload cannot alter the catalog.

## Readiness is not progress

Every plugin has both a state and a `usable` bit. During a transactional rebuild
the state can be `preparing` while `usable` remains true because the attached
generation stays live. Progress describes replacement work; usability describes
what the user can do now. They must never be collapsed into one enum.

Published resource evidence outranks status-message order. For example, fenced
fluid diagnostics keep fluid transport ready even if a later SVO shader compile
reports progress. One plugin's event can only update that plugin.

## Interaction gates

The shell is always interactive. GPU state never disables scene editing,
camera movement, panels, or file actions.

Initialization never installs a modal viewport loader. Platform acquisition,
fluid allocation, and SVO preprocessing all appear in the authoritative
top-right task tray while the editor and any available presentation remain
interactive. Before the first GPU presentation exists, the canvas may be a
placeholder, but it does not take ownership of the UI.

Transport is gated only by the platform and authoritative fluid capabilities.
SVO compilation, debug pipelines, preprocessing refinements, and other optional
work cannot lock play, step, or recording once fluid authority is fenced.

## Adding a resource

1. Export a plugin declaration beside the initializer.
2. Add that export to the static resource catalog.
3. Attach the declaration to progress/ready/failure events emitted by the owner.
4. Report a usable old generation during replacement when the ownership rules
   make that safe.
5. Add a catalog test and a reducer test proving unrelated capabilities remain
   usable while the plugin prepares or fails.

The compatibility reducer still accepts unscoped legacy events during the
migration. New producers must send their plugin declaration; owner-name parsing
is not part of the plugin architecture.
