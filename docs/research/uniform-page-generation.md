# GPU residency and persistent page pool

Implemented in `lib/methods/uniform/uniform-page-generation.ts`. This component is
not connected to production fluid operators yet. The production UI still uses the
initial all-resident page-domain cutover; this change does not claim a frame-time
or scene-memory improvement.

## Transaction

The caller supplies a GPU buffer containing the complete desired page support set.
Initialization or external edits may upload inputs, but per-frame membership is to
be produced by GPU operators. Signed i32 page coordinates have exact hashed lookup;
world extent and distance between components do not enter allocation sizes.

`encodePrepare` plans a candidate directory and slot list, merges duplicate request
roles, retains stable slots, builds six-neighbor adjacency, and initializes newly
allocated fields. Initial field values are provided per canonical lattice coordinate.
Initialization is parallel and indirectly dispatched from GPU counts. Field access
accepts page coordinates plus local coordinates, including seam-crossing offsets.
It returns an explicit missing-address sentinel rather than inventing boundaries.

`encodePublish` copies the candidate metadata to the accepted generation only when
its GPU fault word is clear. Exhausting request or page capacity leaves accepted
metadata and fields intact. Slots occupied in the accepted generation are never
reused during that transition, even when retirement is requested. A later ordered
transaction can reuse them after all old consumers. Empty/refill transitions work
without reallocating any buffers.

The host only encodes fixed passes and a GPU buffer copy for indirect arguments.
There are no map/readback calls, demand-dependent allocations, or host decisions
between preparation and publication.

## Boundaries of this component

- The request producer must prove retirement is legal and keep all mass, interface,
  source and operator support. This allocator does not infer that policy from fields.
- Topology planning uses one bounded GPU invocation over pool metadata and requests.
  Field initialization is parallel. The serial planner avoids publication spinlocks
  but needs measurement at realistic large page counts before production adoption.
- Publication is atomic with respect to **residency**, not a complete fluid advance.
  A full step still needs candidate field/history buffering and a guarded clock,
  source-consumption and presentation commit. Retained fields cannot be updated in
  place and then described as rollback-safe merely because directory commit is gated.
- The pool is fixed-budget. Failure preserves the last accepted generation; it does
  not resize the pool or silently steal a page containing liquid.
- Cached neighbors and generic field addresses are not yet pressure, terrain or
  surface adapters. The production dense textures have not been replaced.

## Validation

`tests/uniform-page-generation-dawn.test.ts` passed on Dawn/Metal for both page edges:

- Duplicate role union, stable physical slots and neighbor links against the host oracle.
- Positive/negative seam ownership and exact distant-page lookup.
- Correct initialization of every field element, including signed zero.
- Persistent retained values, deferred reuse, emptying and refilling.
- Page-capacity and request-capacity rollback, comparing the entire accepted metadata
  and field pool after failure.
- Signed i32 coordinate limits without wrapped adjacency.
- GPU-produced requests followed by prepare/initialize/publish in one submission.
- Unchanged allocated bytes as pages move to distant signed coordinates.

Run this focused gate with the browser GPU unloaded:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/uniform-page-generation-dawn.test.ts
```

It is also included in `npm run test:dawn:uniform-page-domain`. No Sparse CM12
implementation or shared topology code was modified.

Next: define and test the finite phi-band/support contract, connect its GPU support
producer to this pool, then replace operator field reads/writes and the coupled
pressure hierarchy. Production residency cannot shrink safely before those consumers
stop relying on global far-air phi history and dense field addresses.
