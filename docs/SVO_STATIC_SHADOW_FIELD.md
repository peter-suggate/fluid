# Static SVO shadow field

The static shadow field is a conservative accelerator layered on the resident
node-mip topology. It is not a baked shadow image and has no camera, frame, or
history dependency.

Each resident sparse page owns two `u32` bitplanes in the same physical slot as
the node-mip atlas. A bit pair is one authored-light channel:

| Visible bit | Occluded bit | Meaning | Render action |
| --- | --- | --- | --- |
| 0 | 0 | dirty/unavailable | exact current-frame static trace |
| 1 | 0 | every receiver in the page is proven clear of static geometry | skip the static trace |
| 0 | 1 | every receiver in the page is proven blocked by static geometry | reject the light |
| 1 | 1 | mixed/uncertifiable | exact current-frame static trace |

Whenever a light is not statically rejected, moving rigid bodies are tested by
a separate current-frame overlay. A rigid transform changes only overlay data;
it does not invalidate this cache and never requires shader recompilation.

## Safe builder rules

A builder may publish `visible` only after proving the swept segment from every
possible receiver in the page to the whole emitter support is empty in the
static SVO. It may publish `occluded` only after proving every such segment is
blocked. Directional, point, sphere-area, and rectangle-area lights need
different swept bounds. Any inconclusive page must be `mixed`. These rules make
the accelerator exact: uncertain work takes the existing traversal rather than
using an approximate shadow.

Build work can be spread over frames in deterministic page-slot/light-index
order. Payload writes precede certificate writes in command order. Dirty
channels are `00` and therefore cannot take a lit fast path. An older complete
field is retained for allocation reuse and diagnostics, but is consumed only
when its topology generation, light revision, topology hash, and packed-light
hash all match the current request.

`SvoStaticShadowFieldCache.dirtyWork()` exposes that deterministic work list and
accepts a per-update bound. This bounds builder cost without tying correctness
to finishing in one frame: unbuilt entries keep using exact current-frame rays.

## GPU integration

1. Allocate one storage buffer of `atlasCapacity * 8` bytes and clear it when
   the static SVO generation or authored-light revision changes.
2. Dispatch the builder over `(resident page, light channel)`. Store the two
   certificate bits only after the conservative proof finishes.
3. Use the existing node-mip direct table to obtain the shared physical slot.
   Read `[visibleMask, occludedMask]` from the shadow buffer; no shadow page
   table or shader specialization is required.
4. In direct lighting, reject `occluded`, run only the dynamic-rigid overlay for
   `visible`, and run exact static traversal plus the dynamic overlay for
   `mixed` or dirty.

The fixed ABI covers all 32 SVO light records. Light count, light transforms,
rigid transforms, and page residency are buffer/uniform data, not compile-time
constants.

## Resource cost

The resident and allocated costs are exactly `residentPageCount * 8` and
`atlasCapacity * 8` bytes. A safe double-buffered replacement peaks at
`atlasCapacity * 16` bytes. For the 8,192-page hose-tank node-mip publication,
that is 64 KiB resident/allocated and 128 KiB during replacement, independent
of whether the scene publishes 1 or 32 lights.
