# Unified SVO cone tracing

## Architecture

The unified world octree remains the only simulation and structural authority. Ocean, ponds, channels, terrain support, and rigid coupling use its canonical adaptive leaves and faces. Renderer acceleration must never change leaf selection, pressure connectivity, face fluxes, transported volume, or water ownership.

Two disposable views are derived from a complete canonical publication:

1. A 4×4×4 wide-fanout directory collapses two binary octree levels per page. Its terminals point back to canonical node and leaf indices, including mixed-level environment leaves. It improves lookup and empty-region skipping without becoming a second scene model.
2. A sparse node-mip cache stores filterable opacity pages for bounded shadow and ambient-occlusion cones. Pages have an 8³ interior and one-texel apron, producing 10³ physical RGBA8 atlas tiles. Ancestor pages are allocated automatically; no resource scales with the empty finest-level world box.

Primary visibility and exact geometry tests remain available. The wide directory and mip cache are accelerators, not replacements for authoritative geometry.

## Publication and fallback

Every mip key includes the canonical source generation, virtual level, and Morton coordinate. A candidate generation becomes visible only after its directory, payload, and aprons are complete. While a candidate is incomplete, rendering retains the previous complete generation. If no complete matching generation exists, cone sampling is invalid and the renderer follows this ladder:

```text
mip cone visibility → exact bounded SVO visibility → raster structural fallback
```

The dry shader checks the sampled directory generation before cone work. A missing or stale shadow cone continues through the exact SVO shadow path. A missing or stale AO cache returns neutral visibility. Cache allocation, capacity, or upload failure therefore cannot alter canonical simulation or silently turn missing data into an occluder.

## Opacity and memory lanes

The mip atlas uses `rgba8unorm`:

| Lane | Meaning | Reduction |
| --- | --- | --- |
| R | mean solid opacity | mean of eight children |
| G | conservative maximum solid coverage | maximum of eight children |
| B | mean fluid fraction | mean of eight children |
| A | conservative maximum fluid coverage | maximum of eight children |

Mean lanes preserve optical depth; maximum lanes retain thin blockers and conservative skipping. Static terrain and opaque environment proxies populate the solid lanes. Glass and the open presentation wall are excluded. Dynamic analytic rigid bodies are tested exactly alongside cone visibility, so moving objects retain shadows and contact without dirtying the static atlas. The current static publication leaves fluid lanes zero because evolving fluid remains canonical octree state; dynamic fluid mip publication must use the same generation fence before those lanes can affect lighting.

Each physical page costs 4,000 bytes, plus 32 bytes of directory data. The wide hierarchy separately uses 32 bytes per page, 16 bytes per occupied descriptor, and 73 packed opacity words per page. Balanced targets are at most 64 MiB for garden-derived render resources and 96 MiB for ocean opacity pages. A radiance cache is not part of this milestone.

## Lighting behavior and controls

The Visual panel exposes two SVO lighting choices:

- **Exact direct** uses bounded exact SVO visibility.
- **Mip cones** requests cone-traced soft shadows and four bounded hemisphere samples for ambient occlusion.

The URL contract is `svoLighting=direct|cone`. Cone is the canonical default and is omitted from serialized URLs; `svoLighting=direct` records an explicit comparison choice. The control is disabled unless `render=svo` is selected.

Cone AO modulates environment diffuse only. It must not darken emission, direct specular, or the specular environment term. Cone shadow aperture grows with distance, selects LOD from cone diameter, integrates opacity front-to-back, terminates early near full opacity, and remains bounded to 48 steps.

## Acceptance gates

The garden milestone is complete when `garden-pond` and `garden-dam-break` can run in effective SVO mode with stable soft sun shadows and contact AO:

- no light leaks through terrain or thin opaque props;
- no visible page, apron, or LOD seams;
- no camera swimming or paused-frame shimmer;
- no persistent trails after moving water or rigid bodies;
- no partial generation, NaN, unreported exhaustion, or silent fallback;
- mean visibility error ≤ 0.05 and p95 error ≤ 0.10 against a high-sample reference;
- human sign-off on tree, mushroom, stones, cork ball, and terrain contacts.

Performance gates at 1280×720 are:

- wide traversal reduces node/page lookups by at least 35% and relevant p95 time by at least 20%, or remains optional;
- cone shadows plus AO add no more than 1.5 ms p95 in balanced quality;
- unchanged mip maintenance costs no more than 0.1 ms;
- garden dirty rebuild costs no more than 0.5 ms and ocean dirty rebuild no more than 1.0 ms;
- page lookup hit rate is at least 95%, averaging no more than 1.25 lookups per cone step;
- balanced total presentation remains within 4 ms, high within 6 ms, and ultra within 8 ms.
