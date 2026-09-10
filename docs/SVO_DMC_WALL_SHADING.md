# DMC false crease shading

2026-09-11. DMC's zero scalar is a result of sliver elimination: fitting can snap a nearly planar smooth patch onto the zero level. It is not a crease indicator. The extractor formerly treated any zero-valued corner as sharp, making adjacent smooth patches alternate between geometric triangle normals and interpolated field normals.

The extractor now compares valid published field gradients at the eight incident fitted cells. A pair with dot product below 0.8 (~37 degrees) retains sharp geometric shading. Missing/air normals are ignored. This is a local angular crease criterion, not a primitive-specific rule; strongly curved under-resolved cells can also exceed it. The implementation runs in the existing GPU extraction passes, reuses the existing payload normals, and adds no buffer or field evaluation. Vertex positions, scalar fitting, face decisions and geometric shadow-receiver normals are unchanged.

## Validation

- Dawn shapes pass for 8- and 16-cell bricks, including closed meshes spanning brick seams, sphere, sharp box, thin wall, ambiguous faces, and a gently curved capped patch.
- The gently curved patch has no sharp triangles in its smooth interior. Reinstating the old zero-value classifier makes that regression fail with 50 incorrectly sharp interior triangles.
- Scheduler and reconstructed shadow-receiver Dawn tests pass.
- An 800×480 x10 garden A/B with shadows, AO and GI enabled has identical hardware-depth hashes (`0x595b6e59`) and triangle counts (2,677,320 total; 975,167 drawn). The captured normal/shading buffers change. The conspicuous alternating light patches on the front wall are reduced; finer grid patterns remain.

Captures and receipts: `artifacts/svo-dmc-wall-shading/{before,after}`. These captures use the existing requested-depth-3 fallback to built depth 1 on the M1 Max. No topology or resolution policy changed in this shading fix.
