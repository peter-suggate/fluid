# Feature-preserving coarsening in 2D and 3D

## Policy

The direct level-set planners now size material by deformation and representability, rather than absolute translation speed or the range of surface normals across a page. Swept support still follows absolute velocity. Support residency does not independently demand finest resolution; 2:1 grading still applies.

Surface demotion measures candidate bilinear/trilinear restriction against accepted phi. It bounds contour displacement and rejects a candidate box whose same-sign corners would erase represented opposite-sign fluid/air samples. Thin-fluid and injection floors remain. The existing volume/phi mismatch no longer vetoes coarsening: conservative field transfer owns extensive-volume preservation, and disagreement remains diagnostic.

Bulk requests its coarse target directly. Surface transitions retain two accepted proof epochs by default, with generation checks. Retained material has a four-finest-spacing working limit in this first implementation; larger cells require stronger transport reconstruction evidence. Empty support can use larger cells. Authored regional bounds and solid-boundary constraints remain authoritative.

3D phi now follows the accepted solver rung, eliminating unconditional finest-phi interface backing. The 2D reference still stores its direct phi field on its existing dense vertex grid; its solver coarsening criteria match 3D, but this change does not introduce a new sparse 2D phi storage format.

The GPU change reuses the existing census and proof dispatches. Velocity min/max replaces the former normal min/max reduction. The page-wide normal proof, its extra candidate-normal samples and the independent V/phi veto were removed. No runtime topology repair or generation was introduced.

## Checks

- 26 focused Rust resolution tests pass, including thin-feature protection, a droplet lost by coarse interpolation, stale certificates, disabled coarsening and translation-independent support sizing.
- 13 Rust world tests pass. The resting-pool test now requires four-spacing cells after six steps and an exactly unchanged contour.
- Preliminary 3D twin-dam run passed all 91 health checkpoints through 3 seconds and coarsened strongly; the unrestricted eight-spacing material experiment was rejected after Figure 7 showed excessive spreading. Final four-spacing-limit results are recorded below after verification.

The old Figure 7 free-fall assertion expects gravity displacement on the first transport step. It fails at frame one under the concurrent frame-order refactor; it was not weakened. Health/shape receipts are evaluated separately, and passing health alone is not a shape-accuracy claim.

## Build and Figure 7

Scalar, SIMD and threaded Wasm artifacts were rebuilt from stable Rust sources and passed artifact validation.

`artifacts/level-set-volume/figure7-coarsening-four.json`: 26 healthy checkpoints through frame 25, with zero missing-phi, air-phi or low-density interior samples throughout. The sphere remains vertically distorted (variance about 78, 97, 78 at frame 25); this change does not claim to resolve the independent free-fall shape/timestep issue. The discarded unrestricted-eight-spacing experiment had interior dilution and is not the shipped material policy.

### Half pool impact: frame 11 and deep-liquid proof samples

The frame-11 transport failure was a donor containing only
`3.1114633254152634e-28` finest-cell volumes, on a mixed wet/air page.
Whole-page residue deletion could not remove it. The existing uncovered-donor
pass now deletes positively classified air volume at or below `gvRoundoff`
(approximately `9.54e-7` of cell capacity), updates destination density, marks
activity, and records the removed volume in the existing per-frame/cumulative
loss ledger. Meaningful, liquid-sign and unresolved donors retain the existing
missing-receiver failure. This adds no dispatch.

The sphere's blanket refinement also exposed a proof-input bug: valid deep
phase samples were rejected for lacking metric distance. The accepted and
restricted proof samplers now preserve those valid phase signs. At frame 11,
the four central sphere pages at brick coordinates `(3|4,3,3|4)` accept B4
(two finest-grid spacings per cell). Curved peripheral pages can still reject
coarsening through the feature-preservation proof; this change does not claim
that every sphere page reaches four-spacing cells.

A targeted Dawn run with both changes passed health checks through frame 63,
then encountered a separate `MISSING_COMPILED_TOPOLOGY_FACE` failure at frame
64 in `validateAndAuthorizeShadowTopology` (owner 380, operands 59/1/1/1).
That later failure remains unresolved by this frame-11 change. Focused Dawn
residue tests check preservation and that deletion is recorded only once.
