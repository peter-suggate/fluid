# Level-set-plus-volume transport: 2D advance-lab plan

Date: 2026-09-14. Third transport option beside `Baseline` and `CellwiseRemap`; both stay untouched.

## Intention

Carry two fields and one trace per frame, no substeps, no transport geometry:

- **V, the cell liquid volume, is authoritative for mass.** Transported by a conservative semi-Lagrangian gather (Chentanez and Müller 2012, Lentine 2011 doubly-stochastic weights): one backward trace per cell at any Courant, weights normalised so every donor gives exactly its volume. Cells may transiently exceed capacity; the excess drains through the pressure divergence term over frames, as in CM12.
- **φ, a cell-centred signed distance, is authoritative for surface shape.** Advected along the same backward trace by plain semi-Lagrangian sampling of the previous frame's RDF. A distance is scale-free, so coarse and fine neighbours agree at a 2:1 seam by construction; a volume fraction never can.
- **Reconciled per cell, smoothly.** The plane normal comes from ∇φ, the plane position from V clamped to capacity. Those planes feed the existing shared RDF, which becomes next frame's φ. The surface is drawn from φ, so over-capacity volume never appears as geometry. This removes the blobby surface without giving up conservation.

What is dropped relative to the remap: pre-image polygons, area correction, fold detection, band projection, certificates. What is kept from the lab: velocity extension, PLIC planes, the shared RDF, the pressure and resolution stages.

## Where it lives

Rust, in `fluid-core`, because the advance lab already runs the Rust world through wasm and the RDF, PLIC and velocity sampling it needs are all there. No TypeScript beyond exposing the option.

- `world.rs`: a third `TransportExperiment::LevelSetVolume` variant; `fluid-wasm/src/lib.rs` and `lib/physics-wasm/advance-controller.ts` pass the name through like the existing ones.
- New module `levelset_volume.rs` holding the whole step. State added to the world: one `phi: Vec<f32>` per cell, seeded from the initial PLIC RDF.
- Frame order for this mode: extend velocity (existing, 8 generations) → trace → V gather → φ gather → plane fit → RDF rebuild → resample φ. Pressure, resolution and presentation stages unchanged.

## Steps

1. **Trace.** One RK2 backward trace per cell centre through the extended face velocity, using the existing `sample_support` interpolant. Record the landing point and its owner cell. About a day including the option plumbing.
2. **V gather.** Build receiver-by-donor weights from the landing point (bilinear on the fine lattice mapped to owner cells), then three normalisation passes so rows and columns both sum correctly. Apply. Conservation to f32 roundoff is the first checkpoint: total V before and after must match. Report the count and maximum of V > C. About a day.
3. **φ gather.** Sample the previous frame's RDF fine-vertex lattice at the landing point. Resample onto cell centres. Half a day.
4. **Plane fit and RDF rebuild.** A variant of `reconstruct_interfaces` that takes the normal from ∇φ instead of from the density stencil, with the fraction from min(V, C)/C. Feed the planes to `reconstruct_shared_rdf`, then read φ back at cell centres. Rule for inconsistent cells: if V > 0 but φ places the cell more than half a width into air, the cell is sub-cell material, carried in V and not drawn. About a day.
5. **Over-capacity drain.** Add CM12's divergence term, min(λ(V/C − 1), η)/Δx with λ = 0.5, η = 1, to the pressure right-hand side for cells over capacity. Half a day. Skip on the first pass; measure how far over capacity cells go without it.

## Test

Same two lanes as the remap, same native harness, one arm each, dt = 1/30, zero substeps:

- `tools/verify-cm12-figure-7-native.ts --frames=30` and the coarse-first half-pool 10 frames, with `transportExperiment: "level-set-volume"`.
- Report per frame: total V drift, count and max of V > C and how fast it decays, wall per frame split into trace, gathers, plane fit, RDF, and pressure iterations.
- Seam check: across every 2:1 seam, the difference between the two neighbours' PLIC plane offsets evaluated at the seam midpoint. This is the number the whole direction exists to make small; record it for baseline on the same frames.
- Peter checks Figure 7's sheet and the half-pool surface in the advance-lab app himself. No browser gates.

Go/no-go after 30 frames of Figure 7: drift at roundoff, over-capacity bounded and decaying, seam offsets below baseline, wall per frame under baseline's. If the sheet dies of semi-Lagrangian smoothing at Courant 10, the answer is a sharper φ sample (cubic) before anything else.

## Not in scope

3D, GPU, rigid coupling, sources, moving solids. The remap's certificate list and performance plan stay in `sparse-geometric-remap-review-handoff.md` and are paused, not deleted.
