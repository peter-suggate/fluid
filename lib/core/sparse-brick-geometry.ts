/**
 * The brick, as a number every zone is allowed to know.
 *
 * A brick is the unit the sparse solvers actually bind on: the refinement gate
 * raises a region's floor for every brick the region *intersects* and lowers
 * its ceiling only for bricks it fully *contains*
 * (`sparse-cm12-refinement-regions.ts`, and `region_bounds` in
 * `rust/crates/fluid-core/src/resolution.rs` with `B = 8`). So a region edge
 * that falls inside a brick changes nothing for the ceiling and over-applies
 * the floor — which is why the editor snaps a region to this and not to the
 * finest cell.
 *
 * It lives in core because the editor is the consumer that needs it and the
 * module boundary forbids a `feature` reaching a method: `lib/features/*` may
 * import `lib/core`, so this is the one place the 8 can sit where the region
 * package, the 3-D method (`sparse-brick-atlas.ts`) and the 2-D lab's view
 * model (`lib/physics-wasm/advance-view.ts`) can all read the same number.
 * `tests/brick-fine-resolution.test.ts` holds the three equal.
 *
 * This is the *construction-time* brick width, not a runtime one: an atlas
 * built at a different `brickFineResolution` reports its own through
 * `atlas.brickFineResolution`, and geometry reads that. What this constant
 * fixes is the default the shipped ladders and the editor are authored against.
 */
export const BRICK_FINE_CELLS = 8;
