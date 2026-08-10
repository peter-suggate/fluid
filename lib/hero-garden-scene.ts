import { cloneScene, defaultScene, DEFAULT_FINEST_CELL_SIZE_M, type CameraState, type SceneDescription } from "./model";
import { heroGardenCloudTree } from "./hero-garden-tree";
import { applyHeroGardenNodeOverrides, HERO_GARDEN_OVERRIDES } from "./hero-garden-overrides";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import {
  pondVesselWaterline,
  type PondVesselSpec,
} from "./voxel-scenery/pond-vessel";
import { stoneSetBoulderStations, stoneSetSteppingBodies } from "./voxel-scenery/stone-set";
import { ROSETTE_AIR_PLANT, ROSETTE_GRASS_TUFT, type RosetteForm } from "./voxel-scenery/rosette";
import { sweptTubeNodes } from "./voxel-scenery/swept-tube";
import { tanHalfFovFor35mmFocalLength } from "./webgpu-camera";

/**
 * The hero scene: a porcelain pond with a hose filling it.
 *
 * Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`, and the
 * plan in `docs/HERO_GARDEN_HOSE_SCENE_PLAN.md`.
 *
 * This is the phase-0/1 body — the vessel, the water and the framing. The dry
 * set (pebble beds, mushroom-cap boulders, bonsai, air plants, hose geometry)
 * arrives in phase 2 and deliberately does not exist yet: the coping-as-terrain
 * bet is the one thing every later phase stands on, and it wants to be judged in
 * an empty frame rather than through a set that could flatter or hide it.
 *
 * Why the lattice is 25 mm, and why it wants to be 7.5 mm. The reference's two
 * most recognisable water features are the concentric ripple train and the
 * plunge column; the ripples read as ~4 % of the pond's width, so ~5 cm, and the
 * stream is ~4 cm across. At 25 mm both are within a cell or two of unresolvable
 * — this lattice can show that water is moving, not what shape the motion has.
 * 7.5 mm is where they land at ~6.7 and ~5 cells and the reference becomes
 * reachable rather than merely suggested.
 *
 * The scene therefore opens **dry**, and the water is opted into — see
 * `HeroGardenHoseOptions`. None of the walls below stand between opening this
 * scene and looking at it, because none of them are on the renderer's path.
 *
 * 25 mm is where the lattice sits because that is the finest measured to get this
 * scene through solver bring-up. Two walls were found above it, in this order:
 *
 *  - **7.5 mm** overruns a hard device limit. "Seed global fine bricks from
 *    every interface leaf" dispatches one workgroup per interface leaf on a
 *    single axis, and this scene asks for 65 536 against WebGPU's 65 535 ceiling.
 *    That is an engine shape, not a scene budget: no amount of art direction
 *    moves it, and a two-dimensional dispatch would.
 *  - **12.5 mm and 15 mm** refused the t = 0 pressure gate. **Fixed, and it was
 *    the engine, not the vessel.** The inner face was the wrong suspect: the
 *    SPGrid logical-page directory left every never-occupied page at its
 *    zero-fill, which is the legal index of physical page 0, so the coarse
 *    rows over this basin's empty regions resolved a stranger's page and the
 *    MGPCG rejected the publication (ERR_ROW stage 22). This pond has such
 *    rows and the tank scenes do not, which is the whole of why it was
 *    scene-specific. 25/15/12.5 mm all publish now.
 *
 * Width and depth are a whole number of 8-cell bricks at every spacing named
 * here (0.2 m at 25 mm), so the sparse domain has no partial brick at either
 * wall. **The height is not, at 25 mm — see the constant below.**
 */

/**
 * The container, which is also the voxel domain's floor and its cell size.
 *
 * `WebGPULiveSvoScene` derives its spacing as `container.<axis> / dimensions[i]`
 * over the counts `sceneLatticeDimensions` rounds out of these extents
 * (`lib/webgpu-live-svo-scene.ts`), so this box is what sets the voxel size on
 * every axis, and `planSparseSceneDomain` opens its lattice at `[0, dimensions]`
 * before it unions anything else in. Nothing above the lid is *cropped* by it —
 * see the correction below — but everything about the scene's resolution and the
 * solver's own grid is stated here.
 *
 * ---------------------------------------------------------------------------
 * 1.2 m, raised from 0.6, for replacement-tree headroom
 * ---------------------------------------------------------------------------
 * The replacement tree is expected to be roughly 0.80 m
 * standing on the same terrace, whose foot is at about y = 0.21 — so it reaches
 * **y ~= 1.01 m**, nearly twice the old lid. What that has to clear:
 *
 *  - `HERO_LAYOUT_CONTAINER_BOUNDS` is this box, and `tests/hero-layout.test.ts`
 *    asserts that nothing the *layout* places escapes it. That assertion is the
 *    right one to keep and a 1.01 m crown does not fit under a 0.6 m lid — and
 *    note which side of it the tree is on: the bonsai is an authored node rather
 *    than a layout placement, so it has never been checked against this bound at
 *    all, which is how its crown came to sit 103 mm over the lid unremarked.
 *  - The domain's height stops being an accident. Measured on the built document
 *    with `buildEnvironmentProxyCatalog`, the tallest primitive in the set is the
 *    bonsai's crown at y = 0.7035 — already 103 mm *over* a 0.6 m lid — and the
 *    planned domain came out 15 bricks tall (0.75 m) purely because that is where
 *    the crown ended. The extent of the tree was therefore a property of whatever
 *    prop happened to be tallest, and replacing that prop could re-shape it
 *    silently. At 1.2 m the container is the taller of the two and says so.
 *
 * What the added volume costs is **nothing**, and the reason is that all of it
 * is empty air *above* the set. A brick exists because something claims it, and
 * a dry world drops the container claim entirely (`octreeLiveSceneSolverClaim`,
 * `fluid-only` by default), so the render tree's pages follow the ground and the
 * props rather than the box. Measured GPU-free at the authored 6.25 mm, dry,
 * through `sceneLatticeDimensions` -> `planSparseSceneDomain` ->
 * `planAdaptiveSparseBrickOctree`, over the three candidate heights:
 *
 *     depth   height   lattice          bricks         claimed   nodes   leaves
 *     0       0.6 m    288 x  96 x 192  50 x 15 x 30     3,450   4,295    3,450
 *     0       1.1 m    288 x 176 x 192  50 x 22 x 30     3,450   4,295    3,450
 *     0       1.2 m    288 x 192 x 192  50 x 24 x 30     3,450   4,295    3,450
 *     3       0.6 m    288 x  96 x 192  50 x 15 x 30     3,450   4,295    3,450
 *     3       1.1 m    288 x 176 x 192  50 x 22 x 30     3,450   4,295    3,450
 *     3       1.2 m    288 x 192 x 192  50 x 24 x 30     3,450   4,295    3,450
 *
 * The claimed brick set is **identical** at every height and at both rungs, and
 * the plan's voxel count with it — the y lattice grows 96 -> 192 cells and the
 * domain 15 -> 24 bricks, and not one of the added bricks is claimed by
 * anything. The refinement ladder scales with *occupancy*, not with the bounding
 * box, and this is the case that separates the two. Everything downstream is a
 * function of the claimed set, so the pages do not move either. The world origin
 * stays at y = 0 and the node and leaf sets are the same objects, so the frame
 * should not move; the only difference a marcher sees is 600 mm more empty span
 * above the set before it exits.
 *
 * ---------------------------------------------------------------------------
 * The correction: this lid does not clip geometry, and it never did
 * ---------------------------------------------------------------------------
 * An earlier tree plan assumed geometry beyond the container would be clipped.
 * That is not what the planner does. `planSparseSceneDomain`
 * seeds its lattice at the container and then takes the **union** with every
 * proxy AABB, brick-aligned, so a prop that overhangs the box *extends* the
 * domain. Measured directly: adding one synthetic proxy reaching y = 1.01 to
 * the 0.6 m document takes the domain from 15 to 21 bricks on y (0.75 -> 1.05 m)
 * with the container untouched. So the raise is for the two reasons above, and
 * a 0.80 m tree under the old lid would have rendered — just into a domain whose
 * shape no one had authored.
 *
 * ---------------------------------------------------------------------------
 * Why 1.2 and not the 1.1 the replacement tree is expected to need
 * ---------------------------------------------------------------------------
 * The ladder of cell sizes at which every container dimension is a whole number
 * of 8-cell bricks is 25, 12.5, 6.25, 3.125 and 1.5625 mm, and the header above
 * states it as an invariant. 1.1 m holds all of them **except 25 mm**, where 44
 * cells is 5.5 bricks — and 25 mm is `HERO_GARDEN_SOLVER_CELL_M`, the lattice a
 * wet document is clamped to, so a solved pond would carry four cells of a brick
 * above its own ceiling. The visible consequence is small and real:
 * `FLUID_SVO_DRY_SMOKE_CELL_MM=25` and `=15` abort on the brick guard in
 * `tools/run-svo-dry-render-smoke.ts`.
 *
 * 1.2 m is exactly 6 bricks at 25 mm and whole at every finer rung, and it costs
 * nothing to prefer it: measured, the wet lane plans the *same* 324 solver bricks
 * and the same [13, 6, 8] domain at 1.1 and at 1.2, because the rounding has
 * already been paid. The extra 100 mm over what the tree needs is the price of
 * keeping a stated invariant true, which is cheaper than a footnote saying one
 * rung of the ladder no longer works.
 */
export const HERO_GARDEN_CONTAINER = { width_m: 1.8, height_m: 1.2, depth_m: 1.2 } as const;
/** The global lattice; the product's dry refinement presents it at 6.25 mm. */
export const HERO_GARDEN_CELL_M = DEFAULT_FINEST_CELL_SIZE_M;
export const HERO_GARDEN_BRICK_CELLS = 8 as const;

/**
 * The lattice a *solved* hero garden may not go below.
 *
 * The header above is the record: 7.5 mm overruns a one-workgroup-per-interface-
 * leaf dispatch at WebGPU's 65 535 ceiling, and 12.5 mm was the finest measured
 * to publish once the SPGrid page-directory bug was fixed. Held at 25 mm rather
 * than 12.5 because bring-up measured *publication*, not a settled pond, and
 * one path that reaches this is a user toggling water on a scene already open —
 * the wrong moment to discover a finer rung does not hold.
 *
 * The clamp lives with the constraint. `DEFAULT_FINEST_CELL_SIZE_M` is what the
 * picture wants; this is what the solver can carry, and only a document that
 * asks for fluid pays it.
 */
export const HERO_GARDEN_SOLVER_CELL_M = 0.025;

/**
 * How finely the vessel is baked, which is *not* the lattice the solver runs on.
 *
 * A quarter of a cell, so every solver column centre lands exactly on a grid
 * node: (k + 1/2) * 25 mm is 2 + 4k nodes out, and a bilinear fetch at a node is
 * the node. The ground the water meets is therefore the ground that was
 * generated, with no resampling error between them.
 *
 * The renderer sees it too, which it did not always. `TerrainGrid` was once a
 * CPU-and-solver feature — the dry scene's WGSL evaluated terrain from
 * `terrainMeta` plus eight analytic features with no grid sampler anywhere, so
 * a sculpted vessel drew as a flat plane at `baseHeight_m` and this scene could
 * not be looked at at all. `packSvoDrySceneTerrainHeightfield` is the producer
 * the `terrainHeightfield` primitive kind was waiting for, and the ray now hits
 * the surface that was generated here. Sample spacing is therefore a *visual*
 * decision as well as a solver one.
 *
 * ---------------------------------------------------------------------------
 * The alignment argument, generalized — and it is the whole of the rule
 * ---------------------------------------------------------------------------
 * "A quarter of a cell" was never the point; *landing on a node* was. The live
 * voxeliser takes one height per finest column at the column **centre**
 * (`terrainColumnHeightsForLattice`), so with grid spacing `h` and a render cell
 * `c` the sample lands `(k + 1/2) * c / h` nodes out — an integer exactly when
 * `c / h` is even, and worst-case halfway between four nodes when it is one.
 * Measured on this vessel, against the analytic `pondVesselHeightAt`, as the
 * greatest error at any column of the render lattice:
 *
 *     grid vs render cell     worst column error
 *     h = c / 2               0.000 mm      (every centre is a node)
 *     h = c                   0.73 voxels   (every centre is a patch centre)
 *     h = 2c                  1.15 voxels   (0.04 % of columns off by one)
 *     h = 4c                  2.77 voxels   (0.21 % off by one, 0.05 % by two)
 *
 * The last row is what a 6.25 mm bake looks like under a 1.5625 mm picture: a
 * fifth of a percent of the ground is a whole voxel — a visible ledge — away
 * from the vessel that was authored, and the ledges follow the grid rather than
 * the pond. So the spacing follows the *detail* lattice at half its pitch.
 *
 * ---------------------------------------------------------------------------
 * Why there is no clamp here any more, and what used to happen instead
 * ---------------------------------------------------------------------------
 * The rule above is `h = c / 2` and it is the only rule; the `min` is a floor,
 * not a competing objective. It exists because the *solver's* lattice is 25 mm
 * (`HERO_GARDEN_SOLVER_CELL_M`) while the picture's is 6.25 mm, and the
 * alignment argument alone would let a solved document bake at 12.5 mm —
 * coarser than this scene has ever baked, and coarser than the clearances in
 * `tests/pond-vessel.test.ts` and `tests/swept-coping.test.ts` were measured
 * against. So the spacing is half the *finer* of the two lattices, which is
 * 3.125 mm at every rung at or above the authored one and follows the detail
 * cell down from there.
 *
 * What this function used to do was ask for `min(6.25/4, detail/2)` mm and then
 * coarsen by powers of two until the bake fitted `MAX_TERRAIN_GRID_SAMPLES`.
 * That clamp **bound at the authored default** — 1.5625 mm is 1153x769, 3.4x
 * over the 262 144 a document may carry — so it could never stop binding, and
 * the ground came out at 3.125 mm at a 25 mm cell and at a 0.78 mm cell alike.
 * Worst error against the analytic vessel, at the render lattice's own columns:
 * 0.00 leaves at 6.25 mm, 0.73 at 3.125, 1.23 at 1.5625, **2.75 at 0.78** —
 * smaller boxes wrapped around coarser geometry, and 37 148 `terrain-partial`
 * coverage holes at the first refinement rung.
 *
 * The clamp is gone because the grid is gone. `scene.terrain` now carries a
 * `TerrainProcedural` description — the vessel spec, the container and this
 * spacing, about 700 bytes — and `terrainSampleGrid` derives the heightfield on
 * demand at exactly this pitch. Nothing serializes it, nothing structured-clones
 * it, and the only budget left is host memory
 * (`MAX_TERRAIN_DERIVED_GRID_SAMPLES`), which this rule does not approach until
 * a 0.78 mm leaf. The document went from 4.34 MB to 9.6 KB and its clone from
 * 18.8 ms to 0.07 ms.
 */
export function heroGardenTerrainSample_m(
  detailCellSize_m: number = HERO_GARDEN_CELL_M,
): number {
  if (!(detailCellSize_m > 0) || !Number.isFinite(detailCellSize_m)) {
    throw new RangeError("Hero garden terrain sample needs a positive finite detail cell size");
  }
  return Math.min(HERO_GARDEN_CELL_M, detailCellSize_m) / 2;
}

/** The bake this scene has always shipped: `heroGardenTerrainSample_m()` at the authored lattice. */
export const HERO_GARDEN_TERRAIN_SAMPLE_M = heroGardenTerrainSample_m();

/**
 * How far the still-water level sits below the plaster *outside* the pond.
 *
 * This is the number that keeps the water in the basin. A tank fill wets every
 * column the level clears, so a waterline above the outer ground floods the
 * whole 1.8 x 1.2 m floor with a two-cell film — which is not what the reference
 * shows, and which the sparse solver refuses outright: it publishes no
 * liquid-row frontier for a domain that is all film and no body.
 *
 * Expressed in cells rather than millimetres, because that is what the property
 * is actually about: the fill must not reach the outer ground *through rounding*
 * either, so the clearance has to survive the lattice it is sampled on. Under
 * one cell it does not, whatever the metre value says.
 *
 * The lattice it is sampled on is the **solver's**, which is the one thing here
 * that does not follow the picture. A dry render at a finer cell has no fill, no
 * sampler and no rounding to survive; what it does have is a composition — the
 * shore, the bedded stones, the wading path and the plunge point are all solved
 * against this waterline — and re-solving that between two rungs of a resolution
 * ladder would mean the finer frame could not be compared with the coarser one.
 *
 * **So the default is `HERO_GARDEN_SOLVER_CELL_M` and not `HERO_GARDEN_CELL_M`,
 * and the difference between those two is a bug this signature has already
 * had.** They were the same 25 mm when this rule was written; the day the
 * picture's lattice went to `DEFAULT_FINEST_CELL_SIZE_M` the default argument
 * silently became 6.25 mm, the clearance halved to its 20 mm floor, and
 * `HERO_GARDEN_WATERLINE_M` — a module constant that `HERO_STONE_SET`,
 * `heroGardenLayoutWorld`, `HERO_GARDEN_HOSE_IMPACT_M` and every bedded station
 * are composed against — rose 20 mm under the set standing on it. What that
 * cost, measured: 56 % of the wading path's footing came out of the bed against
 * a 50 % bound, the coping stopped holding the still level by a cell, and a
 * document built with `water: true` no longer matched the dry one it is
 * supposed to be, which `withHeroLayout` refuses outright.
 *
 * A fill is only ever sampled on the solver's lattice, so that is the lattice
 * this answers to. 20 mm is the floor either way: past 12.5 mm cells the
 * absolute clearance binds and this stops moving at all.
 */
export function heroGardenWaterBelowGround_m(cellSize_m: number = HERO_GARDEN_SOLVER_CELL_M): number {
  if (!(cellSize_m > 0) || !Number.isFinite(cellSize_m)) {
    throw new RangeError("Hero garden water clearance needs a positive finite cell size");
  }
  return Math.max(0.02, 1.6 * cellSize_m);
}

/** The clearance at the solver's lattice: 40 mm, and 1.6 cells of it. */
export const HERO_GARDEN_WATER_BELOW_GROUND_M = heroGardenWaterBelowGround_m();

/**
 * The vessel.
 *
 * `radius_m` is the coping's crest centreline, not the waterline: the water
 * meets the rim partway down its inner flank, so the wetted plan comes out a
 * little inside these numbers. Seven lobes at an 8 % wobble is enough wander
 * that no two quadrants of the rim repeat, and little enough that the outline
 * still reads as one deliberate curve rather than as a puddle.
 *
 * `HERO_GARDEN_OVERRIDES.vessel` replaces the whole of it when the shape lab has
 * left one there — see `lib/hero-garden-overrides.ts`. It is applied here rather
 * than at any of the call sites because this constant is the *only* place the
 * pond is stated: the terrain, the waterline, the stone set and the layout all
 * derive from it, and a vessel swapped anywhere downstream would move the ground
 * out from under props already standing on it.
 */
const HERO_GARDEN_VESSEL_AUTHORED = Object.freeze({
  center_m: [0, 0] as const,
  radius_m: [0.52, 0.38] as const,
  groundHeight_m: 0.30,
  basinDepth_m: 0.155,
  rimHeight_m: 0.055,
  /**
   * Half the coping band: the rim runs from `-rimHalfWidth_m` to `+rimHalfWidth_m`
   * of signed plan distance, so this is a **60 mm** band.
   *
   * It was 64.4 mm — a 129 mm band — inherited from the swept bullnose's ground
   * footprint so that every bed, stone and disc offsetting from it would stay
   * where it was placed. Keeping a number for continuity is a good reason to keep
   * it and no reason at all for it to be right, and this one was not: measured
   * against the plate at the front of the pond, ours took **5.5 %** of the pond's
   * screen width where the plate's takes **2.45 %**.
   *
   * That comparison is only legitimate between two cameras of similar pitch, and
   * these are: the plate's stepping discs are circles in plan, so their screen
   * aspect gives its pitch directly at about 25 degrees, against this camera's
   * 22.9. Ours foreshortens depth slightly *more*, which shortens a plan band on
   * screen — so 2.2x is if anything an underestimate of how much wider the rim
   * was.
   *
   * The band still has to hold a section. `wallProfile_m` needs 11.9 mm of run
   * either side at this rise, and `sectionWidthVariation` swings the half-width
   * to 23.4 mm at its narrowest, which leaves 11.5 mm of crown there and 36 mm at
   * the mean — a formed lip rather than a shelf, which is what the plate has.
   */
  rimHalfWidth_m: 0.030,
  /**
   * The crest is a swept solid now, not a swelling of the ground.
   *
   * The reason is the *meeting*, not the crown. `pondVesselCrestProfile` is
   * tangent at its foot by construction, so a heightfield rim leaves the plaster
   * with a continuous normal — measured at 0.24 degrees from vertical at the
   * foot — and there is simply no line where the two meet. The reference has a
   * hard one, and a union of a solid with the ground gives it for free, along
   * with the roll-back under the lip that no function of (x, z) can express.
   *
   * What this does *not* change is containment, which is the property that made
   * the vessel terrain in the first place: the still waterline sits 40.9 mm below
   * the lowest plaster outside the rim, and omitting the crest moves that number
   * by less than a nanometre. The wall the water rests against was always the
   * outer ground, never the crest — a coping is by definition the part above the
   * waterline. That is pinned by a test rather than left as an argument.
   */
  /**
   * A **flat-topped coping**, in the heightfield, replacing the swept bullnose
   * solid — see `PondVesselSpec.crestWall`.
   *
   * The solid had to go because it could not be anything but a tube:
   * `sweptCopingNodes` sweeps round cones, so the section is a circular arc and
   * `widthToHeight` only picks how much of the circle shows. At 2.8 the rim
   * showed 142 degrees of a circle and read, correctly, as a pipe laid round the
   * pond. No setting of it is flat.
   */
  crest: "wall",
  // 4 mm and 3 mm on a 55 mm rise: the arcs take 8 % of the rise, the same
  // proportion the raised beds use, which is what separates a formed edge from a
  // rolled one. Measured by `tools/shape-lab.ts wall`.
  crestWall: { crestRadius_m: 0.004, footRadius_m: 0.003, batter_rad: 0.10 },
  // The inner face is a wall, not a beach. At 0.16 m of run for 0.155 m of drop
  // the basin was a 45-degree dish, and a third of the pond's 1.04 m width was
  // spent on the slope — which is why the first renders read as a crater in
  // plaster rather than as a vessel with sides. The reference drops almost
  // vertically from the coping's inner foot: the water meets the rim and the
  // wall goes straight down behind it. Three centimetres of run is as near to
  // vertical as a heightfield gets before the bilinear sample spacing, not the
  // authored profile, is what sets the slope.
  innerFace_m: 0.035,
  // …except on the left, where the reference has a shore. The stepping discs
  // wade in over this sector and the pebble courses run down into the water on
  // it; everywhere else the pond stays a wall. Centred on the bearing of the
  // near-left terrace below, so the boulder group, the beach and the disc path
  // are three consequences of one decision about where the near bank is rather
  // than three separately placed things that have to be kept in agreement.
  beach: { turn: 0.452, width: 0.17, innerFace_m: 0.30 },
  lobes: 7,
  wobble: 0.08,
  // The coping swells and narrows as it runs, which is most of what separates a
  // formed rim from an extruded one.
  sectionHeightVariation: 0.16,
  sectionWidthVariation: 0.22,
  relief_m: 0.0025,
  seed: 0x9a7de11,
  // The references place every prop on one uninterrupted exterior plane. The
  // pond profile still owns its wall, lip and beach; beyond the rim foot there
  // are no raised beds, mounds or hidden composition terraces.
  terraces: [],
});

export const HERO_GARDEN_VESSEL: PondVesselSpec = Object.freeze(
  HERO_GARDEN_OVERRIDES.vessel ?? HERO_GARDEN_VESSEL_AUTHORED,
);

/**
 * Still water, `heroGardenWaterBelowGround_m(cellSize_m)` under the outer ground.
 *
 * The set's one datum: the shore, the bedded stones, the wading path's freeboard
 * and the plunge point are all solved against it, so it is derived once and
 * passed rather than recomputed anywhere.
 *
 * The default is the solver's lattice, so this is one level for the whole scene
 * rather than one per rung — see `heroGardenWaterBelowGround_m`. The parameter
 * is kept because the clearance rule is genuinely a function of a lattice and a
 * caller that solves at a coarser cell than `HERO_GARDEN_SOLVER_CELL_M` needs a
 * deeper pond; what it must not do is hand the *picture's* cell to a rule about
 * a fill.
 */
export function heroGardenWaterline_m(cellSize_m: number = HERO_GARDEN_SOLVER_CELL_M): number {
  return pondVesselWaterline(HERO_GARDEN_VESSEL, heroGardenWaterBelowGround_m(cellSize_m));
}

/** The level at the solver's lattice. `withHeroLayout` is solved against this one. */
export const HERO_GARDEN_WATERLINE_M = heroGardenWaterline_m();

/**
 * Close, and looking down at about twenty-five degrees, so the pond fills the
 * frame the way it does in the reference.
 *
 * The first framing sat at 0.30 rad and 1.62 m, which put the eye 78 cm up and
 * more than a metre back: from there the basin was a slot seen edge-on, the
 * coping's far arc lay flat against its own near arc, and a third of the image
 * was horizon. None of that is in the reference, which is a close
 * three-quarter view with no horizon at all — the pond spans nearly the whole
 * width and the ground behind it is soft and out of focus.
 */
/**
 * Where the camera stands, and — because the light is derived from it — where
 * the sun is.
 *
 * Solved against the set rather than chosen by eye. `cameraPosition` puts the
 * eye at `target + d(sin az, ·, cos az)`, so screen right is `(cos az, -sin az)`
 * and depth is `-(x sin az + z cos az)`. Requiring the bonsai and the hose to
 * land far-right while the boulder group, the shore and the stepping stones stay
 * near-left — which is the reference's whole arrangement — is four inequalities
 * in one unknown, and they hold together only between about 5.1 and 5.7.
 *
 * The middle of that window is the part that matters: it means the framing
 * survives the set being moved rather than balancing on one number. An earlier
 * pass put this at 2.4 by deriving screen right as `(sin az, -cos az)` — the
 * other common convention, and a half-turn out — which stood the tree between
 * the camera and its own pond.
 */
export const HERO_GARDEN_AZIMUTH_RAD = 5.4;

/** Screen right in the XZ plane at that azimuth: `(cos az, -sin az)`. */
const HERO_GARDEN_SCREEN_RIGHT = [Math.cos(HERO_GARDEN_AZIMUTH_RAD), -Math.sin(HERO_GARDEN_AZIMUTH_RAD)] as const;
/** Screen right is retained as the zero of the key bearing below; nothing else reads it. */
void HERO_GARDEN_SCREEN_RIGHT;

/**
 * The lens: a 50 mm, which on the 36 x 24 mm frame is a vertical half-tangent
 * of 0.24 — 27 degrees vertical and 46 horizontal at the studio's 16:9.
 *
 * The renderer's default is 0.72: 71 degrees vertical, 104 horizontal, an
 * ultra-wide by any reckoning and about a 17 mm. The reference is not that. It
 * shows a pond a metre across with the far coping only a little smaller than
 * the near one, which is a normal lens at a couple of metres and nothing else.
 *
 * Fifty rather than something longer because the reference does carry visible
 * perspective — the basin is read as a bowl, not as a disc — and because a
 * longer lens would put the eye far enough back that the set's own terraces
 * start occluding the shore.
 */
const HERO_GARDEN_TAN_HALF_FOV = tanHalfFovFor35mmFocalLength(50);

/**
 * Where the camera stands.
 *
 * Solved against the set, not against the pond. Three quantities move together
 * as the distance changes, and only one of them is a free choice:
 *
 * - The pond's outer plan is 1.10 x 0.82 m, so at this azimuth its silhouette
 *   is 0.473 m of half-extent across screen right. At 1.40 m it reaches both
 *   side crops instead of sitting inside a field of empty plaster.
 * - The bonsai's crown is the highest thing in the set, and it is deliberately
 *   cut by the top edge. The reference is a crop, not a catalog
 *   view; keeping every generated floret inside the picture made the subject
 *   too small.
 * - The near-left boulder terrace approaches the left edge while the near
 *   coping approaches the bottom. Those two near cuts are what make the pond
 *   read as a place the camera is standing over rather than a miniature placed
 *   at the centre of a ground plane.
 *
 * What the longer lens buys is controlled perspective. Closing from 2.70 m to
 * 1.40 m spends the safety margin the old framing reserved around the set; the
 * 50 mm aperture keeps the near coping from acquiring an ultra-wide bow.
 *
 * ---------------------------------------------------------------------------
 * An automated solve was tried here and is deliberately not kept
 * ---------------------------------------------------------------------------
 * `tools/solve-hero-camera.ts` walks the six camera axes against an edge
 * alignment objective and lands on 1.667 m / 0.50 rad, improving that objective
 * from 0.194 to 0.269. **Do not take it.** Put its frame beside the plate and
 * the reason is immediate: pulling back to 1.667 m puts the whole crown inside
 * the frame with headroom above it, and the plate's crown is *cut by the top
 * edge*. The solve bought silhouette overlap and paid for it with the
 * composition — the one thing about this framing that was never in doubt.
 *
 * That is a general lesson about this scene and not a bug in the tool. The
 * frame and the plate differ structurally — no water, smooth forms, a smaller
 * canopy — so a scalar that scores how much of one lies on top of the other is
 * dominated by those differences, and its optimum is wherever they happen to
 * overlap best rather than wherever the camera belongs. **The camera is judged
 * against the plate by eye.** The solver stays because it is useful for
 * bracketing a search once the set actually resembles the plate; its output is
 * a candidate, never an answer.
 *
 * 1.25 m is the standing candidate if this is revisited: it holds the crop and
 * fills the frame more nearly the way the plate does.
 */
export const heroGardenCamera: Partial<CameraState> = {
  azimuth_rad: HERO_GARDEN_AZIMUTH_RAD,
  elevation_rad: 0.44,
  distance_m: 1.90,
  tanHalfFov: HERO_GARDEN_TAN_HALF_FOV,
  target_m: { x: -0.04, y: 0.55, z: 0.0 },
};

/**
 * Where the water leaves the hose, and how fast.
 *
 * One mouth and one aim, shared by the jet the solver injects and the tube the
 * renderer draws. They are the same three numbers on purpose: a hose whose
 * nozzle is anywhere but where the water actually appears is the single most
 * obvious way this scene could look wrong, and deriving both from here means it
 * cannot happen by drift.
 */
/**
 * 335 mm, which is 78 mm of air over the still water rather than 142.
 *
 * The reference's nozzle is *close* to the surface — the falling column is about
 * a tenth of the pond's width long, and that shortness is most of what makes the
 * pour read as a hose someone is holding rather than as a spout built into the
 * garden. At 400 mm the column was nearly twice the pond's own still depth.
 */
export const HERO_GARDEN_HOSE_MOUTH_M = Object.freeze({ x: 0.27, y: 0.335, z: 0.20 });
/** The visible plunge point, so nozzle placement and jet composition share one authority. */
/**
 * Tucked in under the mouth, which is what puts the aim at 40 degrees below
 * horizontal instead of the 33 the old pair described.
 *
 * The aim is derived from these two points and nothing else, so the plunge point
 * and the last hand's breadth of tube are one decision. The reference's nozzle
 * points steeply down and its stream is nearly vertical; a shallow aim throws the
 * column out across the pond and turns the fill into a jet.
 */
export const HERO_GARDEN_HOSE_IMPACT_M = Object.freeze({ x: 0.225, y: HERO_GARDEN_WATERLINE_M, z: 0.12 });
const HOSE_AIM_LENGTH = Math.hypot(
  HERO_GARDEN_HOSE_IMPACT_M.x - HERO_GARDEN_HOSE_MOUTH_M.x,
  HERO_GARDEN_HOSE_IMPACT_M.y - HERO_GARDEN_HOSE_MOUTH_M.y,
  HERO_GARDEN_HOSE_IMPACT_M.z - HERO_GARDEN_HOSE_MOUTH_M.z,
);
const HOSE_AIM = Object.freeze({
  x: (HERO_GARDEN_HOSE_IMPACT_M.x - HERO_GARDEN_HOSE_MOUTH_M.x) / HOSE_AIM_LENGTH,
  y: (HERO_GARDEN_HOSE_IMPACT_M.y - HERO_GARDEN_HOSE_MOUTH_M.y) / HOSE_AIM_LENGTH,
  z: (HERO_GARDEN_HOSE_IMPACT_M.z - HERO_GARDEN_HOSE_MOUTH_M.z) / HOSE_AIM_LENGTH,
});
const HOSE_SPEED_M_S = 1.19;
/**
 * 24 mm, read off the reference against the pond's own width: the tube covers a
 * little over two per cent of it.
 */
const HOSE_BORE_M = 0.012;

/**
 * The hose's run, from the nozzle backwards, in world metres.
 *
 * The first point off the mouth is placed along the aim, so the last hand's
 * breadth of tube points where the water goes. The rest climbs the raised bed's
 * face, lies along its top, and leaves frame to the right.
 *
 * Authored as the polyline it is rather than as placed beads: a capsule already
 * takes the two ends of the run it follows, which is what a hose is described
 * by. When phase 2 brings in the `swept-tube` generator this becomes its input
 * unchanged.
 *
 * **It rests on the bed rather than floating over it.** The back run used to sit
 * at a flat 460-470 mm, which is 85 mm of clear air above the bonsai terrace's
 * 375 mm top — a hose suspended over a garden by nothing. The reference's hose
 * is unambiguously *lying on* the raised bed and draped over its edge, and that
 * contact is what makes the tube read as heavy. So the back stations sit a bore
 * radius over the terrace top, and the two between them take the run down the
 * bed's face on the same curve the water takes off the nozzle.
 *
 * The extra station is the curve. Four records over six points bent the run in
 * two visible places; the hose in the reference is one continuous arc, and a
 * `swept-tube` fits its envelopes to whatever it is given — so the fix is
 * stations, not a different primitive.
 */
const HOSE_PATH_M: readonly (readonly [number, number, number])[] = Object.freeze([
  [HERO_GARDEN_HOSE_MOUTH_M.x, HERO_GARDEN_HOSE_MOUTH_M.y, HERO_GARDEN_HOSE_MOUTH_M.z],
  [HERO_GARDEN_HOSE_MOUTH_M.x + HOSE_AIM.x * -0.075, HERO_GARDEN_HOSE_MOUTH_M.y + HOSE_AIM.y * -0.075, HERO_GARDEN_HOSE_MOUTH_M.z + HOSE_AIM.z * -0.075],
  // Over the rim and up the bed's face.
  [0.365, 0.372, 0.255],
  [0.455, 0.398, 0.305],
  // Along the bed's top, a bore radius clear of it.
  [0.575, 0.389, 0.365],
  [0.720, 0.388, 0.440],
  [0.890, 0.388, 0.520],
  [1.060, 0.390, 0.600],
]);

/**
 * The set, which for now is the ground and the hose.
 *
 * A terrain shell publishes no boxes — the heightfield it stands for is already
 * the authority — so a pond whose every surface is terrain publishes nothing at
 * all, and the SVO's candidate index requires a scene to contain at least one
 * primitive. The hose is the right thing to arrive first regardless: it is the
 * scene's subject, and it is what makes the jet legible as something a person
 * is doing rather than water appearing out of the air.
 *
 * The hose is also the one deliberate departure from the house monochrome rule.
 * Every other surface in this set reads its form from shading alone; the
 * reference's dark slate-teal tube is the single object carrying hue, and it is
 * carrying it on purpose — the same licence the sets already grant a lantern
 * ember, spent on the one object here that is manufactured rather than grown.
 */
const HOSE_MATERIAL = { colorLinear: [0.045, 0.085, 0.082] as const, surface: "organic" as const };
const FERRULE_MATERIAL = { colorLinear: [0.29, 0.275, 0.235] as const, surface: "brushed-metal" as const };

function hoseNodes(): SceneryNode[] {
  // One swept-tube record family over the path as authored, rather than a
  // capsule per segment: the reference's hose is a continuous curve and a chain
  // of capsules creases its shading normal at every bend. `sweptTubeNodes`
  // fits the envelopes, so the only thing decided here is the path and the bore.
  const nodes: SceneryNode[] = sweptTubeNodes({
    id: "hose/tube",
    group: "hose-tube",
    tags: ["hose"],
    stations: HOSE_PATH_M.map(([x, y, z]) => ({ at_m: { x, y, z }, radius_m: HOSE_BORE_M })),
    material: HOSE_MATERIAL,
  });
  // The collar, as a short fat capsule rather than a cone, so it is authored by
  // the run it sits on and needs no orientation solved for it.
  //
  // 1.6 times the bore and a little longer than it was. At 1.3 the collar was
  // barely proud of the tube it clamps and read as a bright patch on the hose
  // rather than as a fitting; the reference's is unmistakably a *step*, and the
  // step is the whole of why the nozzle end is legible against a white pond.
  nodes.push({
    kind: "capsule",
    id: "hose/ferrule",
    group: "metal-ferrule",
    tags: ["hose", "ferrule"],
    place: { units: "metres" },
    from: HERO_GARDEN_HOSE_MOUTH_M,
    to: {
      x: HERO_GARDEN_HOSE_MOUTH_M.x - HOSE_AIM.x * 0.042,
      y: HERO_GARDEN_HOSE_MOUTH_M.y - HOSE_AIM.y * 0.042,
      z: HERO_GARDEN_HOSE_MOUTH_M.z - HOSE_AIM.z * 0.042,
    },
    radius: 1.6 * HOSE_BORE_M,
    material: FERRULE_MATERIAL,
  });
  return nodes;
}

/**
 * The seed the whole dry set is grown from.
 *
 * One number, because the point of a generated set is that "give me another
 * garden" is a re-seed rather than a re-author. Changing it moves every stone
 * and re-packs every bed; it does not move the pond, which has a seed of its
 * own so that the vessel can be pinned while the set is still being explored.
 */
export const HERO_GARDEN_SET_SEED = 0x5701_e5;

/**
 * The set, as a description of itself.
 *
 * This used to be a *composition step*: a function that took a ground query,
 * called `sweptCopingNodes`, `stoneSet` and `bonsaiNodes`, and splatted their
 * output into the node list. What the document then held was 684 baked nodes and
 * 884 kB of ellipsoid centres — the seeds and the parameters gone, "give me
 * another garden" a re-run of this factory that discards every edit the user had
 * made, and every `cloneScene` a copy of the lot. The three nodes below say the
 * same thing in 1.7 kB and publish the same 2 197 primitives in the same order.
 * See lib/scenery-generators.ts.
 *
 * The vessel is named rather than repeated. All three generators lay their runs
 * against the same plan curve — that is why moving a control point carries the
 * rim, the beds and the disc path with it — and a curve that lived on each node
 * would be three copies with three chances to disagree.
 *
 * No ground query is passed. The species still need one and still get one; it
 * arrives from the expansion context, which is the scene's own baked grid, so a
 * root or a pebble is still seated against exactly the surface the renderer
 * draws and the solver collides with.
 */

/**
 * Where the boulder-family plant stands: derived, not authored.
 *
 * The near plant belongs *to the boulder group* — it is the tuft that has taken
 * hold in the gravel heaped at their feet, and that reading is the whole reason
 * it carries a wider `bed_m` than the other two. So it is solved from the
 * family's own stations rather than given a world coordinate beside them.
 *
 * That is not tidiness. This constant used to hold `[-0.60, -0.26]`, and when
 * the stone set re-solved the family's position against the camera the plant
 * stayed behind and stood alone on an empty bank — a coordinate restating
 * another module's answer is a copy that drifts the first time the original
 * moves. `stoneSetSteppingStations` is read the same way by the layout for the
 * same reason.
 *
 * It stands off the outboard end of the family rather than among it, clear by
 * half the outermost stone's own width, so a three-millimetre rosette is never
 * inside a boulder it cannot be seen against.
 */
const HERO_BOULDER_PLANT_CLEARANCE = 0.5;
function heroBoulderPlantStand(waterline_m: number): readonly [number, number] {
  const family = stoneSetBoulderStations({
    vessel: HERO_GARDEN_VESSEL,
    waterline_m,
    seed: HERO_GARDEN_SET_SEED,
  });
  // The outboard end is the station furthest from the pond's centre in x, which
  // is the direction the family runs; standing beyond it keeps the plant on the
  // open bank instead of between two stones.
  const outboard = family.reduce((a, b) => (a.at_m[0] <= b.at_m[0] ? a : b));
  return [
    outboard.at_m[0] - HERO_BOULDER_PLANT_CLEARANCE * outboard.width_m,
    outboard.at_m[1] + HERO_BOULDER_PLANT_CLEARANCE * outboard.width_m,
  ];
}

/**
 * Where the plants stand.
 *
 * All three sit *off* the coping rather than on it: a rosette's blades are
 * three millimetres across and the rim is the one surface in this scene the eye
 * follows as a continuous line, so a tuft on the crest reads as damage to the
 * casting. `bed_m` is the little of the ground each one has disturbed under
 * itself, and the boulder-group plant has more of it because it is bedded into
 * that group's own gravel.
 */
interface HeroPlantStand {
  readonly id: string;
  readonly form: RosetteForm;
  readonly at_m: readonly [number, number];
  readonly bed_m?: number;
  readonly salt: number;
}
function heroPlantStands(waterline_m: number): readonly HeroPlantStand[] {
  return [
    { id: "plant/boulder-group", form: ROSETTE_AIR_PLANT, at_m: heroBoulderPlantStand(waterline_m), bed_m: 0.035, salt: 0x00a1_7e01 },
    { id: "plant/terrace-back", form: ROSETTE_AIR_PLANT, at_m: [0.44, 0.42] as const, bed_m: 0.028, salt: 0x00a1_7e02 },
    { id: "plant/rim-right", form: ROSETTE_GRASS_TUFT, at_m: [0.72, -0.32] as const, bed_m: 0.024, salt: 0x00a1_7e03 },
  ];
}

/**
 * The level is a parameter rather than a module constant because it moves with
 * the *solver's* lattice — see `heroGardenWaterBelowGround_m`. Everything in the
 * graph that knows where the shore is derives from this one number, so a scene
 * built at another cell size cannot end up with stones bedded against a
 * waterline the document does not have.
 */
function heroGardenScenery(waterline_m: number): SceneryGraph {
  const graph = heroGardenAuthoredScenery(waterline_m);
  // The shape lab's node overrides, applied where the graph is assembled rather
  // than inside any one species. `withHeroLayout` appends to this and applies
  // them again over what it added, so an authored node and a laid-out one are
  // replaced by the same rule.
  return { ...graph, nodes: [...applyHeroGardenNodeOverrides(graph.nodes)] };
}

function heroGardenAuthoredScenery(waterline_m: number): SceneryGraph {
  return {
    palettes: {
      clay: { tint: [1, 0.985, 0.955] },
      stone: { tint: [0.972, 0.984, 1] },
    },
    vessels: { pond: HERO_GARDEN_VESSEL },
    nodes: [
      /**
       * Porcelain, not lawn — and it is the whole set that is porcelain, not
       * only the ground.
       *
       * The immediate reason is the ground: the vessel and the plaster it is set
       * into are one fired surface in the reference, and the garden closure would
       * band them by height into liner, soil and grass and speckle the lot with
       * daisies. But this node is also the scene's only declaration of what it is
       * *made of*, so `svoMaterialFunctionIdForEnvironmentProxy` reads it for
       * every prop as well — see `lib/svo-material-abi.ts`. A boulder here is not
       * granite that happens to be pale; it is the same fired white clay as the
       * rim it is bedded against, and it would be granite by default only because
       * its group carries the word "stone". One word decides for the whole set,
       * which is the only way a monochrome set can be kept monochrome as
       * generators come and go.
       */
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
      ...hoseNodes(),
      // No coping node. The rim is the vessel's own `crest: "wall"` section
      // now — a flat top with rolled edges, which a swept chain of round cones
      // cannot express at any setting of its width.
      // Boulders, beds and the wading path, all placed in the vessel's own
      // coordinates — a fraction of a turn round the plan curve and a plan
      // distance either side of it — rather than in world metres that could
      // drift from the coping they hug.
      {
        kind: "generator", id: "stone", generator: "pond-stone-set", vessel: "pond",
        seed: HERO_GARDEN_SET_SEED,
        params: { waterline_m },
      },
      heroGardenCloudTree(),
      // The plants. Built and tested for this scene and then never called by
      // it, which is the failure mode a generator catalog exists to make
      // visible: a species with no node in any document is a species nobody can
      // tell is missing.
      //
      // Two air plants where the reference has them — one wedged against the
      // boulder group on the near-left terrace, one on the back-right plateau
      // behind the tree's stand — and one grass tuft at the near-right rim,
      // which is the only place in frame with a run of plaster wide enough to
      // stand something on without crowding the coping.
      ...heroPlantStands(waterline_m).map(({ id, form, at_m, bed_m, salt }): SceneryNode => ({
        kind: "generator", id, generator: "rosette",
        seed: HERO_GARDEN_SET_SEED ^ salt,
        params: { ...form, at_m, ...(bed_m === undefined ? {} : { bed_m }) },
      })),
    ],
  };
}

/** The hose, as the solver sees it: a round jet leaving the nozzle along its aim. */
function heroGardenInflow(): SceneDescription["fluid"]["inflow"] {
  return {
    center_m: { ...HERO_GARDEN_HOSE_MOUTH_M },
    radius_m: 0.021,
    length_m: 0.05,
    velocity_m_s: {
      x: HOSE_AIM.x * HOSE_SPEED_M_S,
      y: HOSE_AIM.y * HOSE_SPEED_M_S,
      z: HOSE_AIM.z * HOSE_SPEED_M_S,
    },
    start_s: 0,
    end_s: 120,
    ramp_s: 0.35,
  };
}

export interface HeroGardenHoseOptions {
  /**
   * Whether the fluid solver owns this document. Off by default.
   *
   * The set is the part of this scene that is ready to be looked at, and the
   * water is the part still in bring-up — so the default is the one that always
   * opens. With `systems.fluid === false` the renderer attaches the live sparse
   * scene directly: no solver, no t=0 authority fence, no transport gate, and
   * none of the three walls in the header above are on the path to a frame.
   *
   * The water is *authored* either way. The fill, the initial condition and the
   * jet stay on the document because they describe a pond that exists whether or
   * not it is being solved this second; the flag alone decides. That is what
   * makes this one scene with a switch rather than two scenes that will drift.
   */
  readonly water?: boolean;
  /**
   * The octree's own lattice, in metres. Defaults to `HERO_GARDEN_CELL_M`.
   *
   * This is `voxelDomain.finestCellSize_m` — the solver's cell when the fluid
   * owns the document, and the render tree's finest level when it does not. The
   * header above is the record of what it costs to move: 7.5 mm overruns a hard
   * one-workgroup-per-interface-leaf dispatch that only a *fluid* scene issues,
   * so a dry lane may go far below it and a wet one may not.
   *
   * Every container dimension has to stay a whole number of 8-cell bricks, and
   * the ladder that does is 25, 12.5, 6.25, 3.125 and 1.5625 mm. Reserving
   * replacement-tree headroom put that invariant at risk — at 1.1 m the 25 mm rung
   * is 44 cells, which is 5.5 bricks, and 25 mm is what
   * `HERO_GARDEN_SOLVER_CELL_M` pins a wet document to — so the height went to
   * 1.2 m instead, which is exactly 6 bricks there and whole at every finer
   * rung. The ladder is unchanged. See `HERO_GARDEN_CONTAINER`.
   */
  readonly cellSize_m?: number;
  /**
   * The finest voxel the *set* will be drawn at, in metres. Defaults to
   * `cellSize_m`; the rung the *product* opens a scene at is applied by
   * `sceneDocument`, not here.
   *
   * Separate because a dry scene may spend extra octree levels under the tree's
   * own lattice (`SvoRenderTuning.environmentRefinementDepth`), so the voxel a
   * bank or a floret is rasterised into is `cellSize_m / 2^depth` — see
   * `svoSceneryDetailCellSize_m`, which is the function that answers this.
   *
   * It is an input to *construction* and not a field set afterwards, and that is
   * the whole point of it being here. By the time a scene exists its heightfield
   * is baked and every generator has expanded at whatever leaf it was given, so
   * a lattice applied to a finished document changes the tree and nothing that
   * feeds it. `FLUID_SVO_DRY_SMOKE_CELL_MM` did exactly that until this
   * parameter existed.
   */
  readonly detailCellSize_m?: number;
}

export function createHeroGardenHoseScene(options: HeroGardenHoseOptions = {}): SceneDescription {
  const scene = cloneScene(defaultScene);
  // The global lattice, coarsened only for a document that asks to be solved.
  // Every scene wants the finest picture; the solver is the one system that
  // cannot carry it, so it is the one that pays — rather than the whole product
  // rendering at the coarsest thing any subsystem happens to need.
  const requestedCell_m = options.cellSize_m ?? HERO_GARDEN_CELL_M;
  const cellSize_m = options.water === true
    ? Math.max(requestedCell_m, HERO_GARDEN_SOLVER_CELL_M)
    : requestedCell_m;
  // The factory does what it is told. The *product's* default rung is a
  // document-construction policy and lives in `sceneDocument`, so a tool or a
  // test that calls this factory directly gets the lattice it asked for and
  // nothing else — see `SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT`.
  const detailCellSize_m = options.detailCellSize_m ?? cellSize_m;
  if (!(cellSize_m > 0) || !Number.isFinite(cellSize_m)) throw new RangeError("Hero garden cell size must be positive and finite");
  if (!(detailCellSize_m > 0) || detailCellSize_m > cellSize_m + 1e-12) {
    throw new RangeError("Hero garden detail cell size must be positive and no coarser than the lattice");
  }
  scene.sceneId = "hero-garden-hose";
  scene.systems = { ...scene.systems, fluid: options.water === true };
  scene.container.width_m = HERO_GARDEN_CONTAINER.width_m;
  scene.container.height_m = HERO_GARDEN_CONTAINER.height_m;
  scene.container.depth_m = HERO_GARDEN_CONTAINER.depth_m;
  scene.container.top = "open";
  scene.container.vessel = "none";
  scene.voxelDomain = {
    finestCellSize_m: cellSize_m,
    brickSize_cells: HERO_GARDEN_BRICK_CELLS,
    // Written only when it says something the lattice does not, so a document
    // built at the authored size round-trips byte for byte as it always did.
    ...(detailCellSize_m < cellSize_m ? { detailCellSize_m } : {}),
  };
  /**
   * The vessel, described rather than baked.
   *
   * `pondVesselHeightAt` is a pure function of (x, z) and the spec is twenty-odd
   * numbers, so the document has no reason to carry a quarter of a million
   * samples of it — and every reason not to: the bake was 4.32 MB of a 4.34 MB
   * scene and 18.5 ms of its 18.6 ms `structuredClone`, paid again on every
   * pointer-move of an editor drag. `terrainSampleGrid` derives exactly the grid
   * this used to bake, at `spacing_m`, memoized by content so it survives the
   * render worker's clone; every consumer still sees one `TerrainGrid` through
   * one bilinear sampler and none of them learns that it was derived.
   */
  scene.terrain = {
    baseHeight_m: HERO_GARDEN_VESSEL.groundHeight_m,
    features: [],
    procedural: {
      kind: "pond-vessel",
      spec: HERO_GARDEN_VESSEL,
      container: { ...HERO_GARDEN_CONTAINER },
      spacing_m: heroGardenTerrainSample_m(detailCellSize_m),
    },
  };
  /**
   * One level for both documents, and that is the point of it.
   *
   * The clearance is a rule about a *fill* on the *solver's* lattice, and this
   * factory's wet path already clamps to `HERO_GARDEN_SOLVER_CELL_M`, so at
   * every rung either document can be built at, `heroGardenWaterline_m` of the
   * solver's cell is the answer — a dry render simply has no fill to round. A
   * lattice that is coarser still is the one case where the water genuinely has
   * to drop, and it is left to say so: `withHeroLayout` refuses a document whose
   * level is not the one the set was composed against, which is louder than a
   * shore that quietly moves under stones already standing on it.
   */
  const waterline_m = heroGardenWaterline_m(options.water === true ? cellSize_m : HERO_GARDEN_SOLVER_CELL_M);
  scene.container.fillFraction = waterline_m / HERO_GARDEN_CONTAINER.height_m;
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.inflow = heroGardenInflow();
  /**
   * The teal, as a property of *this* pond.
   *
   * Absorption is a rate per metre, and the frozen clean-water table it
   * defaults to is calibrated on tanks: at this pond's 115 mm of still depth it
   * transmits (0.950, 0.990, 0.993), so red is 4.6 % down on green and the
   * water carries no hue a viewer could name. The reference's pond is
   * unmistakably teal at the same depth, and no amount of shader tuning can
   * produce that from a coefficient an order of magnitude too small — Beer's
   * law is the only thing between the two, and it has one input.
   *
   * So these are pond coefficients, not water ones. Red goes up 5.8x, green
   * 4.7x and blue 10x: the last is what makes it *teal* rather than *blue*,
   * because dissolved organics and algae absorb in the blue where pure water
   * barely does, and a pond that attenuates blue least ends up the colour of a
   * swimming pool. What the numbers buy, in transmission:
   *
   *   still depth, 0.115 m   (0.742, 0.953, 0.931)  — red 22 % down on green,
   *                                                   against 4.6 % before
   *   eye path,    0.296 m   (0.463, 0.883, 0.832)  — the slant through the
   *                                                   basin at this camera
   *   sun path,    0.141 m   (0.693, 0.942, 0.916)  — refracted to 35 degrees
   *                                                   off vertical
   *
   * Round trip, the basin floor keeps 32 % of its red against 83 % of its
   * green. That is the whole of the pond's colour, and it comes from the medium
   * rather than from a tint added at the end.
   *
   * The in-scatter goes up with it, and by the same factor rather than by an
   * independently chosen one: `unifiedAbsorbingTransmission` blends toward the
   * scatter colour as the transmission falls, so a body that now extinguishes
   * three times as much light needs three times the scatter to stay luminous
   * instead of going to ink. Three times the default, to within 5 % per channel.
   */
  scene.fluid.optics = {
    absorption_mInv: [2.6, 0.42, 0.62],
    scatter: [0.038, 0.165, 0.145],
  };
  /**
   * The stepping discs, as solids the water has to part around.
   *
   * The path is scenery too — the drawn plate is a tapered footing, a tread and
   * a crown that softens its edge, and that silhouette is what stops five discs
   * reading as five drums — but scenery has no solver term, so a disc that is
   * *only* scenery is a disc the jet pours straight through. These are the same
   * five stones, solved once by `heroSteppingPath` and published twice: the
   * generator node draws them, and this collides them. Neither derives the
   * contour again, which is the only way the two can be guaranteed to agree.
   *
   * The collider is a single static cylinder inscribed in the drawn solid — see
   * `steppingStoneBodies` for why one, and why inscribed rather than at the
   * tread's own radius.
   */
  scene.rigidBodies = stoneSetSteppingBodies({
    vessel: HERO_GARDEN_VESSEL,
    waterline_m,
    seed: HERO_GARDEN_SET_SEED,
  });
  // Lighting is not scene data on the fluid-free SVO path. Every dry scene
  // resolves the same defaults at the renderer boundary; see
  // `svo-dry-scene-lighting.ts`.
  // The set is seated on the grid that was just baked, not on the generator that
  // produced it: the two agree at every grid node by construction and differ
  // between them by the bilinear fetch, and the ground a root meets should be
  // the one the ray actually hits. That is now the expander's doing rather than
  // a closure captured here — it samples `scene.terrain`, which is the grid
  // assigned above.
  scene.scenery = heroGardenScenery(waterline_m);
  return scene;
}
