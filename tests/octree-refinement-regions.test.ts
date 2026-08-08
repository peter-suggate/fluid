import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, validateScene, type FluidRefinementRegion, type SceneDescription } from "../lib/model";
import {
  OCTREE_REFINEMENT_REGION_CAPACITY,
  clampRefinementRegionCellSize,
  packOctreeRefinementRegions,
  refinementRegionBlocksSplit,
  refinementRegionFloorForLeaf,
  refinementRegionLattice,
  sceneRefinementRegions,
} from "../lib/octree-refinement-regions";
import {
  refinementRegionFromDrag,
  refinementRegionResizePolicy,
  snapRefinementRegionBox,
  withRefinementRegion,
} from "../lib/editor-refinement-region";
import { resizeBox } from "../lib/editor-entity";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { getMethod, resolveMethodValues } from "../lib/methods";
import {
  gpuSceneSeedKey,
  gpuSceneStructuralKey,
  gpuSceneUniformKey,
  sceneEditRequiresReset,
  type SimulationRunConfig,
} from "../lib/webgpu-renderer";

const config: SimulationRunConfig = {
  methodId: "octree",
  quality: "balanced",
  values: resolveMethodValues(getMethod("octree"), "balanced", {}),
  simulationEpoch: 0,
};

function region(overrides: Partial<FluidRefinementRegion> = {}): FluidRefinementRegion {
  return {
    id: "region-1",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
    min_m: { x: -0.4, y: 0, z: -0.4 },
    max_m: { x: 0.4, y: 0.4, z: 0.4 },
    ...overrides,
  };
}

/** A unit lattice, so a region's metres and its cells are the same number. */
const UNIT_LATTICE = {
  dimensions: [64, 64, 64] as const,
  cellSize_m: [1, 1, 1] as const,
  origin_m: { x: 0, y: 0, z: 0 },
};

function packed(regions: readonly FluidRefinementRegion[], maximumLeafSize = 32) {
  return packOctreeRefinementRegions(regions, UNIT_LATTICE, maximumLeafSize);
}

test("a region holds only the leaves it fully contains", () => {
  const data = packed([region({
    minimumCellSize_cells: 8,
    min_m: { x: 8, y: 8, z: 8 },
    max_m: { x: 40, y: 40, z: 40 },
  })]);
  // Wholly inside and at the floor: held, so the split is refused.
  assert.equal(refinementRegionBlocksSplit(data, [16, 16, 16], 8), true);
  assert.equal(refinementRegionFloorForLeaf(data, [16, 16, 16], 8), 8);
  // Finer than the floor and inside: also held — that is what caps refinement.
  assert.equal(refinementRegionBlocksSplit(data, [16, 16, 16], 2), true);
  // Coarser than the floor: the ordinary evidence still decides, so the region
  // must not answer. A region caps refinement; it never prevents it.
  assert.equal(refinementRegionBlocksSplit(data, [8, 8, 8], 16), false);
  // Straddling the boundary: not contained, so nothing outside is coarsened.
  assert.equal(refinementRegionBlocksSplit(data, [4, 16, 16], 8), false);
  assert.equal(refinementRegionBlocksSplit(data, [36, 16, 16], 8), false);
  // Entirely elsewhere.
  assert.equal(refinementRegionBlocksSplit(data, [48, 48, 48], 8), false);
});

test("an empty scene's regions are inert rather than special-cased", () => {
  const data = packed([]);
  assert.equal(new Uint32Array(data, 0, 4)[0], 0);
  assert.equal(refinementRegionFloorForLeaf(data, [0, 0, 0], 32), 1);
  // A size-one leaf never splits anyway, so floor one has to read as "no
  // opinion" or every domain without regions would stop refining.
  assert.equal(refinementRegionBlocksSplit(data, [0, 0, 0], 2), false);
});

test("overlapping regions take the coarsest floor", () => {
  const data = packed([
    region({ id: "a", minimumCellSize_cells: 2, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
    region({ id: "b", minimumCellSize_cells: 16, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
  ]);
  // A box drawn over a box is a request for LESS resolution. Taking the finer
  // floor would make the second box do nothing at all.
  assert.equal(refinementRegionFloorForLeaf(data, [0, 0, 0], 16), 16);
});

test("a floor can never exceed the largest leaf the topology builds", () => {
  const data = packed([region({
    minimumCellSize_cells: 32,
    min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 64, y: 64, z: 64 },
  })], 8);
  // Otherwise a region would read as "never refine anything here", which is a
  // different instruction than the one the user gave.
  assert.equal(refinementRegionFloorForLeaf(data, [0, 0, 0], 8), 8);
});

test("degenerate and unimplemented regions are dropped rather than packed", () => {
  const data = packed([
    region({ id: "flat", min_m: { x: 4, y: 4, z: 4 }, max_m: { x: 4, y: 12, z: 12 } }),
    { ...region({ id: "other" }), rule: "something-else" as FluidRefinementRegion["rule"] },
    region({ id: "good", min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
  ]);
  assert.equal(new Uint32Array(data, 0, 4)[0], 1);
  assert.equal(refinementRegionFloorForLeaf(data, [0, 0, 0], 8), 8);
});

test("only the capacity the uniform tail carries is published", () => {
  const many = Array.from({ length: OCTREE_REFINEMENT_REGION_CAPACITY + 4 }, (_unused, index) =>
    region({ id: `region-${index}`, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }));
  const scene = cloneScene(defaultScene);
  scene.fluid.refinementRegions = many;
  assert.equal(sceneRefinementRegions(scene).length, OCTREE_REFINEMENT_REGION_CAPACITY);
  assert.equal(new Uint32Array(packed(many), 0, 4)[0], OCTREE_REFINEMENT_REGION_CAPACITY);
});

test("cell sizes round down onto the dyadic ladder", () => {
  assert.equal(clampRefinementRegionCellSize(1), 1);
  assert.equal(clampRefinementRegionCellSize(6), 4);
  assert.equal(clampRefinementRegionCellSize(0), 1);
  assert.equal(clampRefinementRegionCellSize(1000), 32);
  assert.equal(clampRefinementRegionCellSize(Number.NaN), 8);
});

// ---- the editor -----------------------------------------------------------

/**
 * Every bound is on the region's own floor lattice, or at the container wall.
 *
 * The wall is the documented exception: a domain whose cell count is not a
 * multiple of the floor has no lattice line there, and stopping at the wall is
 * what keeps a region that was dragged to the edge from dropping a row.
 */
function assertOnFloorLattice(
  scene: SceneDescription,
  box: { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } },
  cells: number,
) {
  const lattice = refinementRegionLattice(scene);
  const limits = {
    min: lattice.origin_m,
    max: {
      x: 0.5 * scene.container.width_m,
      y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m,
    },
  };
  (["x", "y", "z"] as const).forEach((axis, index) => {
    const step = lattice.cellSize_m[index]! * cells;
    for (const bound of ["min", "max"] as const) {
      const value = box[bound][axis];
      const offset = (value - lattice.origin_m[axis]) / step;
      const aligned = Math.abs(offset - Math.round(offset)) < 1e-6;
      const atWall = Math.abs(value - limits.min[axis]) < 1e-9 || Math.abs(value - limits.max[axis]) < 1e-9;
      assert.ok(aligned || atWall, `${bound}.${axis} = ${value} is neither on the ${step} m lattice nor at a wall`);
    }
  });
}

test("a drawn region lands on its own floor lattice", () => {
  const scene = cloneScene(defaultScene);
  const cells = 8;
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.31, y: 0.07, z: -0.29 }, { x: 0.42, y: 0.07, z: 0.36 },
    { minimumCellSize_cells: cells });
  assert.equal(drawn.rule, "minimum-cell-size");
  assert.equal(drawn.minimumCellSize_cells, cells);
  // Dyadic leaves of edge S sit at multiples of S in cell space, so only a box
  // on that lattice contains whole leaves of the size it is asking for.
  assertOnFloorLattice(scene, { min: drawn.min_m, max: drawn.max_m }, cells);
  // Outward, so the region covers everything the drag touched.
  assert.ok(drawn.min_m.x <= -0.31 + 1e-9 && drawn.max_m.x >= 0.42 - 1e-9);
});

test("a click with no drag still yields one floor cell rather than an empty box", () => {
  const scene = cloneScene(defaultScene);
  const point = { x: 0.1, y: 0.1, z: 0.1 };
  const drawn = refinementRegionFromDrag(scene, point, point, { minimumCellSize_cells: 4 });
  const step = refinementRegionLattice(scene).cellSize_m;
  assert.ok(drawn.max_m.x - drawn.min_m.x >= step[0]! * 4 - 1e-9);
  assert.ok(drawn.max_m.y - drawn.min_m.y >= step[1]! * 4 - 1e-9);
  assert.ok(drawn.max_m.z - drawn.min_m.z >= step[2]! * 4 - 1e-9);
  assert.deepEqual(validateScene(withRefinementRegion(scene, drawn.id, drawn)), []);
});

test("resizing a region snaps to the cell size it is asking for", () => {
  const scene = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.3, y: 0, z: -0.3 }, { x: 0.3, y: 0, z: 0.3 }, { minimumCellSize_cells: 8 });
  const policy = refinementRegionResizePolicy(scene, drawn);
  const step = refinementRegionLattice(scene).cellSize_m[0]! * 8;
  const next = resizeBox({ min: drawn.min_m, max: drawn.max_m }, { x: "max" },
    { x: drawn.max_m.x + 0.4 * step, y: 0, z: 0 }, policy);
  // Landed on the lattice, not where the pointer happened to be.
  const offset = (next.max.x - policy.limits!.min.x) / step;
  assert.ok(Math.abs(offset - Math.round(offset)) < 1e-6);
});

test("the region list is added to, replaced in, and dropped entirely", () => {
  const scene = cloneScene(defaultScene);
  const first = region({ id: "region-1" });
  const added = withRefinementRegion(scene, first.id, first);
  assert.equal(added.fluid.refinementRegions?.length, 1);

  const retuned = withRefinementRegion(added, first.id, { ...first, minimumCellSize_cells: 16 });
  assert.equal(retuned.fluid.refinementRegions?.length, 1);
  assert.equal(retuned.fluid.refinementRegions?.[0]?.minimumCellSize_cells, 16);

  const second = region({ id: "region-2", min_m: { x: 0.5, y: 0, z: 0.5 }, max_m: { x: 0.9, y: 0.4, z: 0.9 } });
  const both = withRefinementRegion(retuned, second.id, second);
  assert.equal(both.fluid.refinementRegions?.length, 2);

  // An empty list and an absent field mean the same thing, and only one of them
  // leaves an untouched document as it was authored.
  const emptied = withRefinementRegion(
    withRefinementRegion(both, second.id, undefined), first.id, undefined);
  assert.equal("refinementRegions" in emptied.fluid, false);
});

test("snapping to a coarser floor keeps the box aligned to the new lattice", () => {
  const scene = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.3, y: 0, z: -0.3 }, { x: 0.3, y: 0, z: 0.3 }, { minimumCellSize_cells: 2 });
  // Choosing a coarser cell in the flyout re-snaps: a box aligned to the old
  // floor loses a shell of cells to partial containment against the new one.
  const resnapped = snapRefinementRegionBox(scene, { min: drawn.min_m, max: drawn.max_m }, 16);
  assertOnFloorLattice(scene, resnapped, 16);
  // Re-snapping only ever grows, so no part of the region is given up.
  assert.ok(resnapped.min.x <= drawn.min_m.x + 1e-9 && resnapped.max.x >= drawn.max_m.x - 1e-9);
});

// ---- schema ---------------------------------------------------------------

test("authored regions are validated rather than silently reinterpreted", () => {
  const withRegions = (regions: FluidRefinementRegion[]): SceneDescription => {
    const scene = cloneScene(defaultScene);
    scene.fluid.refinementRegions = regions;
    return scene;
  };
  assert.deepEqual(validateScene(withRegions([region()])), []);
  assert.match(validateScene(withRegions([region({ minimumCellSize_cells: 3 })])).join(";"),
    /minimum cell size must be one of/);
  assert.match(validateScene(withRegions([region({ max_m: { x: -0.4, y: 0.4, z: 0.4 } })])).join(";"),
    /positive extent/);
  assert.match(validateScene(withRegions([region(), region()])).join(";"), /duplicate id/);
  assert.match(
    validateScene(withRegions([{ ...region(), rule: "nope" as FluidRefinementRegion["rule"] }])).join(";"),
    /unsupported rule/);
});

// ---- solver tiers ---------------------------------------------------------

test("drawing a region is a uniform write, not a re-seed", () => {
  const before = cloneScene(defaultScene);
  const after = withRefinementRegion(before, "region-1", region());
  // The whole reason a region is an experiment surface: the topology is
  // re-derived from the reset size every epoch, so the running solver adopts a
  // new box without losing its clock.
  assert.equal(gpuSceneSeedKey(before), gpuSceneSeedKey(after));
  assert.equal(gpuSceneStructuralKey(before, config), gpuSceneStructuralKey(after, config));
  assert.notEqual(gpuSceneUniformKey(before), gpuSceneUniformKey(after));
  assert.equal(sceneEditRequiresReset(before, after), false);

  const retuned = withRefinementRegion(after, "region-1", region({ minimumCellSize_cells: 16 }));
  assert.notEqual(gpuSceneUniformKey(after), gpuSceneUniformKey(retuned));
  assert.equal(sceneEditRequiresReset(after, retuned), false);
});

// ---- the shader gate ------------------------------------------------------

test("every refinement decision consults the authored regions", () => {
  // Three call sites, and all three matter: the fine kernel, the coarse
  // workgroup kernel (the only one that ever sees a size-16/32 candidate, which
  // is where a coarse floor is realized), and the retained tile signature —
  // without the last, drawing a region would not dirty the tiles it covers and
  // the delta topology path would never rebuild them.
  assert.match(octreeProjectionShader,
    /fn leafNeedsRefinement\(origin: vec3u, size: u32\) -> bool \{[\s\S]{0,400}?refinementRegionHoldsLeaf\(origin, size\)[\s\S]{0,80}?return false/);
  assert.match(octreeProjectionShader,
    /fn pressureRefinementEvidence\(origin: vec3u, size: u32\) -> bool \{[\s\S]{0,600}?refinementRegionHoldsLeaf\(origin, size\)[\s\S]{0,80}?return false/);
  assert.match(octreeProjectionShader,
    /let decision = !refinementRegionHoldsLeaf\(origin, size\)\s*&& \(pressureEvidence/);
  // Full containment and the coarsest floor, matching `refinementRegionFloorForLeaf`.
  assert.match(octreeProjectionShader,
    /all\(low >= lo\.xyz\) && all\(high <= hi\.xyz\)[\s\S]{0,80}?floorSize = max\(floorSize, u32\(lo\.w\)\)/);
});
