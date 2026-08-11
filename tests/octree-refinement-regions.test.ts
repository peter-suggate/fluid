import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, validateScene, type FluidRefinementRegion } from "../lib/model";
import {
  OCTREE_REFINEMENT_REGION_CAPACITY,
  packOctreeRefinementRegions,
  refinementRegionBlocksSplit,
  refinementRegionCeilingForLeaf,
  refinementRegionFloorForLeaf,
  refinementRegionForcesSplit,
  refinementRegionLattice,
  sceneHasUniformFinestCellCeiling,
} from "../lib/octree-refinement-regions";
import {
  refinementRegionEntity,
  refinementRegionFromDrag,
  refinementRegionSelectionId,
  refinementRegionsFromQuery,
  refinementRegionsToQuery,
  withRefinementRegion,
  withRefinementRegionsFromQuery,
} from "../lib/editor-refinement-region";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { defaultScenePresetId, getScenePreset } from "../lib/scenes";
import { parseQueryState, serializeQueryState } from "../lib/url-state";
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

test("only a full-domain 1^3 ceiling makes the pressure topology immutable", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const full = region({
    minimumCellSize_cells: 1,
    maximumCellSize_cells: 1,
    min_m: { x: -0.5 * scene.container.width_m, y: 0,
      z: -0.5 * scene.container.depth_m },
    max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m },
  });
  scene.fluid.refinementRegions = [full];
  assert.equal(sceneHasUniformFinestCellCeiling(scene), true);
  scene.fluid.refinementRegions = [{ ...full,
    max_m: { ...full.max_m, x: 0 } }];
  assert.equal(sceneHasUniformFinestCellCeiling(scene), false);
  scene.fluid.refinementRegions = [{ ...full, maximumCellSize_cells: 2 }];
  assert.equal(sceneHasUniformFinestCellCeiling(scene), false);
});

/**
 * Every bound is on the region's own floor lattice, or at the container wall.
 *
 * Dyadic leaves of edge S sit at multiples of S in cell space, so only a box on
 * that lattice contains whole leaves of the size it is asking for. The wall is
 * the documented exception: a domain whose cell count is not a multiple of the
 * floor has no lattice line there.
 */
function assertOnFloorLattice(
  scene: ReturnType<typeof cloneScene>,
  box: { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } },
  cells: number,
) {
  const lattice = refinementRegionLattice(scene);
  const walls = {
    min: lattice.origin_m,
    max: { x: 0.5 * scene.container.width_m, y: scene.container.height_m, z: 0.5 * scene.container.depth_m },
  };
  (["x", "y", "z"] as const).forEach((axis, index) => {
    const step = lattice.cellSize_m[index]! * cells;
    for (const bound of ["min", "max"] as const) {
      const value = box[bound][axis];
      const offset = (value - lattice.origin_m[axis]) / step;
      const aligned = Math.abs(offset - Math.round(offset)) < 1e-6;
      const atWall = Math.abs(value - walls.min[axis]) < 1e-9 || Math.abs(value - walls.max[axis]) < 1e-9;
      assert.ok(aligned || atWall, `${bound}.${axis} = ${value} is neither on the ${step} m lattice nor at a wall`);
    }
  });
}

test("a region holds only the leaves it fully contains, at the coarsest floor over them", () => {
  const data = packed([region({
    minimumCellSize_cells: 8,
    min_m: { x: 8, y: 8, z: 8 },
    max_m: { x: 40, y: 40, z: 40 },
  })]);
  // Wholly inside and at or under the floor: held, so the split is refused.
  assert.equal(refinementRegionBlocksSplit(data, [16, 16, 16], 8), true);
  assert.equal(refinementRegionBlocksSplit(data, [16, 16, 16], 2), true);
  // Coarser than the floor: the ordinary evidence still decides. A region caps
  // refinement; it must never prevent it.
  assert.equal(refinementRegionBlocksSplit(data, [8, 8, 8], 16), false);
  // Straddling the boundary: not contained, so nothing outside is coarsened.
  assert.equal(refinementRegionBlocksSplit(data, [4, 16, 16], 8), false);
  assert.equal(refinementRegionBlocksSplit(data, [36, 16, 16], 8), false);

  // A box over a box is a request for LESS resolution. Taking the finer floor
  // would make the second box do nothing at all.
  const stacked = packed([
    region({ id: "a", minimumCellSize_cells: 2, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
    region({ id: "b", minimumCellSize_cells: 16, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
  ]);
  assert.equal(refinementRegionFloorForLeaf(stacked, [0, 0, 0], 16), 16);
});

test("an optional largest cell forces fully contained leaves to its finest overlapping ceiling", () => {
  const bounded = packed([region({
    minimumCellSize_cells: 1,
    maximumCellSize_cells: 4,
    min_m: { x: 8, y: 8, z: 8 },
    max_m: { x: 40, y: 40, z: 40 },
  })]);
  assert.equal(new Float32Array(bounded, 16)[7], 4,
    "the spare hi.w word carries the optional ceiling");
  assert.equal(refinementRegionCeilingForLeaf(bounded, [8, 8, 8], 8), 4);
  assert.equal(refinementRegionForcesSplit(bounded, [8, 8, 8], 8), true);
  assert.equal(refinementRegionForcesSplit(bounded, [8, 8, 8], 4), false);
  assert.equal(refinementRegionForcesSplit(bounded, [4, 8, 8], 8), false,
    "a boundary-straddling leaf remains outside the region's authority");

  const stacked = packed([
    region({ id: "coarse", minimumCellSize_cells: 1, maximumCellSize_cells: 8,
      min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
    region({ id: "fine", minimumCellSize_cells: 1, maximumCellSize_cells: 2,
      min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
  ]);
  assert.equal(refinementRegionCeilingForLeaf(stacked, [0, 0, 0], 16), 2,
    "every overlapping ceiling is satisfied by choosing the finest one");
  assert.equal(refinementRegionCeilingForLeaf(packed([region()]), [0, 0, 0], 8), 0,
    "an omitted largest cell preserves the old evidence-driven coarsening");

  const scene = cloneScene(defaultScene);
  assert.match(validateScene(withRefinementRegion(scene, "region-1", region({
    minimumCellSize_cells: 8,
    maximumCellSize_cells: 4,
  }))).join("\n"), /maximum cell size must not be smaller/);
});

test("a scene with no usable regions is inert rather than special-cased", () => {
  // Floor one has to read as "no opinion" — a size-one leaf never splits anyway
  // — or every domain without regions would stop refining.
  assert.equal(refinementRegionBlocksSplit(packed([]), [0, 0, 0], 2), false);

  const mixed = packed([
    region({ id: "flat", min_m: { x: 4, y: 4, z: 4 }, max_m: { x: 4, y: 12, z: 12 } }),
    { ...region({ id: "other" }), rule: "something-else" as FluidRefinementRegion["rule"] },
    region({ id: "good", min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }),
  ]);
  assert.equal(new Uint32Array(mixed, 0, 4)[0], 1, "degenerate and unimplemented regions must not be packed");

  // A floor above the largest leaf the topology builds would read as "never
  // refine anything here", which is not the instruction the user gave.
  assert.equal(refinementRegionFloorForLeaf(
    packed([region({ minimumCellSize_cells: 32, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 64, y: 64, z: 64 } })], 8),
    [0, 0, 0], 8), 8);

  // The uniform tail is fixed-size; more regions than it carries must be
  // dropped, never written past the end.
  const many = Array.from({ length: OCTREE_REFINEMENT_REGION_CAPACITY + 4 }, (_unused, index) =>
    region({ id: `region-${index}`, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: 32, y: 32, z: 32 } }));
  assert.equal(new Uint32Array(packed(many), 0, 4)[0], OCTREE_REFINEMENT_REGION_CAPACITY);
});

test("a drawn region lands on its own floor lattice", () => {
  const scene = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.31, y: 0.07, z: -0.29 }, { x: 0.42, y: 0.07, z: 0.36 }, { minimumCellSize_cells: 8 });
  assertOnFloorLattice(scene, { min: drawn.min_m, max: drawn.max_m }, 8);
  // Outward, so the region covers everything the drag touched.
  assert.ok(drawn.min_m.x <= -0.31 + 1e-9 && drawn.max_m.x >= 0.42 - 1e-9);
  assert.deepEqual(validateScene(withRefinementRegion(scene, drawn.id, drawn)), []);
});

test("the editor offers AUTO or a dyadic largest cell and keeps both bounds valid", () => {
  const scene = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.3, y: 0, z: -0.3 }, { x: 0.3, y: 0, z: 0.3 },
    { minimumCellSize_cells: 8 });
  const authored = withRefinementRegion(scene, drawn.id, drawn);
  const entity = refinementRegionEntity.find(
    { scene: authored, bodies: [] }, refinementRegionSelectionId(drawn.id));
  assert.ok(entity);
  const largest = entity.choices?.find((choice) => choice.id === "maximumCellSize");
  assert.ok(largest);
  assert.equal(largest.label, "Largest cell");
  assert.equal(largest.value, "auto");
  assert.equal(largest.options[0]?.label, "AUTO");

  const fineOnly = largest.options.find((option) => option.id === "1")?.apply();
  assert.ok(fineOnly?.fluid?.refinementRegions);
  assert.equal(fineOnly.fluid.refinementRegions[0]?.minimumCellSize_cells, 1,
    "lowering the ceiling also lowers an incompatible existing floor");
  assert.equal(fineOnly.fluid.refinementRegions[0]?.maximumCellSize_cells, 1);
  assert.deepEqual(validateScene(fineOnly as ReturnType<typeof cloneScene>), []);
});

// ---- the address bar ------------------------------------------------------

test("a bounded region round-trips through the query as percentages, without creeping", () => {
  const scene = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(scene,
    { x: -0.3, y: 0.1, z: -0.3 }, { x: 0.3, y: 0.1, z: 0.3 },
    { minimumCellSize_cells: 8, maximumCellSize_cells: 16 });
  const authored = withRefinementRegion(scene, drawn.id, drawn);
  const query = refinementRegionsToQuery(authored);

  assert.match(query, /^[0-9._*]+$/, "the value must survive URLSearchParams unescaped");
  assert.ok(query.split("_").every((field) => Number(field) >= 0 && Number(field) <= 100),
    "every positional field must be a percentage or a cell count, never a metre extent");

  const restored = withRefinementRegionsFromQuery(authored, query);
  const region = restored.fluid.refinementRegions?.[0];
  assert.ok(region);
  assert.equal(region.minimumCellSize_cells, drawn.minimumCellSize_cells);
  assert.equal(region.maximumCellSize_cells, drawn.maximumCellSize_cells);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(region.min_m[axis] - drawn.min_m[axis]) < 1e-9, `min.${axis}`);
    assert.ok(Math.abs(region.max_m[axis] - drawn.max_m[axis]) < 1e-9, `max.${axis}`);
  }
  // Open, re-write, open again must not grow the box, which the outward lattice
  // snap does not give for free once a percentage has been through the maths.
  assert.equal(refinementRegionsToQuery(restored), query);

  // A hand-edited link lands nowhere near a lattice line. Trusting it would cost
  // a shell of cells to partial containment and the floor would quietly stop
  // being the floor.
  const hand = refinementRegionsFromQuery(scene, "13.7_4.1_22.9_61.3_48.8_77.2_8")[0];
  assert.ok(hand);
  assert.equal(hand.maximumCellSize_cells, undefined,
    "old seven-field links retain evidence-driven coarsening");
  assertOnFloorLattice(scene, { min: hand.min_m, max: hand.max_m }, 8);
});

test("the same query describes the same fraction of a differently sized tank", () => {
  const small = cloneScene(defaultScene);
  const drawn = refinementRegionFromDrag(small,
    { x: -small.container.width_m / 2, y: 0, z: -small.container.depth_m / 2 },
    { x: 0, y: 0, z: 0 }, { minimumCellSize_cells: 4 });
  const query = refinementRegionsToQuery(withRefinementRegion(small, drawn.id, drawn));

  const large = cloneScene(defaultScene);
  large.container = { ...large.container, width_m: small.container.width_m * 3, depth_m: small.container.depth_m * 2 };
  const restored = withRefinementRegionsFromQuery(large, query);
  const region = restored.fluid.refinementRegions?.[0];
  assert.ok(region);
  // The box occupied the -x/-z quadrant of the small tank, so it must occupy the
  // -x/-z quadrant of the large one — not the same number of metres.
  const fraction = (value: number) => (value + large.container.width_m / 2) / large.container.width_m;
  assert.ok(Math.abs(fraction(region.min_m.x)) < 1e-6);
  assert.ok(Math.abs(fraction(region.max_m.x) - 0.5) < 0.05);
  // The floor is a cell COUNT and deliberately does not scale: it is how many
  // solver cells you are willing to merge, which means the same thing anywhere.
  assert.equal(region.minimumCellSize_cells, 4);
  assert.deepEqual(validateScene(restored), []);
});

test("external region values are clamped, dropped, or capped rather than trusted", () => {
  const scene = cloneScene(defaultScene);
  const parsed = refinementRegionsFromQuery(scene,
    // out of range, too few fields, non-numeric, an unimplemented rule, one good record
    "-40_0_-40_400_400_400_8*1_2_3*a_b_c_d_e_f_g*0_0_0_50_50_50_8_something-else*10_0_10_60_60_60_16");
  assert.equal(parsed.length, 2, "one bad record must not cost the reader the rest of the boxes");
  assert.ok(parsed[0]!.min_m.x >= -scene.container.width_m / 2 - 1e-9);
  assert.ok(parsed[0]!.max_m.x <= scene.container.width_m / 2 + 1e-9);
  assert.equal(parsed[1]!.minimumCellSize_cells, 16);

  const many = Array.from({ length: OCTREE_REFINEMENT_REGION_CAPACITY + 3 }, () => "0_0_0_50_50_50_8").join("*");
  assert.equal(refinementRegionsFromQuery(scene, many).length, OCTREE_REFINEMENT_REGION_CAPACITY);
});

test("regions survive a full query serialize and parse, and an emptied list is not restored", () => {
  const drawn = refinementRegionFromDrag(cloneScene(defaultScene),
    { x: -0.2, y: 0, z: -0.2 }, { x: 0.2, y: 0, z: 0.2 }, { minimumCellSize_cells: 8 });
  const methodState = { methodId: "octree", quality: "balanced" as const, overrides: {} };
  const sceneState = {
    presetId: defaultScenePresetId,
    scene: withRefinementRegion(getScenePreset(defaultScenePresetId).create(), drawn.id, drawn),
  };
  const search = serializeQueryState("", sceneState, methodState);
  assert.match(search, /(^|&)regions=/);
  assert.equal(parseQueryState(search).scene.fluid.refinementRegions?.length, 1);

  // Removing the last region has to write the key as empty rather than omit it,
  // or hydration falls back to the preset and the removal undoes itself.
  const emptied = serializeQueryState(search,
    { ...sceneState, scene: withRefinementRegion(sceneState.scene, drawn.id, undefined) }, methodState);
  assert.doesNotMatch(emptied, /(^|&)regions=[^&]/);
  assert.equal(parseQueryState(emptied).scene.fluid.refinementRegions, undefined);
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
  assert.equal(sceneEditRequiresReset(after,
    withRefinementRegion(after, "region-1", region({ minimumCellSize_cells: 16 }))), false);
  const fineOnly = withRefinementRegion(after, "region-1", region({
    minimumCellSize_cells: 1,
    maximumCellSize_cells: 1,
  }));
  assert.notEqual(gpuSceneUniformKey(after), gpuSceneUniformKey(fineOnly));
  assert.equal(sceneEditRequiresReset(after, fineOnly), false);
});

test("every refinement decision consults the authored regions", () => {
  // Three call sites, and all three matter: the fine kernel, the coarse
  // workgroup kernel (the only one that ever sees a size-16/32 candidate, which
  // is where a coarse floor is realized), and the retained tile signature —
  // without the last, drawing a region would not dirty the tiles it covers and
  // the delta topology path would never rebuild them.
  assert.match(octreeProjectionShader,
    /fn leafNeedsRefinement\(origin: vec3u, size: u32\) -> bool \{[\s\S]{0,400}?refinementRegionForcesSplit\(origin, size\)[\s\S]{0,80}?return true[\s\S]{0,160}?refinementRegionHoldsLeaf\(origin, size\)[\s\S]{0,80}?return false/);
  assert.match(octreeProjectionShader,
    /fn pressureRefinementEvidence\(origin: vec3u, size: u32\) -> bool \{[\s\S]{0,800}?refinementRegionForcesSplit\(origin, size\)[\s\S]{0,80}?return true[\s\S]{0,160}?refinementRegionHoldsLeaf\(origin, size\)[\s\S]{0,80}?return false/);
  assert.match(octreeProjectionShader,
    /let forcedByRegion = refinementRegionForcesSplit\(origin, size\)[\s\S]{0,100}?let decision = forcedByRegion \|\| \(!refinementRegionHoldsLeaf\(origin, size\)/);
});

test("the final split to a regional floor retains only a one-cell interface shell", () => {
  assert.match(octreeProjectionShader,
    /fn refinementRegionFloorCutover\(origin: vec3u, size: u32\) -> bool \{[\s\S]{0,240}?size == 2u \* floorSize/);
  assert.match(octreeProjectionShader,
    /if \(crossesInterface\) \{ return true; \}[\s\S]{0,500}?if \(summary\.sizingRefinement\) \{ return true; \}[\s\S]{0,900}?if \(refinementRegionFloorCutover\(origin, size\)\) \{[\s\S]{0,160}?summary\.minimumPhi < 0\.0 && summary\.minimumAbsolutePhi <= cellWidth/,
    "a cut, transported activity, or its immediate wet neighbor may reach the floor, but the full two-sided halo may not");
  assert.match(octreeProjectionShader,
    /refinementRegionFloorCutover\(origin, size\) && crossesClosedWall[\s\S]{0,180}?return false/,
    "closed-wall look-ahead must not blanket a region's finest permitted tier");
});

test("factor-one coarse pressure membership uses the owner centre, not any wet subcell", () => {
  assert.match(octreeProjectionShader,
    /ownerValidSamples: u32,[\s\S]{0,80}?ownerNegativeSamples: u32/);
  assert.match(octreeProjectionShader,
    /fine\.ownerValidSamples==owner\.size\*owner\.size\*owner\.size[\s\S]{0,160}?wet=2u\*fine\.ownerNegativeSamples>=fine\.ownerValidSamples/,
    "a partially cut 2^3 leaf must not become eight liquid pressure volumes from one negative sample");
  assert.doesNotMatch(octreeProjectionShader,
    /owner\.size>=2u&&fine\.complete\)\{wet=fine\.minimumPhi<0\.0;\}\s*else if\(fine\.centerValid\)/,
    "the conservative refinement interval must not outrank exact centre evidence");
});
