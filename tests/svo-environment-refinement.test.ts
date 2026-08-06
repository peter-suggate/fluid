import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE,
  createSvoEnvironmentRefinement,
  sparseSceneTerrainNodePlanar,
  svoEnvironmentRefinementMode,
} from "../lib/svo-environment-refinement";
import type { SparseSceneTerrainDomain, SparseSceneTerrainField } from "../lib/sparse-scene-terrain-field";
import type { EnvironmentProxyPrimitive } from "../lib/voxel-environments";
import { svoDescriptorForEnvironmentProxy } from "../lib/svo-scene-primitives";

/**
 * A 32-cell cube of 10 mm cells: a finest brick is 80 mm and the tree is two
 * levels deep, so a level-1 node is two finest bricks per axis (160 mm) and a
 * level-0 node is the whole domain. Small enough to reason about a single face.
 */
const CELL_M = 0.01;
const BRICK_SIZE = 8;
const MAXIMUM_DEPTH = 2;
const BRICK_EDGE_M = CELL_M * BRICK_SIZE;
const ORIGIN: readonly [number, number, number] = [0, 0, 0];

const nodeEdge = (): (readonly number[])[] => {
  const table: (readonly number[])[] = [];
  for (let level = 0; level <= MAXIMUM_DEPTH; level += 1) {
    const scale = 2 ** (MAXIMUM_DEPTH - level);
    table.push([BRICK_EDGE_M * scale, BRICK_EDGE_M * scale, BRICK_EDGE_M * scale]);
  }
  return table;
};

/** The identity half every proxy carries; none of it reaches the field. */
const proxy = (shape: Record<string, unknown>): EnvironmentProxyPrimitive => ({
  key: "test/primitive",
  ownerIndex: 0,
  group: "test",
  tags: [],
  material: { albedo: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 },
  ...shape,
} as unknown as EnvironmentProxyPrimitive);

const boxProxy = (
  center: { x: number; y: number; z: number },
  halfSize: { x: number; y: number; z: number },
): EnvironmentProxyPrimitive => proxy({
  kind: "box", center_m: center, halfSize_m: halfSize,
  aabb_m: {
    min: { x: center.x - halfSize.x, y: center.y - halfSize.y, z: center.z - halfSize.z },
    max: { x: center.x + halfSize.x, y: center.y + halfSize.y, z: center.z + halfSize.z },
  },
});

const ellipsoidProxy = (
  center: { x: number; y: number; z: number },
  radius: { x: number; y: number; z: number },
): EnvironmentProxyPrimitive => proxy({
  kind: "ellipsoid", center_m: center, radius_m: radius,
  aabb_m: {
    min: { x: center.x - radius.x, y: center.y - radius.y, z: center.z - radius.z },
    max: { x: center.x + radius.x, y: center.y + radius.y, z: center.z + radius.z },
  },
});

/**
 * The exemption arm, opted into explicitly.
 *
 * Every geometric test below is *about* the exemption — which surfaces earn one
 * and which do not — so they ask for it by name rather than inheriting it. The
 * default is off (see `off` beneath this), and a test that silently rode the
 * default would stop testing anything the day the default moved.
 */
const refinement = (primitives: readonly EnvironmentProxyPrimitive[], terrainField?: SparseSceneTerrainField) =>
  createSvoEnvironmentRefinement({
    planarExemption: true,
    primitives,
    descriptorFor: (primitive) => svoDescriptorForEnvironmentProxy({ rigidBodies: [] }, primitive),
    worldOrigin_m: ORIGIN,
    nodeEdge_m: nodeEdge(),
    brickSize: BRICK_SIZE,
    maximumDepth: MAXIMUM_DEPTH,
    terrainField,
    terrainDomain: {
      worldOrigin_m: ORIGIN,
      cellSize_m: [CELL_M, CELL_M, CELL_M],
      dimensionsCells: [32, 32, 32],
    } satisfies SparseSceneTerrainDomain,
    crowdingTarget: 64,
  });

test("the shipped arm is what an unset or empty lever selects", () => {
  assert.equal(svoEnvironmentRefinementMode({}), "candidates");
  assert.equal(svoEnvironmentRefinementMode({ [SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE]: "" }), "candidates");
  assert.equal(svoEnvironmentRefinementMode({ [SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE]: "surface" }), "surface");
  assert.throws(
    () => svoEnvironmentRefinementMode({ [SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE]: "yes" }),
    RangeError,
  );
});

/**
 * The node every geometric test below is about: level 1, coordinate (1,1,1),
 * which spans [0.16, 0.32] on each axis. Its centre is (0.24, 0.24, 0.24), its
 * half-extent 0.08 m and its enclosing sphere 0.1386 m; its cells are 20 mm, so
 * the planarity residual budget is 5 mm.
 */
const NODE = { x: 1, y: 1, z: 1 };
const NODE_CENTRE = 0.24;

test("a flat floor's face does not split, and a corner between two faces does", () => {
  // A slab far wider than the node whose top face runs through the node centre.
  const floor = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE - 0.5, z: NODE_CENTRE }, { x: 4, y: 0.5, z: 4 });
  const flat = refinement([floor]);
  assert.equal(flat.refineEnvironmentLeaf(1, NODE), false,
    "a node one flat face passes through must stay one leaf");
  assert.equal(flat.statistics.primitiveSplits, 0);
  // The same face, plus the vertical one it meets: the node now holds a crease,
  // and no single plane describes its contents.
  const corner = boxProxy({ x: NODE_CENTRE - 4, y: NODE_CENTRE - 4, z: NODE_CENTRE }, { x: 4, y: 4, z: 8 });
  assert.equal(refinement([corner]).refineEnvironmentLeaf(1, NODE), true,
    "a node two faces meet inside must split");
});

test("a node buried in a primitive's interior does not split", () => {
  // The 54 % case: an AABB overlaps and the surface is nowhere near. One centre
  // evaluation settles it, because |d| exceeds the node's enclosing sphere.
  const enclosing = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE, z: NODE_CENTRE }, { x: 0.5, y: 0.5, z: 0.5 });
  const probe = refinement([enclosing]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), false);
  assert.equal(probe.statistics.straddleSamples, 1, "one sample decides the whole node");
  assert.equal(probe.statistics.planaritySamples, 0, "and the stencil is never paid for");
});

test("a curved surface splits where a primitive-count rule never would", () => {
  // One primitive, so `candidatesInBrick` returns 1 against a budget of 64 and
  // the shipped rule can never split this node. That is the defect.
  // Offset in x as well as y, so the node centre sits on the flank rather than
  // over the pole. A pole is the one place a curved surface *does* present an
  // axis-aligned normal, and the stencil is what rejects it there.
  const cap = ellipsoidProxy({ x: NODE_CENTRE - 0.03, y: NODE_CENTRE - 0.04, z: NODE_CENTRE }, { x: 0.05, y: 0.03, z: 0.05 });
  const probe = refinement([cap]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), true);
  assert.equal(probe.statistics.primitiveSplits, 1);
  assert.equal(probe.statistics.planaritySamples, 0, "an off-axis normal never reaches the stencil");
});

test("a curved surface is rejected at its own pole, where its normal is on axis", () => {
  // Directly over the top of the cap the closest point is the pole, so the
  // normal is exactly +Y and the axis gate passes. The stencil is the only
  // thing standing between that and a coarse patch on a smooth object — which
  // is the artefact this rule exists to make impossible.
  const cap = ellipsoidProxy({ x: NODE_CENTRE, y: NODE_CENTRE - 0.04, z: NODE_CENTRE }, { x: 0.05, y: 0.03, z: 0.05 });
  const probe = refinement([cap]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), true);
  assert.ok(probe.statistics.planaritySamples > 0, "the gate passed and the stencil ran");
});

test("the candidate set narrows down the recursion rather than rescanning", () => {
  const scattered = [
    boxProxy({ x: 0.04, y: 0.04, z: 0.04 }, { x: 0.02, y: 0.02, z: 0.02 }),
    boxProxy({ x: 0.28, y: 0.28, z: 0.28 }, { x: 0.02, y: 0.02, z: 0.02 }),
  ];
  const probe = refinement(scattered);
  probe.refineEnvironmentLeaf(0, { x: 0, y: 0, z: 0 });
  probe.refineEnvironmentLeaf(1, { x: 0, y: 0, z: 0 });
  assert.equal(probe.statistics.fullScans, 1, "only the entry node scans the whole catalogue");
  assert.equal(probe.statistics.narrowedScans, 1);
  // The level-1 child holds one of the two boxes, so its parent's answer of two
  // is what the child filtered — not the catalogue's.
  assert.equal(probe.statistics.candidateVisits, 2 + 2);
});

test("crowding still splits a node no surface passes through", () => {
  // Sixty-five boxes that each contain the node outright, so the detail rule
  // declines every one of them and the cost rule must still fire.
  const crowd = Array.from({ length: 65 }, (_unused, index) => boxProxy(
    { x: NODE_CENTRE, y: NODE_CENTRE, z: NODE_CENTRE },
    { x: 0.2 + index * 1e-4, y: 0.2, z: 0.2 }));
  const probe = refinement(crowd);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), true);
  assert.equal(probe.statistics.crowdingSplits, 1);
  assert.equal(probe.statistics.primitiveSplits, 0);
});

/** A height field of 32 x 32 columns at 10 mm; one level-1 node covers 16 x 16. */
const terrain = (heightAt: (x: number, z: number) => number): SparseSceneTerrainField => {
  const heights = new Float32Array(32 * 32);
  for (let z = 0; z < 32; z += 1) for (let x = 0; x < 32; x += 1) heights[x + 32 * z] = heightAt(x, z);
  let minimum = Infinity, maximum = -Infinity;
  for (const height of heights) { minimum = Math.min(minimum, height); maximum = Math.max(maximum, height); }
  return {
    dimensions: [32, 32], heights_m: heights, minimumHeight_m: minimum, maximumHeight_m: maximum,
    bounds: { minimum: [0, 0, 0], maximum: [0.32, maximum + 0.02, 0.32] },
  };
};

test("level ground is exempt; a constant slope is not", () => {
  const tolerance_m = 0.25 * BRICK_EDGE_M / BRICK_SIZE;
  assert.equal(sparseSceneTerrainNodePlanar(terrain(() => 0.1), BRICK_SIZE, 1, MAXIMUM_DEPTH, NODE, tolerance_m), true);
  // A slope is a plane, and an earlier rule exempted it for being one. It is an
  // *oblique* plane: the voxels-only primary returns a cell face, so the ground
  // staircases by one voxel however flat it is, and halving the leaf halves the
  // step. Only level ground costs nothing to leave coarse.
  assert.equal(
    sparseSceneTerrainNodePlanar(terrain((x, z) => 0.1 + 0.002 * x + 0.001 * z), BRICK_SIZE, 1, MAXIMUM_DEPTH, NODE, tolerance_m),
    false);
  // A ridge inside the node's footprint (columns 16..31), against level ground
  // everywhere else.
  assert.equal(
    sparseSceneTerrainNodePlanar(terrain((x) => (x === 20 ? 0.14 : 0.1)), BRICK_SIZE, 1, MAXIMUM_DEPTH, NODE, tolerance_m),
    false);
});

test("a flat but oblique face is not exempt, however flat it is", () => {
  // The same slab as the floor above, tilted 30 degrees. It is exactly planar
  // across the node — the residual half of the test passes outright — and it
  // still terraces at one voxel, so it must keep refining.
  const angle = Math.PI / 6;
  const flat = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE - 0.5, z: NODE_CENTRE }, { x: 4, y: 0.5, z: 4 });
  const oblique = {
    ...flat, orientation: { w: Math.cos(angle / 2), x: 0, y: 0, z: Math.sin(angle / 2) },
  } as EnvironmentProxyPrimitive;
  const probe = refinement([oblique]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), true);
  assert.equal(probe.statistics.straddleSamples, 1, "the axis gate settles it without the stencil");
  assert.equal(probe.statistics.planaritySamples, 0);
});

test("an exempt node pays for the whole stencil, face centres included", () => {
  // Corners alone cannot see a saddle: its second-order term is
  // (k1 u^2 + k2 v^2) / 2 and |u| = |v| at every corner, so k1 = -k2 reads as
  // exactly planar there. The six face centres close that, and this pins that
  // they are taken.
  const floor = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE - 0.5, z: NODE_CENTRE }, { x: 4, y: 0.5, z: 4 });
  const probe = refinement([floor]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), false);
  assert.equal(probe.statistics.straddleSamples + probe.statistics.planaritySamples, 15);
});

test("flat ground does not split, and a bump in it does", () => {
  // Ground at y = 0.1 is inside the level-1 node spanning y in [0, 0.16].
  const ground = { x: 1, y: 0, z: 1 };
  const flat = refinement([], terrain(() => 0.1));
  assert.equal(flat.refineEnvironmentLeaf(1, ground), false);
  assert.equal(flat.statistics.terrainPlanarExemptions, 1, "the node is ground surface, exempted for flatness");
  const bumpy = refinement([], terrain((x, z) => 0.1 + 0.02 * Math.sin(x) * Math.cos(z)));
  assert.equal(bumpy.refineEnvironmentLeaf(1, ground), true);
  assert.equal(bumpy.statistics.terrainSplits, 1);
});

/** The shipped arm: no exemption, so a surface in the node is the whole rule. */
const off = (primitives: readonly EnvironmentProxyPrimitive[], terrainField?: SparseSceneTerrainField) =>
  createSvoEnvironmentRefinement({
    primitives,
    descriptorFor: (primitive) => svoDescriptorForEnvironmentProxy({ rigidBodies: [] }, primitive),
    worldOrigin_m: ORIGIN,
    nodeEdge_m: nodeEdge(),
    brickSize: BRICK_SIZE,
    maximumDepth: MAXIMUM_DEPTH,
    terrainField,
    terrainDomain: {
      worldOrigin_m: ORIGIN,
      cellSize_m: [CELL_M, CELL_M, CELL_M],
      dimensionsCells: [32, 32, 32],
    } satisfies SparseSceneTerrainDomain,
    crowdingTarget: 64,
  });

test("without the exemption an axis-aligned face still splits, on one sample", () => {
  // The flat floor of the first test, which is the *best* case for the
  // exemption — exactly representable, exactly axis-aligned. Off, it splits
  // anyway, and it costs the centre sample alone: the stencil is what the
  // exemption was for, so removing the exemption removes the stencil.
  const floor = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE - 0.5, z: NODE_CENTRE }, { x: 4, y: 0.5, z: 4 });
  const probe = off([floor]);
  assert.equal(probe.refineEnvironmentLeaf(1, NODE), true);
  assert.equal(probe.statistics.primitiveSplits, 1);
  assert.equal(probe.statistics.straddleSamples, 1);
  assert.equal(probe.statistics.planaritySamples, 0, "no exemption to establish, so no stencil to pay for");
});

test("without the exemption level ground splits, and the interior still does not", () => {
  // The artefact this default exists to remove: a mound's near-level cap is
  // exactly the region the exemption leaves coarse, and a coarse leaf reaches
  // the screen as flat axis-aligned facets against the refined slope.
  const ground = { x: 1, y: 0, z: 1 };
  const flat = off([], terrain(() => 0.1));
  assert.equal(flat.refineEnvironmentLeaf(1, ground), true);
  assert.equal(flat.statistics.terrainSplits, 1);
  assert.equal(flat.statistics.terrainPlanarExemptions, 0);
  // The straddle gate is untouched: a node buried in a solid holds no surface,
  // and this arm splits on surfaces, not on overlap.
  const enclosing = boxProxy({ x: NODE_CENTRE, y: NODE_CENTRE, z: NODE_CENTRE }, { x: 0.5, y: 0.5, z: 0.5 });
  const buried = off([enclosing]);
  assert.equal(buried.refineEnvironmentLeaf(1, NODE), false);
  assert.equal(buried.statistics.straddleSamples, 1);
});
