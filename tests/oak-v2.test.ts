import assert from "node:assert/strict";
import test from "node:test";
import { planOakV2, oakFoliageDescriptor, type OakV2Spec } from "../lib/core/voxel-scenery/oak-v2";
import { sampleSvoPrimitive, validateSvoClusterPacking } from "../lib/svo/contracts/svo-primitive-abi";
import type { SceneryRecursiveShapeNode, SceneryTaperedSweepClusterNode } from "../lib/core/scenery-graph";
const spec: OakV2Spec = { key: "oak", seed: 0x10a2,
  bark: { palette: "clay", value: .955, surface: "architectural" },
  foliage: { palette: "clay", value: .975, surface: "foliage" } };
const separation = (a: {
  x: number;
  y: number;
  z: number;
}, b: {
  x: number;
  y: number;
  z: number;
}) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
for (const seed of [0, 1, 4258, 999, -2147483648])
  test(`oak v2 seed ${seed}: connected acyclic wood and area-conserving forks`, () => {
    const plan = planOakV2({ ...spec, seed });
    const byId = new Map(plan.branches.map(branch => [branch.id, branch]));
    assert.equal(byId.size, plan.branches.length);
    assert.equal(plan.branches.filter(branch => branch.parent === null).length, 1);
    const seen = new Set<string>();
    for (const branch of plan.branches) {
      if (branch.parent) {
        assert.ok(seen.has(branch.parent), "parent must precede child; no cycles");
        const parent = byId.get(branch.parent)!;
        assert.ok(separation(branch.from, parent.to) < 1e-12, `${branch.id}: disconnected fork`);
        assert.ok(branch.baseRadius <= parent.tipRadius + 1e-12);
      }
      seen.add(branch.id);
      assert.ok(separation(branch.from, branch.to) > Math.abs(branch.baseRadius - branch.tipRadius));
      const children = plan.branches.filter(child => child.parent === branch.id);
      if (children.length) {
        assert.equal(children.reduce((n, child) => n + child.leaves, 0), branch.leaves);
        assert.ok(Math.abs(children.reduce((area, child) => area + child.baseRadius ** 2, 0) - branch.tipRadius ** 2) < 1e-12);
      }
    }
    const wood = (plan.node.children[0] as {
      children: readonly SceneryTaperedSweepClusterNode[];
    }).children;
    for (const node of wood) {
      validateSvoClusterPacking({ field: "tapered-sweep", seed: node.seed, smoothRadius_m: node.smoothRadius,
        points: node.points.map(point => ({ position_m: point.position, radius_m: point.radius })) }, node.lobe);
    }
  });
test("every foliage shoot terminates in occupied density, including across scale", () => {
  for (const scale_m of [.5, 1, 4]) {
    const plan = planOakV2({ ...spec, scale_m });
    const pads = (plan.node.children[1] as {
      children: readonly SceneryRecursiveShapeNode[];
    }).children;
    const terminals = plan.branches.filter(branch => !plan.branches.some(child => child.parent === branch.id));
    assert.equal(pads.length, terminals.length);
    for (const pad of pads) {
      const at = pad.place!.position!;
      assert.ok(terminals.some(branch => separation(branch.to, at) < 1e-12));
      const descriptor = oakFoliageDescriptor(pad);
      validateSvoClusterPacking(descriptor.packing!, descriptor.lobeRadii_m);
      assert.ok(sampleSvoPrimitive(descriptor, at).signedDistance_m < 0);
    }
  }
});
test("oak is deterministic JSON geometry with unique editable IDs and bounded cost", () => {
  const a = planOakV2(spec);
  assert.deepEqual(a, planOakV2(spec));
  assert.deepEqual(a.node, JSON.parse(JSON.stringify(a.node)));
  assert.notDeepEqual(a, planOakV2({ ...spec, seed: spec.seed + 1 }));
  const ids = a.node.children.flatMap(group => (group as {
    children: readonly {
      id: string;
    }[];
  }).children.map(node => node.id));
  assert.equal(ids.length, new Set(ids).size);
  assert.ok(ids.length <= 2300, "tree must fit a bounded primitive budget");
  assert.throws(() => planOakV2({ ...spec, scale_m: Infinity }), RangeError);
});
test("foliage fields contain both leaf mass and resolved gaps instead of solid pads", () => {
  const plan = planOakV2(spec);
  const pads = (plan.node.children[1] as {
    children: readonly SceneryRecursiveShapeNode[];
  }).children;
  let occupied = 0, total = 0;
  for (const pad of pads.slice(0, 16)) {
    const descriptor = oakFoliageDescriptor(pad), c = descriptor.center_m;
    for (let x = -4; x <= 4; x++)
      for (let y = -3; y <= 3; y++)
        for (let z = -4; z <= 4; z++) {
          // Interior probes, away from the guaranteed empty envelope boundary.
          const point = { x: c.x + x * .003, y: c.y + y * .003, z: c.z + z * .003 };
          occupied += Number(sampleSvoPrimitive(descriptor, point).signedDistance_m < 0);
          total++;
        }
  }
  const fill = occupied / total;
  assert.ok(fill > .15 && fill < .7, `interior fill ${fill}: mass and gaps must coexist`);
});
test("emitted sweeps cover every fork and retain a resolvable wood core at depth 3", () => {
  const plan = planOakV2(spec);
  const wood = (plan.node.children[0] as {
    children: readonly SceneryTaperedSweepClusterNode[];
  }).children;
  const halfVoxelDiagonal = Math.sqrt(3) * (.00625 / 8) / 2;
  for (let i = 0; i < wood.length; i++) {
    const node = wood[i], branch = plan.branches[i];
    const descriptor = {
      kind: "smooth-union-cluster" as const, primitiveId: i, materialId: 1, clusterReference: 0,
      center_m: node.place!.position!, lobeRadii_m: node.lobe,
      packing: { field: "tapered-sweep" as const, seed: node.seed, smoothRadius_m: node.smoothRadius,
        points: node.points.map(point => ({ position_m: point.position, radius_m: point.radius })) },
    };
    assert.ok(branch.baseRadius >= branch.tipRadius, `${branch.id}: reverse taper`);
    for (const point of node.points)
      assert.ok(point.radius > halfVoxelDiagonal);
    assert.ok(sampleSvoPrimitive(descriptor, branch.from).signedDistance_m < 0);
    assert.ok(sampleSvoPrimitive(descriptor, branch.to).signedDistance_m < 0);
  }
});
test("fractal twig depth adds binary forks without changing the scaffold", () => {
  const coarse = planOakV2({ ...spec, twigDepth: 0 });
  const fine = planOakV2({ ...spec, twigDepth: 3 });
  const coarsePads = (coarse.node.children[1] as {
    children: readonly SceneryRecursiveShapeNode[];
  }).children;
  const finePads = (fine.node.children[1] as {
    children: readonly SceneryRecursiveShapeNode[];
  }).children;
  assert.equal(finePads.length, coarsePads.length * 8);
  const fineScaffold = new Map(fine.branches.filter(b => !b.id.includes("/twig/")).map(b => [b.id, b]));
  for (const branch of coarse.branches) {
    const corresponding = fineScaffold.get(branch.id)!;
    assert.deepEqual(corresponding.from, branch.from);
    assert.deepEqual(corresponding.to, branch.to);
    assert.equal(corresponding.baseRadius, branch.baseRadius);
  }
  assert.throws(() => planOakV2({ ...spec, twigDepth: 4 }), RangeError);
});
test("oak canopy URL round-trip preserves twig-scale periods and density", async () => {
  const { defaultScene } = await import("../lib/core/model");
  const { sceneCanopyQuery, withSceneCanopyQuery, sceneCanopyPads } = await import("../lib/core/tree-canopy-controls");
  const scene = { ...defaultScene, scenery: { palettes: {}, nodes: [planOakV2(spec).node] } };
  const restored = withSceneCanopyQuery(scene, sceneCanopyQuery(scene));
  const before = sceneCanopyPads(scene, "oak"), after = sceneCanopyPads(restored, "oak");
  assert.equal(after.length, before.length);
  for (let i = 0; i < before.length; i++) {
    const a = before[i].form.density, b = after[i].form.density;
    assert.ok(Math.abs(a.clusterPeriod_m - b.clusterPeriod_m) / a.clusterPeriod_m < .002);
    assert.ok(Math.abs(a.detailWeight - b.detailWeight) < .001);
    assert.ok(Math.abs(a.interiorBias - b.interiorBias) < .001);
    assert.equal(a.dotSpacing_m, b.dotSpacing_m);
  }
});
test("Shape Lab refinement retains the v2 canopy dial scale", async () => {
  const { refineFoliageShape } = await import("../lib/core/voxel-scenery/recursive-foliage");
  const { canopyDials } = await import("../lib/core/tree-canopy-controls");
  const plan = planOakV2(spec);
  const pad = (plan.node.children[1] as {
    children: readonly SceneryRecursiveShapeNode[];
  }).children[0];
  const refined = refineFoliageShape(pad);
  for (const child of refined.children!) {
    assert.ok(child.tags?.includes("oak-v2"));
    assert.ok(Math.abs(canopyDials(child).clumpSize - canopyDials(pad).clumpSize) < 1e-12);
  }
});
test("renaming an oak changes identity without reshaping it", () => {
  const a = planOakV2(spec), b = planOakV2({ ...spec, key: "renamed" });
  const aw = (a.node.children[0] as {
    children: readonly SceneryTaperedSweepClusterNode[];
  }).children;
  const bw = (b.node.children[0] as {
    children: readonly SceneryTaperedSweepClusterNode[];
  }).children;
  for (let i = 0; i < aw.length; i++) {
    assert.deepEqual(aw[i].place, bw[i].place);
    assert.deepEqual(aw[i].points, bw[i].points);
  }
});
