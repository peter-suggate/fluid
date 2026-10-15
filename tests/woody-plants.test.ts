import assert from "node:assert/strict";
import test from "node:test";
import type { SceneryNode, SceneryRecursiveShapeNode, SceneryTaperedSweepClusterNode } from "../lib/core/scenery-graph";
import { planWoodyPlant, WOODY_PLANT_SPECIES, type WoodyPlantSpec } from "../lib/core/voxel-scenery/woody-plants";
import { oakFoliageDescriptor } from "../lib/core/voxel-scenery/oak-v2";
import { sampleSvoPrimitive, validateSvoClusterPacking } from "../lib/svo/contracts/svo-primitive-abi";
import { createHeroGardenHoseStressScene, heroGardenStressStands } from "../lib/core/hero-garden-stress-scene";
import { themeParkScenery } from "../lib/core/theme-park-scenery";
import { parseScene, serializeScene } from "../lib/core/model";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";

const material = { colorLinear: [.9, .9, .9] as const, surface: "foliage" as const };
const spec: WoodyPlantSpec = { key: "specimen", species: "birch", seed: 4258, scale_m: 1, bark: material, foliage: material };
const parts = (node: SceneryNode): SceneryNode[] => node.kind === "group" ? node.children.flatMap(parts) : [node];
for (const species of WOODY_PLANT_SPECIES) {
  test(`${species}: seeded, connected wood and seated porous foliage across scales`, () => {
    for (const seed of [0, 4258, -2147483648]) for (const scale_m of [.19, 1, 3.7]) for (const detail of ["full", "miniature"] as const) {
      const input = { ...spec, species, seed, scale_m, detail };
      const plan = planWoodyPlant(input);
      assert.deepEqual(plan, planWoodyPlant(input));
      assert.notDeepEqual(plan, planWoodyPlant({ ...input, seed: seed + 1 }));
      const nodes = parts(plan.node);
      assert.equal(new Set(nodes.map(node => node.id)).size, nodes.length);
      assert.ok(nodes.length < 650, "bounded cost per specimen");
      const seen = new Map<string, typeof plan.branches[number]>();
      for (const branch of plan.branches) {
        assert.ok(branch.baseRadius >= branch.tipRadius && branch.tipRadius > 0);
        if (branch.parent) {
          const parent = seen.get(branch.parent)!;
          assert.ok(parent, "acyclic parent precedes child");
          assert.deepEqual(branch.from, parent.to, "fork has no gap");
          assert.ok(branch.baseRadius <= parent.tipRadius);
        } else assert.equal(branch.from.y, 0, "basal stems meet ground");
        assert.ok(Math.hypot(branch.to.x - branch.from.x, branch.to.y - branch.from.y, branch.to.z - branch.from.z)
          > Math.abs(branch.baseRadius - branch.tipRadius));
        seen.set(branch.id, branch);
      }
      for (const node of nodes as (SceneryRecursiveShapeNode | SceneryTaperedSweepClusterNode)[]) {
        if (node.kind === "recursive-shape") {
          const descriptor = oakFoliageDescriptor(node), tip = node.place!.position!;
          validateSvoClusterPacking(descriptor.packing!, descriptor.lobeRadii_m);
          assert.ok(plan.branches.some(branch => branch.to === tip));
          assert.ok(sampleSvoPrimitive(descriptor, tip).signedDistance_m < 0, "leaf mass touches shoot");
        } else validateSvoClusterPacking({ field: "tapered-sweep", seed: node.seed, smoothRadius_m: node.smoothRadius,
          points: node.points.map(point => ({ position_m: point.position, radius_m: point.radius })) }, node.lobe);
      }
    }
  });
}

test("invalid plant inputs fail before publishing geometry", () => {
  for (const scale_m of [0, -1, NaN, Infinity]) assert.throws(() => planWoodyPlant({ ...spec, scale_m }), RangeError);
  assert.throws(() => planWoodyPlant({ ...spec, seed: .5 }), RangeError);
  assert.throws(() => planWoodyPlant({ ...spec, species: "unknown" as WoodyPlantSpec["species"] }), RangeError);
});

test("dense bushes fill their lower and middle crown while retaining air between leaves", () => {
  for (const species of ["boxwood", "hebe"] as const) for (const detail of ["full", "miniature"] as const) {
    const plan = planWoodyPlant({ ...spec, species, detail });
    const foliage = parts(plan.node).filter((node): node is SceneryRecursiveShapeNode => node.kind === "recursive-shape")
      .map(oakFoliageDescriptor);
    let occupied = 0, samples = 0;
    // Probe actual leaf density throughout the interior, not the enclosing
    // ellipsoids: an empty shell of tufts must fail.
    for (let x = -4; x <= 4; x++) for (let z = -4; z <= 4; z++) for (let y = 0; y < 5; y++) {
      const point = { x: x * .05, y: .12 + y * .05, z: z * .05 };
      if (foliage.some(field => sampleSvoPrimitive(field, point).signedDistance_m < 0)) occupied++;
      samples++;
    }
    assert.ok(occupied / samples > .35, `${species}/${detail}: crown must carry layered leaf mass`);
    // Overlapping sprays can close the crown's interior. Each individual
    // spray still needs holes within its envelope, rather than a solid pad.
    let leafMass = 0, gaps = 0;
    for (const field of foliage) for (let i = 0; i < 24; i++) {
      const y = 1 - 2 * (i + .5) / 24, angle = i * 2.399963, r = Math.sqrt(1 - y * y);
      const point = { x: field.center_m.x + Math.cos(angle) * r * field.lobeRadii_m.x * .65,
        y: field.center_m.y + y * field.lobeRadii_m.y * .65,
        z: field.center_m.z + Math.sin(angle) * r * field.lobeRadii_m.z * .65 };
      if (sampleSvoPrimitive(field, point).signedDistance_m < 0) leafMass++; else gaps++;
    }
    assert.ok(leafMass > 0 && gaps > 0, `${species}/${detail}: sprays retain leaf mass and real gaps`);
  }
});

test("garden x10 and theme park include every species; garden retains stable, grounded stands", () => {
  const garden = createHeroGardenHoseStressScene();
  for (const graph of [garden.scenery!, themeParkScenery()]) {
    for (const species of WOODY_PLANT_SPECIES) assert.ok(graph.nodes.some(node => node.tags?.includes(species)), species);
  }
  assert.deepEqual(heroGardenStressStands(18), heroGardenStressStands(81).slice(0, 18));
  for (const node of garden.scenery!.nodes.filter(node => node.tags?.includes("woody-plant"))) {
    assert.equal(node.place?.anchor, "terrain");
    assert.deepEqual(node.place?.ground, [node.place?.position?.x, node.place?.position?.z]);
  }
  const restored = parseScene(serializeScene(garden));
  assert.equal(JSON.stringify(restored.scenery), JSON.stringify(garden.scenery));
  const catalog = buildEnvironmentProxyCatalog(restored, "garden");
  assert.ok(catalog.primitives.filter(p => p.kind === "cluster").length < 4096,
    "miniature specimens must fit the live aggregate arena with the original oak");
  assert.equal(new Set(catalog.primitives.map(p => p.key)).size, catalog.primitives.length);
  const woody = catalog.primitives.filter(p => p.key.startsWith("garden/stress/") && p.tags?.includes("woody-plant"));
  assert.ok(woody.length > 0);
  for (const primitive of woody) {
    assert.ok(primitive.aabb_m.min.x >= -.9 && primitive.aabb_m.max.x <= .9);
    assert.ok(primitive.aabb_m.min.z >= -.6 && primitive.aabb_m.max.z <= .6);
  }
});
