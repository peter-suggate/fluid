/** Independent midpoint integration of the analytic foliage at UI voxel sizes.
 * This measures sampling convergence; it is not a substitute for GPU render QA.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { planOakV2, oakFoliageDescriptor } from "../lib/core/voxel-scenery/oak-v2";
import { sampleSvoPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
import type { SceneryRecursiveShapeNode } from "../lib/core/scenery-graph";
import assert from "node:assert/strict";
const plan = planOakV2({ key: "oak", seed: 4258, scale_m: 0.85, bark: { palette: "clay", value: .955 }, foliage: { palette: "clay", value: .975 } });
const pads = (plan.node.children[1] as {
  children: readonly SceneryRecursiveShapeNode[];
}).children;
// Disjoint locations and seeds, including opposite boughs, fixed across levels.
const probes = [0, 95, 192, 383, 576, 767].map(i => oakFoliageDescriptor(pads[i]));
const levels = [];
for (let depth = 0; depth <= 4; depth++) {
  const h = .00625 / 2 ** depth;
  let volume = 0, occupied = 0, samples = 0;
  for (const d of probes) {
    const r = d.lobeRadii_m, c = d.center_m;
    for (let iz = Math.floor((c.z - r.z) / h); iz <= Math.ceil((c.z + r.z) / h); iz++)
      for (let iy = Math.floor((c.y - r.y) / h); iy <= Math.ceil((c.y + r.y) / h); iy++)
        for (let ix = Math.floor((c.x - r.x) / h); ix <= Math.ceil((c.x + r.x) / h); ix++) {
          const point = { x: (ix + .5) * h, y: (iy + .5) * h, z: (iz + .5) * h };
          if (((point.x - c.x) / r.x) ** 2 + ((point.y - c.y) / r.y) ** 2 + ((point.z - c.z) / r.z) ** 2 > 1)
            continue;
          samples++;
          if (sampleSvoPrimitive(d, point).signedDistance_m < 0)
            occupied++;
        }
  }
  volume = occupied * h ** 3;
  levels.push({ depth, cell_m: h, volume_m3: volume, occupied, samples });
  console.log(levels.at(-1));
}
const reference = levels.at(-1)!.volume_m3;
const result = levels.map(level => ({ ...level, relativeVolumeError: Math.abs(level.volume_m3 - reference) / reference }));
assert.ok(result[3].relativeVolumeError < .025, "depth 3 must converge within 2.5% of the half-cell reference");
assert.ok(result[3].relativeVolumeError < result[0].relativeVolumeError, "refinement must improve sampled volume");
mkdirSync("artifacts/oak-v2/fractal", { recursive: true });
writeFileSync("artifacts/oak-v2/fractal/convergence.json", JSON.stringify(result, null, 2));
console.log(result);
