/** CPU pointer picking after warm-up.
 * Run: node --import tsx tools/benchmark-sculpt-picking.ts */
import { performance } from "node:perf_hooks";
import { createEmptyScene } from "../lib/core/empty-scene";
import { scenerySurfaceAt } from "../lib/core/editor-scenery";

const outputs = [];
for (const count of [64, 512, 2048]) {
  const scene = createEmptyScene({ extents_m: { x: 2, y: 2, z: 2 }, finestCellSize_m: .01 });
  scene.solidVoxels = Array.from({ length: count }, (_, i) => ({
    operation: "clear", affectsScenery: true, kind: "ellipsoid",
    minimum: [40 + i % 64, 77, 40 + Math.floor(i / 64)],
    maximumExclusive: [46 + i % 64, 83, 46 + Math.floor(i / 64)],
  }));
  scene.scenery = { ...scene.scenery!, nodes: [scene.scenery!.nodes[0]!, {
    kind: "box", id: "block", place: { units: "metres", position: { x: 0, y: .5, z: 0 } },
    halfSize: { x: .8, y: .3, z: .8 }, material: { colorLinear: [1, 1, 1] },
  }] };
  const rays = Array.from({ length: 128 }, (_, i) => ({
    origin: { x: -.57 + (i % 64) * .01, y: 1.5, z: -.57 + Math.floor((i * 17) % count / 64) * .01 },
    direction: { x: 0, y: -1, z: 0 },
  }));
  for (const ray of rays) scenerySurfaceAt(scene, ray);
  const runs: number[] = [];
  let checksum = 0;
  for (let repeat = 0; repeat < 5; repeat++) {
    const start = performance.now();
    for (const ray of rays) checksum += scenerySurfaceAt(scene, ray)?.position_m.y ?? 0;
    runs.push(performance.now() - start);
  }
  outputs.push({ cuts: count, queries: rays.length, median_ms: runs.sort((a, b) => a - b)[2], checksum });
}
console.log(JSON.stringify(outputs, null, 2));
