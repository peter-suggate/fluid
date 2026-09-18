import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { uniformVolumeInitialPhi } from "../lib/methods/uniform/uniform-volume-initial";
import { planUniformHostAllocation } from "../lib/methods/uniform/uniform-host-allocation";
import { planUniformCM11aHierarchy } from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid";

// CPU-only initial geometry census. Bands are sensitivity scenarios, NOT a
// certified transport/velocity/pressure support closure or a speed benchmark.
const scenes = process.argv.slice(2);
if (!scenes.length) scenes.push("minimal-power-dam-break-32", "minimal-power-dam-break-64", "large-power-dam-break");
const product = (a: readonly number[]) => a.reduce((x, y) => x * y, 1);
const memory = [32, 64, 128, 256].map(n => {
  const host = planUniformHostAllocation(n, n, n, "maccormack");
  const hierarchy = planUniformCM11aHierarchy([n, n, n]);
  const pressure = 92 + 88 * hierarchy.levelDimensions.reduce((sum, d) => sum + product(d.map(v => v + 2)), 0) + 4 * (n + 2) ** 3;
  let extensionHierarchy = 0;
  for (let side = n / 2; side >= 1; side /= 2) extensionHierarchy += 32 * side ** 3;
  return { n, stencilMiB: 80 * n ** 3 / 2 ** 20,
    majorPayloadMiB: (host.allocatedBytes + pressure + extensionHierarchy + 8 * (n + 1) ** 3 + 80 * n ** 3) / 2 ** 20 };
});
const results = scenes.map(id => {
  const scene = sceneDocument(getSceneDefinition(id));
  const dims = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (product(dims) > 4_000_000) throw new Error(`${id}: CPU census capped at four million cells`);
  const [nx, ny, nz] = dims;
  const h = Math.max(scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz);
  const phi = uniformVolumeInitialPhi(scene, dims);
  const bands = [0, 4, 8, 16].map(band => {
    const tiles = [4, 8, 16].map(B => ({ B, dims: dims.map(n => Math.ceil(n / B)), keys: new Set<number>() }));
    let cells = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      let minimum = Infinity;
      for (let dz = 0; dz <= 1; dz++) for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
        minimum = Math.min(minimum, phi[x + dx + (nx + 1) * (y + dy + (ny + 1) * (z + dz))]!);
      }
      if (minimum > band * h) continue;
      cells++;
      for (const t of tiles) t.keys.add(Math.floor(x / t.B) + t.dims[0]! * (Math.floor(y / t.B) + t.dims[1]! * Math.floor(z / t.B)));
    }
    return { bandCells: band, geometricCells: cells, tiles: tiles.map(t => ({ B: t.B,
      active: t.keys.size, total: product(t.dims), activePercent: 100 * t.keys.size / product(t.dims) })) };
  });
  return { id, dims, bands };
});
console.log(JSON.stringify({ notes: [
  "Initial phi only; includes liquid interiors and cells with any vertex phi <= band * max(cell size).",
  "Ignores solid clipping, independent V, dynamic sources, characteristic reach, normalization dependencies and velocity hierarchy dependencies.",
  "No band is claimed sufficient. Counts are geometry sensitivity only; no GPU timing or peak residency measurement.",
  "Major payload uses the host allocation plan plus pressure, extension hierarchy, phi and stencils. Excludes small buffers, terrain, rigid resources, driver padding and audits; not an exact VRAM measurement.",
  "Memory includes four velocity fields and two transport fields even under semi-Lagrangian, matching current constructor allocation."
], memory, results }, null, 2));
