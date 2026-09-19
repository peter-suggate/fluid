/**
 * CPU-only structural census for the tall-air A/B. No GPU.
 *
 * Arm A and arm B are the same scene: the same cell size, the same absolute
 * reservoir box, the same floor footprint, the same dt. Only the container's
 * height changes, so only the empty air above the liquid grows.
 *
 * Run: node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/cpu-census.mts
 */
import { writeFileSync } from "node:fs";
import { sceneLatticeDimensions } from "../../../lib/core/scene-lattice-dimensions";
import { planUniformCM11aHierarchy } from "../../../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import { planUniformHostAllocation } from "../../../lib/methods/uniform/uniform-host-allocation";
import { uniformVolumeInitialPhi } from "../../../lib/methods/uniform/uniform-volume-initial";
import { tallAirScene, TALL_AIR_RESERVOIR_M } from "./tall-air-scene.mjs";


const product = (a: readonly number[]) => a.reduce((x, y) => x * y, 1);
/** `while (max(dims) > 1) dims = ceil(dims/2)` — webgpu-uniform-velocity-extrapolation.ts:234. */
function extensionLevels(dims: readonly number[]): number[][] {
  const levels: number[][] = [];
  let d = [...dims];
  while (Math.max(...d) > 1) { d = d.map((v) => Math.ceil(v / 2)); levels.push([...d]); }
  return levels;
}

const arms = [1, 2, 4, 8].map((multiple) => {
  const scene = tallAirScene(multiple);
  const dims = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const [nx, ny, nz] = dims;
  const cells = nx * ny * nz;
  const pressure = planUniformCM11aHierarchy([nx, ny, nz]);
  const extension = extensionLevels(dims);
  const host = planUniformHostAllocation(nx, ny, nz, "semi-lagrangian");
  const pressureBytes = 92 + 88 * pressure.levelDimensions.reduce(
    (sum, d) => sum + product(d.map((v) => v + 2)), 0) + 4 * (nx + 2) * (ny + 2) * (nz + 2);
  const extensionBytes = extension.reduce((sum, d) => sum + 32 * product(d), 0);
  const phiBytes = 8 * (nx + 1) * (ny + 1) * (nz + 1);
  const stencilBytes = 80 * cells;
  const conditioningWords = 2 * cells + 6 * Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4) + 16;
  // Initial liquid census on the real construction phi.
  let liquidCells = 0;
  let bandTiles = 0;
  const tiles = [Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)];
  if (cells <= 6_000_000) {
    const phi = uniformVolumeInitialPhi(scene, dims);
    const live = new Set<number>();
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      let minimum = Infinity;
      for (let dz = 0; dz <= 1; dz++) for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
        minimum = Math.min(minimum, phi[x + dx + (nx + 1) * (y + dy + (ny + 1) * (z + dz))]!);
      }
      if (minimum <= 0) { liquidCells++; live.add(Math.floor(x / 4) + tiles[0]! * (Math.floor(y / 4) + tiles[1]! * Math.floor(z / 4))); }
    }
    bandTiles = live.size;
  }
  return {
    multiple, sceneId: scene.sceneId, height_m: scene.container.height_m, dims, cells,
    cellSize_m: [scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz],
    pressure: {
      levelCount: pressure.levelCount, semiCoarsened: pressure.semiCoarsened,
      rejection: pressure.rejection, coarsestCells: pressure.coarsestCells,
      levelDimensions: pressure.levelDimensions,
      levelWorkgroups: pressure.levelDimensions.map((d) => product(d.map((v) => Math.ceil((v + 2) / 4)))),
    },
    extension: { levelCount: extension.length, levelDimensions: extension },
    tiles: { dims: tiles, total: product(tiles), initialLiquidTiles: bandTiles,
      initialLiquidTilePercent: 100 * bandTiles / product(tiles) },
    initialLiquidCells: liquidCells, initialLiquidPercent: 100 * liquidCells / cells,
    bytes: {
      hostAllocated: host.allocatedBytes, velocity: host.velocityBytes, scalar: host.scalarBytes,
      conditioning: conditioningWords * 4, pressureHierarchy: pressureBytes,
      extensionHierarchy: extensionBytes, vertexPhi: phiBytes, stencilArena: stencilBytes,
      majorPayload: host.allocatedBytes + pressureBytes + extensionBytes + phiBytes + stencilBytes,
      majorPayloadMiB: (host.allocatedBytes + pressureBytes + extensionBytes + phiBytes + stencilBytes) / 2 ** 20,
    },
  };
});

const out = { generated: new Date().toISOString(), reservoir_m: TALL_AIR_RESERVOIR_M, arms };
console.log(JSON.stringify(out, null, 2));
if (process.env.FLUID_TALL_OUT) writeFileSync(process.env.FLUID_TALL_OUT, JSON.stringify(out, null, 2));
