import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  causticShader,
  compositeShader,
  extractionPrepareShader,
  sceneShader,
  surfaceExtractionShader,
  surfaceRasterShader
} from "../lib/webgpu-water-pipeline";
import { gridOverlayShader } from "../lib/webgpu-grid-overlay";
import { secondaryParticleComputeShader, secondaryParticleCorrectionShader, secondaryParticleOpticalShader } from "../lib/webgpu-secondary-particles";
import { sparseBrickDenseFieldShader } from "../lib/sparse-brick-octree";
import { octreeSparseBrickDebugPublicationShader } from "../lib/webgpu-octree-sparse-bricks";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { sparseSceneProxyVoxelizationShader } from "../lib/webgpu-sparse-scene-proxies";
import { sparseSurfaceDynamicsShader, sparseSurfaceFieldShader, sparseSurfaceResidencyShader } from "../lib/webgpu-sparse-surface-band";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { sparseVoxelTemporalAccumulatorShader } from "../lib/webgpu-svo-temporal-accumulator";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "extraction-prepare": extractionPrepareShader,
  "surface-raster": surfaceRasterShader,
  caustics: causticShader,
  scene: sceneShader,
  composite: compositeShader,
  "grid-overlay": gridOverlayShader,
  "secondary-liquid-particle-optics": secondaryParticleOpticalShader,
  "secondary-liquid-particle-compute": secondaryParticleComputeShader,
  "secondary-liquid-particle-correction": secondaryParticleCorrectionShader,
  "sparse-brick-dense-field": sparseBrickDenseFieldShader,
  "sparse-brick-debug-publication": octreeSparseBrickDebugPublicationShader,
  "octree-projection": octreeProjectionShader,
  "sparse-scene-proxy-voxelization": sparseSceneProxyVoxelizationShader,
  "sparse-surface-residency": sparseSurfaceResidencyShader,
  "sparse-surface-field": sparseSurfaceFieldShader,
  "sparse-surface-dynamics": sparseSurfaceDynamicsShader,
  "sparse-voxel-dry-scene": svoDrySceneShader,
  "sparse-voxel-temporal-accumulation": sparseVoxelTemporalAccumulatorShader,
};
const directory = mkdtempSync(join(tmpdir(), "fluid-water-wgsl-"));
try {
  for (const [name, source] of Object.entries(shaders)) {
    const path = join(directory, `${name}.wgsl`);
    writeFileSync(path, source);
    const result = spawnSync(naga, [path], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(`${name}:\n${result.stderr || result.stdout}`);
    console.log(`validated ${name}`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
