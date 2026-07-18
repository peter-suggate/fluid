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
import { secondaryParticleComputeShader, secondaryParticleOpticalShader, secondaryParticleRenderShader } from "../lib/webgpu-secondary-particles";
import { sparseBrickDenseFieldShader } from "../lib/sparse-brick-octree";
import { octreeSparseBrickDebugPublicationShader } from "../lib/webgpu-octree-sparse-bricks";
import { sparseSceneProxyVoxelizationShader } from "../lib/webgpu-sparse-scene-proxies";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "extraction-prepare": extractionPrepareShader,
  "surface-raster": surfaceRasterShader,
  caustics: causticShader,
  scene: sceneShader,
  composite: compositeShader,
  "grid-overlay": gridOverlayShader,
  "secondary-liquid-particles": secondaryParticleRenderShader,
  "secondary-liquid-particle-optics": secondaryParticleOpticalShader,
  "secondary-liquid-particle-compute": secondaryParticleComputeShader,
  "sparse-brick-dense-field": sparseBrickDenseFieldShader,
  "sparse-brick-debug-publication": octreeSparseBrickDebugPublicationShader,
  "sparse-scene-proxy-voxelization": sparseSceneProxyVoxelizationShader
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
