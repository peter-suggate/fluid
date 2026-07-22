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
import {
  octreeTechniqueFaceShader,
  octreeTechniqueFineLifecycleShader,
  octreeTechniqueLifecycleShader,
  octreeTechniqueSection5FaceBandShader,
  octreeTechniqueTopologyShader,
} from "../lib/webgpu-octree-technique-overlay";
import { octreeTechniqueOperatorAuditShader, octreeTechniqueTetraValidityShader } from "../lib/webgpu-octree-technique-audit-overlay";
import { secondaryParticleComputeShader, secondaryParticleCorrectionShader, secondaryParticleOpticalShader } from "../lib/webgpu-secondary-particles";
import { sparseBrickDenseFieldShader } from "../lib/sparse-brick-octree";
import { octreeSparseBrickDebugPublicationShader } from "../lib/webgpu-octree-sparse-bricks";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { octreeMGPCGShader } from "../lib/webgpu-octree-mgpcg";
import { octreeFaceBandWGSL } from "../lib/webgpu-octree-face-fast-march";
import { octreePowerVelocityPrepareFromFaceControlShader, octreePowerVelocityShader } from "../lib/webgpu-octree-power-velocity";
import { octreePowerCoarseLevelSetShader } from "../lib/webgpu-octree-power-coarse-levelset";
import { octreePowerFaceTransferShader } from "../lib/webgpu-octree-power-face-transfer";
import { octreePowerBoundaryPhiShader, octreePowerFaceShader } from "../lib/webgpu-octree-power-faces";
import { octreePowerSolidFaceShader, octreePowerSolidImpulseShader } from "../lib/webgpu-octree-power-solid-faces";
import { octreeSolidVertexSdfShader } from "../lib/webgpu-octree-solid-vertex-sdf";
import { octreeAnalyticBootstrapWorklistShader } from "../lib/webgpu-octree-analytic-bootstrap";
import {
  sparseSurfaceCandidateResidencyShader,
  surfaceCandidateCommitShader,
  surfaceCandidateResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { octreeSurfaceAdapterShader } from "../lib/webgpu-octree-surface-adapter";
import { octreeSurfacePageShader } from "../lib/webgpu-octree-surface-pages";
import { sparseSceneProxyVoxelizationShader } from "../lib/webgpu-sparse-scene-proxies";
import { sparseSurfaceDynamicsShader, sparseSurfaceFieldShader, sparseSurfaceResidencyShader } from "../lib/webgpu-sparse-surface-band";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { svoThickGlassWGSL } from "../lib/svo-thick-glass";
import { sparseVoxelTemporalAccumulatorShader } from "../lib/webgpu-svo-temporal-accumulator";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { globalFineClassifiedCountShader, globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "../lib/webgpu-water-global-fine-tetra";
import { fineLevelSetGPUQueryTransportWGSL } from "../lib/webgpu-octree-fine-levelset-transport";
import { fineLevelSetVolumeCorrectionWGSL } from "../lib/webgpu-octree-fine-levelset-volume";
import { fineLevelSetRedistanceWGSL } from "../lib/webgpu-octree-fine-levelset-redistance";
import { globalFineSurfaceClassificationShader } from "../lib/webgpu-water-global-fine-classify";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "global-fine-classification": globalFineSurfaceClassificationShader,
  "global-fine-classified-count": globalFineClassifiedCountShader,
  "global-fine-classified-scan": globalFineClassifiedScanShader,
  ...Object.fromEntries(globalFineClassifiedEmitShaders.map((source, index) => [`global-fine-classified-tetra-${index}`, source])),
  "global-fine-classified-tetrahedra": globalFineClassifiedEmitShader,
  "global-fine-query-transport": fineLevelSetGPUQueryTransportWGSL,
  "global-fine-volume-correction": fineLevelSetVolumeCorrectionWGSL,
  "global-fine-fast-march": fineLevelSetRedistanceWGSL,
  "extraction-prepare": extractionPrepareShader,
  "surface-raster": surfaceRasterShader,
  caustics: causticShader,
  scene: sceneShader,
  composite: compositeShader,
  "grid-overlay": gridOverlayShader,
  "octree-technique-topology-overlay": octreeTechniqueTopologyShader,
  "octree-technique-face-overlay": octreeTechniqueFaceShader,
  "octree-technique-section5-face-band-overlay": octreeTechniqueSection5FaceBandShader,
  "octree-technique-lifecycle-overlay": octreeTechniqueLifecycleShader,
  "octree-technique-fine-lifecycle-overlay": octreeTechniqueFineLifecycleShader,
  "octree-technique-operator-audit-overlay": octreeTechniqueOperatorAuditShader,
  "octree-technique-tetra-validity-overlay": octreeTechniqueTetraValidityShader,
  "secondary-liquid-particle-optics": secondaryParticleOpticalShader,
  "secondary-liquid-particle-compute": secondaryParticleComputeShader,
  "secondary-liquid-particle-correction": secondaryParticleCorrectionShader,
  "sparse-brick-dense-field": sparseBrickDenseFieldShader,
  "sparse-brick-debug-publication": octreeSparseBrickDebugPublicationShader,
  "octree-projection": octreeProjectionShader,
  "octree-pcg-section43-hybrid": octreeMGPCGShader,
  "octree-face-band-fast-march": octreeFaceBandWGSL,
  "octree-power-velocity-face-authority-gate": octreePowerVelocityPrepareFromFaceControlShader,
  "octree-power-velocity-reconstruction": octreePowerVelocityShader,
  "octree-power-coarse-level-set": octreePowerCoarseLevelSetShader,
  "octree-power-face-transfer": octreePowerFaceTransferShader,
  "octree-power-faces": octreePowerFaceShader,
  "octree-power-boundary-phi": octreePowerBoundaryPhiShader,
  "octree-power-solid-vertex-sdf": octreeSolidVertexSdfShader,
  "octree-power-solid-faces": octreePowerSolidFaceShader,
  "octree-power-solid-impulses": octreePowerSolidImpulseShader,
  "octree-analytic-bootstrap-worklist": octreeAnalyticBootstrapWorklistShader,
  "octree-surface-candidate-residency": surfaceCandidateResidencyShader,
  "octree-sparse-surface-candidate-residency": sparseSurfaceCandidateResidencyShader,
  "octree-surface-candidate-commit": surfaceCandidateCommitShader,
  "octree-surface-adapter": octreeSurfaceAdapterShader,
  "octree-surface-pages": octreeSurfacePageShader,
  "shared-eulerian-compute": legacyUniformComputeShader,
  "sparse-scene-proxy-voxelization": sparseSceneProxyVoxelizationShader,
  "sparse-surface-residency": sparseSurfaceResidencyShader,
  "sparse-surface-field": sparseSurfaceFieldShader,
  "sparse-surface-dynamics": sparseSurfaceDynamicsShader,
  "sparse-voxel-dry-scene": svoDrySceneShader,
  "sparse-voxel-thick-glass-library": svoThickGlassWGSL,
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
