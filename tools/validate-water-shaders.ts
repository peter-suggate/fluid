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
import { octreeSPGridPersistentMGPCGShader, octreeSPGridVCycleShader } from "../lib/webgpu-octree-spgrid-vcycle";
import { octreeFaceBandFusedAuthorityWGSL, octreeFaceBandWGSL } from "../lib/webgpu-octree-face-closest-point";
import { powerVelocityFusedAuthorityWGSL } from "../lib/webgpu-octree-power-velocity-prepass";
import {
  octreePowerVelocityPrepareFromFaceControlShader,
  octreePowerVelocityShader,
} from "../lib/webgpu-octree-power-velocity";
import { octreeCoarsePhiBootstrapShader } from "../lib/webgpu-octree-coarse-levelset";
import { octreeEnergyLedgerWGSL } from "../lib/webgpu-octree-energy-ledger";
import { octreePowerCoarseLevelSetShader } from "../lib/webgpu-octree-power-coarse-levelset";
import { octreePowerDescriptorShader } from "../lib/webgpu-octree-power-descriptor";
import { octreePowerBoundaryPhiShader, octreePowerFaceShader } from "../lib/webgpu-octree-power-faces";
import {
  octreePowerSolidExchangeShader,
  octreePowerSolidFaceShader,
  octreePowerSolidImpulseShader,
} from "../lib/webgpu-octree-power-solid-faces";
import { octreePowerTopologyShader } from "../lib/webgpu-octree-power-topology";
import { octreePowerOperatorShader } from "../lib/webgpu-octree-power-operator";
import { octreeSolidVertexSdfShader } from "../lib/webgpu-octree-solid-vertex-sdf";
import { octreeAnalyticBootstrapWorklistShader } from "../lib/webgpu-octree-analytic-bootstrap";
import { octreeDeterministicOwnerPageLifecycleShader } from "../lib/webgpu-octree-owner-pages";
import {
  sparseFineSeedCandidateResidencyShader,
  fineSeedCandidateCommitShader,
  fineSeedCandidateResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { octreeFineSeedAdapterShader, octreeFineSeedCandidateShader } from "../lib/webgpu-octree-fine-seed-adapter";
import { sparseSceneProxyVoxelizationShader } from "../lib/webgpu-sparse-scene-proxies";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { svoThickGlassWGSL } from "../lib/svo-thick-glass";
import { sparseVoxelTemporalAccumulatorShader } from "../lib/webgpu-svo-temporal-accumulator";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "../lib/webgpu-water-global-fine-tetra";
import { fineLevelSetFusedTransportPublicationWGSL } from "../lib/webgpu-octree-fine-levelset-transport";
import { fineLevelSetVolumeCorrectionWGSL } from "../lib/webgpu-octree-fine-levelset-volume";
import { fineLevelSetJFACPTWGSL } from "../lib/webgpu-octree-fine-levelset-redistance";
import { fineLevelSetSummaryWGSL } from "../lib/webgpu-octree-fine-levelset-summary";
import { makeFineLevelSetTopologyWGSL } from "../lib/webgpu-octree-fine-levelset-topology";
import { globalFineSurfaceClassificationShader } from "../lib/webgpu-water-global-fine-classify";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "global-fine-classification": globalFineSurfaceClassificationShader,
  "global-fine-classified-scan": globalFineClassifiedScanShader,
  ...Object.fromEntries(globalFineClassifiedEmitShaders.map((source, index) => [`global-fine-classified-tetra-${index}`, source])),
  "global-fine-classified-tetrahedra": globalFineClassifiedEmitShader,
  "global-fine-transport-publication": fineLevelSetFusedTransportPublicationWGSL,
  "global-fine-volume-correction": fineLevelSetVolumeCorrectionWGSL,
  "global-fine-jfa-cpt": fineLevelSetJFACPTWGSL,
  "global-fine-summary": fineLevelSetSummaryWGSL,
  "global-fine-topology": makeFineLevelSetTopologyWGSL(`
@group(0) @binding(9) var<storage,read> coarsePhi:array<f32>;
fn sampleCoarseOctreePhi(position:vec3f)->f32{return coarsePhi[u32(position.x)*0u];}`),
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
  "octree-spgrid-vcycle": octreeSPGridVCycleShader,
  "octree-spgrid-persistent-mgpcg": octreeSPGridPersistentMGPCGShader,
  "octree-face-band-closest-point": octreeFaceBandWGSL,
  "octree-face-band-fused-authority": octreeFaceBandFusedAuthorityWGSL,
  "octree-power-velocity-fused-authority": powerVelocityFusedAuthorityWGSL,
  "octree-power-velocity-face-authority-gate": octreePowerVelocityPrepareFromFaceControlShader,
  "octree-power-velocity-reconstruction": octreePowerVelocityShader,
  "octree-coarse-level-set-bootstrap": octreeCoarsePhiBootstrapShader,
  "octree-energy-ledger": octreeEnergyLedgerWGSL,
  "octree-power-coarse-level-set": octreePowerCoarseLevelSetShader,
  "octree-power-descriptor": octreePowerDescriptorShader,
  "octree-power-faces": octreePowerFaceShader,
  "octree-power-boundary-phi": octreePowerBoundaryPhiShader,
  "octree-power-topology": octreePowerTopologyShader,
  "octree-power-operator": octreePowerOperatorShader,
  "octree-power-solid-vertex-sdf": octreeSolidVertexSdfShader,
  "octree-power-solid-faces": octreePowerSolidFaceShader,
  "octree-power-solid-impulses": octreePowerSolidImpulseShader,
  "octree-power-solid-exchange": octreePowerSolidExchangeShader,
  "octree-analytic-bootstrap-worklist": octreeAnalyticBootstrapWorklistShader,
  "octree-owner-page-lifecycle": octreeDeterministicOwnerPageLifecycleShader,
  "octree-fine-seed-candidate-residency": fineSeedCandidateResidencyShader,
  "octree-sparse-fine-seed-candidate-residency": sparseFineSeedCandidateResidencyShader,
  "octree-fine-seed-candidate-commit": fineSeedCandidateCommitShader,
  "octree-fine-seed-adapter": octreeFineSeedAdapterShader,
  "octree-fine-seed-candidates": octreeFineSeedCandidateShader,
  "shared-eulerian-compute": legacyUniformComputeShader,
  "sparse-scene-proxy-voxelization": sparseSceneProxyVoxelizationShader,
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
