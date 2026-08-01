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
  octreeTechniqueFineLifecycleShader,
  octreeTechniqueLifecycleShader,
  octreeTechniqueTopologyShader,
} from "../lib/webgpu-octree-technique-overlay";
import { octreeTechniqueTetraValidityShader } from "../lib/webgpu-octree-technique-audit-overlay";
import {
  fluidBlastRadiusFloodShader,
  fluidBlastRadiusOverlayShader,
} from "../lib/webgpu-fluid-blast-radius";
import { fluidCellTraceGatherShader } from "../lib/webgpu-fluid-cell-trace";
import { secondaryParticleComputeShader, secondaryParticleOpticalShader } from "../lib/webgpu-secondary-particles";
import { sparseBrickDenseFieldShader } from "../lib/sparse-brick-octree";
import { octreeSparseBrickDebugPublicationShader } from "../lib/webgpu-octree-sparse-bricks";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import { octreeSPGridAccurateDispatchGateShader, octreeSPGridAccurateOperatorShader,
  octreeSPGridVCycleShader } from "../lib/webgpu-octree-spgrid-vcycle";
import { directStructuredVelocityPublicationWGSL } from "../lib/webgpu-octree-structured-velocity-gpu";
import { structuredBoundaryCoefficientWGSL } from "../lib/webgpu-octree-structured-boundary";
import { structuredVelocityDynamicsWGSL } from "../lib/webgpu-octree-structured-dynamics";
import { octreeCoarsePhiBootstrapShader } from "../lib/webgpu-octree-coarse-levelset";
import { octreePowerCoarseLevelSetShader } from "../lib/webgpu-octree-power-coarse-levelset";
import { octreePowerDescriptorShader } from "../lib/webgpu-octree-power-descriptor";
import { octreePowerTopologyShader } from "../lib/webgpu-octree-power-topology";
import { octreeSolidVertexSdfShader } from "../lib/webgpu-octree-solid-vertex-sdf";
import { octreeAnalyticBootstrapWorklistShader } from "../lib/webgpu-octree-analytic-bootstrap";
import { octreeDeterministicOwnerPageLifecycleShader } from "../lib/webgpu-octree-owner-pages";
import { octreeTopologyEpochWGSL } from "../lib/webgpu-octree-topology-epoch";
import {
  sparseFineSeedCandidateResidencyShader,
  fineSeedCandidateCommitShader,
  fineSeedCandidateResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { octreeFineSeedAdapterShader, octreeFineSeedCandidateShader } from "../lib/webgpu-octree-fine-seed-adapter";
import { sparseSceneProxyVoxelizationShader } from "../lib/webgpu-sparse-scene-proxies";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { svoThickGlassWGSL } from "../lib/svo-thick-glass";
import { globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "../lib/webgpu-water-global-fine-tetra";
import { structuredFineLevelSetTransportWGSL } from "../lib/webgpu-octree-fine-levelset-transport";
import { fineLevelSetVolumeCorrectionWGSL } from "../lib/webgpu-octree-fine-levelset-volume";
import { fineLevelSetJFACPTWGSL } from "../lib/webgpu-octree-fine-levelset-redistance";
import { fineLevelSetSummaryWGSL } from "../lib/webgpu-octree-fine-levelset-summary";
import { makeFineLevelSetTopologyWGSL } from "../lib/webgpu-octree-fine-levelset-topology";
import { globalFineSurfaceClassificationShader } from "../lib/webgpu-water-global-fine-classify";
import { octreeAirVelocitySupportPublicationWGSL } from "../lib/webgpu-octree-air-velocity-support-gpu";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "global-fine-classification": globalFineSurfaceClassificationShader,
  "global-fine-classified-scan": globalFineClassifiedScanShader,
  ...Object.fromEntries(globalFineClassifiedEmitShaders.map((source, index) => [`global-fine-classified-tetra-${index}`, source])),
  "global-fine-classified-tetrahedra": globalFineClassifiedEmitShader,
  "global-fine-structured-transport": structuredFineLevelSetTransportWGSL,
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
  "octree-technique-lifecycle-overlay": octreeTechniqueLifecycleShader,
  "octree-technique-fine-lifecycle-overlay": octreeTechniqueFineLifecycleShader,
  "octree-technique-tetra-validity-overlay": octreeTechniqueTetraValidityShader,
  "fluid-blast-radius-flood": fluidBlastRadiusFloodShader,
  "fluid-blast-radius-overlay": fluidBlastRadiusOverlayShader,
  "fluid-cell-trace-gather": fluidCellTraceGatherShader,
  "secondary-liquid-particle-optics": secondaryParticleOpticalShader,
  "secondary-liquid-particle-compute": secondaryParticleComputeShader,
  "sparse-brick-dense-field": sparseBrickDenseFieldShader,
  "sparse-brick-debug-publication": octreeSparseBrickDebugPublicationShader,
  "octree-projection": octreeProjectionShader,
  "octree-spgrid-vcycle": octreeSPGridVCycleShader,
  "octree-section63-operator": octreeSPGridAccurateOperatorShader,
  "octree-section63-dispatch-gate": octreeSPGridAccurateDispatchGateShader,
  "octree-direct-structured-velocity": directStructuredVelocityPublicationWGSL,
  "octree-structured-boundary": structuredBoundaryCoefficientWGSL,
  "octree-structured-dynamics": structuredVelocityDynamicsWGSL,
  "octree-air-velocity-support": octreeAirVelocitySupportPublicationWGSL,
  "octree-coarse-level-set-bootstrap": octreeCoarsePhiBootstrapShader,
  "octree-power-coarse-level-set": octreePowerCoarseLevelSetShader,
  "octree-power-descriptor": octreePowerDescriptorShader,
  "octree-power-topology": octreePowerTopologyShader,
  "octree-power-solid-vertex-sdf": octreeSolidVertexSdfShader,
  "octree-analytic-bootstrap-worklist": octreeAnalyticBootstrapWorklistShader,
  "octree-owner-page-lifecycle": octreeDeterministicOwnerPageLifecycleShader,
  "octree-coupled-topology-epoch": octreeTopologyEpochWGSL,
  "octree-fine-seed-candidate-residency": fineSeedCandidateResidencyShader,
  "octree-sparse-fine-seed-candidate-residency": sparseFineSeedCandidateResidencyShader,
  "octree-fine-seed-candidate-commit": fineSeedCandidateCommitShader,
  "octree-fine-seed-adapter": octreeFineSeedAdapterShader,
  "octree-fine-seed-candidates": octreeFineSeedCandidateShader,
  "sparse-scene-proxy-voxelization": sparseSceneProxyVoxelizationShader,
  "sparse-voxel-dry-scene": svoDrySceneShader,
  "sparse-voxel-thick-glass-library": svoThickGlassWGSL,
};
const directory = mkdtempSync(join(tmpdir(), "fluid-water-wgsl-"));
// Naga still rejects the standardized `enable subgroups` directive. These
// modules are compiled by Dawn in the focused backend tests instead; silently
// stripping the directive here would validate a shader that production never runs.
const dawnValidated = new Set(["global-fine-jfa-cpt"]);
try {
  for (const [name, source] of Object.entries(shaders)) {
    if (dawnValidated.has(name)) {
      console.log(`deferred ${name} to Dawn subgroup validation`);
      continue;
    }
    const path = join(directory, `${name}.wgsl`);
    writeFileSync(path, source);
    const result = spawnSync(naga, [path], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(`${name}:\n${result.stderr || result.stdout}`);
    console.log(`validated ${name}`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
