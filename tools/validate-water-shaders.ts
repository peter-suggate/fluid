import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  causticShader,
  compositeShader,
  extractionPrepareShader,
  surfaceExtractionShader,
  surfaceRasterShader,
  surfaceWireframeShader,
} from "../lib/core/webgpu-water-pipeline";
import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";
import {
  octreeTechniqueAdaptiveVelocityShader,
  octreeTechniqueFaceShader,
  octreeTechniqueFineLifecycleShader,
  octreeTechniqueLifecycleShader,
  octreeTechniqueStructuredShader,
  octreeTechniqueTopologyShader,
} from "../lib/methods/octree-shared/webgpu-octree-technique-overlay";
import { octreeTechniqueTetraValidityShader } from "../lib/methods/octree-shared/webgpu-octree-technique-audit-overlay";
import {
  fluidBlastRadiusFloodShader,
  fluidBlastRadiusOverlayShader,
} from "../lib/core/webgpu-fluid-blast-radius";
import { fluidCellTraceGatherShader } from "../lib/core/webgpu-fluid-cell-trace";
import { secondaryParticleComputeShader, secondaryParticleOpticalShader } from "../lib/core/webgpu-secondary-particles";
import { sparseBrickDenseFieldShader } from "../lib/svo/sparse-brick-octree";
import { octreeLosassoProjectionShader } from "../lib/methods/losasso/octree-losasso-projection.wgsl";
import { octreePowerProjectionShader } from "../lib/methods/power/octree-power-projection.wgsl";
import { octreeSPGridAccurateDispatchGateShader, octreeSPGridAccurateOperatorShader,
  octreeSPGridVCycleShader } from "../lib/methods/power/webgpu-octree-spgrid-vcycle";
import { directStructuredVelocityPublicationWGSL } from "../lib/methods/power/webgpu-octree-structured-velocity-gpu";
import { structuredBoundaryCoefficientWGSL } from "../lib/methods/power/webgpu-octree-structured-boundary";
import { structuredVelocityDynamicsWGSL } from "../lib/methods/power/webgpu-octree-structured-dynamics";
import { octreeCoarsePhiBootstrapShader } from "../lib/methods/octree-shared/webgpu-octree-coarse-levelset";
import { octreePowerCoarseLevelSetShader } from "../lib/methods/power/webgpu-octree-power-coarse-levelset";
import { octreePowerDescriptorShader } from "../lib/methods/power/webgpu-octree-power-descriptor";
import { octreePowerTopologyShader } from "../lib/methods/power/webgpu-octree-power-topology";
import { octreeSolidVertexSdfShader } from "../lib/methods/octree-shared/webgpu-octree-solid-vertex-sdf";
import { octreeAnalyticBootstrapWorklistShader } from "../lib/methods/octree-shared/webgpu-octree-analytic-bootstrap";
import { octreeDeterministicOwnerPageLifecycleShader } from "../lib/methods/octree-shared/webgpu-octree-owner-pages";
import { octreeTopologyEpochWGSL } from "../lib/methods/octree-shared/webgpu-octree-topology-epoch";
import {
  sparseFineSeedCandidateResidencyShader,
  fineSeedCandidateCommitShader,
  fineSeedCandidateResidencyShader,
} from "../lib/core/webgpu-fluid-brick-residency";
import { octreeFineSeedAdapterShader, octreeFineSeedCandidateShader } from "../lib/methods/octree-shared/webgpu-octree-fine-seed-adapter";
import { sparseSceneProxyVoxelizationShader } from "../lib/core/webgpu-sparse-scene-proxies";
import { svoDrySceneShader } from "../lib/svo/webgpu-svo-dry-scene";
import { svoThickGlassWGSL } from "../lib/svo/svo-thick-glass";
import { globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedIndirectScanShader, globalFineClassifiedScanShader } from "../lib/core/webgpu-water-global-fine-tetra";
import { structuredFineLevelSetTransportWGSL } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-transport";
import { fineLevelSetVolumeCorrectionWGSL } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-volume";
import { fineLevelSetJFACPTWGSL } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-redistance";
import { fineLevelSetSummaryWGSL } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-summary";
import { makeFineLevelSetTopologyWGSL } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-topology";
import { globalFineSurfaceClassificationShader } from "../lib/core/webgpu-water-global-fine-classify";
import { octreeAirVelocitySupportPublicationWGSL } from "../lib/methods/power/webgpu-octree-air-velocity-support-gpu";
import { octreeLosassoSurfaceGraphWGSL } from "../lib/methods/losasso/webgpu-octree-losasso-surface-graph.wgsl";
import {
  octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
  octreeLosassoAdaptivePhiBacktraceWGSL,
  octreeLosassoAdaptivePhiCommitWGSL,
  octreeLosassoAdaptivePhiEvidenceWGSL,
  octreeLosassoAdaptivePhiGhostWGSL,
  octreeLosassoAdaptivePhiHandoffWGSL,
  octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
  octreeLosassoAdaptivePhiRedistanceBtoAWGSL,
  octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
  octreeLosassoAdaptivePhiScheduleWGSL,
  octreeLosassoAdaptivePhiTransportWGSL,
  octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
  octreeLosassoAdaptivePhiWorklistConstrainedWGSL,
  octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  octreeLosassoAdaptivePhiWorklistFinalizeWGSL,
  octreeLosassoAdaptivePhiWorklistIndependentWGSL,
  octreeLosassoAdaptivePhiWorklistInflowWGSL,
  octreeLosassoAdaptivePhiWorklistPrepareWGSL,
  octreeLosassoAdaptivePhiWorklistProjectWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
  octreeLosassoAdaptivePhiWorklistReceiptWGSL,
  octreeLosassoAdaptivePhiWGSL,
} from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-phi.wgsl";
import {
  octreeLosassoAdaptiveVelocitySamplerWGSL,
  octreeLosassoAdaptiveVelocityWGSL,
} from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-velocity.wgsl";
import { octreeLosassoAdaptiveMassWGSL } from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-mass.wgsl";
import { makeOctreeLosassoAdaptiveDynamicsWGSL } from "../lib/methods/losasso/webgpu-octree-losasso-dynamics.wgsl";
import { SPARSE_CM12_LENSES } from "../lib/methods/adaptive-mass/sparse-cm12-stage-lenses";
import { stageLensProgramWGSL } from "../lib/core/webgpu-stage-lens-overlay";

const naga = process.env.NAGA ?? "naga";
const shaders = {
  "surface-extraction": surfaceExtractionShader,
  "global-fine-classification": globalFineSurfaceClassificationShader,
  "global-fine-classified-scan": globalFineClassifiedScanShader,
  "global-fine-classified-indirect-scan": globalFineClassifiedIndirectScanShader,
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
  "surface-wireframe": surfaceWireframeShader,
  caustics: causticShader,
  composite: compositeShader,
  "grid-overlay": gridOverlayShader,
  "octree-technique-topology-overlay": octreeTechniqueTopologyShader,
  "octree-technique-face-overlay": octreeTechniqueFaceShader,
  "octree-technique-structured-overlay": octreeTechniqueStructuredShader,
  "octree-technique-adaptive-velocity-overlay": octreeTechniqueAdaptiveVelocityShader,
  "octree-technique-lifecycle-overlay": octreeTechniqueLifecycleShader,
  "octree-technique-fine-lifecycle-overlay": octreeTechniqueFineLifecycleShader,
  "octree-technique-tetra-validity-overlay": octreeTechniqueTetraValidityShader,
  "fluid-blast-radius-flood": fluidBlastRadiusFloodShader,
  "fluid-blast-radius-overlay": fluidBlastRadiusOverlayShader,
  "fluid-cell-trace-gather": fluidCellTraceGatherShader,
  "secondary-liquid-particle-optics": secondaryParticleOpticalShader,
  "secondary-liquid-particle-compute": secondaryParticleComputeShader,
  "sparse-brick-dense-field": sparseBrickDenseFieldShader,
  "octree-projection-power": octreePowerProjectionShader,
  "octree-projection-losasso": octreeLosassoProjectionShader,
  "octree-spgrid-vcycle": octreeSPGridVCycleShader,
  "octree-section63-operator": octreeSPGridAccurateOperatorShader,
  "octree-section63-dispatch-gate": octreeSPGridAccurateDispatchGateShader,
  "octree-direct-structured-velocity": directStructuredVelocityPublicationWGSL,
  "octree-structured-boundary": structuredBoundaryCoefficientWGSL,
  "octree-structured-dynamics": structuredVelocityDynamicsWGSL,
  "octree-air-velocity-support": octreeAirVelocitySupportPublicationWGSL,
  "octree-losasso-surface-graph": octreeLosassoSurfaceGraphWGSL,
  "octree-losasso-adaptive-phi": octreeLosassoAdaptivePhiWGSL,
  "octree-losasso-adaptive-phi-worklist-prepare": octreeLosassoAdaptivePhiWorklistPrepareWGSL,
  "octree-losasso-adaptive-phi-worklist-reach": octreeLosassoAdaptivePhiWorklistReachWGSL,
  "octree-losasso-adaptive-phi-worklist-inflow": octreeLosassoAdaptivePhiWorklistInflowWGSL,
  "octree-losasso-adaptive-phi-worklist-independent": octreeLosassoAdaptivePhiWorklistIndependentWGSL,
  "octree-losasso-adaptive-phi-worklist-constrained": octreeLosassoAdaptivePhiWorklistConstrainedWGSL,
  "octree-losasso-adaptive-phi-worklist-constraint-mark": octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  "octree-losasso-adaptive-phi-worklist-finalize": octreeLosassoAdaptivePhiWorklistFinalizeWGSL,
  "octree-losasso-adaptive-phi-worklist-receipt": octreeLosassoAdaptivePhiWorklistReceiptWGSL,
  "octree-losasso-adaptive-phi-worklist-project": octreeLosassoAdaptivePhiWorklistProjectWGSL,
  "octree-losasso-adaptive-phi-redistance-initialize": octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  "octree-losasso-adaptive-phi-redistance-a-to-b": octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
  "octree-losasso-adaptive-phi-redistance-b-to-a": octreeLosassoAdaptivePhiRedistanceBtoAWGSL,
  "octree-losasso-adaptive-phi-redistance-project-a": octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  "octree-losasso-adaptive-phi-redistance-project-b": octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
  "octree-losasso-adaptive-phi-redistance-finish": octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  "octree-losasso-adaptive-phi-backtrace": octreeLosassoAdaptivePhiBacktraceWGSL,
  "octree-losasso-adaptive-phi-transport": octreeLosassoAdaptivePhiTransportWGSL,
  "octree-losasso-adaptive-phi-handoff": octreeLosassoAdaptivePhiHandoffWGSL,
  "octree-losasso-adaptive-phi-commit": octreeLosassoAdaptivePhiCommitWGSL,
  "octree-losasso-adaptive-phi-ghost": octreeLosassoAdaptivePhiGhostWGSL,
  "octree-losasso-adaptive-phi-evidence": octreeLosassoAdaptivePhiEvidenceWGSL,
  "octree-losasso-adaptive-phi-accepted-schedule":
    octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
  "octree-losasso-adaptive-phi-schedule": octreeLosassoAdaptivePhiScheduleWGSL,
  "octree-losasso-adaptive-phi-volume-evidence": octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
  "octree-losasso-adaptive-velocity": octreeLosassoAdaptiveVelocityWGSL,
  "octree-losasso-adaptive-mass": octreeLosassoAdaptiveMassWGSL,
  "octree-losasso-adaptive-dynamics": makeOctreeLosassoAdaptiveDynamicsWGSL(
    octreeLosassoAdaptiveVelocitySamplerWGSL()),
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
  // Every lens program, composed exactly as the overlay composes it. A lens is
  // declaration plus a classifier snippet, so nothing about it is checked until
  // something assembles the module — and the overlay only assembles the one
  // lens a user happens to open.
  ...Object.fromEntries(SPARSE_CM12_LENSES.flatMap((lens) =>
    Object.keys(lens.programs).map((program) =>
      [`stage-lens/${lens.stage}/${program}`, stageLensProgramWGSL(lens, program)]))),
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
    // A lens key carries its stage as a path segment; flatten it rather than
    // mkdir per stage, so the reported name stays the key the roster uses.
    const path = join(directory, `${name.replaceAll("/", "--")}.wgsl`);
    writeFileSync(path, source);
    const result = spawnSync(naga, [path], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(`${name}:\n${result.stderr || result.stdout}`);
    console.log(`validated ${name}`);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
