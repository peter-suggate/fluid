import type {
  MethodHarnessConstructionContext,
  MethodHarnessLane,
  MethodHarnessPlugin,
  MethodHarnessRunDescription,
  MethodHarnessTerminalContext,
  MethodHarnessTerminalVerdict,
} from "../../core/method-harness-contract";
import type { MethodParamValues } from "../../core/method-contract";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import {
  applyOctreeEnvironmentOverrides,
  octreeHarnessLane,
  OCTREE_HARNESS_DIAGNOSTIC_PACKS,
  OCTREE_HARNESS_ENVIRONMENT_VARIABLES,
} from "../octree-shared/harness";
import { OCTREE_LOSASSO_COARSE_PHI_MAGIC } from "./webgpu-octree-losasso-coarse-phi.wgsl";
import {
  OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS,
  unpackAdaptivePhiReceipt,
} from "./webgpu-octree-losasso-adaptive-phi";
import {
  adaptiveMassControlLayout,
  OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC,
  OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS,
  unpackAdaptiveMassReceipt,
} from "./webgpu-octree-losasso-adaptive-mass";
import {
  adaptivePhiRedistanceUnresolvedAudit,
  adaptiveSurfacePublicationD4Audit,
  adaptiveVelocityD4Audit,
  adaptiveVelocityInputD4Audit,
  adaptiveVelocityStencilAudit,
  adaptiveVelocityUnresolvedNodeAudit,
  decodeAdaptiveVelocityGPUFailureDiagnostics,
  losassoAdaptiveRasterCutoverFailures,
  losassoRasterCutoverFailures,
  type AdaptiveSurfacePublicationSnapshot,
} from "./harness-adaptive-audits";

/**
 * The Losasso lane's node-only harness half.
 *
 * The terminal oracle below is the whole reason this contract exists: it reads
 * the reduced row authority, the adaptive graph/scalar/renderer receipts, the
 * conservative mass transaction and the production raster, and every one of
 * those is a buffer only this method builds. It ran inside the smoke
 * executor's run loop behind `method.id === "losasso"`, which meant the
 * generic loop imported eight Losasso ABI modules and carried a thousand lines
 * that no other method could ever execute.
 */

const LOSASSO_CUTOVER_ORACLE_ID = "losasso-cutover";

/**
 * Losasso's terminal oracle reads the accepted native authority directly, so
 * the shared per-step restriction/air-support ring is not encoded on this lane
 * and asking for it would report a wiring failure for evidence the lane
 * deliberately never publishes.
 */
const losassoHarnessLane: MethodHarnessLane = Object.freeze({
  ...octreeHarnessLane,
  nativeTerminalReceipt: true,
});

/**
 * What the freshly constructed t=0 authority contained.
 *
 * By the terminal oracle it has been overwritten a few hundred times, so the
 * only place this can be observed is here. Factor one alone: the separate
 * fine band publishes its own generation diagnostics instead.
 */
async function losassoConstructionRecords(
  context: MethodHarnessConstructionContext,
): Promise<readonly Readonly<Record<string, unknown>>[]> {
  if (context.separateFineLevelSetBand) return [];
  const receipt = await (context.solver as {
    octreeProjection?: { readLosassoAuthorityDiagnostics(): Promise<unknown> };
  }).octreeProjection?.readLosassoAuthorityDiagnostics();
  return [{ phase: "losasso-adaptive-construction-receipt", receipt: receipt ?? null }];
}

const LOSASSO_ENVIRONMENT_VARIABLES: readonly string[] = Object.freeze([
  ...OCTREE_HARNESS_ENVIRONMENT_VARIABLES,
  "FLUID_LOSASSO_VELOCITY_EXTENSION",
  "FLUID_TOPOLOGY_CADENCE_ADVANCES",
  "FLUID_LOSASSO_D4_CUTOVER",
  "FLUID_LOSASSO_CUTOVER_REPORT_ONLY",
  "FLUID_RASTER_MESH_SYMMETRY",
]);

function applyLosassoEnvironmentOverrides(
  values: MethodParamValues,
  env: Readonly<Record<string, string | undefined>>,
): void {
  applyOctreeEnvironmentOverrides(values, env);
  // Both knobs below name Losasso machinery the frozen Power reference lane
  // does not run, and Power pins its topology cadence as a construction
  // constant. Writing them on a Power run would put a setting into the
  // constructed record that nothing read.
  const velocityExtension = env.FLUID_LOSASSO_VELOCITY_EXTENSION;
  if (velocityExtension !== undefined) {
    if (velocityExtension !== "fixed-jacobi" && velocityExtension !== "causal-front") {
      throw new Error("FLUID_LOSASSO_VELOCITY_EXTENSION must be fixed-jacobi or causal-front");
    }
    values.losassoVelocityExtension = velocityExtension;
  }
  const topologyCadence = env.FLUID_TOPOLOGY_CADENCE_ADVANCES === undefined
    ? undefined : Number(env.FLUID_TOPOLOGY_CADENCE_ADVANCES);
  if (topologyCadence !== undefined) {
    if (!Number.isSafeInteger(topologyCadence) || topologyCadence < 1 || topologyCadence > 8) {
      throw new Error("FLUID_TOPOLOGY_CADENCE_ADVANCES must be an integer from 1 through 8");
    }
    values.topologyCadenceAdvances = topologyCadence;
  }
}

/** Identity marker for the production-visible factor-4 D4 cutover command.
 * Generic Losasso lanes still receive the native authority oracle, while this
 * marker additionally refuses any drift in the named release gate's wiring. */
function losassoD4CutoverWiringFailures(run: MethodHarnessRunDescription): readonly string[] {
  if (run.env.FLUID_LOSASSO_D4_CUTOVER !== "1") return [];
  return Object.freeze([
    ...(run.scenarioId === "symmetric-expansion" ? [] : [`scene=${run.scenarioId}`]),
    ...(run.laneId === "fine-factor-4" ? [] : [`lane=${run.laneId}`]),
    ...(run.methodId === "losasso" ? [] : [`method=${run.methodId}`]),
    ...(Number(run.values.maximumLeafSize) === 16
      ? [] : [`maximumLeafSize=${String(run.values.maximumLeafSize)}`]),
    ...(Number(run.values.interfaceRefinementBandCells) === 4
      ? [] : [`interfaceBand=${String(run.values.interfaceRefinementBandCells)}`]),
    ...(Number(run.values.globalFineLevelSetFactor) === 4
      ? [] : [`fineFactor=${String(run.values.globalFineLevelSetFactor)}`]),
    ...(run.collecting.has("initial-final raster") ? [] : ["initial-final-raster=off"]),
    ...(run.collecting.has("checkpoint raster") ? [] : ["checkpoint-raster=off"]),
    ...(run.collecting.has("retained-generation probe") ? [] : ["retained-generation-probe=off"]),
    ...(run.env.FLUID_RASTER_MESH_SYMMETRY === "1" ? [] : ["exact-D4-raster=off"]),
  ]);
}

async function losassoCutoverOracle(
  context: MethodHarnessTerminalContext,
): Promise<MethodHarnessTerminalVerdict | undefined> {
  const { separateFineLevelSetBand } = context;
  const records: Record<string, unknown>[] = [];
  const solver = context.solver as {
    readonly info: GPUEulerianInfo;
    octreeProjection?: {
    readLosassoAuthorityDiagnostics(): Promise<Readonly<{
      authority: readonly number[]; solver: readonly number[]; coarsePhi: readonly number[];
      adaptiveGraph: readonly number[]; adaptivePhi: readonly number[];
      adaptiveRenderer: readonly number[];
      adaptiveMassControl: readonly number[]; adaptiveMassReceipts: readonly number[];
    }> | undefined>;
    readAdaptiveVelocityReceipts(): Promise<readonly number[] | undefined>;
    readAdaptiveVelocityDiagnostics(): Promise<readonly number[] | undefined>;
    readAdaptiveSurfacePublicationDiagnostics():
      Promise<AdaptiveSurfacePublicationSnapshot | undefined>;
    };
  };
  const { info } = solver;
  const projection = solver.octreeProjection;
  const [native, fine, raster, adaptiveVelocity, adaptiveVelocityDiagnostics,
    adaptivePublication] = await Promise.all([
    projection?.readLosassoAuthorityDiagnostics(),
    separateFineLevelSetBand
      ? context.readGlobalFineReceipt() : Promise.resolve(undefined),
    context.renderPresentation(separateFineLevelSetBand),
    !separateFineLevelSetBand
      ? projection?.readAdaptiveVelocityReceipts() : Promise.resolve(undefined),
    !separateFineLevelSetBand
      ? projection?.readAdaptiveVelocityDiagnostics() : Promise.resolve(undefined),
    !separateFineLevelSetBand
      ? projection?.readAdaptiveSurfacePublicationDiagnostics() : Promise.resolve(undefined),
  ]);
  const authority = native?.authority ?? [], pressure = native?.solver ?? [];
  const coarsePhi = native?.coarsePhi ?? [];
  const adaptiveGraph = native?.adaptiveGraph ?? [];
  const adaptivePhi = native?.adaptivePhi ?? [];
  const adaptiveRenderer = native?.adaptiveRenderer ?? [];
  const adaptiveMassControl = native?.adaptiveMassControl ?? [];
  const adaptiveMassReceipts = native?.adaptiveMassReceipts ?? [];
  const adaptivePhiReceipt = adaptivePhi.length >= OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS
    ? unpackAdaptivePhiReceipt(adaptivePhi) : undefined;
  const adaptiveMassReceipt = adaptiveMassReceipts.length
    >= OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS
    ? unpackAdaptiveMassReceipt(adaptiveMassReceipts) : undefined;
  const adaptiveD4 = adaptivePublication
    ? adaptiveSurfacePublicationD4Audit(adaptivePublication) : undefined;
  const adaptivePhiRedistanceUnresolved = adaptivePublication
    ? adaptivePhiRedistanceUnresolvedAudit(adaptivePublication) : undefined;
  const adaptiveVelocityD4 = adaptivePublication
    ? adaptiveVelocityD4Audit(adaptivePublication) : undefined;
  const adaptiveVelocityInputD4 = adaptivePublication
    ? adaptiveVelocityInputD4Audit(adaptivePublication) : undefined;
  const adaptiveVelocityStencil = adaptivePublication
    ? adaptiveVelocityStencilAudit(adaptivePublication) : undefined;
  const adaptiveVelocityCandidateFaceD4 = adaptivePublication
    ? adaptiveVelocityInputD4Audit({ ...adaptivePublication,
      authorityControl: adaptivePublication.candidateAuthorityControl,
      faceGeometry: adaptivePublication.candidateFaceGeometry,
      predictedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity,
      projectedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity,
      advectedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity,
      extendedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity,
    }) : undefined;
  const adaptiveVelocityCandidateNodalD4 = adaptivePublication
    ? adaptiveVelocityD4Audit({ ...adaptivePublication,
      nodalVelocity: adaptivePublication.candidateNodalVelocity }) : undefined;
  const adaptiveVelocityCandidateStencil = adaptivePublication
    ? adaptiveVelocityStencilAudit({ ...adaptivePublication,
      nodalVelocity: adaptivePublication.candidateNodalVelocity,
      projectedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity,
      advectedFaceVelocity: adaptivePublication.candidateExtendedFaceVelocity }) : undefined;
  const adaptiveVelocityUnresolved = adaptivePublication && adaptiveVelocity
    ? adaptiveVelocityUnresolvedNodeAudit(adaptivePublication, adaptiveVelocity) : undefined;
  const adaptiveVelocityGPUFailure = adaptiveVelocityDiagnostics
    ? decodeAdaptiveVelocityGPUFailureDiagnostics(adaptiveVelocityDiagnostics,
      [info.nx, info.ny, info.nz]) : undefined;
  if (adaptiveD4) records.push(({ phase: "losasso-adaptive-publication-d4", metrics: adaptiveD4 }));
  if (adaptivePhiRedistanceUnresolved?.unresolved.length) records.push(({ phase: "losasso-adaptive-phi-redistance-unresolved",
    metrics: adaptivePhiRedistanceUnresolved,
  }));
  if (adaptiveVelocityD4) records.push(({ phase: "losasso-adaptive-velocity-d4", metrics: adaptiveVelocityD4 }));
  if (adaptiveVelocityInputD4) records.push(({ phase: "losasso-adaptive-velocity-input-d4",
    metrics: adaptiveVelocityInputD4 }));
  if (adaptiveVelocityStencil) records.push(({ phase: "losasso-adaptive-velocity-stencil-audit",
    metrics: adaptiveVelocityStencil }));
  if (adaptiveVelocityCandidateFaceD4) records.push(({ phase: "losasso-adaptive-velocity-candidate-face-d4",
    metrics: adaptiveVelocityCandidateFaceD4 }));
  if (adaptiveVelocityCandidateNodalD4) records.push(({ phase: "losasso-adaptive-velocity-candidate-nodal-d4",
    metrics: adaptiveVelocityCandidateNodalD4 }));
  if (adaptiveVelocityCandidateStencil) records.push(({ phase: "losasso-adaptive-velocity-candidate-stencil-audit",
    metrics: adaptiveVelocityCandidateStencil }));
  if (adaptiveVelocityUnresolved && adaptiveVelocityUnresolved.fields.some(
    (field) => (field.unresolvedWithinReach as number) !== 0)) {
    records.push(({ phase: "losasso-adaptive-velocity-unresolved", metrics: adaptiveVelocityUnresolved }));
  }
  if (adaptiveVelocityGPUFailure?.some((field) => field.header.unresolved !== 0)) {
    records.push(({ phase: "losasso-adaptive-velocity-gpu-failure", fields: adaptiveVelocityGPUFailure }));
  }
  const failures: string[] = [];
  if (context.steps < 1) failures.push("no accepted advance executed");
  if (authority.length < 5 || authority[0] === 0 || authority[1] === 0
    || authority[2] === 0 || authority[3] !== 1 || authority[4] !== 0) {
    failures.push("reduced authority is invalid or empty");
  }
  if (pressure.length < 5 || pressure[0] !== 0 || pressure[1] === 0
    || pressure[4] !== authority[1]) {
    failures.push("MGPCG did not converge on the accepted row authority");
  }
  const adaptivePhiValid = !separateFineLevelSetBand && coarsePhi.length >= 15
    && adaptiveGraph.length >= 32
    && adaptivePhi.length >= OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS
    && adaptiveGraph[0] === authority[0] && adaptiveGraph[3] === adaptiveGraph[0]
    && adaptiveGraph[4] === 0 && adaptiveGraph[1]! > authority[1]!
    && adaptiveGraph[28] === authority[1] && adaptiveGraph[29] === 0
    && adaptiveGraph[5] !== 0 && adaptiveGraph[6] === adaptiveGraph[5]
    && adaptiveRenderer[0] === 0x8000_0000
    && adaptiveRenderer[1] === adaptiveGraph[5]
    && adaptiveRenderer[2] === adaptiveGraph[1]
    && coarsePhi[0] === OCTREE_LOSASSO_COARSE_PHI_MAGIC
    && coarsePhi[1] === adaptiveGraph[0] && coarsePhi[2] === adaptiveGraph[1]
    // Factor-one scalar generations advance independently of topology
    // epochs. Topology evidence must pair its two scalar stamps, not force
    // either to equal the reduced-row epoch.
    && coarsePhi[3] === 0 && coarsePhi[12] === adaptiveGraph[5]
    && coarsePhi[13] === 0 && coarsePhi[14] === coarsePhi[12];
  const restrictedPhiValid = separateFineLevelSetBand && coarsePhi.length >= 15
    && coarsePhi[0] === OCTREE_LOSASSO_COARSE_PHI_MAGIC
    && coarsePhi[1] === authority[0] && coarsePhi[2] === authority[1]
    && coarsePhi[3] === authority[2] && coarsePhi[13] === 0;
  if (!adaptivePhiValid && !restrictedPhiValid) {
    failures.push("coarse phi is invalid or disagrees with the reduced authority");
  }
  if (!separateFineLevelSetBand) {
    const scheduledPhiNodes = (adaptivePhi[32] ?? 0) + (adaptivePhi[34] ?? 0);
    const retainedPhiNodes = (adaptivePhi[33] ?? 0) + (adaptivePhi[35] ?? 0);
    const identityAdvance = adaptivePhi[1] === 0 && adaptivePhi[12] === 0;
    if (adaptivePhi[2] !== 0 || adaptivePhi[3] !== 0
      || scheduledPhiNodes + retainedPhiNodes !== adaptiveGraph[2]
      || adaptivePhi[4] !== scheduledPhiNodes || adaptivePhi[5] !== 0
      || adaptivePhi[7] !== (identityAdvance ? 0 : scheduledPhiNodes)
      || adaptivePhi[14] !== 1 || adaptivePhi[15] !== adaptiveGraph[1]
      || adaptivePhi[22] !== 1 || adaptivePhi[23] !== adaptiveGraph[5]) {
      failures.push("adaptive scalar transport/redistance/renderer receipt is incomplete or stale");
    }
  }
  // Surface transport samples the prior projected field. The construction
  // projection has dt=0, so step one may legitimately transport a zero
  // field; by step two the preceding projection tail must produce motion.
  // The graph-owned conservative-mass lane advances phi through an external
  // publication clock, so its adaptive-phi receipt intentionally no longer
  // contains the retired semi-Lagrangian dt/max-delta pair. Validate the
  // conservative transaction itself on that lane.
  if (!separateFineLevelSetBand && context.steps > 1) {
    const massAuthority = adaptiveMassControl.length
      >= adaptiveMassControlLayout.errorBits + 1
      && adaptiveMassControl[adaptiveMassControlLayout.magic]
        === OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC;
    if (massAuthority) {
      const acceptedMass = adaptiveMassReceipt?.acceptedMass_m3 ?? Number.NaN;
      const transportedMass = adaptiveMassReceipt?.transportedMass_m3 ?? Number.NaN;
      if (adaptiveMassControl[adaptiveMassControlLayout.valid] !== 1
        || adaptiveMassControl[adaptiveMassControlLayout.errorBits] !== 0
        || !adaptiveMassReceipt || adaptiveMassReceipt.errors !== 0
        || adaptiveMassReceipt.donors === 0 || adaptiveMassReceipt.transfers === 0
        || adaptiveMassReceipt.missingRecipients !== 0
        || !(acceptedMass > 0) || !(transportedMass > 0)
        || !Number.isFinite(adaptiveMassReceipt.signedTransportDrift_m3)) {
        failures.push("adaptive conservative scalar transaction is invalid or inactive");
      }
    } else {
      const bits = new Uint32Array(1); bits[0] = adaptivePhi[0] ?? 0;
      const dt_s = new Float32Array(bits.buffer)[0]!;
      bits[0] = adaptivePhi[1] ?? 0;
      const maximumDelta_m = new Float32Array(bits.buffer)[0]!;
      if (!(dt_s > 0) || !(maximumDelta_m > 0) || !Number.isFinite(maximumDelta_m)) {
        failures.push(`adaptive scalar did not measurably evolve (dt=${dt_s},maxDelta=${maximumDelta_m})`);
      }
    }
  }
  if (!separateFineLevelSetBand && info.quadtreePressureRejectionSummary) {
    failures.push(`adaptive tuple rejected: ${info.quadtreePressureRejectionSummary}`);
  }
  // D4 equivariance is a scene oracle, not a universal property of every
  // factor-one publication. A one-sided dam is intentionally not invariant
  // under reflection; applying this gate there made an otherwise healthy
  // Dawn raster reproduction fail for the wrong reason.
  // Fixed-point mass transport consumes projected face velocities whose D4
  // oracle is numerical rather than bitwise. Its reconstructed nodal phi may
  // therefore differ by less than one thousandth of a finest cell while the
  // topology remains exact. The downstream fluid-symmetry hook still gates
  // volume, velocity, pressure, RHS, topology, and front evolution.
  // Sixteen published-phi quanta, still less than 0.5% of one 5 cm finest
  // cell. Conservative level-set volume recovery deliberately retains more
  // kinetic energy, so the sub-ulp pressure asymmetry accepted by the field
  // oracle travels farther before phi is quantized. Keep topology bit-exact
  // and bound the resulting scalar displacement geometrically instead of
  // requiring the lower-energy solution's four-quantum envelope.
  const adaptiveScalarD4Tolerance_m = 16 / 65_536;
  if (!separateFineLevelSetBand && context.scenarioId === "symmetric-expansion" && (!adaptiveD4
    || adaptiveD4.graphLeafIdentityMismatches !== 0
    || adaptiveD4.nodalCornerMaximumAbsoluteError > adaptiveScalarD4Tolerance_m
    || adaptiveD4.rendererLeafIdentityMismatches !== 0
    || adaptiveD4.rendererPrimaryMaximumAbsoluteError > adaptiveScalarD4Tolerance_m
    || adaptiveD4.rendererAuxMaximumAbsoluteError > adaptiveScalarD4Tolerance_m
    || adaptiveD4.publicationCornerMaximumAbsoluteError > adaptiveScalarD4Tolerance_m)) {
    failures.push("adaptive graph/scalar/renderer publication exceeds the D4 tolerance");
  }
  if (separateFineLevelSetBand && (!fine || !fine.publicationValid
    || fine.validSamples === 0 || fine.finiteValidSamples !== fine.validSamples
    || fine.negativeValidSamples === 0 || fine.positiveValidSamples === 0
    || coarsePhi[12] !== fine.generation || coarsePhi[14] !== fine.generation)) {
    failures.push("factor-4 fine phi is invalid, non-finite, one-sided, or generation-incoherent");
  }
  if (separateFineLevelSetBand) {
    if (!fine || !raster) failures.push("factor-4 production raster evidence is unavailable");
    else if (context.collecting.has("rigid coupling")) {
      // Coupling oracles own a deliberately narrower rendering assertion:
      // current fine geometry, not silhouette pixel identity around a solid.
      if (raster.surfaceGeometrySource !== "global-fine-coarse"
        || raster.globalFineCrossingPublished !== true
        || raster.presentationFallbackActive !== false) {
        failures.push("rigid coupling raster did not consume current global-fine geometry");
      }
    } else failures.push(...losassoRasterCutoverFailures(raster, fine.generation));
  } else {
    if (!raster) failures.push("factor-one production raster evidence is unavailable");
    else failures.push(...losassoAdaptiveRasterCutoverFailures(raster, coarsePhi[12] ?? 0));
  }
  const authorityRecord = Object.freeze({ backend: "losasso", steps: context.steps,
    submittedTime_s: info.submittedTime_s, authority, solver: pressure, coarsePhi,
    ...(!separateFineLevelSetBand ? { adaptiveGraph, adaptivePhi, adaptiveRenderer,
      adaptiveMassControl, adaptiveMassReceipts,
      adaptiveZeroSetVolume: adaptivePhiReceipt ? {
        // Sum of deterministic subcell volumes cut from the published
        // adaptive trilinear nodal field; unlike wet cell-centre counts this
        // is comparable across procedural/legacy representations.
        measured_m3: adaptivePhiReceipt.measuredVolume_m3,
        signedRedistanceDrift_m3: adaptivePhiReceipt.signedRedistanceVolumeDrift_m3,
        acceptedTransaction: adaptivePhiReceipt.volumeTransaction,
        candidateMeasured_m3: adaptivePhiReceipt.candidate.measuredVolume_m3,
        candidateSignedRedistanceDrift_m3:
          adaptivePhiReceipt.candidate.signedRedistanceVolumeDrift_m3,
        candidateTransaction: adaptivePhiReceipt.candidateVolumeTransaction,
      } : undefined,
      adaptiveVelocity: adaptiveVelocity ?? [], adaptiveD4, adaptiveVelocityUnresolved,
      adaptiveVelocityGPUFailure } : {}),
    fine: fine ? { generation: fine.generation, worklistGeneration: fine.worklistGeneration,
      activePages: fine.activePages, capacity: fine.configuredBrickCapacity,
      validSamples: fine.validSamples, finiteValidSamples: fine.finiteValidSamples,
      negativeValidSamples: fine.negativeValidSamples,
      positiveValidSamples: fine.positiveValidSamples,
      publicationValid: fine.publicationValid } : undefined,
    raster: raster ? {
      geometrySource: raster.surfaceGeometrySource,
      meshPublicationGeneration: raster.meshPublicationGeneration,
      frontInterfacePixels: raster.frontInterfacePixels,
      backInterfacePixels: raster.backInterfacePixels,
      pairedInterfacePixels: raster.pairedInterfacePixels,
      backOnlyInterfacePixels: raster.backOnlyInterfacePixels,
      backOnlyInterfaceLocations: raster.backOnlyInterfaceLocations,
      backOnlyInterfacePositions_m: raster.backOnlyInterfacePositions_m,
      narrowVerticalSlits: raster.narrowVerticalSlits,
      enclosedSurfaceHoles: raster.enclosedSurfaceHoles,
      reverseView: raster.reverseView ? {
        frontInterfacePixels: raster.reverseView.frontInterfacePixels,
        backInterfacePixels: raster.reverseView.backInterfacePixels,
        pairedInterfacePixels: raster.reverseView.pairedInterfacePixels,
        backOnlyInterfacePixels: raster.reverseView.backOnlyInterfacePixels,
        backOnlyInterfaceLocations: raster.reverseView.backOnlyInterfaceLocations,
        backOnlyInterfacePositions_m: raster.reverseView.backOnlyInterfacePositions_m,
        narrowVerticalSlits: raster.reverseView.narrowVerticalSlits,
        enclosedSurfaceHoles: raster.reverseView.enclosedSurfaceHoles,
        unionSurfaceHoles: raster.reverseView.unionSurfaceHoles,
      } : undefined,
      vertexCount: raster.vertexCount, activeCubeCount: raster.activeCubeCount,
      surfaceMeshSymmetry: raster.surfaceMeshSymmetry,
      activeCubeSymmetry: raster.activeCubeSymmetry,
      sharpPatchRaster: raster.sharpPatchRaster,
      authorityTransition: raster.globalFineAuthorityTransition,
    } : undefined,
    failures: Object.freeze(failures) });
  return {
    id: LOSASSO_CUTOVER_ORACLE_ID,
    label: "Losasso cutover",
    failures: Object.freeze(failures),
    authority: authorityRecord,
    reportOnly: context.env.FLUID_LOSASSO_CUTOVER_REPORT_ONLY === "1",
    records: Object.freeze(records),
    presentation: raster,
  };
}

export const losassoHarnessPlugin: MethodHarnessPlugin = {
  methodId: "losasso",
  lane: losassoHarnessLane,
  environmentVariables: LOSASSO_ENVIRONMENT_VARIABLES,
  applyEnvironmentOverrides: applyLosassoEnvironmentOverrides,
  diagnosticPacks: OCTREE_HARNESS_DIAGNOSTIC_PACKS,
  constructionRecords: losassoConstructionRecords,
  runWiringFailures: losassoD4CutoverWiringFailures,
  terminalOracle: losassoCutoverOracle,
};

export default losassoHarnessPlugin;
