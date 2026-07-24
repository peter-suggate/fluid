import type {
  OctreeFaceBandControlSnapshot,
  OctreeFaceBandPointFieldControlSnapshot,
  OctreeFaceBandPowerPublicationSnapshot,
  OctreeFaceBandTransientPowerControlSnapshot,
} from "../lib/webgpu-octree-face-closest-point";
import type { OctreePowerVelocityControl } from "../lib/webgpu-octree-power-velocity";

export interface ExactSection5GenerationAudit {
  /** Fine generation exposed after the accepted advance has completed. */
  readonly publishedFineGeneration: number;
  readonly expectedPowerGeneration: number;
  readonly expectedPowerFaceCount: number;
  readonly previousAcceptedSection5FineGeneration: number;
  readonly previousPowerGeneration: number;
  readonly faceBand: OctreeFaceBandControlSnapshot;
  readonly transientPower: OctreeFaceBandTransientPowerControlSnapshot;
  readonly pointField: OctreeFaceBandPointFieldControlSnapshot;
  readonly powerPublication: OctreeFaceBandPowerPublicationSnapshot;
}

export interface FinalPerformanceAuthorityAudit {
  readonly expectedSteps: number;
  readonly observedSteps: number;
  readonly expectedTime_s: number;
  readonly targetTime_s: number;
  readonly submittedTime_s: number;
  readonly fineSourceGeneration: number;
  /** Five-word GPU-published fine worklist header. */
  readonly fineWorklistHeader: ArrayLike<number>;
  readonly finePageCapacity: number;
  readonly powerFaces: {
    readonly flags: number;
    readonly rowCount: number;
    readonly faceCount: number;
    readonly incidenceCount: number;
    readonly generation: number;
    readonly valid: boolean;
  };
  readonly powerVelocity: OctreePowerVelocityControl;
  readonly faceBand: OctreeFaceBandControlSnapshot;
  readonly transientPower: OctreeFaceBandTransientPowerControlSnapshot;
  readonly pointField: OctreeFaceBandPointFieldControlSnapshot;
  readonly powerPublication: OctreeFaceBandPowerPublicationSnapshot;
}

/**
 * Fail-closed CPU mirror of the exact Dawn Section 5 publication contract.
 *
 * These are four transactions on two clocks, not four independent diagnostic
 * snapshots. The regular band, transient generalized-face graph, and point
 * field consume the fine field that entered the accepted advance. The immutable
 * regular band may be that current field or its exact predecessor when the
 * current transient point field has already republished on the aligned
 * fine/power clock. The advance then publishes its successor fine field, so at
 * a fenced post-advance audit the live fine clock is exactly one ahead. Older
 * retained headers are never accepted.
 */
export function exactSection5GenerationAuditFailures(
  audit: ExactSection5GenerationAudit,
): readonly string[] {
  const failures: string[] = [];
  const positiveSafeGeneration = (value: number) =>
    Number.isSafeInteger(value) && value > 0;

  if (!positiveSafeGeneration(audit.publishedFineGeneration)) {
    failures.push("invalid published fine generation");
  }
  if (!positiveSafeGeneration(audit.expectedPowerGeneration)) {
    failures.push("invalid expected power generation");
  }
  if (!Number.isSafeInteger(audit.expectedPowerFaceCount) || audit.expectedPowerFaceCount <= 0) {
    failures.push("invalid expected power face count");
  }
  if (audit.publishedFineGeneration !== audit.expectedPowerGeneration + 1) {
    failures.push("published fine generation is not the Section 5 successor");
  }
  if (audit.expectedPowerGeneration <= audit.previousAcceptedSection5FineGeneration) {
    failures.push("Section 5 fine generation did not advance");
  }
  if (audit.expectedPowerGeneration <= audit.previousPowerGeneration) {
    failures.push("power generation did not advance");
  }
  if (!audit.faceBand.valid || audit.faceBand.flags !== 0) {
    failures.push("face-band publication is invalid");
  }
  const generationMask = 0x3fff_ffff;
  const faceBandPredecessor = (audit.expectedPowerGeneration + generationMask) & generationMask;
  if (audit.faceBand.generation !== audit.expectedPowerGeneration
    && audit.faceBand.generation !== faceBandPredecessor) {
    failures.push("face-band generation is neither current nor the exact predecessor");
  }
  if (!audit.transientPower.valid || audit.transientPower.flags !== 0) {
    failures.push("transient-power publication is invalid");
  }
  if (audit.transientPower.generation !== audit.expectedPowerGeneration) {
    failures.push("transient-power generation is not current");
  }
  if (!audit.pointField.valid || audit.pointField.flags !== 0) {
    failures.push("point-field publication is invalid");
  }
  if (audit.pointField.generation !== audit.expectedPowerGeneration) {
    failures.push("point-field generation is not current");
  }
  if (audit.transientPower.rowCount !== audit.pointField.rowCount) {
    failures.push("transient-power and point-field row counts differ");
  }
  if (!audit.powerPublication.valid || audit.powerPublication.flags !== 0) {
    failures.push("power publication is invalid");
  }
  if (audit.powerPublication.fineGeneration !== audit.expectedPowerGeneration) {
    failures.push("power publication fine generation is not current");
  }
  if (audit.powerPublication.powerGeneration !== audit.expectedPowerGeneration) {
    failures.push("power publication power generation is not current");
  }
  if (audit.powerPublication.faceCount !== audit.expectedPowerFaceCount) {
    failures.push("power publication face count is not current");
  }
  return failures;
}

/**
 * Final-only, fail-closed authority gate for the fixed-cadence UI performance
 * profile. Dam-break reset publishes power generation 1 before timed advances,
 * so N accepted advances must end at power/Section-5 generation N+1 and expose
 * the N+2 fine successor. Tying the GPU clocks to the exact outer-step count is
 * what rejects a mutually coherent but retained old publication.
 */
export function finalPerformanceAuthorityFailures(
  audit: FinalPerformanceAuthorityAudit,
): readonly string[] {
  const failures: string[] = [];
  const exactTime = (observed: number, expected: number) =>
    Number.isFinite(observed) && Math.abs(observed - expected) <= 1e-9;
  if (!Number.isSafeInteger(audit.expectedSteps) || audit.expectedSteps < 1) {
    failures.push("missing exact positive step contract");
    return failures;
  }
  if (audit.observedSteps !== audit.expectedSteps) failures.push("accepted step count is not exact");
  if (!Number.isFinite(audit.expectedTime_s) || audit.expectedTime_s <= 0) {
    failures.push("invalid exact time contract");
  } else {
    if (!exactTime(audit.targetTime_s, audit.expectedTime_s)) {
      failures.push("target time is not the exact step checkpoint");
    }
    if (!exactTime(audit.submittedTime_s, audit.expectedTime_s)) {
      failures.push("submitted time is not the exact step checkpoint");
    }
  }

  const expectedPowerGeneration = audit.expectedSteps + 1;
  const expectedFineGeneration = expectedPowerGeneration + 1;
  if (audit.fineSourceGeneration !== expectedFineGeneration) {
    failures.push("fine source generation is not current for the exact step");
  }
  if (audit.fineWorklistHeader.length < 5) {
    failures.push("fine worklist publication header is missing");
  } else {
    const activePages = Number(audit.fineWorklistHeader[0]) >>> 0;
    if ((Number(audit.fineWorklistHeader[1]) >>> 0) !== audit.fineSourceGeneration) {
      failures.push("fine worklist generation does not match the exposed source");
    }
    if ((Number(audit.fineWorklistHeader[3]) >>> 0) !== 1
      || (Number(audit.fineWorklistHeader[4]) >>> 0) !== 1) {
      failures.push("fine worklist publication is invalid");
    }
    if (activePages === 0 || activePages > audit.finePageCapacity) {
      failures.push("fine worklist active-page count is invalid");
    }
  }

  if (!audit.powerFaces.valid || audit.powerFaces.flags !== 0) {
    failures.push("power-face publication is invalid");
  }
  if (audit.powerFaces.rowCount === 0 || audit.powerFaces.faceCount === 0
    || audit.powerFaces.incidenceCount === 0) {
    failures.push("power-face publication is empty");
  }
  if (audit.powerFaces.generation !== expectedPowerGeneration) {
    failures.push("power-face generation is not current for the exact step");
  }

  if (audit.powerVelocity.flags !== 0x8000_0000
    || audit.powerVelocity.illConditionedCount !== 0
    || audit.powerVelocity.reconstructedCount !== audit.powerVelocity.rowCount) {
    failures.push("power-velocity publication is invalid");
  }
  if (audit.powerVelocity.rowCount !== audit.powerFaces.rowCount
    || audit.powerVelocity.faceCount !== audit.powerFaces.faceCount
    || audit.powerVelocity.incidenceCount !== audit.powerFaces.incidenceCount) {
    failures.push("power-velocity topology does not match current power faces");
  }
  if (audit.powerVelocity.generation !== expectedPowerGeneration) {
    failures.push("power-velocity generation is not current for the exact step");
  }

  failures.push(...exactSection5GenerationAuditFailures({
    publishedFineGeneration: audit.fineSourceGeneration,
    expectedPowerGeneration,
    expectedPowerFaceCount: audit.powerFaces.faceCount,
    previousAcceptedSection5FineGeneration: expectedPowerGeneration - 1,
    previousPowerGeneration: expectedPowerGeneration - 1,
    faceBand: audit.faceBand,
    transientPower: audit.transientPower,
    pointField: audit.pointField,
    powerPublication: audit.powerPublication,
  }));
  return failures;
}
