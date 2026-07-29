import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  fieldCheckpoints,
  gridFromDiagnostics,
  hookFinding,
  numberPath,
  recordPath,
  recordValue,
  selectedMethodDiagnostics,
  type UnknownRecord,
} from "./scene-hook-evidence";

export interface FieldVelocityParityLimits {
  maximumWeightedRelativeL2: number;
  minimumCosineSimilarity: number;
  minimumEnergyRatio: number;
  maximumEnergyRatio: number;
  minimumPeakRatio: number;
  maximumPeakRatio: number;
}

export interface FieldVelocityParityDiagnosticParameters {
  candidateMethod: string;
  referenceMethod: string;
  limits: FieldVelocityParityLimits;
  minimumWetIntersectionOverUnion: number;
  maximumCentroidDistanceCells: number;
  checkpointTimeTolerance_s: number;
}

interface ScalarComparison {
  wetIntersectionOverUnion: number;
  centroidDistanceCells: number | null;
}

interface VelocityComparison {
  comparedCells: number;
  weightedRelativeL2: number;
  cosineSimilarity: number;
  candidateToReferenceEnergyRatio: number;
  candidateToReferencePeakRatio: number;
}

function compareVelocityFieldsSafe(
  candidate: ArrayLike<number>,
  reference: ArrayLike<number>,
  candidateVolume: ArrayLike<number>,
  referenceVolume: ArrayLike<number>,
): VelocityComparison {
  const cells = candidate.length / 3;
  let error2 = 0, candidate2 = 0, reference2 = 0, dot = 0;
  let candidatePeak = 0, referencePeak = 0, comparedCells = 0;
  for (let cell = 0; cell < cells; cell += 1) {
    const weight = Math.max(0, Math.min(1,
      Math.min(Number(candidateVolume[cell]), Number(referenceVolume[cell]))));
    if (!(weight > 1e-4)) continue;
    comparedCells += 1;
    let candidateLocal2 = 0, referenceLocal2 = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      const c = Number(candidate[3 * cell + axis]), r = Number(reference[3 * cell + axis]);
      if (!Number.isFinite(c) || !Number.isFinite(r)) return {
        comparedCells, weightedRelativeL2: Infinity, cosineSimilarity: -1,
        candidateToReferenceEnergyRatio: Infinity, candidateToReferencePeakRatio: Infinity,
      };
      error2 += weight * (c - r) ** 2;
      candidate2 += weight * c * c; reference2 += weight * r * r; dot += weight * c * r;
      candidateLocal2 += c * c; referenceLocal2 += r * r;
    }
    candidatePeak = Math.max(candidatePeak, Math.sqrt(candidateLocal2));
    referencePeak = Math.max(referencePeak, Math.sqrt(referenceLocal2));
  }
  return {
    comparedCells,
    weightedRelativeL2: Math.sqrt(error2 / Math.max(reference2, 1e-30)),
    cosineSimilarity: dot / Math.sqrt(Math.max(candidate2 * reference2, 1e-30)),
    candidateToReferenceEnergyRatio: candidate2 / Math.max(reference2, 1e-30),
    candidateToReferencePeakRatio: candidatePeak / Math.max(referencePeak, 1e-30),
  };
}

function arrayLike(value: unknown): ArrayLike<number> | undefined {
  if (value === null || typeof value !== "object" || !("length" in value)) return undefined;
  const length = Number((value as { length: unknown }).length);
  return Number.isInteger(length) && length >= 0 ? value as ArrayLike<number> : undefined;
}

function fieldValue(container: unknown): ArrayLike<number> | undefined {
  const direct = arrayLike(container);
  if (direct) return direct;
  const record = recordValue(container);
  return arrayLike(record?.field ?? record?.values);
}

function selectedField(diagnostics: UnknownRecord): ArrayLike<number> | undefined {
  const field = recordPath(diagnostics, "field");
  const final = fieldValue(field?.final);
  if (final) return final;
  const checkpoints = fieldCheckpoints(diagnostics);
  const checkpoint = checkpoints.length > 0 ? recordValue(checkpoints.at(-1)) : undefined;
  return fieldValue(checkpoint) ?? fieldValue(field?.matched) ?? fieldValue(diagnostics.matchedField);
}

function compareScalarFieldsSafe(
  candidate: ArrayLike<number>,
  reference: ArrayLike<number>,
  grid: readonly [number, number, number],
): ScalarComparison | undefined {
  const [nx, ny, nz] = grid;
  const count = nx * ny * nz;
  if (candidate.length !== count || reference.length !== count) return undefined;
  let intersection = 0, union = 0;
  let candidateMass = 0, referenceMass = 0;
  const candidateMoment = [0, 0, 0], referenceMoment = [0, 0, 0];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = x + nx * (y + ny * z);
    const a = Number(candidate[index]), b = Number(reference[index]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return undefined;
    if (a >= 0.5 && b >= 0.5) intersection += 1;
    if (a >= 0.5 || b >= 0.5) union += 1;
    if (a > 0) {
      candidateMass += a; candidateMoment[0] += a * x; candidateMoment[1] += a * y; candidateMoment[2] += a * z;
    }
    if (b > 0) {
      referenceMass += b; referenceMoment[0] += b * x; referenceMoment[1] += b * y; referenceMoment[2] += b * z;
    }
  }
  const centroidDistanceCells = candidateMass > 0 && referenceMass > 0
    ? Math.hypot(
      candidateMoment[0] / candidateMass - referenceMoment[0] / referenceMass,
      candidateMoment[1] / candidateMass - referenceMoment[1] / referenceMass,
      candidateMoment[2] / candidateMass - referenceMoment[2] / referenceMass,
    ) : null;
  return { wetIntersectionOverUnion: union > 0 ? intersection / union : 1, centroidDistanceCells };
}

function comparisonFindings(input: {
  id: string;
  method: string;
  comparison: ScalarComparison | undefined;
  minimumIoU: number;
  maximumCentroid: number;
  time_s?: number;
}): SceneDiagnosticHookFinding[] {
  const suffix = input.time_s === undefined ? "" : ` at t=${input.time_s.toFixed(2)} s`;
  return [
    hookFinding({
      id: `${input.id}.wet-iou`, method: input.method,
      passed: input.comparison !== undefined
        && input.comparison.wetIntersectionOverUnion >= input.minimumIoU,
      message: input.comparison !== undefined
        && input.comparison.wetIntersectionOverUnion >= input.minimumIoU
        ? `wet-field IoU passed${suffix}`
        : `wet-field IoU ${input.comparison?.wetIntersectionOverUnion ?? "unavailable"}${suffix} is below ${input.minimumIoU}`,
      expected: { minimum: input.minimumIoU }, actual: input.comparison?.wetIntersectionOverUnion,
    }),
    hookFinding({
      id: `${input.id}.centroid`, method: input.method,
      passed: input.comparison !== undefined && (input.comparison.centroidDistanceCells === null
        || input.comparison.centroidDistanceCells <= input.maximumCentroid),
      message: input.comparison !== undefined && (input.comparison.centroidDistanceCells === null
        || input.comparison.centroidDistanceCells <= input.maximumCentroid)
        ? `wet-field centroid parity passed${suffix}`
        : `wet-field centroid differs by ${input.comparison?.centroidDistanceCells ?? "unavailable"} cells${suffix}`,
      expected: { maximum: input.maximumCentroid }, actual: input.comparison?.centroidDistanceCells,
    }),
  ];
}

export function evaluateFieldVelocityParityDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: FieldVelocityParityDiagnosticParameters;
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  const methods = new Map(selectedMethodDiagnostics(input.evidence,
    [input.parameters.candidateMethod, input.parameters.referenceMethod]));
  const candidate = methods.get(input.parameters.candidateMethod);
  const reference = methods.get(input.parameters.referenceMethod);
  const methodLabel = `${input.parameters.candidateMethod}-vs-${input.parameters.referenceMethod}`;
  findings.push(hookFinding({
    id: `${methodLabel}.methods-present`,
    passed: candidate !== undefined && reference !== undefined,
    message: candidate && reference ? "both parity methods published evidence" : "one or both parity methods are missing",
    expected: [input.parameters.candidateMethod, input.parameters.referenceMethod],
    actual: [...methods.keys()],
  }));
  if (!candidate || !reference) return findings;
  const candidateGrid = gridFromDiagnostics(candidate), referenceGrid = gridFromDiagnostics(reference);
  const gridsMatch = candidateGrid !== undefined && referenceGrid !== undefined
    && candidateGrid.every((dimension, axis) => dimension === referenceGrid[axis]);
  findings.push(hookFinding({
    id: `${methodLabel}.grid`, passed: gridsMatch,
    message: gridsMatch ? "parity grids match" : "parity grids are missing or differ",
    expected: referenceGrid, actual: candidateGrid,
  }));
  if (!candidateGrid || !referenceGrid || !gridsMatch) return findings;

  const candidateStability = recordPath(candidate, "stability") ?? recordPath(candidate, "stabilityEnvelope");
  const referenceStability = recordPath(reference, "stability") ?? recordPath(reference, "stabilityEnvelope");
  const candidatePeak = numberPath(candidateStability, "peakLiquidSpeed_m_s");
  const referencePeak = numberPath(referenceStability, "peakLiquidSpeed_m_s");
  const peakRatio = candidatePeak !== undefined && referencePeak !== undefined
    ? candidatePeak / Math.max(referencePeak, 1e-9) : undefined;
  findings.push(hookFinding({
    id: `${methodLabel}.peak-speed-ratio`, passed: peakRatio !== undefined
      && peakRatio >= input.parameters.limits.minimumPeakRatio
      && peakRatio <= input.parameters.limits.maximumPeakRatio,
    message: peakRatio !== undefined
      && peakRatio >= input.parameters.limits.minimumPeakRatio
      && peakRatio <= input.parameters.limits.maximumPeakRatio
      ? "peak-speed ratio passed" : `peak-speed ratio ${peakRatio ?? "unavailable"} is outside its declared envelope`,
    expected: { minimum: input.parameters.limits.minimumPeakRatio, maximum: input.parameters.limits.maximumPeakRatio },
    actual: peakRatio,
  }));

  const candidateVelocity = recordPath(candidate, "terminalEvidence", "collocated-velocity") ?? {};
  const referenceVelocity = recordPath(reference, "terminalEvidence", "collocated-velocity") ?? {};
  const candidateVector = fieldValue(candidateVelocity.field);
  const referenceVector = fieldValue(referenceVelocity.field);
  const candidateVolume = fieldValue(candidateVelocity.volume);
  const referenceVolume = fieldValue(referenceVelocity.volume);
  const compact = recordValue(candidateVelocity.compactRaster);
  const cellCount = candidateGrid[0] * candidateGrid[1] * candidateGrid[2];
  const publicationValid = compact?.publicationValid === true
    && numberPath(compact, "coveredCells") === cellCount
    && numberPath(compact, "overlapCells") === 0
    && numberPath(compact, "invalidRows") === 0;
  findings.push(hookFinding({
    id: `${methodLabel}.compact-velocity-publication`, passed: publicationValid,
    message: publicationValid ? "compact velocity raster is a clean cubic partition"
      : "compact velocity raster is absent, invalid, overlapping, or incomplete",
    expected: { publicationValid: true, coveredCells: cellCount, overlapCells: 0, invalidRows: 0 }, actual: compact,
  }));

  const velocityShapesValid = candidateVector?.length === cellCount * 3
    && referenceVector?.length === cellCount * 3
    && candidateVolume?.length === cellCount && referenceVolume?.length === cellCount;
  findings.push(hookFinding({
    id: `${methodLabel}.velocity-fields`, passed: velocityShapesValid,
    message: velocityShapesValid ? "collocated velocity and occupancy fields are complete"
      : "collocated velocity or occupancy evidence is unavailable or malformed",
  }));
  if (velocityShapesValid && candidateVector && referenceVector && candidateVolume && referenceVolume) {
    const metrics = compareVelocityFieldsSafe(candidateVector, referenceVector, candidateVolume, referenceVolume);
    const velocityChecks = [
      ["shared-cells", "no shared liquid velocity cells", metrics.comparedCells > 0],
      ["relative-l2", "velocity relative L2", metrics.weightedRelativeL2 <= input.parameters.limits.maximumWeightedRelativeL2],
      ["direction-cosine", "velocity direction cosine", metrics.cosineSimilarity >= input.parameters.limits.minimumCosineSimilarity],
      ["energy-ratio", "velocity energy ratio", metrics.candidateToReferenceEnergyRatio >= input.parameters.limits.minimumEnergyRatio
        && metrics.candidateToReferenceEnergyRatio <= input.parameters.limits.maximumEnergyRatio],
      ["peak-ratio", "velocity peak ratio", metrics.candidateToReferencePeakRatio >= input.parameters.limits.minimumPeakRatio
        && metrics.candidateToReferencePeakRatio <= input.parameters.limits.maximumPeakRatio],
    ] as const;
    for (const [id, failureLabel, passed] of velocityChecks) findings.push(hookFinding({
      id: `${methodLabel}.velocity-${id}`, passed,
      message: passed ? `${failureLabel} passed` : `${failureLabel} failed declared limits`,
      actual: metrics, expected: input.parameters.limits,
    }));
  }

  const finalComparison = compareScalarFieldsSafe(
    selectedField(candidate) ?? [], selectedField(reference) ?? [], candidateGrid,
  );
  findings.push(...comparisonFindings({
    id: `${methodLabel}.final-field`, method: input.parameters.candidateMethod,
    comparison: finalComparison,
    minimumIoU: input.parameters.minimumWetIntersectionOverUnion,
    maximumCentroid: input.parameters.maximumCentroidDistanceCells,
  }));

  const referenceCheckpoints = fieldCheckpoints(reference).flatMap((value) => {
    const checkpoint = recordValue(value), time_s = numberPath(checkpoint, "time_s");
    return checkpoint && time_s !== undefined ? [{ checkpoint, time_s }] : [];
  });
  fieldCheckpoints(candidate).forEach((value, index) => {
    const checkpoint = recordValue(value), time_s = numberPath(checkpoint, "time_s");
    if (!checkpoint || time_s === undefined) return;
    const matched = referenceCheckpoints.find((entry) =>
      Math.abs(entry.time_s - time_s) <= input.parameters.checkpointTimeTolerance_s);
    if (!matched) return;
    const comparison = compareScalarFieldsSafe(
      fieldValue(checkpoint) ?? [], fieldValue(matched.checkpoint) ?? [], candidateGrid,
    );
    findings.push(...comparisonFindings({
      id: `${methodLabel}.checkpoint-${index}`, method: input.parameters.candidateMethod,
      comparison,
      minimumIoU: input.parameters.minimumWetIntersectionOverUnion,
      maximumCentroid: input.parameters.maximumCentroidDistanceCells,
      time_s,
    }));
  });
  return findings;
}
