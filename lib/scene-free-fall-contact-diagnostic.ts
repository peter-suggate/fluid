import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  fieldCheckpoints,
  gridFromDiagnostics,
  hookFinding,
  numberPath,
  numberValue,
  recordValue,
  runTime,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";

export interface FreeFallContactDiagnosticParameters {
  minimumRunTime_s: number;
  evaluationStart_s: number;
  impactTime_s: number;
  preImpactMargin_s: number;
  minimumMeasuredToAnalyticDropRatio: number;
  maximumMeasuredToAnalyticDropRatio: number;
  maximumDropHeadroom_cells: number;
  minimumPreImpactCheckpoints: number;
  releaseCheckTime_s: number;
  maximumCeilingWetCellsAfterRelease: number;
  maximumCeilingPixelsAfterRelease: number;
  /** Compare corner/seam velocity shortfall with contact-free liquid. */
  includeCornerSeams?: boolean;
  seamEvaluationStart_s?: number;
  maximumSeamShortfallExcess?: number;
}

function wetCeilingCells(
  field: ArrayLike<number>,
  grid: readonly [number, number, number],
  layers = 2,
): number {
  const [nx, ny, nz] = grid;
  let wet = 0;
  for (let z = 0; z < nz; z += 1) for (let y = Math.max(0, ny - layers); y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) if (Number(field[x + nx * (y + ny * z)]) > 0.5) wet += 1;
  }
  return wet;
}

function checkpointCentroidY(checkpoint: Readonly<Record<string, unknown>>): number | undefined {
  return numberValue(checkpoint.centroidY_cells)
    ?? numberPath(checkpoint, "summary", "centroidCells", "y");
}

function checkpointCeilingWet(
  checkpoint: Readonly<Record<string, unknown>>,
  grid: readonly [number, number, number],
): number | undefined {
  const normalized = numberValue(checkpoint.ceilingWetCells);
  if (normalized !== undefined) return normalized;
  const field = checkpoint.field;
  const cellCount = grid[0] * grid[1] * grid[2];
  return field !== null && typeof field === "object" && "length" in field
    && Number((field as ArrayLike<number>).length) === cellCount
    ? wetCeilingCells(field as ArrayLike<number>, grid)
    : undefined;
}

function checkpointCeilingPixels(checkpoint: Readonly<Record<string, unknown>>): number | undefined {
  const normalized = numberValue(checkpoint.ceilingContactPixels);
  if (normalized !== undefined) return normalized;
  const raster = recordValue(checkpoint.raster);
  if (!raster) return undefined;
  const forward = recordValue(raster.ceilingContactPixels);
  const reverse = recordPathValue(raster, "reverseView", "ceilingContactPixels");
  const values = [forward?.front, forward?.back, reverse?.front, reverse?.back];
  return values.every((value) => numberValue(value) !== undefined)
    ? values.reduce<number>((sum, value) => sum + (numberValue(value) ?? 0), 0)
    : undefined;
}

function seamShortfallExcess(checkpoint: Readonly<Record<string, unknown>>): {
  baseline: number;
  seam: number;
  excess: number;
} | undefined {
  const evidence = recordValue(checkpoint.evidence);
  const attribution = recordValue(checkpoint.attribution ?? evidence?.["free-fall-contact-attribution"]);
  const buckets = Array.isArray(attribution?.velocityByContact)
    ? attribution.velocityByContact.flatMap((value) => {
      const bucket = recordValue(value);
      const contacts = numberValue(bucket?.contacts);
      const shortfall = numberValue(bucket?.meanShortfallFraction);
      return bucket && contacts !== undefined && shortfall !== undefined
        ? [{ contacts, shortfall }] : [];
    }) : [];
  const baseline = buckets.find(({ contacts }) => contacts === 0)?.shortfall;
  const seamValues = buckets.filter(({ contacts }) => contacts >= 2).map(({ shortfall }) => shortfall);
  if (baseline === undefined || seamValues.length === 0) return undefined;
  const seam = Math.max(...seamValues);
  return { baseline, seam, excess: seam - baseline };
}

function recordPathValue(value: unknown, ...path: readonly string[]): Readonly<Record<string, unknown>> | undefined {
  let current = recordValue(value);
  for (const segment of path) {
    if (!current) return undefined;
    current = recordValue(current[segment]);
  }
  return current;
}

export function evaluateFreeFallContactDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: FreeFallContactDiagnosticParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const elapsed = runTime(diagnostics);
    if (elapsed === undefined || elapsed < input.parameters.minimumRunTime_s - 1e-9) continue;
    const grid = gridFromDiagnostics(diagnostics);
    const gridValid = grid !== undefined;
    findings.push(hookFinding({
      id: `${method}.grid`, method, passed: gridValid,
      message: gridValid ? "free-fall evidence has a valid grid" : "free-fall evidence has no valid grid",
      expected: "three positive integer dimensions", actual: grid,
    }));
    if (!grid) continue;

    const gravity = input.scene.fluid.gravity_m_s2;
    const g = Math.hypot(gravity.x, gravity.y, gravity.z);
    const cell_m = input.scene.container.height_m / grid[1];
    const initialCentroidY_cells = grid[1] - 4;
    const drop_m = (grid[1] - 8) * cell_m;
    const analyticImpact_s = g > 0 && drop_m >= 0 ? Math.sqrt(2 * drop_m / g) : 0;
    const cutoff_s = Math.min(analyticImpact_s, input.parameters.impactTime_s)
      - input.parameters.preImpactMargin_s;
    let gated = 0;
    fieldCheckpoints(diagnostics).forEach((raw, index) => {
      const checkpoint = recordValue(raw);
      const time_s = numberValue(checkpoint?.time_s);
      if (!checkpoint || time_s === undefined) return;
      if (time_s >= input.parameters.evaluationStart_s - 1e-9 && time_s <= cutoff_s + 1e-9) {
        const centroidY = checkpointCentroidY(checkpoint);
        findings.push(hookFinding({
          id: `${method}.centroid-present.${index}`, method, passed: centroidY !== undefined,
          message: centroidY === undefined
            ? `free-fall oracle has no liquid centroid at t=${time_s.toFixed(2)} s`
            : `free-fall centroid is available at t=${time_s.toFixed(2)} s`,
        }));
        if (centroidY !== undefined) {
          gated += 1;
          const analyticDrop = 0.5 * g * time_s * time_s / cell_m;
          const measuredDrop = initialCentroidY_cells - centroidY;
          const minimum = input.parameters.minimumMeasuredToAnalyticDropRatio * analyticDrop;
          const maximum = input.parameters.maximumMeasuredToAnalyticDropRatio * analyticDrop
            + input.parameters.maximumDropHeadroom_cells;
          findings.push(hookFinding({
            id: `${method}.drop-minimum.${index}`, method, passed: measuredDrop >= minimum,
            message: measuredDrop >= minimum
              ? `free-fall drop kept pace with gravity at t=${time_s.toFixed(2)} s`
              : `free-fall brick centroid fell ${measuredDrop.toFixed(2)} cells at t=${time_s.toFixed(2)} s; free fall predicts ${analyticDrop.toFixed(2)}: the liquid is adhering to the boundary`,
            expected: { minimumDrop_cells: minimum }, actual: { measuredDrop_cells: measuredDrop },
          }));
          findings.push(hookFinding({
            id: `${method}.drop-maximum.${index}`, method, passed: measuredDrop <= maximum,
            message: measuredDrop <= maximum
              ? `free-fall drop did not over-accelerate at t=${time_s.toFixed(2)} s`
              : `free-fall brick centroid fell ${measuredDrop.toFixed(2)} cells at t=${time_s.toFixed(2)} s; free fall predicts ${analyticDrop.toFixed(2)}: the liquid is over-accelerating`,
            expected: { maximumDrop_cells: maximum }, actual: { measuredDrop_cells: measuredDrop },
          }));
        }
      }

      if (input.parameters.includeCornerSeams === true
        && time_s >= (input.parameters.seamEvaluationStart_s ?? Infinity) - 1e-9) {
        const comparison = seamShortfallExcess(checkpoint);
        const maximum = input.parameters.maximumSeamShortfallExcess;
        const passed = comparison !== undefined && maximum !== undefined
          && comparison.excess <= maximum;
        findings.push(hookFinding({
          id: `${method}.seam-shortfall.${index}`, method, passed,
          message: passed
            ? `corner/seam velocity shortfall stayed within the contact-free envelope at t=${time_s.toFixed(2)} s`
            : comparison
              ? `corner/seam velocity shortfall exceeded contact-free liquid by ${comparison.excess.toFixed(4)} at t=${time_s.toFixed(2)} s; maximum ${maximum ?? "undeclared"}`
              : `corner/seam shortfall attribution is unavailable at t=${time_s.toFixed(2)} s`,
          expected: { maximumExcess: maximum }, actual: comparison,
        }));
      }

      if (time_s < input.parameters.releaseCheckTime_s - 1e-9) return;
      const ceilingWet = checkpointCeilingWet(checkpoint, grid);
      if (ceilingWet !== undefined) findings.push(hookFinding({
        id: `${method}.ceiling-wet.${index}`, method,
        passed: ceilingWet <= input.parameters.maximumCeilingWetCellsAfterRelease,
        message: ceilingWet <= input.parameters.maximumCeilingWetCellsAfterRelease
          ? `coarse liquid separated from the ceiling at t=${time_s.toFixed(2)} s`
          : `free-fall oracle holds ${ceilingWet} wet cells in its top two layers at t=${time_s.toFixed(2)} s`,
        expected: { maximum: input.parameters.maximumCeilingWetCellsAfterRelease }, actual: ceilingWet,
      }));
      const ceilingPixels = checkpointCeilingPixels(checkpoint);
      if (ceilingPixels !== undefined) findings.push(hookFinding({
        id: `${method}.ceiling-pixels.${index}`, method,
        passed: ceilingPixels <= input.parameters.maximumCeilingPixelsAfterRelease,
        message: ceilingPixels <= input.parameters.maximumCeilingPixelsAfterRelease
          ? `fine surface separated from the ceiling at t=${time_s.toFixed(2)} s`
          : `free-fall oracle still renders ${ceilingPixels} fine-surface ceiling pixels at t=${time_s.toFixed(2)} s`,
        expected: { maximum: input.parameters.maximumCeilingPixelsAfterRelease }, actual: ceilingPixels,
      }));
    });
    findings.push(hookFinding({
      id: `${method}.pre-impact-checkpoints`, method,
      passed: gated >= input.parameters.minimumPreImpactCheckpoints,
      message: gated >= input.parameters.minimumPreImpactCheckpoints
        ? `free-fall oracle observed ${gated} pre-impact checkpoints`
        : `free-fall oracle observed only ${gated} pre-impact checkpoints; expected at least ${input.parameters.minimumPreImpactCheckpoints}`,
      expected: { minimum: input.parameters.minimumPreImpactCheckpoints }, actual: gated,
    }));
  }
  return findings;
}
