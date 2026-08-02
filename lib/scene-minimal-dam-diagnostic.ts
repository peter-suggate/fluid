import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  arrayPath,
  fieldCheckpoints,
  gridFromDiagnostics,
  hookFinding,
  numberPath,
  numberValue,
  recordPath,
  recordValue,
  runTime,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";

export interface DiagnosticLimitScheduleEntry {
  before_s?: number;
  maximum: number;
}

export interface MinimalDamMotionDiagnosticParameters {
  minimumPeakSpeed_m_s: number;
  minimumLateralSpread_m: number;
  maximumMechanicalEnergyRetention?: number;
  maximumRitterCelerityRatio?: number;
  energyEvaluationAfter_s?: number;
}

export interface MinimalDamCeilingSeparationParameters {
  evaluateAfter_s: number;
  wetCellsStart_s: number;
  contactPixelsStart_s: number;
  ceilingWetCellLimits: readonly DiagnosticLimitScheduleEntry[];
  ceilingContactPixelLimits: readonly DiagnosticLimitScheduleEntry[];
}

function boundsValue(value: unknown): readonly [readonly [number, number, number], readonly [number, number, number]] | undefined {
  if (!Array.isArray(value) || value.length !== 2) return undefined;
  if (!value.every((point) => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))) return undefined;
  return value as unknown as readonly [readonly [number, number, number], readonly [number, number, number]];
}

function scheduledLimit(schedule: readonly DiagnosticLimitScheduleEntry[], time_s: number): number | undefined {
  for (const entry of schedule) {
    if (Number.isFinite(entry.maximum)
      && (entry.before_s === undefined || time_s < entry.before_s - 1e-9)) return entry.maximum;
  }
  return undefined;
}

function aggregateCeilingContactPixels(raster: unknown): number | undefined {
  const value = recordValue(raster);
  if (!value) return undefined;
  const direct = numberValue(value.ceilingContactPixels);
  if (direct !== undefined) return direct;
  const forward = recordValue(value.ceilingContactPixels);
  const reverse = recordPath(value, "reverseView", "ceilingContactPixels");
  const samples = [forward?.front, forward?.back, reverse?.front, reverse?.back];
  return samples.every((sample) => numberValue(sample) !== undefined)
    ? samples.reduce<number>((sum, sample) => sum + (numberValue(sample) ?? 0), 0)
    : undefined;
}

function checkpointCeilingWetCells(
  checkpoint: Readonly<Record<string, unknown>>,
  grid: readonly [number, number, number] | undefined,
): number | undefined {
  const normalized = numberValue(checkpoint.ceilingWetCells);
  if (normalized !== undefined) return normalized;
  if (!grid) return undefined;
  const field = checkpoint.field;
  const cellCount = grid[0] * grid[1] * grid[2];
  if (field === null || typeof field !== "object" || !("length" in field)
      || Number((field as ArrayLike<number>).length) !== cellCount) return undefined;
  let wet = 0;
  const [nx, ny, nz] = grid;
  for (let z = 0; z < nz; z += 1) {
    for (let y = Math.max(0, ny - 2); y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if (Number((field as ArrayLike<number>)[x + nx * (y + ny * z)]) > 0.5) wet += 1;
      }
    }
  }
  return wet;
}

export function evaluateMinimalDamMotionDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: MinimalDamMotionDiagnosticParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const stability = recordPath(diagnostics, "stability") ?? recordPath(diagnostics, "stabilityEnvelope");
    const speed = numberPath(stability, "peakLiquidSpeed_m_s")
      ?? numberPath(diagnostics, "solver", "maxSpeed_m_s");
    findings.push(hookFinding({
      id: `${method}.peak-speed`, method,
      passed: speed !== undefined && speed >= input.parameters.minimumPeakSpeed_m_s,
      message: speed !== undefined && speed >= input.parameters.minimumPeakSpeed_m_s
        ? `minimal dam reached ${speed} m/s`
        : `minimal dam observed motion speed ${speed ?? "unknown"} m/s below ${input.parameters.minimumPeakSpeed_m_s} m/s`,
      expected: { minimum_m_s: input.parameters.minimumPeakSpeed_m_s }, actual: speed,
    }));

    const raster = recordPath(diagnostics, "raster") ?? diagnostics;
    // Checkpoint-authored lanes publish no explicit t=0/final raster; their
    // first and last checkpoint carry that closure evidence instead. Without
    // this fallback the spread resolves to `unknown` and reads as a physics
    // failure. Mirrors `scene-water-raster-integrity-diagnostic`.
    const rasterCheckpoints = (arrayPath(raster, "checkpoints")
      ?? arrayPath(diagnostics, "globalFineGenerationCheckpoints") ?? [])
      .map(recordValue).filter((value) => value !== undefined);
    const initialBounds = boundsValue(recordPath(raster, "initial")?.frontInterfaceBounds_m
      ?? recordPath(diagnostics, "initialGlobalFineRaster")?.frontInterfaceBounds_m
      ?? recordPath(rasterCheckpoints[0], "raster")?.frontInterfaceBounds_m);
    const finalBounds = boundsValue(recordPath(raster, "final")?.frontInterfaceBounds_m
      ?? recordPath(diagnostics, "finalGlobalFineRaster")?.frontInterfaceBounds_m
      ?? recordPath(rasterCheckpoints.at(-1), "raster")?.frontInterfaceBounds_m);
    const spread = initialBounds && finalBounds
      ? Math.max(finalBounds[1][0] - initialBounds[1][0], finalBounds[1][2] - initialBounds[1][2])
      : undefined;
    findings.push(hookFinding({
      id: `${method}.lateral-spread`, method,
      passed: spread !== undefined && spread >= input.parameters.minimumLateralSpread_m,
      message: spread !== undefined && spread >= input.parameters.minimumLateralSpread_m
        ? `minimal dam interface spread ${spread} m`
        : `minimal dam interface spread ${spread ?? "unknown"} m is below ${input.parameters.minimumLateralSpread_m} m`,
      expected: { minimum_m: input.parameters.minimumLateralSpread_m }, actual: spread,
    }));

    const elapsed = runTime(diagnostics) ?? 0;
    const evaluateEnergyAfter = input.parameters.energyEvaluationAfter_s ?? 1;
    if (elapsed < evaluateEnergyAfter - 1e-9) continue;
    const energySamples = arrayPath(diagnostics, "energy", "checkpoints")
      ?? (Array.isArray(diagnostics.compactMechanicalEnergyCheckpoints)
        ? diagnostics.compactMechanicalEnergyCheckpoints : []);
    if (input.parameters.maximumMechanicalEnergyRetention !== undefined) {
      const retentions = energySamples.flatMap((sample) => {
        const value = numberPath(sample, "mechanicalEnergyRetentionRatio");
        return value === undefined ? [] : [value];
      });
      const worst = retentions.length > 0 ? Math.max(...retentions) : undefined;
      findings.push(hookFinding({
        id: `${method}.mechanical-energy-retention`, method,
        passed: worst !== undefined && worst <= input.parameters.maximumMechanicalEnergyRetention,
        message: worst !== undefined && worst <= input.parameters.maximumMechanicalEnergyRetention
          ? `motion mechanical-energy proxy peak ${worst} stayed within its envelope`
          : `motion mechanical-energy proxy peak ${worst ?? "unknown"} exceeds the declared ${input.parameters.maximumMechanicalEnergyRetention} envelope`,
        expected: { maximum: input.parameters.maximumMechanicalEnergyRetention }, actual: worst,
      }));
    }
    if (input.parameters.maximumRitterCelerityRatio !== undefined) {
      const speeds = energySamples.flatMap((sample) => {
        const value = numberPath(sample, "maximumLiquidComponentSpeed_m_s");
        return value === undefined ? [] : [value];
      });
      const peak = speeds.length > 0 ? Math.max(...speeds) : undefined;
      const gravity = input.scene.fluid.gravity_m_s2;
      const gravityMagnitude = Math.hypot(gravity.x, gravity.y, gravity.z);
      const columnHeight_m = Math.max(0.92, input.scene.container.fillFraction)
        * input.scene.container.height_m;
      const celerity = 2 * Math.sqrt(gravityMagnitude * columnHeight_m);
      const maximum = input.parameters.maximumRitterCelerityRatio * celerity;
      findings.push(hookFinding({
        id: `${method}.ritter-celerity`, method,
        passed: peak !== undefined && peak <= maximum,
        message: peak !== undefined && peak <= maximum
          ? `motion liquid peak ${peak} m/s stayed within its Ritter celerity envelope`
          : `motion liquid reached ${peak ?? "unknown"} m/s; the declared ${input.parameters.maximumRitterCelerityRatio}x Ritter celerity bound is ${maximum} m/s`,
        expected: { maximum_m_s: maximum }, actual: peak,
      }));
    }
  }
  return findings;
}

export function evaluateMinimalDamCeilingSeparation(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: MinimalDamCeilingSeparationParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const elapsed = runTime(diagnostics);
    if (elapsed === undefined || elapsed < input.parameters.evaluateAfter_s - 1e-9) continue;
    const grid = gridFromDiagnostics(diagnostics);
    fieldCheckpoints(diagnostics).forEach((raw, index) => {
      const checkpoint = recordValue(raw);
      const time_s = numberValue(checkpoint?.time_s);
      if (!checkpoint || time_s === undefined) return;
      if (time_s >= input.parameters.wetCellsStart_s - 1e-9) {
        const wet = checkpointCeilingWetCells(checkpoint, grid);
        const limit = scheduledLimit(input.parameters.ceilingWetCellLimits, time_s);
        findings.push(hookFinding({
          id: `${method}.ceiling-wet.${index}`, method,
          passed: wet !== undefined && limit !== undefined && wet <= limit,
          message: wet !== undefined && limit !== undefined && wet <= limit
            ? `minimal dam coarse ceiling contact passed at t=${time_s.toFixed(2)} s`
            : `minimal dam holds ${wet ?? "unknown"} wet cells in its top two layers at t=${time_s.toFixed(2)} s; limit ${limit ?? "undeclared"}`,
          expected: { maximum: limit }, actual: wet,
        }));
      }
      if (time_s >= input.parameters.contactPixelsStart_s - 1e-9) {
        const pixels = aggregateCeilingContactPixels(checkpoint.raster);
        const limit = scheduledLimit(input.parameters.ceilingContactPixelLimits, time_s);
        findings.push(hookFinding({
          id: `${method}.ceiling-pixels.${index}`, method,
          passed: pixels !== undefined && limit !== undefined && pixels <= limit,
          message: pixels !== undefined && limit !== undefined && pixels <= limit
            ? `minimal dam fine ceiling contact passed at t=${time_s.toFixed(2)} s`
            : `minimal dam has ${pixels ?? "unknown"} visible fine-surface ceiling pixels at t=${time_s.toFixed(2)} s; limit ${limit ?? "undeclared"}`,
          expected: { maximum: limit }, actual: pixels,
        }));
      }
    });
  }
  return findings;
}
