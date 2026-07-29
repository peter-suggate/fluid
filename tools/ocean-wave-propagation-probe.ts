import type { SceneDescription } from "../lib/model";
import {
  sceneGridFinding,
  type SceneDiagnosticFinding,
  type SceneDiagnosticProbeResult,
  type SceneFieldEvidence,
  usableFieldCheckpoints,
} from "./scene-diagnostic-probe";

export interface OceanWavePropagationProbeOptions {
  /** Scene-authored acceptance floor for the largest far-half surface signal. */
  minimumFarHalfDisturbance_cells: number;
  stationCount?: number;
  /** Do not assert propagation until this many usable time samples exist. */
  minimumCheckpointCount?: number;
}

export interface OceanWaveCheckpointObservation {
  time_s: number;
  crestX_m: number;
  crestHeight_cells: number;
  stationHeights_cells: number[];
}

export interface OceanWavePropagationObservation {
  method: string;
  grid: readonly [number, number, number];
  baselineHeight_cells?: number;
  cellHeight_m?: number;
  minimumFarHalfDisturbance_cells: number;
  farHalfDisturbance_cells?: number;
  crestReach_m?: number;
  stationX_m: number[];
  checkpoints: OceanWaveCheckpointObservation[];
}

export function probeOceanWavePropagation(
  scene: SceneDescription,
  evidence: SceneFieldEvidence,
  options: OceanWavePropagationProbeOptions,
): SceneDiagnosticProbeResult<OceanWavePropagationObservation> {
  const findings: SceneDiagnosticFinding[] = [];
  const gridFinding = sceneGridFinding(scene, evidence);
  if (gridFinding) findings.push(gridFinding);
  const checkpoints = usableFieldCheckpoints(evidence, findings);
  const [nx, ny, nz] = evidence.grid;
  const minimumCheckpointCount = Number.isInteger(options.minimumCheckpointCount)
    && (options.minimumCheckpointCount ?? 0) >= 0 ? options.minimumCheckpointCount ?? 3 : 3;
  const stationCount = Number.isInteger(options.stationCount) && (options.stationCount ?? 0) > 0
    ? options.stationCount ?? 12 : 12;
  const threshold = options.minimumFarHalfDisturbance_cells;

  const observation: OceanWavePropagationObservation = {
    method: evidence.method,
    grid: [...evidence.grid],
    minimumFarHalfDisturbance_cells: threshold,
    stationX_m: [],
    checkpoints: [],
  };

  if (!(threshold >= 0) || !Number.isFinite(threshold)) {
    findings.push({
      id: "ocean-wave.threshold-invalid",
      severity: "error",
      method: evidence.method,
      message: "ocean wave propagation requires a finite non-negative disturbance threshold",
      expected: "finite value >= 0",
      actual: threshold,
    });
  }
  if (!(Number.isInteger(nx) && nx > 0 && Number.isInteger(ny) && ny > 0
    && Number.isInteger(nz) && nz > 0)) return { observations: observation, findings };

  const width_m = scene.container.width_m;
  const height_m = scene.container.height_m;
  if (!(Number.isFinite(width_m) && width_m > 0 && Number.isFinite(height_m) && height_m > 0)) {
    return { observations: observation, findings };
  }
  const xWorld = (x: number) => -0.5 * width_m + (x + 0.5) * width_m / nx;
  const stations = Array.from({ length: stationCount }, (_, index) =>
    Math.min(nx - 1, Math.round((index + 0.5) * nx / stationCount)));
  const baselineHeight_cells = scene.container.fillFraction * ny;
  let crestReach_m = -Infinity;
  let farHalfDisturbance_cells = 0;

  observation.baselineHeight_cells = baselineHeight_cells;
  observation.cellHeight_m = height_m / ny;
  observation.stationX_m = stations.map((x) => Number(xWorld(x).toFixed(3)));

  for (const checkpoint of checkpoints) {
    const heights = new Float64Array(nx);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const value = Number(checkpoint.field[x + nx * (y + ny * z)]);
      if (Number.isFinite(value)) heights[x] += value;
    }
    for (let x = 0; x < nx; x += 1) heights[x] /= nz;

    let crestX = 0;
    for (let x = 0; x < nx; x += 1) {
      if (heights[x] > heights[crestX]) crestX = x;
      if (xWorld(x) > 0) {
        farHalfDisturbance_cells = Math.max(
          farHalfDisturbance_cells,
          Math.abs(heights[x] - baselineHeight_cells),
        );
      }
    }
    crestReach_m = Math.max(crestReach_m, xWorld(crestX));
    observation.checkpoints.push({
      time_s: checkpoint.time_s,
      crestX_m: Number(xWorld(crestX).toFixed(3)),
      crestHeight_cells: Number(heights[crestX].toFixed(2)),
      stationHeights_cells: stations.map((x) => Number(heights[x].toFixed(2))),
    });
  }
  observation.farHalfDisturbance_cells = farHalfDisturbance_cells;
  observation.crestReach_m = Number.isFinite(crestReach_m) ? crestReach_m : undefined;

  if (checkpoints.length >= minimumCheckpointCount
    && Number.isFinite(threshold) && threshold >= 0
    && farHalfDisturbance_cells < threshold) {
    findings.push({
      id: "ocean-wave.far-half-disturbance",
      severity: "error",
      method: evidence.method,
      message: `${evidence.method} far-half surface disturbance reached only ${farHalfDisturbance_cells.toFixed(3)} cells (required ${threshold.toFixed(3)}; global crest max x ${crestReach_m.toFixed(3)} m)`,
      expected: { minimum_cells: threshold },
      actual: { disturbance_cells: farHalfDisturbance_cells, crestReach_m },
    });
  }
  return { observations: observation, findings };
}
