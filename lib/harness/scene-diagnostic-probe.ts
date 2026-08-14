import type { SceneDescription } from "../core/model";

/** A field checkpoint after solver-specific publications have been normalized. */
export interface SceneFieldCheckpointEvidence {
  time_s: number;
  /** Cell-centred liquid occupancy/volume fractions in x-major order. */
  field: ArrayLike<number>;
}

/** The solver-independent evidence accepted by spatial scene probes. */
export interface SceneFieldEvidence {
  method: string;
  grid: readonly [number, number, number];
  checkpoints: readonly SceneFieldCheckpointEvidence[];
}

/** A failed scene-owned expectation. Empty findings mean the probe passed. */
export interface SceneDiagnosticFinding {
  id: string;
  severity: "error";
  message: string;
  method: string;
  checkpointTime_s?: number;
  expected?: unknown;
  actual?: unknown;
}

export interface SceneDiagnosticProbeResult<Observation> {
  observations: Observation;
  findings: SceneDiagnosticFinding[];
}

export function authoredSceneGrid(scene: SceneDescription): readonly [number, number, number] | undefined {
  const spacing = scene.voxelDomain.finestCellSize_m;
  const extents = [scene.container.width_m, scene.container.height_m, scene.container.depth_m] as const;
  if (!(spacing > 0) || !Number.isFinite(spacing)
    || extents.some((extent) => !(extent > 0) || !Number.isFinite(extent))) return undefined;
  return extents.map((extent) => Math.round(extent / spacing)) as unknown as [number, number, number];
}

export function validEvidenceGrid(
  grid: readonly [number, number, number],
): grid is readonly [number, number, number] {
  return grid.length === 3 && grid.every((dimension) => Number.isInteger(dimension) && dimension > 0);
}

export function usableFieldCheckpoints(
  evidence: SceneFieldEvidence,
  findings: SceneDiagnosticFinding[],
): SceneFieldCheckpointEvidence[] {
  if (!validEvidenceGrid(evidence.grid)) {
    findings.push({
      id: "field-evidence.grid-invalid",
      severity: "error",
      method: evidence.method,
      message: `${evidence.method} field evidence has an invalid grid`,
      expected: "three positive integer dimensions",
      actual: [...evidence.grid],
    });
    return [];
  }

  const cellCount = evidence.grid[0] * evidence.grid[1] * evidence.grid[2];
  const usable: SceneFieldCheckpointEvidence[] = [];
  evidence.checkpoints.forEach((checkpoint, index) => {
    if (!Number.isFinite(checkpoint.time_s) || checkpoint.field.length !== cellCount) {
      findings.push({
        id: "field-evidence.checkpoint-invalid",
        severity: "error",
        method: evidence.method,
        message: `${evidence.method} checkpoint ${index} does not match its normalized field contract`,
        checkpointTime_s: Number.isFinite(checkpoint.time_s) ? checkpoint.time_s : undefined,
        expected: { finiteTime: true, fieldLength: cellCount },
        actual: { time_s: checkpoint.time_s, fieldLength: checkpoint.field.length },
      });
      return;
    }
    usable.push(checkpoint);
  });
  return usable;
}

export function sceneGridFinding(
  scene: SceneDescription,
  evidence: SceneFieldEvidence,
): SceneDiagnosticFinding | undefined {
  const expected = authoredSceneGrid(scene);
  if (!expected) return {
    id: "scene-grid.invalid",
    severity: "error",
    method: evidence.method,
    message: `${scene.sceneId} does not declare a valid positive scene lattice`,
    actual: {
      spacing_m: scene.voxelDomain.finestCellSize_m,
      extents_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
    },
  };
  if (expected.every((dimension, axis) => dimension === evidence.grid[axis])) return undefined;
  return {
    id: "scene-grid.mismatch",
    severity: "error",
    method: evidence.method,
    message: `${evidence.method} grid ${evidence.grid.join("x")} differs from ${scene.sceneId}'s authored ${expected.join("x")} lattice`,
    expected: [...expected],
    actual: [...evidence.grid],
  };
}
