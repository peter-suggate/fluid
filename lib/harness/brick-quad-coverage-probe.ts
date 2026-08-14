import type { SceneDescription } from "../core/model";
import {
  authoredSceneGrid,
  sceneGridFinding,
  type SceneDiagnosticFinding,
  type SceneDiagnosticProbeResult,
  type SceneFieldEvidence,
  usableFieldCheckpoints,
} from "./scene-diagnostic-probe";

export interface BrickQuadCoverageProbeOptions {
  liquidThreshold?: number;
  /** The authored footprint this diagnostic expects to exercise. */
  expectedBrickGrid?: readonly [number, number, number];
  minimumFirstCheckpointColumns?: number;
}

export interface BrickQuadCheckpointObservation {
  time_s: number;
  wetBrickColumns: string[];
}

export interface BrickQuadCoverageObservation {
  method: string;
  grid: readonly [number, number, number];
  brickGrid?: readonly [number, number, number];
  wetBrickColumns: string[];
  farColumn: string;
  checkpoints: BrickQuadCheckpointObservation[];
}

export function probeBrickQuadCoverage(
  scene: SceneDescription,
  evidence: SceneFieldEvidence,
  options: BrickQuadCoverageProbeOptions = {},
): SceneDiagnosticProbeResult<BrickQuadCoverageObservation> {
  const findings: SceneDiagnosticFinding[] = [];
  const gridFinding = sceneGridFinding(scene, evidence);
  if (gridFinding) findings.push(gridFinding);
  const checkpoints = usableFieldCheckpoints(evidence, findings);
  const [nx, ny, nz] = evidence.grid;
  const brickSize = scene.voxelDomain.brickSize_cells;
  const authoredGrid = authoredSceneGrid(scene);
  const brickGrid = authoredGrid && authoredGrid.every((dimension) => dimension % brickSize === 0)
    ? authoredGrid.map((dimension) => dimension / brickSize) as unknown as [number, number, number]
    : undefined;
  const expectedBrickGrid = options.expectedBrickGrid ?? [2, 1, 2];
  const farColumn = `${Math.max(0, expectedBrickGrid[0] - 1)},${Math.max(0, expectedBrickGrid[2] - 1)}`;
  const observation: BrickQuadCoverageObservation = {
    method: evidence.method,
    grid: [...evidence.grid],
    brickGrid,
    wetBrickColumns: [],
    farColumn,
    checkpoints: [],
  };

  if (!brickGrid || !brickGrid.every((dimension, axis) => dimension === expectedBrickGrid[axis])) {
    findings.push({
      id: "brick-quad.authored-grid",
      severity: "error",
      method: evidence.method,
      message: `${scene.sceneId} does not declare the brick footprint required by its coverage diagnostic`,
      expected: [...expectedBrickGrid],
      actual: brickGrid ? [...brickGrid] : undefined,
    });
  }
  if (!(Number.isInteger(nx) && nx > 0 && Number.isInteger(ny) && ny > 0
    && Number.isInteger(nz) && nz > 0)) return { observations: observation, findings };

  const threshold = Number.isFinite(options.liquidThreshold) ? options.liquidThreshold ?? 0.5 : 0.5;
  const wetColumns = (field: ArrayLike<number>): Set<string> => {
    const wet = new Set<string>();
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      if (Number(field[x + nx * (y + ny * z)]) >= threshold) {
        wet.add(`${Math.floor(x / brickSize)},${Math.floor(z / brickSize)}`);
      }
    }
    return wet;
  };

  const everWet = new Set<string>();
  for (const checkpoint of checkpoints) {
    const columns = [...wetColumns(checkpoint.field)].sort();
    columns.forEach((column) => everWet.add(column));
    observation.checkpoints.push({ time_s: checkpoint.time_s, wetBrickColumns: columns });
  }
  observation.wetBrickColumns = [...everWet].sort();

  if (checkpoints.length > 0) {
    const minimumFirst = options.minimumFirstCheckpointColumns ?? 2;
    const first = observation.checkpoints[0];
    if (first.wetBrickColumns.length < minimumFirst) findings.push({
      id: "brick-quad.first-boundary-crossing",
      severity: "error",
      method: evidence.method,
      checkpointTime_s: first.time_s,
      message: `${evidence.method} water had not crossed a brick boundary by t=${first.time_s.toFixed(2)} s`,
      expected: { minimumWetBrickColumns: minimumFirst },
      actual: { wetBrickColumns: first.wetBrickColumns },
    });

    const expectedColumnCount = expectedBrickGrid[0] * expectedBrickGrid[2];
    if (everWet.size !== expectedColumnCount) findings.push({
      id: "brick-quad.all-columns",
      severity: "error",
      method: evidence.method,
      message: `${evidence.method} wet only ${everWet.size} of ${expectedColumnCount} brick columns (${[...everWet].sort().join(" | ")})`,
      expected: { wetBrickColumns: expectedColumnCount },
      actual: { wetBrickColumns: [...everWet].sort() },
    });
    if (!observation.checkpoints.some((checkpoint) => checkpoint.wetBrickColumns.includes(farColumn))) {
      findings.push({
        id: "brick-quad.far-column",
        severity: "error",
        method: evidence.method,
        message: `${evidence.method} water never reached the far (+x/+z) brick quadrant opposite the seed`,
        expected: { farColumn },
        actual: { wetBrickColumns: [...everWet].sort() },
      });
    }
  }
  return { observations: observation, findings };
}
