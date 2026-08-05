import type { SceneDescription } from "./model";
import { primitiveVolume } from "./rigid-body";
import type { NormalizedSceneDiagnosticEvidence, RuntimeDiagnosticFinding } from "./scene-diagnostic-runtime";
import { arrayPath, fieldCheckpoints, gridFromDiagnostics, hookFinding, numberPath, recordPath, recordValue } from "./scene-hook-evidence";

interface CouplingCheckpoint {
  time_s: number;
  checkpoint: Readonly<Record<string, unknown>>;
  snapshot: Readonly<Record<string, unknown>>;
  body: Readonly<Record<string, unknown>>;
}

function scalarField(checkpoint: Readonly<Record<string, unknown>>): ArrayLike<number> | undefined {
  const value = checkpoint.field as ArrayLike<number> | undefined;
  return value && typeof value.length === "number" ? value : undefined;
}

function surfaceHeight(checkpoint: Readonly<Record<string, unknown>>,
  grid: readonly [number, number, number], height_m: number): number | undefined {
  const field = scalarField(checkpoint);
  if (!field || field.length < grid[0] * grid[1] * grid[2]) return undefined;
  const tops: number[] = [];
  for (let z = 0; z < grid[2]; z += 1) for (let x = 0; x < grid[0]; x += 1) {
    let heightCells = 0;
    for (let y = 0; y < grid[1]; y += 1) {
      heightCells += Math.max(0, Math.min(1,
        Number(field[x + grid[0] * (y + grid[1] * z)])));
    }
    if (heightCells > 0) tops.push(heightCells * height_m / grid[1]);
  }
  if (tops.length === 0) return undefined;
  tops.sort((a, b) => a - b);
  return tops[Math.floor(tops.length / 2)];
}

function sphereCapFraction(radius: number, centreY: number, waterlineY: number): number {
  const height = Math.max(0, Math.min(2 * radius, waterlineY - (centreY - radius)));
  const cap = Math.PI * height * height * (radius - height / 3);
  return cap / ((4 / 3) * Math.PI * radius ** 3);
}

function wetSamplesAbove(checkpoint: Readonly<Record<string, unknown>>,
  grid: readonly [number, number, number], height_m: number, threshold_m: number): number | undefined {
  const field = scalarField(checkpoint);
  if (!field || field.length < grid[0] * grid[1] * grid[2]) return undefined;
  let count = 0;
  for (let z = 0; z < grid[2]; z += 1) for (let y = 0; y < grid[1]; y += 1) {
    if ((y + 0.5) * height_m / grid[1] <= threshold_m) continue;
    for (let x = 0; x < grid[0]; x += 1) {
      if (Number(field[x + grid[0] * (y + grid[1] * z)]) >= 0.5) count += 1;
    }
  }
  return count;
}

const magnitude = (value: unknown) => {
  const record = recordValue(value);
  return Math.hypot(numberPath(record, "x") ?? Infinity,
    numberPath(record, "y") ?? Infinity, numberPath(record, "z") ?? Infinity);
};

function observations(diagnostics: Readonly<Record<string, unknown>>): CouplingCheckpoint[] {
  return fieldCheckpoints(diagnostics).flatMap((value) => {
    const checkpoint = recordValue(value);
    const snapshot = recordPath(checkpoint, "evidence", "rigid-coupling");
    const body = recordValue(arrayPath(snapshot, "bodies")?.[0]);
    const time_s = numberPath(checkpoint, "time_s");
    return checkpoint && snapshot && body && time_s !== undefined
      ? [{ time_s, checkpoint, snapshot, body }] : [];
  });
}

export function evaluateRigidCouplingDiagnostic(input: {
  scene: SceneDescription;
  evidence: NormalizedSceneDiagnosticEvidence;
  methods: readonly string[];
}): RuntimeDiagnosticFinding[] {
  const bodyDescription = input.scene.rigidBodies[0];
  if (!bodyDescription) return [hookFinding({ id: "body", passed: false,
    message: "rigid coupling oracle scene has no body" })];
  const volume = primitiveVolume(bodyDescription.shape, bodyDescription.dimensions_m);
  const fillVolume = input.scene.container.width_m * input.scene.container.depth_m
    * input.scene.container.height_m * input.scene.container.fillFraction;
  const radius = bodyDescription.dimensions_m.x;
  return input.methods.flatMap((method) => {
    const diagnostics = input.evidence.methods[method]?.diagnostics;
    const samples = diagnostics ? observations(diagnostics) : [];
    const grid = diagnostics ? gridFromDiagnostics(diagnostics) : undefined;
    const cellVolume = grid ? input.scene.container.width_m * input.scene.container.height_m
      * input.scene.container.depth_m / (grid[0] * grid[1] * grid[2]) : NaN;
    const findings: RuntimeDiagnosticFinding[] = [hookFinding({ id: `${method}.checkpoints`, method,
      passed: samples.length >= 4, message: `${samples.length} rigid coupling checkpoints collected`,
      expected: { minimum: 4 }, actual: samples.length })];
    if (samples.length === 0) return findings;

    const generations = samples.map(({ snapshot }) => numberPath(snapshot, "authorityGeneration") ?? -1);
    const live = generations.every((generation, index) => index === 0 || generation > generations[index - 1]!);
    findings.push(hookFinding({ id: `${method}.liveness`, method, passed: live,
      message: live ? "accepted authority advanced at every checkpoint" : "accepted authority stalled",
      actual: generations, expected: "strictly increasing" }));
    const errors = samples.map(({ snapshot }) => ({
      accepted: numberPath(snapshot, "authorityErrorFlags") ?? -1,
      candidate: numberPath(snapshot, "candidateErrorFlags") ?? -1,
      sealed: numberPath(snapshot, "sealedPlugCount") ?? 0,
      refresh: numberPath(snapshot, "rigidBoundaryRefreshErrorFlags") ?? -1,
    }));
    findings.push(hookFinding({ id: `${method}.tripwires`, method,
      passed: errors.every(({ accepted, candidate, sealed, refresh }) =>
        accepted === 0 && candidate === 0 && sealed === 0 && refresh === 0),
      message: "authority, candidate, refresh, and sealed-plug tripwires remained clear",
      actual: errors, expected: { accepted: 0, candidate: 0, refresh: 0, sealed: 0 } }));

    const fineFrames = samples.filter(({ checkpoint }) =>
      recordPath(checkpoint, "raster")?.surfaceGeometrySource === "global-fine-coarse").length;
    findings.push(hookFinding({ id: `${method}.surface-source`, method,
      passed: fineFrames / samples.length >= 0.9,
      message: `${fineFrames}/${samples.length} checkpoints rendered current global-fine geometry`,
      actual: fineFrames / samples.length, expected: { minimum: 0.9 } }));

    const volumeErrors = samples.map(({ checkpoint, body }) => {
      const measured = (numberPath(checkpoint, "summary", "cellSum") ?? NaN) * cellVolume;
      const displaced = numberPath(body, "displacedVolume_m3") ?? NaN;
      const target = fillVolume - displaced;
      return Math.abs(measured - target) / Math.max(target, 1e-9);
    });
    findings.push(hookFinding({ id: `${method}.volume`, method,
      passed: volumeErrors.every((error) => Number.isFinite(error) && error <= 0.005),
      message: "liquid volume followed authored fill minus submerged solid",
      actual: Math.max(...volumeErrors), expected: { maximum: 0.005 } }));

    if (input.scene.sceneId === "rigid-hydrostatic") {
      const expected = input.scene.fluid.density_kg_m3
        * Math.abs(input.scene.fluid.gravity_m_s2.y) * volume;
      const warm = samples.filter(({ time_s }) => time_s >= 0.1);
      const forces = warm.map(({ snapshot, body }) => {
        const dt = numberPath(snapshot, "dt_s") ?? NaN;
        const impulse = recordPath(body, "impulse_N_s");
        const angular = recordPath(body, "angularImpulse_N_m_s");
        return { y: (numberPath(impulse, "y") ?? NaN) / dt,
          xz: Math.hypot(numberPath(impulse, "x") ?? NaN, numberPath(impulse, "z") ?? NaN) / dt,
          torque: magnitude(angular) / dt,
          wetSurfaceCells: numberPath(body, "wetSurfaceCells") ?? NaN };
      });
      findings.push(hookFinding({ id: `${method}.buoyancy`, method,
        passed: forces.length > 0 && forces.every((force) => force.y >= 0.85 * expected
          && force.y <= 1.15 * expected && force.xz <= 0.05 * expected
          && force.torque <= 0.05 * expected * radius),
        message: "pressure reaction matched hydrostatic buoyancy",
        actual: forces, expected: { forceY_N: [0.85 * expected, 1.15 * expected] } }));
      const speedSamples = samples.map(({ snapshot }) => ({
        speed: numberPath(snapshot, "maximumLiquidComponentSpeed_m_s") ?? 0,
        location: recordPath(snapshot, "maximumLiquidComponentLocation"),
      }));
      const speed = Math.max(...speedSamples.map((sample) => sample.speed));
      findings.push(hookFinding({ id: `${method}.still-water`, method, passed: speed <= 0.05,
        message: "hydrostatic water remained still", actual: { peak: speed, samples: speedSamples },
        expected: { maximum: 0.05 } }));
    } else if (input.scene.sceneId === "rigid-float") {
      const late = samples.filter(({ time_s }) => time_s >= 1.5);
      const fractions = late.map(({ checkpoint, body }) => {
        const waterline = grid
          ? surfaceHeight(checkpoint, grid, input.scene.container.height_m) : undefined;
        const centreY = numberPath(body, "position_m", "y");
        return waterline === undefined || centreY === undefined
          ? NaN : sphereCapFraction(radius, centreY, waterline);
      });
      const mean = fractions.reduce((sum, value) => sum + value, 0) / Math.max(1, fractions.length);
      const amplitude = (values: number[]) => Math.max(...values) - Math.min(...values);
      const first = samples.filter(({ time_s }) => time_s >= 0.5 && time_s <= 1)
        .map(({ body }) => numberPath(body, "position_m", "y") ?? NaN);
      const last = late.map(({ body }) => numberPath(body, "position_m", "y") ?? NaN);
      findings.push(hookFinding({ id: `${method}.settling`, method,
        passed: mean >= 0.4 && mean <= 0.6 && first.length > 1 && last.length > 1
          && amplitude(last) < 0.5 * amplitude(first),
        message: "buoyant sphere approached half immersion with decaying oscillation",
        actual: { meanImmersed: mean, earlyAmplitude: amplitude(first), lateAmplitude: amplitude(last) },
        expected: { meanImmersed: [0.4, 0.6], lateToEarlyAmplitude: "< 0.5" } }));
      const minimumY = Math.min(...samples.map(({ body }) => numberPath(body, "position_m", "y") ?? -Infinity));
      findings.push(hookFinding({ id: `${method}.no-plunge`, method, passed: minimumY >= radius - 1e-3,
        message: "buoyant sphere did not pass through the floor", actual: minimumY,
        expected: { minimum: radius - 1e-3 } }));
    } else if (input.scene.sceneId === "rigid-sink") {
      const peak = Math.max(...samples.map(({ checkpoint }) =>
        numberPath(checkpoint, "compactMechanicalEnergy", "maximumLiquidComponentSpeed_m_s") ?? 0));
      findings.push(hookFinding({ id: `${method}.bounded-splash`, method, passed: peak <= 4,
        message: "dense entry stayed below the divergence-pump speed tripwire",
        actual: peak, expected: { maximum: 4 } }));
      const crownCounts = samples.filter(({ time_s }) => time_s >= 0.5).map(({ checkpoint }) =>
        grid ? wetSamplesAbove(checkpoint, grid, input.scene.container.height_m, 0.55) : undefined);
      findings.push(hookFinding({ id: `${method}.bounded-crown`, method,
        passed: crownCounts.length > 0 && crownCounts.every((count) => count === 0),
        message: "dense entry left no wet sample above the bounded crown after 0.5 s",
        actual: crownCounts, expected: 0 }));
      const finalSurface = grid
        ? surfaceHeight(samples.at(-1)!.checkpoint, grid, input.scene.container.height_m) : undefined;
      const expectedSurface = input.scene.container.fillFraction * input.scene.container.height_m
        + volume / (input.scene.container.width_m * input.scene.container.depth_m);
      const fineCell = input.scene.voxelDomain.finestCellSize_m / 4;
      findings.push(hookFinding({ id: `${method}.surface-rise`, method,
        passed: finalSurface !== undefined && Math.abs(finalSurface - expectedSurface) <= fineCell,
        message: "final surface rise matched one submerged body volume",
        actual: finalSurface, expected: { centre_m: expectedSurface, tolerance_m: fineCell } }));
      const floorSamples = samples.filter(({ time_s }) => time_s >= 0.8);
      const floorReached = floorSamples.length > 0 && floorSamples.every(({ body }) =>
        (numberPath(body, "position_m", "y") ?? Infinity) - radius <= 0.05 + 1e-3);
      const finalSpeed = magnitude(recordPath(samples.at(-1)!.body, "linearVelocity_m_s"));
      findings.push(hookFinding({ id: `${method}.floor-rest`, method,
        passed: floorReached && finalSpeed <= 0.2,
        message: "dense sphere reached the floor and came to rest",
        actual: { floorReached, finalSpeed }, expected: { floorBy_s: 0.8, maximumFinalSpeed: 0.2 } }));
    }
    return findings;
  });
}
