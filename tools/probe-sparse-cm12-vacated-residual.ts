/** Dawn-independent long-run residual-density probe for Sparse CM12 mini32. */
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import {
  coarsenLargeQuiescentComponents,
  initializeSparseBrickAtlasFromScene,
  materializeSparseBrickAtlasDensity,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  initializeSparseAtlasDynamics,
  stepSparseAtlasDynamics,
} from "../lib/methods/adaptive-mass/sparse-atlas-dynamics";

const secondsArgument = process.argv.find((value) => value.startsWith("--seconds="));
const seconds = Number(secondsArgument?.slice("--seconds=".length) ?? 16 / 3);
if (!Number.isFinite(seconds) || seconds <= 0) throw new RangeError("seconds must be positive");

const scene = createMinimalPowerDamBreak32Scene();
const dimensions = [32, 32, 32] as const;
let atlas = initializeSparseBrickAtlasFromScene(scene, {
  finestDimensions: dimensions,
  maximumFinestCells: 1_048_576,
});
atlas = coarsenLargeQuiescentComponents(atlas, 8);
let state = initializeSparseAtlasDynamics(atlas);
const dt_s = 1 / 30;
const steps = Math.ceil(seconds / dt_s);
const topology: Array<{ step: number; fine: number; coarse: number }> = [];
for (let step = 1; step <= steps; step += 1) {
  const result = stepSparseAtlasDynamics(state, {
    dt_s,
    accelerationFinePerSecond2: [0, scene.fluid.gravity_m_s2.y / 0.025, 0],
    projection: { relativeTolerance: 1e-10, absoluteTolerance: 1e-12 },
  });
  state = result.state;
  if (step === 1 || step % 30 === 0 || step === steps) {
    const stats = sparseBrickAtlasStats(state.atlas);
    topology.push({ step, fine: stats.fineBrickCount, coarse: stats.coarseBrickCount });
  }
}

const density = materializeSparseBrickAtlasDensity(state.atlas);
const dam = sceneDamBreakBox(scene);
const maxX = Math.ceil(dam.max.x * dimensions[0]);
const maxY = Math.ceil(dam.max.y * dimensions[1]);
const maxZ = Math.ceil(dam.max.z * dimensions[2]);
const vacatedMinimumY = Math.ceil(maxY / 2);
let residualMass = 0, residualMaximum = 0, residualCells = 0;
const rows: Array<{ y: number; mass: number; maximum: number; cells: number }> = [];
for (let y = vacatedMinimumY; y < maxY; y += 1) {
  let mass = 0, maximum = 0, cells = 0;
  for (let z = 0; z < maxZ; z += 1) for (let x = 0; x < maxX; x += 1) {
    const rho = density[x + dimensions[0] * (y + dimensions[1] * z)];
    mass += rho;
    maximum = Math.max(maximum, rho);
    cells += rho > 1e-3 ? 1 : 0;
  }
  residualMass += mass;
  residualMaximum = Math.max(residualMaximum, maximum);
  residualCells += cells;
  rows.push({ y, mass, maximum, cells });
}
console.log(JSON.stringify({
  scene: scene.sceneId,
  dimensions,
  time_s: steps * dt_s,
  originalReservoir: { maxX, maxY, maxZ },
  vacatedRegion: { minimumY: vacatedMinimumY, maximumYExclusive: maxY },
  residualMass_cells: residualMass,
  residualMaximum,
  residualCellCountAbove1e3: residualCells,
  rows,
  topology,
  finalAtlas: sparseBrickAtlasStats(state.atlas),
}, null, 2));
