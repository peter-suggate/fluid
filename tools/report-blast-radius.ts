/**
 * Reports how far one pressure cell's value reaches through the encoded solve.
 *
 * This runs no GPU work. The schedule comes from the solver's own tail policy
 * and the hierarchy depth from the V-cycle's exact one-cell bottom, so the cone
 * it grows is the one the shipped command graph produces.
 *
 * The question it answers is not "how big is the blast radius" — the answer to
 * that is "the whole domain, almost immediately". It is how many stages that
 * takes, and how much level-0 work the encoded schedule spends afterwards.
 *
 *   node --import tsx tools/report-blast-radius.ts [--scene=mini|ceiling|ocean] [--cell=x,y,z]
 */
import assert from "node:assert/strict";
import {
  blastRadiusLevelDimensions,
  blastRadiusLevelsToSingleCell,
  growBlastRadius,
  planBlastRadiusSchedule,
  summarizeBlastRadius,
  type BlastRadiusVec3,
} from "../lib/fluid-blast-radius";
import { planOctreeSolveTail, type OctreeSolveTailSceneProfile } from "../lib/octree-solve-tail-policy";

/**
 * Scene profiles in the same shape the solver's tail policy consumes, so the
 * iteration count reported here is the one the encoder would emit.
 */
const PROFILES: Readonly<Record<string, OctreeSolveTailSceneProfile>> = {
  mini: {
    finestDimensions: [16, 16, 16], maximumLeafSize: 32,
    initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
  },
  ceiling: {
    finestDimensions: [24, 16, 24], maximumLeafSize: 32,
    initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: true, requestedRelativeTolerance: 1e-4,
  },
  ocean: {
    finestDimensions: [320, 96, 80], maximumLeafSize: 32,
    initialCondition: "tank-fill", hasInflow: false, hasTerrain: true,
    movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
  },
};

const argument = (name: string, fallback: string): string => {
  const match = process.argv.slice(2).find((value) => value.startsWith(`--${name}=`));
  return match === undefined ? fallback : match.slice(name.length + 3);
};

const sceneName = argument("scene", "mini");
const profile = PROFILES[sceneName];
assert.ok(profile, `--scene must be one of ${Object.keys(PROFILES).join(", ")}`);
const dimensions = profile.finestDimensions as BlastRadiusVec3;

const cellText = argument("cell", "");
const cell: BlastRadiusVec3 = cellText === ""
  ? [dimensions[0] >> 1, dimensions[1] >> 1, dimensions[2] >> 1]
  : (() => {
    const parts = cellText.split(",").map(Number);
    assert.ok(parts.length === 3 && parts.every(Number.isInteger), "--cell must be x,y,z integers");
    return parts as unknown as BlastRadiusVec3;
  })();

// Chebyshev degree two is the shipped smoother contract; the V-cycle applies it
// as both the pre- and post-smoothing sweep count at every level.
const smoothsPerLevel = Number(argument("smooths", "2"));
assert.ok(Number.isInteger(smoothsPerLevel) && smoothsPerLevel >= 1, "--smooths must be a positive integer");

const policy = planOctreeSolveTail(profile);
const levels = blastRadiusLevelsToSingleCell(dimensions);
const schedule = planBlastRadiusSchedule({
  outerIterations: policy.encodedOuterIterations, levels, smoothsPerLevel,
});
const frontiers = growBlastRadius({ dimensions, schedule, cell });
const summary = summarizeBlastRadius(frontiers, schedule, dimensions);

const percent = (value: number) => `${(100 * value).toFixed(1)}%`;

process.stdout.write(`\nPressure blast radius (${sceneName}, cell ${cell.join(",")})\n`);
process.stdout.write(`${"".padEnd(66, "-")}\n`);
process.stdout.write(`domain                    ${dimensions.join(" x ")} = ${summary.totalCells.toLocaleString()} cells\n`);
process.stdout.write(`hierarchy                 ${levels} levels down to ${blastRadiusLevelDimensions(dimensions, levels - 1).join("x")}\n`);
process.stdout.write(`encoded outer iterations  ${policy.encodedOuterIterations}\n`);
process.stdout.write(`smooths per level         ${smoothsPerLevel} pre and ${smoothsPerLevel} post\n`);
process.stdout.write(`information-moving stages ${summary.stageCount}\n`);

process.stdout.write(`\ncone growth through the first V-cycle\n`);
process.stdout.write(`  ${"".padEnd(30)}${Array.from({ length: levels }, (_, level) => `L${level}`.padStart(6)).join("")}   level-0 reach\n`);

/**
 * Occupancy of each level's box, so the climb up the hierarchy is visible.
 * Level-0 coverage alone shows a long plateau while the cone is travelling
 * through the coarse grids, which reads as nothing happening when in fact that
 * is the only part of the cycle that matters.
 */
const levelMap = (frontier: typeof frontiers[number]): string =>
  frontier.boxes.map((box, level) => {
    if (box.empty) return ".".padStart(6);
    const levelCells = blastRadiusLevelDimensions(dimensions, level)
      .reduce((product, extent) => product * extent, 1);
    const occupied = (box.hi[0] - box.lo[0] + 1) * (box.hi[1] - box.lo[1] + 1) * (box.hi[2] - box.lo[2] + 1);
    return (occupied >= levelCells ? "FULL" : `${Math.max(1, Math.round(100 * occupied / levelCells))}%`).padStart(6);
  }).join("");

const firstIteration = frontiers.filter((frontier) => frontier.stage === undefined || frontier.stage.iteration === 0);
for (const frontier of firstIteration) {
  const label = frontier.stage?.label ?? "selected cell";
  const bar = "#".repeat(Math.max(0, Math.round(frontier.share * 18)));
  process.stdout.write(
    `  ${String(frontier.stageIndex + 1).padStart(3)} ${label.padEnd(26)}${levelMap(frontier)}   ${bar.padEnd(18)} ${percent(frontier.share).padStart(6)}\n`,
  );
  if (frontier.share >= 1) break;
}

process.stdout.write(`\n`);
if (summary.stagesToGlobal === undefined || summary.iterationToGlobal === undefined) {
  process.stdout.write(`The cone never covers the domain within the encoded schedule.\n`);
} else {
  process.stdout.write(`The cone covers every cell after ${summary.stagesToGlobal} of ${summary.stageCount} stages,\n`);
  process.stdout.write(`inside outer iteration ${summary.iterationToGlobal + 1} of ${policy.encodedOuterIterations}.\n`);
  process.stdout.write(`\nSo the blast radius is not a radius. Every cell depends on every other cell\n`);
  process.stdout.write(`within one V-cycle, because the hierarchy bottoms out at a single cell and the\n`);
  process.stdout.write(`correction is prolonged back down through every level. The cost that buys is\n`);
  process.stdout.write(`${summary.fineGridSweeps} full sweeps of the ${summary.totalCells.toLocaleString()}-cell level-0 grid across the encoded solve.\n`);
}
process.stdout.write("\n");
