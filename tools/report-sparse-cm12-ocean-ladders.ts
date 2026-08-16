import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { createOceanSeicheScene } from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
  sparseBrickSpan,
  type SparseBrickFineResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

const scene = createOceanSeicheScene();
const dimensions = sceneLatticeDimensions(scene);
const ladders = [4, 8, 16] as const satisfies readonly SparseBrickFineResolution[];

const receipts = ladders.map((brickFineResolution) => {
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    brickFineResolution,
  });
  const stats = sparseBrickAtlasStats(atlas);
  const rungCensus = new Map<string, number>();
  const verticalProfile = new Map<string, Map<number, number>>();
  for (const brick of atlas.bricks) {
    const span = sparseBrickSpan(brick);
    const rung = `span-${span}/resolution-${brick.resolution}`;
    rungCensus.set(rung, (rungCensus.get(rung) ?? 0) + 1);
    const layer = `${brick.coordinate[1]}..${brick.coordinate[1] + span - 1}`;
    const resolutions = verticalProfile.get(layer) ?? new Map<number, number>();
    resolutions.set(brick.resolution, (resolutions.get(brick.resolution) ?? 0) + 1);
    verticalProfile.set(layer, resolutions);
  }
  return {
    brickFineResolution,
    residentBrickCount: atlas.bricks.length,
    leafCount: stats.leafCount,
    integratedMassFineCells: stats.integratedMassFineCells,
    maximumSpanBricks: atlas.maximumSpanBricks,
    rungCensus: Object.fromEntries(rungCensus),
    verticalProfile: Object.fromEntries([...verticalProfile].map(([layer, resolutions]) =>
      [layer, Object.fromEntries(resolutions)])),
  };
});

const validationErrors: string[] = [];
const masses = new Set(receipts.map((receipt) => receipt.integratedMassFineCells));
if (masses.size !== 1) validationErrors.push("integrated mass differs between ladders");
const sixteen = receipts.find((receipt) => receipt.brickFineResolution === 16)!;
const expectedSixteenProfile = [1, 2, 4, 8, 16];
for (let y = 0; y < expectedSixteenProfile.length; y += 1) {
  const layer = sixteen.verticalProfile[`${y}..${y}`];
  const expected = expectedSixteenProfile[y]!;
  if (!layer || Number(layer[expected]) !== 100 || Object.keys(layer).length !== 1) {
    validationErrors.push(`B16 layer ${y} is not uniformly resolution ${expected}`);
  }
}
if (sixteen.maximumSpanBricks !== 1) {
  validationErrors.push("B16 macro spans cross the five-rung depth skirt");
}

console.log(JSON.stringify({
  scene: scene.sceneId,
  dimensions,
  receipts,
  validationErrors,
  passed: validationErrors.length === 0,
}, null, 2));
if (validationErrors.length > 0) process.exitCode = 1;
