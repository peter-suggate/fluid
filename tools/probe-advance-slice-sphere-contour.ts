import { mkdir, writeFile } from "node:fs/promises";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { reconstructSliceSharedRdf } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import { advanceSlice, createAdvanceSlice, resetAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { contourQualityReceipt, measureSphereDisplayContour,
  sphereContourComparisonSvg } from "../tests/support/advance-slice-contour-quality";

const sceneId = "coarse-first-pool-impact-half";
const center = [32, 36.5] as const;
const radius = 10;
const seed = productionSceneSliceSeedById(sceneId);
let slice = createAdvanceSlice(seed);
const capture = (label: string) => ({ label, quality: measureSphereDisplayContour(slice,
  reconstructSliceSharedRdf(slice.topology.accepted, slice.fields), center, radius) });
const states = [capture("frame 0")];
advanceSlice(slice, { pressureIterations: 4 });
states.push(capture("one step"));
slice = resetAdvanceSlice(slice, seed);
states.push(capture("reset"));

const directory = "artifacts/advance-slice/sphere-contour-quality";
await mkdir(directory, { recursive: true });
const receipt = {
  sceneId,
  sourceCellSize_m: seed.viewport.sourceCellSize,
  analyticCircleFine: { center, radius },
  states: states.map(({ label, quality }) => ({ label, ...contourQualityReceipt(quality) })),
};
await Promise.all([
  writeFile(`${directory}/receipt.json`, `${JSON.stringify(receipt, null, 2)}\n`),
  writeFile(`${directory}/comparison.svg`,
    sphereContourComparisonSvg(states, seed.dimensions)),
]);
console.log(JSON.stringify(receipt));
