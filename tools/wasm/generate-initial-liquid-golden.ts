import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  baseInitialLiquidFractionAtCell,
  initialLiquidFractionAtCell,
} from "../../lib/core/initial-fluid";
import type { SceneDescription } from "../../lib/core/model";
import { sceneDocument } from "../../lib/core/scene-definition";
import { sceneLatticeDimensions } from "../../lib/core/scene-lattice";
import { SCENE_CATALOG } from "../../lib/core/scenes";

const output = resolve(import.meta.dirname, "../../rust/core/testdata/initial-liquid-golden.json");
const definition = (id: string) => {
  const found = SCENE_CATALOG.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`missing scene ${id}`);
  return found;
};
const raster = (scene: SceneDescription) => {
  const dims = sceneLatticeDimensions(scene) as [number, number, number];
  const z = Math.floor(dims[2] / 2);
  const values = new Float32Array(dims[0] * dims[1]);
  for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
    const base = baseInitialLiquidFractionAtCell(scene, x, y, z, dims);
    values[(dims[1] - 1 - y) * dims[0] + x] = initialLiquidFractionAtCell(scene, x, y, z, dims, base);
  }
  return { scene, dimensions: dims, centerSlice: Array.from(values) };
};

const base = sceneDocument(definition("water-box-dam-break"));
const authoredUnion = structuredClone(base);
authoredUnion.sceneId = "initial-liquid-all-volume-arms";
authoredUnion.container = { ...authoredUnion.container, fillFraction: 0 };
authoredUnion.fluid.initialCondition = "tank-fill";
delete authoredUnion.fluid.initialDamBreakDimensions_m;
authoredUnion.fluid.initialLiquidVolumes = [
  { shape: "box", min_m: { x: -.55, y: .05, z: -.25 }, max_m: { x: -.22, y: .34, z: .3 } },
  { shape: "sphere", center_m: { x: -.12, y: .48, z: 0 }, radius_m: .19 },
  { shape: "hemisphere", center_m: { x: .2, y: .28, z: 0 }, radius_m: .22, outwardNormal: { x: 1, y: 2, z: -.5 } },
  { shape: "cylinder", center_m: { x: .43, y: .42, z: 0 }, radius_m: .17, halfHeight_m: .31 },
  { shape: "torus", center_m: { x: 0, y: .7, z: 0 }, radius_m: .32, tubeRadius_m: .1 },
  // Deliberately overlaps the sphere and box: the eight samples must form a
  // boolean union, rather than summing each volume's fraction.
  { shape: "sphere", center_m: { x: -.27, y: .34, z: 0 }, radius_m: .23 },
];

const replacementSeeds = structuredClone(base);
replacementSeeds.sceneId = "initial-liquid-replacement-seeds";
replacementSeeds.fluid.initialBrickSeeds_m = [
  { x: -.51, y: .11, z: -.31 }, { x: .47, y: .42, z: .29 },
  { x: -.49, y: .12, z: -.30 },
];
delete replacementSeeds.fluid.initialBrickSeedsAdditive;

const additiveSeeds = structuredClone(replacementSeeds);
additiveSeeds.sceneId = "initial-liquid-additive-seeds";
additiveSeeds.fluid.initialBrickSeedsAdditive = true;

const quadratic = structuredClone(base);
quadratic.sceneId = "initial-liquid-quadratic-height";
quadratic.fluid.initialCondition = "tank-fill";
delete quadratic.fluid.initialDamBreakDimensions_m;
quadratic.fluid.initialHeightField = { kind: "quadratic", baseHeight_m: .17,
  center_m: { x: .08, z: -.06 }, curvatureX_mInv: .71, curvatureZ_mInv: .39 };

const cosine = structuredClone(quadratic);
cosine.sceneId = "initial-liquid-cosine-height";
cosine.fluid.initialHeightField = { kind: "cosine", baseHeight_m: .42,
  amplitude_m: .19, wavelength_m: .37, originX_m: -.13 };

const production = ["water-box-dam-break", "coarse-first-pool-impact-quarter",
  "twin-dam-collision", "falling-water-torus"].map((id) => raster(sceneDocument(definition(id))));
const artifact = `${JSON.stringify({ schemaVersion: 1, cases: [
  raster(authoredUnion), raster(replacementSeeds), raster(additiveSeeds), raster(quadratic), raster(cosine),
  ...production,
] }, null, 2)}\n`;

if (process.argv.includes("--check")) {
  if (readFileSync(output, "utf8") !== artifact) throw new Error(`${output} is stale`);
} else {
  mkdirSync(resolve(output, ".."), { recursive: true });
  writeFileSync(output, artifact);
  console.log(output);
}
