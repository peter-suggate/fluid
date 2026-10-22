import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import "../lib/methods";
import { validateScene } from "../lib/core/model";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { terrainHeightAt, MAX_TERRAIN_GRID_SAMPLES } from "../lib/core/terrain";
import { initialLiquidCellVolumes } from "../lib/core/initial-liquid-lattice";
import { proceduralSolidField, validateProceduralSolid } from "../lib/core/procedural-solid";
import {
  FOREST_WATERFALL_CELL_M, FOREST_WATERFALL_GRID, FOREST_WATERFALL_ID, FOREST_WATERFALL_TERRAIN_CELL_M,
  FOREST_WATERFALL_ROCKS, FOREST_WATERFALL_SOURCE, FOREST_WATERFALL_WATERLINE_M,
  createForestWaterfallScene,
} from "../lib/core/forest-waterfall-scene";
import colliderBake from "../lib/core/imported-assets/metadata/forest-waterfall-colliders.json";
import forestInstances from "../lib/core/imported-assets/forest-waterfall-instances.json";
import { sceneFluidBoxPage, sceneFluidLatticeDimensions, sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { createInflowGridBoundary } from "../lib/core/inflow-boundary";
import { uniformPageSources } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformPagePoolResidency } from "../lib/methods/uniform/uniform-page-pool";
import { uniformLatticeInitialVolume } from "../lib/methods/uniform/uniform-volume-initial";
import { uniformLatticeAddress, uniformPagePoolBlocksForMemory } from "../lib/methods/uniform/uniform-lattice-address";

test("forest waterfall is a normal wet catalog scene using production narrow-band FLIP", () => {
  const definition = getSceneDefinition(FOREST_WATERFALL_ID);
  assert.equal(definition.methodProfile?.methodId, "uniform-narrow-band-flip");
  assert.equal(definition.presentationMode, "full-scene");
  const scene = sceneDocument(definition);
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.systems?.fluid, true);
  assert.equal(scene.container.fillFraction, 0);
  assert.equal(scene.voxelDomain.finestCellSize_m, .125);
  assert.deepEqual(FOREST_WATERFALL_GRID, [256, 208, 256]);
  assert.equal(scene.numerics.fixedDt_s, 1 / 120);
  assert.equal(scene.fluid.inflow?.end_s, scene.duration_s);
  assert.ok(scene.fluid.inflow!.velocity_m_s.y < 0);
  assert.ok(scene.fluid.inflow!.radius_m >= 4 * FOREST_WATERFALL_CELL_M);
});

test("the source disk is open above the actual Blender ledges, with an approximately 19.5 metre fall", () => {
  const scene = createForestWaterfallScene(), source = scene.fluid.inflow!;
  const field = proceduralSolidField(scene)!;
  assert.ok(Math.abs(source.center_m.y - FOREST_WATERFALL_WATERLINE_M - 19.5) < .1);
  for (let ring = 0; ring <= 4; ring++) for (let i = 0; i < 48; i++) {
    const angle = 2 * Math.PI * i / 48, radius = source.radius_m * ring / 4;
    const p = { x: source.center_m.x + Math.cos(angle) * radius,
      y: source.center_m.y - source.length_m / 2, z: source.center_m.z + Math.sin(angle) * radius };
    assert.ok(p.y > terrainHeightAt(scene.terrain, p.x, p.z) + .02, `terrain blocks source at ${JSON.stringify(p)}`);
    const q = [(p.x + 16) / FOREST_WATERFALL_CELL_M, p.y / FOREST_WATERFALL_CELL_M, (p.z + 16) / FOREST_WATERFALL_CELL_M] as const;
    assert.ok((field.sample(q)?.[3] ?? 1) * FOREST_WATERFALL_CELL_M > .02, `rock blocks source at ${JSON.stringify(p)}`);
  }
  const area = Math.PI * source.radius_m ** 2;
  const added = area * Math.hypot(...Object.values(source.velocity_m_s)) * source.end_s;
  assert.ok(added < 250, "source adds under 250 cubic metres over the full sixty-second run");
  let lowestRim = Infinity;
  for (let i = 0; i <= 128; i++) {
    const p = -16 + i * .25;
    for (const [x, z] of [[-16, p], [16, p], [p, -16], [p, 16]])
      lowestRim = Math.min(lowestRim, terrainHeightAt(scene.terrain, x!, z!));
  }
  let initialStorage = 0, containedStorage = 0;
  for (let z = 0; z < 128; z++) for (let x = 0; x < 128; x++) {
    const bed = terrainHeightAt(scene.terrain, -16 + (x + .5) * .25, -16 + (z + .5) * .25);
    initialStorage += Math.max(0, FOREST_WATERFALL_WATERLINE_M - bed) * .25 ** 2;
    containedStorage += Math.max(0, lowestRim - .1 - bed) * .25 ** 2;
  }
  assert.ok(containedStorage - initialStorage > added * 1.4,
    "rising banks retain the scheduled inflow with capacity margin below the finite domain edge");
});

test("the physical waterfall shoulders are baked from the same posed Blender assets that are displayed", () => {
  assert.equal(colliderBake.sourceAssets.length, 5);
  assert.equal(colliderBake.fluidCellSize_m, FOREST_WATERFALL_CELL_M);
  assert.deepEqual(colliderBake.solidBoundary.dimensions, FOREST_WATERFALL_GRID);
  assert.equal(colliderBake.analyticLandingRockCount, 2);
  for (const source of colliderBake.sourceAssets) {
    const bytes = readFileSync(new URL(`../public/assets/forest-waterfall/${source.asset}.fvx.gz`, import.meta.url));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), source.sha256,
      `${source.asset} changed and needs a collision rebake`);
    const instance = forestInstances.find(value => value.asset === source.asset)!;
    assert.deepEqual(Object.values(source.position), instance.position);
    assert.deepEqual(Object.values(source.orientation), instance.orientation);
  }
  assert.ok(colliderBake.sourceClearance_m.atY22_05 < 0, "old analytic source intersects the actual upper ledge");
  assert.ok(colliderBake.sourceClearance_m.atY22_65 > .02);
});

test("a shared continuous heightfield covers the forest while fractional ledges remain physical", () => {
  const scene = createForestWaterfallScene();
  const grid = scene.terrain!.grid!;
  assert.ok(grid.heights_m.length <= MAX_TERRAIN_GRID_SAMPLES);
  assert.equal(grid.spacing_m, FOREST_WATERFALL_TERRAIN_CELL_M);
  assert.equal((grid.size.nx - 1) * grid.spacing_m, 80);
  assert.equal((grid.size.nz - 1) * grid.spacing_m, 96);
  assert.ok(grid.origin_m.x <= -40 && grid.origin_m.z <= -48);
  assert.ok(terrainHeightAt(scene.terrain, 0, 2) < 1.2);
  assert.ok(terrainHeightAt(scene.terrain, 12, 2) > FOREST_WATERFALL_WATERLINE_M);
  assert.ok(terrainHeightAt(scene.terrain, 8, -12) > 20);
  validateProceduralSolid(scene.solidBoundary!);
  const field = proceduralSolidField(scene)!;
  for (const rock of FOREST_WATERFALL_ROCKS.slice(1, 3)) {
    const p = { ...rock.center, z: rock.center.z + rock.radius.z * .90 };
    const local = [(p.x + 16) / FOREST_WATERFALL_CELL_M, p.y / FOREST_WATERFALL_CELL_M, (p.z + 16) / FOREST_WATERFALL_CELL_M] as const;
    assert.ok(field.sample(local)![3] < 0, "waterfall shoulder has negative interior collision distance");
  }
  const source = [(FOREST_WATERFALL_SOURCE.x + 16) / FOREST_WATERFALL_CELL_M, FOREST_WATERFALL_SOURCE.y / FOREST_WATERFALL_CELL_M,
    (FOREST_WATERFALL_SOURCE.z + 16) / FOREST_WATERFALL_CELL_M] as const;
  assert.ok((field.sample(source)?.[3] ?? 1) > 0);
});

test("initial free-surface water fills the basin and never seeds the cliff or bank interiors", () => {
  const scene = createForestWaterfallScene();
  const seeded = initialLiquidCellVolumes(scene, FOREST_WATERFALL_GRID, true);
  const [nx, ny] = FOREST_WATERFALL_GRID;
  const at = (x: number, y: number, z: number) => seeded.volume[Math.floor((x + 16) / FOREST_WATERFALL_CELL_M)
    + nx * (Math.floor(y / FOREST_WATERFALL_CELL_M) + ny * Math.floor((z + 16) / FOREST_WATERFALL_CELL_M))]!;
  assert.ok(at(0, 2, 2) > .99);
  assert.equal(at(0, 3.5, 2), 0);
  assert.equal(at(0, .5, 2), 0);
  assert.equal(at(11.5, 2, 12), 0);
  assert.equal(at(-1, 14, -9), 0);
  assert.ok(seeded.initial * FOREST_WATERFALL_CELL_M ** 3 > 400 && seeded.initial * FOREST_WATERFALL_CELL_M ** 3 < 700);
});

test("the doubled fluid lattice admits its pond and reserves the full domain plus open-top layer", () => {
  const scene = createForestWaterfallScene();
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const lattice = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  assert.deepEqual(cells, [256, 208, 256]);
  assert.deepEqual(lattice, [256, 224, 256]);
  const source = createInflowGridBoundary(scene.fluid.inflow!, scene.container, cells);
  const residency = uniformPagePoolResidency(lattice, scene.fluid.pagePool,
    uniformPageSources(scene, source, scene.numerics.maxDt_s), true,
    () => uniformLatticeInitialVolume(scene).volume, sceneFluidBoxPage(scene));
  const resident = lattice.reduce((pages, n) => pages * n / 32, 1) - residency.air.length;
  const available = uniformPagePoolBlocksForMemory(scene.fluid.pagePool!.memory_MiB!) - 2;
  assert.ok(resident > uniformPagePoolBlocksForMemory(1024) - 2,
    "the old 1 GiB budget cannot admit the doubled-resolution pond");
  const domainPages = lattice.reduce((pages, n) => pages * n / 32, 1);
  const topLayerPages = lattice[0] * lattice[2] / 32 ** 2;
  assert.ok(available >= domainPages + topLayerPages,
    "the entire authored box and its first open-top layer can coexist in the pool");
  assert.doesNotThrow(() => uniformLatticeAddress(lattice, true, { ...scene.fluid.pagePool, air: residency.air }));
});
