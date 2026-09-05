import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  createDeepPowerHydrostaticScene,
  createOceanSeicheScene,
  getSceneDefinition,
  getScenePreset,
} from "../lib/core/scenes";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { refinementRegionCellBounds, refinementRegionLattice } from
  "../lib/core/refinement-regions";
import { sceneCardForDefinition } from "../lib/core/scene-definition";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickFaceNeighbors,
  sparseBrickContainingCoordinate,
  sparseBrickAtlasStats,
  sparseBrickSpan,
  sparseCM12InitialActiveBrickKeys,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassPresentationDimensionsForScene,
  WebGPUAdaptiveMassSolver,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { SPARSE_CM12_VELOCITY_EXTENSION_DEPTH } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import {
  decodeSparseCM12SignedPresentationKey,
  decodeSparseCM12FinePresentationSource,
  sparseCM12FinePresentationPlan,
  sparseCM12OwnershipTablePlan,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { packFineLevelSetSample } from "../lib/core/fine-levelset-packed-sample";
import { FINE_LEVELSET_SAMPLE_FLAGS } from "../lib/core/fine-levelset-brick-abi";
import {
  GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE,
  globalFineSurfaceClassificationShader,
} from
  "../lib/core/webgpu-water-global-fine-classify";
import { globalFineDirectSharpPatchWGSL } from
  "../lib/core/webgpu-water-global-fine-tetra";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

test("ocean seiche collapses deep water into graded macro-bricks", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const stats = sparseBrickAtlasStats(atlas);

  assert.ok(atlas.maximumSpanBricks >= 4, "deep water never formed a macro-brick");
  const wet = atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0));
  assert.ok(wet.length < stats.logicalBrickCount / 3,
    "liquid storage regressed toward the wet fixed-brick volume");
  const wetLeafCount = wet.reduce((sum, brick) =>
    sum + brick.resolution ** 3, 0);
  const wetCompressionRatio = stats.equivalentFinestCellCount / wetLeafCount;
  assert.ok(wetCompressionRatio > 9.5,
    `deep-water compression regressed to ${wetCompressionRatio}`);
  assert.equal(stats.integratedMassFineCells, 1_853_440,
    "coarsening must preserve the authored pool and raised-slab mass exactly");

  const verticalRungs = (x: number, z: number) => Array.from({ length: 9 }, (_, y) => {
    const brick = sparseBrickContainingCoordinate(atlas, [x, y, z]);
    return brick && 8 * sparseBrickSpan(brick) / brick.resolution;
  });
  assert.deepEqual(verticalRungs(20, 5), [8, 8, 8, 8, 8, 8, 4, 2, 1],
    "calm water must become progressively coarser below the free surface");

  for (const brick of atlas.bricks) {
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = [...brick.coordinate] as [number, number, number];
      coordinate[axis] += sparseBrickSpan(brick);
      if (coordinate[axis] >= atlas.brickDimensions[axis]) continue;
      const neighbor = sparseBrickContainingCoordinate(atlas, coordinate);
      if (!neighbor) continue;
      const ownWidth = 8 * sparseBrickSpan(brick) / brick.resolution;
      const neighborWidth = 8 * sparseBrickSpan(neighbor) / neighbor.resolution;
      assert.ok(Math.max(ownWidth, neighborWidth) <= 2 * Math.min(ownWidth, neighborWidth),
      `brick ${brick.key}/${neighbor.key} exceeds 2:1 grading`);
    }
  }
});

test("generation zero retains the authored dry velocity-extension band", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
    initialSurfaceCoarseningBiasRings: 1,
  });
  const active = sparseCM12InitialActiveBrickKeys(scene, atlas);
  const wet = atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0));
  const drySupport = atlas.bricks.filter((brick) => active.has(brick.key)
    && brick.density.every((density) => density <= 0));

  assert.ok(drySupport.length > 0,
    "the active topology must include air cells, not only liquid cells");
  for (const brick of wet) for (const neighbor of sparseBrickFaceNeighbors(atlas, brick)) {
    if (neighbor.density.every((density) => density <= 0)) {
      assert.ok(active.has(neighbor.key),
        `dry face neighbor ${neighbor.key} of wet brick ${brick.key} is inactive`);
    }
  }
  assert.ok(drySupport.some((brick) => sparseBrickFaceNeighbors(atlas, brick)
    .every((neighbor) => neighbor.density.every((density) => density <= 0))),
  "the band must extend beyond the immediate liquid face receiver");

  let matchedCoarseColumnCount = 0;
  for (let z = 0; z < atlas.brickDimensions[2]; z += 1) {
    for (let x = 0; x < atlas.brickDimensions[0]; x += 1) {
      let surfaceY = -1;
      let surface: (typeof atlas.bricks)[number] | undefined;
      for (let y = 0; y < atlas.brickDimensions[1]; y += 1) {
        const brick = sparseBrickContainingCoordinate(atlas, [x, y, z]);
        if (brick?.density.some((density) => density > 0)) {
          surfaceY = y;
          surface = brick;
        }
      }
      if (!surface) continue;
      const surfaceWidth = 8 * sparseBrickSpan(surface) / surface.resolution;
      const airWidths: number[] = [];
      for (let y = surfaceY + 1; y < atlas.brickDimensions[1]; y += 1) {
        const air = sparseBrickContainingCoordinate(atlas, [x, y, z]);
        if (!air || air.density.some((density) => density > 0)) break;
        airWidths.push(8 * sparseBrickSpan(air) / air.resolution);
      }
      assert.ok(airWidths.reduce((cells, width) => cells + 8 / width, 0)
        >= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH + 1,
      `air column ${x},${z} does not cover the extension receiver`);
      assert.ok(airWidths.every((width) => width <= surfaceWidth),
        `air column ${x},${z} is coarser than its surface`);
      if (surfaceWidth === 2 && airWidths.every((width) => width === 2)) {
        matchedCoarseColumnCount += 1;
      }
    }
  }
  assert.ok(matchedCoarseColumnCount > 0,
    "coarse surface columns must retain coarse matched-rung air support");
});

test("a far-side min-8 region retains ocean macro topology and its hard floor", () => {
  const scene = createOceanSeicheScene();
  scene.fluid.refinementRegions = [{
    id: "far-side-min-8",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
    min_m: { x: 1.2, y: 0, z: -1 },
    max_m: { x: 4, y: 2.4, z: 1 },
  }];
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const bounds = refinementRegionCellBounds(
    scene.fluid.refinementRegions[0]!, refinementRegionLattice(scene));
  const cellWidth = (brick: (typeof atlas.bricks)[number]) =>
    atlas.brickFineResolution * sparseBrickSpan(brick) / brick.resolution;
  const intersectsRegion = (brick: (typeof atlas.bricks)[number]) => {
    const lower = brick.coordinate.map((value) =>
      value * atlas.brickFineResolution);
    const upper = lower.map((value) =>
      value + sparseBrickSpan(brick) * atlas.brickFineResolution);
    return lower.every((value, axis) => value < bounds.max[axis]!
      && upper[axis]! > bounds.min[axis]!);
  };

  assert.ok(atlas.maximumSpanBricks > 1,
    "a partial hard floor must not disable safe macro-brick construction");
  assert.ok(atlas.bricks.length < atlas.brickDimensions.reduce(
    (product, value) => product * value, 1),
  "the partial region must not catalogue the complete logical brick volume");
  const intersecting = atlas.bricks.filter(intersectsRegion);
  assert.ok(intersecting.length > 0);
  assert.ok(intersecting.every((brick) => cellWidth(brick) >= 8),
    "every leaf intersecting the authored region must respect its min-8 floor");
  for (const brick of atlas.bricks) {
    for (const neighbor of sparseBrickFaceNeighbors(atlas, brick)) {
      assert.ok(Math.max(cellWidth(brick), cellWidth(neighbor))
        <= 2 * Math.min(cellWidth(brick), cellWidth(neighbor)),
      `regional face ${brick.key}/${neighbor.key} exceeds strong 2:1 grading`);
    }
  }
});

test("deep hydrostatic authors every maximally graded surface-distance rung", () => {
  const scene = createDeepPowerHydrostaticScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const verticalWidths = Array.from({ length: 5 }, (_, y) => {
    const brick = sparseBrickContainingCoordinate(atlas, [0, y, 0]);
    assert.ok(brick, `missing wet brick at y=${y}`);
    return 8 * sparseBrickSpan(brick) / brick.resolution;
  });
  assert.deepEqual(verticalWidths, [8, 8, 4, 2, 1],
    "the 40-cell pool must reach width 8 at the bottom without a duplicate width-4 rung");
  const bottom = sparseBrickContainingCoordinate(atlas, [0, 0, 0]);
  assert.equal(sparseBrickSpan(bottom!), 2);
  assert.equal(bottom!.resolution, 2,
    "the immutable bottom macro must be authored at its final surface-distance level");
});

test("ocean seiche opens without pressure-hull ribs behind the water", () => {
  const opening = sceneCardForDefinition(getSceneDefinition("ocean-seiche")).open();
  const environment = buildEnvironmentProxyCatalog(
    opening.scene,
    opening.scene.environment ?? "default",
  );
  assert.equal(opening.scene.environment, "stage");
  assert.equal(opening.methodProfile, undefined,
    "the ocean must use ordinary method defaults rather than a scene override");
  assert.ok(environment.primitives.every((primitive) =>
    !primitive.key.includes("hull/rib-")),
  "a station rib behind the transparent tank can look like missing deep water");
});

test("ocean seiche presentation retains deep macro leaves at native scale", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const publication = sparseCM12FinePresentationPlan(atlas);
  const pagesByBrick = new Uint32Array(atlas.bricks.length);
  let macroPageCount = 0;
  let deepestMacroPageY = Number.POSITIVE_INFINITY;

  for (let page = 0; page < publication.plan.maximumResidentBricks; page += 1) {
    const source = decodeSparseCM12FinePresentationSource(publication.metadata[4 * page + 3]!);
    const brick = atlas.bricks[source.brick]!;
    pagesByBrick[source.brick] += 1;
    assert.equal(source.spanBricks, sparseBrickSpan(brick),
      "presentation metadata must preserve the atlas leaf span");
    if (source.spanBricks > 1) {
      macroPageCount += 1;
      deepestMacroPageY = Math.min(deepestMacroPageY, brick.coordinate[1]);
      assert.equal(source.octant, 0, "a macro leaf must cost exactly one page");
    }
  }

  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const expected = sparseBrickSpan(atlas.bricks[brick]!) === 1 ? 8 : 1;
    assert.equal(pagesByBrick[brick], expected,
      `atlas leaf ${brick} disappeared from the global presentation`);
  }
  assert.ok(macroPageCount > 0, "the regression scene must exercise macro publication");
  assert.ok(deepestMacroPageY < 6,
    "native-scale pages must reach the deep half of the authored pool");
  assert.equal((publication.worklist[3]! >>> 8) & 31,
    Math.log2(atlas.maximumSpanBricks),
    "compact lookup must advertise its bounded macro search depth");
  assert.ok(publication.plan.maximumResidentBricks < 8 * atlas.bricks.length,
    "macro publication regressed to fixed-brick page expansion");

  assert.ok(atlas.bricks.every((brick) =>
    Array.from(brick.density).every((density) => density === 1)),
  "the authored pool must remain an intensive, uniform liquid volume");
  assert.match(globalFineSurfaceClassificationShader,
    /base\.x\+scale-1==dims\.x/,
    "scaled pages must detect the real high wall instead of a unit-scale anchor");
  assert.match(globalFineSurfaceClassificationShader,
    /emitScaledBoundaryWallFace\(base,scale/,
    "macro pages must close the pool with an exact boundary-plane owner");
  assert.match(globalFineDirectSharpPatchWGSL,
    /wallScale\*a/,
    "wall emission must retain the native page's tangential span");
});

test("ocean seiche presentation page width is independent of its solver ladder", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
    brickFineResolution: 16,
  });
  for (const pageResolution of [4, 8, 16] as const) {
    const publication = sparseCM12FinePresentationPlan(atlas, pageResolution);
    const pagesPerAxis = 16 / pageResolution;
    assert.equal(publication.plan.brickResolution, pageResolution);
    assert.equal(publication.plan.samplesPerBrick, pageResolution ** 3);
    assert.equal(publication.plan.maximumResidentBricks,
      atlas.bricks.length * pagesPerAxis ** 3);
    assert.equal((publication.worklist[3]! >>> 16) & 31, 16);
    assert.equal((publication.worklist[3]! >>> 21) & 31, pageResolution);
    for (let page = 0; page < publication.plan.maximumResidentBricks; page += 1) {
      const source = decodeSparseCM12FinePresentationSource(
        publication.metadata[4 * page + 3]!, 16, pageResolution,
      );
      assert.ok(source.brick < atlas.bricks.length);
      assert.ok(source.octant < pagesPerAxis ** 3);
      assert.equal(source.spanBricks, 1);
    }
  }
  assert.throws(() => sparseCM12FinePresentationPlan(atlas, 32 as 16), /does not divide/);
});

test("closed vast-depth fills do no finest-domain-shaped planning or allocation", () => {
  const scene = createOceanSeicheScene();
  scene.container = { ...scene.container, fillFraction: 1 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [256, 1_048_576, 256],
  });
  const logicalBricks = 32 * 131_072 * 32;
  assert.equal(atlas.brickDimensions.reduce((product, value) => product * value, 1),
    logicalBricks);
  assert.ok(atlas.bricks.length <= 4096,
    `${atlas.bricks.length} leaves suggests wet-volume enumeration`);
  assert.ok(atlas.maximumSpanBricks >= 32);
  assert.ok(atlas.bricks.every((brick) => brick.density.length === 1),
    "closed quiescent macro-bricks should allocate one physical cell each");
  const ownership = sparseCM12OwnershipTablePlan(atlas.bricks.length);
  assert.ok(ownership.allocatedWords < 32 * atlas.bricks.length,
    "GPU ownership must remain proportional to leaves");
  assert.ok(ownership.allocatedWords < logicalBricks / 1_000,
    "GPU ownership must not recreate a logical-domain brick directory");
});

dawnTest("native ocean pages close interior adaptive seams on the exact tank wall",
  { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-ocean-seiche-coarsening.test.ts");
    let device: GPUDevice | undefined;
    let water: RasterWaterPipeline | undefined;
    const owned: Array<{ destroy(): void }> = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      // Publish the authored Ocean atlas directly. This is intentionally not a
      // tiny solver scene: a 32-cell tank collapses to all-unit pages and cannot
      // reproduce the span-4/span-2/span-1 wall transitions seen in the viewport.
      const scene = createOceanSeicheScene();
      const dimensions = adaptiveMassPresentationDimensionsForScene(scene);
      const atlas = initializeSparseBrickAtlasFromScene(scene, { finestDimensions: dimensions });
      // Four by four brick columns retain complete tangential neighbours at
      // both rung changes without allocating the other 96% of the flat tank.
      const residentBrickKeys = new Set(atlas.bricks.filter((brick) =>
        brick.coordinate[0] < 4 && brick.coordinate[2] < 4).map((brick) => brick.key));
      const publication = sparseCM12FinePresentationPlan(atlas, 8, {
        signedSparseAddressing: true,
        residentBrickKeys,
      });
      const packedSamples = new Uint32Array(publication.plan.maximumResidentBricks
        * publication.plan.samplesPerBrick);
      const pageResolution = publication.plan.brickResolution;
      for (let page = 0; page < publication.plan.maximumResidentBricks; page += 1) {
        const key = publication.metadata[4 * page + 1]!;
        const pageCoordinate = decodeSparseCM12SignedPresentationKey(key);
        const source = decodeSparseCM12FinePresentationSource(
          publication.metadata[4 * page + 3]!, atlas.brickFineResolution, pageResolution,
        );
        const scale = source.spanBricks > 1 ? source.spanBricks : 1;
        for (let local = 0; local < publication.plan.samplesPerBrick; local += 1) {
          const y = Math.floor(local / pageResolution) % pageResolution;
          const qy = pageCoordinate[1] * pageResolution + y * scale;
          const phi = (qy - 71.5) * scene.voxelDomain.finestCellSize_m;
          packedSamples[page * publication.plan.samplesPerBrick + local] =
            packFineLevelSetSample(phi, FINE_LEVELSET_SAMPLE_FLAGS.valid
              | (phi < 0 ? FINE_LEVELSET_SAMPLE_FLAGS.negative : 0));
        }
      }
      const fineMetadata = device.createBuffer({ size: publication.metadata.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const fineWorklist = device.createBuffer({ size: publication.worklist.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const fineSamples = device.createBuffer({ size: packedSamples.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(fineMetadata, 0, publication.metadata.buffer as ArrayBuffer,
        publication.metadata.byteOffset, publication.metadata.byteLength);
      device.queue.writeBuffer(fineWorklist, 0, publication.worklist.buffer as ArrayBuffer,
        publication.worklist.byteOffset, publication.worklist.byteLength);
      device.queue.writeBuffer(fineSamples, 0, packedSamples);
      owned.push(fineMetadata, fineWorklist, fineSamples);

      const uniform = device.createBuffer({ size: 416,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bodies = device.createBuffer({ size: 768,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const volume = device.createTexture({ size: dimensions, dimension: "3d",
        format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
      const columns = device.createTexture({ size: [dimensions[0], dimensions[2]],
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING });
      const output = device.createTexture({ size: [64, 64], format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      owned.push(uniform, bodies, volume, columns, output);
      const view = new Float32Array(104);
      view.set([64, 64, 0, -1], 0);
      view.set([0, 0, 2, 1], 4);
      view.set([0, 1.2, 0, 1], 8);
      view.set([scene.container.width_m, scene.container.height_m,
        scene.container.depth_m, 1.8], 12);
      view.set([0, 0.025, 0, 8], 16);
      view.set([...dimensions, 1], 20);
      device.queue.writeBuffer(uniform, 0, view);

      water = new RasterWaterPipeline(device, "rgba16float", uniform, bodies);
      await water.initialize();
      water.ensureSize(64, 64);
      water.setVolume(volume, columns);
      water.setGlobalFineLevelSet({
        kind: "global-fine-levelset-sampling",
        metadata: { buffer: fineMetadata },
        worklist: { buffer: fineWorklist },
        samples: { buffer: fineSamples },
        sampleDimensions: dimensions,
        brickDimensions: publication.plan.brickDimensions,
        brickResolution: publication.plan.brickResolution,
        samplesPerBrick: publication.plan.samplesPerBrick,
        pageCapacity: publication.plan.maximumResidentBricks,
        fineFactor: 1,
        fineCellWidth: scene.voxelDomain.finestCellSize_m,
        domainOrigin: [0, 0, 0],
        generation: 1,
      });
      water.setSceneOptics({ container: scene.container });
      const encoder = device.createCommandEncoder();
      assert.ok(water.encode(encoder, output, ...dimensions, false, 1, 1,
        undefined, undefined, true));
      device.queue.submit([encoder.finish()]);
      const diagnostics = await water.completeSurfaceDiagnostics();
      assert.ok(diagnostics && diagnostics.vertexCount > 0);
      const source = water.diagnosticSurfaceVertexSource();
      assert.ok(source);
      const readback = device.createBuffer({ size: diagnostics.vertexCount * source.strideBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const cubeReadback = device.createBuffer({ size: diagnostics.activeCubeCount * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      owned.push(readback, cubeReadback);
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(source.buffer, 0, readback, 0, readback.size);
      copy.copyBufferToBuffer(source.classifiedCubes, 0, cubeReadback, 0, cubeReadback.size);
      device.queue.submit([copy.finish()]);
      await Promise.all([
        readback.mapAsync(GPUMapMode.READ),
        cubeReadback.mapAsync(GPUMapMode.READ),
      ]);
      const vertices = new Float32Array(readback.getMappedRange());
      const cubeWords = new Uint32Array(cubeReadback.getMappedRange());
      let transitionCubeCount = 0;
      let transitionFaceCount = 0;
      const descriptorScaleCounts = new Map<number, number>();
      for (let cube = 0; cube < diagnostics.activeCubeCount; cube += 1) {
        const descriptor = cubeWords[2 * cube + 1]! >>> 16;
        const rawCode = descriptor & 0xff;
        if (rawCode >= 224) continue;
        const heightField = rawCode === GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE;
        const transition = !heightField && (rawCode & 0x80) !== 0;
        const scale = heightField ? 1 : transition ? rawCode & 0x7f : rawCode;
        descriptorScaleCounts.set(scale, (descriptorScaleCounts.get(scale) ?? 0) + 1);
        if (!transition) continue;
        transitionCubeCount += 1;
        const faceMask = (descriptor >>> 8) & 0x3f;
        transitionFaceCount += faceMask.toString(2)
          .split("").filter((bit) => bit === "1").length;
      }
      const seamEdges = new Map<string, { count: number;
        a: readonly [number, number, number]; b: readonly [number, number, number] }>();
      const lowWallCoordinates = [-scene.container.width_m / 2,
        -scene.container.depth_m / 2] as const;
      const adaptiveSeamPlanes = [33, 49].map((cell) =>
        cell * scene.voxelDomain.finestCellSize_m);
      for (let vertex = 0; vertex + 2 < diagnostics.vertexCount; vertex += 3) {
        const a = vertex * source.strideBytes / 4;
        const b = (vertex + 1) * source.strideBytes / 4;
        const c = (vertex + 2) * source.strideBytes / 4;
        const points = [
          [vertices[a]!, vertices[a + 1]!, vertices[a + 2]!],
          [vertices[b]!, vertices[b + 1]!, vertices[b + 2]!],
          [vertices[c]!, vertices[c + 1]!, vertices[c + 2]!],
        ] as const;
        const pointKey = (point: readonly number[]) => point.map((value) =>
          new Uint32Array(new Float32Array([value]).buffer)[0]!.toString(16)).join(":");
        for (const [from, to] of [[0, 1], [1, 2], [2, 0]] as const) {
          if (!adaptiveSeamPlanes.some((plane) =>
            Math.abs(points[from][1] - plane) <= 1e-6
            && Math.abs(points[to][1] - plane) <= 1e-6)) continue;
          const liesOnPhysicalWall = [0, 1].some((wall) => {
            const axis = wall === 0 ? 0 : 2;
            return Math.abs(points[from][axis] - lowWallCoordinates[wall]!) <= 1e-6
              && Math.abs(points[to][axis] - lowWallCoordinates[wall]!) <= 1e-6;
          });
          if (!liesOnPhysicalWall) continue;
          const first = pointKey(points[from]), second = pointKey(points[to]);
          const key = first < second ? `${first}|${second}` : `${second}|${first}`;
          const previous = seamEdges.get(key);
          seamEdges.set(key, { count: (previous?.count ?? 0) + 1,
            a: points[from], b: points[to] });
        }
      }
      readback.unmap();
      cubeReadback.unmap();
      assert.ok(transitionCubeCount > 0 && transitionFaceCount > 0,
        "the adaptive ocean must publish explicit coarse/fine transition topology"
          + ` (descriptors ${JSON.stringify(Object.fromEntries(descriptorScaleCounts))})`);
      const unmatchedInteriorWallEdges = [...seamEdges.values()].filter((edge) => {
        if (edge.count !== 1) return false;
        return [0, 1].some((wall) => {
          const axis = wall === 0 ? 0 : 2;
          const tangent = axis === 0 ? 2 : 0;
          const interiorLow = lowWallCoordinates[wall]! + 0.15;
          const interiorHigh = lowWallCoordinates[wall]! + 0.65;
          return Math.abs(edge.a[axis] - lowWallCoordinates[wall]!) <= 1e-6
            && Math.abs(edge.b[axis] - lowWallCoordinates[wall]!) <= 1e-6
            && edge.a[tangent] >= interiorLow && edge.a[tangent] <= interiorHigh
            && edge.b[tangent] >= interiorLow && edge.b[tangent] <= interiorHigh;
        });
      });
      assert.ok(seamEdges.size > 0,
        "the adaptive seam audit must reach an exact physical tank wall");
      assert.equal(unmatchedInteriorWallEdges.length, 0,
        `adaptive wall mesh has ${unmatchedInteriorWallEdges.length} unmatched seam half-edges: `
          + JSON.stringify(unmatchedInteriorWallEdges.slice(0, 8)));
      assert.deepEqual(validationErrors, []);
    } finally {
      if (device) await device.queue.onSubmittedWorkDone();
      water?.destroy();
      for (const resource of owned) resource.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });

dawnTest("production ocean keeps pressure and represented volume across frames",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-ocean-seiche-coarsening.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const scene = getScenePreset("ocean-seiche").create();
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, adaptiveMassSolverOptions({}), () => {},
      );
      await solver.waitForSimulationReady();
      const initialDensity = (await solver.readDiagnosticFields()).density;
      const initialMass = initialDensity.reduce((sum, rho) => sum + Math.max(0, rho), 0);
      const initialWet = initialDensity.reduce((sum, rho) => sum + Number(rho > 0.005), 0);
      let density = initialDensity;
      for (let step = 1; step <= 3; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const [fields, stats] = await Promise.all([
          solver.readDiagnosticFields(), solver.readStats(),
        ]);
        const pcm: {
          readonly cell: { readonly fault: number; readonly firstFault: number };
          readonly row: { readonly fault: number; readonly firstFault: number };
        } = await solver.readPressureCanonicalMembershipQA();
        density = fields.density;
        assert.ok((stats.pressureIterationsExecuted ?? 0) > 0,
          `ocean frame ${step} skipped its pressure solve`);
        assert.equal(pcm.cell.fault, 0,
          `ocean frame ${step} PCM cell authority faulted at ${pcm.cell.firstFault}`);
        assert.equal(pcm.row.fault, 0,
          `ocean frame ${step} PCM row authority faulted at ${pcm.row.firstFault}`);
      }
      for (let step = 4; step <= 20; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
      }
      await device.queue.onSubmittedWorkDone();
      density = (await solver.readDiagnosticFields()).density;
      const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
      let deepMinimum = Number.POSITIVE_INFINITY;
      let missingDeepCells = 0;
      for (let y = 0; y < 32; y += 1) {
        for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
          const rho = density[x + nx * (y + ny * z)]!;
          deepMinimum = Math.min(deepMinimum, rho);
          missingDeepCells += rho < 0.5 ? 1 : 0;
        }
      }
      assert.equal(missingDeepCells, 0,
        `GPU retirement opened ${missingDeepCells} cells in the 32-cell-deep macro rung`
          + ` (minimum rho ${deepMinimum})`);
      const mass = density.reduce((sum, rho) => sum + Math.max(0, rho), 0);
      const wet = density.reduce((sum, rho) => sum + Number(rho > 0.005), 0);
      assert.ok(Math.abs(mass - initialMass) / initialMass <= 5e-4,
        `ocean mass drifted by ${(mass - initialMass) / initialMass}`);
      assert.ok(wet >= 0.98 * initialWet,
        `ocean represented volume fell from ${initialWet} to ${wet} cells`);
    } finally {
      solver?.destroy(); device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
