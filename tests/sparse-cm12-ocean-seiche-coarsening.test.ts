import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  createDeepPowerHydrostaticScene,
  createOceanSeicheScene,
  getSceneDefinition,
} from "../lib/core/scenes";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneCardForDefinition } from "../lib/core/scene-definition";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickContainingCoordinate,
  sparseBrickAtlasStats,
  sparseBrickSpan,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassPresentationDimensionsForScene,
  WebGPUAdaptiveMassSolver,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import {
  decodeSparseCM12FinePresentationSource,
  sparseCM12FinePresentationPlan,
  sparseCM12OwnershipTablePlan,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { globalFineSurfaceClassificationShader } from
  "../lib/core/webgpu-water-global-fine-classify";
import { globalFineDirectSharpPatchWGSL } from
  "../lib/core/webgpu-water-global-fine-tetra";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import { createGlobalFineLevelSetConsumerSource } from
  "../lib/core/octree-consumer-sampling";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

test("ocean seiche collapses deep water into graded macro-bricks", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const stats = sparseBrickAtlasStats(atlas);

  assert.ok(atlas.maximumSpanBricks >= 4, "deep water never formed a macro-brick");
  assert.ok(stats.residentBrickCount < stats.logicalBrickCount / 3,
    "resident brick storage regressed toward the wet fixed-brick volume");
  assert.ok(stats.leafCompressionRatio > 9.5,
    `deep-water compression regressed to ${stats.leafCompressionRatio}`);
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
  assert.equal(opening.scene.environment, "default");
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

dawnTest("native 8-cubed ocean pages close every B16 depth rung on one uniform tank wall",
  { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-ocean-seiche-coarsening.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
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

      // The reduced tank retains span-2 deep leaves while keeping the complete
      // surface-pipeline regression small enough for a normal Dawn test.
      const scene = createOceanSeicheScene();
      scene.container = { ...scene.container,
        width_m: 0.8, height_m: 0.8, depth_m: 0.8, fillFraction: 0.75 };
      scene.voxelDomain = { finestCellSize_m: 0.025, brickSize_cells: 8 };
      delete scene.fluid.initialBrickSeeds_m;
      delete scene.fluid.initialBrickSeedsAdditive;
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, adaptiveMassSolverOptions({
          brickFineResolution: "16",
          presentationPageResolution: "8",
        }), () => {},
      );

      const uniform = device.createBuffer({ size: 416,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bodies = device.createBuffer({ size: 768,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const volume = device.createTexture({ size: [32, 32, 32], dimension: "3d",
        format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
      const columns = device.createTexture({ size: [32, 32], format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING });
      const output = device.createTexture({ size: [64, 64], format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      owned.push(uniform, bodies, volume, columns, output);
      const view = new Float32Array(104);
      view.set([64, 64, 0, -1], 0);
      view.set([0, 0, 2, 1], 4);
      view.set([0, 0.4, 0, 1], 8);
      view.set([0.8, 0.8, 0.8, 0.6], 12);
      view.set([0, 0.025, 0, 8], 16);
      view.set([32, 32, 32, 1], 20);
      device.queue.writeBuffer(uniform, 0, view);

      water = new RasterWaterPipeline(device, "rgba16float", uniform, bodies);
      await water.initialize();
      water.ensureSize(64, 64);
      water.setVolume(volume, columns);
      water.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(
        solver.globalFineLevelSetSource,
      ));
      water.setSceneOptics({ container: scene.container });
      const encoder = device.createCommandEncoder();
      assert.ok(water.encode(encoder, output, 32, 32, 32, false, 1, 1,
        undefined, undefined, undefined, true));
      device.queue.submit([encoder.finish()]);
      const diagnostics = await water.completeSurfaceDiagnostics();
      assert.ok(diagnostics && diagnostics.vertexCount > 0);
      const source = water.diagnosticSurfaceVertexSource();
      assert.ok(source);
      const readback = device.createBuffer({ size: diagnostics.vertexCount * source.strideBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      owned.push(readback);
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(source.buffer, 0, readback, 0, readback.size);
      device.queue.submit([copy.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const vertices = new Float32Array(readback.getMappedRange());
      const xWalls = new Set<string>(), zWalls = new Set<string>();
      const projectedAreas = { xLow: 0, xHigh: 0, zLow: 0, zHigh: 0, top: 0 };
      const normalTriangles = { x: 0, yUp: 0, yDown: 0, z: 0, slanted: 0 };
      const yBounds = { minimum: Number.POSITIVE_INFINITY,
        maximum: Number.NEGATIVE_INFINITY };
      for (let vertex = 0; vertex < diagnostics.vertexCount; vertex += 1) {
        const at = vertex * source.strideBytes / 4;
        if (Math.abs(vertices[at + 4]!) > 0.9) xWalls.add(vertices[at]!.toFixed(6));
        if (Math.abs(vertices[at + 6]!) > 0.9) zWalls.add(vertices[at + 2]!.toFixed(6));
        yBounds.minimum = Math.min(yBounds.minimum, vertices[at + 1]!);
        yBounds.maximum = Math.max(yBounds.maximum, vertices[at + 1]!);
      }
      for (let vertex = 0; vertex + 2 < diagnostics.vertexCount; vertex += 3) {
        const a = vertex * source.strideBytes / 4;
        const b = (vertex + 1) * source.strideBytes / 4;
        const c = (vertex + 2) * source.strideBytes / 4;
        const ab = [vertices[b]! - vertices[a]!, vertices[b + 1]! - vertices[a + 1]!,
          vertices[b + 2]! - vertices[a + 2]!] as const;
        const ac = [vertices[c]! - vertices[a]!, vertices[c + 1]! - vertices[a + 1]!,
          vertices[c + 2]! - vertices[a + 2]!] as const;
        const nx = vertices[a + 4]!, ny = vertices[a + 5]!, nz = vertices[a + 6]!;
        if (Math.abs(nx) > 0.9) {
          normalTriangles.x += 1;
          projectedAreas[nx < 0 ? "xLow" : "xHigh"] +=
            0.5 * Math.abs(ab[1] * ac[2] - ab[2] * ac[1]);
        }
        if (Math.abs(nz) > 0.9) {
          normalTriangles.z += 1;
          projectedAreas[nz < 0 ? "zLow" : "zHigh"] +=
            0.5 * Math.abs(ab[0] * ac[1] - ab[1] * ac[0]);
        }
        if (ny > 0.9) {
          normalTriangles.yUp += 1;
          projectedAreas.top +=
            0.5 * Math.abs(ab[0] * ac[2] - ab[2] * ac[0]);
        } else if (ny < -0.9) normalTriangles.yDown += 1;
        if (Math.max(Math.abs(nx), Math.abs(ny), Math.abs(nz)) <= 0.9) {
          normalTriangles.slanted += 1;
        }
      }
      readback.unmap();
      assert.deepEqual([...xWalls].sort(), ["-0.400000", "0.400000"]);
      assert.deepEqual([...zWalls].sort(), ["-0.400000", "0.400000"]);
      const height = yBounds.maximum - yBounds.minimum;
      const expected = {
        x: scene.container.depth_m * height,
        z: scene.container.width_m * height,
        top: scene.container.width_m * scene.container.depth_m,
      };
      for (const side of ["xLow", "xHigh"] as const) {
        assert.ok(Math.abs(projectedAreas[side] - expected.x) <= 1e-5,
          `${side} wall coverage ${projectedAreas[side]} does not close ${expected.x}`);
      }
      for (const side of ["zLow", "zHigh"] as const) {
        assert.ok(Math.abs(projectedAreas[side] - expected.z) <= 1e-5,
          `${side} wall coverage ${projectedAreas[side]} does not close ${expected.z}`);
      }
      assert.ok(Math.abs(projectedAreas.top - expected.top) <= 1e-5,
        `top coverage ${projectedAreas.top} does not close ${expected.top}`);
      assert.equal(normalTriangles.yDown, 0,
        "uniform ocean publication emitted an internal downward-facing sheet");
      assert.equal(normalTriangles.slanted, 0,
        "uniform ocean publication emitted an internal slanted sheet");
      assert.deepEqual(validationErrors, []);
    } finally {
      water?.destroy();
      solver?.destroy();
      for (const resource of owned) resource.destroy();
      device?.destroy();
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
      const scene = createOceanSeicheScene();
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, adaptiveMassSolverOptions({}), () => {},
      );
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
