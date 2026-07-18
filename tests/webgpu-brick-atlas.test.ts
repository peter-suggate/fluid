import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { octreeMethod } from "../lib/methods/octree";
import {
  brickAtlasLifecycleShader,
  brickAtlasMirrorShader,
  brickAtlasValidateShader,
  fluidBrickAtlasSamplingWGSL,
  planFluidBrickAtlas,
} from "../lib/webgpu-brick-atlas";
import {
  FLUID_BRICK_ACTIVATED,
  FLUID_BRICK_RESIDENT,
  GPUFluidBrickResidency,
  fluidBrickResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("atlas planning packs apron tiles near-cubically inside the texture limit", () => {
  const plan = planFluidBrickAtlas([61, 46, 41], { brickSize: 8 });
  assert.deepEqual(plan.brickDimensions, [8, 6, 6]);
  assert.equal(plan.logicalBrickCount, 288);
  assert.equal(plan.tileSize, 10);
  assert.equal(plan.capacity, 288, "mirror mode backs the whole logical lattice by default");
  assert.deepEqual(plan.tileGridDimensions, [7, 7, 6]);
  assert.deepEqual(plan.atlasDimensions, [70, 70, 60]);
  assert.equal(plan.degraded, false);
  assert.equal(plan.bytesPerTile, 1000 * 20);
  assert.equal(plan.allocatedTextureBytes, 70 * 70 * 60 * 20);
  const fractional = planFluidBrickAtlas([61, 46, 41], { brickSize: 8, maximumResidentFraction: 0.75 });
  assert.equal(fractional.capacity, 216);
});

test("atlas capacity degrades against maxTextureDimension3D instead of failing", () => {
  const plan = planFluidBrickAtlas([128, 128, 128], { brickSize: 8, maximumResidentFraction: 1, maxTextureDimension3D: 40 });
  assert.equal(plan.logicalBrickCount, 4096);
  assert.deepEqual(plan.tileGridDimensions, [4, 4, 4], "only four 10-texel tiles fit per 40-texel axis");
  assert.equal(plan.capacity, 64);
  assert.equal(plan.degraded, true);
  plan.atlasDimensions.forEach((extent) => assert.ok(extent <= 40));
});

test("atlas capacity honors the hard tile ceiling and fraction", () => {
  const capped = planFluidBrickAtlas([128, 128, 128], { brickSize: 8, maximumTiles: 50 });
  assert.equal(capped.capacity, 50);
  assert.equal(capped.degraded, false);
  const fraction = planFluidBrickAtlas([64, 64, 64], { brickSize: 8, maximumResidentFraction: 0.25 });
  assert.equal(fraction.capacity, Math.ceil(512 * 0.25));
  const floor = planFluidBrickAtlas([8, 8, 8], { brickSize: 8, maximumResidentFraction: 0.0001 });
  assert.equal(floor.capacity, 1, "at least one physical tile always exists");
  assert.throws(() => planFluidBrickAtlas([0, 8, 8]), /positive integer/);
});

test("atlas shaders follow the pooled sparse-page template", () => {
  assert.match(brickAtlasLifecycleShader, /atomicExchange\(&pageTable\[brickIndex\], INVALID\)/);
  assert.match(brickAtlasLifecycleShader, /atomicStore\(&control\[2\], 1u\)/, "pool exhaustion raises the overflow flag");
  assert.match(brickAtlasLifecycleShader, /atomicSub\(&control\[0\], 1u\)/);
  assert.match(brickAtlasLifecycleShader, /let x = min\(blocks, 65535u\)/, "indirect work tiles into two dimensions");
  assert.doesNotMatch(brickAtlasLifecycleShader, /states\[pageIndex\] = state & ~RESIDENT/, "atlas overflow must not clear residency it does not own");
  assert.match(brickAtlasMirrorShader, /vec3i\(local\) - vec3i\(1\)/, "tiles mirror a one-voxel apron around the payload");
  assert.match(brickAtlasMirrorShader, /clamp\(cell, vec3i\(0\), vec3i\(atlasParams\.dims\.xyz\) - vec3i\(1\)\)/, "apron reads clamp to the dense domain like the dense sampler");
  for (const filterable of [true, false]) {
    const sampling = fluidBrickAtlasSamplingWGSL(filterable);
    assert.match(sampling, /fn brickAtlasSamplePhi\(position: vec3f\) -> f32/);
    assert.match(sampling, /fn brickAtlasSampleVelocity\(position: vec3f\) -> vec3f/);
    assert.match(sampling, /brickAtlasDensePhi/, "missing slots fall back to the dense field");
    const validate = brickAtlasValidateShader(filterable);
    assert.match(validate, /fn compareAtlasToDense/);
    assert.match(validate, /atomicMax\(&stats\[index\], bitcast<u32>/);
  }
  assert.match(fluidBrickAtlasSamplingWGSL(true), /textureSampleLevel\(brickAtlasPhi, brickAtlasSampler/, "filterable devices use one hardware trilinear fetch");
});

test("residency pre-activation entry points exist and widen support by swept velocity", () => {
  assert.match(fluidBrickResidencyShader, /fn classifySwept/);
  assert.match(fluidBrickResidencyShader, /fn expandDownstream/);
  assert.match(fluidBrickResidencyShader, /fn emitWorklist/);
  assert.match(fluidBrickResidencyShader, /params\.settings\.z \* 1\.5/, "swept support = |v| dt with a 1.5 safety factor");
  assert.match(fluidBrickResidencyShader, /max\(params\.settings\.x, sweptSupport\)/);
  assert.match(fluidBrickResidencyShader, /minimumPhi <= 0\.0 && maximumPhi >= 0\.0/, "legacy classify keeps its exact band semantics");
});

async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredFeatures: optionalFluidDeviceFeatures(adapter.features),
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  return { device, validationErrors };
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number) {
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return words;
}

test("GPU pre-activation schedules downstream and swept bricks before phi arrives", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const dims = [24, 8, 8] as const;
    const cell = [0.1, 0.1, 0.1] as const;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    const levelSet = device.createTexture({ size: [...dims], dimension: "3d", format: "r32float", usage });
    const velocity = device.createTexture({ size: [...dims], dimension: "3d", format: "rgba32float", usage });
    // Brick 0 (x 0..7) straddles phi=0; bricks 1 and 2 sit far outside the
    // two-cell halo band (minimum phi 0.45 m and 1.25 m against 0.2 m).
    const phi = new Float32Array(dims[0] * dims[1] * dims[2]);
    for (let z = 0; z < dims[2]; z += 1) for (let y = 0; y < dims[1]; y += 1) for (let x = 0; x < dims[0]; x += 1) {
      phi[x + dims[0] * (y + dims[1] * z)] = x < 4 ? -0.05 : (x - 4 + 0.5) * cell[0];
    }
    device.queue.writeTexture({ texture: levelSet }, phi, { bytesPerRow: dims[0] * 4, rowsPerImage: dims[1] }, [...dims]);
    const writeVelocity = (vx: number) => {
      const data = new Float32Array(dims[0] * dims[1] * dims[2] * 4);
      for (let i = 0; i < data.length; i += 4) data[i] = vx;
      device.queue.writeTexture({ texture: velocity }, data, { bytesPerRow: dims[0] * 16, rowsPerImage: dims[1] }, [...dims]);
    };
    const run = async (vx: number, preActivation: boolean, dt_s: number) => {
      const residency = new GPUFluidBrickResidency(device, dims, cell, { brickSize: 8, haloCells: 2, retireAfterFrames: 0 });
      writeVelocity(vx);
      const encoder = device.createCommandEncoder();
      residency.encode(encoder, levelSet, velocity, { dt_s, preActivation });
      device.queue.submit([encoder.finish()]);
      const states = await readBuffer(device, residency.stateBuffer, 3 * 4);
      residency.destroy();
      return states;
    };
    // Baseline: no pre-activation keeps the dry downstream bricks unscheduled.
    const baseline = await run(2, false, 0.05);
    assert.ok((baseline[0] & FLUID_BRICK_RESIDENT) !== 0, "the interface brick is resident");
    assert.equal(baseline[1] & FLUID_BRICK_RESIDENT, 0);
    assert.equal(baseline[2] & FLUID_BRICK_RESIDENT, 0);
    // Downstream expansion: swept support 2*0.05*1.5 = 0.15 m cannot reach
    // brick 1 (0.45 m), so residency must come from the core neighbor whose
    // face velocity points into it — before any of its cells are wet.
    const downstream = await run(2, true, 0.05);
    assert.ok((downstream[1] & FLUID_BRICK_RESIDENT) !== 0, "downstream neighbor of a core brick pre-activates");
    assert.ok((downstream[1] & FLUID_BRICK_ACTIVATED) !== 0);
    assert.equal(downstream[2] & FLUID_BRICK_RESIDENT, 0, "bricks beyond the immediate downstream neighbor stay unscheduled");
    // Swept support: upstream flow disables the neighbor rule, but a fast
    // front widens the band itself: 8*0.05*1.5 = 0.6 m > 0.45 m.
    const upstreamSlow = await run(-2, true, 0.05);
    assert.equal(upstreamSlow[1] & FLUID_BRICK_RESIDENT, 0, "upstream flow must not pre-activate the neighbor");
    const upstreamFast = await run(-8, true, 0.05);
    assert.ok((upstreamFast[1] & FLUID_BRICK_RESIDENT) !== 0, "swept support widens the residency band");
    levelSet.destroy();
    velocity.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("GPU atlas mirrors dam-break fields exactly, including brick seams, and pre-activates ahead of the front", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const scene: SceneDescription = cloneScene(defaultScene);
    scene.sceneId = "test-brick-atlas-dam-break";
    scene.fluid.initialCondition = "dam-break";
    delete scene.fluid.inflow;
    scene.rigidBodies = [];
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
    const values = Object.fromEntries(octreeMethod.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
    const solver = octreeMethod.createSolver!(device, scene, "balanced", values, undefined) as unknown as {
      advanceTo(time_s: number, bodies: never[]): boolean;
      readStats(): Promise<Record<string, number | undefined>>;
      destroy(): void;
    };
    const internals = solver as unknown as {
      octreeProjection: {
        levelSetTexture: GPUTexture;
        sparseBrickWorld: { residency: GPUFluidBrickResidency; atlas?: { plan: { capacity: number } } };
      };
    };
    const residency = internals.octreeProjection.sparseBrickWorld.residency;
    const capacity = residency.capacity;
    const [bx, by, bz] = residency.brickDimensions;
    const dims = [61, 46, 41] as const;
    const bytesPerRow = Math.ceil(dims[0] * 4 / 256) * 256;
    const phiReadback = device.createBuffer({ size: bytesPerRow * dims[1] * dims[2], usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readPhiPerBrick = async () => {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: internals.octreeProjection.levelSetTexture },
        { buffer: phiReadback, bytesPerRow, rowsPerImage: dims[1] },
        [...dims],
      );
      device.queue.submit([encoder.finish()]);
      await phiReadback.mapAsync(GPUMapMode.READ);
      const raw = new Uint8Array(phiReadback.getMappedRange().slice(0));
      phiReadback.unmap();
      const minimumPhi = new Float32Array(capacity).fill(Number.POSITIVE_INFINITY);
      const view = new DataView(raw.buffer);
      for (let z = 0; z < dims[2]; z += 1) for (let y = 0; y < dims[1]; y += 1) for (let x = 0; x < dims[0]; x += 1) {
        const value = view.getFloat32((z * dims[1] + y) * bytesPerRow + x * 4, true);
        const brick = Math.floor(x / 8) + bx * (Math.floor(y / 8) + by * Math.floor(z / 8));
        if (value < minimumPhi[brick]) minimumPhi[brick] = value;
      }
      return minimumPhi;
    };
    const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));
    const dt = scene.numerics.fixedDt_s!;
    const firstResident = new Int32Array(capacity).fill(-1);
    const firstWet = new Int32Array(capacity).fill(-1);
    const steps = 50;
    let time = 0;
    for (let step = 0; step < steps; step += 1) {
      while (!solver.advanceTo(time + dt, [])) await nextTurn();
      time += dt;
      await device.queue.onSubmittedWorkDone();
      const states = await readBuffer(device, residency.stateBuffer, capacity * 4);
      const minimumPhi = await readPhiPerBrick();
      for (let brick = 0; brick < capacity; brick += 1) {
        if (firstResident[brick] < 0 && (states[brick] & FLUID_BRICK_RESIDENT) !== 0) firstResident[brick] = step;
        if (firstWet[brick] < 0 && minimumPhi[brick] < 0) firstWet[brick] = step;
      }
    }
    // Pre-activation invariant: no brick may get wet without having been
    // scheduled on an earlier step (bricks wet from the first sample carry the
    // initial condition and are excluded).
    let lateWetBricks = 0;
    for (let brick = 0; brick < capacity; brick += 1) {
      if (firstWet[brick] <= 0) continue;
      lateWetBricks += 1;
      assert.ok(firstResident[brick] >= 0 && firstResident[brick] < firstWet[brick],
        `brick ${brick} became wet at step ${firstWet[brick]} but resident only at ${firstResident[brick]}`);
    }
    assert.ok(lateWetBricks > 0, "the dam front must reach previously dry bricks during the test window");
    const stats = await solver.readStats();
    assert.ok((stats.fluidBrickAtlasResidentTiles ?? 0) > 0, "resident bricks hold atlas tiles");
    assert.equal(stats.fluidBrickAtlasOverflow, 0);
    assert.equal(stats.fluidBrickAtlasResidentTiles, stats.fluidBrickResidentCount,
      "every resident brick owns exactly one atlas tile when the pool has headroom");
    assert.ok((stats.fluidBrickAtlasCapacity ?? 0) <= internals.octreeProjection.sparseBrickWorld.atlas!.plan.capacity);
    // Mirror-mode round trip: FP32 trilinear through the atlas tile (payload +
    // apron) must reproduce dense sampling exactly, including across seams.
    assert.ok((stats.fluidBrickAtlasMaxPhiErrorManual ?? 1) <= 1e-5,
      `atlas phi round-trip error ${stats.fluidBrickAtlasMaxPhiErrorManual}`);
    assert.ok((stats.fluidBrickAtlasMaxVelocityErrorManual ?? 1) <= 1e-5,
      `atlas velocity round-trip error ${stats.fluidBrickAtlasMaxVelocityErrorManual}`);
    if (device.features.has("float32-filterable" as GPUFeatureName)) {
      // Hardware trilinear quantizes footprint weights; bound the error by a
      // small multiple of the per-cell field variation.
      assert.ok((stats.fluidBrickAtlasMaxPhiError ?? 1) <= 0.02,
        `hardware phi sampling error ${stats.fluidBrickAtlasMaxPhiError}`);
      assert.ok((stats.fluidBrickAtlasMaxVelocityError ?? 1) <= 0.05,
        `hardware velocity sampling error ${stats.fluidBrickAtlasMaxVelocityError}`);
    }
    phiReadback.destroy();
    solver.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});
