import assert from "node:assert/strict";
import test from "node:test";
import "../../lib/methods";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { resolvedMethodValues } from "../../lib/core/stores/method-store";
import type { RigidBodyState } from "../../lib/core/rigid-body";
import { RustFluid3DGPUSolverAdapter, type RustFluid3DClient }
  from "../../lib/physics-wasm/fluid3d-gpu-adapter";
import { parsePhysicsReceipt, type PhysicsCommandReceipt, type PhysicsPublication }
  from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

interface NativeWorld {
  advance(sequence: number, dt_s: number): string;
  apply_command(command: string): string;
  receipt(): string;
  snapshot(mask: number): Uint8Array;
  free(): void;
}

class DirectWorldClient implements RustFluid3DClient {
  private world?: NativeWorld;
  private sequence = 0;
  private runEpoch = 0;
  private publicationId = 0;
  lastReceipt?: PhysicsCommandReceipt;
  releasedPublications = 0;

  constructor(private readonly wasm: { FluidWorld: {
    from_scene(scene: string, options: string): NativeWorld;
  } }) {}

  async load(scene: unknown, options: unknown): Promise<PhysicsCommandReceipt> {
    const record = options as { commandSequence?: number; runEpoch?: number };
    this.sequence = record.commandSequence ?? 0;
    this.runEpoch = record.runEpoch ?? 0;
    this.world = this.wasm.FluidWorld.from_scene(JSON.stringify(scene), JSON.stringify(options));
    return this.remember(this.world.receipt());
  }

  async advance(dt_s: number, viewMask = 0xffff_ffff): Promise<PhysicsPublication> {
    const world = this.requireWorld();
    this.sequence += 1;
    return this.publication(this.remember(world.advance(this.sequence, dt_s)), viewMask);
  }

  async applyCommand(command: unknown, viewMask?: number):
  Promise<PhysicsCommandReceipt | PhysicsPublication> {
    const world = this.requireWorld();
    this.sequence += 1;
    const receipt = this.remember(world.apply_command(JSON.stringify({
      ...(command as Record<string, unknown>),
      commandSequence: this.sequence, runEpoch: this.runEpoch,
    })));
    return viewMask === undefined ? receipt : this.publication(receipt, viewMask);
  }

  async snapshot(viewMask = 0xffff_ffff): Promise<PhysicsPublication> {
    return this.applyCommand({ type: "snapshot" }, viewMask) as Promise<PhysicsPublication>;
  }

  async destroy(): Promise<void> {
    this.world?.free();
    this.world = undefined;
  }

  private publication(receipt: PhysicsCommandReceipt, mask: number): PhysicsPublication {
    const bytes = this.requireWorld().snapshot(mask).slice();
    return {
      id: ++this.publicationId, revision: receipt, bytes,
      release: () => { this.releasedPublications += 1; },
    };
  }

  private remember(json: string): PhysicsCommandReceipt {
    return (this.lastReceipt = parsePhysicsReceipt(json));
  }

  private requireWorld(): NativeWorld {
    if (!this.world) throw new Error("direct FluidWorld client is not loaded");
    return this.world;
  }
}

function recordingGPUDevice() {
  let textureWrites = 0, bufferWrites = 0, destroyedTextures = 0, destroyedBuffers = 0;
  const device = {
    createTexture: () => ({ destroy: () => { destroyedTextures += 1; } }),
    createBuffer: () => ({ destroy: () => { destroyedBuffers += 1; } }),
    queue: {
      writeTexture: () => { textureWrites += 1; },
      writeBuffer: () => { bufferWrites += 1; },
    },
  } as unknown as GPUDevice;
  return { device, counts: () => ({ textureWrites, bufferWrites,
    destroyedTextures, destroyedBuffers }) };
}

test("real threaded Rust adapter bounds target-clock debt and publishes paused edits", async () => {
  Object.assign(globalThis, {
    GPUTextureUsage: { TEXTURE_BINDING: 1, COPY_DST: 2 },
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 },
  });
  const definition = findSceneDefinition("water-box-dam-break");
  assert.ok(definition, "water-box-dam-break must remain in the production catalogue");
  const scene = sceneDocument(definition);
  scene.numerics.fixedDt_s = 1 / 60;
  scene.numerics.maxDt_s = 1 / 60;
  scene.rigidBodies = [{
    id: "adapter-held-sphere", name: "Adapter held sphere", shape: "sphere",
    dimensions_m: { x: 0.06, y: 0.06, z: 0.06 }, density_kg_m3: 500,
    position_m: { x: 0.4, y: 0.65, z: 0.25 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.4,
  }];
  const methodValues = resolvedMethodValues({
    methodId: "adaptive-volume", quality: "balanced",
    overrides: { "adaptive-volume": { physicsExecutionBackend: "cpu" } },
  });
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "threaded", threadCount: 8 });
  const client = new DirectWorldClient(wasm);
  const gpu = recordingGPUDevice();
  const solver = await RustFluid3DGPUSolverAdapter.create(gpu.device, scene, {
    quality: "balanced", methodValues, runEpoch: 73, commandSequence: 0, tracerBudget: 0,
  }, { client });
  try {
    assert.equal(solver.info.completedTime_s, 0);
    for (let frame = 1; frame <= 4; frame += 1) {
      assert.equal(solver.advanceTo(1, []), true, `target-clock debt should admit frame ${frame}`);
      await solver.awaitFrameCompletion();
      assert.equal(client.lastReceipt?.frame, frame);
      assert.equal(solver.info.encodedSteps, frame,
        "each Rust frame must advance the renderer's surface-extraction revision");
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame / 30) < 1e-12);
    }
    assert.ok((solver.info.submittedTime_s ?? 0) < 1,
      "one admission must not collapse a one-second target jump into one Rust frame");

    const writesBeforeRuntime = gpu.counts().textureWrites;
    solver.applyRuntimeValues({ ...methodValues, timeStep: "scene" });
    assert.equal(solver.framePending, true);
    await solver.awaitFrameCompletion();
    assert.equal(solver.info.completedTime_s, 4 / 30,
      "a paused runtime edit must publish without advancing time");
    assert.equal(gpu.counts().textureWrites, writesBeforeRuntime + 3);

    assert.equal(solver.advanceTo(1, []), true);
    await solver.awaitFrameCompletion();
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - (4 / 30 + 1 / 60)) < 1e-12,
      "the next admission must use dtS from the paused Rust publication");

    const writesBeforeInjection = gpu.counts().textureWrites;
    const frameBeforeInjection = solver.info.encodedSteps;
    const surfaceBeforeInjection = solver.info.surfaceRevision;
    solver.injectLiquidBall({ centre_m: { x: 0.2, y: 0.6, z: 0 }, radius_m: 0.04 });
    await solver.awaitFrameCompletion();
    assert.equal(client.lastReceipt?.injections, 1);
    assert.equal(solver.info.encodedSteps, frameBeforeInjection,
      "a paused injection must not claim a physics step");
    assert.ok((solver.info.surfaceRevision ?? 0) > (surfaceBeforeInjection ?? 0),
      "a paused injection must invalidate retained surface geometry");
    assert.equal(gpu.counts().textureWrites, writesBeforeInjection + 3,
      "a paused injection must immediately refresh renderer textures");

    const held = {
      description: scene.rigidBodies[0], held: true,
      position_m: { x: 0.38, y: 0.67, z: 0.24 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 },
      angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    } as RigidBodyState;
    assert.equal(solver.advanceTo(solver.info.submittedTime_s ?? 0, [held]), false);
    await solver.awaitFrameCompletion();
    assert.deepEqual((await solver.readRigidBodyPoses())[0]?.position_m, {
      x: Math.fround(held.position_m.x), y: Math.fround(held.position_m.y),
      z: Math.fround(held.position_m.z),
    });
    assert.equal(client.lastReceipt?.commandSequence, 9);
    assert.equal(client.releasedPublications, 9,
      "initial, frame, and paused-command publications must all release exactly once");
  } finally { solver.destroy(); }
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.deepEqual(gpu.counts(), {
    textureWrites: 27, bufferWrites: 9, destroyedTextures: 3, destroyedBuffers: 1,
  });
});
