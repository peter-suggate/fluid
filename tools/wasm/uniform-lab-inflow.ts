import "../../lib/methods";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { UniformLabController, uniformLabSceneLimitation } from "../../lib/physics-wasm/uniform-controller";
import { PhysicsWasmWorkerRuntime } from "../../lib/physics-wasm/worker-runtime";
import type { WorkerPort } from "../../lib/physics-wasm/client";
import type { PhysicsWorkerRequest, PhysicsWorkerResponse } from "../../lib/physics-wasm/protocol";
import type { FluidWasmModule } from "../../lib/physics-wasm/module";
const baseline = new Map<string, unknown>();
for (const artifact of ["scalar", "simd"] as const) {
  const root = new URL(
    `../../public/wasm/fluid-wasm/${artifact}/`,
    import.meta.url,
  );
  const wasm = (await import(
    new URL("fluid_wasm.js", root).href
  )) as FluidWasmModule;
  await wasm.default({
    module_or_path: readFileSync(new URL("fluid_wasm_bg.wasm", root)),
  });
  const factory = (): WorkerPort => {
    const listeners = new Set<
      (e: MessageEvent<PhysicsWorkerResponse>) => void
    >();
    const runtime = new PhysicsWasmWorkerRuntime(
      (message, transfer) => {
        const data = structuredClone(message, { transfer: transfer ?? [] });
        queueMicrotask(() =>
          listeners.forEach((listener) =>
            listener({ data } as MessageEvent<PhysicsWorkerResponse>),
          ),
        );
      },
      async () => ({ ...wasm, default: async () => {} }),
    );
    return {
      postMessage(message: PhysicsWorkerRequest, transfer?: Transferable[]) {
        runtime.receive(structuredClone(message, { transfer: transfer ?? [] }));
      },
      addEventListener(type: string, listener: unknown) {
        if (type === "message")
          listeners.add(
            listener as (e: MessageEvent<PhysicsWorkerResponse>) => void,
          );
      },
      terminate() {
        listeners.clear();
      },
    };
  };
  const controller = await UniformLabController.create({
    artifact,
    workerFactory: factory,
  });
  try {
    for (const id of ["hero-garden-hose", "water-box-dam-break"]) {
      const scene = sceneDocument(findSceneDefinition(id)!);
      if (id === "water-box-dam-break") {
        scene.container.fillFraction = 0;
        scene.fluid.initialCondition = "tank-fill";
        scene.fluid.inflow = {
          center_m: { x: 0, y: scene.container.height_m * 0.7, z: 0.2 },
          velocity_m_s: { x: 0.3, y: -0.6, z: 0 },
          radius_m: 0.04, length_m: 0.05, start_s: 0.05, end_s: 0.2, ramp_s: 0.03,
        };
      }
      assert.equal(uniformLabSceneLimitation(scene), undefined);
      let view = await controller.load(scene);
      const mass = view.volume.reduce((a, b) => a + b, 0);
      let dust = 0;
      for (let frame = 0; frame < 30; frame++) {
        view = await controller.advance(1 / 60);
        for (const field of [view.volume, view.phi, view.velocity, view.pressure]) {
          assert.ok(field.every(Number.isFinite), `${id}: finite frame ${frame}`);
        }
        const receipt = view.receipt.uniform as {
          injectedVolume: number; sharpeningDust: number; transport: { dustVolume: number };
        };
        dust += receipt.sharpeningDust + receipt.transport.dustVolume;
        if (id === "water-box-dam-break" && (frame < 2 || frame > 12)) {
          assert.equal(receipt.injectedVolume, 0, "source respects start and end times");
        }
      }
      const injected = Number(view.receipt.injectedVolume);
      assert.ok(injected > 0, `${id}: source emits water`);
      assert.ok(view.velocity.some(v => Math.abs(v) > 0.01), `${id}: water moves`);
      const finalMass = view.volume.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(finalMass + dust - mass - injected) < 1e-4 * Math.max(1, finalMass),
        `${id}: accounts for source mass: ${finalMass + dust - mass - injected}`);
      const fields = { volume: [...view.volume], phi: [...view.phi], velocity: [...view.velocity] };
      if (artifact === "scalar") baseline.set(id, fields);
      else assert.deepEqual(fields, baseline.get(id), `${id}: scalar/SIMD parity`);
      const reset = await controller.load(scene);
      assert.equal(reset.receipt.injectedVolume, 0);
      assert.equal(reset.revision.time, 0);
      const withoutSource = structuredClone(scene);
      delete withoutSource.fluid.inflow;
      await controller.load(withoutSource);
      const noSource = await controller.advance(1 / 60);
      assert.equal(noSource.receipt.injectedVolume, 0, "loading another scene clears inflow");
      console.log(JSON.stringify({ artifact, scene: id, frames: 30, injected, finalMass }));
    }
  } finally {
    await controller.destroy();
  }
}
