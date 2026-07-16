import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { uniformMethod } from "../lib/methods/uniform";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { advanceRigidBodies, initializeRigidBodies, type RigidBodyState } from "../lib/rigid-body";
import { externalLoadsFromGPU } from "../lib/simulation/gpu-loads";
import { mergeGPURigidLoads, type GPURigidLoad } from "../lib/webgpu-eulerian";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("GPU buoyancy does not eject a light sphere", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU buoyancy checks" }, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice(), validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));

  try {
    for (const method of [tallCellMethod, uniformMethod]) {
      const scene: SceneDescription = cloneScene(defaultScene), radius = 0.08, waterHeight = scene.container.height_m * 0.6;
      scene.sceneId = `test-buoyant-sphere-${method.id}`;
      scene.fluid.initialCondition = "tank-fill";
      scene.fluid.surfaceTension_N_m = 0;
      delete scene.fluid.inflow;
      scene.container.fillFraction = 0.6;
      scene.container.top = "open";
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
      scene.rigidBodies = [{
        id: "buoyant-sphere", name: "Buoyant sphere", shape: "sphere",
        dimensions_m: { x: radius, y: radius, z: radius }, density_kg_m3: 100,
        position_m: { x: 0, y: 0.3, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
        linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
        restitution: 0.05, friction: 0.8, motion: "static"
      }];

      const values = Object.fromEntries(method.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
      let pending: GPURigidLoad[] = [];
      const solver = method.createSolver!(device, scene, "balanced", values, (incoming) => { pending = mergeGPURigidLoads(pending, incoming); }) as {
        advanceTo(time_s: number, bodies: RigidBodyState[]): boolean;
        destroy(): void;
      };
      const bodies = initializeRigidBodies(scene.rigidBodies), body = bodies[0], dt = scene.numerics.fixedDt_s;
      const sphereVolume = 4 / 3 * Math.PI * radius ** 3;
      let time = 0, peakVelocity = 0, peakHeight = body.position_m.y, maximumAirborneBuoyancy_g = 0, maximumResolvedWetRatio = 0;
      const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));

      while (time < 1.5 - 1e-9) {
        if (time >= 0.25 && body.description.motion === "static") body.description.motion = "dynamic";
        const coupling = externalLoadsFromGPU(scene, pending, dt, bodies);
        const displaced = coupling.diagnostics.displacedVolume_m3;
        if (body.position_m.y > waterHeight + radius) maximumAirborneBuoyancy_g = Math.max(maximumAirborneBuoyancy_g, scene.fluid.density_kg_m3 * displaced / body.mass_kg);
        const depth = Math.max(0, Math.min(2 * radius, waterHeight - (body.position_m.y - radius)));
        const analytic = Math.PI * depth ** 2 * (3 * radius - depth) / 3;
        // Ratios become grid-noise dominated for the final sliver of a cap.
        if (analytic > 0.05 * sphereVolume) maximumResolvedWetRatio = Math.max(maximumResolvedWetRatio, displaced / analytic);
        advanceRigidBodies(bodies, scene, dt, 6, coupling.loads);
        while (!solver.advanceTo(time + dt, bodies)) await nextTurn();
        time += dt;
        await device.queue.onSubmittedWorkDone();
        await nextTurn();
        peakVelocity = Math.max(peakVelocity, body.linearVelocity_m_s.y);
        peakHeight = Math.max(peakHeight, body.position_m.y);
      }

      solver.destroy();
      assert.ok(maximumAirborneBuoyancy_g <= 1, `${method.id}: airborne buoyancy reached ${maximumAirborneBuoyancy_g.toFixed(3)} g`);
      assert.ok(maximumResolvedWetRatio <= 2, `${method.id}: displaced/analytic volume reached ${maximumResolvedWetRatio.toFixed(3)}`);
      assert.ok(peakVelocity <= 2.5, `${method.id}: peak vy reached ${peakVelocity.toFixed(3)} m/s`);
      assert.ok(peakHeight - radius - waterHeight <= 0.3, `${method.id}: sphere rose ${(peakHeight - radius - waterHeight).toFixed(3)} m above the surface`);
    }
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
    Reflect.deleteProperty(globalThis, "navigator");
  }
});
