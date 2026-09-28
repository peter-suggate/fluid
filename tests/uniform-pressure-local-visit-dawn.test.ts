import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { refinementRegionLattice } from "../lib/core/refinement-regions";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("local coarse Jacobi visits match texture visits on minidam64", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform local pressure visits");
  let raw: GPUDevice | undefined;
  let solver: WebGPUUniformReferenceSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    raw = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    for (const mode of ["fine", "air"] as const) {
      const snapshots: { fields: Float32Array[]; residual: number | undefined }[] = [];
      let redirected = 0;
      for (const reference of [true, false]) {
        const visits = new Set<GPUComputePipeline>();
        let dispatchedVisits = 0;
        // Compile the old visit under the new plan's entry label. Bindings,
        // constants, sweep counts and all other simulation kernels are identical.
        const device = managedGPUDevice(new Proxy(raw, {
          get(target, key) {
            if (key === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
              const localVisit = descriptor.compute.entryPoint === "mgSmoothVisitLocalInPlace";
              if (reference && descriptor.compute.entryPoint === "mgSmoothVisitLocalInPlace") {
                redirected++;
                descriptor = { ...descriptor, compute: { ...descriptor.compute, entryPoint: "mgSmoothVisitInPlace" } };
              }
              const pipeline = await target.createComputePipelineAsync(descriptor);
              if (localVisit) visits.add(pipeline);
              return pipeline;
            };
            if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
              const encoder = target.createCommandEncoder(descriptor);
              return new Proxy(encoder, { get(e, property) {
                if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
                  const pass = e.beginComputePass(descriptor);
                  return new Proxy(pass, { get(p, name) {
                    if (name === "setPipeline") return (pipeline: GPUComputePipeline) => {
                      if (visits.has(pipeline)) dispatchedVisits++;
                      p.setPipeline(pipeline);
                    };
                    const value = Reflect.get(p, name, p);
                    return typeof value === "function" ? value.bind(p) : value;
                  } });
                };
                const value = Reflect.get(e, property, e);
                return typeof value === "function" ? value.bind(e) : value;
              } });
            };
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        }), { requireWorkerRealm: false });
        const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
        scene.fluid.refinementRegions = [];
        if (mode === "air") {
          const lattice = refinementRegionLattice(scene), axes = ["x", "y", "z"] as const;
          const corner = (offset: number) => Object.fromEntries(axes.map((axis, i) => [axis,
            lattice.origin_m[axis] + (lattice.dimensions[i]! - offset) * lattice.cellSize_m[i]!])) as {x:number;y:number;z:number};
          scene.fluid.refinementRegions = [{ id: "air", rule: "minimum-cell-size", minimumCellSize_cells: 4,
            maximumCellSize_cells: 4, min_m: corner(8), max_m: corner(4) }];
        }
        solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined,
          uniformGeometricSolverOptions({}, scene), () => {});
        for (let step = 1; step <= 4; step++) {
          assert.ok(solver.advanceTo(step / 30, [])); await solver.awaitFrameCompletion();
        }
        const fields = solver as unknown as { volumeA: GPUTexture; velocityA: GPUTexture; vertexPhiField: GPUTexture; boundaryVelocityA: GPUBuffer };
        snapshots.push({ fields: [await readMixedTexture(device, fields.volumeA), await readMixedTexture(device, fields.velocityA),
          await readMixedTexture(device, fields.vertexPhiField), await readMixedBuffer(device, fields.boundaryVelocityA)],
          residual: solver.info.uniformPressureAcceptedResidual });
        assert.ok(dispatchedVisits > 0, `${mode}: must execute coarse visits (${reference ? "reference" : "local"})`);
        solver.destroy(); solver = undefined;
      }
      assert.ok(redirected > 0, "reference must compile the previous visit");
      for (let field = 0; field < snapshots[0]!.fields.length; field++)
        assert.deepEqual(snapshots[1]!.fields[field], snapshots[0]!.fields[field], `${mode} field ${field} changed`);
      assert.equal(snapshots[1]!.residual, snapshots[0]!.residual, `${mode} residual changed`);
      assert.deepEqual(errors, []);
      console.log(`${mode}: all fields and accepted residual matched exactly`);
    }
  } finally { solver?.destroy(); raw?.destroy(); await releaseWebGPUExclusiveLock(); }
});
