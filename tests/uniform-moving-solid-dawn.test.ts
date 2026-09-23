import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { CM12_TRANSPORT_FIXED_SCALE } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene, type RigidBodyDescription } from "../lib/core/model";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
async function read(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    const components = texture.format === "rgba32float" ? 4 : 1;
    const row = Math.ceil(texture.width * components * 4 / 256) * 256;
    const buffer = device.createBuffer({ size: row * texture.height * texture.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
        const e = device.createCommandEncoder();
        e.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: texture.height }, [texture.width, texture.height, texture.depthOrArrayLayers]);
        device.queue.submit([e.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const mapped = new Float32Array(buffer.getMappedRange());
        const result = new Float32Array(texture.width * texture.height * texture.depthOrArrayLayers * components);
        for (let z = 0; z < texture.depthOrArrayLayers; z++)
            for (let y = 0; y < texture.height; y++)
                result.set(mapped.subarray((z * texture.height + y) * row / 4, (z * texture.height + y) * row / 4 + texture.width * components), (z * texture.height + y) * texture.width * components);
        return result;
    }
    finally {
        buffer.unmap();
        buffer.destroy();
    }
}
function write(device: GPUDevice, texture: GPUTexture, values: Float32Array) { const components = texture.format === "rgba32float" ? 4 : 1; device.queue.writeTexture({ texture }, values as Float32Array<ArrayBuffer>, { bytesPerRow: texture.width * components * 4, rowsPerImage: texture.height }, [texture.width, texture.height, texture.depthOrArrayLayers]); }
const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0);
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("Uniform Geometric moving solids conserve and displace water", { timeout: 600000 }, async (t) => {
    await acquireWebGPUExclusiveLock("dawn-test", "uniform moving solids");
    let device: GPUDevice | undefined;
    try {
        const dawn = await import(pathToFileURL(modulePath!).href);
        Object.assign(globalThis, dawn.globals);
        const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
        const adapter = await gpu.requestAdapter();
        assert.ok(adapter);
        device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
        const errors: string[] = [];
        device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
        for (const mode of ["body", "voxel"] as const)
            await t.test(mode, async (fixture) => {
                let scene = cloneScene(defaultScene);
                scene.sceneId = "uniform-moving-solid-regression";
                scene.container = { ...scene.container, width_m: 0.8, height_m: 0.8, depth_m: 0.8, fillFraction: 0.5, top: "open", fluidWallMode: "free-slip" };
                scene.fluid.initialCondition = "tank-fill";
                scene.fluid.initialLiquidVolumes = [];
                scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
                scene.fluid.surfaceTension_N_m = 0;
                scene.fluid.dynamicViscosity_Pa_s = 0;
                delete scene.fluid.inflow;
                delete scene.terrain;
                scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
                scene.voxelDomain.finestCellSize_m = 0.05;
                scene.solidVoxels = [...solidVoxelShellForScene(scene)];
                const description: RigidBodyDescription = { id: "plunger", name: "Plunger", shape: "box", dimensions_m: { x: 0.3, y: 0.2, z: 0.3 }, density_kg_m3: 1000,
                    position_m: { x: 0, y: 0.6, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 }, linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0, friction: 0, motion: "dynamic" };
                scene.rigidBodies = mode === "body" ? [description] : [];
                const bodies = initializeRigidBodies(scene.rigidBodies);
                if (bodies[0])
                    bodies[0].held = true;
                const solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined, uniformGeometricSolverOptions({ timeStep: "scene", rigidCoupling: "off" }, scene), () => { });
                try {
                    const access = solver as unknown as {
                        shaderSource: string;
                        mainPipelineLayout: GPUPipelineLayout;
                        solidEntryScatterGroup: GPUBindGroup;
                        solidEntryResolveGroup: GPUBindGroup;
                        conditioningScratch: GPUBuffer;
                        pipelines: {
                            scatterSolidExcess: GPUComputePipeline;
                            resolveSolidExcess: GPUComputePipeline;
                        };
                        volumeB: GPUTexture;
                        velocityA: GPUTexture;
                        writeParams(dt: number, count: number, strength: number): void;
                        rigidSystem: {
                            syncBodies(b: typeof bodies): void;
                        };
                    };
                    if (mode === "body")
                        await fixture.test("co-moving fluid has zero relative solid flux", async () => {
                            bodies[0]!.position_m.y = 0.25;
                            bodies[0]!.linearVelocity_m_s = { x: 0.7, y: -1, z: 0.3 };
                            access.rigidSystem.syncBodies(bodies);
                            access.writeParams(1 / 120, 1, 0);
                            const velocity = new Float32Array(16 * 16 * 16 * 4);
                            for (let i = 0; i < velocity.length; i += 4)
                                velocity.set([0.7, -1, 0.3, 0], i);
                            write(device!, access.velocityA, velocity);
                            const module = device!.createShaderModule({ code: access.shaderSource.replace("if(params.dropExtent.w>0.5&&valid(id))", "if(false)") + `
@compute @workgroup_size(4,4,4) fn probeRelativeFlux(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!valid(id)){return;}textureStore(volumeOut,id,vec4f(divergenceAt(id,true)));}` });
                            const pipeline = await device!.createComputePipelineAsync({ layout: access.mainPipelineLayout, compute: { module, entryPoint: "probeRelativeFlux" } });
                            const encoder = device!.createCommandEncoder();
                            const pass = encoder.beginComputePass();
                            pass.setPipeline(pipeline);
                            pass.setBindGroup(0, access.solidEntryScatterGroup);
                            pass.dispatchWorkgroups(4, 4, 4);
                            pass.end();
                            device!.queue.submit([encoder.finish()]);
                            const divergence = await read(device!, access.volumeB);
                            let maximum = 0;
                            for (let z = 3; z < 13; z++)
                                for (let y = 2; y < 8; y++)
                                    for (let x = 3; x < 13; x++)
                                        maximum = Math.max(maximum, Math.abs(divergence[x + 16 * (y + 16 * z)]!));
                            console.log(JSON.stringify({ coMovingDivergence: maximum }));
                            // A tilted box and co-rotating liquid exercise off-axis
                            // quadrature, not just translation of aligned geometry.
                            bodies[0]!.orientation = { w: Math.cos(0.185), x: 0, y: Math.sin(0.185), z: 0 };
                            bodies[0]!.angularVelocity_rad_s = { x: 0, y: 2, z: 0 };
                            access.rigidSystem.syncBodies(bodies);
                            for (let z = 0; z < 16; z++)
                                for (let y = 0; y < 16; y++)
                                    for (let x = 0; x < 16; x++) {
                                        const wx = -0.4 + (x + 0.5) * 0.05, wz = -0.4 + (z + 0.5) * 0.05;
                                        velocity.set([0.7 + 2 * wz, -1, 0.3 - 2 * wx, 0], 4 * (x + 16 * (y + 16 * z)));
                                    }
                            write(device!, access.velocityA, velocity);
                            const rotatedEncoder = device!.createCommandEncoder(), rotatedPass = rotatedEncoder.beginComputePass();
                            rotatedPass.setPipeline(pipeline);
                            rotatedPass.setBindGroup(0, access.solidEntryScatterGroup);
                            rotatedPass.dispatchWorkgroups(4, 4, 4);
                            rotatedPass.end();
                            device!.queue.submit([rotatedEncoder.finish()]);
                            const rotated = await read(device!, access.volumeB);
                            let rotatingMaximum = 0;
                            for (let z = 3; z < 13; z++)
                                for (let y = 2; y < 8; y++)
                                    for (let x = 3; x < 13; x++)
                                        rotatingMaximum = Math.max(rotatingMaximum, Math.abs(rotated[x + 16 * (y + 16 * z)]!));
                            console.log(JSON.stringify({ coRotatingDivergence: rotatingMaximum }));
                            bodies[0]!.orientation = { w: 1, x: 0, y: 0, z: 0 };
                            bodies[0]!.angularVelocity_rad_s = { x: 0, y: 0, z: 0 };
                            bodies[0]!.position_m.y = 0.6;
                            bodies[0]!.linearVelocity_m_s = { x: 0, y: 0, z: 0 };
                            access.rigidSystem.syncBodies(bodies);
                            velocity.fill(0);
                            write(device!, access.velocityA, velocity);
                            assert.ok(maximum < 1e-5, `co-moving divergence ${maximum}`);
                            assert.ok(rotatingMaximum < 1e-4, `co-rotating divergence ${rotatingMaximum}`);
                        });
                    const initial = sum(await read(device!, solver.volumeTexture));
                    let worstLoss = 0;
                    let peakOutsideSpeed = 0;
                    let discardedDust = 0;
                    let dustRoundingBound = 0;
                    let worstUnexplainedLoss = 0;
                    const started = performance.now();
                    for (let step = 1; step <= 48; step++) {
                        if (mode === "body") {
                            bodies[0]!.position_m.y = 0.6 - Math.min(step, 36) / 120;
                            bodies[0]!.linearVelocity_m_s.y = step <= 36 ? -1 : 0;
                        }
                        if (mode === "voxel" && step === 1) {
                            scene = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [5, 4, 5], maximumExclusive: [11, 8, 11] }]);
                            solver.applySceneUniforms(scene);
                        }
                        assert.ok(solver.advanceTo(step / 120, bodies));
                        await solver.awaitFrameCompletion();
                        const v = await read(device!, solver.volumeTexture);
                        worstLoss = Math.max(worstLoss, (initial - sum(v)) / initial);
                        if (process.env.FLUID_COUPLING_TRACE)
                            console.log(JSON.stringify({ mode, step, mass: sum(v) }));
                        const stats = await solver.readStats() as typeof solver.info & {
                            uniformVolumeDustMass_cells?: number;
                            uniformVolumeDustCells?: number;
                            uniformVolumeDustThreshold?: number;
                        };
                        discardedDust += stats.uniformVolumeDustMass_cells ?? 0;
                        dustRoundingBound += (stats.uniformVolumeDustCells ?? 0) * (stats.uniformVolumeDustThreshold ?? 0) / 64;
                        worstUnexplainedLoss = Math.max(worstUnexplainedLoss, (initial - sum(v) - discardedDust - dustRoundingBound) / initial);
                        assert.ok(sum(v) + discardedDust - initial < 0.0001 * initial, "solid coupling must not create water");
                        assert.ok(v.every(x => Number.isFinite(x) && x >= -1e-6));
                        const vel = await read(device!, (solver as unknown as {
                            velocityA: GPUTexture;
                        }).velocityA);
                        for (let z = 0; z < 16; z++)
                            for (let y = 0; y < 8; y++)
                                for (let x = 0; x < 16; x++)
                                    if (x < 4 || x > 11 || z < 4 || z > 11) {
                                        const i = 4 * (x + 16 * (y + 16 * z));
                                        peakOutsideSpeed = Math.max(peakOutsideSpeed, Math.hypot(vel[i]!, vel[i + 1]!, vel[i + 2]!));
                                    }
                    }
                    const final = await read(device!, solver.volumeTexture);
                    let above = 0, buried = 0;
                    for (let z = 0; z < 16; z++)
                        for (let y = 0; y < 16; y++)
                            for (let x = 0; x < 16; x++) {
                                const v = final[x + 16 * (y + 16 * z)]!;
                                if (y >= 8)
                                    above += v;
                                if (x >= 5 && x < 11 && z >= 5 && z < 11 && y >= 4 && y < 8)
                                    buried += v;
                            }
                    const result = { mode, initial, final: sum(final), worstLoss, discardedDust, dustRoundingBound, worstUnexplainedLoss, above, buried, peakOutsideSpeed, msPerStep: (performance.now() - started) / 48 };
                    console.log(JSON.stringify(result));
                    assert.ok(worstUnexplainedLoss < 0.0001, `covered water lost: ${JSON.stringify(result)}`);
                    assert.ok(buried < 1e-3, `water remains inside solid: ${JSON.stringify(result)}`);
                    assert.ok(above > 70, `body volume must raise free surface: ${JSON.stringify(result)}`);
                    assert.ok(peakOutsideSpeed > 0.01, `solid must drive surrounding water: ${JSON.stringify(result)}`);
                    if (mode === "body") {
                        const timed = performance.now();
                        for (let step = 49; step <= 80; step++) {
                            bodies[0]!.position_m.x = (step - 49) * 0.002;
                            bodies[0]!.linearVelocity_m_s = { x: 0.24, y: 0, z: 0 };
                            assert.ok(solver.advanceTo(step / 120, bodies));
                            await solver.awaitFrameCompletion();
                        }
                        console.log(JSON.stringify({ mode, advanceWithoutFieldReadback_ms: (performance.now() - timed) / 32 }));
                    }
                    if (mode === "voxel") {
                        // Undo the submerged fill on the same solver. No reset or water
                        // reinjection may conceal a deletion at either edit boundary.
                        scene = sceneWithSolidStroke(scene, [{ operation: "clear", minimum: [5, 4, 5], maximumExclusive: [11, 8, 11] }]);
                        solver.applySceneUniforms(scene);
                        const beforeRemoval = sum(await read(device!, solver.volumeTexture));
                        // Disable the intentional dust sink for an exact edit/undo check.
                        solver.applyRuntimeValues({ volumeDustThreshold: 0, timeStep: "scene", rigidCoupling: "off" });
                        for (let step = 49; step <= 60; step++) {
                            assert.ok(solver.advanceTo(step / 120, bodies));
                            await solver.awaitFrameCompletion();
                        }
                        assert.ok(Math.abs(sum(await read(device!, solver.volumeTexture)) - beforeRemoval) < 0.0001 * initial, "removing the block conserves existing water");
                        const sealed = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [16, 16, 16] }]);
                        solver.applySceneUniforms(sealed);
                        assert.ok(solver.advanceTo(61 / 120, bodies));
                        await solver.awaitFrameCompletion();
                        assert.ok(Math.abs(sum(await read(device!, solver.volumeTexture)) - beforeRemoval) < 0.0001 * initial, "a sealed edit retains its unplaceable reservoir");
                        const sealedStats = await solver.readStats() as typeof solver.info & {
                            uniformUnplaceableSolidExcess_cells?: number;
                        };
                        assert.ok((sealedStats.uniformUnplaceableSolidExcess_cells ?? 0) > 0.9 * beforeRemoval, "sealed edits expose unresolved water");
                        solver.applySceneUniforms(scene);
                        assert.ok(solver.advanceTo(62 / 120, bodies));
                        await solver.awaitFrameCompletion();
                        assert.ok(Math.abs(sum(await read(device!, solver.volumeTexture)) - beforeRemoval) < 0.0001 * initial, "unsealing recovers the conserved reservoir");
                        // Four fixed-point units split six ways used to round
                        // the first five shares up and make the last negative.
                        solver.applySceneUniforms(sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [8, 4, 8], maximumExclusive: [9, 5, 9] }]));
                        const tiny = new Float32Array(16 * 16 * 16);
                        tiny[8 + 16 * (4 + 16 * 8)] = 4 / CM12_TRANSPORT_FIXED_SCALE;
                        write(device!, solver.volumeTexture, tiny);
                        access.writeParams(1 / 120, 0, 0);
                        const repair = device!.createCommandEncoder();
                        repair.clearBuffer(access.conditioningScratch, 0, tiny.length * 4);
                        for (const [pipeline, group] of [[access.pipelines.scatterSolidExcess, access.solidEntryScatterGroup], [access.pipelines.resolveSolidExcess, access.solidEntryResolveGroup]] as const) {
                            const pass = repair.beginComputePass();
                            pass.setPipeline(pipeline);
                            pass.setBindGroup(0, group);
                            pass.dispatchWorkgroups(4, 4, 4);
                            pass.end();
                        }
                        device!.queue.submit([repair.finish()]);
                        const tinyResult = await read(device!, solver.volumeTexture);
                        assert.ok(tinyResult.every(v => v >= 0), "small displacement cannot deposit negative water");
                        assert.equal(sum(tinyResult), sum(tiny), "rounded displacement conserves every integer unit");
                    }
                }
                finally {
                    solver.destroy();
                }
            });
        assert.deepEqual(errors, []);
    }
    finally {
        device?.destroy();
        await releaseWebGPUExclusiveLock();
    }
});
