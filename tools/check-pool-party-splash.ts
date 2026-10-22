/** Full authored pool scene through ball impact, using the UI's Box3D/fluid coupling.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-pool-party-splash.ts
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { interactionScene } from "../lib/interaction/adapters/scene";
import { SceneInteractionHost } from "../lib/interaction/runtime/host";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";

import { sceneFluidFrame } from "../lib/core/scene-lattice";
import type { FluidExchange } from "../lib/interaction/contracts";
import type { Vec3 } from "../lib/core/model";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_DETAIL_CANONICAL_LOAD } from "../lib/core/uniform-detail-abi";
import { readMixedBuffer } from "../tests/helpers/uniform-mixed-native-fields";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";

/** Query the live GPU ownership, including reconstructed coarse vertices. Raw
 * fine phi texels are not authoritative after narrow-band retirement. */
async function createWaterPointProbe(solver: WebGPUUniformReferenceSolver) {
  const { device, mixedFrame: frame } = solver as unknown as { device: GPUDevice;
    mixedFrame: { ownership: UniformMixedOwnership; solid: UniformMixedSolid; fields: { phi: GPUTexture } } };
  const ownership = frame.ownership, capacity = 32;
  const points = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const layout = uniformDetailBindLayout(device, { entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const module = uniformDetailModule(device, { label: "Pool exterior water diagnostics", code: uniformMixedTopologyWGSL(ownership.capacity, 0) + `
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> points:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> result:array<vec4f>;
fn umLoadVertex(p:vec3u)->f32{return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("", false)}
${uniformMixedSolidWGSL(2, frame.solid.coarse?.count)}
@compute @workgroup_size(32) fn sample(@builtin(local_invocation_index) i:u32){
 let p=points[i].xyz+vec3f(umWallLow());let c=vec3i(floor(p));
 result[i]=vec4f(0);if(points[i].w==0.0||!umCellPresent(c)){return;}
 let owner=umOwnerAt(c);result[i]=vec4f(umSampleVertex(p),umCellOpen(c),f32(owner.width),1.0);
}` });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, [], "exterior-water diagnostic shader compiles");
  const pipeline = await uniformDetailPipeline(device, ownership, { layout: device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, layout, frame.solid.tileLayout] }), compute: { module, entryPoint: "sample" } });
  const group = uniformDetailGroup(device, { layout, entries: [
    { binding: 0, resource: frame.fields.phi }, { binding: 1, resource: { buffer: points } }, { binding: 2, resource: { buffer: output } },
  ] });
  return {
    async read(coordinates: readonly (readonly number[])[]) {
      assert.ok(coordinates.length <= capacity);
      const input = new Float32Array(capacity * 4);
      coordinates.forEach((p, i) => input.set([p[0]!, p[1]!, p[2]!, 1], i * 4));
      device.queue.writeBuffer(points, 0, input);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(uniformDetailPick(pipeline)); pass.setBindGroup(0, ownership.bindGroup);
      pass.setBindGroup(1, group.group); pass.setBindGroup(2, frame.solid.tileGroup); pass.dispatchWorkgroups(1); pass.end();
      device.queue.submit([encoder.finish()]);
      return readMixedBuffer(device, output);
    },
    destroy() { points.destroy(); output.destroy(); },
  };
}

paceDawnEventPolling();
const output = "artifacts/svo-build-edit/pool-party-splash.json";
const rows: Record<string, unknown>[] = [], gpuErrors: string[] = [];
const began = performance.now();
const report: Record<string, unknown> = {
  scene: "backyard-pool-party", harness: "authored Box3D + production NB-FLIP",
  requestedSteps: 960, dt_s: 1 / 60, requestedTime_s: 16,
  completedSteps: 0, completedTime_s: 0, passed: false, rows, gpuErrors,
};
const save = () => {
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
};
save();
try {
  await withUniformDevice("pool party splash", async device => {
    device.addEventListener("uncapturederror", event => { gpuErrors.push(event.error.message); });
    const scene = createBackyardPoolPartyScene();
    console.log("Preparing authored pool solver");
    const manager = gpuCompilationManagerFor(device);
    const progress = setInterval(() => {
      const snapshot = manager.snapshot();
      console.log(JSON.stringify({ phase: "compile", active: snapshot.active, queued: snapshot.queued, current: snapshot.progress?.label }));
    }, 30_000);
    const solver = await createUniformSolver(device, interactionScene(scene)).finally(() => clearInterval(progress));
    report.grid = [solver.info.nx, solver.info.ny, solver.info.nz];
    report.allocatedBytes = solver.info.allocatedBytes;
    report.initialization_ms = performance.now() - began;
    let latestExchange: FluidExchange | undefined;
    let latestFluidPosition: Vec3 | undefined;
    const totalImpulse = { x: 0, y: 0, z: 0 };
    const host = new SceneInteractionHost(scene, { fluid: { step: async request => {
      latestFluidPosition = request.bodies.find(body => body.description.id === "pool-party/ball")?.position_m;
      const packets = await solver.advanceExternalRigidStep(request);
      latestExchange = packets.find(packet => packet.bodyId === "pool-party/ball");
      if (latestExchange) for (const axis of ["x", "y", "z"] as const) totalImpulse[axis] += latestExchange.linearImpulse_N_s[axis];
      return packets;
    } } });
    const fluidFrame = sceneFluidFrame(scene), cellVolume = fluidFrame.cellSize_m.reduce((a, b) => a * b, 1);
    let volumeProbe: Awaited<ReturnType<typeof createNbflipVolumeProbe>> | undefined;
    let pointProbe: Awaited<ReturnType<typeof createWaterPointProbe>> | undefined;
    let initialVolume = 0, minimumVolume = Infinity;
    const volumeFailures: string[] = [], couplingAnomalies: string[] = [];
    report.couplingAnomalies = couplingAnomalies;
    const measureWater = async (position: Vec3) => {
      const values = await volumeProbe!.read();
      let volume = 0, invalid = 0;
      for (const amount of values) {
        if (!Number.isFinite(amount) || amount < 0) invalid++;
        volume += amount * cellVolume;
      }
      assert.equal(invalid, 0, "finite nonnegative represented water volume");
      minimumVolume = Math.min(minimumVolume, volume);
      // Sample beyond the analytic sphere and its nearest solid voxel layer.
      // Use the pre-Box3D-step pose: the fluid fields/exchange belong to it.
      const radius = scene.rigidBodies.find(body => body.id === "pool-party/ball")!.dimensions_m.x;
      const shellRadius = radius + 2 * Math.max(...fluidFrame.cellSize_m);
      const samples: { kind: string; position_m: number[] }[] = [];
      if (position.y - radius < 1.8) for (let i = 0; i < 24; i++) {
        const y = 1 - 2 * (i + .5) / 24, r = Math.sqrt(1 - y * y), angle = i * Math.PI * (3 - Math.sqrt(5));
        samples.push({ kind: "ball-shell", position_m: [position.x + shellRadius * r * Math.cos(angle), position.y + shellRadius * y, position.z + shellRadius * r * Math.sin(angle)] });
      }
      for (const x of [-3, 0, 3]) for (const z of [-1, 1]) samples.push({ kind: "pool-reference", position_m: [x, 1, z] });
      const points = await pointProbe!.read(samples.map(sample => sample.position_m.map((value, axis) => (value - fluidFrame.container_m[axis]!) / fluidFrame.cellSize_m[axis]!)));
      const sampled = samples.map((sample, i) => ({ ...sample, phi_m: points[4 * i]!, openFraction: points[4 * i + 1]!, ownerWidth_cells: points[4 * i + 2]!, present: points[4 * i + 3] === 1 }));
      assert.ok(sampled.every(sample => Number.isFinite(sample.phi_m) && Number.isFinite(sample.openFraction)), "finite exterior water samples");
      const openShell = sampled.filter(sample => sample.kind === "ball-shell" && sample.present && sample.openFraction > .5);
      return { representedVolume_m3: volume, retainedVolumeFraction: initialVolume > 0 ? volume / initialVolume : 1,
        shellRadius_m: shellRadius, shellCenter_m: position, openShellSamples: openShell.length,
        wetOpenShellSamples: openShell.filter(sample => sample.phi_m < 0).length, samples: sampled };
    };
    let submerged = false, minY = Infinity, previousTime = 0;
    try {
      await host.ready;
      assert.equal(host.getView().readiness.state, "ready");
      volumeProbe = await createNbflipVolumeProbe(device, solver);
      pointProbe = await createWaterPointProbe(solver);
      const initialBall = host.currentBodies().find(body => body.description.id === "pool-party/ball")!;
      const initialWater = await measureWater(initialBall.position_m);
      initialVolume = initialWater.representedVolume_m3;
      assert.ok(initialVolume > 0, "authored pool starts with represented water");
      report.initialWater = initialWater;
      report.volumeInvariant = { minimumRetainedFraction: .8, initialVolume_m3: initialVolume, failures: volumeFailures };
      save();
      console.log("Pool solver ready");
      for (let frame = 1; frame <= 960; frame++) {
        report.attemptedStep = frame;
        await host.advance(1 / 60);
        const view = host.getView(), time = solver.info.completedTime_s;
        report.completedTime_s = time;
        report.readiness = view.readiness;
        report.simulationPipelineError = solver.info.simulationPipelineError ?? null;
        assert.equal(solver.info.simulationPipelineError, undefined, `step ${frame}: solver pipeline healthy`);
        assert.ok(typeof time === "number" && Number.isFinite(time) && time > previousTime,
          `step ${frame}: fluid clock must advance`);
        assert.ok(Math.abs(time - frame / 60) < 1e-7, `step ${frame}: requested fluid interval completed`);
        previousTime = time;
        assert.equal(view.readiness.state, "ready", `step ${frame}: interaction host remains ready`);
        assert.equal(view.snapshot?.tick, frame, `step ${frame}: Box3D clock agrees with fluid`);
        const bodies = host.currentBodies();
        for (const body of bodies) {
          assert.ok([body.position_m, body.orientation, body.linearVelocity_m_s, body.angularVelocity_rad_s]
            .every(vector => Object.values(vector).every(Number.isFinite)),
          `step ${frame}: finite pose and velocity for ${body.description.id}`);
        }
        const ball = bodies.find(body => body.description.id === "pool-party/ball");
        assert.ok(ball, `step ${frame}: authored pool ball remains present`);
        const ballSnapshot = view.snapshot!.bodies.find(body => body.bodyId === ball.description.id);
        assert.ok(ballSnapshot);
        const fraction = ballSnapshot.submergedFraction ?? 0;
        assert.ok(Number.isFinite(fraction) && fraction >= 0 && fraction <= 1, `step ${frame}: valid submerged fraction`);
        const particles = solver.narrowBandFlipInfo;
        if (particles) assert.ok(Number.isFinite(particles.particles) && particles.particles >= 0
          && particles.particles <= particles.capacity, `step ${frame}: particle count remains within capacity`);
        assert.deepEqual(gpuErrors, [], `step ${frame}: no uncaptured GPU errors`);
        minY = Math.min(minY, ball.position_m.y);
        submerged ||= fraction > .1;
        report.completedSteps = frame;
        report.minY_m = minY;
        report.submerged = submerged;
        if (frame === 1 || frame % 60 === 0) {
          const water = await measureWater(latestFluidPosition ?? ball.position_m);
          if (water.openShellSamples >= 8 && water.wetOpenShellSamples >= Math.ceil(.75 * water.openShellSamples)
            && (!latestExchange || latestExchange.displacedVolume_m3 <= 1e-8)) {
            couplingAnomalies.push(`step ${frame}: wet exterior shell but no displaced-water exchange`);
          }
          if (water.retainedVolumeFraction < .8) volumeFailures.push(`step ${frame}: retained ${(100 * water.retainedVolumeFraction).toFixed(2)}% of initial water`);
          const row = { water, exchange: latestExchange, cumulativeFluidImpulse_N_s: { ...totalImpulse },
            meanFluidForce_N: latestExchange ? Object.fromEntries(Object.entries(latestExchange.linearImpulse_N_s).map(([axis, impulse]) => [axis, impulse * 60])) : undefined,
            frame, time_s: frame / 60, completedTime_s: time, tick: view.snapshot!.tick,
            readiness: view.readiness.state, position: { ...ball.position_m }, submerged: fraction,
            particles: particles?.particles, particleCapacity: particles?.capacity,
            pagePool: solver.pagePoolInfo ? { ...solver.pagePoolInfo } : undefined };
          rows.push(row); save(); console.log(JSON.stringify({ ...row, water: { ...water, samples: undefined } }));
        }
      }
      report.minimumRepresentedVolume_m3 = minimumVolume;
      report.runtimePassed = true;
      assert.deepEqual(volumeFailures, [], "closed pool must not catastrophically lose represented water");
      assert.deepEqual(couplingAnomalies, [], "wet ball exterior must produce displaced-water exchange");
      assert.equal(host.getView().snapshot!.tick, 960);
      assert.equal(host.getView().readiness.state, "ready");
      assert.ok(submerged, "the ball must enter the water");
      assert.ok(minY < 1.8, "the ball must pass through the waterline");
      console.log(JSON.stringify({ result: "passed", minY, submerged, completedTime_s: solver.info.completedTime_s }));
    } finally {
      report.completedTime_s = solver.info.completedTime_s;
      report.simulationPipelineError = solver.info.simulationPipelineError ?? null;
      report.readiness = host.getView().readiness;
      try { await host.dispose(); } finally { pointProbe?.destroy(); volumeProbe?.destroy(); solver.destroy(); }
    }
  }, ["subgroups"]);
  report.passed = true;
} catch (error) {
  report.failure = error instanceof Error ? error.stack ?? error.message : String(error);
  throw error;
} finally {
  report.elapsed_ms = performance.now() - began;
  save();
}
