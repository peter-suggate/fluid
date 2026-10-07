import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { ApicTransport, type ApicInfo } from "../lib/methods/particle/transport";
import { apicMethod, APIC_CONFIGURATION } from "../lib/methods/particle/method";
import { MacGridSolver } from "../lib/methods/mac-shared/solver";
import type { MacMultigrid } from "../lib/methods/mac-shared/multigrid";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("PIC/FLIP preserves increments, blends with PIC and switches live", { timeout: 120_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "particle transfer modes");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    Object.assign(scene.container, { width_m: 1, height_m: 1, depth_m: 1, fillFraction: 0.5, top: "closed", fluidWallMode: "free-slip" });
    scene.voxelDomain.finestCellSize_m = 1 / 8; scene.rigidBodies = []; scene.solidVoxels = [];
    Object.assign(scene.fluid, { initialCondition: "tank-fill", initialLiquidVolumes: [], initialBrickSeeds_m: [], initialHeightField: undefined,
      initialVelocity_m_s: { x: 0, y: 0, z: 0 }, inflow: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0, gravity_m_s2: { x: 0, y: 0, z: 0 } });
    const read = async (buffer: GPUBuffer) => {
      const target = device!.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size); device!.queue.submit([encoder.finish()]);
        await target.mapAsync(GPUMapMode.READ); return new Float32Array(target.getMappedRange().slice(0));
      } finally { target.unmap(); target.destroy(); }
    };
    await t.test("pre-force snapshot survives projection overwrite and affine rows are ignored", async () => {
      const n = 8, h = 1 / n, cells = n ** 3, owned: GPUBuffer[] = [];
      const buffer = (size: number, uniform = false) => {
        const result = device!.createBuffer({ size, usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); owned.push(result); return result;
      };
      const parameters = buffer(80, true), scalars = buffer(128), solidBuffer = buffer(cells * 4), velocity = buffer(cells * 16), transferredVelocity = buffer(cells * 16), phi = buffer((n + 1) ** 3 * 4);
      const params = new ArrayBuffer(80); new Uint32Array(params).set([n, n, n]); new Float32Array(params).set([h, h, h, 0], 4); new Float32Array(params)[12] = 1000;
      device!.queue.writeBuffer(parameters, 0, params);
      const receipt = new Float32Array(32); receipt[20] = 1; receipt[21] = 1; device!.queue.writeBuffer(scalars, 0, receipt);
      const transport = await ApicTransport.create({ device: device!, scene, values: { transferMode: "flip", flipRatio: 1 }, dimensions: [n, n, n], h: [h, h, h],
        initialPhi: new Float32Array((n + 1) ** 3).fill(-1), solids: new Uint32Array(cells), parameters, scalars, solidBuffer, velocity, transferredVelocity, phi });
      try {
        const particlesBuffer = transport.debug.apicParticles as GPUBuffer, original = await read(particlesBuffer);
        for (let i = 0; i < original.length; i += 20) {
          original.set([0.3 + ((i / 20) % 2 ? 0.2 : -0.2), -0.1, 0.15], i + 4);
          // These stale APIC rows must not leak into a live PIC/FLIP transfer.
          original.set([4, -3, 2, 0, -2, 1, 5, 0, 2, -4, 3, 0], i + 8);
          original[i + 8] *= (i / 20) % 2 ? 1 : -1;
        }
        const weight = (x: number) => { const a = Math.abs(x); return a < 0.5 ? 0.75 - a * a : a < 1.5 ? 0.5 * (1.5 - a) ** 2 : 0; };
        const sample = (field: Float32Array, position: Float32Array, axis: number) => {
          const g = Array.from(position, (x, a) => x / h - (a === axis ? 1 : 0.5)), base = g.map(x => Math.floor(x - 0.5)); let value = 0;
          for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
            const q = [base[0] + x, base[1] + y, base[2] + z];
            value += weight(q[0] - g[0]) * weight(q[1] - g[1]) * weight(q[2] - g[2]) * field[4 * (q[0] + n * (q[1] + n * q[2])) + axis];
          }
          return value;
        };
        // First case uses constructor values; remaining cases exercise live updates.
        for (const [mode, ratio] of [["flip", 1], ["flip", 0], ["flip", 0.95], ["pic", 1]] as const) {
          if (mode !== "flip" || ratio !== 1) transport.applyRuntimeValues({ transferMode: mode, flipRatio: ratio });
          device!.queue.writeBuffer(particlesBuffer, 0, original);
          const transfer = device!.createCommandEncoder(); transport.encodeInitial(transfer); transport.encodeTransfer(transfer); device!.queue.submit([transfer.finish()]);
          const before = await read(transferredVelocity), delta = [0.07, -0.03, 0.02];
          for (let a = 0; a < 3; a++) {
            const q = [3.5, 3.5, 3.5]; q[a] += 0.5; let mass = 0, momentum = 0;
            for (let i = 0; i < original.length; i += 20) {
              const w = weight(q[0] - original[i] / h) * weight(q[1] - original[i + 1] / h) * weight(q[2] - original[i + 2] / h) * original[i + 3];
              mass += w; momentum += w * original[i + 4 + a];
            }
            assert.ok(Math.abs(before[4 * (3 + n * (3 + n * 3)) + a] - momentum / mass) < 1e-6, "PIC/FLIP P2G must ignore affine rows");
          }
          const after = before.slice(); for (let i = 0; i < after.length; i += 4) for (let a = 0; a < 3; a++) after[i + a] += delta[a];
          device!.queue.writeBuffer(velocity, 0, after);
          device!.queue.writeBuffer(transferredVelocity, 0, new Float32Array(before.length).fill(999));
          const move = device!.createCommandEncoder(); transport.encodeMove(move); device!.queue.submit([move.finish()]);
          const actual = await read(particlesBuffer); let checked = 0;
          for (let i = 0; i < original.length; i += 20) {
            const position = original.slice(i, i + 3); if (position.some(x => x < 0.3 || x > 0.7)) continue;
            for (let a = 0; a < 3; a++) {
              const pic = sample(after, position, a), flip = original[i + 4 + a] + pic - sample(before, position, a);
              const blend = mode === "pic" ? 0 : ratio, expected = (1 - blend) * pic + blend * flip;
              assert.ok(Math.abs(actual[i + 4 + a] - expected) < 2e-6, `${mode} ${ratio}: ${actual[i + 4 + a]} vs ${expected}`);
            }
            assert.ok(actual.slice(i + 8, i + 20).every(x => x === 0), "non-affine mode must clear old rows"); checked++;
          }
          assert.ok(checked > 100);
        }
      } finally { transport.destroy(); owned.forEach(b => b.destroy()); }
    });
    await t.test("solver switches APIC, PIC and FLIP without resetting particles or time", async () => {
      const solver = await apicMethod.createSolverAsync!(device!, scene, "balanced", {}, undefined, () => {}) as MacGridSolver;
      try {
        const particles = solver.debug!.apicParticles; let target = 0;
        for (const transferMode of ["pic", "flip", "apic"]) {
          solver.applyRuntimeValues({ transferMode, flipRatio: 0.95 }); target += 1 / 120;
          solver.advanceTo(target, []); await solver.awaitFrameCompletion();
          assert.equal(solver.debug!.apicParticles, particles); assert.ok(Math.abs(solver.info.completedTime_s! - target) < 1e-8);
          assert.ok(solver.info.pressureSolveConverged); assert.ok((await read(particles as GPUBuffer)).every(Number.isFinite));
          assert.equal((solver.info as ApicInfo).apicTransferMode, transferMode === "pic" ? 1 : transferMode === "flip" ? 2 : 0);
        }
      } finally { solver.destroy(); }
    });
    await t.test("bounded continuations preserve the complete frame and defer runtime edits", async () => {
      const moving = structuredClone(scene); moving.container.fillFraction = 0;
      moving.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.2 }];
      moving.fluid.initialVelocity_m_s = { x: 0.1, y: -2, z: 0.05 }; moving.numerics.fixedDt_s = 0.03;
      // Submit this copy synchronously, before any continuation callback can
      // enqueue the next chunk. It observes the first chunk's publication.
      const surface = async (texture: GPUTexture) => {
        const n = 9, stride = 256, target = device!.createBuffer({ size: stride * n * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const encoder = device!.createCommandEncoder();
          encoder.copyTextureToBuffer({ texture }, { buffer: target, bytesPerRow: stride, rowsPerImage: n }, [n, n, n]); device!.queue.submit([encoder.finish()]);
          await target.mapAsync(GPUMapMode.READ); const mapped = target.getMappedRange(), values = new Float32Array(n ** 3);
          for (let row = 0; row < n * n; row++) values.set(new Float32Array(mapped, stride * row, n), n * row);
          return values;
        } finally { target.unmap(); target.destroy(); }
      };
      const surfaces: Float32Array[] = [], substeps: number[] = [];
      for (const continuationSubsteps of [undefined, 1]) {
        const solver = await MacGridSolver.createAsync(device!, moving, "balanced", { cfl: 0.1 }, undefined, undefined,
          { ...APIC_CONFIGURATION, continuationSubsteps });
        try {
          await solver.awaitFrameCompletion(); const initial = await surface(solver.denseLevelSetVolumeSource.vertexPhi);
          const writtenClocks: number[] = []; let clock = solver.info.completedTime_s!;
          Object.defineProperty(solver.info, "completedTime_s", { configurable: true, enumerable: true,
            get: () => clock, set: (value: number) => { writtenClocks.push(value); clock = value; } });
          const mg = (solver as unknown as { multigrid: MacMultigrid }).multigrid, encode = mg.encode.bind(mg), limits: number[] = [];
          mg.encode = (encoder, limit) => { limits.push(limit); encode(encoder, limit); };
          assert.equal(solver.advanceTo(0.03, []), true);
          if (continuationSubsteps) {
            assert.equal(solver.framePending, true); assert.equal(solver.info.completedTime_s, 0);
            assert.equal(limits.length, 1, "only one substep may be encoded before its receipt");
            assert.equal(solver.advanceTo(0.04, []), false, "a pending frame must not accept a second advance");
            const incompleteSurface = surface(solver.denseLevelSetVolumeSource.vertexPhi);
            solver.applyRuntimeValues({ cfl: 0.1, pressureLimit: 1, transferMode: "pic" });
            assert.deepEqual(await incompleteSurface, initial, "an incomplete chunk must retain the last accepted publication");
          }
          await solver.awaitFrameCompletion();
          assert.equal(solver.framePending, false); assert.ok(Math.abs(solver.info.completedTime_s! - 0.03) < 1e-8);
          assert.ok(writtenClocks.length > 0 && writtenClocks.every(value => Math.abs(value - 0.03) < 1e-8), "completed time advances only to the complete target");
          assert.ok(solver.info.lastSubsteps! >= 3, "fixture must exercise several actual substeps");
          assert.ok(limits.every(limit => limit === 32), "runtime pressure edits must not change an in-flight frame's iteration limit");
          assert.equal((solver.info as ApicInfo).apicTransferMode, 0, "in-flight frame retains APIC transfer");
          assert.ok(solver.info.pressureSolveConverged);
          const final = await surface(solver.denseLevelSetVolumeSource.vertexPhi); surfaces.push(final); substeps.push(solver.info.lastSubsteps!);
          assert.ok(final.some((value, i) => Math.abs(value - initial[i]) > 0.001), "the accepted surface must actually move");
          if (continuationSubsteps) {
            solver.applyRuntimeValues({ cfl: 0.1, pressureLimit: 32, transferMode: "pic" });
            assert.equal(solver.advanceTo(0.031, []), true); await solver.awaitFrameCompletion();
            assert.equal((solver.info as ApicInfo).apicTransferMode, 1, "the following frame adopts the live transfer edit");
          }
        } finally { solver.destroy(); }
      }
      assert.equal(substeps[0], substeps[1]);
      let maximumDifference = 0;
      for (let i = 0; i < surfaces[0].length; i++) maximumDifference = Math.max(maximumDifference, Math.abs(surfaces[0][i] - surfaces[1][i]));
      assert.ok(maximumDifference < 2e-5, `continuation changed the surface by ${maximumDifference} m`);
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); releaseWebGPUExclusiveLock(); }
});
