import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { apicMethod, APIC_CONFIGURATION } from "../lib/methods/particle/method";
import { ApicTransport, type ApicInfo } from "../lib/methods/particle/transport";
import { MacGridSolver } from "../lib/methods/mac-shared/solver";
import { MacMultigrid } from "../lib/methods/mac-shared/multigrid";
import { readFloatTexture3D, smokeRenderHybridPresentation } from "../lib/harness/webgpu-smoke-readbacks";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("APIC particle transport, incompressibility and publication", { timeout: 240_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "APIC particle plugin");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const base = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    Object.assign(base.container, { width_m: 1, height_m: 1, depth_m: 1, fillFraction: 0.5, top: "closed", fluidWallMode: "free-slip" });
    base.voxelDomain.finestCellSize_m = 1 / 16; base.rigidBodies = []; base.solidVoxels = [];
    Object.assign(base.fluid, { initialCondition: "tank-fill", initialLiquidVolumes: [], initialBrickSeeds_m: [], initialHeightField: undefined,
      initialVelocity_m_s: undefined, inflow: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0, gravity_m_s2: { x: 0, y: -9.81, z: 0 } });
    // Preserve the original stringent physics regressions independently of app defaults.
    const strictPressure = { pressureTolerance: 0.001, pressureRelativeReduction: 0 };
    const create = async (scene = base) => await apicMethod.createSolverAsync!(device!, scene, "balanced", strictPressure, undefined, () => {}) as MacGridSolver;
    const readBuffer = async (buffer: GPUBuffer) => {
      const target = device!.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size); device!.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ); const values = new Float32Array(target.getMappedRange().slice(0)); target.unmap(); target.destroy(); return values;
    };
    await t.test("quadratic MAC transfers reproduce affine velocity and its gradient in the interior", async () => {
      const n = 8, h = 1 / n, count = n ** 3;
      const buffers: GPUBuffer[] = [];
      const buffer = (size: number, uniform = false) => { const b = device!.createBuffer({ size, usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); buffers.push(b); return b; };
      const parameters = buffer(80, true), scalars = buffer(128), solidBuffer = buffer(count * 4), velocity = buffer(count * 16), transferredVelocity = buffer(count * 16), phi = buffer((n + 1) ** 3 * 4);
      const params = new ArrayBuffer(80); new Uint32Array(params).set([n, n, n]); new Float32Array(params).set([h, h, h, 0.001], 4); new Float32Array(params)[12] = 1000;
      device!.queue.writeBuffer(parameters, 0, params);
      const receipts = new Float32Array(32); receipts[20] = 1; receipts[21] = 1; device!.queue.writeBuffer(scalars, 0, receipts);
      const field = new Float32Array(count * 4), expected = (x: number, y: number, z: number) => [0.2 + 0.3 * x - 0.4 * y, -0.1 + 0.4 * x + 0.2 * z, 0.1 - 0.3 * y];
      for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let a = 0; a < 3; a++) {
        const q = [x + 0.5, y + 0.5, z + 0.5]; q[a] += 0.5;
        field[4 * (x + n * (y + n * z)) + a] = expected(q[0] * h, q[1] * h, q[2] * h)[a];
      }
      device!.queue.writeBuffer(velocity, 0, field);
      const transport = await ApicTransport.create({ device: device!, scene: base, dimensions: [n, n, n], h: [h, h, h], initialPhi: new Float32Array((n + 1) ** 3).fill(-1), solids: new Uint32Array(count), parameters, scalars, solidBuffer, velocity, transferredVelocity, phi });
      try {
        const encoder = device!.createCommandEncoder(); transport.encodeInitial(encoder); transport.encodeMove(encoder); transport.encodeTransfer(encoder); device!.queue.submit([encoder.finish()]);
        const particles = await readBuffer(transport.debug.apicParticles as GPUBuffer);
        let tested = 0;
        for (let i = 0; i < particles.length; i += 20) {
          const position = particles.slice(i, i + 3); if (position.some(v => v < 0.3 || v > 0.7)) continue;
          const v = expected(position[0], position[1], position[2]);
          for (let a = 0; a < 3; a++) assert.ok(Math.abs(particles[i + 4 + a] - v[a]) < 1e-5);
          assert.ok(Math.abs(particles[i + 8] - 0.3) < 1e-5); assert.ok(Math.abs(particles[i + 9] + 0.4) < 1e-5);
          assert.ok(Math.abs(particles[i + 12] - 0.4) < 1e-5); tested++;
        }
        assert.ok(tested > 100);
        const gathered = await readBuffer(transferredVelocity);
        for (let z = 3; z < 5; z++) for (let y = 3; y < 5; y++) for (let x = 3; x < 5; x++) for (let a = 0; a < 3; a++) {
          const i = 4 * (x + n * (y + n * z)) + a; assert.ok(Math.abs(gathered[i] - field[i]) < 1e-5, `affine round trip ${gathered[i]} vs ${field[i]}`);
        }
      } finally { transport.destroy(); for (const b of buffers) b.destroy(); }
    });
    await t.test("fresh multigrid residual retains small errors beside large pressure terms", async () => {
      const owned: GPUBuffer[] = [];
      const buffer = (bytes: number, uniform = false) => {
        const result = device!.createBuffer({ size: bytes, usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        owned.push(result); return result;
      };
      const matrix = new Float32Array(27 * 4), cg = new Float32Array(27 * 4), rhs = new Float32Array(27);
      const edge = Math.fround(1 / 0.0125 ** 2);
      for (let i = 0; i < 27; i++) {
        const x = i % 3, y = Math.floor(i / 3) % 3, z = Math.floor(i / 9);
        matrix.set([x < 2 ? edge : 0, y < 2 ? edge : 0, z < 2 ? edge : 0, Math.fround(6 * edge)], i * 4);
        cg[4 * i] = 0.1 + i * 0.001;
      }
      let expected = 0;
      for (let i = 0; i < 27; i++) {
        const q = [i % 3, Math.floor(i / 3) % 3, Math.floor(i / 9)], strides = [1, 3, 9];
        let applied = matrix[4 * i + 3] * cg[4 * i];
        for (let a = 0; a < 3; a++) {
          if (q[a] < 2) applied -= edge * cg[4 * (i + strides[a])];
          if (q[a] > 0) applied -= edge * cg[4 * (i - strides[a])];
        }
        rhs[i] = applied;
        expected = Math.max(expected, Math.abs(rhs[i] - applied));
      }
      assert.ok(expected > 1e-5, "fixture must contain cancellation error that binary32 Ax would hide");
      const fine = { matrix: buffer(matrix.byteLength), rhs: buffer(rhs.byteLength), cg: buffer(cg.byteLength), partial: buffer(32), scalars: buffer(144), params: buffer(80, true) };
      const scalars = new Float32Array(32); scalars[21] = 1;
      const params = new Float32Array(20); params[14] = 1e-9;
      device!.queue.writeBuffer(fine.matrix, 0, matrix); device!.queue.writeBuffer(fine.rhs, 0, rhs); device!.queue.writeBuffer(fine.cg, 0, cg);
      device!.queue.writeBuffer(fine.scalars, 0, scalars); device!.queue.writeBuffer(fine.params, 0, params);
      const mg = await MacMultigrid.create(device!, [3, 3, 3], fine);
      try {
        const encoder = device!.createCommandEncoder(); mg.encode(encoder, 0); device!.queue.submit([encoder.finish()]);
        const measured = (await readBuffer(fine.scalars))[4];
        assert.ok(Math.abs(measured - expected) < 1e-6, `GPU residual ${measured}, binary64 reference ${expected}`);
        // A single-float pressure cannot generally satisfy this stricter bound
        // beside the fixture's large terms. Verify the stored high/low solution
        // independently in binary64, not just the solver's own acceptance flag.
        params[14] = 1e-5;
        device!.queue.writeBuffer(fine.params, 0, params);
        device!.queue.writeBuffer(fine.scalars, 0, scalars);
        device!.queue.writeBuffer(fine.cg, 0, new Float32Array(cg.length));
        const solve = device!.createCommandEncoder(); mg.encode(solve, 64); device!.queue.submit([solve.finish()]);
        const receipt = await readBuffer(fine.scalars), solution = await readBuffer(fine.cg);
        let referenceResidual = 0;
        const pressure = (i: number) => solution[4 * i] + solution[4 * i + 2];
        for (let i = 0; i < 27; i++) {
          const q = [i % 3, Math.floor(i / 3) % 3, Math.floor(i / 9)], strides = [1, 3, 9];
          let residual = rhs[i] - matrix[4 * i + 3] * pressure(i);
          for (let a = 0; a < 3; a++) {
            if (q[a] < 2) residual += edge * pressure(i + strides[a]);
            if (q[a] > 0) residual += edge * pressure(i - strides[a]);
          }
          referenceResidual = Math.max(referenceResidual, Math.abs(residual));
        }
        t.diagnostic(`Pressure-pair residual: GPU ${receipt[4]}, binary64 reference ${referenceResidual}`);
        assert.equal(receipt[20], 1, `pressure pair did not converge: ${receipt[4]}`);
        assert.ok(referenceResidual <= 1.1e-5, `binary64 pressure-pair residual ${referenceResidual}`);
        // A short final substep can scale a valid pressure warm start far beyond
        // 1e7 per stencil term. Verify the range independently of convergence:
        // the previous accumulator returned its 1e30 overflow sentinel here.
        const largeMatrix = Float32Array.from(matrix, value => value * 100_000);
        const largeEdge = largeMatrix[0], largeRhs = new Float32Array(27);
        let largeExpected = 0;
        for (let i = 0; i < 27; i++) {
          const q = [i % 3, Math.floor(i / 3) % 3, Math.floor(i / 9)], strides = [1, 3, 9];
          let applied = largeMatrix[4 * i + 3] * cg[4 * i];
          for (let a = 0; a < 3; a++) {
            if (q[a] < 2) applied -= largeEdge * cg[4 * (i + strides[a])];
            if (q[a] > 0) applied -= largeEdge * cg[4 * (i - strides[a])];
          }
          largeRhs[i] = applied;
          largeExpected = Math.max(largeExpected, Math.abs(largeRhs[i] - applied));
        }
        assert.ok(largeMatrix[3] * cg[0] > 1e7 && largeExpected > 0.001);
        device!.queue.writeBuffer(fine.matrix, 0, largeMatrix);
        device!.queue.writeBuffer(fine.rhs, 0, largeRhs);
        device!.queue.writeBuffer(fine.cg, 0, cg);
        device!.queue.writeBuffer(fine.scalars, 0, scalars);
        const measureLarge = device!.createCommandEncoder(); mg.encode(measureLarge, 0);
        device!.queue.submit([measureLarge.finish()]);
        const largeReceipt = await readBuffer(fine.scalars);
        assert.ok(Math.abs(largeReceipt[4] - largeExpected) < 1e-6,
          `large-term GPU residual ${largeReceipt[4]}, binary64 reference ${largeExpected}`);
        assert.equal(largeReceipt[20], 0, "large cancellation error must not be accepted");
      } finally { mg.destroy(); owned.forEach(b => b.destroy()); }
    });
    await t.test("APIC timestep includes affine velocity even when particle centres are stationary", async () => {
      const scene = structuredClone(base); scene.fluid.gravity_m_s2.y = 0; scene.numerics.fixedDt_s = 1 / 30;
      const solver = await create(scene);
      try {
        await solver.awaitFrameCompletion();
        const buffer = solver.debug!.apicParticles as GPUBuffer, particles = await readBuffer(buffer);
        for (let i = 0; i < particles.length; i += 20) particles[i + 8] = 40;
        device!.queue.writeBuffer(buffer, 0, particles);
        const encoder = device!.createCommandEncoder();
        (solver as unknown as { transport: ApicTransport }).transport.encodeStats(encoder);
        device!.queue.submit([encoder.finish()]);
        const before = await solver.readStats() as ApicInfo;
        assert.ok(Math.abs(before.apicAffineSpeedBound_m_s! - 3.75) < 1e-6);
        assert.equal(before.maxSpeed_m_s, 0, "the bound is separate from actual centre/grid speed");
        solver.advanceTo(1 / 30, []); await solver.awaitFrameCompletion();
        assert.ok((solver.info.lastSubsteps ?? 0) >= 2, "affine motion must constrain the first timestep");
        assert.ok(solver.info.pressureSolveConverged);
      } finally { solver.destroy(); }
    });
    await t.test("hydrostatic pool stays quiet and conserves particle material", async () => {
      const solver = await create();
      try {
        for (let frame = 1; frame <= 12; frame++) { solver.advanceTo(frame / 120, []); await solver.awaitFrameCompletion(); }
        const info = await solver.readStats() as ApicInfo;
        assert.ok(info.apicParticleCount! > 1000); assert.ok(Math.abs(info.apicMaterialDrift!) < 1e-5);
        assert.ok(info.pressureSolveConverged); assert.ok(info.pressureResidual! <= 0.001);
        assert.ok(info.maxSpeed_m_s! < 0.005, `rest speed ${info.maxSpeed_m_s}`);
        assert.ok(Math.abs(info.volumeDrift!) < 0.005, `surface drift ${info.volumeDrift}`);
        const phi = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 17, 17, 17);
        assert.ok(phi.some(v => v < 0) && phi.some(v => v > 0)); assert.ok(phi.every(Number.isFinite));
      } finally { solver.destroy(); }
    });
    await t.test("free particles accelerate under gravity without losing their material", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }];
      const solver = await create(scene);
      try {
        for (let frame = 1; frame <= 6; frame++) { solver.advanceTo(frame / 120, []); await solver.awaitFrameCompletion(); }
        const particles = await readBuffer(solver.debug!.apicParticles as GPUBuffer);
        let velocity = 0, count = 0; for (let i = 0; i < particles.length; i += 20) if (particles[i + 3] > 0) { velocity += particles[i + 5]; count++; }
        assert.ok(Math.abs(velocity / count + 9.81 * 0.05) < 0.01, `drop speed ${velocity / count}`);
        assert.ok(Math.abs((solver.info as ApicInfo).apicMaterialDrift!) < 1e-5);
      } finally { solver.destroy(); }
    });
    await t.test("authored dam runs through impact with finite particles and an accepted surface", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      const solver = await create(scene), start = performance.now();
      try {
        let steps = 0;
        while ((solver.info.submittedTime_s ?? 0) < 0.5 - 1e-10) {
          assert.ok(++steps < 1000); solver.advanceTo(0.5, []); await solver.awaitFrameCompletion();
        }
        const info = await solver.readStats() as ApicInfo;
        assert.ok(info.pressureSolveConverged); assert.ok(info.maxSpeed_m_s! > 0.1);
        assert.ok(Math.abs(info.apicMaterialDrift!) < 1e-5); assert.ok(info.maxDivergenceAfter_s! < 0.003);
        const particles = await readBuffer(solver.debug!.apicParticles as GPUBuffer); assert.ok(particles.every(Number.isFinite));
        // The bin scan crosses all three hierarchy levels at 32³. Validate
        // the complete permutation and ownership, not only its rendered phi.
        const countsRaw = await readBuffer(solver.debug!.apicBinCounts as GPUBuffer);
        const offsetsRaw = await readBuffer(solver.debug!.apicBinOffsets as GPUBuffer);
        const counts = new Uint32Array(countsRaw.buffer), offsets = new Uint32Array(offsetsRaw.buffer);
        const sorted = await readBuffer(solver.debug!.apicSortedParticles as GPUBuffer);
        let prefix = 0, energy = 0, volume = 0, affineBound = 0;
        const h = 0.8 / 32;
        for (let cell = 0; cell < counts.length; cell++) {
          assert.equal(offsets[cell], prefix, `exclusive scan at bin ${cell}`);
          for (let j = prefix; j < prefix + counts[cell]; j++) {
            const bin = [cell % 32, Math.floor(cell / 32) % 32, Math.floor(cell / 1024)];
            for (let axis = 0; axis < 3; axis++) {
              const coordinate = sorted[j * 20 + axis] / Math.fround(h);
              // GPU division may use a rounded reciprocal. At an exact cell
              // face either neighboring bin is valid within a few f32 ulps;
              // test geometric ownership instead of JS/GPU bit identity.
              const tolerance = Math.max(1, Math.abs(coordinate)) * 2 ** -22;
              assert.ok(coordinate >= bin[axis] - tolerance && coordinate <= bin[axis] + 1 + tolerance,
                `particle coordinate ${coordinate} outside bin ${bin[axis]} on axis ${axis}`);
            }
          }
          prefix += counts[cell];
        }
        assert.equal(prefix, info.apicParticleCount);
        const sourceWords = new Uint32Array(particles.buffer), sortedWords = new Uint32Array(sorted.buffer);
        const records = new Map<string, number>();
        for (let i = 0; i < particles.length; i += 20) if (particles[i + 3] > 0) {
          const key = sourceWords.subarray(i, i + 20).join(","); records.set(key, (records.get(key) ?? 0) + 1);
        }
        for (let i = 0; i < prefix; i++) {
          const key = sortedWords.subarray(i * 20, (i + 1) * 20).join(","), remaining = records.get(key) ?? 0;
          assert.ok(remaining > 0, "scatter must preserve every complete particle record exactly once");
          if (remaining === 1) records.delete(key); else records.set(key, remaining - 1);
        }
        assert.equal(records.size, 0, "scatter omitted no particle records");
        for (let i = 0; i < particles.length; i += 20) if (particles[i + 3] > 0) {
          const speed2 = particles[i + 4] ** 2 + particles[i + 5] ** 2 + particles[i + 6] ** 2;
          volume += particles[i + 3]; energy += 0.5 * scene.fluid.density_kg_m3 * particles[i + 3] * speed2;
          const bound = [0, 1, 2].map(a => Math.abs(particles[i + 4 + a]) + 1.5 * h *
            (Math.abs(particles[i + 8 + 4 * a]) + Math.abs(particles[i + 9 + 4 * a]) + Math.abs(particles[i + 10 + 4 * a])));
          affineBound = Math.max(affineBound, Math.hypot(...bound));
        }
        assert.ok(Math.abs(info.apicMaterialVolume_m3! - volume) < 1e-6 * volume);
        assert.ok(Math.abs(info.apicKineticEnergy_J! - energy) < 1e-5 * Math.max(1, energy));
        assert.ok(Math.abs(info.apicAffineSpeedBound_m_s! - affineBound) < 1e-5 * Math.max(1, affineBound));
        t.diagnostic(`APIC dam: ${steps} advances, ${((performance.now() - start) / steps).toFixed(2)} ms/advance including test fence, ${info.apicParticleCount} particles, ${info.allocatedBytes} bytes`);
      } finally { solver.destroy(); }
    });
    await t.test("an empty scene stays empty without invalid particle dispatches", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      const solver = await create(scene);
      try {
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        const info = await solver.readStats() as ApicInfo;
        assert.equal(info.apicParticleCount, 0); assert.equal(info.apicMaterialVolume_m3, 0);
        assert.equal(info.apicMaterialDrift, 0); assert.equal(info.volumeCellSum, 0);
      } finally { solver.destroy(); }
    });
    await t.test("live timestep edits retain particle allocation, material and the running clock", async () => {
      const scene = structuredClone(base); scene.fluid.gravity_m_s2.y = 0;
      const solver = await create(scene);
      try {
        const particles = solver.debug!.apicParticles;
        let target = 0;
        for (const dt of [0.004, 1 / 60, 0.033, 0.05, 0.008]) {
          scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
          solver.applySceneUniforms!(structuredClone(scene));
          target += dt;
          assert.equal(solver.advanceTo(target, []), true); await solver.awaitFrameCompletion();
          assert.ok(Math.abs(solver.info.completedTime_s! - target) < 1e-8);
          assert.ok(Math.abs(solver.info.lastDt_s! - dt) < 1e-7, `actual dt ${solver.info.lastDt_s} vs requested ${dt}`);
          assert.equal(solver.debug!.apicParticles, particles);
          assert.ok(Math.abs((solver.info as ApicInfo).apicMaterialDrift!) < 1e-5);
        }
      } finally { solver.destroy(); }
    });
    await t.test("direct pressure dispatch matches the indirect pressure result", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      const surfaces: Float32Array[] = [];
      for (const directPressure of [false, true]) {
        const solver = await MacGridSolver.createAsync(device!, scene, "balanced", strictPressure, undefined, undefined, {
          ...APIC_CONFIGURATION, directPressure, multigridPressure: false, advanceStepCapacity: 1,
          resolveOptions: (s, v) => ({ ...APIC_CONFIGURATION.resolveOptions!(s, v), cfl: 0.5, pressureLimit: 256 }),
        });
        try {
          for (let step = 1; step <= 6; step++) {
            while (solver.info.submittedTime_s! < step / 120 - 1e-9) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
          }
          assert.ok(solver.info.pressureSolveConverged); assert.ok(solver.info.maxDivergenceAfter_s! < 0.003);
          assert.ok(Math.abs((solver.info as ApicInfo).apicMaterialDrift!) < 1e-5);
          surfaces.push(await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 33, 33, 33));
        } finally { solver.destroy(); }
      }
      let maximum = 0; for (let i = 0; i < surfaces[0].length; i++) maximum = Math.max(maximum, Math.abs(surfaces[0][i] - surfaces[1][i]));
      assert.ok(maximum < 1e-5, `dispatch surface difference ${maximum} m`);
    });
    await t.test("direct dispatch cannot publish a frame that exhausts its substep capacity", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }];
      scene.fluid.initialVelocity_m_s = { x: 20, y: 0, z: 0 };
      const solver = await MacGridSolver.createAsync(device!, scene, "balanced", strictPressure, undefined, undefined, {
        ...APIC_CONFIGURATION, directPressure: true, multigridPressure: false, advanceStepCapacity: 1,
        resolveOptions: (s, v) => ({ ...APIC_CONFIGURATION.resolveOptions!(s, v), cfl: 0.5, pressureLimit: 256 }),
      });
      try {
        await solver.awaitFrameCompletion();
        const before = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 17, 17, 17);
        (solver as unknown as { capacitySpeed: number }).capacitySpeed = 0;
        solver.advanceTo(1 / 120, []);
        await assert.rejects(solver.awaitFrameCompletion(), /substep capacity exhausted/);
        assert.deepEqual(await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 17, 17, 17), before);
        assert.equal(solver.info.completedTime_s, 0);
      } finally { solver.destroy(); }
    });
    await t.test("multigrid-preconditioned pressure agrees with diagonal PCG at the same timestep", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      const surfaces: Float32Array[] = [];
      for (const multigridPressure of [false, true]) {
        const solver = await MacGridSolver.createAsync(device!, scene, "balanced", strictPressure, undefined, undefined, {
          ...APIC_CONFIGURATION, multigridPressure, advanceStepCapacity: 1,
          resolveOptions: (s, v) => ({ ...APIC_CONFIGURATION.resolveOptions!(s, v), cfl: 0.5, pressureLimit: multigridPressure ? 16 : 256 }),
        });
        try {
          for (let step = 1; step <= 6; step++) {
            while (solver.info.submittedTime_s! < step / 120 - 1e-9) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
          }
          assert.ok(solver.info.pressureSolveConverged); assert.ok(solver.info.pressureResidual! <= 0.001);
          assert.ok(solver.info.maxDivergenceAfter_s! < 0.003);
          if (multigridPressure) assert.ok(solver.info.pressureIterationsExecuted! <= 16);
          surfaces.push(await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 33, 33, 33));
        } finally { solver.destroy(); }
      }
      let maximum = 0; for (let i = 0; i < surfaces[0].length; i++) maximum = Math.max(maximum, Math.abs(surfaces[0][i] - surfaces[1][i]));
      assert.ok(maximum < 1e-4, `preconditioner surface difference ${maximum} m`);
    });
    await t.test("multigrid advances larger particle steps through dam impact without relaxing incompressibility", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      scene.numerics.fixedDt_s = 1 / 30;
      const solver = await create(scene);
      try {
        let advances = 0, largestDt = 0;
        while (solver.info.submittedTime_s! < 0.5 - 1e-9) {
          assert.ok(++advances <= 20, "large-step scheduler must not collapse to millisecond host advances");
          solver.advanceTo(0.5, []); await solver.awaitFrameCompletion();
          largestDt = Math.max(largestDt, solver.info.lastDt_s!);
          assert.ok(solver.info.pressureResidual! <= 0.001); assert.ok(solver.info.maxDivergenceAfter_s! < 0.003);
        }
        assert.ok(largestDt > 0.016); assert.ok(Math.abs((solver.info as ApicInfo).apicMaterialDrift!) < 1e-5);
        assert.ok((await readBuffer(solver.debug!.apicParticles as GPUBuffer)).every(Number.isFinite));
      } finally { solver.destroy(); }
    });
    await t.test("APIC pressure warm start survives a short remainder followed by a large step", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      scene.numerics.fixedDt_s = 1 / 30;
      const solver = await create(scene);
      try {
        for (const target of [1 / 30, 1 / 30 + 1e-6, 2 / 30, 0.1]) {
          while (solver.info.submittedTime_s! < target - 1e-9) {
            solver.advanceTo(target, []); await solver.awaitFrameCompletion();
            assert.ok(solver.info.pressureResidual! <= 0.001);
            assert.ok(solver.info.maxDivergenceAfter_s! < 0.003);
          }
        }
        assert.ok(Math.abs(solver.info.completedTime_s! - 0.1) < 1e-8);
      } finally { solver.destroy(); }
    });
    await t.test("64³ APIC strict reference converges through two seconds at a requested 33.3 ms", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
      scene.numerics.fixedDt_s = 1 / 30;
      const solver = await create(scene);
      try {
        let advances = 0, maximumIterations = 0;
        for (let frame = 1; frame <= 60; frame++) {
          const target = frame / 30;
          while (solver.info.submittedTime_s! < target - 1e-9) {
            // Large grids use smaller host advances to bound Chrome GPU work.
            // Check actual progress, independently of that scheduling choice.
            const previousTime = solver.info.submittedTime_s!;
            advances++;
            solver.advanceTo(target, []);
            try { await solver.awaitFrameCompletion(); }
            catch (error) {
              const resources = (solver as unknown as { resources: Map<number, GPUBufferBinding> }).resources;
              const [matrix, cg, rhs, scalars] = await Promise.all([9, 10, 19, 12].map(key => readBuffer(resources.get(key)!.buffer)));
              let largestTerm = 0;
              for (let i = 0; i < rhs.length; i++) largestTerm = Math.max(largestTerm, Math.abs(matrix[4 * i + 3] * cg[4 * i]));
              t.diagnostic(JSON.stringify({ target, completedTime_s: solver.info.completedTime_s,
                residual: solver.info.pressureResidual, iterations: solver.info.pressureIterationsExecuted,
                maxSpeed_m_s: solver.info.maxSpeed_m_s, previousDt: scalars[15], currentDt: scalars[16], largestTerm,
                nonfiniteInputs: [matrix, cg, rhs].map(values => values.filter(value => !Number.isFinite(value)).length) }));
              throw error;
            }
            assert.ok(solver.info.pressureSolveConverged);
            assert.ok(solver.info.completedTime_s! > previousTime);
            assert.ok(solver.info.pressureResidual! <= 0.001, `residual at ${target}: ${solver.info.pressureResidual}`);
            assert.ok(solver.info.maxDivergenceAfter_s! < 0.003);
            maximumIterations = Math.max(maximumIterations, solver.info.pressureIterationsExecuted!);
          }
        }
        assert.ok(Math.abs(solver.info.completedTime_s! - 2) < 1e-8);
        assert.ok((await readBuffer(solver.debug!.apicParticles as GPUBuffer)).every(Number.isFinite));
        t.diagnostic(`64³: ${advances} advances, maximum final-substep iterations ${maximumIterations}, residual ${solver.info.pressureResidual}`);
      } finally { solver.destroy(); }
    });
    await t.test("an exhausted multigrid pressure budget cannot publish an unconverged frame", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      const solver = await apicMethod.createSolverAsync!(device!, scene, "balanced", { ...strictPressure, pressureLimit: 1 }, undefined, () => {}) as MacGridSolver;
      try {
        await solver.awaitFrameCompletion();
        const before = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 33, 33, 33);
        solver.advanceTo(1 / 30, []);
        await assert.rejects(solver.awaitFrameCompletion(), /pressure did not converge/);
        assert.equal(solver.info.completedTime_s, 0);
        assert.deepEqual(await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 33, 33, 33), before);
      } finally { solver.destroy(); }
    });
    await t.test("open-top outflow is counted instead of silently losing material", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0; scene.container.top = "open";
      scene.fluid.gravity_m_s2.y = 0; scene.fluid.initialVelocity_m_s = { x: 0, y: 1, z: 0 };
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.93, z: 0 }, radius_m: 0.12 }];
      const solver = await create(scene);
      try {
        for (let step = 1; step <= 24; step++) { solver.advanceTo(step / 120, []); await solver.awaitFrameCompletion(); }
        const info = await solver.readStats() as ApicInfo;
        assert.ok(info.apicEscapedVolume_m3! > 0);
        assert.ok(info.apicMaterialVolume_m3! < info.apicInitialVolume_m3!);
        assert.ok(Math.abs(info.apicMaterialDrift!) < 1e-5);
      } finally { solver.destroy(); }
    });
    await t.test("the shared water renderer consumes the reconstructed particle surface", async () => {
      const solver = await create();
      try {
        solver.advanceTo(1 / 120, []); await solver.awaitFrameCompletion();
        const rendered = await smokeRenderHybridPresentation(device!, solver, base, []);
        assert.ok(rendered.frontInterfacePixels > 0); assert.ok(rendered.pairedInterfacePixels > 0);
        assert.equal(rendered.rendererValidationErrorCount, 0); assert.equal(rendered.rendererUncapturedErrorCount, 0);
      } finally { solver.destroy(); }
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
