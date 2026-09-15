import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { LEVELSET_VOLUME_SUPPORT } from
  "../lib/methods/adaptive-volume/levelset-volume-layout";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";
import { radialFrontsFromFields } from "../tools/sparse-geometric-radial-front";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function scalarD4Maximum(field: ArrayLike<number>, nx: number, ny: number, nz: number) {
  let maximum = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const source = Number(field[x + nx * (y + ny * z)]);
    for (const [tx, tz] of [[nx - 1 - x, z], [x, nz - 1 - z], [z, x]]) {
      const target = Number(field[tx! + nx * (y + ny * tz!)]);
      if (Number.isFinite(source) !== Number.isFinite(target)) return Infinity;
      if (Number.isFinite(source)) maximum = Math.max(maximum, Math.abs(source - target));
    }
  }
  return maximum;
}

function vertexAuthority(receipt: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readAdaptiveLevelSetQA"]>>) {
  return (receipt.vertices ?? []).map(vertex => ({ positionFine: vertex.positionFine,
    phiFine: vertex.phiFine, support: vertex.support }))
    .sort((left, right) => left.positionFine[2] - right.positionFine[2]
      || left.positionFine[1] - right.positionFine[1]
      || left.positionFine[0] - right.positionFine[0]);
}

function assertValidSupport(vertices: ReturnType<typeof vertexAuthority>) {
  for (const vertex of vertices) {
    assert.notEqual(vertex.support, LEVELSET_VOLUME_SUPPORT.absent,
      `absent support at ${vertex.positionFine.join(",")}`);
    if (Math.abs(vertex.phiFine) <= 4) assert.equal(vertex.support,
      LEVELSET_VOLUME_SUPPORT.metric,
      `near-interface phi ${vertex.phiFine} has support ${vertex.support} at ${
        vertex.positionFine.join(",")}`);
    if (vertex.support === LEVELSET_VOLUME_SUPPORT.deepAir) assert.ok(vertex.phiFine > 0,
      `deep-air phi ${vertex.phiFine} is not positive at ${vertex.positionFine.join(",")}`);
    if (vertex.support === LEVELSET_VOLUME_SUPPORT.deepLiquid) assert.ok(vertex.phiFine < 0,
      `deep-liquid phi ${vertex.phiFine} is not negative at ${vertex.positionFine.join(",")}`);
  }
}

function projectedPhaseD4Mismatch(phi: ArrayLike<number>, nx: number, ny: number, nz: number) {
  const occupied = new Uint8Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) if (Number(phi[x + nx * (y + ny * z)]) <= 0) {
      occupied[x + nx * z] = 1; break;
    }
  }
  let mismatches = 0;
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const value = occupied[x + nx * z]!;
    if (occupied[nx - 1 - x + nx * z] !== value) mismatches++;
    if (occupied[x + nx * (nz - 1 - z)] !== value) mismatches++;
    if (occupied[z + nx * x] !== value) mismatches++;
  }
  return mismatches;
}

dawnTest("zero-motion symmetric box preserves its geometric surface exactly",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "sparse-geometric zero-motion surface");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    const validationErrors: string[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn, [
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache",
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); validationErrors.push(event.error.message);
      });
      const scene = createSymmetricExpansionScene();
      scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
      solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
        device, scene, "balanced", undefined, sparseCM12DawnDefaultOptions(), () => {});
      await solver.waitForSimulationReady();
      const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
      const initialPublished = await readPublishedCM12Field(device, solver);
      const initialPhi = await solver.readAdaptiveLevelSetQA(true);
      const initialVolume = await solver.readAcceptedGeometricVolumeQA();
      assert.equal(initialVolume.volumeFine3, 2048);
      assert.equal(scalarD4Maximum(initialPublished.values, ...dimensions), 0);

      while (!solver.advanceTo(CM12_PAPER_DT_S, [])) await new Promise<void>(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      const [published, phi, volume, fields, transport] = await Promise.all([
        readPublishedCM12Field(device, solver), solver.readAdaptiveLevelSetQA(true),
        solver.readAcceptedGeometricVolumeQA(), solver.readDiagnosticFields(true),
        solver.readGeometricVolumeTransportReceiptQA(),
      ]);

      const initialVertices = vertexAuthority(initialPhi);
      const vertices = vertexAuthority(phi);
      assert.equal(vertices.length, initialVertices.length,
        `zero motion changed vertex count ${initialVertices.length} -> ${vertices.length}`);
      for (let vertex = 0; vertex < vertices.length; vertex++) {
        const before = initialVertices[vertex]!, after = vertices[vertex]!;
        assert.deepEqual(after.positionFine, before.positionFine,
          `zero motion changed vertex ${vertex} position`);
        assert.equal(after.phiFine, before.phiFine,
          `zero motion changed phi at ${after.positionFine.join(",")}: ${before.phiFine} -> ${
            after.phiFine}`);
      }
      assertValidSupport(vertices);
      assert.deepEqual(published.values, initialPublished.values,
        "zero motion changed the packed published surface");
      assert.deepEqual(published.floorContinuation, initialPublished.floorContinuation);
      assert.equal(scalarD4Maximum(published.values, ...dimensions), 0);
      assert.ok(Math.abs(volume.volumeFine3 - initialVolume.volumeFine3) <= 2 ** -18,
        `zero motion changed accepted volume: ${volume.volumeFine3}/${initialVolume.volumeFine3}`);
      assert.equal(transport.fault, 0); assert.equal(transport.transportCompleted, true);
      assert.equal(transport.coupling.maximumTraceDisplacementFine, 0);
      let maximumVelocity_m_s = 0;
      for (let cell = 0; cell < fields.density.length; cell++) maximumVelocity_m_s = Math.max(
        maximumVelocity_m_s, Math.hypot(fields.velocity[4 * cell]!,
          fields.velocity[4 * cell + 1]!, fields.velocity[4 * cell + 2]!));
      assert.ok(maximumVelocity_m_s <= 1e-7,
        `zero-motion velocity is ${maximumVelocity_m_s} m/s`);
      assert.deepEqual(validationErrors, []);
    } finally {
      try { solver?.destroy(); device?.destroy(); } finally { await releaseWebGPUExclusiveLock(); }
    }
  });

dawnTest("moving symmetric box evolves a conserved D4 surface for three production steps",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "sparse-geometric moving surface");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    const validationErrors: string[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn, [
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache",
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); validationErrors.push(event.error.message);
      });
      const scene = createSymmetricExpansionScene();
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
      solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
        device, scene, "balanced", undefined,
        { ...sparseCM12DawnDefaultOptions(), pressureRelativeTolerance: 0 }, () => {});
      await solver.waitForSimulationReady();
      const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
      const initialVolume = (await solver.readAcceptedGeometricVolumeQA()).volumeFine3;
      assert.equal(initialVolume, 2048);

      for (let step = 1; step <= 3; step++) {
        while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise<void>(setImmediate);
        await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
        const published = await readPublishedCM12Field(device, solver);
        const fields = await solver.readDiagnosticFields(true);
        const volume = await solver.readAcceptedGeometricVolumeQA();
        const phiReceipt: Awaited<ReturnType<
          WebGPUAdaptiveMassSolver["readAdaptiveLevelSetQA"]>> =
          await solver.readAdaptiveLevelSetQA(false);
        const transport: Awaited<ReturnType<
          WebGPUAdaptiveMassSolver["readGeometricVolumeTransportReceiptQA"]>> =
          await solver.readGeometricVolumeTransportReceiptQA();
        assert.equal(phiReceipt.fault, 0, `step ${step}: adaptive phi fault`);
        assert.equal(transport.fault, 0, `step ${step}: transport fault`);
        assert.equal(transport.transportCompleted, true, `step ${step}: incomplete transport`);
        assert.ok(Math.abs(volume.volumeFine3 - initialVolume) / initialVolume <= 1e-6,
          `step ${step}: accepted-volume drift ${volume.volumeFine3}/${initialVolume}`);
        assert.equal(projectedPhaseD4Mismatch(published.values, ...dimensions), 0,
          `step ${step}: rendered phase footprint is not D4 symmetric`);
        const radial = radialFrontsFromFields(published.values, fields.density, dimensions);
        assert.ok(radial.phiProjected.d4MaximumError_cells <= 0.125,
          `step ${step}: rendered radial D4 error ${radial.phiProjected.d4MaximumError_cells}`);
        assert.ok(radial.phiProjected.roughnessAfterLowModes.rms_cells !== null
          && radial.phiProjected.roughnessAfterLowModes.rms_cells <= 0.3,
        `step ${step}: rendered radial roughness ${
          radial.phiProjected.roughnessAfterLowModes.rms_cells}`);
        assert.ok(transport.coupling.maximumTraceDisplacementFine >= 0.2,
          `step ${step}: projected motion did not reach transport`);
      }
      assert.deepEqual(validationErrors, []);
    } finally {
      try { solver?.destroy(); device?.destroy(); } finally { await releaseWebGPUExclusiveLock(); }
    }
  });
