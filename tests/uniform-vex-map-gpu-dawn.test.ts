import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { GPUUniformVexMapCompiler, UNIFORM_VEX_MAP_FAULT } from "../tools/implicit-density/uniform-vex-map-gpu";
import { inspectUniformVexFixture, uniformVexCorruptions, uniformVexFixture, uploadUniformVexFixture } from "./helpers/uniform-vex-map-fixture";

const modulePath = process.env.WEBGPU_NODE_MODULE, live = new Set<GPU>();
(modulePath ? test : test.skip)("immutable actual-native-format uniform VEX GPU map compiler", { timeout: 30_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform-native-vex-map-compiler");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const compiler = await GPUUniformVexMapCompiler.create(device);
    for (const velocity of [[15, -2.5, 4], [0, 0, 0], [160, -16, 0]]) await t.test(`uniform [${velocity}] maps physical units and preserves invalid coverage`, async () => {
      const fixture = uniformVexFixture(velocity), expected = inspectUniformVexFixture(fixture);
      const uploaded = uploadUniformVexFixture(device!, fixture);
      const encoder = device!.createCommandEncoder(), attempt = compiler.encodeSnapshotAndCompile(encoder, uploaded.source, fixture.dimensions);
      // A following native stage can overwrite live source data. The map and
      // its coverage must remain tied to the earlier immutable snapshot.
      encoder.clearBuffer(uploaded.source.effectiveTransportVelocity);
      encoder.clearBuffer(uploaded.source.activity);
      encoder.clearBuffer(uploaded.source.topologyArena);
      device!.queue.submit([encoder.finish()]);
      try {
        const receipt = await attempt.readReceiptForQA();
        assert.equal(receipt.frameGeneration, 17); assert.equal(receipt.sealedCandidateGeneration, 18); assert.equal(receipt.topologyGeneration, 9);
        assert.equal(receipt.scalarParity, 1); assert.equal(receipt.faceParity, 1);
        assert.equal(receipt.validNativeCells, expected.valid); assert.equal(receipt.acceptedNativeCells, expected.accepted);
        assert.equal(receipt.anchorNativeCell, expected.anchor);
        assert.deepEqual(receipt.map.matrix, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
        for (let axis = 0; axis < 3; axis++) {
          assert.ok(Math.abs(receipt.velocity_m_s[axis]! - expected.physicalVelocity[axis]!) <= 2e-7);
          assert.ok(Math.abs(receipt.map.translation[axis]! - expected.translation[axis]!) <= 3e-7);
        }
        assert.equal(receipt.timeStep_s, expected.dt); assert.equal(receipt.finestCellSize_m, expected.h);
        assert.deepEqual(await attempt.readCoverageForQA(), expected.coverage);
        console.log(JSON.stringify({ fixture: "uniform-native-vex-map", ...receipt }));
      } finally { attempt.destroy(); uploaded.destroy(); }
    });
    for (const [name, fault, corrupt] of uniformVexCorruptions) await t.test(name, async () => {
      const fixture = uniformVexFixture(); corrupt(fixture); const uploaded = uploadUniformVexFixture(device!, fixture);
      const encoder = device!.createCommandEncoder(), attempt = compiler.encodeSnapshotAndCompile(encoder, uploaded.source, fixture.dimensions);
      device!.queue.submit([encoder.finish()]);
      try {
        await assert.rejects(attempt.readReceiptForQA(), /not admitted/);
        const raw = await attempt.readRawReceiptForQA();
        assert.equal(raw[2], 2); assert.ok((raw[3]! & UNIFORM_VEX_MAP_FAULT[fault as keyof typeof UNIFORM_VEX_MAP_FAULT]) !== 0,
          `${name}: wrong failure bits ${raw[3]}`);
        assert.ok(Array.from(raw.subarray(16, 28)).every(word => word === 0), "rejected snapshot cannot publish an affine map");
      } finally { attempt.destroy(); uploaded.destroy(); }
    });
    await t.test("discarded encoding does not produce an accepted zero map", async () => {
      const fixture = uniformVexFixture(), uploaded = uploadUniformVexFixture(device!, fixture);
      const attempt = compiler.encodeSnapshotAndCompile(device!.createCommandEncoder(), uploaded.source, fixture.dimensions);
      try { await assert.rejects(attempt.readReceiptForQA(), /not admitted/); }
      finally { attempt.destroy(); uploaded.destroy(); }
    });
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
  } finally { device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock(); }
});
