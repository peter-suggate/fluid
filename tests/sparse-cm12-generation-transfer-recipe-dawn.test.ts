import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createSolidWorld } from "../lib/core/solid-world";
import { realizeCM12ResourceRecipe } from "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import { PreparedSparseCM12GenerationTransfer } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { referenceGenerationTransfer, transferFixtures, transferSourceRecipe,
  transferSourceValues } from "./helpers/cm12-generation-transfer-oracle";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const liveDawn = new Set<GPU>();
const fixtures = transferFixtures().filter(fixture =>
  fixture.name === "clipped macro split" || fixture.name === "signed dynamic source page");

for (const fixture of fixtures) dawnTest(`${fixture.name}: recorded worker transfer preserves live banks on a real device`,
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", `cm12-recorded-transfer-${fixture.name}`);
    let device: GPUDevice | undefined, gpu: GPU | undefined;
    let realized: Awaited<ReturnType<typeof realizeCM12ResourceRecipe>> | undefined;
    let transfer: PreparedSparseCM12GenerationTransfer | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); liveDawn.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      device.pushErrorScope("validation");
      const before = transferSourceRecipe(fixture.source, fixture.dynamicPage);
      const sourceValues = transferSourceValues(fixture.source, before.layout, before.physicalCells, before.physicalRows);
      const make = (label: string, values: Uint32Array | Float32Array) => {
        const buffer = device!.createBuffer({ label, size: values.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        device!.queue.writeBuffer(buffer, 0, values.buffer as ArrayBuffer, values.byteOffset, values.byteLength);
        buffers.push(buffer); return buffer;
      };
      const sourceState = make("live source banks", sourceValues);
      const sourceTopology = make("live source native geometry and parity", before.topology);
      const { templateWords: _hostShadow, ...geometryRecipe } = before.recipe;
      const recipe = structuredClone(await WebGPUSparseCM12Resident.recordPreparedGeneration({
        atlas: fixture.target.atlas, active: new Set(fixture.target.atlas.bricks.map(brick => brick.key)),
        finestCellSize_m: .05, solidWorld: createSolidWorld(), maximumBytes: 256 * 1024 * 1024,
        topologyPageCapacityMaximum: 0, symmetry: { scalar: false, face: false }, limits: device.limits,
        source: { geometryRecipe, cellIds: before.physicalCells, rowIds: before.physicalRows,
          densityOffset: before.layout.densityOffset, gammaOffset: before.layout.gammaOffset,
          velocityOffset: before.layout.velocityOffset, pressureOffset: before.layout.pressureOffset,
          faceOffset: before.layout.faceOffset,
          stateDescriptor: { size: sourceState.size, usage: sourceState.usage },
          controlDescriptor: { size: sourceTopology.size, usage: sourceTopology.usage },
          liveControl: { scalarParityWord: before.scalarParityWord, faceParityWord: before.faceParityWord,
            densityOffsets: [before.layout.densityOffset, before.layout.densityOtherOffset],
            gammaOffsets: [before.layout.gammaOffset, before.layout.gammaOtherOffset],
            velocityOffsets: [before.layout.velocityOffset, before.layout.velocityOtherOffset],
            faceOffsets: [before.layout.faceOffset, before.layout.faceOtherOffset] } },
      }));
      realized = await realizeCM12ResourceRecipe(device, recipe, [sourceState, sourceTopology]);
      const data = realized.state as { resident: { state: GPUBuffer; layout: Record<string, number>;
        initialGenerationCellIds: Uint32Array; initialGenerationRowIds: Uint32Array }; transfer: object };
      transfer = Object.assign(Object.create(PreparedSparseCM12GenerationTransfer.prototype), data.transfer, { device });
      const target = data.resident, layout = target.layout;
      assert.equal(target.initialGenerationCellIds.length, fixture.target.cells.length);
      assert.equal(target.initialGenerationRowIds.length, fixture.target.gradientRows.length);
      const readback = device.createBuffer({ size: target.state.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      buffers.push(readback);
      for (const [scalar, face] of [[1, 0], [0, 1]] as const) {
        // These controls change after CPU recording and GPU recipe hydration.
        device.queue.writeBuffer(sourceTopology, 4 * before.scalarParityWord, new Uint32Array([scalar, face]));
        const encoder = device.createCommandEncoder(); transfer!.encode(encoder);
        encoder.copyBufferToBuffer(target.state, 0, readback, 0, target.state.size);
        device.queue.submit([encoder.finish()]); await transfer!.validate();
        await readback.mapAsync(GPUMapMode.READ);
        const actual = new Float32Array(readback.getMappedRange());
        const expected = referenceGenerationTransfer(fixture.source, fixture.target, sourceValues,
          before.layout, before.physicalCells, before.physicalRows, scalar, face, fixture.air);
        let maximumError = 0;
        const close = (at: number, value: number) => {
          const error = Math.abs(actual[at]! - value); maximumError = Math.max(maximumError, error);
          assert.ok(Number.isFinite(actual[at]) && error <= 2e-6 * Math.max(1, Math.abs(value)),
            `recorded ${fixture.name} bank${scalar}/${face} word${at}: ${actual[at]} != ${value}`);
        };
        for (const cell of fixture.target.cells) {
          const id = target.initialGenerationCellIds[cell.id]!, value = expected.cells[cell.id]!;
          for (const key of ["densityA", "densityB"]) close(layout[key]! + id, value.density);
          for (const key of ["gammaA", "gammaB"]) close(layout[key]! + id, value.gamma);
          close(layout.pressure! + id, value.pressure);
          for (const key of ["cellVelocityA", "cellVelocityB"]) for (let axis = 0; axis < 3; axis++)
            close(layout[key]! + 4 * id + axis, value.velocity[axis]!);
        }
        for (const row of fixture.target.gradientRows) for (const key of ["faceA", "faceB"])
          close(layout[key]! + target.initialGenerationRowIds[row.id]!, expected.faces[row.id]!);
        readback.unmap();
        console.log(JSON.stringify({ fixture: fixture.name, path: "recorded-worker-recipe", scalar, face,
          checkedCells: fixture.target.cells.length, checkedFaces: fixture.target.gradientRows.length, maximumError }));
      }
      assert.equal(await device.popErrorScope(), null);
    } finally {
      transfer?.destroy(); realized?.destroy();
      for (const buffer of buffers) { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
      device?.destroy(); if (gpu) liveDawn.delete(gpu); await releaseWebGPUExclusiveLock();
    }
  });
