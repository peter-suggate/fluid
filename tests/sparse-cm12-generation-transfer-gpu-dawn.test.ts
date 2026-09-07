import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { SparseCM12GenerationBudgetDeferred } from "../lib/methods/adaptive-mass/sparse-cm12-generation-budget";
import { prepareSparseCM12GPUGenerationTransfer } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer-gpu";
import type { PreparedSparseCM12GenerationTransfer } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";
import { transferFixtures, transferSourceRecipe, transferSourceValues, transferLayout,
  packedTransferGrid, referenceGenerationTransfer } from "./helpers/cm12-generation-transfer-oracle";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const liveDawn = new Set<GPU>();
const fixtures = transferFixtures();

async function withDevice(name: string, run: (device: GPUDevice) => Promise<void>) {
  await acquireWebGPUExclusiveLock("dawn-test", `gpu-generation-transfer-${name}`);
  let device: GPUDevice | undefined, gpu: GPU | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); liveDawn.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    await run(device);
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(errors, [], "no WebGPU validation failure may masquerade as a transfer receipt");
  } finally { device?.destroy(); if (gpu) liveDawn.delete(gpu); await releaseWebGPUExclusiveLock(); }
}

function resources(device: GPUDevice, fixture: (typeof fixtures)[number]) {
  const buffers: GPUBuffer[] = [];
  const make = (name: string, values: ArrayBufferView<ArrayBuffer>) => {
    const result = device.createBuffer({ label: name, size: values.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(result, 0, values); buffers.push(result); return result;
  };
  const before = transferSourceRecipe(fixture.source, fixture.dynamicPage);
  const packed = packedTransferGrid(fixture.target);
  const layout = transferLayout(fixture.target.cells.length, fixture.target.gradientRows.length);
  const data = transferSourceValues(fixture.source, before.layout, before.physicalCells, before.physicalRows);
  const topology = make("source accepted topology and live parity", before.topology);
  const source = { ...before.layout, state: make("source two field banks", data),
    cellIds: before.physicalCells, rowIds: before.physicalRows, liveControl: {
      buffer: topology, scalarParityWord: before.scalarParityWord, faceParityWord: before.faceParityWord,
      densityOffsets: [before.layout.densityOffset, before.layout.densityOtherOffset] as const,
      gammaOffsets: [before.layout.gammaOffset, before.layout.gammaOtherOffset] as const,
      velocityOffsets: [before.layout.velocityOffset, before.layout.velocityOtherOffset] as const,
      faceOffsets: [before.layout.faceOffset, before.layout.faceOtherOffset] as const,
    } };
  const target = { ...layout, state: make("target fields", new Float32Array(layout.length).fill(-999)),
    topology: make("target accepted native topology", packed.topology), atlas: fixture.target.atlas,
    cellIds: Uint32Array.from(fixture.target.cells, cell => cell.id).reverse(),
    rowIds: packed.physicalRows.slice().reverse() };
  const setParity = (scalar: number, face: number) => {
    device.queue.writeBuffer(topology, 4 * before.scalarParityWord, Uint32Array.from([scalar, face]));
  };
  return { before, packed, layout, source, target, data, setParity,
    destroy: () => buffers.forEach(buffer => buffer.destroy()) };
}

async function execute(device: GPUDevice, prepared: PreparedSparseCM12GenerationTransfer) {
  const encoder = device.createCommandEncoder(); prepared.encode(encoder);
  device.queue.submit([encoder.finish()]); await prepared.validate();
}
async function read(device: GPUDevice, buffer: GPUBuffer) {
  const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange()).slice();
  } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
}

for (const fixture of fixtures) dawnTest(`${fixture.name}: GPU geometry transfer agrees with independent physical overlaps`,
  { timeout: 120_000 }, () => withDevice(fixture.name, async device => {
    const r = resources(device, fixture); let prepared: PreparedSparseCM12GenerationTransfer | undefined;
    try {
      await assert.rejects(prepareSparseCM12GPUGenerationTransfer(device, r.before.recipe, r.source, r.target, 0, fixture.air),
        SparseCM12GenerationBudgetDeferred);
      prepared = await prepareSparseCM12GPUGenerationTransfer(device, r.before.recipe, r.source, r.target, undefined, fixture.air);
      // Independent bank selection happens after preparation. Reusing the
      // command object also proves its GPU hash/receipt is reset per encode.
      for (const [scalar, face] of [[1, 0], [0, 1]]) {
        r.setParity(scalar!, face!);
        await execute(device, prepared);
        const actual = await read(device, r.target.state);
        const expected = referenceGenerationTransfer(fixture.source, fixture.target, r.data, r.before.layout,
          r.before.physicalCells, r.before.physicalRows, scalar!, face!, fixture.air);
        let maximumError = 0;
        const close = (at: number, value: number) => {
          const error = Math.abs(actual[at]! - value); maximumError = Math.max(maximumError, error);
          assert.ok(Number.isFinite(actual[at]) && error <= 2e-6 * Math.max(1, Math.abs(value)),
            `${fixture.name} scalar=${scalar} face=${face} word=${at}: ${actual[at]} != ${value}`);
        };
        for (const cell of fixture.target.cells) {
          const value = expected.cells[cell.id]!;
          for (const offset of [r.layout.densityOffset, r.layout.densityOtherOffset]) close(offset + cell.id, value.density);
          for (const offset of [r.layout.gammaOffset, r.layout.gammaOtherOffset]) close(offset + cell.id, value.gamma);
          close(r.layout.pressureOffset + cell.id, value.pressure);
          for (const offset of [r.layout.velocityOffset, r.layout.velocityOtherOffset])
            for (let axis = 0; axis < 3; axis++) close(offset + 4 * cell.id + axis, value.velocity[axis]!);
        }
        for (const row of fixture.target.gradientRows) for (const offset of [r.layout.faceOffset, r.layout.faceOtherOffset])
          close(offset + r.packed.physicalRows[row.id]!, expected.faces[row.id]!);
        console.log(JSON.stringify({ fixture: fixture.name, scalarParity: scalar, faceParity: face,
          checkedCells: expected.cells.length, checkedFaces: expected.faces.length, maximumError }));
      }
    } finally { prepared?.destroy(); r.destroy(); }
  }));

dawnTest("GPU transfer rejects missing execution receipts and invalid accepted fields", { timeout: 120_000 },
  () => withDevice("rejection", async device => {
    const fixture = fixtures[0]!, r = resources(device, fixture);
    let prepared: PreparedSparseCM12GenerationTransfer | undefined;
    try {
      prepared = await prepareSparseCM12GPUGenerationTransfer(device, r.before.recipe, r.source, r.target);
      await assert.rejects(prepared.validate(), /receipt/);
      // Discarded command buffers must not inherit a zero or a prior success.
      prepared.encode(device.createCommandEncoder());
      await assert.rejects(prepared.validate(), /fault 4294967295/);
      await execute(device, prepared);
      prepared.encode(device.createCommandEncoder());
      await assert.rejects(prepared.validate(), /fault 4294967295/);
      const rejected = async (bit: number) => assert.rejects(execute(device, prepared!), error => {
        const match = error instanceof Error && /fault (\d+)/.exec(error.message);
        return Boolean(match && (Number(match[1]) & bit) !== 0);
      });
      for (const [word, value, bit] of [
        [r.source.densityOffset + r.before.physicalCells[0]!, NaN, 1],
        [r.source.densityOffset + r.before.physicalCells[0]!, -.25, 1],
        [r.source.faceOffset + r.before.physicalRows[0]!, NaN, 2],
      ]) {
        const broken = r.data.slice(); broken[word!] = value!;
        device.queue.writeBuffer(r.source.state, 0, broken); await rejected(bit!);
        device.queue.writeBuffer(r.source.state, 0, r.data); await execute(device, prepared);
      }
      await assert.rejects(prepared.validate(), /receipt/, "a receipt is consumed exactly once");
    } finally { prepared?.destroy(); r.destroy(); }
  }));

for (const defect of ["unproven new air", "duplicate accepted face", "invalid face geometry"] as const) dawnTest(
  `GPU transfer rejects ${defect}`, { timeout: 120_000 }, () => withDevice(defect, async device => {
    const fixture = defect === "unproven new air" ? fixtures.find(value => value.air)! : fixtures[0]!;
    const r = resources(device, fixture); let prepared: PreparedSparseCM12GenerationTransfer | undefined;
    try {
      const recipe = defect === "duplicate accepted face" ? { ...r.before.recipe,
        rows: Uint32Array.from([...r.before.recipe.rows, r.before.recipe.rows[0]!]) } : r.before.recipe;
      if (defect === "invalid face geometry") {
        const rows = r.before.topology[3]!, base = r.before.topology[7]!;
        device.queue.writeBuffer(r.source.liveControl.buffer, 4 * (base + 3 * rows + r.before.physicalRows[0]!), new Float32Array([0]));
      }
      prepared = await prepareSparseCM12GPUGenerationTransfer(device, recipe, r.source, r.target);
      const requiredBit = defect === "unproven new air" ? 16 : defect === "duplicate accepted face" ? 32 : 4;
      await assert.rejects(execute(device, prepared), error => {
        const match = error instanceof Error && /fault (\d+)/.exec(error.message);
        return Boolean(match && (Number(match[1]) & requiredBit) !== 0);
      });
    } finally { prepared?.destroy(); r.destroy(); }
  }));
