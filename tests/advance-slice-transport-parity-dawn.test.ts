import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { createSparseCM12ComplexityScene, getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { productionSceneSliceSeed } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import type { SliceNumericalTopology, SliceTransportMicrostepReceipt } from
  "../lib/methods/adaptive-volume/advance-slice/slice-stage-numerics";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const INVALID = 0xffff_ffff;
const TRANSPORT_SOURCE_SHA256 = "4792eff69e8f5eddef056f09a8fa8c47ed627c0076e495f905d22a1f34463bec";

type ResidentAccess = {
  readonly state: GPUBuffer;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly layout: {
    readonly geometricInterfacePlanes: number;
    readonly pressure: number;
    readonly faceA: number;
    readonly faceB: number;
  };
  readFrameControlQA(): Promise<{ readonly faceParity: number }>;
};

type MovingSnapshot = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGeometricMovingDualSnapshotQA"]>>;

const floatBits = (value: number): number =>
  new Uint32Array(new Float32Array([value]).buffer)[0]!;
const bitsFloat = (bits: number): number =>
  new Float32Array(new Uint32Array([bits]).buffer)[0]!;

async function readFloats(device: GPUDevice, source: GPUBuffer,
  sourceFloat: number, count: number): Promise<Float32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * count),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 4 * sourceFloat, readback, 0, 4 * count);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function transportSourceFingerprint(): Promise<string> {
  const paths = [
    "lib/methods/adaptive-volume/resident-volume.wgsl.ts",
    "lib/methods/adaptive-volume/geometric-interface-resident.wgsl.ts",
    "lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts",
    "lib/methods/adaptive-volume/advance-slice/slice-stage-numerics.ts",
  ];
  const source = await Promise.all(paths.map(path => readFile(resolve(path), "utf8")));
  const extracts = [
    source[0]!, source[1]!,
    source[2]!.slice(source[2]!.indexOf('stage("conservative-transport"'),
      source[2]!.indexOf("if (pendingFrame) return pendingFrame")),
    source[3]!.slice(source[3]!.indexOf("export function reconstructSliceInterfaces"),
      source[3]!.indexOf("export function transportSliceVolumeMicrostep") +
      "export function transportSliceVolumeMicrostep".length),
  ];
  assert.ok(extracts.every(part => part.length > 100), "transport source anchors moved");
  return createHash("sha256").update(extracts.join("\nSOURCE\n")).digest("hex");
}

interface SlabCell {
  readonly cpu: number;
  readonly gpu: number;
  readonly widthZ: number;
  readonly z0: number;
  readonly z1: number;
}

interface CpuTransportFrame {
  readonly topology: SliceNumericalTopology;
  readonly receipt: SliceTransportMicrostepReceipt;
  readonly committedNormal: Float32Array;
  readonly committedOffset: Float32Array;
  readonly pressureMember: Uint8Array;
  readonly pressureRhs: Float32Array;
  readonly pressureDiagonal: Float32Array;
  readonly pressure: Float32Array;
  readonly projectedFaceVelocity: Float32Array;
}

const CPU_TRANSPORT_CAPTURED = Symbol("CPU transport captured");

function runCpuTransportFrame(cpu: ReturnType<typeof createAdvanceSlice>): CpuTransportFrame {
  let topology: SliceNumericalTopology | undefined;
  let receipt: SliceTransportMicrostepReceipt | undefined;
  let committedNormal: Float32Array | undefined;
  let committedOffset: Float32Array | undefined;
  let pressureMember: Uint8Array | undefined;
  let pressureRhs: Float32Array | undefined;
  let pressureDiagonal: Float32Array | undefined;
  let pressure: Float32Array | undefined;
  let projectedFaceVelocity: Float32Array | undefined;
  try {
    advanceSlice(cpu, { pressureIterations: 64, pressureRelativeTolerance: 1e-6,
      onTransportMicrostep: (_step, slice, value) => {
        topology = slice.numericalTopology; receipt = value;
      }, onStageComplete: (stage, slice) => {
        if (stage !== "conservative-transport") return;
        committedNormal = Float32Array.from(slice.fields.interfaceNormal);
        committedOffset = Float32Array.from(slice.fields.interfaceOffset);
        pressureMember = Uint8Array.from(slice.fields.pressureMember);
        pressureRhs = Float32Array.from(slice.fields.pressureRhs);
        pressureDiagonal = Float32Array.from(slice.fields.pressureDiagonal);
        pressure = Float32Array.from(slice.fields.pressure);
        projectedFaceVelocity = Float32Array.from(slice.fields.faceVelocity);
        // Stop before activity/candidate commit so the second differential
        // frame uses the same frozen accepted topology as the resident.
        throw CPU_TRANSPORT_CAPTURED;
      } });
  } catch (error) {
    if (error !== CPU_TRANSPORT_CAPTURED) throw error;
  }
  assert.equal(cpu.fields.fault, null);
  assert.ok(topology && receipt && committedNormal && committedOffset
    && pressureMember && pressureRhs && pressureDiagonal && pressure
    && projectedFaceVelocity);
  return { topology, receipt, committedNormal, committedOffset,
    pressureMember, pressureRhs, pressureDiagonal, pressure, projectedFaceVelocity };
}

function centerSlabMap(snapshot: MovingSnapshot, topology: SliceNumericalTopology,
  centerCellZ: number): readonly SlabCell[] {
  const plane = centerCellZ + 0.5;
  const gpu = snapshot.acceptedCells.map(cell => {
    const at = 8 * cell;
    const center = snapshot.descriptors.slice(at, at + 3);
    const widths = snapshot.descriptors.slice(at + 4, at + 7);
    return { cell, center, widths,
      z0: center[2]! - 0.5 * widths[2]!, z1: center[2]! + 0.5 * widths[2]! };
  }).filter(cell => plane >= cell.z0 && plane < cell.z1);
  return topology.cells.map(cell => {
    const match = gpu.filter(candidate => candidate.center[0] === cell.center[0]
      && candidate.center[1] === cell.center[1]
      && candidate.widths[0] === cell.widths[0]
      && candidate.widths[1] === cell.widths[1]);
    assert.equal(match.length, 1, `center-slab GPU owner for CPU cell ${cell.id}`);
    return { cpu: cell.id, gpu: match[0]!.cell, widthZ: match[0]!.widths[2]!,
      z0: match[0]!.z0, z1: match[0]!.z1 };
  });
}

function assertSameWord(label: string, cpu: number, gpu: number, mismatches: string[]): void {
  if (floatBits(cpu) !== floatBits(gpu)) {
    mismatches.push(`${label}: cpu=${cpu} [${floatBits(cpu).toString(16)}] gpu=${gpu} [${
      floatBits(gpu).toString(16)}]`);
  }
}

function compareTransport(snapshot: MovingSnapshot, topology: SliceNumericalTopology,
  frame: CpuTransportFrame, slab: readonly SlabCell[], gpuPressure: Float32Array,
  gpuFaceVelocity: Float32Array): readonly string[] {
  const mismatches: string[] = [];
  const receipt = frame.receipt;
  const byCpu = new Map(slab.map(cell => [cell.cpu, cell]));
  const comparedRows = new Set<number>();
  for (const face of topology.subfaces) {
    if (face.negativeCell < 0 || face.positiveCell < 0) continue;
    const negative = byCpu.get(face.negativeCell), positive = byCpu.get(face.positiveCell);
    assert.ok(negative && positive);
    const candidates: number[] = [];
    for (let gpuFace = 0; gpuFace < snapshot.faceCount; gpuFace += 1) {
      const at = 4 * gpuFace, a = snapshot.metadata[at]!, b = snapshot.metadata[at + 1]!;
      if (a === negative.gpu && b === positive.gpu) candidates.push(gpuFace);
    }
    assert.equal(candidates.length, 1, `GPU physical subface for CPU face ${face.id}`);
    const gpuFace = candidates[0]!, zWidth = Math.min(negative.z1, positive.z1)
      - Math.max(negative.z0, positive.z0);
    const metadataAt = 4 * gpuFace, stateAt = 4 * gpuFace;
    const gpuRow = snapshot.metadata[metadataAt + 2]!;
    if (!comparedRows.has(face.rowId)) {
      comparedRows.add(face.rowId);
      assertSameWord(`row ${face.rowId}/${gpuRow} projected velocity`,
        frame.projectedFaceVelocity[face.rowId]!, gpuFaceVelocity[gpuRow]!, mismatches);
    }
    assertSameWord(`face ${face.id}/${gpuFace} area`, Math.fround(face.area * zWidth),
      bitsFloat(snapshot.metadata[metadataAt + 3]!), mismatches);
    assertSameWord(`face ${face.id}/${gpuFace} low`,
      Math.fround(receipt.lowFlux[face.id]! * zWidth), snapshot.faceState[stateAt]!, mismatches);
    assertSameWord(`face ${face.id}/${gpuFace} sweep`,
      Math.fround(receipt.sweep[face.id]! * zWidth), snapshot.faceState[stateAt + 3]!, mismatches);
    assertSameWord(`face ${face.id}/${gpuFace} high`,
      Math.fround(receipt.highFlux[face.id]! * zWidth), snapshot.faceState[stateAt + 1]!, mismatches);
    assertSameWord(`face ${face.id}/${gpuFace} limited`,
      Math.fround(receipt.limitedFlux[face.id]! * zWidth), snapshot.faceState[stateAt + 2]!, mismatches);
  }
  for (const cell of slab) {
    const expected = Math.fround(receipt.nextVolume[cell.cpu]! * cell.widthZ);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} next-volume`, expected,
      snapshot.fields.currentVolume[cell.gpu]!, mismatches);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} increase`, receipt.increase[cell.cpu]!,
      snapshot.fields.currentDual[cell.gpu]!, mismatches);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} decrease`, receipt.decrease[cell.cpu]!,
      snapshot.fields.proposedProximal[cell.gpu]!, mismatches);
    assert.equal(snapshot.fields.pressureMembership[cell.gpu] !== 0,
      frame.pressureMember[cell.cpu] !== 0,
      `cell ${cell.cpu}/${cell.gpu} pressure membership`);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} pressure rhs`,
      Math.fround(frame.pressureRhs[cell.cpu]! * cell.widthZ),
      snapshot.fields.rhs[cell.gpu]!, mismatches);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} pressure diagonal`,
      Math.fround(frame.pressureDiagonal[cell.cpu]! * cell.widthZ),
      snapshot.fields.diagonal[cell.gpu]!, mismatches);
    assertSameWord(`cell ${cell.cpu}/${cell.gpu} pressure`,
      frame.pressure[cell.cpu]!, gpuPressure[cell.gpu]!, mismatches);
  }
  return mismatches;
}

dawnTest("2-D mixed-rung reconstruction and geometric FCT match a production Z extrusion",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-transport-parity-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const fingerprint = await transportSourceFingerprint();
      assert.equal(fingerprint, TRANSPORT_SOURCE_SHA256,
        "production or CPU transport source changed; refresh this differential deliberately");
      const dawn = await import(pathToFileURL(resolve(dawnModule!)).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); errors.push(event.error.message);
      });

      const definition = getSceneDefinition("coarse-surface-translation");
      const scene = sceneDocument(definition);
      // The authored analytic scene is Z invariant. Symmetry at the omitted
      // pair of depth faces makes the resident stencil the exact extrusion
      // counterpart rather than a one-sided finite-depth boundary stencil.
      scene.container.depthBoundary = "symmetry";
      const validationDefinition = { ...definition, build: () => structuredClone(scene) };
      const values = resolveMethodValues(adaptiveMassMethod,
        definition.methodProfile!.quality, definition.methodProfile!.overrides);
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene,
        definition.methodProfile!.quality, values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady(); await solver.waitForTopologyReady();
      solver.setTopologyFrozen(true);
      const cpu = createAdvanceSlice(productionSceneSliceSeed(validationDefinition));
      const dt = scene.numerics.fixedDt_s!;
      const resident = (solver.sparseWorldTrace as unknown as { resident: ResidentAccess }).resident;
      let before = await solver.readGeometricMovingDualSnapshotQA();
      for (let frame = 1; frame <= 2; frame += 1) {
        const prePlanes = await readFloats(device, resident.state,
          resident.layout.geometricInterfacePlanes, 4 * resident.cellCount);
        const initialDensity = Float32Array.from(cpu.fields.density);
        const cpuFrame = runCpuTransportFrame(cpu);
        const initialSlab = centerSlabMap(before, cpuFrame.topology,
          cpu.scene.viewport.centerCellZ);
        const density = before.acceptedScalarParity ? before.fields.densityB : before.fields.densityA;
        const initialMismatches: string[] = [];
        for (const cell of initialSlab) assertSameWord(
          `frame ${frame} initial density ${cell.cpu}/${cell.gpu}`,
          initialDensity[cell.cpu]!, density[cell.gpu]!, initialMismatches);
        assert.deepEqual(initialMismatches, []);

        while (!solver.advanceTo(frame * dt, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
        await solver.assertSimulationHealthy();
        const after: MovingSnapshot = await solver.readGeometricMovingDualSnapshotQA();
        const slab = centerSlabMap(after, cpuFrame.topology, cpu.scene.viewport.centerCellZ);
        const frameControl = await resident.readFrameControlQA();
        const gpuPressure = await readFloats(device, resident.state,
          resident.layout.pressure, resident.cellCount);
        const gpuFaceVelocity = await readFloats(device, resident.state,
          frameControl.faceParity ? resident.layout.faceB : resident.layout.faceA,
          resident.rowCount);
        const mismatches: string[] = [...compareTransport(after,
          cpuFrame.topology, cpuFrame, slab, gpuPressure, gpuFaceVelocity)];
        const planes = await readFloats(device, resident.state,
          resident.layout.geometricInterfacePlanes, 4 * resident.cellCount);
        let certifiedPlanes = 0, nonAxisPlanes = 0;
        for (const cell of slab) {
          const gx = planes[4 * cell.gpu]!, gy = planes[4 * cell.gpu + 1]!;
          const gz = planes[4 * cell.gpu + 2]!;
          // Non-axis plane normalization contains division and sqrt, whose
          // WGSL accuracy is backend-bounded rather than raw-word portable.
          // Axis-aligned planes contain neither operation and remain exact.
          const exactExtrusion = gz === 0 && (Math.abs(gx) === 1 || Math.abs(gy) === 1);
          if (!exactExtrusion) {
            if (gx !== 0 || gy !== 0 || gz !== 0) nonAxisPlanes += 1;
            continue;
          }
          certifiedPlanes += 1;
          assertSameWord(`frame ${frame} plane ${cell.cpu}/${cell.gpu} nx`,
            cpuFrame.committedNormal[2 * cell.cpu]!, gx, mismatches);
          assertSameWord(`frame ${frame} plane ${cell.cpu}/${cell.gpu} ny`,
            cpuFrame.committedNormal[2 * cell.cpu + 1]!, gy, mismatches);
          assertSameWord(`frame ${frame} plane ${cell.cpu}/${cell.gpu} offset`,
            cpuFrame.committedOffset[cell.cpu]!, planes[4 * cell.gpu + 3]!, mismatches);
        }
        assert.ok(nonAxisPlanes > 0, `frame ${frame} fixture must exercise non-axis PLIC`);
        // WGSL f32 division permits 2.5 ULP. Metal rounds this one ratio one
        // ULP below ECMAScript's correctly-rounded f64 quotient cast to f32;
        // retain the exact operand/result receipt while requiring all material
        // fluxes and committed volumes above to remain bit identical.
        const permittedBackendDivision = frame === 2 ? new Set([
          "cell 22/42 increase: cpu=0.9999799132347107 [3f7ffeaf] gpu=0.9999798536300659 [3f7ffeae]",
          "cell 23/43 increase: cpu=0.9999799132347107 [3f7ffeaf] gpu=0.9999798536300659 [3f7ffeae]",
        ]) : new Set<string>();
        assert.deepEqual(mismatches.filter((value: string) => !permittedBackendDivision.has(value)), [],
          mismatches.slice(0, 30).join("\n"));
        assert.deepEqual(new Set(mismatches), permittedBackendDivision);
        before = after;
      }
      assert.deepEqual(errors, []);
      assert.match(fingerprint, /^[0-9a-f]{64}$/);
    } finally {
      solver?.destroy(); device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });

dawnTest("the invariant-axis reconstruction branch preserves genuine 3-D normals",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-transport-parity-dawn.test.ts true-3d-negative");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(resolve(dawnModule!)).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); errors.push(event.error.message);
      });
      const scene = createSparseCM12ComplexityScene("symmetric-3d");
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "scene", selectorMode: "coarse-first",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
        values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady(); await solver.waitForTopologyReady();
      solver.setTopologyFrozen(true);
      const dt = scene.numerics.fixedDt_s!;
      while (!solver.advanceTo(dt, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();
      const snapshot = await solver.readGeometricMovingDualSnapshotQA();
      const resident = (solver.sparseWorldTrace as unknown as { resident: ResidentAccess }).resident;
      const planes = await readFloats(device, resident.state,
        resident.layout.geometricInterfacePlanes, 4 * resident.cellCount);
      const density = snapshot.acceptedScalarParity
        ? snapshot.fields.densityB : snapshot.fields.densityA;
      let positiveZ = 0, negativeZ = 0, oblique3d = 0;
      for (const cell of snapshot.acceptedCells) {
        const rho = density[cell]!;
        if (!(rho > 0 && rho < 1)) continue;
        const nx = planes[4 * cell]!, ny = planes[4 * cell + 1]!, nz = planes[4 * cell + 2]!;
        if (nz > 1e-5) positiveZ += 1;
        if (nz < -1e-5) negativeZ += 1;
        if (Math.abs(nx) > 1e-5 && Math.abs(ny) > 1e-5 && Math.abs(nz) > 1e-5) oblique3d += 1;
      }
      assert.ok(positiveZ > 0, "true 3-D surface must retain +Z normals");
      assert.ok(negativeZ > 0, "true 3-D surface must retain -Z normals");
      assert.ok(oblique3d > 0, "true 3-D surface must retain oblique XYZ normals");
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });

dawnTest("a hydrostatic Z extrusion matches a nonzero pressure solve and projection",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-transport-parity-dawn.test.ts nonzero-pressure");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(resolve(dawnModule!)).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); errors.push(event.error.message);
      });
      const definition = getSceneDefinition("hydrostatic-power-two-level");
      const scene = sceneDocument(definition);
      scene.container.depthBoundary = "symmetry";
      const validationDefinition = { ...definition, build: () => structuredClone(scene) };
      const quality = definition.methodProfile?.methodId === "adaptive-volume"
        ? definition.methodProfile.quality : "balanced";
      const overrides = definition.methodProfile?.methodId === "adaptive-volume"
        ? definition.methodProfile.overrides : {};
      const values = resolveMethodValues(adaptiveMassMethod, quality, overrides);
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene,
        quality, values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady(); await solver.waitForTopologyReady();
      solver.setTopologyFrozen(true);
      const cpu = createAdvanceSlice(productionSceneSliceSeed(validationDefinition));
      const resident = (solver.sparseWorldTrace as unknown as { resident: ResidentAccess }).resident;
      const before = await solver.readGeometricMovingDualSnapshotQA();
      const cpuInitialDensity = Float32Array.from(cpu.fields.density);
      const cpuFrame = runCpuTransportFrame(cpu);
      assert.ok(cpu.pressureReceipt && cpu.pressureReceipt.iterations > 0,
        "fixture must execute the pressure recurrence");
      assert.ok(cpuFrame.pressure.some(value => value !== 0),
        "fixture must publish nonzero pressure");
      assert.ok(cpuFrame.pressureRhs.some(value => value !== 0),
        "fixture must publish a nonzero pressure RHS");
      const initialSlab = centerSlabMap(before, cpuFrame.topology,
        cpu.scene.viewport.centerCellZ);
      const initialDensity = before.acceptedScalarParity
        ? before.fields.densityB : before.fields.densityA;
      const initialMismatches: string[] = [];
      for (const cell of initialSlab) assertSameWord(
        `initial density ${cell.cpu}/${cell.gpu}`, cpuInitialDensity[cell.cpu]!,
        initialDensity[cell.gpu]!, initialMismatches);
      assert.deepEqual(initialMismatches, []);

      const dt = scene.numerics.fixedDt_s!;
      while (!solver.advanceTo(dt, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();
      const after: MovingSnapshot = await solver.readGeometricMovingDualSnapshotQA();
      const slab = centerSlabMap(after, cpuFrame.topology, cpu.scene.viewport.centerCellZ);
      const frameControl = await resident.readFrameControlQA();
      const gpuPressure = await readFloats(device, resident.state,
        resident.layout.pressure, resident.cellCount);
      const gpuFaceVelocity = await readFloats(device, resident.state,
        frameControl.faceParity ? resident.layout.faceB : resident.layout.faceA,
        resident.rowCount);
      const mismatches = compareTransport(after, cpuFrame.topology, cpuFrame,
        slab, gpuPressure, gpuFaceVelocity);
      assert.deepEqual(mismatches, [], mismatches.slice(0, 30).join("\n"));
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
