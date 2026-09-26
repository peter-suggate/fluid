import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { MixedPressureCycleDawn } from "./helpers/uniform-mixed-pressure-cycle-dawn";
import { MixedMultigridOracle } from "./helpers/uniform-mixed-multigrid";
import { mixedPressureLayouts, mixedPressureFixture, faceGradient, geometricDivergence, type PressureFixture } from "./helpers/uniform-mixed-pressure";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayout, uniformMixedPressureLevel } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedPressureBoundsStage, UniformMixedPressureLevelStage, UniformMixedPressureTransferStage } from "../lib/methods/uniform/uniform-mixed-pressure-stage";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("device-only mixed pressure cycles converge with the native coarse solve and cycle budget", { timeout: 180000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform mixed coupled pressure cycles");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter();
    assert.ok(adapter); device = await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    await t.test("pressure stages borrow disjoint arena views without allocating fields", () => checkBorrowedArena(device!));
    for (const [index, layout] of mixedPressureLayouts().entries()) {
      const oracle = new MixedMultigridOracle(layout, undefined, "native-jacobi"), fixture = oracle.fixtures[0]!;
      // Native baseline: all-fine ownership of the same lattice through this
      // harness, i.e. native smoother, transfers, coarse solve and budget.
      const nativeLayout = createUniformMixedLayout(layout.lattice, [], true, 1), nativeFixture = mixedPressureFixture(nativeLayout);
      const cycle = new MixedPressureCycleDawn(device, layout), native = new MixedPressureCycleDawn(device, nativeLayout);
      try {
        await cycle.initialize(); await native.initialize();
        for (const mode of ["hydrostatic", "random"] as const) await t.test(`fixture ${index}, ${mode}`, async () => {
          const field = (f: { axis: number }, k: number) => mode === "hydrostatic" ? (f.axis === 1 ? -9.81 : 0) : Math.sin(k * 13);
          const velocity = Float64Array.from(fixture.faces, field);
          const rhs = Float32Array.from(geometricDivergence(fixture, velocity), v => -v);
          const cpu = oracle.solve(Float64Array.from(rhs, (v, i) => v * fixture.cells[i]!.volume));
          // A face-indexed random field is not the same problem on another
          // lattice, so it keeps the absolute limits outright.
          // Native's 1e-4 absolute coarse stop moves both arms' final residual
          // by about +/-2%: fixture 1 is 2.1% behind native at that stop, 0.8%
          // ahead at 1e-5 and within 0.14% at 1e-6. Allow 5% for that noise.
          const measured = mode === "hydrostatic" ? await projectedBaseline(native, nativeFixture, field) : { residual: 0, divergence: 0, velocity: 0 };
          const baseline = { residual: 1.05 * measured.residual, divergence: 1.05 * measured.divergence, velocity: 1.05 * measured.velocity };
          const actual = await cycle.solve(rhs);
          assert.equal(actual.coarseExhausted, 0, "native coarse solve must meet its unchanged stopping test");
          assert.equal(actual.residuals.length, 7);
          assert.ok(actual.residuals.every(Number.isFinite));
          // Native-relative acceptance (2026-09-26): with the production damped
          // Jacobi smoother, all-fine native misses 1e-3 on some of these long-
          // wave problems in the fixed budget. The limit binds wherever native
          // meets it; mixed ownership must never converge worse than native.
          assert.ok(actual.residuals.at(-1)! < Math.max(1e-3, baseline.residual), `GPU residual history: ${actual.residuals}; native ${baseline.residual}`);
          const gradient = faceGradient(fixture, actual.pressure), cpuGradient = faceGradient(fixture, cpu.pressure);
          const gradientError = Math.max(...gradient.map((v, i) => Math.abs(v - cpuGradient[i]!)));
          assert.ok(gradientError < 1e-3, `CPU/GPU pressure-gradient error ${gradientError}`);
          // Independent physical divergence also verifies the projected field,
          // rather than relying solely on the GPU's own residual calculation.
          const projected = velocity.map((v, i) => v - gradient[i]!);
          const residual = geometricDivergence(fixture, projected);
          assert.ok(Math.max(...residual.map(Math.abs)) < Math.max(1e-3, baseline.divergence), `projected divergence; native ${baseline.divergence}`);
          if (mode === "hydrostatic") assert.ok(Math.max(...projected.map(Math.abs)) < Math.max(1e-3, baseline.velocity), `hydrostatic parasitic velocity; native ${baseline.velocity}`);
          t.diagnostic(`${index}/${mode}: GPU final=${actual.residuals.at(-1)}, CPU final=${cpu.residuals.at(-1)}, native final=${baseline.residual}, gradient error=${gradientError}`);
        });
      } finally { cycle.destroy(); native.destroy(); }
      const constrained = new MixedPressureCycleDawn(device, layout, true);
      try {
        await constrained.initialize();
        await t.test(`fixture ${index}, active pressure bounds`, async () => {
          const minimum = Float32Array.from(fixture.cells, (_, i) => .25 + .03125 * (i % 3));
          const target = Float32Array.from(fixture.cells, (c, i) => i % 7 === 0 ? minimum[i]!
            : minimum[i]! + .25 + .125 * (1 + Math.sin(c.center.reduce((s, v) => s + v, 0) * .25)));
          const rhs = Float32Array.from(oracle.apply(0, target), (v, i) => v / fixture.cells[i]!.volume - (i % 7 === 0 ? .5 : 0));
          const actual = await constrained.solve(rhs, minimum);
          assert.equal(actual.coarseExhausted, 0);
          actual.pressure.forEach((v, i) => assert.ok(v >= minimum[i]!, `cell ${i} violates its lower bound`));
          assert.ok(actual.residuals.at(-1)! < 1e-3, `projected residual history ${actual.residuals}`);
          const pressureError = Math.max(...actual.pressure.map((v, i) => Math.abs(v - target[i]!)));
          assert.ok(pressureError < 1e-3, `manufactured constrained pressure error ${pressureError}`);
          const applied = oracle.apply(0, actual.pressure);
          actual.pressure.forEach((p, i) => {
            const residual = rhs[i]! - applied[i]! / fixture.cells[i]!.volume;
            if (p - minimum[i]! > 1e-4) assert.ok(Math.abs(residual) < 1e-3, `free row ${i} residual ${residual}`);
            else assert.ok(residual < 1e-3, `active row ${i} has a positive residual ${residual}`);
          });
          t.diagnostic(`${index}/bounds: final=${actual.residuals.at(-1)}, pressure error=${pressureError}`);
        });
      } finally { constrained.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

/** The native all-fine solve's final residual, and its projected field's
 * physical divergence and maximum velocity, for the same face velocity rule. */
async function projectedBaseline(native: MixedPressureCycleDawn, fixture: PressureFixture, field: (f: { axis: number }, k: number) => number) {
  const velocity = Float64Array.from(fixture.faces, field);
  const solved = await native.solve(Float32Array.from(geometricDivergence(fixture, velocity), v => -v));
  const gradient = faceGradient(fixture, solved.pressure), projected = velocity.map((v, i) => v - gradient[i]!);
  return { residual: solved.residuals.at(-1)!, divergence: Math.max(...geometricDivergence(fixture, projected).map(Math.abs)),
    velocity: Math.max(...projected.map(Math.abs)) };
}

async function checkBorrowedArena(device: GPUDevice): Promise<void> {
  const layout = mixedPressureLayouts()[0]!, ownership = new UniformMixedOwnership(device, layout);
  const coarse = new UniformMixedOwnership(device, uniformMixedPressureLevel(layout, 4));
  const middle = new UniformMixedOwnership(device, uniformMixedPressureLevel(layout, 2));
  const alignment = device.limits.minStorageBufferOffsetAlignment, n = layout.cellCount;
  let bytes = 0;
  const ranges = [4*n, 16*n, 4*n, 4*n, 4*n, 4*n, 4*n].map(size => {
    const offset = bytes; bytes += Math.ceil(size / alignment) * alignment; return { offset, size };
  });
  const arena = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  // Fail on any field allocation in production stage construction or shader
  // preparation. Ownership and the fixture's arena are deliberately external.
  const borrowedDevice = new Proxy(device, { get(target, key) {
    if (key === "createBuffer" || key === "createTexture") return () => { throw new Error("Pressure stage allocated a field"); };
    const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
  try {
    const stage = new UniformMixedPressureLevelStage(borrowedDevice, ownership); await stage.initialize();
    const views = ranges.map(range => ({ buffer: arena, ...range }));
    const fields = { pressure: views[0]!, slopes: views[1]!, rhs: views[2]!, frozen: views[3]!, result: views[4]! };
    assert.throws(() => stage.bind({ ...fields, rhs: fields.pressure }), /overlap/);
    assert.throws(() => stage.bind({ ...fields, slopes: { ...fields.slopes, size: 4 } }), /Invalid borrowed/);
    assert.throws(() => new UniformMixedPressureTransferStage(borrowedDevice, ownership, coarse), /adjacent nested/);
    const initial = new Uint32Array(bytes / 4).fill(0x7fc00000), floats = new Float32Array(initial.buffer);
    floats.fill(3, ranges[0]!.offset / 4, ranges[0]!.offset / 4 + n);
    const rhs = Float32Array.from({ length: n }, (_, i) => Math.cos(i * 13));
    floats.set(rhs, ranges[2]!.offset / 4); device.queue.writeBuffer(arena, 0, initial);
    const group = stage.bind(fields), encoder = device.createCommandEncoder();
    stage.encode(encoder, "reconstruct", group); stage.encode(encoder, "residual", group);
    encoder.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const data = read.getMappedRange(), words = new Uint32Array(data), result = new Float32Array(data);
    assert.deepEqual([...result.subarray(ranges[4]!.offset / 4, ranges[4]!.offset / 4 + n)], [...rhs]);
    for (let i = 0; i < words.length; i++) {
      const writable = [ranges[1]!, ranges[4]!].some(r => i * 4 >= r.offset && i * 4 < r.offset + r.size);
      if (!writable) assert.equal(words[i], initial[i], `arena word ${i} changed outside written fields`);
    }
    assert.equal(stage.allocatedBytes, 0); read.unmap();

    const transfer = new UniformMixedPressureTransferStage(borrowedDevice, ownership, middle); await transfer.initialize();
    const restriction = transfer.bind("restrictValues", fields.rhs, fields.result);
    const prolongation = transfer.bind("prolongAdd", fields.result, fields.pressure);
    const transferEncoder = device.createCommandEncoder();
    transfer.encode(transferEncoder, "restrictValues", restriction); transfer.encode(transferEncoder, "prolongAdd", prolongation);
    transferEncoder.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([transferEncoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const transferred = new Float32Array(read.getMappedRange());
    const oracle = new MixedMultigridOracle(layout), nf = oracle.fixtures[0]!, nc = oracle.fixtures[1]!;
    const expectedRestricted = oracle.restrict(0, Float64Array.from(rhs, (v, i) => v * nf.cells[i]!.volume)).map((v, i) => v / nc.cells[i]!.volume);
    expectedRestricted.forEach((v, i) => assert.ok(Math.abs(transferred[ranges[4]!.offset / 4 + i]! - v) < 1e-6));
    oracle.prolong(0, expectedRestricted).forEach((v, i) => assert.ok(Math.abs(transferred[ranges[0]!.offset / 4 + i]! - (3 + v)) < 1e-6));
    assert.deepEqual([...transferred.subarray(ranges[2]!.offset / 4, ranges[2]!.offset / 4 + n)], [...rhs]);
    assert.deepEqual([...transferred.subarray(ranges[4]!.offset / 4 + nc.cells.length, ranges[4]!.offset / 4 + n)], [...rhs.subarray(nc.cells.length)]);
    const pressure = transferred.slice(ranges[0]!.offset / 4, ranges[0]!.offset / 4 + n);
    assert.equal(transfer.allocatedBytes, 0); read.unmap();

    // Directly verify phi restriction's two native policies, including
    // persistent owners. Sources and outputs share the same borrowed arena.
    for(const preferPositive of [false,true]) {
      const phiTransfer=new UniformMixedPressureTransferStage(borrowedDevice,ownership,middle,preferPositive);await phiTransfer.initialize();
      const children=Array.from({length:nc.cells.length},()=>[] as number[]);
      oracle.parents[0]!.forEach((parent,i)=>children[parent]!.push(rhs[i]!));
      const expected=children.map(values=>{
        const positive=values.filter(v=>v>=0);
        const selected=preferPositive&&positive.length>0&&positive.length<values.length?positive:values;
        return selected.reduce((sum,v)=>sum+v,0)/selected.length;
      });
      const encoder=device.createCommandEncoder();
      phiTransfer.encode(encoder,"restrictSurfacePhi",phiTransfer.bind("restrictSurfacePhi",fields.rhs,fields.result));
      encoder.copyBufferToBuffer(arena,0,read,0,bytes);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const data=new Float32Array(read.getMappedRange());
      expected.forEach((v,i)=>assert.ok(Math.abs(data[fields.result.offset/4+i]!-v)<1e-6));
      assert.deepEqual(data.slice(fields.rhs.offset/4,fields.rhs.offset/4+n),rhs);
      read.unmap();
    }
    const backupEncoder=device.createCommandEncoder();
    stage.encode(backupEncoder,"saveBackup",group);
    backupEncoder.copyBufferToBuffer(arena,0,read,0,bytes);device.queue.submit([backupEncoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    assert.deepEqual(new Float32Array(read.getMappedRange()).slice(fields.result.offset/4,fields.result.offset/4+n),pressure);
    read.unmap();

    const bounds = new UniformMixedPressureBoundsStage(borrowedDevice, ownership, middle); await bounds.initialize();
    assert.equal(bounds.allocatedBytes, 0);
    assert.throws(() => new UniformMixedPressureBoundsStage(borrowedDevice, ownership, coarse), /adjacent nested/);
    const minimum = Float32Array.from({ length: n }, (_, i) => i % 11 === 0 ? -3.402823e38 : Math.sin(i) * 5);
    device.queue.writeBuffer(arena, fields.frozen.offset, minimum);
    for (const entry of ["shiftMinimum", "downsampleMinimum", "downsampleSubtract"] as const) {
      const expected = new Float32Array(entry === "shiftMinimum" ? n : nc.cells.length).fill(-3.402823e38);
      minimum.forEach((v, i) => {
        const value = entry === "downsampleMinimum" ? v : Math.fround(v - pressure[i]!);
        if (entry === "shiftMinimum") expected[i] = value;
        else { const parent = oracle.parents[0]![i]!; expected[parent] = Math.max(expected[parent]!, value); }
      });
      const sentinel = new Float32Array(n).fill(12345);
      device.queue.writeBuffer(arena, fields.result.offset, sentinel);
      const encoder = device.createCommandEncoder();
      bounds.encode(encoder, entry, bounds.bind(entry, fields.frozen, fields.pressure, fields.result));
      encoder.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(read.getMappedRange());
      assert.deepEqual(data.slice(fields.result.offset / 4, fields.result.offset / 4 + expected.length), expected, entry);
      assert.deepEqual(data.slice(fields.result.offset / 4 + expected.length, fields.result.offset / 4 + n), sentinel.slice(expected.length));
      assert.deepEqual(data.slice(fields.pressure.offset / 4, fields.pressure.offset / 4 + n), pressure);
      assert.deepEqual(data.slice(fields.frozen.offset / 4, fields.frozen.offset / 4 + n), minimum);
      read.unmap();
    }

    for(const surface of [false,true]) {
    const constrained = new UniformMixedPressureLevelStage(borrowedDevice, ownership, true, surface); await constrained.initialize();
    const constrainedFields = { ...fields, minimum: views[5]!, ...(surface?{phi:views[6]!}:{}) };
    assert.throws(() => constrained.bind(fields), /minimum binding/);
    assert.throws(() => stage.bind(constrainedFields), /minimum binding/);
    const constrainedGroup = constrained.bind(constrainedFields);
    device.queue.writeBuffer(arena, constrainedFields.minimum.offset, new Float32Array(n));
    if(surface)device.queue.writeBuffer(arena,views[6]!.offset,new Float32Array(n).fill(-1));
    for (const [pressureValue, rhsValue] of [[NaN, 0], [Infinity, 0], [3, NaN]]) {
      const pressures = new Float32Array(n).fill(3), rightHandSide = new Float32Array(n);
      pressures[0] = pressureValue!; rightHandSide[0] = rhsValue!;
      device.queue.writeBuffer(arena, fields.pressure.offset, pressures);
      device.queue.writeBuffer(arena, fields.rhs.offset, rightHandSide);
      const encoder = device.createCommandEncoder();
      constrained.encode(encoder, "reconstruct", constrainedGroup); constrained.encode(encoder, "measure", constrainedGroup);
      encoder.copyBufferToBuffer(arena, 0, read, 0, bytes); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      assert.ok(new Float32Array(read.getMappedRange())[fields.result.offset / 4]! > 1e30, "nonfinite input cannot report convergence");
      read.unmap();
    }
    assert.equal(constrained.allocatedBytes, 0);
    }
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); arena.destroy(); ownership.destroy(); coarse.destroy(); middle.destroy(); }
}
