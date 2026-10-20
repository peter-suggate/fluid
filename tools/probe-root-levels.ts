/** Holds the pressure root's launches to their host reference
 * (tools/root-levels-reference.ts), one launch at a time, on a scene's own
 * pool. Browser GPU must be off.
 *   node --import tsx tools/probe-root-levels.ts <scene> [steps] [--dt=0.017]
 * The scene advances `steps` steps (1 by default; a solver failure there is
 * printed and the probe goes on, since it reads only the pool and the phi the
 * step left). The probe then writes its own pressure, RHS and bound over the
 * root's rows, opens the pressure schedule's gate, and issues the root's own
 * launches itself, each in its own submission: the setup, a measure, a
 * V-cycle, a measure, a Full-cycle, a measure. Around every launch it reads
 * back the arena's two ranges, the rounding pairs and the job list; the
 * reference runs the same launch on the state the device started it from,
 * and every word is compared (a mask and a list exactly, a value to a few
 * parts in ten thousand of the launch's own writes in its field). So a
 * launch is judged on its own, whatever the ones before it did.
 * First the block neighbour rows the kernels read are held to the host's
 * (uniformBlockNeighbourWords of the page table read back).
 * A line a launch: ok, or DEVIATES with, per field, the words that differ
 * (the device wrote and the reference did not, the reference wrote and the
 * device did not, both wrote) and the first of them by level, block and
 * place. The last line names the first deviating launch; exits non-zero on
 * any. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import type { UniformPressureLevelWords } from '../lib/methods/uniform/pressure-plan';
import { UNIFORM_PAGE_TABLE_WORDS, uniformBlockNeighbourLayout, uniformBlockNeighbourWords } from '../lib/methods/uniform/uniform-lattice-address';
import { UniformMixedPressurePasses } from '../lib/methods/uniform/uniform-mixed-pressure-cycles';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { rootLevelsReference, type RootLevelsFacts, type RootLevelsState } from './root-levels-reference';

const method = uniformNarrowBandMethod;
const [sceneId, stepsText] = process.argv.slice(2).filter(a => !a.startsWith("--"));
const steps = Number(stepsText ?? 1);
assert.ok(sceneId && Number.isInteger(steps) && steps >= 0 && steps <= 240, "usage: probe-root-levels.ts <scene> [steps] [--dt=0.017]");
const dt = Number(process.argv.find(a => a.startsWith("--dt="))?.split("=")[1] ?? 0.017);
assert.ok(dt > 0 && dt <= 0.1);
const scene = structuredClone(sceneDocument(getSceneDefinition(sceneId)));
scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };

type Words = RootLevelsFacts["words"];
interface Launch { tiles: number }
interface Range { buffer: GPUBuffer; offset?: number; size?: number }
interface Cycles {
  pipelines: Map<string, Launch>; traversals: Record<"setup" | "setupFull" | "measure" | "vCycle" | "fullCycle", Launch[]>;
  dispatch(passes: UniformMixedPressurePasses, launch: Launch, capacity: boolean): void;
  resourceEntries: { binding: number; resource: Range }[]; list: GPUBuffer; rounding: GPUBuffer; words: Words; tiled: number; listed: boolean;
}
interface Frame {
  cycles: Cycles; fields: { pressure: { levels: readonly UniformPressureLevelWords[] } };
  levels: { ownership: { support: GPUBuffer; pageMirror: { buffer: GPUBuffer; offset: number }; layout: { address: { blocks: number }; lattice: { cellSize_m: readonly [number, number, number] } } } }[];
}
const hash = (...v: number[]): number => { let h = 2166136261; for (const x of v) { h ^= x + 0x9e3779b9 + (h << 6) + (h >>> 2); h = Math.imul(h, 16777619); } h ^= h >>> 15; h = Math.imul(h, 2246822519); h ^= h >>> 13; return (h >>> 0) / 4294967296; };

let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ['backend=metal']); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const gpuDevice = device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  gpuDevice.addEventListener('uncapturederror', e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  solver = await method.createSolverAsync!(gpuDevice, scene, 'balanced', { ...method.appDefaults, timeStep: 'scene' }, undefined, () => { }) as WebGPUUniformReferenceSolver;
  for (let step = 1; step <= steps; step++) {
    try { solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion(); }
    catch (e) { console.log(`step ${step}: the solver stopped (${(e as Error).message.slice(0, 160)}); the probe goes on from the pool and phi it left`); break; }
    if (solver.info.simulationPipelineError) { console.log(`step ${step}: ${solver.info.simulationPipelineError.slice(0, 160)}`); break; }
  }
  await gpuDevice.queue.onSubmittedWorkDone();

  const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame, cycles = frame.cycles, root = frame.levels[0]!.ownership;
  const N = root.layout.address.blocks, keys = 512 * N, count = 896 * N, h = root.layout.lattice.cellSize_m, levels = frame.fields.pressure.levels;
  const entry = (binding: number): Range | undefined => cycles.resourceEntries.find(e => e.binding === binding)?.resource;
  const mixedRange = entry(0)!, nativeRange = entry(1)!, phiRange = entry(2)!, recordRange = entry(4);
  const read = async (buffer: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer> => {
    const staging = gpuDevice.createBuffer({ label: "Root levels probe readback", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = gpuDevice.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, staging, 0, size); gpuDevice.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy(); return out;
  };
  const neighbours = uniformBlockNeighbourLayout(N);
  const mirror = new Uint32Array(await read(root.pageMirror.buffer, 4 * root.pageMirror.offset, 4 * neighbours.wallCount));
  const facts: RootLevelsFacts = {
    blocks: N, h, pages: mirror.subarray(0, UNIFORM_PAGE_TABLE_WORDS + N), support: new Uint32Array(await read(root.support, 0, root.support.size)),
    phi: new Float32Array(await read(phiRange.buffer, phiRange.offset ?? 0, 4 * keys)),
    record: recordRange ? new Float32Array(await read(recordRange.buffer, recordRange.offset ?? 0, 16 * count)) : undefined,
    levels, words: cycles.words, listed: cycles.listed, tiled: cycles.tiled,
  };
  const reference = rootLevelsReference(facts);
  {
    const held: string[] = []; for (let b = 1; b < N - 1; b++) if (reference.stored(b)) held.push(`${b}:${reference.pageOf(b).join(",")}`);
    console.log(`${sceneId} after ${steps} steps: pool of ${N} blocks, ${held.length} hold a page (block:page ${held.join(" ")}); ${facts.record ? "solid record" : "no solid record"}; root jobs ${cycles.listed ? "listed" : "every octant"}; levels ${cycles.tiled} and below in the one-workgroup launch; h ${h.join(" ")}`);
    const rows = uniformBlockNeighbourWords(facts.pages.subarray(0, UNIFORM_PAGE_TABLE_WORDS), facts.pages.subarray(UNIFORM_PAGE_TABLE_WORDS)); let wrong = 0, first = "";
    for (let b = 0; b < N; b++) {
      // A row is read only for a block that holds a page.
      if (!reference.stored(b)) continue;
      for (let i = 0; i < 27; i++) if (rows[27 * b + i] !== mirror[neighbours.rows + 27 * b + i]) { if (!wrong++) first = `block ${b} word ${i}: device ${mirror[neighbours.rows + 27 * b + i]}, host ${rows[27 * b + i]}`; }
    }
    console.log(wrong ? `DEVIATES block neighbour rows: ${wrong} words (${first})` : `ok   block neighbour rows (${27 * held.length} words)`);
    if (wrong) process.exitCode = 1;
  }

  // The probe's own problem over the rows of the blocks that hold a page; zero elsewhere, unbounded.
  {
    const p = new Float32Array(count), b = new Float32Array(count), m = new Float32Array(count).fill(-3.402823e38), scale = 0.2 / (4 * Math.min(...h)) ** 2;
    for (let j = 0; j < count; j++) {
      const block = j < keys ? j >>> 9 : Math.floor((j - keys) / 384); if (!reference.stored(block)) continue;
      p[j] = 0.2 * (hash(1, j) - 0.5); b[j] = scale * (hash(2, j) - 0.5); if (hash(3, j) < 1 / 4000) m[j] = p[j]! - 0.01;
    }
    const at = (word: number): number => (mixedRange.offset ?? 0) + 4 * word;
    gpuDevice.queue.writeBuffer(mixedRange.buffer, at(cycles.words.p), p); gpuDevice.queue.writeBuffer(mixedRange.buffer, at(cycles.words.b0), b); gpuDevice.queue.writeBuffer(mixedRange.buffer, at(cycles.words.m0), m);
    // The pressure schedule's gate (support 9n+24), open.
    gpuDevice.queue.writeBuffer(root.support, 4 * (9 * keys + 24), new Uint32Array([0]));
    const encoder = gpuDevice.createCommandEncoder(); encoder.clearBuffer(cycles.rounding); encoder.clearBuffer(cycles.list, 0, 4); encoder.clearBuffer(cycles.list, cycles.list.size - 4 * (2 * N + 1), 4); gpuDevice.queue.submit([encoder.finish()]);
  }
  const state = async (): Promise<RootLevelsState> => ({
    mixed: new Float32Array(await read(mixedRange.buffer, mixedRange.offset ?? 0, mixedRange.size!)), native: new Float32Array(await read(nativeRange.buffer, nativeRange.offset ?? 0, nativeRange.size!)),
    rounding: new Float32Array(await read(cycles.rounding, 0, cycles.rounding.size)), list: new Uint32Array(await read(cycles.list, 0, cycles.list.size)),
  });

  // Every word of the two arena ranges by field: a level and a row for the report; exact: the row's mask.
  interface Field { name: string; array: "mixed" | "native" | "rounding"; start: number; length: number; level: number; stride: number; first: number; exact?: number }
  const fields: Field[] = [];
  for (const [name, word] of Object.entries(cycles.words)) fields.push({ name: `root ${name}`, array: "mixed", start: word, length: count, level: 0, stride: 1, first: 0 });
  levels.forEach((x, l) => {
    const base = levels[0]!.pressure[0], rows = x.cells + x.slots, add = (name: string, word: number, length: number, stride: number, first: number, exact?: number): void => { fields.push({ name: `level ${l} ${name}`, array: "native", start: word - base, length, level: l, stride, first, exact }); };
    add("p0", x.pressure[0], rows, 1, 0); add("p1", x.pressure[1], rows, 1, 0); add("rhs0", x.rhs[0], rows, 1, 0); add("rhs1", x.rhs[1], rows, 1, 0); add("bound0", x.minimum[0], rows, 1, 0); add("bound1", x.minimum[1], rows, 1, 0);
    add("coefficients and mask", x.coefficients, 7 * rows, 7, 0, 6); add("phi", x.phi[0], x.cells, 1, 0); add("phi continued", x.phi[1], x.cells, 1, 0); add("fractions", x.topology, 4 * x.cells, 4, 0); add("wall fraction", x.wall, x.slots, 1, x.cells);
  });
  fields.push({ name: "rounding", array: "rounding", start: 2, length: 2 * count, level: 0, stride: 2, first: 0 });
  const bits = (a: Float32Array): Uint32Array => new Uint32Array(a.buffer, a.byteOffset, a.length);
  /** The words of a launch that differ, by field. */
  const compare = (before: RootLevelsState, expected: RootLevelsState, device: RootLevelsState): { lines: string[]; wrote: number } => {
    const lines: string[] = []; let wrote = 0;
    const covered = { mixed: new Uint8Array(device.mixed.length), native: new Uint8Array(device.native.length), rounding: new Uint8Array(device.rounding.length) };
    let pressureScale = 0;
    for (const field of fields) {
      const pre = bits(before[field.array]), ref = bits(expected[field.array]), dev = bits(device[field.array]), value = expected[field.array], got = device[field.array];
      let scale = 0;
      for (let i = field.start; i < field.start + field.length; i++) { covered[field.array][i] = 1; if (ref[i] !== pre[i]) { wrote++; const v = Math.abs(value[i]!); if (v < 1e30 && v > scale) scale = v; } }
      if (field.name === "root p") pressureScale = scale;
      // The rounding pairs are the pressure's own low words.
      if (field.array === "rounding") scale = Math.max(scale, pressureScale);
      let deviceOnly = 0, referenceOnly = 0, both = 0, first = "";
      for (let i = field.start; i < field.start + field.length; i++) {
        if (ref[i] === dev[i]) continue;
        const a = value[i]!, b = got[i]!, exact = field.exact !== undefined && (i - field.start) % field.stride === field.exact;
        if (!exact && Math.abs(a - b) <= 2e-4 * Math.max(Math.abs(a), Math.abs(b)) + 2e-5 * scale) continue;
        if (!exact && Math.abs(a) >= 1e30 && Math.abs(b) >= 1e30 && Math.sign(a) === Math.sign(b)) continue;
        if (ref[i] === pre[i]) deviceOnly++; else if (dev[i] === pre[i]) referenceOnly++; else both++;
        if (!first) first = `${reference.describe(field.level, field.first + Math.floor((i - field.start) / field.stride))}${field.stride > 1 ? ` word ${(i - field.start) % field.stride}` : ""}: device ${b}, reference ${a}, before ${before[field.array][i]}`;
      }
      if (deviceOnly + referenceOnly + both) lines.push(`${field.name}: ${deviceOnly + referenceOnly + both} words (${deviceOnly} only the device wrote, ${referenceOnly} only the reference wrote, ${both} both wrote); first ${first}`);
    }
    for (const array of ["mixed", "native", "rounding"] as const) {
      const pre = bits(before[array]), dev = bits(device[array]); let stray = 0;
      for (let i = 0; i < dev.length; i++) if (!covered[array][i] && pre[i] !== dev[i]) stray++;
      if (stray) lines.push(`${array}: ${stray} words outside every field changed`);
    }
    // The list: its count, its jobs as a set, a class a job.
    const jobs = (s: RootLevelsState): string => Array.from(s.list.subarray(4, 4 + Math.min(s.list[0]!, 8 * N))).sort((a, b) => a - b).join(",");
    if (expected.list[0] !== device.list[0] || jobs(expected) !== jobs(device)) lines.push(`job list: device ${device.list[0]} jobs, reference ${expected.list[0]}${jobs(expected) === jobs(device) ? "" : "; the jobs differ"}`);
    let classes = 0; for (let i = 4 + 8 * N; i < 4 + 16 * N; i++) if (expected.list[i] !== device.list[i]) classes++;
    if (classes) lines.push(`job classes: ${classes} words`);
    return { lines, wrote };
  };

  const names = new Map<Launch, string>(); for (const [name, launch] of cycles.pipelines) names.set(launch, name);
  const t = cycles.traversals, sequence = [...t.setupFull, ...t.measure, ...t.vCycle, ...t.measure, ...t.fullCycle, ...t.measure];
  const passes = new UniformMixedPressurePasses(undefined, "Root levels probe");
  let before = await state(), firstDeviation: string | undefined, deviations = 0;
  for (let i = 0; i < sequence.length; i++) {
    const launch = sequence[i]!, name = names.get(launch)!;
    const encoder = gpuDevice.createCommandEncoder(); passes.begin(encoder); cycles.dispatch(passes, launch, false); passes.finish(); gpuDevice.queue.submit([encoder.finish()]);
    const after = await state();
    const expected: RootLevelsState = { mixed: before.mixed.slice(), native: before.native.slice(), rounding: before.rounding.slice(), list: before.list.slice() };
    reference.run(name.replace(/\.full$/, ""), expected);
    const { lines, wrote } = compare(before, expected, after);
    if (lines.length) { deviations++; firstDeviation ??= `launch ${i + 1} (${name})`; console.log(`DEVIATES ${i + 1} ${name}\n  ${lines.join("\n  ")}`); }
    else console.log(`ok   ${i + 1} ${name} (${wrote} words, list ${after.list[0]})`);
    before = after;
  }
  console.log(deviations ? `${deviations} of ${sequence.length} launches deviate; the first is ${firstDeviation}` : `all ${sequence.length} launches agree with the reference`);
  console.log(`errors ${JSON.stringify(errors)}`);
  if (deviations || errors.length) process.exitCode = 1;
} finally { solver?.destroy(); await device?.queue.onSubmittedWorkDone(); device?.destroy(); }
