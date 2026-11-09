/**
 * Drape accuracy of the Bridson 2002 rotating sphere: positions at fixed ticks against a reference made once with a
 * converged Newton (iterations x pcgIterations and tolerances divided as given), kept in fluid-scratch.
 *   --mode ref    run the converged settings from tick 0 and save the positions at --ticks (default 30,45)
 *   --mode check  run the scene's own settings and report the rms and max distance to the reference per tick
 *     node --import tsx tools/benchmark-cloth-drape.ts --mode ref|check [--nodes 150] [--ticks 30,45] [--ref file]
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { cloneScene } from '../lib/core/model';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1]! : fallback; };
const MODE = opt('mode', 'check'), NODES = Number(opt('nodes', '150')), TICKS = opt('ticks', '30,45').split(',').map(Number).sort((a, b) => a - b);
const REF = opt('ref', join(homedir(), `code/me/fluid-scratch/cloth-bench/drape-ref-${NODES}.json`));
if (MODE !== 'ref' && MODE !== 'check') throw new Error(`Unknown mode ${MODE}`);

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
if (!adapter) throw new Error('no WebGPU adapter');
const device: GPUDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = [];
device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });

const scene = cloneScene(bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: NODES })));
const doc = scene.cloth![0]!;
const own = doc.solver?.newton;
if (!own) throw new Error('the scene names no Newton settings');
const settings = MODE === 'ref'
  ? { iterations: 32, pcgIterations: 800, pcgTolerance: (own.pcgTolerance ?? 1e-3) / 16, residualTolerance: (own.residualTolerance ?? 1e-4) / 16 }
  : own;
(scene.cloth as unknown as { solver?: unknown }[])[0] = { ...doc, solver: { ...doc.solver, newton: settings } };
console.log(`rotating sphere ${NODES}^2, ${MODE}: Newton ${JSON.stringify(settings)}`);
const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
const solver = cloth.solverFor(doc.id)!;
const saved: Record<number, number[]> = {};
const t0 = performance.now();
let tick = 0;
const newton: number[] = [];
for (const target of TICKS) {
  while (tick < target) {
    if (!cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S)) throw new Error(`step refused at tick ${tick}: ${cloth.failure?.message ?? 'pending'}`);
    await device.queue.onSubmittedWorkDone();
    newton.push((await solver.readNewtonControl())[3]!);
    tick++;
    if (cloth.failure) throw new Error(`tick ${tick}: ${cloth.failure.message}`);
  }
  saved[target] = Array.from(await solver.readPositions());
  console.log(`tick ${target} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
}
console.log(`Newton iterations per step: ${newton.join(' ')}`);
if (MODE === 'ref') {
  writeFileSync(REF, JSON.stringify({ nodes: NODES, settings, ticks: saved }));
  console.log(`reference saved to ${REF}`);
} else {
  if (!existsSync(REF)) throw new Error(`no reference at ${REF} (run --mode ref)`);
  const ref = JSON.parse(readFileSync(REF, 'utf8')) as { ticks: Record<number, number[]> };
  for (const t of TICKS) {
    const a = saved[t]!, b = ref.ticks[t];
    if (!b || b.length !== a.length) throw new Error(`the reference has no tick ${t} at this resolution`);
    let sum = 0, worst = 0;
    for (let v = 0; v < a.length / 3; v++) {
      const d2 = (a[3 * v]! - b[3 * v]!) ** 2 + (a[3 * v + 1]! - b[3 * v + 1]!) ** 2 + (a[3 * v + 2]! - b[3 * v + 2]!) ** 2;
      sum += d2; worst = Math.max(worst, d2);
    }
    console.log(`tick ${t}: rms ${(1e3 * Math.sqrt(sum / (a.length / 3))).toFixed(3)} mm, max ${(1e3 * Math.sqrt(worst)).toFixed(3)} mm`);
  }
}
cloth.destroy();
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`${errors.length} WebGPU errors`); process.exitCode = 1; }
device.destroy();
