/**
 * Probe (lane L1, cloth-m1-max-locality): is the Bridson rotating sphere with newton.contactLinearization on the same
 * simulation as with it off? From the benchmark's saved states (tools/benchmark-cloth-sphere.ts, `--cache`).
 *
 *   --mode iterates   From one saved state, one step under each (Newton cap, PCG cap) of a fixed list: first the PCG cap
 *                     swept at one Newton iteration, then the Newton cap swept at the scene's PCG cap. Every run is
 *                     deterministic, so the positions after a step capped at k are the k-th iterate. Writes the
 *                     positions of each (Float32, `--out`.bin) and the counts (`--out`.json). Runs are compared with
 *                     `--mode compare --a x --b y`.
 *                     `--perturb seed` moves every saved position coordinate by -1, 0 or +1 f32 ulp (seeded): the
 *                     scale of f32 rounding, put into the same solver.
 *   --mode quality    `--steps` consecutive steps from the saved state; per tick: intersecting face pairs and faces in
 *                     a solid (the detector's count), the least true distance to the scene's solids (vertices and a
 *                     4-per-side face lattice), |g| at the last Newton iteration's start and |g| / |m g|, the greatest
 *                     edge strain, held and travel-held vertices, plane-record peak, status, Newton and PCG iterations,
 *                     rounds, pairs, and the compiled records' fallback rows.
 *   --mode compare    CPU only: per config of two `iterates` outputs, the greatest and rms position difference.
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh check \
 *     node --import tsx tools/probe-cloth-linearization.ts --mode iterates|quality --tick 35 [--nodes 150] [--cache dir]
 *       [--lin off|factorized|blocks] [--perturb seed] [--steps n] --out path
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { cloneScene, type SceneDescription } from '../lib/core/model';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { OgcContactDetector } from '../lib/cloth/contact/ogc-detector';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription, bridson2002SceneLayout } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { bridsonSolidDistance } from '../lib/cloth/scenes/bridson-2002/solids';
import type { ClothDocument } from '../lib/cloth/types';

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1]! : fallback; };
const MODE = opt('mode', ''), OUT = opt('out', '');
if (MODE !== 'iterates' && MODE !== 'quality' && MODE !== 'compare') throw new Error(`Unknown --mode ${MODE}`);

interface IterateRun { tick: number; lin: string; perturb: number; vertices: number; configs: { newton: number; pcg: number; newtonUsed: number; pcgUsed: number; g2: number; alpha: number; candidates: number; pairs: number; rounds: number; solidContacts: number }[] }

if (MODE === 'compare') {
  const load = (p: string) => ({ meta: JSON.parse(readFileSync(`${p}.json`, 'utf8')) as IterateRun, x: new Float32Array(new Uint8Array(readFileSync(`${p}.bin`)).buffer) });
  const a = load(opt('a', '')), b = load(opt('b', '')), n = 3 * a.meta.vertices;
  console.log(`tick ${a.meta.tick}: ${a.meta.lin}${a.meta.perturb ? ` (ulp seed ${a.meta.perturb})` : ''} vs ${b.meta.lin}${b.meta.perturb ? ` (ulp seed ${b.meta.perturb})` : ''}`);
  console.log('  newton cap, pcg cap | PCG used a / b | max |dx| m | rms |dx| m | vertices past 1e-6 m | pairs a / b | candidates a / b | g2 a / b');
  a.meta.configs.forEach((c, k) => {
    const d = b.meta.configs[k]!;
    let max = 0, sum = 0, far = 0;
    for (let v = 0; v < a.meta.vertices; v++) {
      const o = k * n + 3 * v, e = Math.hypot(a.x[o]! - b.x[o]!, a.x[o + 1]! - b.x[o + 1]!, a.x[o + 2]! - b.x[o + 2]!);
      max = Math.max(max, e); sum += e * e; if (e > 1e-6) far++;
    }
    console.log(`  ${c.newton}, ${String(c.pcg).padStart(3)} | ${c.pcgUsed} / ${d.pcgUsed} | ${max.toExponential(2)} | ${Math.sqrt(sum / a.meta.vertices).toExponential(2)} | ${far} | ${c.pairs} / ${d.pairs} | ${c.candidates} / ${d.candidates} | ${c.g2.toExponential(5)} / ${d.g2.toExponential(5)}`);
  });
  process.exit(0);
}

const NODES = Number(opt('nodes', '150')), TICK = Number(opt('tick', '35')), STEPS = Number(opt('steps', '60'));
const LIN = opt('lin', 'off'), PERTURB = Number(opt('perturb', '0'));
const CACHE = opt('cache', join(homedir(), 'code/me/fluid-scratch/cloth-locality/states'));
if (LIN !== 'off' && LIN !== 'factorized' && LIN !== 'blocks') throw new Error(`Unknown --lin ${LIN}`);
if (!OUT) throw new Error('--out is required');

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
if (!adapter) throw new Error('no WebGPU adapter');
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups' as GPUFeatureName] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = [];
device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });

const source = bridson2002RotatingSphere({ nodesAcross: NODES }), baseScene = bridson2002SceneDescription(source);
const documentId = baseScene.cloth![0]!.id;
const saved = JSON.parse(readFileSync(join(CACHE, `sphere-${NODES}-tick${TICK}.json`), 'utf8')) as { nodes: number; tick: number; document: ClothDocument };
if (saved.nodes !== NODES || saved.tick !== TICK) throw new Error(`not the ${NODES}-node state at tick ${TICK}`);
const N = saved.document.geometry.data.restPositions_m.length / 3;

/** mulberry32. */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function restored(newton: { iterations?: number; pcgIterations?: number }): SceneDescription {
  const scene = cloneScene(baseScene), d = saved.document, base = d.solver?.newton ?? baseScene.cloth![0]!.solver!.newton!;
  let data = d.geometry.data;
  if (PERTURB) {
    const rnd = random(PERTURB), f = new Float32Array(1), u = new Uint32Array(f.buffer);
    const x = Array.from(data.initialPositions_m!, value => {
      const r = rnd(); f[0] = value;
      if (f[0] !== 0) { if (r < 1 / 3) u[0]! += 1; else if (r < 2 / 3) u[0]! -= 1; }
      return f[0]!;
    });
    data = { ...data, initialPositions_m: x };
  }
  scene.cloth = [{ ...d, geometry: { ...d.geometry, data }, solver: { ...d.solver, newton: { ...base, ...newton, contactLinearization: LIN as 'off' } } }];
  return scene;
}
async function step(cloth: ClothSystem): Promise<void> {
  if (!cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S)) throw new Error(`step refused at ${cloth.time_s} s: ${cloth.failure?.message ?? 'pending'}`);
  await device.queue.onSubmittedWorkDone();
}

if (MODE === 'iterates') {
  const base = saved.document.solver!.newton!, configs: [number, number][] = [];
  for (const pcg of [1, 2, 5, 10, 20, 50, 100]) configs.push([1, pcg]);
  for (let k = 1; k <= base.iterations; k++) configs.push([k, base.pcgIterations]);
  const run: IterateRun = { tick: TICK, lin: LIN, perturb: PERTURB, vertices: N, configs: [] }, x = new Float32Array(3 * N * configs.length);
  for (const [k, [newton, pcg]] of configs.entries()) {
    const cloth = await ClothSystem.create(device, restored({ iterations: newton, pcgIterations: pcg }), { statusInterval: 1 });
    await step(cloth);
    const solver = cloth.solverFor(documentId)!, contact = cloth.contactFor(documentId)!;
    const ctl = await solver.readNewtonControl(), c = await contact.readCounters(), st = await solver.readState();
    if (cloth.failure) throw new Error(`cap ${newton}/${pcg}: ${cloth.failure.message}`);
    x.set(st.positions_m, 3 * N * k);
    run.configs.push({ newton, pcg, newtonUsed: ctl[3]!, pcgUsed: ctl[7]!, g2: ctl[5]!, alpha: ctl[2]!, candidates: c.candidates, pairs: c.vfPairs + c.eePairs, rounds: c.detections + c.redetections, solidContacts: c.solidContacts });
    console.log(`  cap ${newton}/${pcg}: Newton ${ctl[3]}, PCG ${ctl[7]}, g2 ${ctl[5]!.toExponential(5)}, alpha ${ctl[2]}, pairs ${c.vfPairs + c.eePairs}, candidates ${c.candidates}, rounds ${c.detections + c.redetections}`);
    cloth.destroy();
  }
  writeFileSync(`${OUT}.bin`, new Uint8Array(x.buffer)); writeFileSync(`${OUT}.json`, JSON.stringify(run));
}

if (MODE === 'quality') {
  const d = saved.document, tris = d.geometry.data.triangles, rest = d.geometry.data.restPositions_m, mass = d.geometry.data.vertexMass_kg!;
  const t = bridson2002SceneLayout(source).translation_m, limit = d.materials[0]!.strainLimit ?? 0;
  const seen = new Set<number>(), edges: number[] = [];
  for (let f = 0; f < tris.length; f += 3) for (let k = 0; k < 3; k++) {
    const a = tris[f + k]!, b = tris[f + (k + 1) % 3]!, key = Math.min(a, b) * N + Math.max(a, b);
    if (!seen.has(key)) { seen.add(key); edges.push(a, b); }
  }
  const restLength = (a: number, b: number) => Math.hypot(rest[3 * a]! - rest[3 * b]!, rest[3 * a + 1]! - rest[3 * b + 1]!, rest[3 * a + 2]! - rest[3 * b + 2]!);
  let mg = 0; for (let i = 0; i < N; i++) mg += (9.81 * mass[i]!) ** 2; mg = Math.sqrt(mg);
  const cloth = await ClothSystem.create(device, restored({}), { statusInterval: 1 });
  const solver = cloth.solverFor(documentId)!, contact = cloth.contactFor(documentId)!;
  const rows: Record<string, number | string>[] = [];
  let halt = '';
  console.log(`quality: ${NODES}^2 from tick ${TICK}, ${STEPS} steps, contactLinearization ${LIN}${PERTURB ? `, positions perturbed by an ulp (seed ${PERTURB})` : ''}; start time ${cloth.time_s.toFixed(4)} s; strain limit ${limit}; |m g| ${mg.toExponential(3)}`);
  for (let s = 1; s <= STEPS; s++) {
    // A failing status halts the system: the capture throws. The run ends there, with the ticks it completed.
    try { await step(cloth); await cloth.capture([cloth.time_s]); }
    catch (e) { halt = `tick ${TICK + s}: ${String(e).slice(0, 200)}`; console.log(`  HALT ${halt}`); break; }
    const ctl = await solver.readNewtonControl(), c = await contact.readCounters(), x = (await solver.readState()).positions_m;
    const { intersections, solidIntersections } = await contact.countIntersections();
    const lin = await solver.readContactLinearization();
    // Positions are in the cloth's frame: the scene frame less the document's origin, which the capture restores.
    const pose = (await cloth.capture([cloth.time_s]))[0]![0]!.positions_m, time = cloth.time_s;
    const dist = (q: readonly number[]) => { let m = Infinity; for (const sd of source.solids) m = Math.min(m, bridsonSolidDistance(sd, { x: q[0]! - t.x, y: q[1]! - t.y, z: q[2]! - t.z }, time)); return m; };
    const P = (i: number) => [pose[3 * i]!, pose[3 * i + 1]!, pose[3 * i + 2]!];
    let least = Infinity;
    for (let i = 0; i < N; i++) least = Math.min(least, dist(P(i)));
    for (let f = 0; f < tris.length; f += 3) {
      const [A, B, C] = [P(tris[f]!), P(tris[f + 1]!), P(tris[f + 2]!)];
      for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4 - i; j++) {
        if ((i === 0 && (j === 0 || j === 4)) || (i === 4 && j === 0)) continue; // the corners are the vertices
        const u = i / 4, w = j / 4;
        least = Math.min(least, dist([0, 1, 2].map(k => (1 - u - w) * A[k]! + u * B[k]! + w * C[k]!)));
      }
    }
    let strain = 0, over = 0;
    for (let e = 0; e < edges.length; e += 2) {
      const a = edges[e]!, b = edges[e + 1]!, sv = Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / restLength(a, b) - 1;
      strain = Math.max(strain, sv); if (limit > 0 && sv > 1.05 * limit) over++;
    }
    const row = { tick: TICK + s, newton: ctl[3]!, pcg: ctl[7]!, g: Math.sqrt(ctl[5]!), gOverMg: Math.sqrt(ctl[5]!) / mg, alpha: ctl[2]!, travelHeld: ctl[11]!, intersections, solidIntersections, leastDistance: least,
      maxStrain: strain, edgesPastLimit: over, held: c.held, planeRecordHeld: c.planeRecordHeld, planeRecordPeak: c.framePeakPlaneRecords, rounds: c.detections + c.redetections, pairs: c.vfPairs + c.eePairs, solidContacts: c.solidContacts,
      candidates: c.candidates, capped: c.capped, uncertified: c.uncertified, status: c.status, statusText: [OgcContactDetector.describeStatus(c.status)].flat().join(','), fallbackRows: lin?.fallbackRows ?? 0, linRecords: lin?.demandRecords ?? 0 };
    rows.push(row);
    console.log(`  tick ${row.tick}: newton ${row.newton} pcg ${row.pcg} |g| ${row.g.toExponential(2)} (${row.gOverMg.toExponential(2)} |mg|) alpha ${+row.alpha.toPrecision(3)} x ${intersections}/${solidIntersections} least ${least.toExponential(3)} m strain ${strain.toFixed(4)} (${over} past) `
      + `held ${c.held} travel-held ${row.travelHeld} overflow-held ${c.planeRecordHeld} records ${c.framePeakPlaneRecords} rounds ${row.rounds} pairs ${row.pairs} solid ${c.solidContacts} status ${row.statusText || 0} fallback ${row.fallbackRows}`);
    if (cloth.failure) { halt = `tick ${row.tick}: ${cloth.failure.message}`; console.log(`  HALT ${halt}`); break; }
  }
  cloth.destroy();
  writeFileSync(`${OUT}.json`, JSON.stringify({ nodes: NODES, tick: TICK, lin: LIN, perturb: PERTURB, mg, limit, halt, rows }, null, 1));
}
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`${errors.length} WebGPU errors`); process.exitCode = 1; }
device.destroy();
