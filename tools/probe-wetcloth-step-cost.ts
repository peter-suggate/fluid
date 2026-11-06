/** Per-step cost of the dry cloth host's mechanics on a released fixture, split
 * into the GPU Float32 solve and the CPU Float64 refinement it falls back to.
 * Same owner, inputs and dt as the app's dry host (lib/cloth/dry-host.ts).
 *   tools/gpu/fluid-gpu-run.sh check npx tsx tools/probe-wetcloth-step-cost.ts [--scene=ID] [--steps=N]
 * --synthetic=drape|fold [--n=180] [--dt=2e-4] [--contact-capacity=PAIRS]: the GPU Newton solve alone on an n x n sheet
 * (default 32400 vertices) that is in contact within a few steps: dropped onto a sphere, or
 * folded with its upper layer thrown onto the fixed lower one (self contact). No scene is edited.
 */
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {prepareWetClothSource, buildWetClothSourceScene} from '../lib/cloth/scenes/definitions';
import {ClothRuntime} from '../lib/cloth/runtime';
import {loadClothTopology} from '../lib/cloth/source-assets';
import {clothContainerColliders, clothSceneStaticColliders} from '../lib/cloth/scene-contact';
import {composeClothFeatures} from '../lib/cloth/definition';
import {withUniformDevice} from '../tests/helpers/uniform-geometric';
import {sceneWithSolidStroke} from '../lib/core/solid-world';
import {ClothContactSet} from '../lib/cloth/contact-constraints';
import type {ClothStepOptions} from '../lib/cloth/dynamics';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology} from '../lib/cloth/topology';
import type {ClothCollider} from '../lib/cloth/contact';
import '../lib/methods';

const option = (key: string) => process.argv.find(v => v.startsWith(`--${key}=`))?.slice(key.length + 3);
const id = option('scene') ?? 'wetcloth-source-simple-cloth', steps = Number(option('steps') ?? 60);
const request: typeof fetch = async input => new Response(await readFile(resolve('public' + String(input))));

// --cpu: the Float64 CPU owner only (no device); --solver=gauss-seidel|newton-pcg.
const cpu = process.argv.includes('--cpu'), solver = (option('solver') ?? 'newton-pcg') as 'newton-pcg' | 'gauss-seidel';
const measure = async (device?: GPUDevice) => {
  composeClothFeatures();
  await prepareWetClothSource(id, request);
  // --cut: the one-cell pen erase the app's browser check makes on the lower ribbon.
  const built = buildWetClothSourceScene(id), scene = process.argv.includes('--cut')
    ? sceneWithSolidStroke(built, [{operation: 'clear', minimum: [15, 7, 14], maximumExclusive: [16, 8, 15], affectsScenery: true}])
    : built, document = scene.cloth![0]!;
  const runtime = new ClothRuntime(document, await loadClothTopology(document.geometry), {
    sourceFields: scene.wetClothSource?.fields, excludedSourceColliderIndices: scene.wetClothSource?.bakedSolidFieldIndices,
    sourceOffset_m: scene.wetClothSource?.sourceOffset_m, nonlinearSolver: solver});
  if (device) await runtime.initializeGPU(device);
  // --cpu-detect: GPU solve with the Float64 contact set detecting between solves.
  const gpuSolver = (runtime as unknown as {gpu?: {contactDetection?: 'auto' | 'cpu'; detectContacts?: (state: unknown, dt: number, options: unknown) => Promise<{contacts: unknown[]; detect_ms: number; errors: number}>}}).gpu;
  if (gpuSolver && process.argv.includes('--cpu-detect')) gpuSolver.contactDetection = 'cpu';
  const detection: Record<string, number> = {}, fallbacks = new Set<string>(), refineStatus: Record<number, number> = {};
  const g = scene.fluid.gravity_m_s2, colliders = [...clothContainerColliders(scene), ...clothSceneStaticColliders(scene)];
  const dt = scene.numerics.maxDt_s, rows: {ms: number; gpu: number; cpu: number; linear: number; solve: number; finish: number; contact: number; solves: number; submissions: number; outer: number}[] = [];
  for (let i = 0; i < steps; i++) {
    const t0 = performance.now();
    let report;
    try {
      report = await runtime.stepAsync(dt, colliders, {gravity_m_s2: [g.x, g.y, g.z], ...(option('iterations') ? {iterations: Number(option('iterations'))} : {})}) as {iterations: number; gpuIterations?: number; refinementIterations?: number; linearIterations?: number; solve_ms?: number; finish_ms?: number; contact_ms?: number; gpuSolves?: number; submissions?: number; contactIterations?: number; contactDetection?: string; contactFallback?: string; gpuStatus?: number};
    } catch (error) {
      console.log(JSON.stringify({id, failedAtStep: i, error: String(error).slice(0, 160)}));
      break;
    }
    rows.push({ms: performance.now() - t0, gpu: report.gpuIterations ?? 0, cpu: device ? report.refinementIterations ?? 0 : report.iterations, linear: report.linearIterations ?? 0, solve: report.solve_ms ?? NaN, finish: report.finish_ms ?? NaN, contact: report.contact_ms ?? NaN, solves: report.gpuSolves ?? NaN, submissions: report.submissions ?? NaN, outer: report.contactIterations ?? 0});
    if (report.contactDetection) detection[report.contactDetection] = (detection[report.contactDetection] ?? 0) + 1;
    if (report.contactFallback) fallbacks.add(report.contactFallback);
    if (report.refinementIterations) refineStatus[report.gpuStatus ?? -1] = (refineStatus[report.gpuStatus ?? -1] ?? 0) + 1;
  }
  const mean = (f: (r: typeof rows[0]) => number, from = 0) => rows.slice(from).reduce((s, r) => s + f(r), 0) / Math.max(1, rows.length - from);
  const refined = rows.filter(r => r.cpu > 0);
  if (!device) {
    console.log(JSON.stringify({id, owner: 'cpu', solver, cut: process.argv.includes('--cut'), dt_s: dt, completedSteps: rows.length, vertices: runtime.prepared.vertexCount,
      meanStep_ms: +mean(r => r.ms, 5).toFixed(2), maxStep_ms: +Math.max(...rows.map(r => r.ms)).toFixed(2),
      meanIterations: +mean(r => r.cpu).toFixed(1), maxIterations: Math.max(...rows.map(r => r.cpu)), meanLinearIterations: +mean(r => r.linear).toFixed(1)}));
    return;
  }
  console.log(JSON.stringify({id, dt_s: dt, steps, vertices: document.geometry.kind === 'inline' ? document.geometry.data.restPositions_m.length / 3 : undefined,
    solver, vertexCount: runtime.prepared.vertexCount, meanStep_ms: +mean(r => r.ms, 5).toFixed(2), maxStep_ms: +Math.max(...rows.map(r => r.ms)).toFixed(2),
    meanGpuIterations: +mean(r => r.gpu).toFixed(1), meanLinearIterations: +mean(r => r.linear).toFixed(1),
    // GPU Newton only: solve (upload, submissions, readbacks), CPU contact work
    // between solves (detection, multipliers), Float64 finish (report, refinement).
    meanSolve_ms: +mean(r => r.solve, 5).toFixed(2), meanContact_ms: +mean(r => r.contact, 5).toFixed(2), meanFinish_ms: +mean(r => r.finish, 5).toFixed(2),
    meanGpuSolves: +mean(r => r.solves).toFixed(2), meanSubmissions: +mean(r => r.submissions).toFixed(1),
    refinedSteps: refined.length, meanRefinementIterations: +(refined.reduce((s, r) => s + r.cpu, 0) / Math.max(1, refined.length)).toFixed(1),
    meanRefinedStep_ms: +(refined.reduce((s, r) => s + r.ms, 0) / Math.max(1, refined.length)).toFixed(2),
    meanUnrefinedStep_ms: +(rows.filter(r => !r.cpu).reduce((s, r) => s + r.ms, 0) / Math.max(1, rows.length - refined.length)).toFixed(2),
    meanContactIterations: +mean(r => r.outer).toFixed(2), detection, fallbacks: [...fallbacks], refinedByGpuStatus: refineStatus, ...await detectionCost(),
    simSecondsPerWallSecond: +(dt / (mean(r => r.ms, 5) / 1000)).toFixed(4)}));
  // One GPU detection pass at the final state (the first detection of a step; each outer round repeats it).
  async function detectionCost(): Promise<object> {
    if (!gpuSolver?.detectContacts || gpuSolver.contactDetection === 'cpu' || !detection.gpu) return {};
    const options = (runtime as unknown as {prepareStep(dt: number, c: unknown, o: unknown): unknown}).prepareStep(dt, colliders, {gravity_m_s2: [g.x, g.y, g.z]}), times: number[] = [];
    let keys: string[] = [], errors = 0;
    for (let k = 0; k < 3; k++) {const r = await gpuSolver.detectContacts(runtime.state, dt, options); times.push(r.detect_ms); keys = (r.contacts as {key: string}[]).map(c => c.key); errors = r.errors;}
    // The Float64 set's first detection at the same prediction, for parity.
    const o = options as ClothStepOptions, state = runtime.state, top = runtime.prepared, x = state.positions_m.slice(), target = o.supportPositions_m ?? top.data.restPositions_m, gravity = o.gravity_m_s2 ?? [0, -9.81, 0];
    for (let i = 0; i < top.vertexCount; i++) {
      const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]!*top.materials[top.vertexMaterial[i]!]!.waterDensity_kg_m3;
      for (let a = 0; a < 3; a++) {const k = 3*i + a; x[k] = (top.kinematicFixed[i] ?? 0) & 1 ? target[k]! : x[k]! + dt*state.velocities_m_s[k]! + dt*dt*(gravity[a]! + (o.externalForces_N?.[k] ?? 0)/mass);}
    }
    const t0 = performance.now(), reference = new ClothContactSet(top, state.positions_m.slice(), dt, o.colliders ?? [], o.selfCollision ?? false, o.tolerance_m ?? 1e-8);
    reference.detect(x);
    const cpuDetect_ms = performance.now() - t0, cpuKeys = new Set(reference.constraints.map(c => c.key)), gpuKeys = new Set(keys);
    return {detect_ms: +times.sort((a, b) => a - b)[1]!.toFixed(2), cpuDetect_ms: +cpuDetect_ms.toFixed(1), detectedPairs: keys.length, detectErrors: errors,
      onlyGpu: keys.filter(k => !cpuKeys.has(k)).slice(0, 8), onlyCpu: [...cpuKeys].filter(k => !gpuKeys.has(k)).slice(0, 8), cpuPairs: cpuKeys.size};
  }
};
/** A sheet in contact within a few steps (see the header); per-step GPU cost and contact counts. */
const synthetic = async (device: GPUDevice, kind: string) => {
  const n = Number(option('n') ?? 180), dt = Number(option('dt') ?? 2e-4), restPositions_m: number[] = [], start: number[] = [], triangles: number[] = [], fixed: number[] = [];
  const thickness = DEFAULT_CLOTH_MATERIAL.thickness_m, colliders: ClothCollider[] = [];
  let h: number, velocity: (i: number, j: number) => number;
  if (kind === 'drape') {
    // A 0.45 m sheet 0.2 mm above the top of a 0.15 m sphere, falling at 0.5 m/s.
    h = .45/(n - 1); const R = .15;
    colliders.push({kind: 'sphere', center_m: [.225, 1 - R - .5*thickness - 2e-4, .225], radius_m: R});
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*h, 1, j*h); fixed.push(0);}
    start.push(...restPositions_m); velocity = () => -.5;
  } else if (kind === 'fold') {
    // Folded at mid-length around a 1 mm crease; the lower half fixed, the upper half thrown down at 1 m/s.
    h = .002; const R = .001, crease = (n/2 - 1)*h;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = i*h, z = j*h, s = x - crease; restPositions_m.push(x, 1, z); fixed.push(s <= 0 ? 1 : 0);
      if (s <= 0) start.push(x, 1, z); else if (s <= Math.PI*R) start.push(crease + R*Math.sin(s/R), 1 + R*(1 - Math.cos(s/R)), z); else start.push(crease - (s - Math.PI*R), 1 + 2*R, z);
    }
    velocity = i => i*h - crease > Math.PI*R ? -1 : 0;
  } else throw new Error(`--synthetic=${kind}: expected drape or fold`);
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, c, b, b, c, d);}
  const topology = prepareClothTopology({restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed}, [DEFAULT_CLOTH_MATERIAL]), state = createClothState(topology);
  state.positions_m.set(start);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) state.velocities_m_s[3*(j*n + i) + 1] = velocity(i, j);
  const options: ClothStepOptions = {iterations: 64, tolerance_m: 1e-8, selfCollision: kind === 'fold', colliders, supportPositions_m: Float64Array.from(start)};
  const solver = new GPUClothNewton(device, topology, {contactCapacity: Number(option('contact-capacity') ?? 0)}), rows: {ms: number; newton: number; refine: number; outer: number; contacts: number; solve: number; finish: number; detection: string; fallback?: string; status: number; attempts: number}[] = [];
  try {
    await solver.initialize();
    for (let i = 0; i < steps; i++) {
      const t0 = performance.now(), r = await solver.step(state, dt, options) as unknown as {gpuIterations: number; refinementIterations: number; contactIterations?: number; contacts?: {contacts: number; selfContacts: number}; solve_ms: number; finish_ms: number; contactDetection: string; contactFallback?: string; gpuStatus: number; contactAttempts?: number};
      rows.push({ms: performance.now() - t0, newton: r.gpuIterations, refine: r.refinementIterations, outer: r.contactIterations ?? 0, contacts: (r.contacts?.contacts ?? 0) + (r.contacts?.selfContacts ?? 0), solve: r.solve_ms, finish: r.finish_ms, detection: r.contactDetection, fallback: r.contactFallback, status: r.gpuStatus, attempts: r.contactAttempts ?? 1});
    }
  } finally {solver.destroy();}
  const tail = rows.slice(5), mean = (f: (r: typeof rows[0]) => number, from = tail) => +(from.reduce((s, r) => s + f(r), 0)/Math.max(1, from.length)).toFixed(2);
  const contact = tail.filter(r => r.contacts > 0), count = (f: (r: typeof rows[0]) => string | undefined) => rows.reduce((m, r) => {const k = f(r); if (k !== undefined) m[k] = (m[k] ?? 0) + 1; return m;}, {} as Record<string, number>);
  console.log(JSON.stringify({synthetic: kind, vertices: topology.vertexCount, dt_s: dt, steps: rows.length, meanStep_ms: mean(r => r.ms), maxStep_ms: +Math.max(...rows.map(r => r.ms)).toFixed(1),
    meanSolve_ms: mean(r => r.solve), meanFinish_ms: mean(r => r.finish), meanNewton: mean(r => r.newton), meanOuter: mean(r => r.outer), contactSteps: contact.length,
    meanContactStep_ms: mean(r => r.ms, contact), meanContactNewton: mean(r => r.newton, contact), meanContactOuter: mean(r => r.outer, contact), maxContacts: Math.max(...rows.map(r => r.contacts)),
    refinedSteps: rows.filter(r => r.refine > 0).length, meanRefinement: mean(r => r.refine, rows.filter(r => r.refine > 0)), detection: count(r => r.detection), fallbacks: count(r => r.fallback),
    refinedByGpuStatus: count(r => r.refine > 0 ? String(r.status) : undefined), redoneSteps: rows.map((r, k) => [k, r.attempts]).filter(([, attempts]) => attempts! > 1), perStep: rows.map(r => [+r.ms.toFixed(1), r.newton, r.outer, r.contacts, r.refine])}));
};
if (option('synthetic')) await withUniformDevice('wetcloth synthetic contact', device => synthetic(device, option('synthetic')!));
else if (cpu) await measure(); else await withUniformDevice('wetcloth step cost', measure);
