/** Per-step cost of the dry cloth host's mechanics on a released fixture, split
 * into the GPU Float32 solve and the CPU Float64 refinement it falls back to.
 * Same owner, inputs and dt as the app's dry host (lib/cloth/dry-host.ts).
 *   tools/gpu/fluid-gpu-run.sh check npx tsx tools/probe-wetcloth-step-cost.ts [--scene=ID] [--steps=N]
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
  const g = scene.fluid.gravity_m_s2, colliders = [...clothContainerColliders(scene), ...clothSceneStaticColliders(scene)];
  const dt = scene.numerics.maxDt_s, rows: {ms: number; gpu: number; cpu: number; linear: number; solve: number; finish: number; submissions: number}[] = [];
  for (let i = 0; i < steps; i++) {
    const t0 = performance.now();
    let report;
    try {
      report = await runtime.stepAsync(dt, colliders, {gravity_m_s2: [g.x, g.y, g.z], ...(option('iterations') ? {iterations: Number(option('iterations'))} : {})}) as {iterations: number; gpuIterations?: number; refinementIterations?: number; linearIterations?: number; solve_ms?: number; finish_ms?: number; submissions?: number};
    } catch (error) {
      console.log(JSON.stringify({id, failedAtStep: i, error: String(error).slice(0, 160)}));
      break;
    }
    rows.push({ms: performance.now() - t0, gpu: report.gpuIterations ?? 0, cpu: device ? report.refinementIterations ?? 0 : report.iterations, linear: report.linearIterations ?? 0, solve: report.solve_ms ?? NaN, finish: report.finish_ms ?? NaN, submissions: report.submissions ?? NaN});
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
    // GPU Newton only: solve (upload, submissions, readbacks) vs Float64 finish (contact, refinement).
    meanSolve_ms: +mean(r => r.solve, 5).toFixed(2), meanFinish_ms: +mean(r => r.finish, 5).toFixed(2), meanSubmissions: +mean(r => r.submissions).toFixed(1),
    refinedSteps: refined.length, meanRefinementIterations: +(refined.reduce((s, r) => s + r.cpu, 0) / Math.max(1, refined.length)).toFixed(1),
    meanRefinedStep_ms: +(refined.reduce((s, r) => s + r.ms, 0) / Math.max(1, refined.length)).toFixed(2),
    meanUnrefinedStep_ms: +(rows.filter(r => !r.cpu).reduce((s, r) => s + r.ms, 0) / Math.max(1, rows.length - refined.length)).toFixed(2),
    simSecondsPerWallSecond: +(dt / (mean(r => r.ms, 5) / 1000)).toFixed(4)}));
};
if (cpu) await measure(); else await withUniformDevice('wetcloth step cost', measure);
