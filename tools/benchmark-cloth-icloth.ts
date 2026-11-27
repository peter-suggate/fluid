/**
 * One cloth scene, one method, its whole authored run through ClothSystem (one step per 1/60 s frame): the baseline
 * matrix for method 'icloth' (I-Cloth: proximity terms in the solve, then CCD + impact-zone rounds) against 'newton'.
 *
 * Reports: fenced ms/step over the whole run (mean, median, max; the stage-timed steps are left out of it) and per
 * fifth of the run; the stage split (timestamped passes, sampled); dispatches encoded per step by stage and, for
 * I-Cloth's collision stages, those that did work (derived from the step's counters); Newton and PCG iterations;
 * I-Cloth's rounds, impacts, impact-zone iterations and list peaks; at checkpoints and at the end the pose's
 * self-intersections, vertices inside solids and strain tail (Float64, CPU); the notes and failure; and for the
 * analytic scenes the quantities tests/cloth-analytic.test.ts gates, with their pass/fail under this method.
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh timing \
 *     node --import tsx tools/benchmark-cloth-icloth.ts --scene <scene> [--method icloth|newton] [--newton 4x50]
 *       [--steps N] [--nodes 150] [--checkpoints 6] [--max-wall-s 240] [--tag name] [--out file.json]
 *   scenes: curtain, sphere, tablecloth (Bridson 2002); free-fall, hanging-curtain, swinging-sheet, peirce-cantilever,
 *   friction-incline (analytic). --steps defaults to the scene's authored length; --nodes resamples the sphere's sheet.
 */
import { writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import type { SceneDescription } from '../lib/core/model';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { clothStageOf } from '../lib/cloth/stage-timing';
import { prepareClothTopology } from '../lib/cloth/topology';
import { buildClothBodyField, buildClothSceneField, clothBodyPoses, clothContactRadius, clothSceneMovingSolids } from '../lib/cloth/scene-contact';
import { sampleClothSolidField } from '../lib/cloth/contact/solid-field';
import { bridson2002Curtain, bridson2002RotatingSphere, bridson2002Tablecloth } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import {
  GRAVITY, INCLINE, PEIRCE, SWING, bdf2FreeFall, clothAnalyticSceneDescription, hangingBandStrain,
  clothAnalyticFreeFall, clothAnalyticFrictionIncline, clothAnalyticHangingCurtain, clothAnalyticPeirceCantilever, clothAnalyticSwingingSheet,
  type ClothAnalyticScene,
} from '../lib/cloth/scenes/analytic';
import { iclothSelfIntersections } from '../lib/cloth/icloth/oracle';
import type { ClothDocument, ClothMethod, ClothNewtonSettings } from '../lib/cloth/types';

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1]! : fallback; };
const SCENE = opt('scene', 'sphere'), METHOD = opt('method', 'icloth') as ClothMethod, NODES = Number(opt('nodes', '150'));
const BUDGET = opt('newton', ''), CHECKPOINTS = Number(opt('checkpoints', '6')), MAX_WALL_S = Number(opt('max-wall-s', '240')), TAG = opt('tag', '');
const budget = BUDGET ? BUDGET.split('x').map(Number) : undefined;
// --icloth RxK: I-Cloth's caps (collision rounds x impact-zone iterations) in place of the document's.
const ICLOTH = opt('icloth', ''), iclothCaps = ICLOTH ? ICLOTH.split('x').map(Number) : undefined;
if (iclothCaps && !(iclothCaps.length === 2 && iclothCaps.every(c => Number.isInteger(c) && c >= 1))) throw new Error(`--icloth takes <rounds>x<iterations>, got ${ICLOTH}`);
if (budget && !(budget.length === 2 && budget.every(c => Number.isInteger(c) && c >= 1))) throw new Error(`--newton takes <newton>x<pcg>, got ${BUDGET}`);
const g = GRAVITY.value, h = CLOTH_SCENE_STEP_S;
const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const sum = (a: readonly number[]) => a.reduce((x, y) => x + y, 0), mean = (a: readonly number[]) => a.length ? sum(a) / a.length : NaN;
const median = (a: readonly number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? (s[(s.length - 1) >> 1]! + s[s.length >> 1]!) / 2 : NaN; };

// ---------------------------------------------------------------- the scene
const withSolver = (scene: SceneDescription, newton?: (own: ClothNewtonSettings) => ClothNewtonSettings): SceneDescription => {
  (scene as { cloth?: ClothDocument[] }).cloth = scene.cloth!.map(d => ({ ...d, solver: { ...d.solver, method: METHOD, ...(iclothCaps ? { icloth: { rounds: iclothCaps[0]!, impactIterations: iclothCaps[1]! } } : {}), ...(newton && d.solver?.newton ? { newton: newton(d.solver.newton) } : {}) } }));
  return scene;
};
const capped = budget ? (own: ClothNewtonSettings) => ({ ...own, iterations: budget[0]!, pcgIterations: budget[1]! }) : undefined;
const BRIDSON = { curtain: () => bridson2002Curtain(), sphere: () => bridson2002RotatingSphere({ nodesAcross: NODES }), tablecloth: () => bridson2002Tablecloth() } as const;
const ANALYTIC = { 'free-fall': clothAnalyticFreeFall, 'hanging-curtain': clothAnalyticHangingCurtain, 'swinging-sheet': clothAnalyticSwingingSheet,
  'peirce-cantilever': clothAnalyticPeirceCantilever, 'friction-incline': clothAnalyticFrictionIncline } as const;
let analytic: ClothAnalyticScene | undefined, sceneId: string, duration_s: number;
const buildScene = (newton = capped): SceneDescription => {
  if (SCENE in BRIDSON) { const s = BRIDSON[SCENE as keyof typeof BRIDSON](); sceneId = s.id; duration_s = s.duration_s; return withSolver(bridson2002SceneDescription(s), newton); }
  if (SCENE in ANALYTIC) { const s = analytic = ANALYTIC[SCENE as keyof typeof ANALYTIC](); sceneId = s.id; duration_s = s.duration_s; return withSolver(clothAnalyticSceneDescription(s), newton); }
  throw new Error(`Unknown scene '${SCENE}': one of ${[...Object.keys(BRIDSON), ...Object.keys(ANALYTIC)].join(', ')}`);
};
const scene = buildScene();
const STEPS = Number(opt('steps', String(Math.round(duration_s! / h))));
if (!(STEPS >= 1 && STEPS <= 800)) throw new Error('--steps must be in 1..800');
const name = `${SCENE}${SCENE === 'sphere' && NODES !== 150 ? `-${NODES}` : ''}-${METHOD}-${BUDGET || 'doc'}${ICLOTH ? `-ic${ICLOTH}` : ''}${TAG ? `-${TAG}` : ''}`;
const OUT = opt('out', join(homedir(), `code/me/fluid-scratch/icloth/matrix/${name}.json`));

// ---------------------------------------------------------------- the device, with every dispatch counted by stage
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
if (!adapter) throw new Error('no WebGPU adapter');
const features = (['timestamp-query', 'subgroups'] as GPUFeatureName[]).filter(f => adapter.features.has(f));
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = [];
device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
const dispatched = new Map<string, number>();
let counting = true;
{
  const create = device.createCommandEncoder.bind(device);
  try {
    device.createCommandEncoder = (d?: GPUCommandEncoderDescriptor) => {
      const encoder = create(d), begin = encoder.beginComputePass.bind(encoder);
      encoder.beginComputePass = (pd?: GPUComputePassDescriptor) => {
        const pass = begin(pd), setPipeline = pass.setPipeline.bind(pass), direct = pass.dispatchWorkgroups.bind(pass), indirect = pass.dispatchWorkgroupsIndirect.bind(pass);
        let stage = 'other';
        const bump = () => dispatched.set(stage, (dispatched.get(stage) ?? 0) + 1);
        pass.setPipeline = p => { stage = clothStageOf(p.label); setPipeline(p); };
        pass.dispatchWorkgroups = (...a) => { bump(); direct(...a); };
        pass.dispatchWorkgroupsIndirect = (...a) => { bump(); indirect(...a); };
        return pass;
      };
      return encoder;
    };
    device.createCommandEncoder().finish();
  } catch { counting = false; }
}

// ---------------------------------------------------------------- pose measures (Float64, CPU)
const rotateInverse = (q: readonly [number, number, number, number], v: [number, number, number]): [number, number, number] => {
  const [x, y, z, w] = [-q[0], -q[1], -q[2], q[3]];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};
/** A document's solids as the methods collide with them: vertices inside the static field and the moving solid at time t. */
function solidProbe(doc: ClothDocument) {
  const r = clothContactRadius(doc), X = doc.geometry.data.restPositions_m;
  const origin = [0, 1, 2].map(a => { let lo = Infinity, hi = -Infinity; for (let i = a; i < X.length; i += 3) { lo = Math.min(lo, X[i]!); hi = Math.max(hi, X[i]!); } return (lo + hi) / 2; }) as [number, number, number];
  const field = buildClothSceneField(scene, origin, r), body = clothSceneMovingSolids(scene)[0], bodyField = body && buildClothBodyField(scene, body, r);
  return (x: Float64Array, t: number) => {
    const pose = body && clothBodyPoses(body, origin, t, 0, CLOTH_SCENE_STEP_S)[0]!;
    let fixed = 0, moving = 0, least = Infinity;
    for (let v = 0; v < x.length / 3; v++) {
      const p: [number, number, number] = [x[3 * v]! - origin[0], x[3 * v + 1]! - origin[1], x[3 * v + 2]! - origin[2]];
      const a = sampleClothSolidField(field, p).phi;
      if (a < 0) fixed++;
      least = Math.min(least, a);
      if (pose && bodyField) {
        const b = sampleClothSolidField(bodyField, rotateInverse(pose.orientation, [p[0] - pose.position[0], p[1] - pose.position[1], p[2] - pose.position[2]])).phi;
        if (b < 0) moving++;
        least = Math.min(least, b);
      }
    }
    return { fixed, moving, least };
  };
}
/** Edge strain tail: edges stretched over 10.1% and over 15%, the largest stretch and the largest compression. */
function strainTail(x: ArrayLike<number>, edges: ArrayLike<number>, rest: ArrayLike<number>) {
  const stride = edges.length / rest.length;
  let over10 = 0, over15 = 0, max = 0, min = 0;
  for (let e = 0; e < rest.length; e++) {
    const a = edges[stride * e]!, b = edges[stride * e + 1]!;
    const s = Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / rest[e]! - 1;
    if (s > 0.101) over10++;
    if (s > 0.15) over15++;
    max = Math.max(max, s); min = Math.min(min, s);
  }
  return { over10, over15, max, min };
}

// ---------------------------------------------------------------- the analytic gates (tests/cloth-analytic.test.ts's, same thresholds)
interface Gate { readonly name: string; readonly pass: boolean; readonly detail: string }
interface TickState { readonly positions_m: Float64Array; readonly velocities_m_s: Float64Array }
interface GateRun { readonly wants: (tick: number) => boolean; readonly tick: (tick: number, docs: readonly TickState[]) => void; readonly finish: () => Promise<Gate[]> | Gate[] }
const fmt = (v: number, d = 3) => Number.isFinite(v) ? v.toFixed(d) : String(v);

function ringdown(t: readonly number[], series: readonly (readonly number[])[]) {
  const basis = (a: number, w: number) => t.map(s => [1, Math.exp(-a * s) * Math.cos(w * s), Math.exp(-a * s) * Math.sin(w * s)] as const);
  const fit = (B: readonly (readonly [number, number, number])[], y: readonly number[]) => {
    const M = [0, 1, 2].map(i => [0, 1, 2].map(j => B.reduce((q, r) => q + r[i]! * r[j]!, 0))), v = [0, 1, 2].map(i => B.reduce((q, r, k) => q + r[i]! * y[k]!, 0));
    for (let i = 0; i < 3; i++) for (let r = 0; r < 3; r++) if (r !== i) { const f = M[r]![i]! / M[i]![i]!; for (let c = i; c < 3; c++) M[r]![c]! -= f * M[i]![c]!; v[r]! -= f * v[i]!; }
    const c = [v[0]! / M[0]![0]!, v[1]! / M[1]![1]!, v[2]! / M[2]![2]!];
    return { c, res: B.reduce((q, r, k) => q + (r[0] * c[0]! + r[1] * c[1]! + r[2] * c[2]! - y[k]!) ** 2, 0) };
  };
  let best = { res: Infinity, a: 0, w: 0, c: [0, 0, 0] };
  for (let w = 2; w <= 60; w += 0.01) for (let a = 0; a <= 2; a += 0.025) { const f = fit(basis(a, w), series[0]!); if (f.res < best.res) best = { ...f, a, w }; }
  const B = basis(best.a, best.w);
  return { rest: series.map(y => fit(B, y).c[0]!), w: best.w, a: best.a, amplitude: Math.hypot(best.c[1]!, best.c[2]!), rms: Math.sqrt(best.res / t.length) };
}

function gatesFor(source: ClothAnalyticScene): GateRun {
  const doc0 = source.cloth[0]!, data = doc0.geometry.data, rest = data.restPositions_m, tri = data.triangles;
  switch (source.kind) {
    case 'free-fall': {
      const n = source.probe.gateSteps as number, y0 = source.probe.y0 as number, ref = bdf2FreeFall(n, h, -g);
      let centroid = 0, velocity = 0, flat = 0, strain = 0, euler = 0;
      const track: { y: number; vy: number }[] = [];
      return {
        wants: t => t <= 45,
        tick: (t, [s]) => {
          const x = s!.positions_m, v = s!.velocities_m_s, N = x.length / 3;
          let yc = 0, vc = 0;
          for (let i = 0; i < N; i++) { yc += x[3 * i + 1]! / N; vc += v[3 * i + 1]! / N; }
          track.push({ y: yc, vy: vc });
          if (t > n) return;
          for (let i = 0; i < N; i++) flat = Math.max(flat, Math.abs(x[3 * i + 1]! - yc), Math.abs(x[3 * i]! - rest[3 * i]!), Math.abs(x[3 * i + 2]! - rest[3 * i + 2]!));
          centroid = Math.max(centroid, Math.abs(yc - (y0 + ref.displacement_m[t - 1]!)));
          velocity = Math.max(velocity, Math.abs(vc - ref.velocity_m_s[t - 1]!));
          euler = Math.max(euler, Math.abs(yc - (y0 - g * h * h * t * (t + 1) / 2)));
          for (let f = 0; f < tri.length; f += 3) for (const [a, b] of [[tri[f]!, tri[f + 1]!], [tri[f + 1]!, tri[f + 2]!], [tri[f + 2]!, tri[f]!]]) {
            const l = Math.hypot(x[3 * a!]! - x[3 * b!]!, x[3 * a! + 1]! - x[3 * b! + 1]!, x[3 * a! + 2]! - x[3 * b! + 2]!);
            const l0 = Math.hypot(rest[3 * a!]! - rest[3 * b!]!, rest[3 * a! + 1]! - rest[3 * b! + 1]!, rest[3 * a! + 2]! - rest[3 * b! + 2]!);
            strain = Math.max(strain, Math.abs(l / l0 - 1));
          }
        },
        finish: () => {
          let k = 0;
          track.forEach((t, i) => { if (t.vy < track[k]!.vy) k = i; });
          const fall = track[k]?.vy ?? NaN, up = Math.max(0, ...track.slice(k + 1).map(t => t.vy));
          return [
            { name: `centroid on the BDF2 recurrence over ${n} steps (< 0.1 mm)`, pass: centroid < 1e-4, detail: `${fmt(1e3 * centroid, 4)} mm off BDF2; ${fmt(1e3 * euler, 4)} mm off the implicit Euler recurrence y0 - g h^2 n (n + 1) / 2` },
            { name: 'velocity on the BDF2 recurrence (< 1 mm/s)', pass: velocity < 1e-3, detail: `${fmt(1e3 * velocity, 4)} mm/s` },
            { name: 'rigid translation (< 0.1 mm)', pass: flat < 1e-4, detail: `${fmt(1e3 * flat, 4)} mm` },
            { name: 'unstrained (< 1e-4)', pass: strain < 1e-4, detail: strain.toExponential(2) },
            { name: 'landing: reached the floor (fall < -4 m/s)', pass: fall < -4, detail: `${fmt(fall)} m/s` },
            { name: 'landing: no rebound (<= 0.05 m/s)', pass: up <= 0.05, detail: `${fmt(up, 4)} m/s upward after the fastest fall; rests at ${fmt(1e3 * (track.at(-1)?.y ?? NaN), 2)} mm` },
          ];
        },
      };
    }
    case 'hanging-curtain': {
      const m = source.probe.columns as number, n = source.probe.rows as number, hc = source.probe.cell as number, top = source.probe.top as number;
      const settle = Math.round((source.probe.settle as number) / h), window = Math.round(1 / h), first = settle - window + 1;
      const material = doc0.materials[0]!, k = material.stretchWeft_N_m, rho = material.arealDensity_kg_m2;
      const bottom = Math.floor(m / 2) + m * (n - 1), gated = range(2, n - 5), hem: number[] = [], bands = gated.map(() => [] as number[]);
      return {
        wants: t => t >= first && t <= settle,
        tick: (_t, [s]) => {
          const x = s!.positions_m;
          gated.forEach((j, q) => {
            let e = 0;
            for (let i = 1; i < m - 1; i++) { const a = i + m * j, b = a + m; e += (Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / hc - 1) / (m - 2); }
            bands[q]!.push(e);
          });
          hem.push(top - x[3 * bottom + 1]! - (n - 1) * hc);
        },
        finish: () => {
          if (hem.length < window) return [{ name: 'hanging equilibrium', pass: false, detail: `the run ended before the settle window (${hem.length} of ${window} steps)` }];
          const fit = ringdown(range(0, hem.length - 1).map(i => i * h), [hem, ...bands]), stretch = fit.rest[0]!;
          let worst = 0, drop = 0;
          for (let j = 0; j + 1 < n; j++) drop += hangingBandStrain(j, n, hc, k, rho, g) * hc;
          gated.forEach((j, q) => { const exact = hangingBandStrain(j, n, hc, k, rho, g); worst = Math.max(worst, Math.abs(fit.rest[q + 1]! - exact) / exact); });
          return [
            { name: 'the hem is a ringdown (fit rms < 1% of the drop)', pass: fit.rms < 0.01 * drop, detail: `rms ${fmt(1e6 * fit.rms, 1)} um; sway ${fmt(1e3 * fit.amplitude, 2)} mm at ${fmt(fit.w, 2)} rad/s, decay ${fmt(fit.a)} / s` },
            { name: 'band strain at equilibrium (< 1%)', pass: worst < 0.01, detail: `${fmt(100 * worst)} %` },
            { name: 'hem drop (within 1%)', pass: Math.abs(stretch - drop) < 0.01 * drop, detail: `${fmt(1e3 * stretch)} mm, exact ${fmt(1e3 * drop)} mm` },
          ];
        },
      };
    }
    case 'swinging-sheet': {
      const edge = source.probe.freeEdge as readonly number[], n = source.probe.gateSteps as number, y0 = source.probe.y0 as number, ticks = 45, ref = bdf2FreeFall(n, h, -g);
      const track = (x: number[], y: number[]) => (s: TickState) => { const p = s.positions_m; y.push(mean(edge.map(v => p[3 * v + 1]!))); x.push(mean(edge.map(v => p[3 * v]!))); };
      const crossing = (x: number[]) => { for (let k = 1; k < x.length; k++) if (x[k]! <= 0 && x[k - 1]! > 0) return (k + x[k - 1]! / (x[k - 1]! - x[k]!)) * h; return NaN; };
      const xs: number[] = [], ys: number[] = [], own = track(xs, ys);
      return {
        wants: t => t <= ticks,
        tick: (_t, [s]) => own(s!),
        finish: async () => {
          let worst = 0, at = 0, euler = 0;
          for (let k = 0; k < Math.min(n, ys.length); k++) {
            const e = Math.abs(ys[k]! - (y0 + ref.displacement_m[k]!)); if (e > worst) { worst = e; at = k + 1; }
            euler = Math.max(euler, Math.abs(ys[k]! - (y0 - g * h * h * (k + 1) * (k + 2) / 2)));
          }
          // The reference: the same scene and method at 4x the caps and tolerances / 4^2 (the test's converged run).
          const f = SWING.convergedFactor.value, cx: number[] = [], cy: number[] = [], conv = track(cx, cy);
          const reference = buildScene(o => ({ iterations: f * (budget?.[0] ?? o.iterations), pcgIterations: f * (budget?.[1] ?? o.pcgIterations), pcgTolerance: o.pcgTolerance! / f ** 2, residualTolerance: o.residualTolerance! / f ** 2 }));
          const cloth = await ClothSystem.create(device, reference, { statusInterval: 1 }), solver = cloth.solverFor(reference.cloth![0]!.id)!;
          for (let t = 1; t <= ticks && cloth.stepTo(t * h) && !cloth.failure; t++) conv(await solver.readState());
          cloth.destroy();
          const a = crossing(xs), b = crossing(cx);
          return [
            { name: `free edge on BDF2 free fall over ${n} steps (< 1 mm)`, pass: worst < 1e-3, detail: `${fmt(1e3 * worst)} mm at step ${at}; ${fmt(1e3 * euler)} mm off the implicit Euler recurrence` },
            { name: 'the free edge passes under the pin', pass: Number.isFinite(a), detail: `${fmt(a, 4)} s` },
            { name: `crossing within one step of the ${f}x converged run`, pass: Math.abs(a - b) <= h, detail: `${fmt(a, 4)} s against ${fmt(b, 4)} s (${cx.length} reference steps)` },
          ];
        },
      };
    }
    case 'peirce-cantilever': {
      const settle = Math.round(PEIRCE.settle.value / h);
      let at: readonly TickState[] | undefined;
      return {
        wants: t => t === settle,
        tick: (_t, docs) => { at = docs; },
        finish: () => {
          if (!at) return [{ name: 'Peirce chord angles', pass: false, detail: 'the run ended before the settle time' }];
          return source.cloth.map((_, s) => {
            const [clampCells, cols, rows, overhang, exact] = source.probe[`strip${s}`] as number[], x = at![s]!.positions_m;
            let dx = 0, dy = 0;
            for (let j = 0; j < rows!; j++) { const root = clampCells! + cols! * j, tip = cols! - 1 + cols! * j; dx += (x[3 * tip]! - x[3 * root]!) / rows!; dy += (x[3 * root + 1]! - x[3 * tip + 1]!) / rows!; }
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            return { name: `strip ${s} (l = ${fmt(1e3 * overhang!, 0)} mm) chord within 1 deg of the heavy elastica`, pass: Math.abs(angle - exact!) < 1, detail: `${fmt(angle, 2)} deg, exact ${fmt(exact!, 2)} (through ClothSystem with contact; the test runs the bare BDF2 solver)` };
          });
        },
      };
    }
    case 'friction-incline': {
      const t0 = Math.round(INCLINE.gateStart.value / h), t1 = Math.round(INCLINE.gateEnd.value / h);
      const along = source.cloth.map(() => [] as number[]);
      return {
        wants: t => t >= t0 && t <= t1,
        tick: (_t, docs) => source.cloth.forEach((_, s) => {
          const [, , dx, dy] = source.probe[`slope${s}`] as number[], p = docs[s]!.positions_m, N = p.length / 3;
          let a = 0; for (let i = 0; i < N; i++) a += (p[3 * i]! * dx! + p[3 * i + 1]! * dy!) / N;
          along[s]!.push(a);
        }),
        finish: () => {
          if (along[0]!.length < t1 - t0 + 1) return [{ name: 'friction on the incline', pass: false, detail: 'the run ended before the gate window' }];
          const fit = (s: number[]) => {
            const rows = s.map((_, k) => { const t = (t0 + k) * h; return [1, t, t * t / 2]; });
            const A = [0, 1, 2].map(i => [0, 1, 2].map(j => rows.reduce((q, r) => q + r[i]! * r[j]!, 0))), b = [0, 1, 2].map(i => rows.reduce((q, r, k) => q + r[i]! * s[k]!, 0));
            for (let i = 0; i < 3; i++) for (let r = i + 1; r < 3; r++) { const f = A[r]![i]! / A[i]![i]!; for (let c = i; c < 3; c++) A[r]![c]! -= f * A[i]![c]!; b[r]! -= f * b[i]!; }
            const x = [0, 0, 0]; for (let i = 2; i >= 0; i--) x[i] = (b[i]! - [0, 1, 2].slice(i + 1).reduce((q, c) => q + A[i]![c]! * x[c]!, 0)) / A[i]![i]!;
            return x[2]!;
          };
          const creep = Math.abs(along[0]!.at(-1)! - along[0]![0]!), a = fit(along[1]!), exact = (source.probe.slope1 as number[])[1]!;
          return [
            { name: 'the patch below the friction angle sticks (< 2 mm)', pass: creep < 2e-3, detail: `${fmt(1e3 * creep)} mm from ${INCLINE.gateStart.value} to ${INCLINE.gateEnd.value} s` },
            { name: 'the patch above it slides at g (sin - mu cos) (within 5%)', pass: Math.abs(a / exact - 1) < 0.05, detail: `${fmt(a, 4)} m/s^2, exact ${fmt(exact, 4)} (${fmt(100 * (a / exact - 1), 2)} %)` },
          ];
        },
      };
    }
  }
}

// ---------------------------------------------------------------- the run
const docs = scene.cloth!, gates = analytic && gatesFor(analytic);
// --status-interval N: steps between the system's status-word reads (the product's default when absent).
const STATUS = opt('status-interval', '');
const cloth = await ClothSystem.create(device, scene, STATUS ? { statusInterval: Number(STATUS) } : {});
const each = docs.map(doc => ({ doc, solver: cloth.solverFor(doc.id)!, icloth: cloth.iclothFor(doc.id), topology: prepareClothTopology(doc.geometry.data, doc.materials), inside: solidProbe(doc) }));
const canTime = features.includes('timestamp-query');
const ms: number[] = [], msAt: number[] = [], newton: number[] = [], pcg: number[] = [], alpha: number[] = [], stageSums = new Map<string, number>(), stagePasses = new Map<string, number>();
const ic = { rounds: [] as number[], iterations: [] as number[], perRound: [] as number[][], proxPairs: [] as number[], proxSolid: [] as number[], worked: [] as number[], remainingPairs: 0, remainingSolid: 0, unclean: 0, capped: 0, stalled: 0, degenerate: 0 };
const checkpoints: { tick: number; intersections: number; insideStatic: number; insideMoving: number; leastField_mm: number; over10: number; over15: number; maxStrain: number; maxCompression: number }[] = [];
const checkAt = new Set(range(1, CHECKPOINTS).map(i => Math.round(i * STEPS / CHECKPOINTS)));
let stageSteps = 0, lastSample = 0, steps = 0, truncated = false;
dispatched.clear();
const encodedBefore = () => new Map(dispatched);
const started = performance.now();
async function checkpoint(tick: number): Promise<void> {
  const c = { tick, intersections: 0, insideStatic: 0, insideMoving: 0, leastField_mm: Infinity, over10: 0, over15: 0, maxStrain: 0, maxCompression: 0 };
  for (const e of each) {
    const x = await e.solver.readPositions(), at = e.inside(x, cloth.time_s), tail = strainTail(x, e.topology.edges, e.topology.edgeRestLength_m);
    c.intersections += iclothSelfIntersections(x, e.topology.faces); c.insideStatic += at.fixed; c.insideMoving += at.moving; c.leastField_mm = Math.min(c.leastField_mm, 1e3 * at.least);
    c.over10 += tail.over10; c.over15 += tail.over15; c.maxStrain = Math.max(c.maxStrain, tail.max); c.maxCompression = Math.min(c.maxCompression, tail.min);
  }
  checkpoints.push(c);
}
const stepDispatches = encodedBefore();
let encodedSteps = 0;
for (let tick = 1; tick <= STEPS; tick++) {
  // Every 16th step may be stage-timed (one pass per stage change and a readback): left out of the ms figures.
  const instrumented = canTime && tick % 16 === 8;
  cloth.timing = instrumented;
  const t0 = performance.now();
  if (!cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S)) break;
  await device.queue.onSubmittedWorkDone();
  const dt = performance.now() - t0;
  steps = tick; encodedSteps++;
  if (!instrumented) { ms.push(dt); msAt.push(tick); }
  // Reads between the fences (not timed).
  for (const e of each) {
    const control = await e.solver.readNewtonControl();
    newton.push(control[3]!); pcg.push(control[7]!); alpha.push(control[2]!);
    if (e.icloth) {
      const c = await e.icloth.readCounters(), R = c.rounds, cap = e.icloth.config.rounds;
      ic.rounds.push(R); ic.iterations.push(c.impactIterations); ic.perRound.push(c.impactsPerRound); ic.proxPairs.push(c.proximityPairs); ic.proxSolid.push(c.proximitySolid);
      ic.remainingPairs += c.remainingPairs; ic.remainingSolid += c.remainingSolid; if (c.remainingPairs + c.remainingSolid) ic.unclean++;
      ic.capped = Math.max(ic.capped, c.cappedSolves); ic.stalled = Math.max(ic.stalled, c.stalledInfeasible); ic.degenerate = Math.max(ic.degenerate, c.degenerate);
      // Dispatches that did work: step begin + proximity (5), CCD begin, a detection (4) per round until one is clean,
      // per round with impacts the zone init, the commit, its descent (2 an iteration, + the closing decision), the verify (4).
      ic.worked.push(5 + 1 + 4 * Math.min(R + 1, cap) + 3 * R + 2 * c.impactIterations + 4);
    }
  }
  if (gates?.wants(tick)) gates.tick(tick, await Promise.all(each.map(e => e.solver.readState())));
  const sample = cloth.performance;
  if (sample && sample.sampleId !== lastSample && !sample.unavailable) {
    lastSample = sample.sampleId; stageSteps += sample.steps;
    for (const s of sample.stages) { stageSums.set(s.id, (stageSums.get(s.id) ?? 0) + s.duration_ms); stagePasses.set(s.id, (stagePasses.get(s.id) ?? 0) + s.passes); }
  }
  if (cloth.failure) break;
  if (checkAt.has(tick) && tick !== STEPS) await checkpoint(tick);
  if ((performance.now() - started) / 1e3 > MAX_WALL_S) { truncated = true; break; }
}
const encoded = [...dispatched].map(([id, n]) => ({ id, perStep: (n - (stepDispatches.get(id) ?? 0)) / Math.max(encodedSteps, 1) }));
await device.queue.onSubmittedWorkDone();
await new Promise(resolve => setTimeout(resolve, 50));
await checkpoint(steps);
const gateResults = gates ? await gates.finish() : [];
const failure = cloth.failure?.message, notes = [...cloth.notes].map(([n, t]) => `${n} (since ${t.toFixed(3)} s)`);
const peaks = each.map(e => e.icloth ? { capacity: e.icloth.capacity } : undefined).filter(Boolean);
const facts = each.map(e => ({ id: e.doc.id, vertices: e.solver.N, triangles: e.solver.T, newton: e.doc.solver?.newton }));
cloth.destroy();
await device.queue.onSubmittedWorkDone();

const windows = range(0, 4).map(w => { const lo = w * STEPS / 5, hi = (w + 1) * STEPS / 5, part = ms.filter((_, i) => msAt[i]! > lo && msAt[i]! <= hi); return { from: Math.floor(lo) + 1, to: Math.floor(hi), mean: mean(part), max: part.length ? Math.max(...part) : NaN }; });
const complete = steps === STEPS && !failure, end = checkpoints.at(-1)!;
const clean = end.intersections === 0 && end.insideStatic === 0 && end.insideMoving === 0 && checkpoints.every(c => c.intersections === 0 && c.insideStatic === 0 && c.insideMoving === 0);
const result = {
  scene: sceneId!, method: METHOD, budget: BUDGET || 'document', documents: facts, steps, authoredSteps: Math.round(duration_s! / h), requestedSteps: STEPS, complete, truncated, failure, notes, clean,
  ms: { mean: mean(ms), median: median(ms), max: ms.length ? Math.max(...ms) : NaN, min: ms.length ? Math.min(...ms) : NaN, steps: ms.length }, windows,
  stages: [...stageSums].map(([id, v]) => ({ id, ms: v / Math.max(stageSteps, 1), passes: (stagePasses.get(id) ?? 0) / Math.max(stageSteps, 1) })), stageSamples: stageSteps,
  dispatches: counting ? { encodedPerStep: encoded, encodedTotalPerStep: sum(encoded.map(e => e.perStep)), iclothWorkedPerStep: ic.worked.length ? { mean: mean(ic.worked) * each.length, max: Math.max(...ic.worked) } : undefined } : undefined,
  newtonIterations: { mean: mean(newton), max: Math.max(...newton) }, pcgIterations: { mean: mean(pcg), max: Math.max(...pcg) },
  // The step's last accepted line-search step length (control word 2); per-iteration lengths are not kept by the solver.
  lastAlpha: { mean: mean(alpha), min: Math.min(...alpha), below1: alpha.filter(a => a < 1).length, zero: alpha.filter(a => a === 0).length, steps: alpha.length,
    byFifth: range(0, 4).map(w => { const part = alpha.slice(Math.floor(w * alpha.length / 5), Math.floor((w + 1) * alpha.length / 5)); return { mean: mean(part), below1: part.filter(a => a < 1).length }; }) },
  icloth: ic.rounds.length ? { roundsMean: mean(ic.rounds), roundsMax: Math.max(...ic.rounds), impactIterationsMean: mean(ic.iterations), impactIterationsMax: Math.max(...ic.iterations),
    impactsPerRoundMean: range(0, 7).map(r => mean(ic.perRound.map(p => p[r] ?? 0))), impactsTotal: sum(ic.perRound.map(sum)),
    proximityPairsMean: mean(ic.proxPairs), proximityPairsMax: Math.max(...ic.proxPairs), proximitySolidMean: mean(ic.proxSolid), proximitySolidMax: Math.max(...ic.proxSolid),
    remainingPairs: ic.remainingPairs, remainingSolid: ic.remainingSolid, uncleanSteps: ic.unclean, cappedSolves: ic.capped, stalledInfeasible: ic.stalled, degenerate: ic.degenerate, pools: peaks } : undefined,
  checkpoints, gates: gateResults,
};

console.log(`${result.scene} [${METHOD}, Newton ${facts.map(f => `${f.newton?.iterations} x ${f.newton?.pcgIterations}`).join(' / ')}${BUDGET ? ' (override)' : ''}]: ${facts.map(f => `${f.vertices} vertices`).join(' + ')}, ${steps} of ${STEPS} steps`
  + `${failure ? `, HALTED: ${failure}` : ''}${truncated ? `, STOPPED at the ${MAX_WALL_S} s wall cap` : ''}`);
console.log(`  ms/step mean ${fmt(result.ms.mean, 2)} median ${fmt(result.ms.median, 2)} max ${fmt(result.ms.max, 2)} over ${ms.length} steps; by fifth of the run (mean/max): ${windows.map(w => `${fmt(w.mean, 1)}/${fmt(w.max, 0)}`).join('  ')}`);
console.log(`  stages (GPU ms per step over ${stageSteps} sampled steps; passes): ${result.stages.map(s => `${s.id} ${fmt(s.ms, 2)} (${fmt(s.passes, 0)})`).join(', ') || 'no sample'}`);
if (result.dispatches) console.log(`  dispatches encoded per step ${fmt(result.dispatches.encodedTotalPerStep, 0)}: ${encoded.map(e => `${e.id} ${fmt(e.perStep, 0)}`).join(', ')}`
  + (result.dispatches.iclothWorkedPerStep ? `; I-Cloth collision dispatches that did work: mean ${fmt(result.dispatches.iclothWorkedPerStep.mean, 0)}, max ${result.dispatches.iclothWorkedPerStep.max} (derived from the counters)` : ''));
console.log(`  Newton iterations mean ${fmt(result.newtonIterations.mean, 2)} max ${result.newtonIterations.max}; PCG iterations per step mean ${fmt(result.pcgIterations.mean, 1)} max ${result.pcgIterations.max}; `
  + `last line-search alpha of a step: mean ${fmt(result.lastAlpha.mean)}, min ${fmt(result.lastAlpha.min)}, below 1 in ${result.lastAlpha.below1} (0 in ${result.lastAlpha.zero}) of ${alpha.length} document-steps; by fifth (mean, below 1): ${result.lastAlpha.byFifth.map(w => `${fmt(w.mean, 2)}, ${w.below1}`).join('  ')}`);
if (result.icloth) {
  const c = result.icloth;
  console.log(`  I-Cloth: proximity records per step mean ${fmt(c.proximityPairsMean, 0)} (max ${c.proximityPairsMax}) pairs + ${fmt(c.proximitySolidMean, 0)} (max ${c.proximitySolidMax}) solid; rounds with impacts mean ${fmt(c.roundsMean, 2)} max ${c.roundsMax}; `
    + `impact-zone iterations per step mean ${fmt(c.impactIterationsMean, 1)} max ${c.impactIterationsMax}; impacts per round (mean) ${c.impactsPerRoundMean.map(x => fmt(x, 1)).join(' ')}; ${c.impactsTotal} impacts in all`);
  console.log(`  I-Cloth: closing detection left ${c.remainingPairs} pair + ${c.remainingSolid} solid impacts over ${c.uncleanSteps} steps; ${c.cappedSolves} solves at the iteration cap (${c.stalledInfeasible} still violated); ${c.degenerate} degenerate pairs`);
}
for (const c of checkpoints) console.log(`  tick ${String(c.tick).padStart(3)}: ${c.intersections} self-intersections, ${c.insideStatic} + ${c.insideMoving} vertices inside static + moving solids (least field ${fmt(c.leastField_mm)} mm), `
  + `edges over 10.1% ${c.over10}, over 15% ${c.over15}, max stretch ${fmt(100 * c.maxStrain, 2)}%, max compression ${fmt(-100 * c.maxCompression, 2)}%`);
for (const gate of gateResults) console.log(`  gate ${gate.pass ? 'PASS' : 'FAIL'}: ${gate.name}: ${gate.detail}`);
if (notes.length) console.log(`  notes: ${notes.join('; ')}`);
console.log(`  SUMMARY ${name}: ${complete ? 'complete' : 'INCOMPLETE'}, ${clean ? 'clean' : 'NOT CLEAN'}, mean ${fmt(result.ms.mean, 2)} ms/step${gateResults.length ? `, gates ${gateResults.filter(x => x.pass).length}/${gateResults.length}` : ''}`);
writeFileSync(OUT, JSON.stringify(result, null, 1));
console.log(`  saved ${OUT}`);
if (errors.length) { console.error(`${errors.length} WebGPU errors`); process.exitCode = 1; }
device.destroy();
