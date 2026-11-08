import assert from 'node:assert/strict';
import test from 'node:test';
import { clothGridData } from '../lib/cloth/grid';
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothAttachmentScript, ClothMaterial, ClothTopologyData } from '../lib/cloth/types';
import {
  buildClothSolverModel, clothAugmentedEnergy, clothBendingBlock, clothHingeAngleGradient, clothLocalSystem, clothMechanicalEnergy,
  clothMembraneBlock, clothMembraneConstants, clothOracleStep, clothProjectPSD, clothStepFrame, clothSymmetricEigen3, clothVertexUpdate,
  clothWarmStartDuals, createClothOracleState, type ClothStepOptions, type Mat3, type Vec3,
} from '../lib/cloth/oracle';

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0, thickness_m: 5e-4 };

function rng(seed: number): () => number { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }
function fdGradient(f: (x: Vec3) => number, x: Vec3, h: number): Vec3 {
  return [0, 1, 2].map(c => { const a = [...x] as Vec3, b = [...x] as Vec3; a[c] += h; b[c] -= h; return (f(a) - f(b)) / (2 * h); }) as Vec3;
}
function close(a: readonly number[], b: readonly number[], tol: number, label: string): void {
  const scale = Math.max(1e-12, ...b.map(Math.abs));
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i]! - b[i]!) <= tol * scale, `${label}[${i}]: ${a[i]} vs ${b[i]} (scale ${scale})`);
}
function minEigen(H: Mat3): number { return Math.min(...clothSymmetricEigen3(H).values); }

test('membrane block: FD gradient and Hessian, including compressed fibres', () => {
  const c = clothMembraneConstants(cotton), r = rng(7);
  const dm = [1 / 0.02, 0, -0.5 / 0.02 * 0, 1 / 0.025]; // rest edges (0.02, 0) and (0, 0.025)
  for (let trial = 0; trial < 24; trial++) {
    const x0: Vec3 = [r() * 1e-3, r() * 1e-3, r() * 1e-3];
    const s1 = 0.9 + 0.2 * r(), s2 = 0.9 + 0.2 * r(), shear = 0.004 * (r() - 0.5);
    const x1: Vec3 = [x0[0] + 0.02 * s1, x0[1] + 0.003 * (r() - 0.5), x0[2] + shear];
    const x2: Vec3 = [x0[0] + shear, x0[1] + 0.003 * (r() - 0.5), x0[2] + 0.025 * s2];
    for (let k = 0; k < 3; k++) {
      const X = [x0, x1, x2] as [Vec3, Vec3, Vec3];
      const at = (p: Vec3) => { const Y = [...X] as [Vec3, Vec3, Vec3]; Y[k] = p; return Y; };
      const block = clothMembraneBlock(c, dm, 0, 2.5e-4, k, ...X, false);
      close(block.gradient, fdGradient(p => clothMembraneBlock(c, dm, 0, 2.5e-4, k, ...at(p), false).energy, X[k]!, 1e-8), 1e-5, `gradient ${trial}.${k}`);
      const fdH: number[] = [];
      for (let col = 0; col < 3; col++) {
        const a = [...X[k]!] as Vec3, b = [...X[k]!] as Vec3; a[col] += 1e-8; b[col] -= 1e-8;
        const ga = clothMembraneBlock(c, dm, 0, 2.5e-4, k, ...at(a), false).gradient, gb = clothMembraneBlock(c, dm, 0, 2.5e-4, k, ...at(b), false).gradient;
        for (let row = 0; row < 3; row++) fdH[3 * row + col] = (ga[row]! - gb[row]!) / 2e-8;
      }
      close(block.hessian, fdH, 1e-4, `hessian ${trial}.${k}`);
      const projected = clothMembraneBlock(c, dm, 0, 2.5e-4, k, ...X, true).hessian;
      const scale = Math.max(...block.hessian.map(Math.abs));
      assert.ok(minEigen(projected) >= -1e-10 * scale, 'projected block is PSD');
      if (minEigen(block.hessian) >= 0) close(projected, block.hessian, 1e-12, 'PSD blocks are unchanged');
    }
  }
});

test('membrane energy is C1 across fibre compression and positive definite in strain', () => {
  const c = clothMembraneConstants({ ...cotton, compressionRatio: 0.01 });
  const dm = [1, 0, 0, 1];
  // Stretch the warp fibre across zero strain with a stretched weft and Poisson coupling active.
  const at = (s: number) => clothMembraneBlock(c, dm, 0, 1, 1, [0, 0, 0], [s, 0, 0], [0, 0, 1.05], false).gradient[0];
  const below = at(1 - 1e-9), above = at(1 + 1e-9);
  assert.ok(Math.abs(below - above) < 1e-6 * Math.abs(above) + 1e-9, `gradient continuous at zero strain: ${below} vs ${above}`);
  // Every softened law stays PD: C - (1 - r) diag(softWarp [e11<0], softWeft [e22<0]).
  for (const [w1, w2] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const a = c.c11 - (1 - c.ratio) * c.softWarp * w1!, d = c.c22 - (1 - c.ratio) * c.softWeft * w2!;
    assert.ok(a > 0 && a * d - c.c12 * c.c12 > 0, `softened law ${w1}${w2} is PD`);
  }
});

test('bending: FD angle gradient and a PSD Gauss-Newton Hessian', () => {
  const r = rng(3);
  for (let trial = 0; trial < 16; trial++) {
    const X: [Vec3, Vec3, Vec3, Vec3] = [[0.3 * r(), 0.8 + 0.2 * r(), 0.3 * r()], [0.3 * r(), -0.8 - 0.2 * r(), 0.4 * r() - 0.2], [0, 0, 0], [1 + 0.1 * r(), 0.1 * r(), 0.1 * r()]];
    const hinge = clothHingeAngleGradient(...X)!;
    for (let k = 0; k < 4; k++) {
      const fd = fdGradient(p => { const Y = [...X] as typeof X; Y[k] = p; return clothHingeAngleGradient(...Y)!.angle; }, X[k]!, 1e-7);
      close(hinge.gradient[k]!, fd, 1e-6, `angle gradient ${trial}.${k}`);
      const block = clothBendingBlock(0.3, 0.2, k, ...X);
      const fdE = fdGradient(p => { const Y = [...X] as typeof X; Y[k] = p; return clothBendingBlock(0.3, 0.2, k, ...Y).energy; }, X[k]!, 1e-7);
      close(block.gradient, fdE, 1e-6, `bending gradient ${trial}.${k}`);
      assert.ok(minEigen(block.hessian) >= -1e-12 * Math.max(...block.hessian.map(Math.abs)), 'GN Hessian PSD');
    }
  }
});

test('PSD projection clamps an indefinite matrix to its nearest PSD part', () => {
  const H: Mat3 = [2, 1, 0, 1, -3, 0.5, 0, 0.5, 1];
  const P = clothProjectPSD(H), { values } = clothSymmetricEigen3(H);
  assert.ok(Math.min(...clothSymmetricEigen3(P).values) >= -1e-12);
  const trace = P[0] + P[4] + P[8];
  close([trace], [values.filter(v => v > 0).reduce((a, b) => a + b, 0)], 1e-12, 'trace keeps positive eigenvalues');
});

function sheet(columns: number, rows: number, opts: { pins?: 'corners' | 'edge' | 'none'; strainLimit?: number; damping?: number; bending?: number } = {}) {
  const pins = opts.pins ?? 'corners';
  const data: ClothTopologyData = clothGridData({ columns, rows, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0],
    group: (i, j) => pins === 'corners' ? (j === 0 && (i === 0 || i === columns - 1) ? 1 : 0) : pins === 'edge' ? (j === 0 ? 1 : 0) : 0 });
  const material = { ...cotton, strainLimit: opts.strainLimit, damping_s: opts.damping ?? 0, bending_N_m: opts.bending ?? cotton.bending_N_m };
  const topology = prepareClothTopology(data, [material]);
  const colours = colourClothVertices(clothVertexNeighbours(topology));
  return { data, topology, colours, material };
}
const g: [number, number, number] = [0, -9.81, 0];

test('local vertex system: FD gradient with pins, strain limits, damping and a frozen damping Hessian', () => {
  const { topology, colours, material } = sheet(5, 5, { strainLimit: 0.02, damping: 0.01 });
  const model = buildClothSolverModel(topology, [material], colours);
  const r = rng(11), x0 = Float64Array.from(topology.data.restPositions_m);
  const state = createClothOracleState(model, x0);
  for (let i = 0; i < state.x.length; i++) state.x[i] += 0.01 * (r() - 0.5);
  for (let i = 0; i < state.v.length; i++) state.v[i] = 0.2 * (r() - 0.5);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'implicit-euler', iterations: 1, chebyshevRho: 0 };
  const { frame } = clothStepFrame(model, state, options);
  clothWarmStartDuals(model, state, frame.hEff);
  for (let e = 0; e < topology.edgeCount; e++) { state.edgeDuals[2 * e] = 0.5 * r(); }
  for (let v = 0; v < topology.vertexCount; v++) for (let c = 0; c < 3; c++) state.pinDuals[4 * v + c] = 0.01 * (r() - 0.5);
  // Stretch the sheet so several strain limits are active.
  for (let v = 0; v < topology.vertexCount; v++) state.x[3 * v] *= 1.04;
  for (const v of [0, 6, 12, 4]) {
    const xv: Vec3 = [state.x[3 * v]!, state.x[3 * v + 1]!, state.x[3 * v + 2]!];
    const local = clothLocalSystem(model, frame, state, v, xv);
    const fd = fdGradient(p => clothLocalSystem(model, frame, state, v, p, local.damping).energy, xv, 1e-7);
    close(local.gradient, fd, 2e-5, `local gradient v${v}`);
    assert.ok(minEigen(local.hessian) > 0, `local Hessian PD at v${v}`);
    const update = clothVertexUpdate(model, frame, state, v);
    assert.ok(update.energyAfter <= update.energyBefore, `update descends at v${v}`);
  }
});

test('repeated block updates of one vertex reach its local stationary point', () => {
  const { topology, colours, material } = sheet(4, 4, { pins: 'none', damping: 0.02 });
  const model = buildClothSolverModel(topology, [material], colours);
  const state = createClothOracleState(model, topology.data.restPositions_m);
  const r = rng(5); for (let i = 0; i < state.x.length; i++) state.x[i] += 0.02 * (r() - 0.5);
  const { frame } = clothStepFrame(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'implicit-euler', iterations: 1, chebyshevRho: 0 });
  const v = 5;
  let last = Infinity;
  for (let i = 0; i < 40; i++) { const u = clothVertexUpdate(model, frame, state, v); state.x.set(u.position, 3 * v); last = Math.hypot(...u.gradient); }
  const local = clothLocalSystem(model, frame, state, v, [state.x[15]!, state.x[16]!, state.x[17]!]);
  assert.ok(Math.hypot(...local.gradient) < 1e-9 * (1 + topology.mass_kg[v]! * 9.81), `stationary: |g| = ${Math.hypot(...local.gradient)} (last ${last})`);
});

test('augmented-Lagrangian energies differentiate to the clamped AVBD force', () => {
  for (const [k, lambda, C, inequality] of [[10, 0.3, 0.02, true], [10, 0.3, -0.05, true], [10, -0.2, 0.01, false]] as const) {
    const h = 1e-7, fd = (clothAugmentedEnergy(k, lambda, C + h, inequality) - clothAugmentedEnergy(k, lambda, C - h, inequality)) / (2 * h);
    const force = inequality ? Math.max(k * C + lambda, 0) : k * C + lambda;
    assert.ok(Math.abs(fd - force) < 1e-6, `${fd} vs ${force}`);
  }
});

test('free fall: implicit Euler and BDF2 integrate a rigid drop exactly', () => {
  for (const integrator of ['implicit-euler', 'bdf2'] as const) {
    const { topology, colours, material } = sheet(4, 4, { pins: 'none' });
    const model = buildClothSolverModel(topology, [material], colours);
    const state = createClothOracleState(model, topology.data.restPositions_m);
    const steps = 10, dt = 1 / 60;
    for (let s = 0; s < steps; s++) clothOracleStep(model, state, { dt_s: dt, gravity_m_s2: g, integrator, iterations: 4, chebyshevRho: 0 });
    for (let v = 0; v < topology.vertexCount; v++) assert.ok(Math.abs(state.v[3 * v + 1]! + 9.81 * steps * dt) < 1e-10, `${integrator}: v_y = ${state.v[3 * v + 1]}`);
  }
});

test('a moving attachment carries its pins to the scripted target', () => {
  const { topology, colours, material } = sheet(6, 6, { pins: 'edge' });
  const scripts: ClothAttachmentScript[] = [{ kind: 'translate', group: 1, start_s: 0, end_s: 0.25, vector: [0.1, 0, 0.05], easing: 'cosine' }];
  const model = buildClothSolverModel(topology, [material], colours, scripts);
  const state = createClothOracleState(model, topology.data.restPositions_m);
  for (let s = 0; s < 30; s++) clothOracleStep(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 16, chebyshevRho: 0 });
  let worst = 0;
  for (let v = 0; v < topology.vertexCount; v++) if (topology.groups[v]) {
    const target = [topology.data.restPositions_m[3 * v]! + 0.1, topology.data.restPositions_m[3 * v + 1]!, topology.data.restPositions_m[3 * v + 2]! + 0.05];
    worst = Math.max(worst, Math.hypot(...[0, 1, 2].map(c => state.x[3 * v + c]! - target[c]!)));
  }
  assert.ok(worst < 1e-4, `pin error ${worst} m`);
});

test('the AVBD strain limit holds a soft sheet near its limit', () => {
  const soft = (strainLimit?: number) => {
    const { topology, colours, material } = sheet(6, 6, { pins: 'edge', strainLimit, damping: 0.02 });
    const m = { ...material, stretchWarp_N_m: 2, stretchWeft_N_m: 2, shear_N_m: 0.5, poisson: 0 };
    const model = buildClothSolverModel(prepareClothTopology(topology.data, [m]), [m], colours);
    const state = createClothOracleState(model, topology.data.restPositions_m);
    let report;
    for (let s = 0; s < 40; s++) report = clothOracleStep(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 24, chebyshevRho: 0 });
    return report!.maxStrain;
  };
  const free = soft(), limited = soft(0.03);
  assert.ok(free > 0.06, `unlimited soft sheet stretches: ${free}`);
  assert.ok(limited < 0.035, `limited strain ${limited}`);
});

test('energy loss of a free-swinging pinned sheet at dt 1/60: BDF2 keeps more than implicit Euler', t => {
  const run = (integrator: 'implicit-euler' | 'bdf2') => {
    // Pinned at two corners of one edge and released horizontal: an undamped swing.
    const data = clothGridData({ columns: 8, rows: 8, width_m: 0.4, height_m: 0.4, origin_m: [0, 1, 0],
      group: (i, j) => j === 0 && (i === 0 || i === 7) ? 1 : 0 });
    const topology = prepareClothTopology(data, [cotton]);
    const model = buildClothSolverModel(topology, [cotton], colourClothVertices(clothVertexNeighbours(topology)));
    const state = createClothOracleState(model, data.restPositions_m);
    const e0 = clothMechanicalEnergy(model, state.x, state.v, g).total;
    let lowest = Infinity, final = e0;
    for (let s = 0; s < 60; s++) {
      clothOracleStep(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator, iterations: 48, chebyshevRho: 0 });
      const e = clothMechanicalEnergy(model, state.x, state.v, g);
      lowest = Math.min(lowest, e.potential); final = e.total;
    }
    // Normalize by the potential the swing released at its lowest point.
    const swing = clothMechanicalEnergy(model, data.restPositions_m, new Float64Array(3 * topology.vertexCount), g).potential - lowest;
    return { lost: e0 - final, fraction: (e0 - final) / swing, swing };
  };
  const ie = run('implicit-euler'), bdf2 = run('bdf2');
  t.diagnostic(`1 s swing (released ${ie.swing.toExponential(3)} J): implicit Euler loses ${(100 * ie.fraction).toFixed(1)} %, BDF2 ${(100 * bdf2.fraction).toFixed(1)} %`);
  assert.ok(ie.lost > 0 && bdf2.lost > 0, 'both integrators dissipate');
  assert.ok(bdf2.lost < ie.lost, `BDF2 loses less: ${bdf2.lost} vs ${ie.lost}`);
});

test('one step converges with K: the residual falls and Chebyshev accelerates it', () => {
  const { topology, colours, material } = sheet(10, 10, { damping: 0.005 });
  const model = buildClothSolverModel(topology, [material], colours);
  const residualsAt = (rho: number) => {
    const state = createClothOracleState(model, topology.data.restPositions_m);
    for (let s = 0; s < 5; s++) clothOracleStep(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 32, chebyshevRho: rho });
    return clothOracleStep(model, state, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 32, chebyshevRho: rho }, true).residuals;
  };
  const plain = residualsAt(0), accelerated = residualsAt(0.9);
  assert.ok(plain.at(-1)! < 0.1 * plain[0]!, `residual falls: ${plain[0]} -> ${plain.at(-1)}`);
  assert.ok(accelerated.at(-1)! <= plain.at(-1)!, `Chebyshev: ${accelerated.at(-1)} vs ${plain.at(-1)}`);
});
