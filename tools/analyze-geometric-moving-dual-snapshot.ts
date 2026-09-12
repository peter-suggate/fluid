import { readFile } from 'node:fs/promises';

const INVALID = 0xffff_ffff;
const ROUND_EPS = 9.5367431640625e-7;

type Edge = { to: number; rev: number; cap: number; initial: number };
class Dinic {
  graph: Edge[][];
  constructor(readonly size: number) { this.graph = Array.from({ length: size }, () => []); }
  add(from: number, to: number, cap: number) {
    const f = { to, rev: this.graph[to].length, cap, initial: cap };
    const r = { to: from, rev: this.graph[from].length, cap: 0, initial: 0 };
    this.graph[from].push(f); this.graph[to].push(r);
  }
  solve(source: number, sink: number) {
    let total = 0;
    for (;;) {
      const level = new Int32Array(this.size).fill(-1); level[source] = 0;
      const queue = [source];
      for (let q = 0; q < queue.length; q++) for (const e of this.graph[queue[q]]) {
        if (e.cap > 1e-13 && level[e.to] < 0) { level[e.to] = level[queue[q]] + 1; queue.push(e.to); }
      }
      if (level[sink] < 0) return { total, reachable: level };
      const next = new Int32Array(this.size);
      const push = (v: number, amount: number): number => {
        if (v === sink) return amount;
        for (; next[v] < this.graph[v].length; next[v]++) {
          const e = this.graph[v][next[v]];
          if (e.cap <= 1e-13 || level[e.to] !== level[v] + 1) continue;
          const sent = push(e.to, Math.min(amount, e.cap));
          if (sent > 1e-13) { e.cap -= sent; this.graph[e.to][e.rev].cap += sent; return sent; }
        }
        return 0;
      };
      for (;;) { const sent = push(source, Number.POSITIVE_INFINITY); if (sent <= 1e-13) break; total += sent; }
    }
  }
}

const path = process.argv[2] ?? '/tmp/sparse-cm12-rigid-moving-dual-snapshot.json';
const snapshot = JSON.parse(await readFile(path, 'utf8'));
const { faceCount, fields, metadata, faceState, descriptors } = snapshot;
const cells: number[] = snapshot.acceptedCells;
const n = cells.length;
const dense = new Map(cells.map((cell, index) => [cell, index]));

function run(tolerant: boolean) {
  const source = n, sink = n + 1, flow = new Dinic(n + 2);
  let target = 0, negativeVolume = 0, storage = 0, allowedRoundoff = 0;
  for (let i = 0; i < n; i++) {
    const cell = cells[i], volume = fields.currentVolume[cell], oldCap = fields.oldCapacity[cell];
    const fraction = snapshot.controlWords
      ? (snapshot.controlWords[3] + 1) / Math.max(1, snapshot.controlWords[2]) : 1;
    const cap = oldCap + fraction * (fields.newCapacity[cell] - oldCap);
    const tol = cap > 0 ? ROUND_EPS * cap : ROUND_EPS * oldCap;
    if (volume >= 0) { flow.add(source, i, volume); target += volume; } else negativeVolume += -volume;
    const tolerance = cap > 0 ? ROUND_EPS * cap : ROUND_EPS * oldCap;
    allowedRoundoff += tolerance;
    const bound = Math.max(0, cap + (tolerant ? tolerance : 0));
    flow.add(i, sink, bound); storage += bound;
  }
  let exteriorCapacity = 0, directedCapacity = 0;
  for (let f = 0; f < faceCount; f++) {
    const a = metadata[4 * f] >>> 0, b = metadata[4 * f + 1] >>> 0;
    const sweep = faceState[4 * f + 3], cap = Math.abs(sweep);
    if (!(cap > 0)) continue;
    const donor = sweep >= 0 ? a : b, receiver = sweep >= 0 ? b : a;
    const donorDense = dense.get(donor);
    if (donor === INVALID || donorDense === undefined) continue;
    const receiverDense = dense.get(receiver);
    if (receiver === INVALID || receiverDense === undefined) { flow.add(donorDense, sink, cap); exteriorCapacity += cap; }
    else { flow.add(donorDense, receiverDense, cap); directedCapacity += cap; }
  }
  const result = flow.solve(source, sink);
  const reachable: number[] = [];
  for (let i = 0; i < n; i++) if (result.reachable[i] >= 0) reachable.push(i);
  const reachableSet = new Set(reachable);
  let cutInitial = 0, cutStorage = 0, cutAllowedRoundoff = 0, cutOutgoingCapacity = 0;
  for (const i of reachable) {
    const cell = cells[i], oldCap = fields.oldCapacity[cell];
    const fraction = snapshot.controlWords
      ? (snapshot.controlWords[3] + 1) / Math.max(1, snapshot.controlWords[2]) : 1;
    const cap = oldCap + fraction * (fields.newCapacity[cell] - oldCap);
    const tol = cap > 0 ? ROUND_EPS * cap : ROUND_EPS * fields.oldCapacity[cell];
    cutInitial += Math.max(0, fields.currentVolume[cell]); cutStorage += cap + (tolerant ? tol : 0); cutAllowedRoundoff += tol;
  }
  for (let f = 0; f < faceCount; f++) {
    const a = metadata[4 * f] >>> 0, b = metadata[4 * f + 1] >>> 0, sweep = faceState[4 * f + 3], cap = Math.abs(sweep);
    const donor = dense.get(sweep >= 0 ? a : b), receiver = dense.get(sweep >= 0 ? b : a);
    if (donor !== undefined && reachableSet.has(donor) && (receiver === undefined || !reachableSet.has(receiver))) cutOutgoingCapacity += cap;
  }
  const closing = reachable.filter(i => fields.newCapacity[cells[i]] === 0).map(i => cells[i]);
  const cut = reachable.slice(0, 24).map(i => ({
    cell: cells[i],
    center: descriptors.slice(8 * cells[i], 8 * cells[i] + 3),
    volume: fields.currentVolume[cells[i]], oldCapacity: fields.oldCapacity[cells[i]], newCapacity: fields.newCapacity[cells[i]],
  }));
  return { tolerant, physicalNodes: snapshot.templateCellCount, acceptedLiquidCells: n, faces: faceCount, target, maxFlow: result.total,
    deficit: target - result.total, negativeVolume, storage, allowedRoundoff,
    exteriorCapacity, directedCapacity, reachableCount: reachable.length,
    reachableCells: reachable.map(i => cells[i]),
    cutInitial, cutStorage, cutAllowedRoundoff, cutOutgoingCapacity,
    cutDeficit: cutInitial - cutStorage - cutOutgoingCapacity,
    reachableClosingCount: closing.length, reachableClosingCells: closing, reachableSample: cut };
}

function dualResidual() {
  const volume = cells.map(cell => fields.currentVolume[cell]);
  for (let f = 0; f < faceCount; f++) {
    const a = metadata[4 * f] >>> 0, b = metadata[4 * f + 1] >>> 0;
    const sweep = faceState[4 * f + 3], mag = Math.abs(sweep);
    if (!(mag > 0)) continue;
    const donor = sweep >= 0 ? a : b, receiver = sweep >= 0 ? b : a;
    const donorDense = dense.get(donor); if (donor === INVALID || donorDense === undefined) continue;
    const receiverDense = dense.get(receiver);
    const base = Math.max(0, Math.min(mag, sweep >= 0 ? faceState[4 * f] : -faceState[4 * f]));
    const yd = fields.currentDual[donor] ?? 0, yr = receiverDense === undefined ? 0 : fields.currentDual[receiver];
    const flux = Math.max(0, Math.min(mag, base + mag * (yd - yr)));
    volume[donorDense] -= flux; if (receiverDense !== undefined) volume[receiverDense] += flux;
  }
  const invalid = [];
  for (let i = 0; i < n; i++) {
    const cell = cells[i], oldCap = fields.oldCapacity[cell];
    const fraction = snapshot.controlWords
      ? (snapshot.controlWords[3] + 1) / Math.max(1, snapshot.controlWords[2]) : 1;
    const cap = oldCap + fraction * (fields.newCapacity[cell] - oldCap);
    const tol = cap > 0 ? ROUND_EPS * cap : ROUND_EPS * oldCap;
    if (volume[i] < -tol || volume[i] > cap + tol) invalid.push({ cell, volume: volume[i], cap, tol, dual: fields.currentDual[cell], proposed: fields.proposedProximal[cell] });
  }
  invalid.sort((a, b) => Math.max(Math.abs(b.volume - b.cap), -b.volume) - Math.max(Math.abs(a.volume - a.cap), -a.volume));
  return { invalidCount: invalid.length, worst: invalid.slice(0, 20) };
}

function gclResidual() {
  if (!snapshot.controlWords || !fields.pressureMembership) return null;
  const count = Math.max(1, snapshot.controlWords[2]);
  const net = new Map(cells.map(cell => [cell, 0]));
  for (let f = 0; f < faceCount; f++) {
    const a = metadata[4 * f] >>> 0, b = metadata[4 * f + 1] >>> 0, sweep = faceState[4 * f + 3];
    if (net.has(a)) net.set(a, net.get(a)! - sweep);
    if (net.has(b)) net.set(b, net.get(b)! + sweep);
  }
  return cells.filter(cell => fields.pressureMembership[cell] > 0.5).map(cell => {
    const expected = (fields.newCapacity[cell] - fields.oldCapacity[cell]) / count;
    return { cell, residual: net.get(cell)! - expected, netSweep: net.get(cell), expected,
      rhs: fields.rhs[cell], diagonal: fields.diagonal[cell], pressureResidual: fields.residual[cell],
      sourceRate: fields.sourceRate[cell], center: descriptors.slice(8 * cell, 8 * cell + 3) };
  }).sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual)).slice(0, 32);
}

function projectedEndpointAudit() {
  if (!snapshot.controlWords || fields.densityA === undefined) return null;
  const count = Math.max(1, snapshot.controlWords[2]);
  const completed = snapshot.controlWords[3] === snapshot.controlWords[2];
  const sourceParity = completed ? snapshot.acceptedScalarParity ^ 1 : snapshot.acceptedScalarParity;
  const sourceDensity = sourceParity ? fields.densityB : fields.densityA;
  const net = new Map(cells.map(cell => [cell, 0]));
  for (let f = 0; f < faceCount; f++) {
    const a = metadata[4 * f] >>> 0, b = metadata[4 * f + 1] >>> 0, sweep = faceState[4 * f + 3];
    if (net.has(a)) net.set(a, net.get(a)! - sweep);
    if (net.has(b)) net.set(b, net.get(b)! + sweep);
  }
  const rows = cells.map(cell => {
    const cellVolume = descriptors[8 * cell + 3], initialVolume = sourceDensity[cell] * cellVolume;
    const finalCapacity = fields.newCapacity[cell], projectedDelta = net.get(cell)! * count;
    return { cell, initialVolume, finalCapacity, projectedDelta,
      endpointVolume: initialVolume + projectedDelta,
      pressureMember: fields.pressureMembership[cell] > 0.5,
      center: descriptors.slice(8 * cell, 8 * cell + 3) };
  });
  const nonmemberHeadroomFailures = rows.filter(row => !row.pressureMember
    && row.endpointVolume > row.finalCapacity + ROUND_EPS * Math.max(row.finalCapacity, fields.oldCapacity[row.cell]))
    .sort((a, b) => (b.endpointVolume - b.finalCapacity) - (a.endpointVolume - a.finalCapacity));
  const lowDensityMembers = rows.filter(row => row.pressureMember
    && row.initialVolume < .5 * fields.oldCapacity[row.cell]);
  return { completed, sourceParity, nonmemberHeadroomFailureCount: nonmemberHeadroomFailures.length,
    nonmemberHeadroomFailures: nonmemberHeadroomFailures.slice(0, 32),
    lowDensityMemberCount: lowDensityMembers.length, lowDensityMembers: lowDensityMembers.slice(0, 32) };
}

console.log(JSON.stringify({ exact: run(false), declaredTolerance: run(true),
  currentDual: dualResidual(), pressureGclWorst: gclResidual(),
  projectedEndpoint: projectedEndpointAudit() }, null, 2));
