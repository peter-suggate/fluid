import assert from "node:assert/strict";
import test from "node:test";

interface Edge { from: number; to: number; capacity: number; base: number }

function maximumEvacuation(initial: readonly number[], capacity: readonly number[], edges: readonly Edge[]) {
  const n = initial.length, source = n, sink = n + 1;
  const residual = Array.from({ length: n + 2 }, () => new Float64Array(n + 2));
  let required = 0;
  for (const edge of edges) residual[edge.from]![edge.to] += edge.capacity;
  for (let i = 0; i < n; i++) {
    const excess = Math.max(0, initial[i]! - capacity[i]!);
    const room = Math.max(0, capacity[i]! - initial[i]!);
    residual[source]![i] = excess; residual[i]![sink] = room; required += excess;
  }
  let flow = 0;
  while (true) {
    const parent = new Int32Array(n + 2).fill(-1); parent[source] = source;
    const queue = [source];
    for (let head = 0; head < queue.length && parent[sink] < 0; head++) {
      const from = queue[head]!;
      for (let to = 0; to < n + 2; to++) if (parent[to] < 0 && residual[from]![to]! > 1e-14) {
        parent[to] = from; queue.push(to);
      }
    }
    if (parent[sink] < 0) break;
    let amount = Number.POSITIVE_INFINITY;
    for (let at = sink; at !== source; at = parent[at]!) amount = Math.min(amount, residual[parent[at]!]![at]!);
    for (let at = sink; at !== source; at = parent[at]!) {
      const from = parent[at]!; residual[from]![at] -= amount; residual[at]![from] += amount;
    }
    flow += amount;
  }
  return { feasible: Math.abs(flow - required) <= 1e-12, flow, required };
}

function solveDual(initial: readonly number[], capacity: readonly number[], edges: readonly Edge[], limit = 1024) {
  const n = initial.length, y = new Float64Array(n), prior = new Float64Array(n);
  let volume = new Float64Array(initial);
  for (let pass = 0; pass < limit; pass++) {
    volume = new Float64Array(initial); const degree = new Float64Array(n);
    for (const edge of edges) {
      const flux = Math.max(0, Math.min(edge.capacity,
        edge.base + edge.capacity * (y[edge.from]! - y[edge.to]!)));
      volume[edge.from] -= flux; volume[edge.to] += flux;
      degree[edge.from] += edge.capacity; degree[edge.to] += edge.capacity;
    }
    if (volume.every((v, i) => v >= -1e-10 && v <= capacity[i]! + 1e-10)) {
      return { converged: true, passes: pass + 1, volume };
    }
    const x = new Float64Array(n), beta = pass / (pass + 3);
    for (let i = 0; i < n; i++) {
      if (degree[i] === 0) { x[i] = y[i]!; continue; }
      const tau = 1 / (2 * degree[i]!);
      x[i] = Math.max(0, y[i]! + tau * (volume[i]! - capacity[i]!))
        + Math.min(0, y[i]! + tau * volume[i]!);
    }
    for (let i = 0; i < n; i++) {
      y[i] = x[i]! + beta * (x[i]! - prior[i]!); prior[i] = x[i]!;
    }
  }
  return { converged: false, passes: limit, volume };
}

test("moving dual evacuates a feasible directed closing chain", () => {
  const count = 32;
  const initial = Array.from({ length: count }, (_, i) => i === 0 ? 1 : 0);
  const capacity = Array.from({ length: count }, (_, i) => i === count - 1 ? 1 : 0);
  const edges = Array.from({ length: count - 1 }, (_, i) => ({ from: i, to: i + 1, capacity: 1, base: 0 }));
  assert.deepEqual(maximumEvacuation(initial, capacity, edges), { feasible: true, flow: 1, required: 1 });
  const result = solveDual(initial, capacity, edges);
  assert.equal(result.converged, true);
  assert.ok(result.passes < 1024, `feasible chain required ${result.passes} passes`);
  result.volume.forEach((volume, i) => assert.ok(volume >= -1e-10 && volume <= capacity[i]! + 1e-10));
});

test("moving dual does not accept an infeasible reverse-only closing component", () => {
  const initial = [1, 0], capacity = [0, 1];
  const edges = [{ from: 1, to: 0, capacity: 1, base: 0 }];
  assert.deepEqual(maximumEvacuation(initial, capacity, edges), { feasible: false, flow: 0, required: 1 });
  assert.equal(solveDual(initial, capacity, edges, 128).converged, false);
});

test("moving dual retains a bounded directed cycle through-flow", () => {
  const initial = [1, 1, 1], capacity = [1, 1, 1];
  const edges = [0, 1, 2].map(from => ({ from, to: (from + 1) % 3, capacity: 1, base: 0.4 }));
  const result = solveDual(initial, capacity, edges);
  assert.equal(result.converged, true); assert.equal(result.passes, 1);
  result.volume.forEach((volume, i) => assert.ok(Math.abs(volume - initial[i]!) <= 1e-12));
});
