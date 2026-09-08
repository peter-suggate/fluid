import assert from "node:assert/strict";
import test from "node:test";
import { SPARSE_CM12_RETAINED_RIGID_DISPLACEMENT_HOPS } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

type Cell = Readonly<{ id: number; open: boolean; amount: number }>;
type Face = Readonly<{ a: number; b: number; staticArea: number; distance: number }>;

/** Small independent finite-volume reference. The graph is geometric; IDs
 * address cells but do not choose a route or receive a rounding remainder.
 * Amount packets use f32, like the GPU, with compensated receiver sums.
 */
function displace(cells: readonly Cell[], faces: readonly Face[], hops = SPARSE_CM12_RETAINED_RIGID_DISPLACEMENT_HOPS) {
  const f = Math.fround;
  const adjacency = new Map(cells.map(cell => [cell.id, [] as { id: number; weight: number }[]]));
  for (const face of faces) if (face.staticArea > 0) {
    assert.ok(face.distance > 0);
    const weight = f(face.staticArea / face.distance);
    adjacency.get(face.a)!.push({ id: face.b, weight });
    adjacency.get(face.b)!.push({ id: face.a, weight });
  }
  const distance = new Map(cells.map(cell => [cell.id, cell.open ? 0 : Infinity]));
  const queue = cells.filter(cell => cell.open).map(cell => cell.id);
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const id = queue[cursor]!, nextDistance = distance.get(id)! + 1;
    if (nextDistance > hops) continue;
    for (const face of adjacency.get(id)!) if (nextDistance < distance.get(face.id)!) {
      distance.set(face.id, nextDistance); queue.push(face.id);
    }
  }
  const compensated = (terms: readonly number[]) => {
    let sum = 0, correction = 0;
    for (const value of terms) {
      const next = f(sum + value);
      correction = f(correction + (sum >= value ? f(f(sum - next) + value) : f(f(value - next) + sum)));
      sum = next;
    }
    return f(sum + correction);
  };
  const weights = new Map(cells.map(cell => [cell.id, compensated(adjacency.get(cell.id)!
    .filter(face => distance.get(face.id)! < distance.get(cell.id)!).map(face => face.weight))]));
  for (const cell of cells) if (!cell.open && cell.amount > 0
    && (!Number.isFinite(distance.get(cell.id)) || !weights.get(cell.id))) {
    throw new RangeError(`No admitted static-open route for cell ${cell.id}`);
  }
  let packets = new Map(cells.map(cell => [cell.id, cell.open ? 0 : f(cell.amount)]));
  for (let hop = 0; hop < hops; hop++) {
    const next = new Map<number, number>();
    for (const cell of cells) {
      const terms = cell.open ? [packets.get(cell.id)!] : [];
      for (const face of adjacency.get(cell.id)!) {
        const sourceDistance = distance.get(face.id)!;
        if (!Number.isFinite(sourceDistance) || sourceDistance <= distance.get(cell.id)!) continue;
        terms.push(f(packets.get(face.id)! * f(face.weight / weights.get(face.id)!)));
      }
      next.set(cell.id, compensated(terms));
    }
    packets = next;
  }
  for (const cell of cells) if (!cell.open) assert.equal(packets.get(cell.id), 0, "no mass remains in a closed support");
  return new Map(cells.map(cell => [cell.id, cell.open ? f(cell.amount + packets.get(cell.id)!) : 0]));
}

test("a covered cell shares its full amount equally across six geometric neighbors", () => {
  const cells = [{ id: 51, open: false, amount: 1 },
    ...[83, 2, 96, 14, 70, 31].map(id => ({ id, open: true, amount: 0 }))];
  const faces = cells.slice(1).map(cell => ({ a: 51, b: cell.id, staticArea: 1, distance: 1 }));
  const result = displace(cells, faces);
  for (const cell of cells.slice(1)) assert.equal(result.get(cell.id), Math.fround(1 / 6));
  assert.equal(result.get(51), 0);
  assert.ok(Math.abs([...result.values()].reduce((sum, amount) => sum + amount, 0) - 1) < 4e-8);
  for (let axis = 0; axis < 3; axis++) assert.equal(result.get(cells[1 + 2 * axis]!.id), result.get(cells[2 + 2 * axis]!.id),
    "opposite faces have zero net displacement moment");
});

test("routing uses physical area divided by center distance", () => {
  const cells = [0, 1, 2, 3].map(id => ({ id, open: id !== 0, amount: id === 0 ? 7 : 0 }));
  const faces = [
    { a: 0, b: 1, staticArea: 1, distance: 1 },
    { a: 0, b: 2, staticArea: 2, distance: 1 },
    { a: 0, b: 3, staticArea: 1, distance: 2 },
  ];
  const result = displace(cells, faces);
  for (const [id, expected] of [[1, 2], [2, 4], [3, 1]]) assert.ok(Math.abs(result.get(id!)! - expected!) < 5e-7);
});

function coveredCube() {
  const cells: Cell[] = [], faces: Face[] = [];
  const id = (x: number, y: number, z: number) => x + 5 * (y + 5 * z);
  for (let z = 0; z < 5; z++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) {
    const open = Math.min(x, y, z) === 0 || Math.max(x, y, z) === 4;
    cells.push({ id: id(x, y, z), open, amount: open ? .25 : .375 });
    if (x < 4) faces.push({ a: id(x, y, z), b: id(x + 1, y, z), staticArea: 1, distance: 1 });
    if (y < 4) faces.push({ a: id(x, y, z), b: id(x, y + 1, z), staticArea: 1, distance: 1 });
    if (z < 4) faces.push({ a: id(x, y, z), b: id(x, y, z + 1), staticArea: 1, distance: 1 });
  }
  return { cells, faces, id };
}

test("a multicell covered component conserves mass and reflection symmetry", () => {
  const { cells, faces, id } = coveredCube();
  const result = displace(cells, faces);
  const before = cells.reduce((sum, cell) => sum + cell.amount, 0);
  const after = [...result.values()].reduce((sum, amount) => sum + amount, 0);
  assert.ok(Math.abs(after - before) < 2e-6, `floating packet receipt ${before} -> ${after}`);
  for (let z = 0; z < 5; z++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) {
    const amount = result.get(id(x, y, z))!;
    for (const reflected of [id(4 - x, y, z), id(x, 4 - y, z), id(x, y, 4 - z)]) {
      assert.ok(Math.abs(amount - result.get(reflected)!) < 1e-7);
    }
  }
});

test("native ID relabeling and incidence order do not choose different receivers", () => {
  const { cells, faces } = coveredCube();
  const result = displace(cells, faces);
  const rename = (id: number) => ((id * 37) % 127) + 1000;
  const relabeled = displace([...cells].reverse().map(cell => ({ ...cell, id: rename(cell.id) })),
    [...faces].reverse().map(face => ({ ...face, a: rename(face.b), b: rename(face.a) })));
  for (const cell of cells) assert.ok(Math.abs(result.get(cell.id)! - relabeled.get(rename(cell.id))!) < 1e-7);
});

test("a static wall or an excessive closed extent rejects the transaction", () => {
  const blocked = [{ id: 0, open: false, amount: 1 }, { id: 1, open: true, amount: 0 }];
  assert.throws(() => displace(blocked, [{ a: 0, b: 1, staticArea: 0, distance: 1 }]), /No admitted static-open route/);
  const { cells, faces } = coveredCube();
  assert.throws(() => displace(cells, faces, 1), /No admitted static-open route/);
  assert.equal(blocked[0]!.amount, 1, "failed route validation does not debit the source");
});
