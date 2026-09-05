import assert from "node:assert/strict";
import test from "node:test";
import { compileSparseCM12StableLeafFaceNeighbors } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology";

type Cube = { q: readonly [number, number, number]; span: number };
const compile = (cubes: readonly Cube[]) => compileSparseCM12StableLeafFaceNeighbors({
  coordinates: cubes.map((cube) => cube.q), spans: cubes.map((cube) => cube.span),
});
// Independent box-overlap oracle; never uses the production origin index.
const oracle = (cubes: readonly Cube[]) => cubes.map((a, i) => cubes.flatMap((b, j) => {
  if (i === j) return [];
  const sharesFace = [0, 1, 2].some((axis) =>
    (a.q[axis]! + a.span === b.q[axis] || b.q[axis]! + b.span === a.q[axis])
    && [0, 1, 2].filter((t) => t !== axis).every((t) =>
      Math.min(a.q[t]! + a.span, b.q[t]! + b.span) > Math.max(a.q[t]!, b.q[t]!)));
  return sharesFace ? [j] : [];
}));

test("stable face discovery follows sparse dyadic coverage, including signed roots", () => {
  let cubes: Cube[] = [{ q: [-16, 0, 0], span: 16 }, { q: [0, 0, 0], span: 16 }];
  let random = 42;
  const next = () => (random = (Math.imul(random, 1664525) + 1013904223) >>> 0);
  for (let step = 0; step < 40; step += 1) {
    const choices = cubes.map((c, i) => c.span > 1 ? i : -1).filter((i) => i >= 0);
    const at = choices[next() % choices.length]!;
    const parent = cubes[at]!, half = parent.span / 2;
    const children: Cube[] = [];
    for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
      // Omitted octants exercise holes rather than merely a dense partition.
      if (next() % 5 === 0) continue;
      children.push({ q: [parent.q[0] + x * half, parent.q[1] + y * half,
        parent.q[2] + z * half], span: half });
    }
    cubes.splice(at, 1, ...children);
    assert.deepEqual(compile(cubes), oracle(cubes));
  }
  cubes = cubes.reverse();
  assert.deepEqual(compile(cubes), oracle(cubes));
});

test("a million-brick macro face does not materialize its covered volume or area", () => {
  const span = 2 ** 20;
  const cubes: Cube[] = [
    { q: [0, 0, 0], span }, { q: [span, 0, 0], span },
    { q: [-1, 0, 0], span: 1 }, { q: [-1, span - 1, span - 1], span: 1 },
    { q: [0, span, 0], span },
  ];
  assert.deepEqual(compile(cubes), oracle(cubes));
});

test("stable geometry rejects overlaps in either order and invalid dyadic origins", () => {
  const nested: Cube[] = [{ q: [0, 0, 0], span: 4 }, { q: [1, 1, 1], span: 1 }];
  assert.throws(() => compile(nested), /overlap/);
  assert.throws(() => compile(nested.reverse()), /overlap/);
  for (const cube of [{ q: [1, 0, 0], span: 2 }, { q: [0, 0, 0], span: 3 },
    { q: [NaN, 0, 0], span: 1 }] as Cube[]) {
    assert.throws(() => compile([cube]), /aligned dyadic/);
  }
});
