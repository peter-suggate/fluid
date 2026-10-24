import assert from "node:assert/strict";
import test from "node:test";
import { packSvoSculptCuts, sampleSvoSculptCuts } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { prepareSvoSculptCutDistance } from "../lib/svo/features/scene-publication/svo-sculpt-distance";

test("indexed cut distances match the full union inside, outside and across overlapping stretched cuts", () => {
  const cuts = Array.from({ length: 257 }, (_, i) => {
    const low = [i % 17 - 8, (i * 7) % 19 - 9, (i * 11) % 23 - 11] as const;
    const size = i % 5 === 0 ? [1, 64, 2] : [2 + i % 7, 1 + i % 5, 3 + i % 11];
    return { operation: "clear" as const, affectsScenery: true, kind: i % 3 ? "ellipsoid" : "box",
      minimum: low, maximumExclusive: low.map((v, a) => v + size[a]!) as [number, number, number] };
  });
  const lattice = { origin_m: [-2, -3, -4] as const, cellSize_m: [.3, .01, 2] as const };
  const packed = packSvoSculptCuts(cuts, lattice);
  // Also exercise a packed view with a nonzero byte offset.
  const storage = new Uint32Array(packed.length + 7); storage.set(packed, 7);
  const view = storage.subarray(7), distance = prepareSvoSculptCutDistance(view);
  for (let i = 0; i < 1024; i++) {
    const p = [-8 + (i * 73 % 1031) / 1030 * 16, -3.2 + (i * 37 % 1031) / 1030,
      -40 + (i * 19 % 1031) / 1030 * 80] as const;
    assert.ok(Math.abs(distance({ x: p[0], y: p[1], z: p[2] }) - sampleSvoSculptCuts(view, p).distance) < 1e-12);
  }
  assert.equal(prepareSvoSculptCutDistance(packSvoSculptCuts([], lattice))({ x: 0, y: 0, z: 0 }), 1e20);
});

test("the nearest cut keeps deterministic normals on ties and at degenerate centres", () => {
  const words = packSvoSculptCuts([
    { operation: "clear", affectsScenery: true, minimum: [0, 0, 0], maximumExclusive: [2, 2, 2] },
    { operation: "clear", affectsScenery: true, kind: "ellipsoid", minimum: [0, 0, 0], maximumExclusive: [2, 2, 2] },
  ], { origin_m: [0, 0, 0], cellSize_m: [1, 1, 1] });
  assert.deepEqual(sampleSvoSculptCuts(words, [1, 1, 1]), { distance: -1, normal: [1, 0, 0] });
  assert.deepEqual(sampleSvoSculptCuts(words, [3, 1, 1]), { distance: 1, normal: [1, 0, 0] });
  const diagonal = sampleSvoSculptCuts(words, [3, 3, 1]).normal;
  assert.ok(Math.abs(diagonal[0] - Math.SQRT1_2) < 1e-15 && Math.abs(diagonal[1] - Math.SQRT1_2) < 1e-15);
  assert.equal(diagonal[2], 0);
});
