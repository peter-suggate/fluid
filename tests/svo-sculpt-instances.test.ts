import assert from "node:assert/strict";
import test from "node:test";
import type { SvoPrimitiveDescriptor } from "../lib/svo/contracts/svo-primitive-abi";
import { packSvoSculptCuts } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { classifySvoSculptInstances } from "../lib/svo/features/scene-publication/svo-sculpt-instances";

test("appended cuts retain affected instances; undo, replacement and scenery changes reclassify", () => {
  let visited = 0;
  const primitive = (x: number): SvoPrimitiveDescriptor => ({
    kind: "field-program", center_m: { x, y: 0, z: 0 }, envelopeRadii_m: { x: .1, y: .1, z: .1 },
    fieldProgramReference: 0, materialId: 1, primitiveId: x,
    get orientation() { visited++; return { x: 0, y: 0, z: 0, w: 1 }; },
    program: { ops: [], result: 0, sampledVolume: { asset: "honed-travertine-paver" } },
  });
  const primitives = [primitive(0), primitive(10), primitive(20)];
  const stamp = (x: number) => ({ operation: "clear" as const, affectsScenery: true,
    minimum: [x, 0, 0] as const, maximumExclusive: [x + 1, 1, 1] as const });
  const lattice = { origin_m: [0, 0, 0] as const, cellSize_m: [1, 1, 1] as const };
  const first = packSvoSculptCuts([stamp(0)], lattice);
  const indices = classifySvoSculptInstances(primitives, first);
  assert.deepEqual(indices, [0]);
  const previous = { primitives, cuts: first, indices };
  visited = 0;
  assert.equal(classifySvoSculptInstances(primitives, first, previous), indices);
  assert.equal(classifySvoSculptInstances(primitives, first.slice(), previous), indices);
  assert.equal(visited, 0, "unchanged cuts do not reread primitive bounds");
  const appended = packSvoSculptCuts([stamp(0), stamp(10)], lattice);
  const next = classifySvoSculptInstances(primitives, appended, previous);
  assert.deepEqual(next, [0, 1]);
  assert.equal(visited, 2, "already affected instances need no new bounds test");
  assert.deepEqual(classifySvoSculptInstances(primitives, first, { primitives, cuts: appended, indices: next }), [0]);
  assert.deepEqual(classifySvoSculptInstances(primitives, packSvoSculptCuts([stamp(20)], lattice), previous), [2]);
  assert.deepEqual(classifySvoSculptInstances([primitive(30), ...primitives.slice(1)], first, previous), []);
  assert.deepEqual(classifySvoSculptInstances(primitives, packSvoSculptCuts([], lattice), previous), []);
});
