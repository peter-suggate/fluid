import assert from "node:assert/strict";
import test from "node:test";
import { PARTICLE_MASS_HEADER_WORDS, particleMassReadbackBytes, particleMassRecords } from "../tools/particle-mass-readback";

for (const stride of [12, 28]) {
  test(`particle mass reads ${stride}-float records, current spills and no stale tail`, () => {
    const capacity = 8192;
    const raw = new ArrayBuffer(particleMassReadbackBytes(capacity, stride));
    const f = new Float32Array(raw), u = new Uint32Array(raw);
    u[0] = 3; u[21] = 1; // Three slab slots and one spill; slot 1 is dead.
    for (const i of [0, 2, 3, 4, capacity - 1]) {
      const at = PARTICLE_MASS_HEADER_WORDS + i * stride;
      f[at] = i; f[at + 1] = 5; f[at + 2] = 6; f[at + 3] = 1;
      f[at + 7] = -2;
      if (stride === 28) f[at + 25] = 7;
      else u[at + 11] = 0x80000007; // Ballistic flag does not affect mass.
    }
    const live = [...particleMassRecords(raw, capacity, stride, true)];
    assert.deepEqual(live.map(p => p.x), [0, 2, 3]);
    assert.equal(live.reduce((sum, p) => sum + p.mass, 0), 24);
    assert.deepEqual(live[0], { x: 0, y: 5, z: 6, depth: -2, mass: 8 });
    assert.equal([...particleMassRecords(raw, capacity, stride, false)].reduce((sum, p) => sum + p.mass, 0), 3);
  });
}
