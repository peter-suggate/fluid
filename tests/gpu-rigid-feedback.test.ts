import assert from "node:assert/strict";
import test from "node:test";
import { consumeGPURigidLoad, decodeGPURigidLoad, legacyUniformComputeShader, mergeGPURigidLoads, type GPURigidLoad } from "../lib/webgpu-eulerian";
import { tallCellComputeShader } from "../lib/tall-cell-kernels";

const load = (impulse: number, interval: number): GPURigidLoad => ({
  bodyId: "body",
  impulse_N_s: { x: impulse, y: 0, z: 0 },
  angularImpulse_N_m_s: { x: 0, y: impulse / 2, z: 0 },
  couplingInterval_s: interval,
  displacedVolume_m3: 0.001,
  meanFluidVelocity_m_s: { x: impulse, y: 0, z: 0 }
});

test("GPU rigid impulse is consumed once over its source interval", () => {
  const pending = load(0.8, 0.008);
  const first = consumeGPURigidLoad(pending, 0.004), second = consumeGPURigidLoad(pending, 0.004), third = consumeGPURigidLoad(pending, 0.004);
  assert.ok(Math.abs(first.impulse_N_s.x - 0.4) < 1e-12);
  assert.ok(Math.abs(second.impulse_N_s.x - 0.4) < 1e-12);
  assert.equal(third.impulse_N_s.x, 0);
  assert.ok(Math.abs(first.impulse_N_s.x + second.impulse_N_s.x - 0.8) < 1e-12);
});

test("GPU readbacks queue impulses instead of replacing or repeating them", () => {
  const [pending] = mergeGPURigidLoads([load(0.25, 0.004)], [load(0.35, 0.004)]);
  assert.ok(Math.abs(pending.impulse_N_s.x - 0.6) < 1e-12);
  assert.ok(Math.abs(pending.couplingInterval_s - 0.008) < 1e-12);
  assert.equal(pending.meanFluidVelocity_m_s.x, 0.35, "the latest fluid snapshot replaces the stale one");
  const delivered = consumeGPURigidLoad(pending, 0.004).impulse_N_s.x + consumeGPURigidLoad(pending, 0.004).impulse_N_s.x;
  assert.ok(Math.abs(delivered - 0.6) < 1e-12);
});

test("rigid exchange decoding averages snapshots without dividing impulses", () => {
  const words = new Int32Array(24), base = 12;
  words[base] = 600_000;
  words[base + 6] = 6 * 65536;
  words[base + 7] = 12 * 10000;
  words[base + 8] = -6 * 10000;
  words[base + 9] = 3 * 10000;
  const decoded = decodeGPURigidLoad("body", words, 1, 0.03, 0.001, 3);
  assert.equal(decoded.impulse_N_s.x, 0.6);
  assert.equal(decoded.displacedVolume_m3, 0.002);
  assert.deepEqual(decoded.meanFluidVelocity_m_s, { x: 2, y: -1, z: 0.5 });
});

test("GPU coupling shaders publish twelve-word wet-velocity snapshots", () => {
  assert.match(legacyUniformComputeShader, /base=bodyIndex\*12u/);
  assert.match(legacyUniformComputeShader, /phi\*ambientVelocity\.x\*10000\.0/);
  assert.match(tallCellComputeShader, /exchangeBase=owner\*12u/);
  assert.match(tallCellComputeShader, /alpha\*solid\*ambientVelocity\.x\*1e4/);
  assert.match(tallCellComputeShader, /var phiNext=phi/);
  assert.match(tallCellComputeShader, /rigidVelocityAt\(neighborWorld\)\.w==0\.0/);
  assert.match(tallCellComputeShader, /reaction=-fluidImpulse\*select\(0\.0,1\.0,solid>0\.9\)/,
    "partially covered collocated cells must not feed their grid-phase impulse back to the body");
  assert.match(tallCellComputeShader, /fluidOpen\*params\.cellGravity\.w\*dt/,
    "body-interior liquid samples must not accumulate gravity before rigid coupling");
  assert.match(tallCellComputeShader, /if\(solidFractionCell\(plus\)>0\.9\)\{v\[axis\]=solidVelocityCell\(plus\)\[axis\];\}else if\(solidFractionCell\(q\)>0\.9\)\{v\[axis\]=solidVelocityCell\(q\)\[axis\];\}/,
    "projection must apply the same moving-solid face constraint as divergence");
});
