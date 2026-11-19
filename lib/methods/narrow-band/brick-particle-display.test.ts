/**
 * The brick engine's particle layer (Speed and Motion views, webgpu-particle-overlay.ts):
 * - pDelta stores the Motion energy |velocity - before|^2 of every live record of the bank it files, before it
 *   overwrites `before` (the grid's velocity) with the FLIP delta; a dead record's word is left alone.
 * - The solver's particleSource is the completed step's bank: bank stepsSubmitted & 1, from record bank * PCAP, its
 *   live count PCOUNT[bank], and its live records sit inside the bank.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { BrickLayout, FAIL, H } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions } from "./brick-engine";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import { createBrickSolver } from "./brick-method";
import { P_ENERGY_BUFFER } from "./particles-kernels.wgsl";
import type { BrickStageContext } from "./brick-stages";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const METHOD = "uniform-narrow-band-flip";

function methodValues(): Record<string, unknown> {
  const method = getMethod(METHOD);
  return resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
}

test("pDelta stores the Motion energy before it overwrites the grid velocity, and the stages own its buffer", () => {
  const L = new BrickLayout({ lattice: [32, 32, 32], f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 });
  const context = { layout: L, dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values: methodValues() } as BrickStageContext;
  for (const tiered of [false, true]) {
    const stages = fluidStagesAt(tiered)(context);
    assert.equal(stages.buffers[P_ENERGY_BUFFER], L.cap.particles, `tiered ${tiered}: a word a record of a bank`);
    const delta = stages.tail.find(p => p.kernel.label === "pDelta")!.kernel;
    assert.ok(delta.bindings.some(b => b.name === P_ENERGY_BUFFER && b.kind === "rw"));
    const code = delta.code, store = code.indexOf("pEnergy[i]=dot(motion,motion);");
    assert.ok(store > 0 && store < code.indexOf("p.before=vec4f(nbfDeltaSample(q)") && store < code.indexOf("p.before=vec4f(vec3f(NB_DELTA_NONE)"), `tiered ${tiered}: the store precedes both writes of before`);
  }
});

function tinyDam(): BrickEngineOptions {
  return {
    lattice: [32, 32, 32], h_m: 0.0125, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0], capacities: { f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 },
    liquid: [{ lo: [-64, -64, -64], hi: [16, 12, 96] }], solver: compositeMGPCG(), stages: fluidStagesAt(true), values: methodValues(),
    fluid: { density_kg_m3: 1000, viscosity_Pa_s: 0, surfaceTension_N_m: 0 }, label: "display",
  };
}

test("brick particle display (GPU): pDelta's Motion energy is |velocity - before|^2 of each live record", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick particle display energy", async device => {
    const e = await BrickEngine.create(device, tinyDam());
    try {
      await e.fence();
      const tail = e.schedule.part2, at = tail.findIndex(p => p.kernel.label === "pDelta");
      assert.ok(at > 0);
      const args = [...tail.slice(0, at)].reverse().find(p => p.kernel.label === "pArgs")!;
      // run() binds ring slot 0 (pSrc 0): the tail's pArgs and pDelta read bank 1, from record PCAP.
      const P = e.layout.cap.particles, S = e.layout.particleStride;
      // A coupled record, a ballistic one (the mode in the sign of word 11), a dead one, and a coupled one off any run.
      const records = [
        [4.5, 4.5, 4.5, 1, 0.3, -0.2, 0.1, -0.5, 0.1, -0.1, 0.05, 0],
        [6.5, 5.5, 4.5, 1, -1.5, 2.0, 0.25, -0.5, 0, 0, 0, -0],
        [5.5, 4.5, 6.5, 0, 9, 9, 9, -0.5, 0, 0, 0, 0],
        [10.25, 3.75, 7.5, 1, 0.75, 0.5, -0.125, -2, 0.75, 0.5, -0.125, 0],
      ];
      e.write("parts", S * P, Float32Array.from(records.flat()));
      e.write("set", H.PCOUNT0, Uint32Array.of(0, records.length));
      e.write(P_ENERGY_BUFFER, 0, Float32Array.of(-1, -1, -1, -1));
      await e.run([args, tail[at]!]);
      const energy = new Float32Array((await e.read(P_ENERGY_BUFFER, 0, records.length)).buffer);
      records.forEach((r, i) => {
        if (r[3]! <= 0) { assert.equal(energy[i], -1, "a dead record's word is untouched"); return; }
        const d = [0, 1, 2].map(a => Math.fround(r[4 + a]!) - Math.fround(r[8 + a]!));
        assert.ok(Math.abs(energy[i]! - (d[0]! ** 2 + d[1]! ** 2 + d[2]! ** 2)) < 1e-6, `record ${i}: ${energy[i]} for ${d.join(", ")}`);
      });
      assert.equal(energy[3], 0, "a record moving with the grid has no Motion energy");
      const h = await e.read("set", 0, 256);
      assert.equal(h[H.FAIL] & ~FAIL.PARTICLES, 0);
    } finally { e.destroy(); }
  });
});

test("brick particle display (GPU): the solver's particle source is the completed step's bank", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick particle display banks", async device => {
    const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
    const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
    const solver = await createBrickSolver(device, scene, values as Record<string, unknown>, quality);
    try {
      const e = solver.brickEngine, P = e.layout.cap.particles, S = e.layout.particleStride;
      let seen = 0;
      for (let step = 0; step <= 6; step++) {
        if (step > 0) assert.ok(solver.advanceTo(step / 60 + 1e-6, []), `step ${step} advances`);
        await e.fence();
        const source = solver.particleSource, bank = step & 1, header = await e.read("set", 0, 256);
        assert.equal(header[H.STEP]! & 1, bank, `the GPU's step word agrees with the host's ${step} steps`);
        assert.equal(source.firstRecord, bank * P, `after ${step} steps the records are bank ${bank}'s`);
        assert.equal(source.capacity, P);
        assert.equal(source.liveCount!.buffer, e.buffer("set"));
        assert.equal(source.liveCount!.byteOffset, 4 * (H.PCOUNT0 + bank));
        const n = header[H.PCOUNT0 + bank]!;
        assert.ok(n <= P, `bank ${bank} holds ${n} records`);
        if (n === 0) continue;
        seen++;
        const words = new Float32Array((await e.read("parts", S * bank * P, S * n)).buffer);
        let live = 0;
        for (let i = 0; i < n; i++) if (words[S * i + 3]! > 0) {
          live++;
          for (let a = 0; a < 3; a++) assert.ok(words[S * i + a]! >= 0 && words[S * i + a]! <= e.layout.lattice[a]!, `record ${i} lies in the lattice`);
        }
        assert.ok(live > n / 2, `bank ${bank}: ${live} of ${n} records live`);
        if (step > 0) {
          const energy = new Float32Array((await e.read(P_ENERGY_BUFFER, 0, n)).buffer);
          for (let i = 0; i < n; i++) if (words[S * i + 3]! > 0) assert.ok(Number.isFinite(energy[i]) && energy[i]! >= 0, `record ${i}: energy ${energy[i]}`);
        }
      }
      assert.ok(seen > 0, "the dam seeds records within six steps");
    } finally { solver.destroy(); }
  });
});
