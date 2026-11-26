/**
 * The brick engine through the app's path (lane E): createBrickSolver on the dam with the app's resolved values and
 * the scene's authored pools, driven by the renderer's own advance ledger (submitNextPreparedGPUAdvance,
 * canQueuePreparedGPUAdvance), one presentation completion a frame and the health receipt after it.
 *
 * Regression: the solver set no `info.submittedTime_s`, so the ledger never counted an advance. Every frame then
 * queued another ~100 ms step with no bound, the app froze, and t stayed 0.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { canQueuePreparedGPUAdvance, submitNextPreparedGPUAdvance, BROWSER_GPU_THROUGHPUT_DEPTH } from "../../core/webgpu-renderer";
import { FAIL, H, PHI_AIR_C, brickAbiWGSL } from "./brick-abi";
import { SimulationFailureError } from "../../core/simulation-failure";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import type { BrickNBFlipSolver } from "./brick-solver";
import { createBrickSolver } from "./brick-method";
import { BrickPresentation } from "./brick-present";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };

async function damSolver(device: GPUDevice): Promise<BrickNBFlipSolver> {
  const METHOD = "uniform-narrow-band-flip";
  const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  return createBrickSolver(device, scene, values as Record<string, unknown>, quality);
}

test("brick solver (GPU): the app's ledger advances the dam one step a frame, bounded, and t retires", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick solver app", async device => {
    const solver = await damSolver(device);
    try {
      const dt = 1 / 60, frames = 30;
      let pending = 0, accounted = 0, maxPending = 0;
      const retire: Promise<void>[] = [];
      const t0 = performance.now();
      for (let frame = 1; frame <= frames; frame++) {
        // The renderer's prepared clock runs ahead in real time; the ledger alone bounds the queue.
        const clock = frame * 4 * dt;
        if (canQueuePreparedGPUAdvance(pending, BROWSER_GPU_THROUGHPUT_DEPTH)) {
          const { submittedTime } = submitNextPreparedGPUAdvance(solver, clock, []);
          if (submittedTime > accounted) { accounted = submittedTime; pending++; }
        }
        maxPending = Math.max(maxPending, pending);
        // The presentation's completion retires the admitted advances, then the health receipt (no second fence).
        const done = device.queue.onSubmittedWorkDone();
        const at = accounted;
        retire.push(done.then(async () => { await solver.assertSimulationHealthy(done); solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, at); pending = 0; }));
        if (retire.length >= BROWSER_GPU_THROUGHPUT_DEPTH) await retire.shift();
      }
      await Promise.all(retire);
      const ms = (performance.now() - t0) / frames;
      await solver.awaitFrameCompletion();
      const stats = await solver.readStats();
      const step = stats.brickEngine?.step ?? 0;
      console.log(`brick app: ${frames} frames, ${(stats.submittedTime_s! / dt).toFixed(0)} steps submitted, engine step ${step}, `
        + `completed t ${stats.completedTime_s?.toFixed(4)}, max pending ${maxPending}, ${ms.toFixed(1)} ms/frame (check slot), liveF ${stats.brickEngine?.liveF}`);
      assert.ok(maxPending <= BROWSER_GPU_THROUGHPUT_DEPTH, `the ledger bounds the queue (max pending ${maxPending})`);
      assert.ok((stats.submittedTime_s ?? 0) > 0, "submittedTime_s advances");
      assert.ok((stats.completedTime_s ?? 0) > 0, "completedTime_s retires (t leaves 0)");
      assert.equal(Math.round(stats.submittedTime_s! / solver.brickEngine.options.dt_s), step, "one engine step per accounted advance");
      assert.equal(solver.info.simulationFailure, undefined, "no latched failure");
      const header = await solver.brickEngine.read("set", 0, 4);
      assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info 0x${header[H.FAIL_INFO]!.toString(16)}`);
    } finally { solver.destroy(); }
  });
});

/**
 * The presentation shows the tier authority's phi (brick-owner nbVertexPhi) at every vertex, bitwise, on the tiered
 * dam at t = 70 dt, where 4h-owned regions hold no F brick. The old copy read F bricks only: the probe counts the
 * vertices where that reading's sign differs (grey holes along the floor, stair banding on the side faces in the app).
 */
/**
 * The presentation's plain copy (smooth off) is the tier authority's φ at every vertex. The drawn copy (the solver's)
 * is presentVertex's: the CPU oracle below rebuilds it from the authority, the C corner words and the tiles' coarse
 * bits, and at every tile corner it is the authority's word (the cubic takes the corners, and C mirrors F's word at a
 * fine-owned corner, lcMirror).
 */
test("brick solver (GPU): the presentation is the tier authority's phi at every vertex", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick presentation", async device => {
    const solver = await damSolver(device);
    try {
      const e = solver.brickEngine, L = e.layout, N = L.lattice, dt = e.options.dt_s, steps = 70;
      for (let i = 1; i <= steps; i++) solver.advanceTo(i * dt, []);
      await solver.assertSimulationHealthy();
      const V = N.map(n => n + 1), count = V[0]! * V[1]! * V[2]!;
      const code = `${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> out:array<f32>;
@group(0) @binding(5) var<storage,read_write> corners:array<f32>;
${brickOwnerWGSL({ phi: "phi", vel: "vel" })}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) n:vec3u){
  let i=g.x+64u*n.x*g.y;if(i>=${count}u){return;}
  if(i==0xffffffffu){out[0]=vel[0]+f32(nbSet[0]+dir[0]);} // every binding in the auto layout
  let V=vec3u(${V.join(",")});let v=vec3i(vec3u(i%V.x,(i/V.x)%V.y,i/(V.x*V.y)));
  out[2u*i]=nbVertexPhi(v);
  let s=nbFSlotOfBrick(v>>vec3u(3u));
  out[2u*i+1u]=select(NB_PHI_AIR,bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]),s<NB_SEAM);
  // Tile corner T (0..N/4 an axis): its C word, and whether tile T is resident and 4h-owned.
  let C=vec3u(${N.map(n => n / 4 + 1).join(",")});
  if(i<C.x*C.y*C.z){
    let T=vec3i(vec3u(i%C.x,(i/C.x)%C.y,i/(C.x*C.y)));
    corners[2u*i]=nbCVertex(T);
    corners[2u*i+1u]=select(0.0,1.0,nbTileInLattice(T)&&nbCSlotOfPage(T>>vec3u(3u))!=NB_NONE&&!nbTileOwnerH(T));
  }
}`;
      const module = device.createShaderModule({ code });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      assert.equal(errors.length, 0, errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
      const out = device.createBuffer({ size: 8 * count, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const C = N.map(n => n / 4 + 1), cornerCount = C[0]! * C[1]! * C[2]!;
      const cornerBuf = device.createBuffer({ size: 8 * cornerCount, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const cornerRead = device.createBuffer({ size: 8 * cornerCount, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const farM0 = (solver as unknown as { options: { farPhi_m: number } }).options.farPhi_m;
      const plain = await BrickPresentation.create(e, farM0, N, true, false);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: ["set", "dir", "phi", "vel"].map((name, binding) => ({ binding, resource: { buffer: e.buffer(name) } })).concat([{ binding: 4, resource: { buffer: out } }, { binding: 5, resource: { buffer: cornerBuf } }]) });
      const rowBytes = Math.ceil((4 * V[0]!) / 256) * 256;
      const tex = device.createBuffer({ size: rowBytes * V[1]! * V[2]!, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const smoothTex = device.createBuffer({ size: rowBytes * V[1]! * V[2]!, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const probe = device.createBuffer({ size: 8 * count, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const groups = Math.ceil(count / 64), gx = Math.min(groups, 32768), gy = Math.ceil(groups / gx);
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      plain.encode(enc);
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(gx, gy); pass.end();
      enc.copyBufferToBuffer(out, 0, probe, 0, 8 * count);
      enc.copyBufferToBuffer(cornerBuf, 0, cornerRead, 0, 8 * cornerCount);
      enc.copyTextureToBuffer({ texture: plain.vertexPhi }, { buffer: tex, bytesPerRow: rowBytes, rowsPerImage: V[1]! }, V);
      enc.copyTextureToBuffer({ texture: solver.brickPresentation.vertexPhi }, { buffer: smoothTex, bytesPerRow: rowBytes, rowsPerImage: V[1]! }, V);
      device.queue.submit([enc.finish()]);
      const invalid = await device.popErrorScope();
      assert.equal(invalid, null, `probe: ${invalid?.message}`);
      await Promise.all([tex.mapAsync(GPUMapMode.READ), probe.mapAsync(GPUMapMode.READ), smoothTex.mapAsync(GPUMapMode.READ), cornerRead.mapAsync(GPUMapMode.READ)]);
      const shown = new Float32Array(tex.getMappedRange()), want = new Float32Array(probe.getMappedRange());
      const drawn = new Float32Array(smoothTex.getMappedRange()), cw = new Float32Array(cornerRead.getMappedRange());
      const h = Math.fround(e.options.h_m), farM = (solver as unknown as { options: { farPhi_m: number } }).options.farPhi_m;
      const FAR_H = farM / h;
      let mismatch = 0, first = "", liquid = 0, oldWrong = 0, minQ = Infinity, minOld = Infinity, finite = 0, minShown = Infinity;
      const rowFloats = rowBytes / 4;
      for (let z = 0; z < V[2]!; z++) for (let y = 0; y < V[1]!; y++) for (let x = 0; x < V[0]!; x++) {
        const i = x + V[0]! * (y + V[1]! * z), q = want[2 * i]!, old = want[2 * i + 1]!;
        // The band's 4h air value (PHI_AIR_C) and anything past it show as far (brick-present.ts, since 06d4ca89).
        const expect = q < PHI_AIR_C ? Math.fround(q * h) : Math.fround(farM);
        const got = shown[x + rowFloats * (y + V[1]! * z)]!;
        if (got !== expect) { if (!mismatch) first = `(${x},${y},${z}) shown ${got} want ${expect} (q ${q})`; mismatch++; }
        if (q < 0) liquid++;
        minQ = Math.min(minQ, q); minOld = Math.min(minOld, old); minShown = Math.min(minShown, got); if (q < FAR_H - 1e-3) finite++;
        if ((q < 0) !== (old < 0)) oldWrong++;
      }
      // The drawn copy: presentVertex rebuilt (brick-present.ts), from the authority q, the C corner words and coarse bits.
      const vi = (x: number, y: number, z: number) => x + V[0]! * (y + V[1]! * z);
      const at = (x: number, y: number, z: number) => shown[x + rowFloats * (y + V[1]! * z)]!;
      const cl = (v: number, n: number) => Math.min(Math.max(v, 0), n);
      const corner = (T: number[]) => cw[2 * (cl(T[0]!, C[0]! - 1) + C[0]! * (cl(T[1]!, C[1]! - 1) + C[1]! * cl(T[2]!, C[2]! - 1)))]!;
      const coarse = (t: number[]) => t.every((c, a) => c >= 0 && 4 * c < N[a]!) && cw[2 * (t[0]! + C[0]! * (t[1]! + C[1]! * t[2]!)) + 1] === 1;
      const cr = (t: number) => [0.5 * (-t * t * t + 2 * t * t - t), 0.5 * (3 * t * t * t - 5 * t * t + 2), 0.5 * (-3 * t * t * t + 4 * t * t + t), 0.5 * (t * t * t - t * t)];
      const nb = V.map(n => Math.ceil(n / 8));
      const surface = new Uint8Array(nb[0]! * nb[1]! * nb[2]!);
      for (let bz = 0; bz < nb[2]!; bz++) for (let by = 0; by < nb[1]!; by++) for (let bx = 0; bx < nb[0]!; bx++) {
        let near = false, any = false;
        for (let z = 8 * bz; z < Math.min(8 * bz + 8, V[2]!); z++) for (let y = 8 * by; y < Math.min(8 * by + 8, V[1]!); y++) for (let x = 8 * bx; x < Math.min(8 * bx + 8, V[0]!); x++) {
          if (at(x, y, z) !== Math.fround(farM) && Math.abs(want[2 * vi(x, y, z)]!) < 4) near = true;
        }
        for (let k = 0; k < 64; k++) if (coarse([2 * bx - 1 + (k & 3), 2 * by - 1 + ((k >> 2) & 3), 2 * bz - 1 + (k >> 4)])) any = true;
        surface[bx + nb[0]! * (by + nb[1]! * bz)] = near && any ? 1 : 0;
      }
      let drawnBad = 0, drawnFirst = "", moved = 0, cornerBad = 0, worst = 0;
      for (let z = 0; z < V[2]!; z++) for (let y = 0; y < V[1]!; y++) for (let x = 0; x < V[0]!; x++) {
        const plainV = at(x, y, z), got = drawn[x + rowFloats * (y + V[1]! * z)]!, q = want[2 * vi(x, y, z)]!, v = [x, y, z];
        let expect = plainV;
        const b = v.map(c => c >> 3);
        if (plainV !== Math.fround(farM) && Math.abs(q) < 4 && surface[b[0]! + nb[0]! * (b[1]! + nb[1]! * b[2]!)]) {
          const own = v.map(c => c >> 2);
          let d = 4;
          for (let k = 0; k < 27 && d > 0; k++) {
            const n = [own[0]! + (k % 3) - 1, own[1]! + (Math.floor(k / 3) % 3) - 1, own[2]! + Math.floor(k / 9) - 1];
            if (n.some((c, a) => c < 2 * b[a]! - 1 || c > 2 * b[a]! + 2) || !coarse(n)) continue;
            d = Math.min(d, Math.hypot(...n.map((c, a) => Math.max(4 * c - v[a]!, v[a]! - 4 * c - 4, 0))));
          }
          if (d < 4) {
            const w = own.map((c, a) => cr((v[a]! - 4 * c) / 4));
            let sum = 0, lo = Infinity, hi = -Infinity;
            for (let k = 0; k < 64; k++) {
              const c = [k & 3, (k >> 2) & 3, k >> 4], x0 = corner(own.map((o, a) => o + c[a]! - 1));
              sum += w[0]![c[0]!]! * w[1]![c[1]!]! * w[2]![c[2]!]! * x0;
            }
            for (let k = 0; k < 8; k++) { const x0 = corner(own.map((o, a) => o + ((k >> a) & 1))); lo = Math.min(lo, x0); hi = Math.max(hi, x0); }
            const cubic = Math.min(Math.max(sum, lo), hi), s = Math.min(Math.max(d / 4, 0), 1), ws = s * s * (3 - 2 * s);
            // A vertex in a coarse tile takes the cubic; elsewhere the blend keeps F's sign (q's where the cubic's differs).
            const ownCoarse = coarse(own);
            expect = (!ownCoarse && (cubic < 0) !== (q < 0) ? q : cubic * (1 - ws) + q * ws) * h;
          }
        }
        const err = Math.abs(got - expect) / h; worst = Math.max(worst, plainV === Math.fround(farM) ? 0 : err);
        if (err > 2e-4) { if (!drawnBad) drawnFirst = `(${x},${y},${z}) drawn ${got} want ${expect} (q ${q})`; drawnBad++; }
        if (Math.abs(got - plainV) > 0.01 * h) moved++;
        if (x % 4 === 0 && y % 4 === 0 && z % 4 === 0 && plainV !== Math.fround(farM) && Math.abs(got - plainV) > 2e-4 * h) cornerBad++;
      }
      smoothTex.unmap(); cornerRead.unmap(); smoothTex.destroy(); cornerRead.destroy(); cornerBuf.destroy(); plain.destroy();
      console.log(`drawn copy: ${moved} vertices moved past 0.01 h by presentVertex, worst oracle error ${worst.toExponential(2)} h, ${drawnBad} past 2e-4 h, ${cornerBad} tile corners off the authority`);
      assert.equal(drawnBad, 0, `the drawn copy differs from presentVertex's oracle at ${drawnBad} vertices, first ${drawnFirst}`);
      assert.equal(cornerBad, 0, `the drawn copy leaves the authority at ${cornerBad} tile corners`);
      assert.ok(moved > 0, "presentVertex smooths some 4h vertex near the surface");
      tex.unmap(); probe.unmap(); tex.destroy(); probe.destroy(); out.destroy();
      console.log(`brick presentation at step ${steps}: min q ${minQ} old ${minOld} shown ${minShown}, ${finite} finite of ${count}, FAR_H ${FAR_H}; ${liquid} liquid vertices; the old F-only copy's sign differs at ${oldWrong}; mismatches ${mismatch}`);
      assert.equal(mismatch, 0, `presentation differs from the tier authority at ${mismatch} vertices, first ${first}`);
      assert.ok(liquid > 0, "the dam has liquid vertices");
    } finally { solver.destroy(); }
  });
});

/**
 * A latched FAIL reaches the app as a SimulationFailureError ("Simulation HALTED", the renderer's terminal status via
 * stopAfterSimulationFailure), and the solver refuses further steps: never a silent reset to t = 0 or to another
 * scene, so a halt cannot be confused with a dev-server hot reload.
 */
test("brick solver (GPU): a latched FAIL halts the app explicitly and refuses further steps", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick solver halt", async device => {
    const solver = await damSolver(device);
    try {
      const e = solver.brickEngine, dt = e.options.dt_s;
      solver.advanceTo(dt, []); await solver.assertSimulationHealthy();
      // A kernel's latch, injected: nbFail ORs FAIL and sets FAIL_INFO once; nothing clears them.
      e.write("set", H.FAIL, new Uint32Array([FAIL.REACH, 0x52454143]));
      let halted: unknown;
      for (let i = 2; i <= 12 && !halted; i++) {
        solver.advanceTo(i * dt, []);
        try { await solver.assertSimulationHealthy(); } catch (error) { halted = error; }
      }
      assert.ok(halted instanceof SimulationFailureError && halted.message.startsWith("Simulation HALTED: BRICK_REACH")
        && halted.failure.operands[1] === 0x52454143 && solver.info.simulationFailure?.code === "BRICK_REACH",
        `a latched FAIL is a SimulationFailureError naming it (got ${String(halted)})`);
      const submitted = solver.info.submittedTime_s ?? 0, step = Math.round(submitted / dt);
      assert.equal(solver.advanceTo((step + 5) * dt, []), false, "no step after a halt");
      assert.equal(solver.info.submittedTime_s, submitted, "the clock holds (no reset, no advance)");
    } finally { solver.destroy(); }
  });
});

/**
 * The default engine, "auto", through the method's own createSolverAsync: the dam (no gaps) runs the brick engine and
 * a scene with gaps (the stair cube's interactive bodies; open worlds run on brick since b33d9bec) runs the old engine, each
 * named in info.nbEngine, and each advances ten frames through the app's ledger with no latched failure.
 */
test("brick solver (GPU): auto, the default engine, runs the dam on brick and a gap scene on uniform", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD);
  await withUniformDevice("brick solver auto", async device => {
    for (const [sceneId, ran] of [["high-resolution-dam-break", "brick"], ["nbflip-stair-cube", "uniform"]] as const) {
      const definition = getSceneDefinition(sceneId), scene = sceneDocument(definition), profile = definition.methodProfile;
      const quality = profile?.methodId === METHOD ? profile.quality : "balanced";
      const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}) });
      assert.equal(values.engine, "auto", `${sceneId}: the default engine`);
      const solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
      try {
        assert.equal(solver.info.nbEngine?.ran, ran, `${sceneId}: ${JSON.stringify(solver.info.nbEngine)}`);
        assert.equal("brickEngine" in solver, ran === "brick", `${sceneId}: the solver is the ${ran} engine's`);
        if (ran === "uniform") assert.match(solver.info.nbEngine!.gap ?? "", /interactive bodies/);
        await solver.waitForSimulationReady?.();
        let pending = 0, accounted = 0;
        for (let frame = 1; frame <= 10; frame++) {
          if (canQueuePreparedGPUAdvance(pending, BROWSER_GPU_THROUGHPUT_DEPTH)) {
            const { submittedTime } = submitNextPreparedGPUAdvance(solver, frame * 2 / 60, []);
            if (submittedTime > accounted) { accounted = submittedTime; pending++; }
          }
          const done = device.queue.onSubmittedWorkDone();
          await done; await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.(done);
          solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, accounted); pending = 0;
        }
        console.log(`auto ${sceneId}: ${JSON.stringify(solver.info.nbEngine)}, t ${solver.info.completedTime_s?.toFixed(4)}`);
        assert.ok((solver.info.completedTime_s ?? 0) > 0, `${sceneId}: t leaves 0`);
        assert.equal(solver.info.simulationFailure, undefined, `${sceneId}: no latched failure`);
        assert.equal(solver.info.nbEngine?.ran, ran, `${sceneId}: the status survives the steps`);
      } finally { solver.destroy(); }
    }
  });
});

/**
 * Static voxels through the app's path (lane E, M3): the dam with a block on its floor (P's 0xa2 obstacle, x 96..108,
 * y 0..24, z 40..88; its x = 108 face cuts bricks, so the pools carry K slots for the cuts) runs through the app's
 * createBrickSolver. The scene's own mask reaches the engine, and the front
 * meets the block's face with no latch.
 */
test("brick solver (GPU): the dam against a voxel block, the scene's mask bound through createBrickSolver", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { sceneWithSolidStroke } = await import("../../core/solid-world");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD);
  const definition = getSceneDefinition("high-resolution-dam-break"), dam = sceneDocument(definition);
  const scene = sceneWithSolidStroke(dam, [{ operation: "fill", minimum: [96, 0, 40], maximumExclusive: [108, 24, 88], materialId: 1 }]);
  scene.fluid.brickPools = { ...scene.fluid.brickPools!, k: 512 };
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" }) as Record<string, unknown>;
  await withUniformDevice("brick solver statics", async device => {
    const solver = await createBrickSolver(device, scene, values, quality);
    try {
      const e = solver.brickEngine, dt = e.options.dt_s, steps = 30;
      assert.ok(e.options.staticMask, "the scene's mask is bound");
      for (let i = 1; i <= steps; i++) { solver.advanceTo(i * dt, []); if (i % 10 === 0) await solver.assertSimulationHealthy(); }
      await solver.awaitFrameCompletion();
      const header = await e.read("set", 0, 4);
      const stats = await solver.readStats();
      console.log(`brick statics app: ${steps} steps, engine step ${stats.brickEngine?.step}, liveF ${stats.brickEngine?.liveF}, FAIL ${header[H.FAIL]}`);
      assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info 0x${header[H.FAIL_INFO]!.toString(16)}`);
      assert.equal(stats.brickEngine?.step, steps);
    } finally { solver.destroy(); }
  });
});
