/** The pressure solve's per-frame record on the pure BULK16 rest tank
 * (lib/core/bulk-rest-tank-scene.ts): each frame's initial residual (the warm
 * start's, header word 26), the native rows' initial (27), the cycle
 * history (8..), the cycles run (4), the accepted residual (2), and the
 * volume drift and max speed at the end. Frames are not pipelined: each is
 * awaited, then its header read.
 *   tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-tank-pressure-trace.ts
 *     [--px=16 --py=8 --pz=8] [--cell=0.01] [--steps=30] [--values='{...}'] [--schedule='{...}'] [--diagnostics]
 * --diagnostics: UniformPressureArena.diagnostics, and per measure (i initial,
 * 0..3 cycles, f final; eps: f32 eps times the start's largest terms) and row class (4 native interior, 5 seam, 6 wall, 3 bulk
 * member, 7 BULK16 page, 2 BULK16 page beside a 4h block,
 * 1 BULK16 page on a wall (initial only)); rho1: a BULK16 row's residual after the
 * first cycle over its start, the largest sign flip (7) and same-sign ratio (0)) the largest residual and the rows past the target. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import "../lib/methods";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createBulkRestTankScene, BULK_REST_TANK_ID } from "../lib/core/bulk-rest-tank-scene";
import { getSceneDefinition } from "../lib/core/scenes";
import type { MethodParamValues } from "../lib/core/method-contract";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { UniformPressureArena, UNIFORM_PRESSURE_ARENA_SCHEDULE } from "../lib/methods/uniform/uniform-pressure-arena";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const cell = Number(option("cell") ?? 0.01), steps = Number(option("steps") ?? 30);
const values: MethodParamValues = JSON.parse(option("values") ?? "{}");
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const scene = createBulkRestTankScene(cell, { pagesX: Number(option("px") ?? 16), pagesY: Number(option("py") ?? 8), pagesZ: Number(option("pz") ?? 8) });
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const diagnostics = process.argv.includes("--diagnostics");
if (diagnostics) UniformPressureArena.diagnostics = true;
// --schedule='{"tailScale":1}': UNIFORM_PRESSURE_ARENA_SCHEDULE overrides.
Object.assign(UNIFORM_PRESSURE_ARENA_SCHEDULE, JSON.parse(option("schedule") ?? "{}"));
const WORDS = diagnostics ? 63 + 128 : 32;
const f = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0]!;
await withUniformDevice("tank pressure trace", async device => {
  const solver = await createUniformSolver(device, scene, { ...getSceneDefinition(BULK_REST_TANK_ID).methodProfile?.overrides, ...values });
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  const index = (frame as unknown as { band: { index: GPUBuffer } }).band.index;
  const read = device.createBuffer({ size: 4 * WORDS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    for (let step = 1; step <= steps; step++) {
      assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s));
      await solver.awaitFrameCompletion();
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(index, 0, read, 0, 4 * WORDS); device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ); const h = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const stats = await solver.readStats();
      const g = (x: number) => +x.toPrecision(3);
      console.log(JSON.stringify({ step, cycles: h[4], accepted: g(f(h[2]!)), initial: g(f(h[26]!)), nativeInitial: g(f(h[27]!)),
        valid: h[25]!.toString(2), history: [...h.slice(8, 13)].map(u => g(f(u))), drift: stats.volumeDrift, maxSpeed: stats.maxSpeed_m_s,
        ...(diagnostics ? { diag: Object.fromEntries([[6, "eps"], [7, "i"], [0, "0"], [1, "1"], [2, "2"], [3, "3"], [4, "f"], [5, "rho1"]].map(([row, name]) => [name,
          Object.fromEntries([0, 1, 2, 3, 4, 5, 6, 7].filter(c => h[63 + 8 * (row as number) + c]! > 0).map(c => [c, `${g(f(h[63 + 8 * (row as number) + c]!))}/${h[63 + 64 + 8 * (row as number) + c]}`]))])) } : {}) }));
    }
  } finally { read.destroy(); solver.destroy(); }
});
