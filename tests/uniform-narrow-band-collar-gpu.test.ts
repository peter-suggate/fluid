import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandActivityWGSLFor } from "../lib/methods/uniform/uniform-narrow-band-activity.wgsl";
import { narrowBandCollarTargets, NARROW_BAND_COLLAR_SPREADS } from "../lib/methods/uniform/uniform-narrow-band-collar";
import { UNIFORM_STAGE_IMPORTANCE } from "../lib/methods/uniform/uniform-stage-grids";
import { withUniformDevice } from "./helpers/uniform-geometric";

// The solver's own activitySpread entry, run on a synthetic 7 x 6 x 5 tile grid with absent tiles,
// dry tiles and hot ones, against the CPU model (uniform-narrow-band-collar.ts): under "liquid" a dry
// tile keeps its own heat and a wet one takes today's target; "neighbours" is today's spread.
test("activitySpread on the GPU matches the collar model under each collarSpread", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("collar spread", async device => {
    const N = [7, 6, 5] as const, tiles = N[0] * N[1] * N[2];
    let seed = 11; const r = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
    const heat = new Float32Array(tiles), wet: boolean[] = [], present: boolean[] = [];
    for (let i = 0; i < tiles; i++) { present.push(r() > 0.08); wet.push(r() > 0.4); const u = r(); heat[i] = u < 0.15 ? 2 : u < 0.3 ? Math.fround(2 * r()) : 0; }
    for (const spread of NARROW_BAND_COLLAR_SPREADS) {
      const source = narrowBandActivityWGSLFor(spread), entry = source.slice(source.indexOf("@compute @workgroup_size(64) fn activitySpread"));
      // Words: scores (2 a tile), heat, target, new, then the present mask; the live list is every present tile.
      const SCORES = 0, HEAT = 2 * tiles, TARGET = 3 * tiles, NEW = 4 * tiles, PRESENT = 5 * tiles, LIVE = 6 * tiles, words = 7 * tiles + 1;
      const init = new Uint32Array(words), f = new Float32Array(init.buffer), live = [...Array(tiles).keys()].filter(t => present[t]);
      for (let t = 0; t < tiles; t++) {
        init[SCORES + 2 * t + 1] = (wet[t] ? UNIFORM_STAGE_IMPORTANCE.wet : 0) | (1 << 16);
        f[HEAT + t] = heat[t]!; f[TARGET + t] = t % 3 === 0 ? 0.5 : 0; init[PRESENT + t] = present[t] ? 1 : 0;
      }
      init[LIVE + tiles] = live.length; live.forEach((t, j) => { init[LIVE + j] = t; });
      const bins = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(bins, 0, init);
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
const NB_ACTIVITY_SCORES=${SCORES}u;const NB_ACTIVITY_HEAT=${HEAT}u;const NB_ACTIVITY_TARGET=${TARGET}u;const NB_ACTIVITY_NEW=${NEW}u;
fn umTileAt(p:vec3u)->u32{return p.x+${N[0]}u*(p.y+${N[1]}u*p.z);}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%${N[0]}u,(t/${N[0]}u)%${N[1]}u,t/${N[0] * N[1]}u);}
fn umTilePresent(q:vec3i)->bool{if(any(q<vec3i(0))||any(q>=vec3i(${N[0]},${N[1]},${N[2]}))){return false;}return atomicLoad(&bins[${PRESENT}u+umTileAt(vec3u(q))])!=0u;}
fn umLiveTileCount()->u32{return atomicLoad(&bins[${LIVE + tiles}u]);}
fn umLiveTile(j:u32)->u32{return atomicLoad(&bins[${LIVE}u+j]);}
${entry}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), [], spread);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "activitySpread" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bins } }] });
      const read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const e = device.createCommandEncoder(), p = e.beginComputePass(); p.setPipeline(pipeline); p.setBindGroup(0, group); p.dispatchWorkgroups(2); p.end();
        e.copyBufferToBuffer(bins, 0, read, 0, 4 * words); device.queue.submit([e.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Uint32Array(read.getMappedRange().slice(0)), target = new Float32Array(out.buffer);
        const expected = narrowBandCollarTargets({ n: N, heat, wet, present }, spread), today = narrowBandCollarTargets({ n: N, heat, wet, present }, "neighbours");
        let dryCold = 0;
        for (const t of live) {
          assert.equal(target[TARGET + t], expected[t], `${spread} tile ${t}`);
          assert.equal(out[NEW + t], (t % 3 === 0 ? 0.5 : 0) === 0 && expected[t]! > 0 ? 1 : 0, `${spread} new ${t}`);
          if (spread === "liquid" && !wet[t]) { assert.equal(target[TARGET + t], heat[t]); if (heat[t] === 0 && today[t]! > 0) dryCold++; }
        }
        if (spread === "liquid") assert.ok(dryCold >= 10, `dry cold collar tiles exercised: ${dryCold}`);
      } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); bins.destroy(); }
    }
  });
});
