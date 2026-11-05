import {test} from "node:test";
import assert from "node:assert/strict";
import {readFileSync, readdirSync} from "node:fs";
import {join} from "node:path";
import {createNbflipAnalyticScene, NBFLIP_ANALYTIC_DT_S} from "../lib/core/nbflip-analytic-scenes";
import {advanceUniform, createUniformSolver, withUniformDevice} from "./helpers/uniform-geometric";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";

// Every NB-FLIP launch strides its items by the grid it is given
// (num_workgroups), so a narrower grid does the same work. Kernels once
// strided by a literal 65536 (64 lanes x the fixed 1024-workgroup grid):
// correct at the fixed 1024 they are launched at, but a narrower grid
// skipped every item past its own lanes. Sizing a launch by its live work
// needs every kernel to stride by the grid it is given.

const UNIFORM = join(import.meta.dirname, "../lib/methods/uniform");

test("NB-FLIP kernels stride by their launch, not a literal grid", () => {
 for (const name of readdirSync(UNIFORM).filter(n => n.startsWith("uniform-narrow-band"))) {
  const source = readFileSync(join(UNIFORM, name), "utf8");
  const literal = source.split("\n").filter(line => /for\s*\(var\s+\w+=[^;]*;[^;]*;\s*\w+\+=65536u\)/.test(line));
  assert.deepEqual(literal, [], `${name} strides by a literal grid`);
 }
});

/** A device whose NB-FLIP direct launches run at most `width` workgroups. */
function narrowed(raw: GPUDevice, width: number): GPUDevice {
 const labels = new WeakMap<GPUComputePipeline, string>();
 const named = (p: GPUComputePipeline, d: GPUComputePipelineDescriptor) => { labels.set(p, d.label ?? ""); return p; };
 return new Proxy(raw, {get(target, key) {
  if (key === "createComputePipeline") return (d: GPUComputePipelineDescriptor) => named(target.createComputePipeline(d), d);
  if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => named(await target.createComputePipelineAsync(d), d);
  if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
   const encoder = target.createCommandEncoder(d);
   return new Proxy(encoder, {get(e, k) {
    if (k === "beginComputePass") return (pd?: GPUComputePassDescriptor) => {
     const pass = e.beginComputePass(pd); let nb = false;
     return new Proxy(pass, {get(p, pk) {
      if (pk === "setPipeline") return (q: GPUComputePipeline) => { nb = (labels.get(q) ?? "").startsWith("Narrow-band FLIP"); p.setPipeline(q); };
      if (pk === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => p.dispatchWorkgroups(nb ? Math.min(x, width) : x, y, z);
      const v = Reflect.get(p, pk); return typeof v === "function" ? v.bind(p) : v;
     }});
    };
    const v = Reflect.get(e, k); return typeof v === "function" ? v.bind(e) : v;
   }});
  };
  const v = Reflect.get(target, key); return typeof v === "function" ? v.bind(target) : v;
 }}) as GPUDevice;
}

async function band(raw: GPUDevice, width: number): Promise<{n: number; centre: number[]}> {
 const device = narrowed(raw, width);
 const solver = await createUniformSolver(device, createNbflipAnalyticScene("dam-break"), {particleTransfer: "flip"});
 try {
  const flip = (solver as unknown as {mixedFrame: {narrowBandFlip: UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
  for (let frame = 1; frame <= 6; frame++) await advanceUniform(solver, frame * NBFLIP_ANALYTIC_DT_S);
  const stride = flip.particleStride, bytes = flip.activeParticles.size;
  const read = raw.createBuffer({size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const encoder = raw.createCommandEncoder(); encoder.copyBufferToBuffer(flip.activeParticles, 0, read, 0, bytes); raw.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ); const a = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
  let n = 0; const sum = [0, 0, 0];
  for (let o = 0; o + stride <= a.length; o += stride) {
   if (!(a[o]! > 0 && a[o + 3]! > 0)) continue;
   n++; for (let c = 0; c < 3; c++) sum[c]! += a[o + c]!;
  }
  return {n, centre: sum.map(s => s / Math.max(1, n))};
 } finally { solver.destroy(); }
}

test("NB-FLIP launched three workgroups wide seeds and moves the band it does at full width", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
 await withUniformDevice("NB-FLIP launch width", async device => {
  const full = await band(device, Infinity), narrow = await band(device, 3);
  assert.ok(full.n > 1000, `the dam break seeds a band (${full.n} samples)`);
  assert.ok(Math.abs(narrow.n - full.n) <= 0.02 * full.n, `samples ${narrow.n} at 3 workgroups against ${full.n}`);
  for (let c = 0; c < 3; c++) assert.ok(Math.abs(narrow.centre[c]! - full.centre[c]!) <= 0.5,
   `band centre ${narrow.centre.map(v => v.toFixed(2))} at 3 workgroups against ${full.centre.map(v => v.toFixed(2))} (cells)`);
 }, ["subgroups"]);
});
