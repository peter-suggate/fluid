/** Parity harness for lane F's pure formulas (lib/methods/narrow-band/*-formulas.wgsl.ts): compiles a kernel of the
 * old engine's resolved text (parity/uniform-dam.wgsl, tools/nbflip-parity-extract.ts) and one of the new pure
 * function on the same inputs, and returns both outputs' bits. GPU tests skip without WEBGPU_NODE_MODULE. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../../../harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../../../core/webgpu-device-limits";

export const parityGPU = !!process.env.WEBGPU_NODE_MODULE;

const fixtures = new Map<string, Map<string, string>>();
/** The old engine's resolved function `name`, from the dam's fixture (parity/uniform-dam.wgsl). */
export function oldFunction(name: string): string { return oldFunctionIn("uniform-dam", name); }
/** The old engine's resolved function `name` from fixture `file` ("uniform-tension": the capillarity's, resolved for
 * water-box-dam-break). */
export function oldFunctionIn(file: string, name: string): string {
  let fixture = fixtures.get(file);
  if (!fixture) {
    fixture = new Map(); fixtures.set(file, fixture);
    const parts = readFileSync(resolve(import.meta.dirname, `${file}.wgsl`), "utf8").split(/^\/\/ ==== /m).slice(1);
    for (const part of parts) { const name = /^(\w+)/.exec(part)![1]!; fixture.set(name, part.slice(part.indexOf("\n") + 1).trimEnd()); }
  }
  return fixture.get(name) ?? assert.fail(`parity fixture ${file} has no ${name}`);
}
/** An excerpt of the old function `name`, checked to occur in it verbatim (whitespace runs compared as one). */
export function oldExcerpt(name: string, excerpt: string): string {
  const squash = (s: string) => s.replace(/\s+/g, " ").trim();
  assert.ok(squash(oldFunction(name)).includes(squash(excerpt)), `excerpt not verbatim in old ${name}:\n${excerpt}`);
  return excerpt;
}

export async function withParityDevice(run: (device: GPUDevice) => Promise<void>): Promise<void> {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => { e.preventDefault(); errors.push((e as GPUUncapturedErrorEvent).error.message); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

export interface ParityKernel {
  /** WGSL before the entry point: the functions under test and any stubs. */
  readonly prelude: string;
  /** The body of `fn main`, with `t` the thread index (t < threads); it reads `inp`, `aux` and any bindings the
   * prelude declares, and writes `outp[OUT * t + j]`. */
  readonly body: string;
}
/** Bindings every parity kernel sees: inputs as words (read with bitcast), a second word buffer and the output
 * words. A kernel declares further storage bindings (3 and up) in its prelude and passes their contents as `extra`;
 * it must use each (the layout is the pipeline's own). */
const BINDINGS = /* wgsl */`
@group(0) @binding(0) var<storage,read> inp:array<u32>;
@group(0) @binding(1) var<storage,read> aux:array<u32>;
@group(0) @binding(2) var<storage,read_write> outp:array<u32>;
`;
export const PARITY_READERS = /* wgsl */`
fn f(i:u32)->f32{return bitcast<f32>(inp[i]);}
fn v3(i:u32)->vec3f{return vec3f(f(i),f(i+1u),f(i+2u));}
fn put(i:u32,x:f32){outp[i]=bitcast<u32>(x);}
fn put3(i:u32,x:vec3f){put(i,x.x);put(i+1u,x.y);put(i+2u,x.z);}
`;

function storage(device: GPUDevice, words: Uint32Array | Float32Array, usage = 0): GPUBuffer {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(words.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | usage });
  device.queue.writeBuffer(b, 0, words.buffer, words.byteOffset, words.byteLength);
  return b;
}

/** The module `runParity` compiles (also for offline validation with naga). */
export function parityCode(kernel: ParityKernel, threads: number): string {
  return `${kernel.prelude}\n${BINDINGS}${PARITY_READERS}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 _=inp[0];_=aux[0];let t=gid.x;if(t>=${threads}u){return;}
 ${kernel.body}
}`;
}

/** Runs `kernel` over `threads` threads; returns `outWords` output words. */
export async function runParity(device: GPUDevice, kernel: ParityKernel, io: { inp: Uint32Array | Float32Array; aux?: Uint32Array | Float32Array; extra?: Readonly<Record<number, Uint32Array | Float32Array>>; threads: number; outWords: number }): Promise<Uint32Array> {
  const code = parityCode(kernel, io.threads);
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === "error");
  assert.equal(errs.length, 0, `WGSL errors:\n${errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const out = device.createBuffer({ size: Math.max(16, io.outWords * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: storage(device, io.inp) } },
    { binding: 1, resource: { buffer: storage(device, io.aux ?? new Uint32Array(4)) } },
    { binding: 2, resource: { buffer: out } },
  ];
  for (const [binding, words] of Object.entries(io.extra ?? {})) entries.push({ binding: Number(binding), resource: { buffer: storage(device, words) } });
  // A binding the pipeline does not use leaves the auto layout without it, the bind group invalid and the output
  // zero: fail here rather than compare two zeroed outputs.
  device.pushErrorScope("validation");
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const read = device.createBuffer({ size: out.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(io.threads / 64)); pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, out.size); device.queue.submit([enc.finish()]);
  const invalid = await device.popErrorScope();
  assert.equal(invalid, null, `parity kernel invalid: ${invalid?.message}`);
  await read.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(read.getMappedRange().slice(0, io.outWords * 4)); read.unmap();
  return words;
}

/** Compares two output word arrays as f32: the count of words that differ in bits, the largest difference in ulps
 * among them and where (NaN-free inputs; +0 and -0 count as one ulp apart), and the largest absolute difference. */
export function ulpCompare(a: Uint32Array, b: Uint32Array): { differ: number; maxUlp: number; first?: number; worst?: number; maxAbs: number } {
  assert.equal(a.length, b.length);
  const ord = (w: number) => (w & 0x80000000 ? 0x80000000 - (w & 0x7fffffff) - 1 : 0x80000000 + w);
  const fa = new Float32Array(a.buffer, a.byteOffset, a.length), fb = new Float32Array(b.buffer, b.byteOffset, b.length);
  let differ = 0, maxUlp = 0, maxAbs = 0, first: number | undefined, worst: number | undefined;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) continue;
    differ++; first ??= i;
    const ulp = Math.abs(ord(a[i]!) - ord(b[i]!));
    if (ulp > maxUlp) { maxUlp = ulp; worst = i; }
    maxAbs = Math.max(maxAbs, Math.abs(fa[i]! - fb[i]!));
  }
  return { differ, maxUlp, first, worst, maxAbs };
}

/** A deterministic PRNG for synthetic inputs (mulberry32). */
export function prng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
