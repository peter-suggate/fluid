/** particleTransfer "apic" on the brick engine (particles-formulas.wgsl.ts nbfApicWGSL; Jiang et al. 2015's quadratic
 * MAC APIC, the old engine's uniform-narrow-band-apic.wgsl.ts): the gate, the record and its layout, every kernel of an
 * APIC layout on the APIC record (and every FLIP kernel without APIC text), and on the GPU (WEBGPU_NODE_MODULE) the
 * transfer's round trip, which carries an affine (rotating and shearing) field exactly where PIC does not. */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { BrickLayout, type BrickLayoutOptions } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
import type { BrickStageContext, BrickStages } from "./brick-stages";
import { fluidLayout } from "./fluid-grid.wgsl";
import { NBF_PARTICLE_STRUCT_APIC, NBF_PARTICLE_STRUCT_FLIP, nbfApicWGSL, nbfP2GApicWGSL, nbfParticleApic, nbfParticleRecordApicWGSL, nbfParticleRecordFor, nbfParticleRecordWGSL, nbfParticleStride, nbfParticleSupported } from "./particles-formulas.wgsl";
import { nbfP2GWGSL } from "./velocity-formulas.wgsl";
import { parityGPU, prng, withParityDevice } from "./parity/harness";
import "../index";

const OPTIONS: BrickLayoutOptions = { lattice: [32, 32, 32], f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 };
const DETAIL: BrickLayoutOptions = { lattice: [64, 64, 64], box: [64, 64, 64], f: 512, c: 8, b: 16, k: 0, particles: 1024, detail: { factor: 2, tiles: [16, 16, 16], budget: 64, slots: 192 } };

function context(layout: BrickLayout, transfer?: string): BrickStageContext {
  const method = getMethod("uniform-narrow-band-flip");
  const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick", ...(transfer ? { particleTransfer: transfer } : {}) });
  return { layout, dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values } as BrickStageContext;
}
const kernels = (stages: BrickStages) => {
  const all = [...stages.move, ...stages.rebin, ...stages.levelSet, ...stages.velocity, ...stages.tail, ...stages.boot,
    ...(stages.stamp ?? []), ...(stages.kick ? [...stages.kick.force, ...stages.kick.after, ...(stages.kick.seed ?? [])] : [])].map(p => p.kernel);
  return [...new Set(all)];
};

test("the brick engine takes particleTransfer flip or apic, and fails explicitly on any other", () => {
  nbfParticleSupported({ particleTransfer: "flip" });
  nbfParticleSupported({ particleTransfer: "apic" });
  nbfParticleSupported({});
  for (const bad of ["pic", "asflip", "APIC", 0, null]) {
    assert.throws(() => nbfParticleSupported({ particleTransfer: bad }), /particleTransfer ".*" is not implemented \(only "flip" or "apic"\)/, String(bad));
    assert.throws(() => nbfParticleApic({ particleTransfer: bad }), /is not implemented/);
  }
  assert.equal(nbfParticleApic({ particleTransfer: "apic" }), true);
  assert.equal(nbfParticleApic({ particleTransfer: "flip" }), false);
  assert.equal(nbfParticleApic({}), false, "FLIP by default");
  // The other settings still take only their one value.
  assert.throws(() => nbfParticleSupported({ particleTransfer: "apic", crowdedCells: "retire" }), /crowdedCells "retire" is not implemented/);
});

test("an APIC layout holds 24-float records at the same capacity, FLIP's 12 first", () => {
  const flip = new BrickLayout(OPTIONS), apic = new BrickLayout({ ...OPTIONS, apic: true });
  assert.equal(flip.apic, false); assert.equal(apic.apic, true);
  assert.equal(flip.particleStride, 12); assert.equal(apic.particleStride, 24);
  assert.equal(nbfParticleStride(false), 12); assert.equal(nbfParticleStride(true), 24);
  assert.equal(apic.cap.particles, flip.cap.particles, "the bigger record is sized from the same capacity");
  assert.equal(apic.partsWords, 2 * OPTIONS.particles * 24);
  assert.equal(fluidLayout(apic).fmoved.words, 24 * OPTIONS.particles);
  // Six vec4f: 96 bytes (WGSL's struct size), the stride. FLIP's struct is the record's first three.
  assert.equal((NBF_PARTICLE_STRUCT_APIC.match(/:vec4f/g) ?? []).length * 4, 24);
  assert.ok(NBF_PARTICLE_STRUCT_APIC.startsWith(NBF_PARTICLE_STRUCT_FLIP.slice(0, -1) + ","), "the FLIP record's words lead");
  // The accessors (mode, age, mass in before.w) are FLIP's, so every kernel reads the same packed word.
  assert.ok(nbfParticleRecordApicWGSL.startsWith(nbfParticleRecordWGSL.replace(NBF_PARTICLE_STRUCT_FLIP, NBF_PARTICLE_STRUCT_APIC)));
  assert.equal(nbfParticleRecordFor(flip), nbfParticleRecordWGSL);
  assert.equal(nbfParticleRecordFor(apic), nbfParticleRecordApicWGSL);
});

test("every kernel of an APIC layout reads the APIC record, and no FLIP kernel carries APIC text", () => {
  for (const [name, options] of [["plain", OPTIONS], ["detail", DETAIL]] as const) for (const tiered of [false, true]) {
    if (options.detail && tiered) continue;
    const flip = kernels(fluidStagesAt(tiered)(context(new BrickLayout(options))));
    const apic = kernels(fluidStagesAt(tiered)(context(new BrickLayout({ ...options, apic: true }), "apic")));
    assert.deepEqual(apic.map(k => k.label), flip.map(k => k.label), `${name} ${tiered}: the same stages`);
    for (const k of flip) {
      assert.ok(!k.code.includes(NBF_PARTICLE_STRUCT_APIC) && !/nbfApicRow|nbfP2GScatterApic|nbApicRowOf|\.cx\b/.test(k.code), `${name} ${tiered} ${k.label}: FLIP text`);
      if (k.code.includes("struct Particle")) assert.ok(k.code.includes(NBF_PARTICLE_STRUCT_FLIP), k.label);
    }
    for (const k of apic) {
      assert.ok(!k.code.includes(NBF_PARTICLE_STRUCT_FLIP), `${name} ${tiered} ${k.label}: no FLIP record`);
      if (/\b(parts|fmoved)\b/.test(k.code)) assert.ok(k.code.includes(NBF_PARTICLE_STRUCT_APIC), `${name} ${tiered} ${k.label}: the APIC record`);
    }
    const code = (label: string) => apic.filter(k => k.label === label).map(k => k.code).join("\n");
    assert.ok(code("pMove").includes("let rows=pApicSample(q);") && code("pMove").includes("fn nbfApicRow("), "G2P takes the rows");
    assert.ok(!code("pMove").includes("nbfG2P(v,p.before.xyz"), "no FLIP blend under APIC");
    assert.ok(code("pDeposit").includes("nbfP2GScatterApic(q,s.velocity.xyz,nbApicRows(s),nbParticleMass(s));"), "P2G deposits the affine velocity");
    assert.ok(!code("pDelta").includes("nbfDeltaSample(q)"), "APIC records take no FLIP delta");
    if (tiered) assert.ok(code("uTransfer").includes("momentum+=m*w*(s.velocity[a]+dot(nbApicRowOf(s,a),r));"), "the 4h gather takes the affine velocity");
  }
});

test("the stages fail where the layout's records are not the values' transfer", () => {
  assert.throws(() => fluidStagesAt(false)(context(new BrickLayout(OPTIONS), "apic")), /particleTransfer "apic" with FLIP records/);
  assert.throws(() => fluidStagesAt(true)(context(new BrickLayout({ ...OPTIONS, apic: true }), "flip")), /particleTransfer "flip" with APIC records/);
  const pic = context(new BrickLayout({ ...OPTIONS, apic: true }));
  assert.throws(() => fluidStagesAt(false)({ ...pic, values: { ...pic.values, particleTransfer: "pic" } }), /particleTransfer "pic" is not implemented/);
});

// ---- GPU: the transfer's round trip on a 12^3 MAC grid. Records one a cell (jittered) carry an affine field
// u(x) = u0 + A (x - x0) (a rigid rotation and a shear): v_p = u(x_p), rows c_a = A[a]. P2G (nbfP2GScatterApic through
// nbxDeposit into the fixed-point words) then G2P (nbfApicRow from the resolved faces). APIC's faces are u at the face
// and its G2P recovers u(p) and A; PIC's deposit (rows 0) misses the faces by the weighted centroid's offset times A.
const D = 12;
const A = [[0.2, -1.1, 0.4], [1.1, -0.1, 0.7], [-0.4, -0.7, -0.1]], U0 = [0.3, -0.5, 0.8], X0 = [6, 6, 6];
const affine = (x: readonly number[], a: number) => U0[a]! + A[a]!.reduce((s, v, k) => s + v * (x[k]! - X0[k]!), 0);
async function readBack(device: GPUDevice, buffer: GPUBuffer): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, read, 0, buffer.size); device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ); const out = read.getMappedRange().slice(0); read.unmap(); read.destroy(); return out;
}
test("APIC's P2G -> G2P carries an affine field exactly (PIC's does not), on the deposit's own weights", { skip: !parityGPU }, async t => {
  const rand = prng(20261123), records: number[][] = [];
  for (let z = 1; z < D - 1; z++) for (let y = 1; y < D - 1; y++) for (let x = 1; x < D - 1; x++) records.push([x + rand(), y + rand(), z + rand()]);
  const points = Array.from({ length: 32 }, () => [4 + 3 * rand(), 4 + 3 * rand(), 4 + 3 * rand()]);
  const n = records.length, faces = 3 * D * D * D;
  // A record: p, v, then the rows cx, cy, cz (vec4f each).
  const input = new Float32Array(20 * n);
  records.forEach((p, i) => { input.set(p, 20 * i); input.set([0, 1, 2].map(a => affine(p, a)), 20 * i + 4); for (let a = 0; a < 3; a++) input.set(A[a]!, 20 * i + 8 + 4 * a); });
  const pts = new Float32Array(points.flatMap(p => [...p, 0]));
  await withParityDevice(async device => {
    const module = device.createShaderModule({ code: /* wgsl */`
@group(0) @binding(0) var<storage,read> recs:array<vec4f>;
@group(0) @binding(1) var<storage,read> pts:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> mom:array<i32>;
@group(0) @binding(3) var<storage,read_write> wt:array<u32>;
@group(0) @binding(4) var<storage,read_write> out:array<vec4f>;
override APIC:bool=true;
const D:i32=${D};
// Face axis of node j (the +face of cell j, as nbxDeposit's callers), nodes past the grid held to it (unread here).
fn faceIndex(axis:u32,node:vec3i)->u32{let c=clamp(node,vec3i(0),vec3i(D-1));return axis*u32(D*D*D)+u32(c.x+D*(c.y+D*c.z));}
fn nbxDeposit(axis:u32,node:vec3i,momentum:i32,weight:u32){let i=faceIndex(axis,node);mom[i]+=momentum;wt[i]+=weight;}
fn nbxTransferHeat(tile:u32)->vec2f{return vec2f(0.0);}
fn nbxTransferDepth(vertex:vec3u)->f32{return 0.0;}
fn nbxApicTap(node:vec3i,axis:u32)->f32{let i=faceIndex(axis,node);return (f32(mom[i])/PB_MOMENTUM)/(f32(wt[i])/PB_WEIGHT);}
${nbfP2GWGSL}
${nbfP2GApicWGSL}
${nbfApicWGSL}
@compute @workgroup_size(1) fn main(){
 for(var i=0u;i<${n}u;i++){
  var c=array<vec3f,3>(recs[5u*i+2u].xyz,recs[5u*i+3u].xyz,recs[5u*i+4u].xyz);
  if(!APIC){c=array<vec3f,3>(vec3f(0.0),vec3f(0.0),vec3f(0.0));}
  nbfP2GScatterApic(recs[5u*i].xyz,recs[5u*i+1u].xyz,c,1.0);
 }
 for(var k=0u;k<${points.length}u;k++){for(var a=0u;a<3u;a++){out[3u*k+a]=nbfApicRow(pts[k].xyz,a);}}
}` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => `${m.lineNum}:${m.linePos} ${m.message}`);
    assert.deepEqual(errors, [], "the formulas compile");
    const run = async (apic: boolean) => {
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main", constants: { APIC: apic ? 1 : 0 } } });
      const make = (size: number, data?: Float32Array<ArrayBuffer>) => { const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); if (data) device.queue.writeBuffer(b, 0, data); return b; };
      const bufs = [make(input.byteLength, input), make(pts.byteLength, pts), make(4 * faces), make(4 * faces), make(16 * 3 * points.length)];
      const e = device.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
      pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([e.finish()]);
      const [mom, wt, out] = [new Int32Array(await readBack(device, bufs[2]!)), new Uint32Array(await readBack(device, bufs[3]!)), new Float32Array(await readBack(device, bufs[4]!))];
      for (const b of bufs) b.destroy();
      // The interior faces (every node 3..8: full support from the records' cells 1..10).
      let faceError = 0;
      for (let a = 0; a < 3; a++) for (let z = 3; z <= 8; z++) for (let y = 3; y <= 8; y++) for (let x = 3; x <= 8; x++) {
        const i = a * D * D * D + x + D * (y + D * z), at = [x + 0.5, y + 0.5, z + 0.5]; at[a] = [x, y, z][a]! + 1;
        faceError = Math.max(faceError, Math.abs((mom[i]! / 8192) / (wt[i]! / 65536) - affine(at, a)));
      }
      let velocityError = 0, rowError = 0;
      points.forEach((p, k) => { for (let a = 0; a < 3; a++) {
        const r = out.subarray(4 * (3 * k + a), 4 * (3 * k + a) + 4);
        velocityError = Math.max(velocityError, Math.abs(r[3]! - affine(p, a)));
        for (let j = 0; j < 3; j++) rowError = Math.max(rowError, Math.abs(r[j]! - A[a]![j]!));
      } });
      return { faceError, velocityError, rowError };
    };
    const apic = await run(true), pic = await run(false);
    t.diagnostic(`APIC ${JSON.stringify(apic)}; PIC ${JSON.stringify(pic)}`);
    assert.ok(apic.faceError < 2e-3, `APIC's faces are the field's: ${apic.faceError}`);
    assert.ok(apic.velocityError < 2e-3 && apic.rowError < 4e-3, `APIC's G2P recovers u(p) and A: ${apic.velocityError} ${apic.rowError}`);
    assert.ok(pic.faceError > 0.05, `PIC's deposit misses the affine field: ${pic.faceError}`);
  });
});
