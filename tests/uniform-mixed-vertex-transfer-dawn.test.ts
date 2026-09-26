import { uniformMixedPresentationWGSL } from "../lib/methods/uniform/uniform-mixed-presentation.wgsl";
import { UniformMixedSurfaceGeometry } from "../lib/methods/uniform/uniform-mixed-surface-geometry";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedVertexTransfer } from "../lib/methods/uniform/uniform-mixed-vertex-transfer";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import { seamLayout } from "./helpers/uniform-geometric-seam";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("mixed vertex handoff uses unique coarse authority, continuous hanging values and no allocated fields", { timeout: 180000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform mixed vertex transfer");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    for (const layout of [seamLayout(0, "fine"), seamLayout(0, "coarse"), ...mixedPressureLayouts()]) {
      const d = layout.lattice.dimensions, size = d.map(n => n + 1), count = size.reduce((a, b) => a * b);
      const cells = mixedPressureFixture(layout).cells.map(c => ({ width: c.width,
        origin: c.center.map((v, a) => Math.round(v / layout.lattice.cellSize_m[a]! - c.width / 2)) }));
      const index = (p: readonly number[]) => p[0]! + size[0]! * (p[1]! + size[1]! * p[2]!);
      // Independent physical-box authority; no packed topology or GPU lookup.
      const authority = (p: readonly number[]) => cells.filter(c => c.origin.every((v, a) => p[a]! >= v && p[a]! <= v + c.width))
        .sort((a, b) => b.width - a.width)[0]!;
      const canonical = (p: readonly number[]) => { const c = authority(p); return c.origin.every((v, a) => (p[a]! - v) % c.width === 0); };
      const points: number[][] = [];
      for (let z = 0; z <= d[2]; z++) for (let y = 0; y <= d[1]; y++) for (let x = 0; x <= d[0]; x++) points.push([x, y, z]);
      const ownership = new UniformMixedOwnership(device, layout), textures: GPUTexture[] = [], buffers: GPUBuffer[] = [];
      const texture = () => { const t = device!.createTexture({ size, dimension: "3d", format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST }); textures.push(t); return t; };
      const fine = texture(), compact = texture(), expanded = texture();
      const cellTexture=()=>{const t=device!.createTexture({size:[...d],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
      const target=cellTexture(),centerPhi=cellTexture();
      try {
        const borrowed: GPUDevice = new Proxy(device, { get(target, key) {
          if (key === "createBuffer" || key === "createTexture") return () => { throw new Error("Mixed vertex transfer allocated a field"); };
          const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
        } });
        const restriction: UniformMixedVertexTransfer = new UniformMixedVertexTransfer(borrowed, ownership, fine, compact, "restrict");
        const prolongation: UniformMixedVertexTransfer = new UniformMixedVertexTransfer(borrowed, ownership, compact, expanded, "prolong");
        await restriction.initialize(); await prolongation.initialize();
        const geometry:UniformMixedSurfaceGeometry=new UniformMixedSurfaceGeometry(borrowed,ownership);await geometry.initialize();
        const geometryGroup=geometry.bind(compact,target,centerPhi);
        assert.equal(restriction.allocatedBytes + prolongation.allocatedBytes + geometry.allocatedBytes, 0);
        const upload = (t: GPUTexture, data: Float32Array<ArrayBuffer>) => device!.queue.writeTexture({ texture: t }, data, { bytesPerRow: size[0]! * 4, rowsPerImage: size[1]! }, size);
        const isCanonical = points.map(canonical);
        for (const mode of ["affine", "curved"] as const) {
          const source = Float32Array.from(points, p => mode === "affine" ? .3 + .11 * p[0]! - .07 * p[1]! + .13 * p[2]!
            : Math.sin(.7 * p[0]! + .9 * p[1]! - .4 * p[2]!) + .2 * Math.cos(p[0]! - p[2]!));
          upload(fine, source); upload(compact, new Float32Array(count).fill(NaN));
          const encoder = device.createCommandEncoder(); restriction.encode(encoder); prolongation.encode(encoder); geometry.encode(encoder,geometryGroup); device.queue.submit([encoder.finish()]);
          const sparse = await readMixedTexture(device, compact), dense = await readMixedTexture(device, expanded);
          const fills=await readMixedTexture(device,target),centers=await readMixedTexture(device,centerPhi);
          if(mode==="affine")for(const cell of cells){
            const at=cell.origin[0]!+d[0]*(cell.origin[1]!+d[1]*cell.origin[2]!);
            const normal=[.11,-.07,.13],weights=normal.map(v=>Math.abs(v)*cell.width);
            const threshold=-.3-cell.origin.reduce((sum,v,a)=>sum+normal[a]!*v,0)-Math.min(0,normal[1]!*cell.width);
            let fraction=threshold<=0?0:threshold>=weights.reduce((a,b)=>a+b)?1:0;
            if(threshold>0&&threshold<weights.reduce((a,b)=>a+b))for(let k=0;k<8;k++){
              let t=threshold,sign=1;for(let a=0;a<3;a++)if((k>>a)&1){t-=weights[a]!;sign=-sign;}
              fraction+=sign*Math.max(0,t)**3/(6*weights[0]!*weights[1]!*weights[2]!);
            }
            const expectedCenter=.3+cell.origin.reduce((sum,v,a)=>sum+normal[a]!*(v+cell.width/2),0);
            assert.ok(Math.abs(fills[at]!-fraction)<2e-5,`geometric planar fill ${fills[at]} != ${fraction}`);
            assert.ok(Math.abs(centers[at]!-expectedCenter)<2e-6,"cell phi disagrees with vertex authority");
          }
          const cache = new Map<number, number>();
          const vertex = (p: readonly number[]): number => {
            const at = index(p); if (isCanonical[at]) return source[at]!;
            const saved = cache.get(at); if (saved !== undefined) return saved;
            const c = authority(p); let value = 0;
            for (let k = 0; k < 8; k++) {
              const q = c.origin.map((v, a) => v + ((k >> a) & 1) * c.width);
              const weight = c.origin.reduce((w, v, a) => w * (((k >> a) & 1) ? (p[a]! - v) / c.width : 1 - (p[a]! - v) / c.width), 1);
              if (weight > 0) value += weight * vertex(q);
            }
            cache.set(at, value); return value;
          };
          const sample = (p: readonly number[]) => {
            const c = cells.find(c => c.origin.every((v, a) => p[a]! >= v && (p[a]! < v + c.width || p[a] === d[a] && v + c.width === d[a])))!;
            let value = 0;
            for (let k = 0; k < 8; k++) {
              const q = c.origin.map((v, a) => v + ((k >> a) & 1) * c.width);
              const weight = c.origin.reduce((w, v, a) => w * (((k >> a) & 1) ? (p[a]! - v) / c.width : 1 - (p[a]! - v) / c.width), 1);
              if (weight > 0) value += weight * vertex(q);
            }
            return value;
          };
          for (let i = 0; i < count; i++) {
            if (isCanonical[i]) assert.equal(sparse[i], source[i], "restriction changed an authoritative value");
            else assert.ok(Number.isNaN(sparse[i]), "restriction materialized an inactive/hanging vertex");
            const expected = mode === "affine" ? source[i]! : sample(points[i]!);
            assert.ok(Number.isFinite(dense[i]) && Math.abs(dense[i]! - expected) < 2e-6, `${mode} vertex ${points[i]}: ${dense[i]} != ${expected}`);
          }
          // Arbitrary values across all tile seams, sampled on both sides at
          // non-vertex tangential coordinates. A per-owner interpolation that
          // reads stale hanging vertices fails with NaN or a visible jump.
          if (mode === "curved") {
            const queries: number[][] = [];
            for (let axis = 0; axis < 3; axis++) for (let plane = 4; plane < d[axis]!; plane += 4) for (let k = 0; k < 9; k++) {
              const p = d.map((n, a) => a === axis ? plane : n * ((k * 7 + a * 3) % 17 + .37) / 18);
              for (const sign of [-1, 1]) { const q = [...p]; q[axis]! += sign * 1e-5; queries.push(q); }
            }
            const input = device.createBuffer({ size: queries.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            const output = device.createBuffer({ size: queries.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }); buffers.push(input, output);
            device.queue.writeBuffer(input, 0, Float32Array.from(queries.flatMap(q => [...q, 0])));
            const module = device.createShaderModule({ code: uniformMixedTopologyWGSL(layout, 0) + `
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> queries:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> results:array<f32>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&queries)){results[id.x]=umSampleVertex(queries[id.x].xyz);}}` });
            const resources = device.createBindGroupLayout({ entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ] });
            const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, resources] }), compute: { module, entryPoint: "main" } });
            const group = device.createBindGroup({ layout: resources, entries: [
              { binding: 0, resource: compact.createView() }, { binding: 1, resource: { buffer: input } }, { binding: 2, resource: { buffer: output } },
            ] });
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, group);
            pass.dispatchWorkgroups(Math.ceil(queries.length / 64)); pass.end(); device.queue.submit([encoder.finish()]);
            const values = await readMixedBuffer(device, output);
            const consumerModule=device.createShaderModule({code:`
@group(0) @binding(0) var phi:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> queries:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> results:array<f32>;
${uniformMixedPresentationWGSL(3,"phi","textureDimensions(phi)-vec3u(1)")}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&queries)){results[id.x]=umSampleVertex(queries[id.x].xyz);}}`});
            const consumerPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:consumerModule,entryPoint:"main"}});
            const consumerGroup=device.createBindGroup({layout:consumerPipeline.getBindGroupLayout(0),entries:[
              {binding:0,resource:compact.createView()},{binding:1,resource:{buffer:input}},
              {binding:2,resource:{buffer:output}},{binding:3,resource:ownership.presentation},
            ]});
            const consumerEncoder=device.createCommandEncoder(),consumerPass=consumerEncoder.beginComputePass();
            consumerPass.setPipeline(consumerPipeline);consumerPass.setBindGroup(0,consumerGroup);consumerPass.dispatchWorkgroups(Math.ceil(queries.length/64));consumerPass.end();device.queue.submit([consumerEncoder.finish()]);
            const presented=await readMixedBuffer(device,output);
            for(let i=0;i<values.length;i++)assert.ok(Math.abs(presented[i]!-values[i]!)<1e-6,"presentation must match canonical simulation sampling");

            for (let i = 0; i < queries.length; i++) {
              assert.ok(Number.isFinite(values[i]) && Math.abs(values[i]! - sample(queries[i]!)) < 3e-6, `sample ${queries[i]}: ${values[i]}`);
              if (i % 2) assert.ok(Math.abs(values[i]! - values[i - 1]!) < 1e-4, `discontinuous seam at ${queries[i]}`);
            }
          }
        }
      } finally { buffers.forEach(b => b.destroy()); textures.forEach(t => t.destroy()); ownership.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
