import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { geometricSeamRows, gradedSeamLayout, seamLayout } from "./helpers/uniform-geometric-seam";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("graded Uniform MAC patches have reciprocal ownership and exact geometric face coverage", { timeout: 180000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform mixed MAC face ownership");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    for (const mode of ["fine", "coarse", "graded"] as const) for (let axis = 0; axis < 3; axis++) {
      const layout = mode === "graded" ? gradedSeamLayout(axis) : seamLayout(axis, mode);
      const ownership = new UniformMixedOwnership(device, layout);
      const size = layout.cellCount * 24 * 12 * 4;
      const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback: GPUBuffer = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const module = device.createShaderModule({ code: uniformMixedTopologyWGSL(layout, 0) + /* wgsl */ `
@group(1) @binding(0) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(64) fn faces(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){for(var k=0u;k<4u;k++){
  let face=umFace(owner,axis,select(-1,1,side==1u),k);let at=(owner.index*24u+axis*8u+side*4u+k)*12u;
  result[at]=face.width;if(face.width==0u){continue;}
  result[at+1u]=select(face.neighbor.index,0xffffffffu,face.neighbor.width==0u);
  result[at+2u]=u32(face.anchor.x);result[at+3u]=u32(face.anchor.y);result[at+4u]=u32(face.anchor.z);
  result[at+5u]=face.axis;result[at+6u]=u32(face.sign);result[at+7u]=face.count;
  let center=umFaceCenter(face);result[at+8u]=bitcast<u32>(center.x);result[at+9u]=bitcast<u32>(center.y);result[at+10u]=bitcast<u32>(center.z);
 }}}
}
` });
        const resources = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, resources] });
        const pipelines = await Promise.all([1, 2, 4].map(umCellWidth => device!.createComputePipelineAsync({ layout: pipelineLayout,
          compute: { module, entryPoint: "faces", constants: { umCellWidth, umDispatchX: ownership.dispatchX } } })));
        const group = device.createBindGroup({ layout: resources, entries: [{ binding: 0, resource: { buffer: output } }] });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, group); ownership.dispatch(pass, pipelines); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, size); device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const bytes: ArrayBuffer = readback.getMappedRange();
        const words: Uint32Array = new Uint32Array(bytes), floats = new Float32Array(bytes);
        const cells = geometricSeamRows(layout, () => [0, 0, 0]).cells;
        const faces = new Map<string, { self: number; neighbor: number; sign: number; area: number }[]>();
        let splitFaces = 0;
        for (let i = 0; i < cells.length; i++) for (let a = 0; a < 3; a++) for (let side = 0; side < 2; side++) {
          let area = 0;
          const c = cells[i]!, sign = side ? 1 : -1;
          for (let k = 0; k < 4; k++) {
            const at = (i * 24 + a * 8 + side * 4 + k) * 12, width = words[at]!;
            if (!width) continue;
            const neighbor = words[at + 1]!, anchor = Array.from(words.subarray(at + 2, at + 5), v => v | 0);
            assert.equal(words[at + 5], a); assert.equal(words[at + 6]! | 0, sign);
            assert.ok(words[at + 7]! <= 4);
            assert.equal(anchor[a], c.min[a]! + (side ? c.width : 0) - 1);
            for (let b = 0; b < 3; b++) assert.equal(floats[at + 8 + b], anchor[b]! + (b === a ? 1 : width / 2));
            if (neighbor !== 0xffffffff) {
              assert.equal(width, Math.min(c.width, cells[neighbor]!.width));
              assert.ok(Math.max(c.width, cells[neighbor]!.width) <= 2 * width);
            }
            const key = `${a}:${anchor}:${width}`, patch = { self: i, neighbor, sign, area: width ** 2 };
            const existing = faces.get(key) ?? []; existing.push(patch); faces.set(key, existing);
            area += width ** 2;
            if (words[at + 7] === 4 && k === 0) splitFaces++;
          }
          assert.equal(area, c.width ** 2, `${mode} cell ${i} face ${a}/${side} coverage`);
        }
        const divergence = new Float64Array(cells.length);
        for (const [key, incident] of faces) {
          const f = incident[0]!;
          if (f.neighbor === 0xffffffff) { assert.equal(incident.length, 1); continue; }
          assert.equal(incident.length, 2, key);
          const g = incident[1]!; assert.equal(f.self, g.neighbor); assert.equal(f.neighbor, g.self); assert.equal(f.sign, -g.sign);
          // An arbitrary canonical flux cancels exactly between both cells,
          // including anisotropic physical face areas and unequal cell volumes.
          const a = Number(key[0]), h = layout.lattice.cellSize_m;
          const flux = Math.sin(f.self * 13 + g.self * 7) * f.area * h[(a + 1) % 3]! * h[(a + 2) % 3]!;
          divergence[f.self] += f.sign * flux; divergence[g.self] += g.sign * flux;
        }
        assert.ok(Math.abs(divergence.reduce((sum, v) => sum + v, 0)) < 1e-10);
        assert.equal(splitFaces > 0, mode === "graded");
        readback.unmap();
      } finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); output.destroy(); ownership.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
