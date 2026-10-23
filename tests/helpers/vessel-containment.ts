import type { Quaternion, Vec3 } from "../../lib/core/model";
import type { UniformNarrowBandFlip } from "../../lib/methods/uniform/uniform-narrow-band-flip";

/** Read every live sample on the GPU, returning only containment counters. */
export async function vesselContainmentProbe(device: GPUDevice, particles: UniformNarrowBandFlip) {
  device.pushErrorScope("validation");
  const groups = Math.ceil(particles.capacity / 64), dispatchX = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
  const params = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const module = device.createShaderModule({ code: `
struct Params { position:vec4f, orientation:vec4f, halfSize:vec4f, origin:vec4f, cell:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> samples:array<f32>;
@group(0) @binding(2) var<storage,read> state:array<u32>;
@group(0) @binding(3) var<storage,read_write> result:array<atomic<u32>>;
@compute @workgroup_size(64) fn probe(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32) {
 let index=64u*(group.x+group.y*${dispatchX}u)+lane;
 if(index>=min(state[0]+state[21],arrayLength(&samples)/${particles.particleStride}u)){return;}let at=index*${particles.particleStride}u;
 if(samples[at+3u]<=0.0){return;}
 let p=params.origin.xyz+params.cell.xyz*vec3f(samples[at],samples[at+1u],samples[at+2u])-params.position.xyz;
 let q=params.orientation;let uv=cross(-q.yzw,p);let local=p+2.0*(q.x*uv+cross(-q.yzw,uv));
 let outside=max(abs(local)-params.halfSize.xyz,vec3f(0));let distance=max(outside.x,max(outside.y,outside.z));
 atomicAdd(&result[0],1u);if(distance>1e-5){atomicAdd(&result[1],1u);}
 atomicMax(&result[2],bitcast<u32>(distance));
}` });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: particles.activeParticles } },
    { binding: 2, resource: { buffer: particles.state } }, { binding: 3, resource: { buffer: output } },
  ] });
  const setupError = await device.popErrorScope(); if (setupError) throw new Error(setupError.message);
  return {
    async sample(position: Vec3, orientation: Quaternion, halfSize: Vec3, origin: Vec3, cellSize: number) {
      device.pushErrorScope("validation");
      device.queue.writeBuffer(params, 0, new Float32Array([
        position.x, position.y, position.z, 0, orientation.w, orientation.x, orientation.y, orientation.z,
        halfSize.x, halfSize.y, halfSize.z, 0, origin.x, origin.y, origin.z, 0, cellSize, cellSize, cellSize, 0,
      ]));
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(output);
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(dispatchX, Math.ceil(groups / dispatchX)); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope(); if (error) throw new Error(error.message);
      await read.mapAsync(GPUMapMode.READ);
      const mapped = read.getMappedRange(), counts = new Uint32Array(mapped), values = new Float32Array(mapped);
      const result = { count: counts[0]!, outside: counts[1]!, maxDistance_m: values[2]! }; read.unmap();
      return result;
    },
    destroy() { params.destroy(); output.destroy(); read.destroy(); },
  };
}
