/** Read a compact GPU census, without transferring the dense scene payload. */
export async function censusSvoCellContours(device: GPUDevice, payload: GPUBuffer, geometryWords: number,
  strideWords: number, voxelCount: number): Promise<{ cells: number; meanCode: number }> {
  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage,read> payload:array<u32>;
    @group(0) @binding(1) var<storage,read_write> totals:array<atomic<u32>>;
    @compute @workgroup_size(256) fn count(@builtin(global_invocation_id) g:vec3u){
      let i=g.x+g.y*65535u*256u;if(i>=${voxelCount}u){return;}
      let q=payload[${geometryWords}u+i*${strideWords}u]>>24u;
      if(q>0u&&q<255u){atomicAdd(&totals[q],1u);}
    }` });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "count" } });
  const totals = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 1024, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: payload } }, { binding: 1, resource: { buffer: totals } }] });
    const encoder = device.createCommandEncoder();const pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);
    const groups=Math.ceil(voxelCount/256);pass.dispatchWorkgroups(Math.min(groups,65535),Math.max(1,Math.ceil(groups/65535)));pass.end();
    encoder.copyBufferToBuffer(totals,0,readback,0,1024);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);const counts=new Uint32Array(readback.getMappedRange().slice(0));readback.unmap();
    let cells=0, sum=0;
    for(let q=1;q<255;q++){cells+=counts[q];sum+=counts[q]*q;}
    return { cells, meanCode: cells ? sum/cells : 0 };
  } finally { totals.destroy();readback.destroy(); }
}
