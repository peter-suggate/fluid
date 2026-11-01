/** Optional fixed-input A/B/B/A timing for algorithm experiments. Production
 * dispatches are untouched. Each sample repeats the same kernel to amortize
 * timestamp quantization; timestamps exclude host encoding and readback. */
export async function pairedComputeTimings(device:GPUDevice,pipelines:readonly GPUComputePipeline[],groups:readonly (GPUBindGroup|readonly GPUBindGroup[])[],dispatch:number,repeats=8){
 if(!device.features.has("timestamp-query"))throw new Error("Algorithm timings require timestamp-query");
 const query=device.createQuerySet({type:"timestamp",count:8});
 const resolve=device.createBuffer({size:64,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=device.createBuffer({size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 const samples:number[][]=[[],[]];
 try{
  for(let round=0;round<14;round++){
   const e=device.createCommandEncoder();
   for(const [j,variant] of [0,1,1,0].entries()){
    const p=e.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:2*j,endOfPassWriteIndex:2*j+1}});
    p.setPipeline(pipelines[variant]!);
    const bound=groups[variant]!;
    if(Array.isArray(bound)){for(const [index,group] of bound.entries())p.setBindGroup(index,group);}
    else p.setBindGroup(0,bound as GPUBindGroup);
    for(let n=0;n<repeats;n++)p.dispatchWorkgroups(dispatch);
    p.end();
   }
   e.resolveQuerySet(query,0,8,resolve,0);e.copyBufferToBuffer(resolve,0,read,0,64);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange());
   if(round>=2)for(const [j,variant] of [0,1,1,0].entries())samples[variant]!.push(Number(t[2*j+1]!-t[2*j]!)/1e6/repeats);
   read.unmap();
  }
  const median=(a:number[])=>{const s=[...a].sort((a,b)=>a-b);return (s[s.length/2-1]!+s[s.length/2]!)/2;};
  const medians=samples.map(median);return {baselineMedian_ms:medians[0],candidateMedian_ms:medians[1],speedup:medians[0]!/medians[1]!,samples_ms:samples};
 }finally{query.destroy();resolve.destroy();read.destroy();}
}
