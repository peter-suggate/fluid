export interface RegionTierCommands { [label: string]: { passes: number; dispatches: number; indirectDispatches: number; workgroups: number } }

/** Benchmark-only observation of actual direct dispatches, including compute
 * blits (which call the native encoder prototype). Optional timestamps use
 * separate diagnostic runs; count-only observation adds no GPU queries. */
export function profileRegionTierCommands(device: GPUDevice, timings = false) {
  const prototype = Object.getPrototypeOf(device.createCommandEncoder()) as GPUCommandEncoder;
  const begin = prototype.beginComputePass;
  let current: RegionTierCommands | undefined;
  const queries=timings?device.createQuerySet({type:"timestamp",count:64}):undefined;
  let timed:{label:string;index:number}[]=[];
  const labels=new Set(["Uniform shared frame schedule","Narrow-band FLIP regional air update","Narrow-band FLIP crossing search","Narrow-band FLIP retire empty air","Narrow-band FLIP shared frame crossing domain","APIC bin prefix scan","Narrow-band FLIP clear occupied bins","Narrow-band FLIP occupied cell offsets"]);
  prototype.beginComputePass = function(descriptor?: GPUComputePassDescriptor) {
    let measured=descriptor;
    if(current&&queries&&labels.has(String(descriptor?.label))){
      const index=2*timed.length;if(index+1>=64)throw new Error("Regional timing capacity exceeded");
      timed.push({label:String(descriptor!.label),index});
      measured={...descriptor,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}};
    }
    const pass = begin.call(this, measured);
    if (current) {
      const row = current[String(descriptor?.label ?? "unlabelled")] ??= { passes: 0, dispatches: 0, indirectDispatches: 0, workgroups: 0 };
      row.passes++;
      const indirect = pass.dispatchWorkgroupsIndirect;
      pass.dispatchWorkgroupsIndirect = function(buffer: GPUBuffer, offset: number) { row.dispatches++; row.indirectDispatches++; indirect.call(this, buffer, offset); };
      const dispatch = pass.dispatchWorkgroups;
      pass.dispatchWorkgroups = function(x: number, y = 1, z = 1) { row.dispatches++; row.workgroups += x * y * z; dispatch.call(this, x, y, z); };
    }
    return pass;
  };
  return { begin: () => { current = {}; timed=[]; }, end: () => { const result = current; current = undefined; return result; },
    readTimings:async()=>{
      if(!queries||timed.length===0)return [];
      const size=16*timed.length;
      const resolve=device.createBuffer({size,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try{
        const e=device.createCommandEncoder();e.resolveQuerySet(queries,0,2*timed.length,resolve,0);e.copyBufferToBuffer(resolve,0,read,0,size);
        device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(read.getMappedRange());
        if(times.every(t=>t===0n))throw new Error("GPU timestamps unavailable");
        const result=timed.map(({label,index})=>({label,ms:Number(times[index+1]!-times[index]!)/1e6}));read.unmap();return result;
      }finally{resolve.destroy();read.destroy();}
    },
    destroy: () => { prototype.beginComputePass = begin;queries?.destroy(); } };
}
