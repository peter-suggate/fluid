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
  // Pipelines by name (label, else module label and entry point): the shared
  // frame schedule's work is counted by pipeline.
  const names=new WeakMap<GPUComputePipeline,string>();
  const nameOf=(d:GPUComputePipelineDescriptor)=>d.label||`${d.compute.module.label||"module"}:${d.compute.entryPoint??"main"}`;
  const devices=Object.getPrototypeOf(device) as GPUDevice,create=devices.createComputePipeline,createAsync=devices.createComputePipelineAsync;
  devices.createComputePipeline=function(this:GPUDevice,d:GPUComputePipelineDescriptor){const p=create.call(this,d);names.set(p,nameOf(d));return p;};
  devices.createComputePipelineAsync=async function(this:GPUDevice,d:GPUComputePipelineDescriptor){const p=await createAsync.call(this,d);names.set(p,nameOf(d));return p;};
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
      const label = String(descriptor?.label ?? "unlabelled"), rows = current;
      let row = rows[label] ??= { passes: 0, dispatches: 0, indirectDispatches: 0, workgroups: 0 };
      row.passes++;
      if (label === "Uniform shared frame schedule") {
        const set = pass.setPipeline;
        pass.setPipeline = function(pipeline: GPUComputePipeline) {
          row = rows[`schedule: ${names.get(pipeline) ?? (pipeline.label || "unnamed")}`] ??= { passes: 0, dispatches: 0, indirectDispatches: 0, workgroups: 0 };
          set.call(this, pipeline);
        };
      }
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
    destroy: () => { prototype.beginComputePass = begin;devices.createComputePipeline=create;devices.createComputePipelineAsync=createAsync;queries?.destroy(); } };
}
