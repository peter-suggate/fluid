import assert from "node:assert/strict";

/** Frozen workgroup-per-row normalization for paired checks. QA only;
 * the application always compiles its current operator. */
export function uniformTransportWorkgroupReference(code:string):string{
 const start=code.indexOf("// Four coarse rows per workgroup,"),end=code.indexOf("// Fine row tiles:",start);
 assert.ok(start>=0&&end>start,"Transport reference no longer matches the production operator");
 const previous=String.raw`// One coarse row per workgroup: distribute up to 125 donor overlaps,
// retaining the scalar row's summation order and exact integer donor sums.
var<workgroup> coarseWeights:array<f32,126>;
var<workgroup> coarseDonors:array<u32,126>;
var<workgroup> coarseScale:f32;
fn normalizeCoarseRow(job:u32,lane:u32,divide:bool){
 let r=rowAt(vec3u(job,0u,0u));let base=rowBase(r);
 for(var k=lane;k<126u;k+=64u){if(r.width!=0u&&k<r.count){
  let donor=donorFrom(r,base,k);var weight=bitcast<f32>(edges[r.address+(1u+k)*r.stride]);
  if(divide){weight=weight*donor.capacity/max(sums[donor.index],1e-20);}
  else if(k==r.count-1u&&!tpSampled(r.index)){weight=f32(r.width*r.width*r.width);}
  coarseWeights[k]=weight;coarseDonors[k]=donor.index;
 }}
 workgroupBarrier();
 if(lane==0u&&r.width!=0u){
  var sum=0.0;for(var k=0u;k<r.count;k++){sum+=coarseWeights[k];}
  coarseScale=umRowCapacity(r)/max(sum,1e-20);
 }
 workgroupBarrier();
 for(var k=lane;k<126u;k+=64u){if(r.width!=0u&&k<r.count){
  let weight=tpQuantize(coarseWeights[k]*coarseScale);
  edges[r.address+(1u+k)*r.stride]=bitcast<u32>(weight);uvAddDonor(coarseDonors[k],weight);
 }}
}
// Grid-stride over the listed coarse rows, one row per workgroup job.
var<workgroup> tpCoarseJobs:u32;
fn normalizeCoarseRows(group:u32,groups:u32,lane:u32,divide:bool){
 if(lane==0u){tpCoarseJobs=atomicLoad(&live[tpCountWord(umTransportList,1u)]);}
 let jobs=workgroupUniformLoad(&tpCoarseJobs);
 for(var job=group;job<jobs;job+=groups){normalizeCoarseRow(job,lane,divide);workgroupBarrier();}
}
@compute @workgroup_size(64) fn rowsFallbackCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){normalizeCoarseRows(group.x,groups.x,lane,false);}
@compute @workgroup_size(64) fn rowsDivideCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){normalizeCoarseRows(group.x,groups.x,lane,true);}
`;
 return code.slice(0,start)+previous+code.slice(end);
}

/** Restore the old one-workgroup-per-coarse-row launch budget as well as
 * the shader above. Only the reference stage instance is intercepted. */
export function restoreUniformTransportWorkgroupDispatch(stage:any):void{
 const dispatch=stage.ownership.dispatchBuffered.bind(stage.ownership);
 const pipelines=new Set<GPUComputePipeline>();
 for(const [key,tiers] of stage.pipelines as Map<string,GPUComputePipeline[]>){
  if(key.startsWith("rowsFallback:")||key.startsWith("rowsDivide:")){pipelines.add(tiers[1]!);pipelines.add(stage.variant(tiers[1]!));}
 }
 assert.ok(pipelines.size>0,"Initialize the reference transport before restoring its dispatch");
 stage.ownership.dispatchBuffered=(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,kind:string,maximum:number,jobsPerGroup=1)=>{
  if(pipelines.has(pipeline))dispatch(pass,pipeline,"coarse",Math.min(2048,stage.ownership.capacity.tiles),1);
  else dispatch(pass,pipeline,kind,maximum,jobsPerGroup);
 };
}
