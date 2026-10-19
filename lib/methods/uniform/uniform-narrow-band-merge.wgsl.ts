/** Work-parallel segmented merge of canonical particle keys. Each group
 * merges a bounded tile using shared memory after two diagonal searches. Even a
 * single crowded cell is distributed across the GPU. Compact xyz/source
 * keys alternate banks; work items retain immutable cell identity. */
export function narrowBandMergeWGSL(scratchKeys:number,headerWords:number,mergeQueues:readonly number[]):string{return /* wgsl */`
const ORDER_HEADER_WORDS:u32=${headerWords}u;
struct OrderScratch{counts:array<atomic<u32>,${headerWords}>,words:array<u32>}
@group(0) @binding(8) var<storage,read_write> orderScratch:OrderScratch;
const ORDER_SCRATCH_KEYS:u32=${scratchKeys}u;
const ORDER_MERGE_LEVELS:u32=${mergeQueues.length}u;
const ORDER_MERGE_QUEUES=array<u32,${Math.max(1,mergeQueues.length)}>(${mergeQueues.length?mergeQueues.map(n=>n+"u").join(","):"0u"});
override ORDER_MERGE_LEVEL:u32=0u;
override ORDER_MERGE_WIDTH:u32=128u;
override ORDER_MERGE_BANK:u32=1u;
// Retain compact keys through every level. Looking positions up through
// shuffled source indices here creates random full-record traffic and a
// cache-capacity cliff. Only exact position ties read payload components.
fn mergeSource(slot:u32,bank:u32)->vec4u{
 if(bank==0u){return orderKey(slot);}
 let at=ORDER_SCRATCH_KEYS-ORDER_HEADER_WORDS+4u*slot;
 return vec4u(orderScratch.words[at],orderScratch.words[at+1u],orderScratch.words[at+2u],orderScratch.words[at+3u]);
}
fn mergeStore(slot:u32,bank:u32,key:vec4u){
 if(bank==0u){let at=UM_CELL_KEYS()+4u*slot;links[at]=key.x;links[at+1u]=key.y;links[at+2u]=key.z;links[at+3u]=key.w;}
 else{let at=ORDER_SCRATCH_KEYS-ORDER_HEADER_WORDS+4u*slot;orderScratch.words[at]=key.x;orderScratch.words[at+1u]=key.y;orderScratch.words[at+2u]=key.z;orderScratch.words[at+3u]=key.w;}
}
fn mergeBefore(a:vec4u,b:vec4u)->bool{return orderBefore(a,b);}
// A workgroup loads one 256-key output tile once. Two diagonal searches
// locate its input slices; all remaining comparisons use shared keys.
var<workgroup> mergeJobs:u32;
var<workgroup> mergeRange:vec4u;
var<workgroup> mergeCuts:array<vec2u,2>;
var<workgroup> mergeKeys:array<vec4u,256>;
@compute @workgroup_size(256) fn canonicalMerge(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){mergeJobs=atomicLoad(&orderScratch.counts[4u*(ORDER_MERGE_LEVEL+1u)+3u]);}
 let jobs=workgroupUniformLoad(&mergeJobs);
 for(var job=group.x;job<jobs;job+=groups.x){
  if(lane==0u){
   let at=ORDER_MERGE_QUEUES[ORDER_MERGE_LEVEL]-ORDER_HEADER_WORDS+2u*job;
   let slot=orderScratch.words[at];let cell=orderScratch.words[at+1u];
   let start=links[cell];let count=atomicLoad(&bins[cell]);mergeRange=vec4u(slot,start,start+count,count);
  }
  let r=workgroupUniformLoad(&mergeRange);let slot=r.x;let start=r.y;let end=r.z;let count=r.w;
  let a=start+((slot-start)/(2u*ORDER_MERGE_WIDTH))*(2u*ORDER_MERGE_WIDTH);
  let b=min(a+ORDER_MERGE_WIDTH,end);let finish=min(b+ORDER_MERGE_WIDTH,end);
  let na=b-a;let nb=finish-b;let n=min(256u,finish-slot);
  if(lane<2u){
   let diagonal=slot-a+select(0u,n,lane==1u);
   var low=diagonal-min(diagonal,nb);var high=min(diagonal,na);
   while(low<high){let mid=(low+high)/2u;let j=diagonal-mid;
    if(j>0u&&mid<na&&mergeBefore(mergeSource(a+mid,ORDER_MERGE_BANK),mergeSource(b+j-1u,ORDER_MERGE_BANK))){low=mid+1u;}else{high=mid;}
   }
   mergeCuts[lane]=vec2u(low,diagonal-low);
  }
  workgroupBarrier();
  let cuts=workgroupUniformLoad(&mergeCuts[0]);let next=workgroupUniformLoad(&mergeCuts[1]);let left=next.x-cuts.x;
  if(lane<n){
   var at=b+cuts.y+lane-left;if(lane<left){at=a+cuts.x+lane;}
   mergeKeys[lane]=mergeSource(at,ORDER_MERGE_BANK);
  }
  workgroupBarrier();
  if(lane<n){
   let own=mergeKeys[lane];let otherStart=select(0u,left,lane<left);let otherEnd=select(left,n,lane<left);
   var low=otherStart;var high=otherEnd;
   while(low<high){let mid=(low+high)/2u;if(orderBefore(mergeKeys[mid],own)){low=mid+1u;}else{high=mid;}}
   let rank=select(lane-left,lane,lane<left)+low-otherStart;
   if(count<=2u*ORDER_MERGE_WIDTH){output[slot+rank]=input[own.w];}
   else{mergeStore(slot+rank,1u-ORDER_MERGE_BANK,own);}
  }
  workgroupBarrier();
 }
}
`;}
