/** Diagnostic ordering control; not a production sorting algorithm. */
export async function damProbeOrder(device:GPUDevice,nb:any){
  // One invocation owns a cell's disjoint run. Deliberately simple, unbounded
  // insertion sort for diagnosis, not a production performance proposal.
  const canonicalPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{entryPoint:'main',module:device.createShaderModule({code:`
struct Record { words:array<u32,${nb.particleStride}> }
@group(0) @binding(0) var<storage,read_write> records:array<Record>;
@group(0) @binding(1) var<storage,read> bins:array<u32>;
@group(0) @binding(2) var<storage,read_write> links:array<u32>;
fn less(a:Record,b:Record)->bool {
 for(var k=0u;k<${nb.particleStride}u;k++){if(a.words[k]!=b.words[k]){return a.words[k]<b.words[k];}}
 return false;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let cell=gid.x;if(cell>=${nb.order.cells}u){return;}
 let count=bins[cell];if(count==0u){return;}let start=links[cell];
 for(var k=1u;k<count;k++){
  let item=records[start+k];var j=k;
  while(j>0u){if(!less(item,records[start+j-1u])){break;}records[start+j]=records[start+j-1u];j--;}
  records[start+j]=item;
 }
 for(var k=0u;k<count;k++){let at=${nb.order.cells}u+4u*(start+k);for(var a=0u;a<4u;a++){links[at+a]=records[start+k].words[a];}}
}`})}});
  const canonicalGroups=[0,1].map(parity=>device.createBindGroup({layout:canonicalPipeline.getBindGroupLayout(0),entries:[nb.particles[parity],nb.bins,nb.next].map((buffer,binding)=>({binding,resource:{buffer}}))}));
  return (e:GPUCommandEncoder,parity:number)=>{const pass=e.beginComputePass({label:'Diagnostic canonical particle order'});pass.setPipeline(canonicalPipeline);pass.setBindGroup(0,canonicalGroups[parity]!);pass.dispatchWorkgroups(Math.ceil(nb.order.cells/64));pass.end();};
}
