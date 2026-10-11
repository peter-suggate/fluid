/** A cut owner's least-squares plane, evaluated at its tile centre.
 * Empty and fully open masks retain the original ordered result. Partial
 * masks use four 32-lane fits per workgroup, with four phi values per lane
 * instead of one thread retaining all 125. Only the sum order changes.
 * Physics changes must also update the ordered cutTile reference in
 * uniform-mixed-pressure-authority.ts and its frozen comparison. */
export const uniformMixedPressureCutWGSL=/* wgsl */`
// Cooperative cut fit; each workgroup owns the same 64-tile batch as the serial entry.
var<workgroup> pcWorkCount:atomic<u32>;
var<workgroup> pcCount:u32;
var<workgroup> pcTiles:array<u32,64>;
var<workgroup> pcFirst:array<vec4f,128>;
var<workgroup> pcSecond:array<vec4f,128>;
var<workgroup> pcThird:array<vec4f,128>;
fn pcFitTile(t:u32,validTile:bool,l:u32){
 let origin=vec3i(4u*umTileCoord(t));let open=umSolidOpenCells(t);let lane=l&31u;let base=l&~31u;
 var values:array<f32,4>;var lives=0u;var first=vec4f(0);var total=0.0;
 for(var k=0u;k<4u;k++){
  let v=lane+32u*k;
  if(validTile&&v<125u&&cutLive(open,v)){
   lives|=1u<<k;let value=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;values[k]=value;
   first+=vec4f(vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0,1);total+=value;
  }
 }
 pcFirst[l]=first;pcSecond[l]=vec4f(total,0,0,0);
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];}
  workgroupBarrier();
 }
 let count=pcFirst[base].w;let mean=pcFirst[base].xyz/max(count,1.0);let level=pcSecond[base].x/max(count,1.0);
 workgroupBarrier();
 var diagonal=vec3f(0);var cross3=vec3f(0);var moment=vec3f(0);
 for(var k=0u;k<4u;k++){if((lives&(1u<<k))!=0u){
  let v=lane+32u*k;let r=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0-mean;
  diagonal+=r*r;cross3+=r.xxy*r.yzz;moment+=r*(values[k]-level);
 }}
 pcFirst[l]=vec4f(diagonal,0);pcSecond[l]=vec4f(cross3,0);pcThird[l]=vec4f(moment,0);
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];pcThird[l]+=pcThird[l+stride];}
  workgroupBarrier();
 }
 if(lane==0u&&validTile){
  var result=2.0*UM_HMIN;
  if(count>0.0){
   let diagonal=pcFirst[base].xyz;let cross3=pcSecond[base].xyz;let moment=pcThird[base].xyz;
   let c0=vec3f(diagonal.x,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z);
   let slope=vec3f(dot(moment,cross(c1,c2)),dot(c0,cross(moment,c2)),dot(c0,cross(c1,moment)))/dot(c0,cross(c1,c2));
   result=level-dot(slope,mean);
  }
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(result,0.0);
 }
 workgroupBarrier();
}
@compute @workgroup_size(128) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){
  if(l==0u){atomicStore(&pcWorkCount,0u);}
  workgroupBarrier();
  if(l<64u){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){
  let open=umSolidOpenCells(t);
  if(all(open==vec2u(0u))){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(2.0*UM_HMIN,0.0);}
  else if(all(open==vec2u(0xffffffffu))){pcFullFit(t);}
  else{let at=atomicAdd(&pcWorkCount,1u);pcTiles[at]=t;}
 }}
  workgroupBarrier();
  if(l==0u){pcCount=atomicLoad(&pcWorkCount);}
  let count=workgroupUniformLoad(&pcCount);
  for(var j=0u;j<count;j+=4u){let at=j+(l>>5u);pcFitTile(pcTiles[min(at,count-1u)],at<count,l);}
  workgroupBarrier();
 }
}
fn pcFullFit(t:u32){
 // Every vertex is live and its centred geometric mean is exactly zero.
 // The fitted centre value is the same ordered phi mean as the serial fit.
 let origin=vec3i(4u*umTileCoord(t));var count=0.0;var total=0.0;
 for(var v=0u;v<125u;v++){count+=1.0;total+=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;}
 balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/count,0.0);
}
`;
