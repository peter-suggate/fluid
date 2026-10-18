/** A cut owner's least-squares plane, evaluated at its tile centre.
 * Empty and fully open masks retain the ordered result. Partial masks use
 * four 32-lane fits per workgroup, with four phi values per lane.
 * Where liquid is present, only negative samples adjacent to open cells
 * enter the fit: positive
 * distance bends around a voxel shoreline and includes air beyond the wall,
 * which must not tilt the liquid's hydrostatic pressure plane. This uses
 * the existing vertex loads and removes the connected-component flood fill.
 * Wholly dry tiles retain their air fit for the free-surface distance ratio.
 * Samples on a coordinate plane through the centre can determine its value
 * without a perpendicular derivative. Other rank-deficient sets leave the
 * root owner in air (2h), with the interface resolved by the fine pressure.
 * The slot's second word is 0 for an empty mask and 1 otherwise.
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
 // Dry distance bends around the shoreline. Fit the liquid samples already
 // loaded by this kernel; no neighbourhood reads or additional pass.
 var values:array<f32,4>;var lives=0u;var allLives=0u;var first=vec4f(0);var total=0.0;var airFirst=vec4f(0);var airTotal=0.0;
 for(var k=0u;k<4u;k++){
  let v=lane+32u*k;if(!validTile||v>=125u||!cutLive(open,v)){continue;}
  let value=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;
  values[k]=value;allLives|=1u<<k;let at=vec4f(vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0,1);
  if(value<0.0){lives|=1u<<k;first+=at;total+=value;}else{airFirst+=at;airTotal+=value;}
 }
 pcFirst[l]=first;pcSecond[l]=vec4f(total,airTotal,0,0);pcThird[l]=airFirst;
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];pcThird[l]+=pcThird[l+stride];}
  workgroupBarrier();
 }
 // Entirely dry tiles still supply the air-side distance in free-surface theta.
 let wet=pcFirst[base].w>0.0;let moments=select(pcThird[base],pcFirst[base],wet);
 let count=moments.w;let mean=moments.xyz/max(count,1.0);let level=select(pcSecond[base].y,pcSecond[base].x,wet)/max(count,1.0);
 if(!wet){lives=allLives;}
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
   var diagonal=pcFirst[base].xyz;let cross3=pcSecond[base].xyz;let moment=pcThird[base].xyz;
   // Samples on a coordinate plane through the centre still determine its
   // value. The absent derivative is immaterial there, so fix it to zero.
   diagonal=select(diagonal,vec3f(1.0),(diagonal==vec3f(0.0))&(mean==vec3f(0.0)));
   let c0=vec3f(diagonal.x,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z);
   let determinant=dot(c0,cross(c1,c2));let scale=diagonal.x+diagonal.y+diagonal.z;
   if(determinant>0.000001*scale*scale*scale){
    let slope=vec3f(dot(moment,cross(c1,c2)),dot(c0,cross(moment,c2)),dot(c0,cross(c1,moment)))/determinant;
    result=level-dot(slope,mean);
   }
  }
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(result,1.0);
 }
 workgroupBarrier();
}
@compute @workgroup_size(128) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){
  if(l==0u){atomicStore(&pcWorkCount,0u);}
  workgroupBarrier();
  if(l<64u){let t=start+l;if(umTileLive(t)&&(umSolidCut(t)||(pcHeld&&umSolidFineTile(t)))){
  let open=umSolidOpenCells(t);
  if(!umSolidCut(t)){pcHeldFit(t);}
  else if(all(open==vec2u(0u))){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(2.0*UM_HMIN,0.0);}
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
// An uncut tile the simulation holds at h: the mean of its h cells' centre
// phi, each the mean of its corners, as the band's rows hold them. The tile's
// eight corners alone do not see the h surface between them.
fn pcHeldFit(t:u32){
 let origin=vec3i(4u*umTileCoord(t));var total=0.0;
 for(var v=0u;v<125u;v++){
  let g=vec3u(v%5u,(v/5u)%5u,v/25u);let w=select(vec3f(1.0),vec3f(0.5),(g==vec3u(0u))|(g==vec3u(4u)));
  total+=w.x*w.y*w.z*textureLoad(vertexPhi,origin+vec3i(g),0).x;
 }
 balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/64.0,0.0);
}
fn pcFullFit(t:u32){
 // Every vertex is live and its centred geometric mean is exactly zero.
 // The fitted centre value is the same ordered phi mean as the serial fit.
 let origin=vec3i(4u*umTileCoord(t));var count=0.0;var total=0.0;
 for(var v=0u;v<125u;v++){count+=1.0;total+=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;}
 balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/count,1.0);
}
`;
