/** A cut owner's least-squares plane, evaluated at its tile centre.
 * Empty and fully open masks retain the original ordered result. Partial
 * masks use four 32-lane fits per workgroup, with four phi values per lane
 * instead of one thread retaining all 125. Only the sum order changes.
 * A wall no thicker than the tile leaves open cells on both of its sides, and
 * phi is not one field across it: the fit is through the liquid's side
 * (cutSide), or the bath's corner tiles, a sliver of deep liquid and the air
 * outside, fit to air and were a p=0 face at depth. With liquid on neither
 * side the fit is through the side nearer it, the one of the lower mean phi:
 * an air owner's phi is then the continuation of the liquid's beside it,
 * which is what that liquid's free-surface theta reads (through both sides
 * the host model of a resting tank puts a row under a dry wall tile up to
 * 6% of rho g 4h off hydrostatic). The slot's second word names the sides
 * the fit is through, 1 the side cutSide grows and 2 the rest, for the
 * root's wet faces (umWetSide).
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
 // The vertices of each side of the tile (cutSide), and whether it holds liquid.
 let side=cutSide(open);let other=open&~side;
 var values:array<f32,4>;var lives=0u;var others=0u;var first=vec4f(0);var total=0.0;var firstOther=vec4f(0);var totalOther=0.0;var wet=vec2f(0);
 for(var k=0u;k<4u;k++){
  let v=lane+32u*k;if(!validTile||v>=125u){continue;}
  let own=cutLive(side,v);if(!own&&!cutLive(other,v)){continue;}
  let value=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;values[k]=value;
  let moment=vec4f(vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0,1);
  if(own){lives|=1u<<k;first+=moment;total+=value;if(value<0.0){wet.x=1.0;}}
  else{others|=1u<<k;firstOther+=moment;totalOther+=value;if(value<0.0){wet.y=1.0;}}
 }
 pcFirst[l]=first;pcSecond[l]=vec4f(total,totalOther,wet);pcThird[l]=firstOther;
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];pcThird[l]+=pcThird[l+stride];}
  workgroupBarrier();
 }
 // Liquid on one side alone: the fit is that side's. Liquid on neither: the
 // side nearer it, by its mean phi.
 let sums=pcSecond[base];let near=sums.x*pcThird[base].w<=sums.y*pcFirst[base].w;
 let keep=sums.z>0.0||(sums.w<=0.0&&near);let keepOther=sums.w>0.0||(sums.z<=0.0&&!near);
 var moments=vec4f(0);var sum=0.0;
 if(keep){moments+=pcFirst[base];sum+=sums.x;}
 if(keepOther){moments+=pcThird[base];sum+=sums.y;lives|=others;}
 if(!keep){lives=others;}
 let count=moments.w;let mean=moments.xyz/max(count,1.0);let level=sum/max(count,1.0);
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
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(result,select(0.0,1.0,keep)+select(0.0,2.0,keepOther));
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
