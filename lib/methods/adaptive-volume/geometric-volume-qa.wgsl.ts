/** QA-only float reductions. No production dispatch, fixed-point telemetry, or
 * presentation materialization participates in this receipt. */
export const geometricVolumeQAWGSL = /* wgsl */ `
@group(1) @binding(1) var<storage,read_write> geometricVolumeQA:array<vec4f>;
var<workgroup> gvqaSum:array<vec4f,64>;
var<workgroup> gvqaCounts:array<vec4f,64>;
var<workgroup> gvqaExtrema:array<vec4f,64>;
var<workgroup> gvqaExcess:array<vec4f,64>;
var<workgroup> gvqaOutside:array<vec4f,64>;
fn gvqaFinite(x:f32)->bool{return x==x&&abs(x)<=3.402823466e38;}
@compute @workgroup_size(64)
fn reduceAcceptedGeometricVolumeQA(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
 let brick=wid.x;
 var sums=vec4f(0.0);var counts=vec4f(0.0);
 var extrema=vec4f(3.402823466e38,-3.402823466e38,0.0,0.0);
 var outside=vec4f(0.0);var excess=vec4f(0.0);
 if(brickActive(brick)){
  let cells=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  for(var local=lane;local<cells.y;local+=64u){
   let cell=cells.x+local;let volume=cellVolume(cell);
   let rho=state[sourceDensity()+cell];let open=cellOpenFraction(cell);
   let v=rho*volume;let c=open*volume;
   counts.x+=1.0;
   let velocityAt=sourceCellVelocity()+4u*cell;
   let dynamicsFinite=gvqaFinite(state[velocityAt])
    &&gvqaFinite(state[velocityAt+1u])&&gvqaFinite(state[velocityAt+2u])
    &&gvqaFinite(state[p.stateOffsets2.x+cell])
    &&gvqaFinite(state[p.stateOffsets2.y+cell])
    &&gvqaFinite(state[p.stateOffsets2.z+cell]);
   if(!dynamicsFinite){sums.w+=1.0;}
   let finite=gvqaFinite(v)&&gvqaFinite(c)&&gvqaFinite(rho)&&gvqaFinite(open);
   if(!finite){counts.z+=1.0;counts.w+=1.0;continue;}
   sums.x+=v;sums.y+=c;
   let margin=8.0*1.1920928955078125e-7*c;
   if(c<0.0||v< -margin||(c==0.0&&v!=0.0)){counts.w+=1.0;}
   if(c==0.0){counts.y+=1.0;if(v!=0.0){sums.z+=1.0;}}
   else if(c>0.0){extrema.x=min(extrema.x,v/c);extrema.y=max(extrema.y,v/c);}
   let extra=max(v-c,0.0);excess.x+=extra;excess.y+=select(0.0,1.0,extra>margin);
   excess.z=max(excess.z,extra);
   extrema.z=max(extrema.z,max(-v,select(0.0,v,c==0.0)));
   extrema.w=max(extrema.w,max(-rho,select(0.0,rho,c==0.0)));
   let lo=cellCenter(cell)-0.5*cellWidths(cell);
   let hi=cellCenter(cell)+0.5*cellWidths(cell);
   let domain=vec3f(p.dimensions.xyz);
   if(any(hi<=vec3f(0.0))||any(lo>=domain)){outside.x+=v;outside.z+=1.0;}
   else if(any(lo<vec3f(0.0))||any(hi>domain)){
    // A cell average does not locate liquid within a straddling cell. Keep
    // that entire amount separate rather than claiming an exact outside split.
    outside.y+=v;outside.w+=1.0;
   }
  }
 }
 gvqaSum[lane]=sums;gvqaCounts[lane]=counts;
 gvqaExcess[lane]=excess;
 gvqaExtrema[lane]=extrema;gvqaOutside[lane]=outside;
 workgroupBarrier();var width=32u;
 loop{if(lane<width){
  gvqaSum[lane]+=gvqaSum[lane+width];gvqaCounts[lane]+=gvqaCounts[lane+width];
  gvqaOutside[lane]+=gvqaOutside[lane+width];
  gvqaExcess[lane]=vec4f(gvqaExcess[lane].xy+gvqaExcess[lane+width].xy,
    max(gvqaExcess[lane].z,gvqaExcess[lane+width].z),0.0);
  gvqaExtrema[lane]=vec4f(min(gvqaExtrema[lane].x,gvqaExtrema[lane+width].x),
   max(gvqaExtrema[lane].y,gvqaExtrema[lane+width].y),
   max(gvqaExtrema[lane].z,gvqaExtrema[lane+width].z),
   max(gvqaExtrema[lane].w,gvqaExtrema[lane+width].w));
 }workgroupBarrier();if(width==1u){break;}width/=2u;}
 if(lane==0u){geometricVolumeQA[5u*brick]=gvqaSum[0];
  geometricVolumeQA[5u*brick+1u]=gvqaCounts[0];
  geometricVolumeQA[5u*brick+2u]=gvqaExtrema[0];
  geometricVolumeQA[5u*brick+3u]=gvqaOutside[0];
  geometricVolumeQA[5u*brick+4u]=gvqaExcess[0];}
}
`;
