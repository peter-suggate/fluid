/** Dense render vertices only: these do not request simulation h owners.
 * A weighted particle surface overlaps a shrunken coarse bulk interior. */
export const narrowBandSurfaceWGSL=/* wgsl */`
fn surfaceWeight(d:vec3f)->f32{let t=max(0.0,1.0-dot(d,d)/4.0);return t*t*t;}
@compute @workgroup_size(64) fn surface(@builtin(global_invocation_id) gid:vec3u){
 let dims=UM_D+1u;let count=dims.x*dims.y*dims.z;
 for(var i=gid.x;i<count;i+=65536u){
  let q=vec3i(vec3u(i%dims.x,(i/dims.x)%dims.y,i/(dims.x*dims.y)));let x=vec3f(q);
  let bulk=bandPhi(x);var value=bulk;
  if(bulk>=-2.0&&bulk<=4.0){
   var sum=vec3f(0);var total=0.0;var wallMask=0u;var mirrorOrigin=vec3f(0);
   for(var a=0u;a<3u;a++){
    if(x[a]<2.0){wallMask|=1u<<a;}
    else if(x[a]>f32(UM_D[a])-2.0&&!(a==1u&&params.settings.y>0.5)){wallMask|=1u<<a;mirrorOrigin[a]=2.0*f32(UM_D[a]);}
   }
   for(var z=-2;z<=1;z++){for(var y=-2;y<=1;y++){for(var k=-2;k<=1;k++){
    let cell=q+vec3i(k,y,z);if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
    var link=atomicLoad(&bins[2u*cellIndex(cell)]);
    for(var j=0u;j<16u&&link!=0u;j++){
     let index=link-1u;let centre=particles[index].position.xyz;
     let w=surfaceWeight(x-centre);sum+=w*centre;total+=w;
     if(wallMask!=0u){let mirror=mirrorOrigin-centre;
      for(var mask=1u;mask<8u;mask++){
       if((mask&wallMask)!=mask){continue;}
       let c=select(centre,mirror,vec3<bool>((mask&1u)!=0u,(mask&2u)!=0u,(mask&4u)!=0u));
       let mw=surfaceWeight(x-c);sum+=mw*c;total+=mw;
      }
     }
     link=links[index];
    }
   }}}
   var particlePhi=4.0;if(total>1e-6){particlePhi=length(x-sum/total)-0.5;}
   // The coarse interior fills the particle-free bulk. The two-cell overlap
   // avoids exposing the inner edge of the four-cell particle shell.
   value=min(bulk+2.0,particlePhi);
  }
  textureStore(surfacePhi,q,vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
  if(all(q<vec3i(UM_D))){textureStore(surfaceOpen,q,vec4f(umCellOpen(q)));}
 }
}
`;

/** One workgroup per coarse face, with an exact particle gather and parallel
 * reduction. It uses the same quadratic kernel as the h transfer, at H=4h. */
export const narrowBandCoarseTransferWGSL=/* wgsl */`
var<workgroup> coarseSums:array<vec2f,64>;
@compute @workgroup_size(64) fn transferCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+256u*group.y;if(tile>=UM_T.x*UM_T.y*UM_T.z){return;}
 let axis=group.z;let origin=vec3i(vec3u(tile%UM_T.x,(tile/UM_T.x)%UM_T.y,tile/(UM_T.x*UM_T.y)))*4;
 var anchor=origin;anchor[axis]+=3;var q=vec3f(origin)+2.0;q[axis]+=2.0;
 let depth=bandPhi(q);let blend=clamp((depth+4.0)/2.0,0.0,1.0);
 let original=textureLoad(velocity,anchor,0);var sum=vec2f(0);
 if(blend>0.0&&depth<=1.5&&q[axis]<f32(UM_D[axis])){
  let lo=max(vec3i(floor(q-6.0)),vec3i(0));let hi=min(vec3i(ceil(q+6.0)),vec3i(UM_D));let size=vec3u(hi-lo);
  for(var cell=lane;cell<size.x*size.y*size.z;cell+=64u){
   let p=lo+vec3i(vec3u(cell%size.x,(cell/size.x)%size.y,cell/(size.x*size.y)));var link=atomicLoad(&bins[2u*cellIndex(p)]);
   for(var j=0u;j<16u&&link!=0u;j++){
    let i=link-1u;let r=(q-particles[i].position.xyz)/4.0;let w=weight(r.x)*weight(r.y)*weight(r.z);
    sum+=vec2f(w*particles[i].velocity[axis],w);link=links[i];
   }
  }
 }
 coarseSums[lane]=sum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){coarseSums[lane]+=coarseSums[lane+stride];}workgroupBarrier();}
 if(lane==0u){var value=original;let result=coarseSums[0];if(result.y>1e-5){value[axis]=mix(original[axis],result.x/result.y,blend);}textureStore(output,anchor,value);}
}
`;
