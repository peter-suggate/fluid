/** Reuse the reconstructed surface's bounded crossing search for redistance
 * and particle membership. Vertex signs stay fixed so the crossing-cell cache remains valid.
 * Coarse values are advected surface samples, not a particle distance field;
 * keep them unchanged, which also preserves hanging-vertex signs. */
export const narrowBandRedistanceWGSL=/* wgsl */`
fn nbTrilinear(v:array<f32,8>,q:vec3f)->vec4f{
 let x00=mix(v[0],v[1],q.x);let x10=mix(v[2],v[3],q.x);
 let x01=mix(v[4],v[5],q.x);let x11=mix(v[6],v[7],q.x);
 let y0=mix(x00,x10,q.y);let y1=mix(x01,x11,q.y);
 let dx=mix(mix(v[1]-v[0],v[3]-v[2],q.y),mix(v[5]-v[4],v[7]-v[6],q.y),q.z);
 return vec4f(dx,mix(x10-x00,x11-x01,q.z),y1-y0,mix(y0,y1,q.z));
}
fn nbSurfaceDistance(p:vec3f,cell:u32)->f32{
 let origin=vec3f(nbCell(cell));var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=bandPhi(origin+vec3f(umCorner(k,2u)));}
 // Edge crossings supply a zero-set fallback if a clamped Newton solve
 // cannot reach zero inside this cell (thin or badly conditioned patches).
 var distance=1e20;
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
  distance=min(distance,length(p-q));
 }}
 var q=clamp(p-origin,vec3f(0),vec3f(1));
 for(var i=0u;i<6u;i++){
  let f=nbTrilinear(v,q);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbTrilinear(v,q).w)<1e-4){distance=min(distance,length(p-origin-q));}
 return distance;
}
@compute @workgroup_size(64) fn buildDistance(@builtin(global_invocation_id) gid:vec3u){
 let dims=UM_D+1u;let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 for(var i=gid.x;i<dims.x*dims.y*dims.z;i+=65536u){
  let p=vec3u(i%dims.x,(i/dims.x)%dims.y,i/(dims.x*dims.y));let initial=umSampleVertex(vec3f(p));
  var value=initial/h;
  if(abs(initial)>1e-8){
   let nearest=atomicLoad(&bins[NB_DEPTH_B+cellIndex(vec3i(min(p,UM_D-1u)))]);
   var distance=max(abs(value),6.0);
   if(nearest!=NB_NO_SURFACE){distance=nbSurfaceDistance(vec3f(p),nearest);}
   // Cell-centre proximity alone can select the wrong interface beside a
   // tiny ripple. Also trace the local phi gradient across cell boundaries;
   // unlike a cell-clamped solve this reaches the surface above that vertex.
   if(abs(value)<4.5){
    var q=vec3f(p);
    for(var iteration=0u;iteration<5u;iteration++){
     let f=bandPhi(q);if(abs(f)<1e-5){distance=min(distance,length(q-vec3f(p)));break;}
     var g=vec3f(0);
     for(var axis=0u;axis<3u;axis++){
      var e=vec3f(0);e[axis]=0.5;
      let lo=clamp(q-e,vec3f(0),vec3f(UM_D));let hi=clamp(q+e,vec3f(0),vec3f(UM_D));
      g[axis]=(bandPhi(hi)-bandPhi(lo))/max(hi[axis]-lo[axis],0.01);
     }
     let g2=dot(g,g);if(g2<1e-8){break;}
     q=clamp(q-clamp(f/sqrt(g2),-1.5,1.5)*g*inverseSqrt(g2),vec3f(0),vec3f(UM_D));
    }
   }
   value=sign(initial)*max(distance,1e-8/h);
  }
  atomicStore(&bins[NB_DEPTH_A+i],bitcast<u32>(value));
 }
}
fn nbInterfaceVertex(p:vec3u)->bool{
 // Every crossing cell retains itself as its nearest seed. A vertex touching
 // any such cell determines its interpolated zero set: changing its magnitude
 // would move that surface even if its sign stayed fixed.
 for(var k=0u;k<8u;k++){
  let c=vec3i(p)-vec3i(umCorner(k,2u));
  if(any(c<vec3i(0))||any(c>=vec3i(UM_D))){continue;}
  let index=cellIndex(c);
  if(atomicLoad(&bins[NB_DEPTH_B+index])==index){return true;}
 }
 return false;
}
fn nbRedistanceVertex(p:vec3u,width:u32){
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 // Coarse samples retain the reconstructed field and hanging-vertex signs.
 var value=umSampleVertex(vec3f(p));
 if(width==1u&&!nbInterfaceVertex(p)){value=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbVertexIndex(p)]))*h;}
 textureStore(outputPhi,vec3i(p),vec4f(value));
}
@compute @workgroup_size(128) fn redistanceFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(group.x*64u,0u,0u));let origin=umTileCoord(owner.tile)*4u;
 if(lane>=125u){return;}
 let local=vec3u(lane%5u,(lane/5u)%5u,lane/25u);let p=origin+local;
 let regular=umTileMaximumWidth(owner.tile)==1u&&umTileMinimumWidth(owner.tile)==1u;
 if(regular){if(!all((local>vec3u(0))|(origin==vec3u(0)))){return;}}
 else{let a=umVertexAuthority(p);if(a.width!=1u||a.tile!=owner.tile){return;}}
 nbRedistanceVertex(p,1u);
}
@compute @workgroup_size(64) fn redistanceCoarse(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<umCounts.y;i+=65536u){
  let owner=umAllOwner(vec3u(umCounts.x*64u+i,0u,0u));let origin=umOrigin(owner);
  let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let p=origin+corner*4u;
   if(regular){if(!all((corner!=vec3u(0))|(origin==vec3u(0)))){continue;}}
   else{if(umVertexAuthority(p).index!=owner.index){continue;}}
   nbRedistanceVertex(p,4u);
  }
 }
}
`;
