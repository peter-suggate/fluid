/** Searching extension requests are fixed MAC lattice sites. The caller has
 * rejected exterior and direct requests. Compile their remaining geometry
 * from width and plane alignment, then retain the original geometric tie,
 * distance tie and ordered value reduction. No owner/face discovery. */
export const uniformCompiledExtensionNeighborWGSL = /* wgsl */ `
fn ueCompiledNeighbor(point:vec3f,center:vec3f,component:u32,step:u32,width:u32)->UMNeighbor{
 let u=(component+1u)%3u;let v=(component+2u)%3u;
 let plane=i32(round(point[component]));
 var candidates:array<vec4i,4>;var n=0u;
 if(width==4u){
  // A non-direct 4h request lies on an h face plane. Its transverse
  // coordinates are integers: four h patches meet at the requested point.
  for(var k=0u;k<4u;k++){
   var anchor=vec3i(point);anchor[component]=plane-1;
   anchor[u]+=select(-1,0,(k&1u)!=0u);anchor[v]+=select(-1,0,(k&2u)!=0u);
   candidates[k]=vec4i(anchor,1);
  }
  n=4u;
 }else if((plane&3)==0){
  // A non-direct unit request on a 4h plane names one coarse patch.
  var anchor=(vec3i(floor(point))/4)*4;anchor[component]=plane-1;
  candidates[0]=vec4i(anchor,4);n=1u;
 }else{
  // A unit request inside a 4h cell sees its two bounding faces. Each
  // bounding face is either one coarse patch or the h patch at this column.
  let tile=min(vec3u(point),UM_D-vec3u(1))/4u;let origin=vec3i(tile*4u);
  let fine=umFineFaceSides(umTileAt(tile));
  for(var side=0u;side<2u;side++){
   let code=2u*component+1u-side;let patchWidth=select(4,1,(fine&(1u<<code))!=0u);
   var anchor=origin;anchor[component]+=select(-1,3,side!=0u);
   anchor[u]+=(i32(floor(point[u]))-origin[u])/patchWidth*patchWidth;
   anchor[v]+=(i32(floor(point[v]))-origin[v])/patchWidth*patchWidth;
   candidates[side]=vec4i(anchor,patchWidth);
  }
  n=2u;
 }
 var spatials:array<f32,4>;var slots:array<vec2f,4>;var nearest=UM_INF;
 for(var i=0u;i<n;i++){
  spatials[i]=UM_INF;let face=candidates[i];
  var location=vec3f(face.xyz)+vec3f(0.5*f32(face.w));location[component]=f32(face[component]+1);
  let delta=(location-center)*h.xyz;
  if(abs(location[step]-center[step])<1e-5){continue;}
  spatials[i]=dot(delta,delta);slots[i]=umSlotState(face.xyz,component,u32(face.w));nearest=min(nearest,spatials[i]);
 }
 if(nearest>=0.5*UM_INF){return UMNeighbor(0,UM_INF,1);}
 var distance=UM_INF;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)){distance=min(distance,slots[i].y);}}
 var sum=0.0;var count=0.0;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)&&abs(slots[i].y-distance)<=1e-6*max(1.0,distance)){sum+=slots[i].x;count+=1.0;}}
 return UMNeighbor(select(0.0,sum/count,count>0.0),distance,sqrt(nearest));
}
`;
