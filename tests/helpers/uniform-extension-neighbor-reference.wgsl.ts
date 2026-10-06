/** Frozen pre-recipe extension neighbor traversal for independent parity. */
export const uniformExtensionNeighborReferenceWGSL = /* wgsl */ `
struct UMNeighbor {value:f32,distance:f32,spacing:f32}
fn umNeighbor(point:vec3f,center:vec3f,component:u32,step:u32,width:u32)->UMNeighbor{
 if(any(point<vec3f(0))||any(point>vec3f(UM_D))){return UMNeighbor(0,UM_INF,1);}
 let tile=umTileAt(min(vec3u(point),UM_D-vec3u(1))/4u);
 // Both direct cases read the width-w patch slot below the plane (w is 1 on
 // the all-h stencil path). Its anchor needs no topology, so the support and
 // width words issue beside the slot's address instead of behind its value.
 var offset=vec3f(0.5*f32(width));offset[component]=1.0;
 // The first case below always reads a unit patch (every cell of an all-h
 // stencil, and so the one below the plane, is h); the second reads the
 // width-w patch. Both issue here, before the topology words resolve.
 let anchor=vec3i(round(point-offset));let unit=umSlotState(anchor,component,1u);
 var direct=unit;if(width!=1u){direct=umSlotState(anchor,component,width);}
 // A unit request whose plane is interior to a unit tile has unit cells on
 // both sides: the case below with lowWidth = highWidth = 1, one load.
 let interior=width==1u&&(u32(round(point[component]))&3u)!=0u&&umTileWidth(tile)==1u;
 if(umRegularFine||interior||umTileMaximumWidth(tile)==1u){return UMNeighbor(unit.x,unit.y,h[step]);}
 // The point is a width-w lattice patch centre (the requesting face has
 // width w). When the owners on both sides of its plane are no finer than w
 // and one of them has width w (or the plane is a domain wall and the inner
 // owner has width w), the lower owner's positive face has patches of width
 // min(low, high) = w, one of them centred here: the search below selects
 // exactly it.
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 let lowWidth=select(0u,umTileWidth(umTileAt(vec3u(max(below,vec3i(0)))/4u)),plane>0);
 let highWidth=select(0u,umTileWidth(umTileAt(min(vec3u(above),UM_D-vec3u(1))/4u)),plane<i32(UM_D[component]));
 if(select((lowWidth==width&&(highWidth==0u||highWidth>=width))||(highWidth==width&&lowWidth>width),highWidth==width,lowWidth==0u)){
  // The point is center ± width along step only: the spacing is exactly
  // width*h, as on the fast path above. sqrt(dot) can land an ulp above it,
  // and the root cutoff below sits exactly on whole-cell distances, so which
  // path a face took (the +side tile decides) would decide its reach.
  return UMNeighbor(direct.x,direct.y,abs(point[step]-center[step])*h[step]);
 }
 let site=umVelocitySite(point,component);
 // Candidate real faces; choose geometrically, never read an unowned fine
 // texel. A requested plane inside a coarser cell has two incident faces. A
 // wider patch's centre on finer owners sits on fine cell boundaries in both
 // tangential axes: every tied cell's face is a candidate. Ties are resolved
 // by value, never by order: floor() (and a strict first-wins compare) takes
 // the +side (-side) cell of a tie, and a mirror maps it to the other side.
 let owner=umOwnerAt(min(vec3i(floor(point)),vec3i(UM_D)-vec3i(1)));
 let u=(component+1u)%3u;let v=(component+2u)%3u;
 var faces:array<UMFace,4>;var n=0u;
 if(site.interior){
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,component,sign,0u);
   let local=clamp(point-vec3f(umOrigin(owner)),vec3f(0),vec3f(f32(owner.width)-1e-4));
   faces[n]=umFace(owner,component,sign,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));n++;
  }
 }else if(site.width<width){
  for(var k=0u;k<4u;k++){
   var tied=point;tied[u]+=select(-0.5,0.5,(k&1u)!=0u);tied[v]+=select(-0.5,0.5,(k&2u)!=0u);
   let s=umVelocitySite(tied,component);if(!s.interior){faces[n]=s.face;n++;}
  }
 }else{faces[0]=site.face;n=1u;}
 // Nearest faces, then the least slot distance among them, then the mean
 // value of the faces holding it.
 var spatials:array<f32,4>;var slots:array<vec2f,4>;var nearest=UM_INF;
 for(var i=0u;i<n;i++){
  spatials[i]=UM_INF;let face=faces[i];if(face.width==0u){continue;}
  let location=umFaceCenter(face);let delta=(location-center)*h.xyz;
  if(abs(location[step]-center[step])<1e-5){continue;}
  spatials[i]=dot(delta,delta);slots[i]=umSlotState(face.anchor,component,face.width);nearest=min(nearest,spatials[i]);
 }
 if(nearest>=0.5*UM_INF){return UMNeighbor(0,UM_INF,1);}
 var distance=UM_INF;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)){distance=min(distance,slots[i].y);}}
 var sum=0.0;var count=0.0;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)&&abs(slots[i].y-distance)<=1e-6*max(1.0,distance)){sum+=slots[i].x;count+=1.0;}}
 return UMNeighbor(select(0.0,sum/count,count>0.0),distance,sqrt(nearest));
}
`;
