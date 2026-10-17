/** Conservative single-sample raster coverage. This is not a size/area LOD:
 * a subpixel triangle is kept whenever it can cover a pixel centre. The full
 * mesh remains resident and coverage is recomputed for every camera/viewport.
 * Only use with the primary pass's default viewport and sampleCount=1. */
export const svoMeshSampleCoverageWGSL = /* wgsl */ `
fn meshCoverageEdge(a:vec2f,b:vec2f,p:vec2f)->f32{
  let e=b-a;let q=p-a;return e.x*q.y-e.y*q.x;
}
fn meshTriangleMayCoverSample(aClip:vec4f,bClip:vec4f,cClip:vec4f,viewport:vec2f)->bool{
  // Let hardware clip near-plane crossings. Avoid division at/behind the eye
  // and large offscreen coordinates where edge arithmetic loses precision.
  if(min(aClip.w,min(bClip.w,cClip.w))<=aClip.z){return true;}
  let a=(aClip.xy/aClip.w*vec2f(.5,-.5)+vec2f(.5))*viewport;
  let b=(bClip.xy/bClip.w*vec2f(.5,-.5)+vec2f(.5))*viewport;
  let c=(cClip.xy/cClip.w*vec2f(.5,-.5)+vec2f(.5))*viewport;
  let lo=min(a,min(b,c));let hi=max(a,max(b,c));
  if(any(lo<vec2f(-32768))||any(hi>vec2f(32768))){return true;}
  // Expand by a full 1/16 pixel in both axes for rasterizer vertex snapping
  // and floating-point projection differences. Boundary cases stay drawable.
  let guard=1./16.;
  let first=max(vec2f(0),ceil(lo-vec2f(.5+guard)));
  let last=min(viewport-vec2f(1),floor(hi-vec2f(.5-guard)));
  if(any(first>last)){return false;}
  // Only enumerate small bounds; larger geometry goes straight to hardware.
  if(any(last-first>vec2f(1))){return true;}
  let area=meshCoverageEdge(a,b,c);
  if(abs(area)<1e-8){return true;}
  let winding=select(-1.,1.,area>0.);
  let ab=guard*(abs(b.x-a.x)+abs(b.y-a.y))+1e-4;
  let bc=guard*(abs(c.x-b.x)+abs(c.y-b.y))+1e-4;
  let ca=guard*(abs(a.x-c.x)+abs(a.y-c.y))+1e-4;
  for(var y=first.y;y<=last.y;y+=1.){for(var x=first.x;x<=last.x;x+=1.){
    let p=vec2f(x,y)+vec2f(.5);
    if(winding*meshCoverageEdge(a,b,p)>=-ab&&winding*meshCoverageEdge(b,c,p)>=-bc&&winding*meshCoverageEdge(c,a,p)>=-ca){return true;}
  }}
  return false;
}
`;
