/**
 * LANE F OWNS THIS FILE. The sources' WGSL (M4, brick-source.ts): this step's drop from the ring's SOURCE words and
 * F's `src` buffer (its extrusion edges), in cells. Every function is the old engine's, its metres over h
 * (uniform-source.wgsl.ts dropDistance and uvSourcePhi; uniform-mixed-momentum.ts umDropFace; uniform-mixed-dynamic.ts
 * umSourceTile; uniform-region-heat.ts's source trigger), with traceWorld the identity: positions are cells.
 *
 * A kernel that reads it binds `src` (BRICK_SOURCE_BINDING, read-only f32) and reads the ring at cfg.x.
 */
import type { BrickBinding } from "./brick-kernel";
import { SRC_PIN_MARGIN, SRC_RING } from "./brick-source";

export const BRICK_SOURCE_BINDING: BrickBinding = { name: "src", kind: "ro", type: "array<f32>" };
/** The `src` buffer's words: the scene's extrusion edges (4 a edge), at least 4 (a scene with no extrusion). */
export const brickSourceWords = (edges: number) => Math.max(4, 4 * edges);

/** The ring's source words alone (no `src` binding): the drop's presence, the added cells and the boxes. */
export const BRICK_SOURCE_RING_WGSL = /* wgsl */`
const NB_SRC_KIND:u32=${SRC_RING.KIND}u;const NB_SRC_CENTRE:u32=${SRC_RING.CENTRE}u;const NB_SRC_RADIUS:u32=${SRC_RING.RADIUS}u;
const NB_SRC_HALF_HEIGHT:u32=${SRC_RING.HALF_HEIGHT}u;const NB_SRC_EDGE_START:u32=${SRC_RING.EDGE_START}u;const NB_SRC_EDGE_COUNT:u32=${SRC_RING.EDGE_COUNT}u;
const NB_SRC_OFFSET:u32=${SRC_RING.OFFSET}u;const NB_SRC_HALF_DEPTH:u32=${SRC_RING.HALF_DEPTH}u;const NB_SRC_EDGE_RADIUS:u32=${SRC_RING.EDGE_RADIUS}u;
const NB_SRC_BOX:u32=${SRC_RING.BOX}u;const NB_SRC_AHEAD:u32=${SRC_RING.AHEAD}u;const NB_SRC_BOXES:u32=${SRC_RING.BOXES}u;const NB_SRC_ADDED:u32=${SRC_RING.ADDED}u;
const NB_SRC_PIN_MARGIN:i32=${SRC_PIN_MARGIN};
// This step's drop (the old params.drop.w > 0).
fn nbDropOn()->bool{return nbRing(cfg.x,NB_SRC_KIND)!=0u;}
fn nbDropCentre()->vec3f{return vec3f(nbRingF(cfg.x,NB_SRC_CENTRE),nbRingF(cfg.x,NB_SRC_CENTRE+1u),nbRingF(cfg.x,NB_SRC_CENTRE+2u));}
// The cells this step adds (the drop's volume, cells).
fn nbSourceAdded()->f32{return nbRingF(cfg.x,NB_SRC_ADDED);}
// The ring's drop box k (0 this step's, 1 the next step's): lo inclusive, hi exclusive (cells); empty when absent.
fn nbSourceBox(k:u32)->array<vec3i,2>{
  if((nbRing(cfg.x,NB_SRC_BOXES)&(1u<<k))==0u){return array<vec3i,2>(vec3i(0),vec3i(0));}
  let at=select(NB_SRC_BOX,NB_SRC_AHEAD,k==1u);
  let lo=vec3i(i32(nbRing(cfg.x,at)),i32(nbRing(cfg.x,at+1u)),i32(nbRing(cfg.x,at+2u)));
  let hi=vec3i(i32(nbRing(cfg.x,at+3u)),i32(nbRing(cfg.x,at+4u)),i32(nbRing(cfg.x,at+5u)));
  return array<vec3i,2>(clamp(lo,vec3i(0),NB_LATTICE),clamp(hi,vec3i(0),NB_LATTICE));
}
`;

/** The drop's geometry (binds `src`), with the ring's words. */
export const BRICK_SOURCE_WGSL = BRICK_SOURCE_RING_WGSL + /* wgsl */`
// Signed distance in the x/y plane to the drop's raw outline; negative inside. The fill is even-odd, so a contour
// inside another is a hole (dropOutlineDistance). The edge count is held to the binding.
fn nbDropOutline(point:vec2f)->f32{
  let start=nbRing(cfg.x,NB_SRC_EDGE_START);let edges=arrayLength(&src)/4u;
  let count=min(nbRing(cfg.x,NB_SRC_EDGE_COUNT),edges-min(start,edges));
  var nearest=3.0e38;var inside=false;
  for(var i=0u;i<count;i++){
    let at=4u*(start+i);let a=vec2f(src[at],src[at+1u]);let b=vec2f(src[at+2u],src[at+3u]);let e=b-a;let w=point-a;
    let off=w-clamp(dot(w,e)/dot(e,e),0.0,1.0)*e;nearest=min(nearest,dot(off,off));
    // The ray toward +x; the half-open test counts a shared vertex once.
    if((a.y>point.y)!=(b.y>point.y)&&point.x<a.x+(point.y-a.y)*e.x/e.y){inside=!inside;}
  }
  return select(sqrt(nearest),-sqrt(nearest),inside);
}
// Signed distance (cells) from point p (cells) to this step's drop; negative is liquid (dropDistance). An extrusion
// past its bounding ball by ceiling returns the ball's distance, a lower bound: a caller's min is unchanged.
fn nbDropDistance(p:vec3f,ceiling:f32)->f32{
  let kind=nbRing(cfg.x,NB_SRC_KIND);let d=p-nbDropCentre();let r=nbRingF(cfg.x,NB_SRC_RADIUS);
  if(kind==1u){return length(d)-r;}
  if(kind==2u){return max(length(d.xy)-r,abs(d.z)-nbRingF(cfg.x,NB_SRC_HALF_HEIGHT));}
  let bound=length(d)-r;if(bound>=ceiling){return bound;}
  // A rounded extrusion of the outline grown by its offset.
  let er=nbRingF(cfg.x,NB_SRC_EDGE_RADIUS);
  let w=vec2f(nbDropOutline(p.xy)-nbRingF(cfg.x,NB_SRC_OFFSET),abs(d.z)-nbRingF(cfg.x,NB_SRC_HALF_DEPTH))+vec2f(er);
  return min(max(w.x,w.y),0.0)+length(max(w,vec2f(0.0)))-er;
}
// φ (cells) at p with this step's source: the union min(φ, drop) (uvSourcePhi; nbSourcePhi in h).
fn nbSourcePhi(p:vec3f,value:f32)->f32{
  if(!nbDropOn()){return value;}
  return min(value,nbDropDistance(p,value));
}
// A face (centre c, cells) of width w (1 an h face, 4 a 4h face) within a face width of the drop: its liquid arrives
// at rest (umDropFace).
fn nbDropFace(c:vec3f,w:f32)->bool{
  if(!nbDropOn()){return false;}
  return nbDropDistance(c,w)<w;
}
// A 4h tile the drop can fill (umSourceTile, the drop's half): its centre within half the tile's diagonal and two
// cells of the drop.
fn nbSourceTile(t:vec3i)->bool{
  if(!nbDropOn()){return false;}
  return nbSourcePhi(4.0*vec3f(t)+vec3f(2.0),3.0e38)<=0.5*length(vec3f(4.0))+2.0;
}
// A 16-cell region the drop reaches (uniform-region-heat.ts's trigger 16): its centre within the region's half
// diagonal and a tile.
fn nbSourceRegion(r:vec3i)->bool{
  if(!nbDropOn()){return false;}
  return nbSourcePhi(16.0*vec3f(r)+vec3f(8.0),1.0e30)<8.0*1.7320508+4.0;
}
`;
