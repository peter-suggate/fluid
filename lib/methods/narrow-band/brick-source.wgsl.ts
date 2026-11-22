/**
 * LANE F OWNS THIS FILE. The sources' WGSL (M4, brick-source.ts): this step's drop from the ring's SOURCE words and
 * F's `src` buffer (its extrusion edges), in cells. Every function is the old engine's, its metres over h
 * (uniform-source.wgsl.ts dropDistance and uvSourcePhi; uniform-mixed-momentum.ts umDropFace; uniform-mixed-dynamic.ts
 * umSourceTile; uniform-region-heat.ts's source trigger), with traceWorld the identity: positions are cells.
 *
 * A kernel that reads it binds `src` (BRICK_SOURCE_BINDING, read-only f32) and reads the ring at cfg.x.
 *
 * The nozzle (BRICK_SOURCE_RING_WGSL, ring words only): inflow-boundary.ts's inflowApertureFraction,
 * isInflowVelocityCell, applyInflowVelocity (the projection's override), inflowSweptPlugSource,
 * isInflowSweptVelocityCell and applyInflowSweptVelocity (the forces' override), and uvSourcePhi's plug, in cells: a
 * cell's centre q + 1/2, the face plane face + 1, an edge width of half a face diagonal (sqrt 2 / 2). The old face's
 * texel q is the cell whose +face it is: the brick's lower face (c, a) is the old (c - e_a, a). A vector component at a
 * dynamic axis is read, never written (nbAxisPick).
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding } from "./brick-kernel";
import { brickDetailWGSL } from "./brick-detail.wgsl";
import { SRC_PIN_MARGIN, SRC_PLUG_OVERLAP, SRC_RING } from "./brick-source";

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
// The ring's source box k (0 this step's drop, 1 the next step's, 2 the nozzle's plug): lo inclusive, hi exclusive
// (cells); empty when absent.
fn nbSourceBox(k:u32)->array<vec3i,2>{
  if((nbRing(cfg.x,NB_SRC_BOXES)&(1u<<k))==0u){return array<vec3i,2>(vec3i(0),vec3i(0));}
  let at=select(select(NB_SRC_BOX,NB_SRC_AHEAD,k==1u),${SRC_RING.INFLOW_BOX}u,k==2u);
  let lo=vec3i(i32(nbRing(cfg.x,at)),i32(nbRing(cfg.x,at+1u)),i32(nbRing(cfg.x,at+2u)));
  let hi=vec3i(i32(nbRing(cfg.x,at+3u)),i32(nbRing(cfg.x,at+4u)),i32(nbRing(cfg.x,at+5u)));
  return array<vec3i,2>(clamp(lo,vec3i(0),NB_LATTICE),clamp(hi,vec3i(0),NB_LATTICE));
}
// The nozzle (inflow-boundary.ts, in cells). On: this step's strength > 0 (inflowStrength() > 0).
const NB_IN_FLAGS:u32=${SRC_RING.INFLOW}u;const NB_IN_STRENGTH:u32=${SRC_RING.STRENGTH}u;const NB_IN_OUTLET:u32=${SRC_RING.OUTLET}u;
const NB_IN_RADIUS:u32=${SRC_RING.IN_RADIUS}u;const NB_IN_VELOCITY:u32=${SRC_RING.VELOCITY}u;const NB_IN_SCALE:u32=${SRC_RING.SCALE}u;
const NB_IN_SWEEP:u32=${SRC_RING.SWEEP}u;const NB_IN_AXIS:u32=${SRC_RING.AXIS}u;const NB_IN_FACE:u32=${SRC_RING.FACE}u;
const NB_IN_OVERLAP:f32=${SRC_PLUG_OVERLAP}.0;
fn nbInflowOn()->bool{return (nbRing(cfg.x,NB_IN_FLAGS)&1u)!=0u;}
// The nozzle ran before this step: last step's plug lies past this one's front (physical.w).
fn nbInflowContinues()->bool{return (nbRing(cfg.x,NB_IN_FLAGS)&2u)!=0u;}
// inflowNeedsFine (the host's, from the step's strength and sweep).
fn nbInflowNeedsFine()->bool{return (nbRing(cfg.x,NB_IN_FLAGS)&4u)!=0u;}
fn nbInflowStrength()->f32{return nbRingF(cfg.x,NB_IN_STRENGTH);}
fn nbInflowOutlet()->vec3f{return vec3f(nbRingF(cfg.x,NB_IN_OUTLET),nbRingF(cfg.x,NB_IN_OUTLET+1u),nbRingF(cfg.x,NB_IN_OUTLET+2u));}
fn nbInflowVelocity()->vec3f{return vec3f(nbRingF(cfg.x,NB_IN_VELOCITY),nbRingF(cfg.x,NB_IN_VELOCITY+1u),nbRingF(cfg.x,NB_IN_VELOCITY+2u));}
fn nbInflowAxis()->u32{return nbRing(cfg.x,NB_IN_AXIS);}
fn nbInflowFace()->i32{return bitcast<i32>(nbRing(cfg.x,NB_IN_FACE));}
fn nbAxisPick(v:vec3f,a:u32)->f32{return select(select(v.z,v.y,a==1u),v.x,a==0u);}
fn nbAxisPickI(v:vec3i,a:u32)->i32{return select(select(v.z,v.y,a==1u),v.x,a==0u);}
fn nbAxisMask(a:u32)->vec3<bool>{return vec3<bool>(a==0u,a==1u,a==2u);}
// inflowReceiverIndex, and the donor upstream of it.
fn nbInflowReceiver()->i32{let f=nbInflowFace();return select(f,f+1,nbAxisPick(nbInflowVelocity(),nbInflowAxis())>=0.0);}
fn nbInflowDonor()->i32{let r=nbInflowReceiver();return select(r+1,r-1,nbAxisPick(nbInflowVelocity(),nbInflowAxis())>=0.0);}
// The aperture's radial coverage at relative point rel along direction dir: a smoothstep over half a face diagonal.
fn nbInflowCoverage(rel:vec3f,dir:vec3f)->f32{
  let axial=dot(rel,dir);let radial=length(rel-axial*dir);
  let c=clamp(0.5+0.5*(nbRingF(cfg.x,NB_IN_RADIUS)-radial)/(0.5*sqrt(2.0)),0.0,1.0);return c*c*(3.0-2.0*c);
}
// inflowApertureFraction: q's tangential centre on the nozzle's face plane.
fn nbInflowAperture(q:vec3i)->f32{
  let v=nbInflowVelocity();let speed=length(v);if(speed<=1e-6){return 0.0;}
  let point=select(vec3f(q)+vec3f(0.5),vec3f(f32(nbInflowFace()+1)),nbAxisMask(nbInflowAxis()));
  return nbInflowCoverage(point-nbInflowOutlet(),v/speed);
}
// isInflowVelocityCell: the receiver or donor layer within the aperture.
fn nbInflowVelocityCell(q:vec3i)->bool{
  let qa=nbAxisPickI(q,nbInflowAxis());
  return nbInflowOn()&&(qa==nbInflowReceiver()||qa==nbInflowDonor())&&nbInflowAperture(q)>0.0;
}
// applyInflowVelocity's component a at q (the projection's): the nozzle's axis takes the desired velocity times the
// aperture scale; the receiver's other components the desired velocity; the donor's others stand.
fn nbInflowProjected(q:vec3i,a:u32,value:f32)->f32{
  if(!nbInflowOn()||!nbInflowVelocityCell(q)){return value;}
  let axis=nbInflowAxis();let desired=nbInflowVelocity()*nbInflowStrength();
  if(a==axis){return nbAxisPick(desired,axis)*nbRingF(cfg.x,NB_IN_SCALE);}
  if(nbAxisPickI(q,axis)==nbInflowReceiver()){return nbAxisPick(desired,a);}
  return value;
}
// inflowSweptPlugSource > 0: q's cell overlaps the plug the reservoir sweeps this step (its sweep in cells) within the
// aperture's coverage.
fn nbInflowSwept(q:vec3i)->bool{
  let v=nbInflowVelocity();let sweep=nbRingF(cfg.x,NB_IN_SWEEP);
  if(length(v)*nbInflowStrength()<=1e-6||sweep<=0.0){return false;}
  let dir=v/max(length(v),1e-6);let rel=vec3f(q)+vec3f(0.5)-nbInflowOutlet();
  let axial=dot(rel,dir);let half=0.5*dot(abs(dir),vec3f(1.0));
  let overlap=max(0.0,min(axial+half,sweep)-max(axial-half,0.0));
  return overlap>0.0&&nbInflowCoverage(rel,dir)*nbRingF(cfg.x,NB_IN_SCALE)>0.0;
}
// isInflowSweptVelocityCell: the donor within the aperture (upstream support) or a swept plug cell.
fn nbInflowSweptCell(q:vec3i)->bool{
  if(!nbInflowOn()){return false;}
  return (nbAxisPickI(q,nbInflowAxis())==nbInflowDonor()&&nbInflowAperture(q)>0.0)||nbInflowSwept(q);
}
// applyInflowSweptVelocity's component a at q (the forces'): the plug's own velocity, unscaled.
fn nbInflowForced(q:vec3i,a:u32,value:f32)->f32{
  if(!nbInflowSweptCell(q)){return value;}
  return nbAxisPick(nbInflowVelocity(),a)*nbInflowStrength();
}
// uvSourcePhi's plug (cells): the stream's side distance across its axial span; past the span the capped distance; on
// the stream's first step the front is a surface.
fn nbInflowPhi(p:vec3f,value:f32)->f32{
  let v=nbInflowVelocity();if(!nbInflowOn()||length(v)*nbInflowStrength()<=1e-6){return value;}
  let dir=normalize(v);let delta=p-nbInflowOutlet();let axial=dot(delta,dir);
  let side=length(delta-axial*dir)-nbRingF(cfg.x,NB_IN_RADIUS);
  let front=axial-nbRingF(cfg.x,NB_IN_SWEEP)-NB_IN_OVERLAP;let cap=max(-axial,front);
  return min(value,select(max(side,cap),select(max(side,front),side,nbInflowContinues()),cap<=0.0));
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
  var result=value;
  if(nbDropOn()){result=min(result,nbDropDistance(p,result));}
  return nbInflowPhi(p,result);
}
// Any source this step (the old params.drop.w > 0 || inflowStrength() > 0).
fn nbSourceOn()->bool{return nbDropOn()||nbInflowOn();}
// A face (centre c, cells) of width w (1 an h face, 4 a 4h face) within a face width of the drop: its liquid arrives
// at rest (umDropFace).
fn nbDropFace(c:vec3f,w:f32)->bool{
  if(!nbDropOn()){return false;}
  return nbDropDistance(c,w)<w;
}
// A 4h tile a source can fill (umSourceTile): its centre within half the tile's diagonal and two cells (the aperture
// edge and the donor layer upstream of the nozzle face) of the drop or the plug.
fn nbSourceTile(t:vec3i)->bool{
  if(!nbSourceOn()){return false;}
  return nbSourcePhi(4.0*vec3f(t)+vec3f(2.0),3.0e38)<=0.5*length(vec3f(4.0))+2.0;
}
// A 16-cell region the drop reaches (uniform-region-heat.ts's trigger 16): its centre within the region's half
// diagonal and a tile.
fn nbSourceRegion(r:vec3i)->bool{
  if(!nbSourceOn()){return false;}
  return nbSourcePhi(16.0*vec3f(r)+vec3f(8.0),1.0e30)<8.0*1.7320508+4.0;
}
`;

/**
 * The sources' WGSL for a φ writer (lAdvect, lUnion, the C level set's): BRICK_SOURCE_WGSL, and on a detail layout
 * (brick-detail.wgsl.ts) its relaxation after the sources, nbSourcePhi = nbDetailPhi(p, the sources' φ). The sources'
 * own tests (nbSourceTile, nbSourceRegion: the census's and the heat's) keep the sources alone (nbSourcePhiCore), so
 * the shell is neither a fine source nor a heat trigger. Off a detail layout the text is BRICK_SOURCE_WGSL's.
 */
export function brickSourceWGSL(L: BrickLayout): string {
  if (!L.detail) return BRICK_SOURCE_WGSL;
  const head = "fn nbSourcePhi(p:vec3f,value:f32)->f32{";
  if (BRICK_SOURCE_WGSL.split(head).length !== 2) throw new Error("brickSourceWGSL: nbSourcePhi's definition moved");
  const core = BRICK_SOURCE_WGSL.replace(head, "fn nbSourcePhiCore(p:vec3f,value:f32)->f32{")
    .replace("return nbSourcePhi(4.0*vec3f(t)+vec3f(2.0),3.0e38)", "return nbSourcePhiCore(4.0*vec3f(t)+vec3f(2.0),3.0e38)")
    .replace("return nbSourcePhi(16.0*vec3f(r)+vec3f(8.0),1.0e30)", "return nbSourcePhiCore(16.0*vec3f(r)+vec3f(8.0),1.0e30)");
  if (/nbSourcePhi\(/.test(core)) throw new Error("brickSourceWGSL: a sources' test still reads nbSourcePhi");
  return core + brickDetailWGSL(L) + /* wgsl */`
// φ (cells) at p with this step's sources, relaxed toward the parent's in the coupled shell.
fn nbSourcePhi(p:vec3f,value:f32)->f32{return nbDetailPhi(p,nbSourcePhiCore(p,value));}
`;
}
