import {uniformPageDomainWGSL,type UniformPageDomain} from "./uniform-page-domain";
import { uniformVolumePagesWGSL, type UniformVolumePageShaderOptions } from "./uniform-volume-pages.wgsl";
import { uniformVolumeWGSL } from "./uniform-volume.wgsl";
import { sceneShapeWgsl } from "../../core/scene-shape";
import { inflowBoundaryWGSL } from "../../core/inflow-boundary";
import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

const uniformMacCormackAuditEnabled = typeof process !== "undefined"
  && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1";

/**
 * Dense uniform-grid reference kernels.
 *
 * This module is deliberately independent of both adaptive coarse backends.
 * It provides a matched-lattice GPU baseline for transport and projection
 * comparisons without octree topology, sparse residency, or backend cutovers.
 */
export function createUniformReferenceComputeShader(geometric = false, referenceDimension: 2 | 3 = 3, pages?: UniformVolumePageShaderOptions, domain?:UniformPageDomain): string { return /* wgsl */ `
// The dimensional oracle suppresses the absent derivative; symmetry walls alone
// do not prevent roundoff from creating a transverse level-set gradient.
const UNIFORM_REFERENCE_DIMENSION: u32 = ${referenceDimension}u;
const MACCORMACK_AUDIT_ENABLED: bool = ${uniformMacCormackAuditEnabled};
${createCm12NumericsWGSL()}
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
  tuning: vec4f,
  // A ball of liquid dropped into the running solve: centre in metres, radius
  // in w. Zero radius is the resting state, which is every step but the one
  // immediately after the user let go of a drop.
  drop: vec4f,
  // x: the half-depth a dropped disk spans along z, zero for a ball. A 2D
  // case's liquid is extruded through the slab, so its drops are too.
  dropExtent: vec4f,
  // x: the shell reach in 4h tiles, i.e. how far past the fine set the velocity
  // extension must still be exact. y: 1 when the extension runs on those tiles
  // rather than densely. Both are inert unless physical.z is non-negative.
  twoLevel: vec4f,
  // Geometric only: the phi/V agreement stages, all inert at zero.
  // x: inward compaction of V. y: seed phi from V where phi has no surface.
  // z: gain of the slow normal shift (0 is off). w: its clamp in cells a step.
  agreement: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var heightIn: texture_2d<f32>;
@group(0) @binding(8) var heightOut: texture_storage_2d<rg32float, write>;
// 0..4 are published diagnostics. The geometric method adds 5 and 6 for the
// volume dust floor (cells zeroed, discarded mass in sixty-fourths of the
// threshold) and 7 for the count of fine tiles in the E1 two-level map.
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,8>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
  inverseMassInertia: vec4f,
  angularMomentumRestitution: vec4f,
  material: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(12) var predictedVelocityIn: texture_3d<f32>;
@group(0) @binding(13) var reversedVelocityIn: texture_3d<f32>;
// Precomputed transport velocity with a one-texel zero shell so hardware
// trilinear sampling reproduces the zero wall-face boundary condition.
@group(0) @binding(14) var transportIn: texture_3d<f32>;
@group(0) @binding(15) var transportSampler: sampler;
// Previous-step rho'=rho/V. Liquid momentum advection uses this phase mask
// with velocityIn and never reads momentum from the extrapolated transport.
@group(0) @binding(16) var velocityPhaseIn: texture_3d<f32>;
@group(0) @binding(19) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;
// The adaptive method binds its resident signed-distance field here. Uniform
// reference solvers bind volumeIn instead, preserving their VOF formulation.
@group(0) @binding(20) var surfaceIn: texture_3d<f32>;
// Per-column terrain heights in cell units; params.container.w enables it so
// terrain-free scenes never pay the extra load. Static for the whole run.
@group(0) @binding(21) var terrainIn: texture_2d<f32>;
@group(0) @binding(24) var gammaIn: texture_3d<f32>;
@group(0) @binding(25) var gammaOut: texture_storage_3d<r32float, write>;
// The physical rgba velocity texture owns positive MAC faces. The separate
// field owns the three negative domain faces so the CM11a pressure halo has
// persistent velocity DOFs on both sides of every closed wall.
@group(0) @binding(26) var<storage,read> boundaryVelocityIn:array<f32>;
@group(0) @binding(27) var<storage,read_write> boundaryVelocityOut:array<f32>;
// Eight vec4s per (cell, component), populated only in the opt-in Dawn stage
// audit shader variant. Production compiles the constant-false branch away.
@group(0) @binding(28) var<storage,read_write> macCormackAudit:array<vec4f>;
// GPU-resident wet bounds followed by indirect dispatch records. Words 7..9
// are the main-grid origin; the pressure hierarchy consumes the level records
// beginning at word 16.
@group(0) @binding(29) var<storage,read> activeRegion:array<u32>;
@group(0) @binding(30) var<storage,read_write> activeScratch:array<u32>;
// Header words 0..255 mirror activeRegion; eight words per workgroup follow.
// Words 176..178 are the (n+1)-lattice vertex dispatch Uniform Geometric's phi
// passes use; 179 pads the record to a four-word stride. Words 180..236 are the
// group counts the HOST chose for this step's direct dispatches, which the
// finalize below checks its own exact counts against. Words 237/238 carry this
// step's packed per-axis travel from the reduce to the finalize, 239 publishes
// the two-sided total the host pads its lagged counts with, and 240..242 are
// the pressure-window origin the host writes for the CM11a lattice.
const ACTIVE_SUMMARY_BASE:u32=256u;
const ACTIVE_VERTEX_DISPATCH_WORD:u32=176u;
const ACTIVE_CPU_LEVEL_BASE_WORD:u32=180u;
const ACTIVE_CPU_MAIN_WORD:u32=228u;
const ACTIVE_CPU_VERTEX_WORD:u32=231u;
const ACTIVE_CPU_MODE_WORD:u32=234u;
const ACTIVE_VIOLATION_WORD:u32=235u;
const ACTIVE_VIOLATION_AXES_WORD:u32=236u;
const ACTIVE_TRAVEL_PLUS_WORD:u32=237u;
const ACTIVE_TRAVEL_MINUS_WORD:u32=238u;
const ACTIVE_TRAVEL_TOTAL_WORD:u32=239u;
const ACTIVE_PRESSURE_ORIGIN_WORD:u32=240u;
const ACTIVE_PRESSURE_CAPACITY_WORD:u32=243u;
const ACTIVE_PRESSURE_MODE_WORD:u32=246u;
const ACTIVE_SCAN_GROUPS_WORD:u32=247u;
fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
// Three per-axis counts to a word, ten bits each. Travel saturates, which is
// harmless: a step that moves a thousand cells is already outside the regime.
fn activePackTravel(value:vec3u)->u32{
  let c=min(value,vec3u(1023u));
  return (c.x|(c.y<<10u))|(c.z<<20u);
}
fn activeUnpackTravel(packed:u32)->vec3u{
  return vec3u(packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u);
}
// Field-wise max. A plain max() on the packed word would let the high field
// zero the low ones, which is not the conservative direction.
fn activeMaxPacked(a:u32,b:u32)->u32{
  return (max(a&1023u,b&1023u)|(max((a>>10u)&1023u,(b>>10u)&1023u)<<10u))
    |(max((a>>20u)&1023u,(b>>20u)&1023u)<<20u);
}
// Packed extents, bit 30 a usable flag: a count that does not fit clears the
// flag and the reader falls back to no clip at all, always the safe direction.
fn activePackExtent(value:vec3u)->u32{
  if(any(value>=vec3u(1024u))){return 0u;}
  return ((value.x|(value.y<<10u))|(value.z<<20u))|0x40000000u;
}
fn activeUnpackExtent(packed:u32)->vec3u{
  if((packed&0x40000000u)==0u){return vec3u(0xffffffffu);}
  return vec3u(packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u);
}
fn activeWindowOrigin()->vec3u{return vec3u(activeRegion[7],activeRegion[8],activeRegion[9]);}
fn activeWindowExtent()->vec3u{
  return vec3u(activeRegion[10],activeRegion[11],activeRegion[12])-activeWindowOrigin();
}
// The host sizes this dispatch from a box a couple of steps old, so its last
// workgroups overrun the exact window. Those threads exit here, at
// activeRegion[10..12] minus the origin: the union box the finalize published,
// which is also what the DISPATCH clips to when the box outgrows the counts.
// The overrun band is then genuinely free and the host's lag allowance is a
// launch-size allowance only, with no bearing on the answer.
//
// That was not true until the vertex lattice below carried its own read reach.
// While the phi passes wrote only the box, a vertex on its face redistanced
// from phi the previous step had left outside it, the overrun band was
// refreshing it every step, and exiting here moved the tall-air far-wall
// impact peak from 3.11 m/s to 4.39. With the reach written, clip-all and
// no-clip agree on peak speed, front position and volume at 1x and 8x
// (docs/research/uniform-geometric-tall-air-2026-09-19/pressure-window-report.md).
fn activeId(gid:vec3u)->vec3i{
  ${domain ? "return pageDomainCell(gid);" : `
  if(any(gid>=activeWindowExtent())){return vec3i(-1);}
  return vec3i(activeWindowOrigin()+gid);
  `}
}
// The (n+1)^3 vertex lattice the geometric phi passes run on. It is one wider
// than the cell box on every axis, and it is DILATED by the phi stage's own
// internal read reach on both sides, which is the one thing the window's
// padding cannot express: the padding covers what a CELL reads, and the phi
// passes read each other.
//
//   uvAdvectPhi writes the advected phi; uvRedistancePhi then reads it through
//   uvPhi at the closest point, q clamped to p +- 4, with uvGradient probing
//   +- 0.25 and the trilinear tap one vertex past that ................... 6
//   uvAgreementShift's tent gather, when the shift runs: base + [-4,4)
//   plus a tap ............................................................ 5
//
// Write only the box and a vertex on its face redistances from phi four
// vertices outside it, which the previous step's ping-pong partner still
// holds. That is what the host's lag allowance was covering: measured on the
// tall-air dam, stopping at the box moves the far-wall impact peak to
// 4.392 m/s against the whole-domain control's 3.106, and a margin of four
// vertices removes the whole difference. With the reach written, every phi a
// phi pass reads was written this step and the lattice is sound on its own
// box, which is what lets every kernel's overrun threads exit again.
//
// With the window off this folds to the plain [0,dims] domain test.
const VERTEX_PHI_REACH:u32=6u;
fn activeVertexLow()->vec3u{
  let origin=activeWindowOrigin();
  return origin-min(origin,vec3u(VERTEX_PHI_REACH));
}
fn activeVertexId(gid:vec3u)->vec3i{
  ${domain ? "return pageDomainVertex(gid);" : `
  let low=activeVertexLow();
  let high=min(vec3u(dims()),
    vec3u(activeRegion[10],activeRegion[11],activeRegion[12])+vec3u(VERTEX_PHI_REACH));
  if(any(gid>high-low)){return vec3i(-1);}
  return vec3i(low+gid);
  `}
}
// The CM11a pressure hierarchy can be planned on the WINDOW rather than the
// domain. When it is, the host publishes the lattice origin and capacity in
// simulation cells and the whole hierarchy is window-local: a pressure cell
// id maps to simulation id - 1 + origin, and every pressure dispatch covers
// the whole capacity lattice with a static plan, so the level records are not
// consulted at all.
fn pressureWindowLattice()->bool{return activeRegion[ACTIVE_PRESSURE_MODE_WORD]==1u;}
fn pressureWindowOrigin()->vec3i{
  if(!pressureWindowLattice()){return vec3i(0);}
  return vec3i(vec3u(activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD],
    activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD+1u],activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD+2u]));
}
${geometric ? `
// Uniform Geometric's solve-window predicate. A cell is a seed when it holds
// liquid above the dust floor, or when any of its eight vertices is on the
// liquid side of the 4h band that the redistance, uvTarget and the two-level
// classify all treat as near-surface. Solids and terrain are deliberately NOT
// seeds: a solid far from the liquid needs no fluid work of any kind, and
// seeding on it would make the window the whole domain in any scene with a
// container floor. Sources are added by the external-source scan, which is the
// only pass that can see liquid arriving outside the previous window.
fn geometricActiveSeed(id:vec3i)->bool{
  if(!valid(id)){return false;}
  let dust=select(params.tuning.z,1e-6,params.tuning.z<=0.0);
  if(abs(volume(id))>dust){return true;}
  let band=4.0*max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  for(var k=0u;k<8u;k++){
    if(textureLoad(uvPhiIn,id+uvCorner(k),0).x<band){return true;}
  }
  return false;
}
` : ""}
fn inflowGridDims()->vec3i{return dims();}
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
fn solidVoxelWord(index:u32)->u32{
  return activeScratch[u32(round(params.dropExtent.z))+index];
}
fn staticSolidVoxelOccupied(p:vec3i)->bool{
  if(solidVoxelWord(0u)!=0x53565731u){return false;}
  let shape=vec3i(i32(solidVoxelWord(1u)),i32(solidVoxelWord(2u)),
    i32(solidVoxelWord(3u)));
  let q=p+vec3i(1);
  if(any(q<vec3i(0))||any(q>=shape)){return false;}
  let index=u32(q.x+shape.x*(q.y+shape.y*q.z));
  return (solidVoxelWord(4u+(index>>5u))&(1u<<(index&31u)))!=0u;
}
// Canonical reductions for the horizontal D4 group. Reflections exchange
// operands inside opposite-direction pairs; x/z exchange operands of the
// horizontal pair sum. The papers prescribe the stencil, not its add order.
fn d4Sum6(value:array<f32,6>)->f32{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}
fn d4Sum8(value:array<f32,8>)->f32{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn d4Sum8Vec2(value:array<vec2f,8>)->vec2f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn d4Sum6Vec3(value:array<vec3f,6>)->vec3f{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}
fn worldCell(id:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);}
fn hasTerrain()->bool{return params.container.w>0.5;}
fn depthSymmetry()->bool{return params.tuning.w>0.5;}
fn terrainHeightCells(x:i32,z:i32)->f32{let d=dims();return textureLoad(terrainIn,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
// Ground handling mirrors the rigid-body solid treatment with zero velocity:
// the heightfield closes faces, drops pressure unknowns, and blocks deposits.
fn cellInsideTerrain(p:vec3i)->bool{if(!hasTerrain()){return false;}return f32(p.y)+0.5<terrainHeightCells(p.x,p.z);}
fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}
${inflowBoundaryWGSL}
/**
 * Share of a cell covered by a ball dropped into the running solve.
 *
 * Eight sub-cell samples, which is the same estimator the host uses to seed a
 * ball at t = 0 — so a ball dropped at t > 0 wets exactly the cells it would
 * have wetted had it been authored in the document from the start. Anything
 * cheaper (a centre-in-sphere test) would make a small ball's mass depend on
 * where it happened to land relative to the lattice.
 */
fn dropSource(q:vec3i)->f32{
  let radius=params.drop.w;if(radius<=0.0){return 0.0;}
  let h=params.cellGravity.xyz;
  let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);
  var covered=0.0;
  for(var sample=0u;sample<8u;sample+=1u){
    let offset=vec3f(f32(sample&1u),f32((sample>>1u)&1u),f32((sample>>2u)&1u))*0.5+vec3f(0.25);
    let point=minimum+(vec3f(q)+offset)*h;
    let d=point-params.drop.xyz;
    let inside=select(length(d)<=radius,
      length(d.xy)<=radius&&abs(d.z)<=params.dropExtent.x,
      params.dropExtent.x>0.0);
    if(inside){covered+=0.125;}
  }
  return covered;
}
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
fn levelSetAuthority() -> bool { return ${geometric ? "true" : "params.physical.w > 0.5"}; }
fn surfaceValue(p: vec3i) -> f32 {
  if (!valid(p)) { return select(0.0, 5.0 * min(params.cellGravity.x, min(params.cellGravity.y, params.cellGravity.z)), levelSetAuthority()); }
  return ${geometric ? "uvPhi(vec3f(p)+vec3f(0.5))" : "textureLoad(surfaceIn, p, 0).x"};
}
fn surfaceOccupancy(p: vec3i) -> f32 {
  if (!valid(p)) { return 0.0; }
  let value = surfaceValue(p);
  return select(clamp(value, 0.0, 1.0), clamp(0.5 - value / (4.0 * params.cellGravity.y), 0.0, 1.0), levelSetAuthority());
}
fn surfaceLiquid(p: vec3i) -> bool { return valid(p) && select(surfaceValue(p) >= 0.5, surfaceValue(p) < 0.0, levelSetAuthority()); }
fn velocity(p: vec3i) -> vec3f { return textureLoad(velocityIn,clampCell(p),0).xyz; }
fn faceVelocity(p:vec3i)->vec3f{if(!valid(p)){return vec3f(0.0);}return textureLoad(velocityIn,p,0).xyz;}
fn boundaryFaceIndex(p:vec3i,axis:u32)->u32{
  let d=dims();
  if(axis==0u){return u32(p.y+d.y*p.z);}
  let yOffset=d.y*d.z;
  if(axis==1u){return u32(yOffset+p.x+d.x*p.z);}
  return u32(yOffset+d.x*d.z+p.x+d.x*p.y);
}
fn boundaryVelocity(p:vec3i)->vec3f{
  if(!valid(p)){return vec3f(0.0);}var value=vec3f(0.0);
  if(p.x==0){value.x=boundaryVelocityIn[boundaryFaceIndex(p,0u)];}
  if(p.y==0){value.y=boundaryVelocityIn[boundaryFaceIndex(p,1u)];}
  if(p.z==0){value.z=boundaryVelocityIn[boundaryFaceIndex(p,2u)];}
  return value;
}
fn storeBoundaryVelocity(id:vec3i,value:vec3f){
  if(id.x==0){boundaryVelocityOut[boundaryFaceIndex(id,0u)]=value.x;}
  if(id.y==0){boundaryVelocityOut[boundaryFaceIndex(id,1u)]=value.y;}
  if(id.z==0){boundaryVelocityOut[boundaryFaceIndex(id,2u)]=value.z;}
}
fn carryBoundaryVelocity(id:vec3i){storeBoundaryVelocity(id,boundaryVelocity(id));}
fn liquid(p:vec3i)->bool{return surfaceLiquid(p);}
fn pressureValue(p:vec3i)->f32{
  return textureLoad(pressureIn,clampCell(p),0).x;
}
// Projection alone binds the CM11a finest level, whose physical cells are
// enclosed by a one-cell solid/domain halo. Keep this addressing explicit:
// other main-shader pressure scratch remains an unpadded simulation texture.
fn projectPressureValue(p:vec3i)->f32{
  let pressureDims=vec3i(textureDimensions(pressureIn));
  let local=p+vec3i(1)-pressureWindowOrigin();
  // A window lattice ends before the domain does. A lookup that leaves it on
  // a side still INSIDE the domain has left the solved region into far air,
  // whose pressure is zero; one that leaves the domain keeps the clamp, which
  // is the wall/lid ghost the projection has always read. The second case can
  // only arise where the window is snapped to that wall, so the clamped cell
  // is still the domain halo it was without a window.
  if(valid(p)&&(any(local<vec3i(0))||any(local>=pressureDims))){return 0.0;}
  return textureLoad(pressureIn,clamp(local,vec3i(0),pressureDims-vec3i(1)),0).x;
}
fn cellOpenFraction(p:vec3i)->f32{
  if(!valid(p)||staticSolidVoxelOccupied(p)){return 0.0;}
  return clamp((1.0-cellSolidFraction(p))*(1.0-cellTerrainFraction(p)),0.0,1.0);
}
// Chentanez--Mueller Sec. 3.7, Eq. 20.  Surface density represents mass in
// the non-solid part of a cut cell, so pressure classification and the ghost
// fluid distance must use rho'=rho/V rather than raw rho.
fn pressureDensityOpen(p:vec3i)->f32{
  let open=cellOpenFraction(p);
  if(open<=1e-5){return 0.0;}
  return volume(p)/open;
}
fn pressureDensity(p:vec3i)->f32{
  if(!valid(p)){return 0.0;}
  let open=cellOpenFraction(p);
  if(open>1e-5){return pressureDensityOpen(p);}
  // Eq. 20 is extrapolated from V>0 into adjacent V=0 cells.  Although fully
  // solid cells are not pressure unknowns, this continuation keeps the free
  // surface distance well-defined at a cut boundary.
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  // A phase-diluting average can turn an adjacent solid continuation back
  // into air when any of the other neighbors are dry. The max continuation
  // guarantees that every V=0 cell adjacent to liquid is retained as the
  // pressure unknown required by Sec. 3.7.
  var continued=0.0;
  for(var index=0;index<6;index+=1){let q=p+offsets[index];if(cellOpenFraction(q)>1e-5){continued=max(continued,pressureDensityOpen(q));}}
  return continued;
}
fn geometricVolumeEnabled()->bool{return ${geometric ? "true" : "false"};}
fn pressureSurfacePhi(p:vec3i)->f32{
  ${geometric ? `
  // Vertex phi is transported non-conservatively and nothing returns it to V,
  // so a film thinner than half a cell has a positive centre and, on phi
  // alone, no pressure row: its faces are zeroed by the projection, V rides
  // the extrapolated field into the far wall and stacks there, and Sec. 3.7's
  // excess divergence -- the only V-to-dynamics coupling -- sits inside the
  // row it was denied. params.physical.w lets V claim the row: a cell holding
  // at least half its open capacity is liquid, at the ghost distance that
  // fill implies. Every consumer of the pressure interface (rows, RHS, ghost
  // fractions, the multigrid topology and the extension's source test) reads
  // this one function, so they cannot disagree about which cells are liquid.
  //
  // w=1 grants that row only to a cell the projection abandons: one with no
  // phi-liquid face neighbour, whose faces are all air-air. V is conservative,
  // not geometric -- beside a phi surface it sits in a patchy one-cell layer
  // (V near 0.9 where phi's fill reads 0.15), so letting it place the free
  // surface there makes random columns a full cell taller than their
  // neighbours: rho*g*h across one face, every step, which is the bubbling.
  // A cell beside phi-liquid already has a projected face, so phi keeps it.
  // w=2 is that unrestricted rule, min(phi, V distance) everywhere.
  let q=clampCell(p);let phi=uvPhi(vec3f(q)+vec3f(0.5));
  let open=cellOpenFraction(q);
  if(params.physical.w<0.5||open<=1e-5){return phi;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let volumePhi=h*(0.5-volume(q)/open);
  if(params.physical.w>1.5){return min(phi,volumePhi);}
  if(phi<0.0||volumePhi>=0.0){return phi;}
  for(var axis=0;axis<3;axis+=1){for(var side=-1;side<=1;side+=2){
    var n=q;n[axis]+=side;if(valid(n)&&uvPhi(vec3f(n)+vec3f(0.5))<0.0){return phi;}}}
  return max(volumePhi,-0.5*h);` : `
  let dx=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  return -(pressureDensity(p)-0.5)*dx;`}
}
// CM11a includes one layer of solid pressure unknowns. Continue the liquid
// interface from open neighbours, never from stale phi trapped inside solid.
fn pressurePhi(p:vec3i)->f32{
  if(!geometricVolumeEnabled()||cellOpenFraction(p)>1e-5){return pressureSurfacePhi(p);}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(params.boundary.w>0.5&&p.y==dims().y&&p.x>=0&&p.x<dims().x&&p.z>=0&&p.z<dims().z){return 0.5*h;}
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;var weights:array<f32,6>;
  for(var i=0u;i<6u;i++){let q=p+offsets[i];let open=cellOpenFraction(q);
    terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
    let phi=pressureSurfacePhi(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
  let weight=d4Sum6(weights);
  return select(0.5*h,d4Sum6(terms)/max(weight,1e-9),weight>0.0);
}
// Sec. 3.7 explicitly extrapolates rho' into adjacent V=0 cells so those
// cells participate in the pressure system. Do not filter them back out by V.
fn pressureLiquid(p:vec3i)->bool{return valid(p)&&${geometric ? "pressurePhi(p)<0.0" : "pressureDensity(p)>0.5"};}
fn ghostFluidFraction(liquidCell:vec3i,airCell:vec3i)->f32{
  let liquidPhi=pressurePhi(liquidCell);let airPhi=pressurePhi(airCell);
  return cm12GhostFluidTheta(liquidPhi,airPhi,${geometric ? "1e-9" : "1e-6"});
}
fn sampledFaceVelocity(p:vec3i,component:u32)->f32{
  let d=dims();if(p[component]<0||p[component]>=d[component]){return 0.0;}
  return textureLoad(transportIn,clampCell(p)+vec3i(1),0)[component];
}
fn transportCoordinate(q:vec3f)->vec3f{return (q+vec3f(1.5))/vec3f(dims()+vec3i(2));}
fn sampleVolume(p:vec3f)->f32{
  let q=p-vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;
    let weight=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    terms[corner]=select(0.0,weight*volume(donor),valid(donor)&&!cellInsideSolid(donor));
  }
  return d4Sum8(terms);
}
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  // Experiment E1's single choke point. The decision is per sample point, not
  // per thread, so a characteristic leaving the fine band crosses the level
  // interface part way exactly as a shrunk lattice would make it.
  ${geometric ? "if(params.physical.z>=0.0&&!uvTwoLevelFineAt(p)){return uvCoarseVelocityComponent(p,component);}" : ""}
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;let q=clamp(p-offset,lower,vec3f(dims()-vec3i(1)));
  let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(o)>vec3f(0.5));
    terms[corner]=weights.x*weights.y*weights.z*textureLoad(transportIn,base+o+vec3i(1),0)[component];
  }
  return d4Sum8(terms);
}
fn sampleVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
fn velocityPhaseWeight(p:vec3i)->f32{
  if(!valid(p)){return 0.0;}
  // velocityPhaseIn stores prior rho'=rho/V. Thin numerical density halos are
  // not full-strength velocity donors. This is support selection only: neither
  // density nor wall aperture scales the intensive transported velocity.
  let liquidFraction=clamp(textureLoad(velocityPhaseIn,p,0).x,0.0,1.0);
  return select(0.0,1.0,liquidFraction>0.5&&cellOpenFraction(p)>1e-5);
}
fn physicalVelocityFaceWeight(p:vec3i,component:u32)->f32{
  var axis=vec3i(0);axis[component]=1;let neighbor=p+axis;
  // MAC storage keeps positive faces in velocityIn and the three negative
  // domain faces in boundaryVelocityIn. Both are authoritative boundary
  // conditions, not extrapolated air. A stencil location is addressable when
  // at least one of its adjacent cells is in the domain; invalid transverse
  // coordinates leave both cells invalid and are rejected.
  if(!valid(p)&&!valid(neighbor)){return 0.0;}
  let liquidWeight=max(velocityPhaseWeight(p),velocityPhaseWeight(p+axis));
  return liquidWeight;
}
fn physicalVelocityFaceValue(p:vec3i,component:u32)->f32{
  if(valid(p)){return textureLoad(velocityIn,p,0)[component];}
  var axis=vec3i(0);axis[component]=1;let neighbor=p+axis;
  if(valid(neighbor)&&p[component]==-1){return boundaryVelocity(neighbor)[component];}
  return 0.0;
}
// Interpolate the authoritative pre-advection MAC field using only faces
// adjacent to prior liquid. Exterior transport values are not candidates and
// therefore cannot contribute momentum, regardless of how they were filled.
fn samplePhysicalVelocityComponent(p:vec3f,component:u32)->vec2f{
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;
  let q=clamp(p-offset,lower,vec3f(dims()-vec3i(1)));let base=vec3i(floor(q));let fraction=fract(q);
  var terms:array<vec2f,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(o)>vec3f(0.5));let weight=weights.x*weights.y*weights.z;
    let donor=base+o;let combinedWeight=weight*physicalVelocityFaceWeight(donor,component);
    terms[corner]=vec2f(combinedWeight*physicalVelocityFaceValue(donor,component),combinedWeight);
  }
  let sum=d4Sum8Vec2(terms);
  return vec2f(select(0.0,sum.x/sum.y,sum.y>0.0),sum.y);
}
fn clampVelocityTraceToDomain(p:vec3f)->vec3f{
  let upper=vec3f(dims());var q=p;
  q.x=clamp(q.x,0.0,upper.x);q.z=clamp(q.z,0.0,upper.z);q.y=max(q.y,0.0);
  if(params.boundary.w<=0.5){q.y=min(q.y,upper.y);}
  return q;
}
// Reconstruct every RK2 stage at the actual MAC component offsets. Long
// characteristics remain semi-Lagrangian, but are integrated in local pieces
// no longer than the accurate extension band. The extension defines only the
// characteristic map; samplePhysicalVelocityComponent owns transported
// momentum below.
fn departurePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{
  var point=position;var remaining=abs(dt);let direction=select(-1.0,1.0,dt>=0.0);
  for(var step=0;step<32;step+=1){
    if(remaining<=1e-7){break;}
    let first=sampleVelocity(point);let rate=max(abs(first.x)/h.x,max(abs(first.y)/h.y,abs(first.z)/h.z));
    let stepSeconds=min(remaining,1.5/max(rate,1e-6));let signedStep=direction*stepSeconds;
    let midpoint=clampVelocityTraceToDomain(point-0.5*first*signedStep/h);
    point=clampVelocityTraceToDomain(point-sampleVelocity(midpoint)*signedStep/h);remaining-=stepSeconds;
  }
  return point;
}
// advectVelocityComponent follows rigidBodyIndexAt below: it clips the
// characteristic against the bodies, and WGSL requires declaration before use.
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
${sceneShapeWgsl()}
// Body space: every shape function above is written as if the body sat at the
// origin unrotated, so these two are the whole of what this shader knows about
// what a shape *is*. Adding one is an entry in SCENE_SHAPE_TABLE and nothing
// here.
fn rigidLocalPoint(body:RigidBody,world:vec3f)->vec3f{
  return quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);
}
fn rigidShapeTag(body:RigidBody)->i32{return i32(round(body.positionShape.w));}
fn insideRigid(body:RigidBody,world:vec3f)->bool{
  return rigidShapeInside(rigidShapeTag(body),body.dimensions.xyz,rigidLocalPoint(body,world));
}
fn rigidSignedDistance(body:RigidBody,world:vec3f)->f32{
  return rigidShapeDistance(rigidShapeTag(body),body.dimensions.xyz,rigidLocalPoint(body,world));
}
fn rigidBodyIndexAt(world:vec3f)->i32{
  let bodyCount=u32(round(params.boundary.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return i32(bodyIndex);}}
  return -1;
}
fn rigidVelocityAt(bodyIndex:i32,world:vec3f)->vec3f{
  let body=rigidBodies[u32(bodyIndex)];
  return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);
}
// Trace-space to world.  Advection positions are cell coordinates carrying a
// per-component MAC offset (cell+(1,.5,.5) is the +x face of cell), so the
// half-cell that worldCell adds is already present in the position.
fn traceWorld(p:vec3f)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(-0.5*params.container.x+p.x*h.x,p.y*h.y,-0.5*params.container.z+p.z*h.z);
}
// Gates the moving-solid corrections.  Each reduces to the original expression
// when no body is present, but the gate keeps body-free scenes on the literal
// original path so their trajectories stay bit-identical rather than merely
// algebraically equal.  Terrain is deliberately excluded: it is static, so it
// gains far less from these corrections than it would cost in re-blessing
// every shipped terrain scene.
fn hasRigidBodies()->bool{return params.boundary.z>=0.5;}
fn insideAnyRigid(world:vec3f)->bool{return rigidBodyIndexAt(world)>=0;}
fn staticSolidVoxelAtWorld(world:vec3f)->bool{
  let h=params.cellGravity.xyz;
  let p=vec3i(floor(vec3f((world.x+0.5*params.container.x)/h.x,
    world.y/h.y,(world.z+0.5*params.container.z)/h.z)));
  return staticSolidVoxelOccupied(p);
}
fn insideAnyTraceSolid(world:vec3f)->bool{return staticSolidVoxelAtWorld(world)||insideAnyRigid(world);}
// Sec. 3.4 stops the density characteristic at a solid boundary; the velocity
// characteristic had no such test.  The RK2 backtrace therefore read straight
// through a body, so liquid ahead of a moving obstacle sampled the liquid
// behind it and the obstacle never pushed it.  Clip the chord at the first
// crossing and sample just outside the surface, where the Sec. 3.3 extension
// has already written u_s.  A departure that is not inside a body -- the
// overwhelmingly common case -- costs exactly one primitive test.
fn clipDepartureAtSolid(position:vec3f,departure:vec3f)->vec3f{
  if(!insideAnyTraceSolid(traceWorld(departure))){return departure;}
  // A departure inside a body converges to lo=0 and samples in place, the
  // correct degenerate answer for a face the body has already swallowed.
  var lo=0.0;var hi=1.0;
  for(var step=0;step<8;step+=1){
    let mid=0.5*(lo+hi);
    if(insideAnyTraceSolid(traceWorld(mix(position,departure,mid)))){hi=mid;}else{lo=mid;}
  }
  return mix(position,departure,lo);
}
// Both branches spell the body-free case as the original expression rather
// than as a clip that happens to be the identity.  Routing it through the
// wrapper instead measured a 1.5e-4 relative shift in maxSpeed on the
// body-free hydrostatic lane -- pure float reassociation, but enough to make
// every still-scene lane need re-blessing for no physical reason.
fn clippedDeparturePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{
  return clipDepartureAtSolid(position,departurePoint(position,dt,h));
}
fn velocityFaceLiquid(position:vec3f,component:u32)->bool{
  var offset=vec3f(0.5);offset[component]=1.0;let face=vec3i(floor(position-offset+vec3f(0.5)));
  var axis=vec3i(0);axis[component]=1;
  // Sub-isovalue density is still physical liquid. Switching it back to the
  // exterior gather at 0.5 produces a temporal discontinuity at energetic
  // interfaces and separating walls.
  return volume(face)>1e-5||volume(face+axis)>1e-5;
}
fn liquidOnlyVelocityAdvection()->bool{return params.dropExtent.y>0.5;}
fn advectVelocityComponent(position:vec3f,component:u32,dt:f32,h:vec3f)->f32{
  let rawDeparture=departurePoint(position,dt,h);
  var departure=rawDeparture;
  departure=clipDepartureAtSolid(position,rawDeparture);
  // Air-side values still need the complete hierarchy for interface transport
  // and the following extrapolation. When the toggle is on, a face belonging
  // to updated liquid gathers momentum exclusively from prior liquid faces.
  if(!liquidOnlyVelocityAdvection()||!velocityFaceLiquid(position,component)){
    return sampleVelocityComponent(departure,component);
  }
  let supported=samplePhysicalVelocityComponent(departure,component);
  if(supported.y>0.0){return supported.x;}
  // Discrete high-CFL traces can miss the old liquid stencil. Search back
  // along the same characteristic for authoritative liquid support. This is
  // phase-aware interpolation of velocityIn, not sampling of the air field.
  for(var probe=1;probe<=16;probe+=1){
    let candidate=mix(departure,position,f32(probe)/16.0);
    let recovered=samplePhysicalVelocityComponent(candidate,component);
    if(recovered.y>0.0){return recovered.x;}
  }
  // No prior-liquid donor exists on this characteristic. Zero is neutral and
  // deterministic; projection may reconstruct it from neighboring liquid,
  // while exterior velocity cannot steer the liquid.
  return 0.0;
}
// Conservative bounding-sphere reject so cells away from every body (and
// body-free scenes) skip the per-cell primitive tests in the solid-aware
// pressure, projection, and coupling kernels.
fn nearAnyBody(world:vec3f)->bool{
  let bodyCount=u32(round(params.boundary.z));
  let margin=2.0*max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    let body=rigidBodies[bodyIndex];
    // One bounding radius, from the shape table. This ladder used to spell the
    // cylinder's as sqrt(x*x+0.25*y*y) while the octree lane spelled the same
    // radius as length(vec2f(x,0.5*y)); they agree to the last ulp on almost
    // every input and were free to disagree on the rest. The margin below is
    // two cells wide, so the reject was never sensitive to that difference —
    // but nothing said so, and a reader had to prove it twice.
    let radius=rigidShapeBoundingRadius(rigidShapeTag(body),body.dimensions.xyz);
    if(distance(world,body.positionShape.xyz)<=radius+margin){return true;}
  }
  return false;
}
// Paper Sec 3.9.1 treats a cell as solid in the divergence when its solid
// fraction is high; the cell-centre point-in-primitive test is our s>0.9.
fn cellRigidBody(p:vec3i)->i32{
  if(!valid(p)){return -1;}
  return rigidBodyIndexAt(worldCell(p));
}
// Sub-cell solid fraction with the CPU voxelizer's 8-corner sampling
// (solidFieldsFromBodies), so mixed cells blend rather than snap.
fn bodySolidFraction(body:RigidBody,p:vec3i)->f32{
  var inside=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(insideRigid(body,worldCell(p)+offset*params.cellGravity.xyz)){inside+=1.0;}
  }
  return inside/8.0;
}
fn cellSolidFraction(p:vec3i)->f32{
  if(staticSolidVoxelOccupied(p)){return 1.0;}
  let bodyCount=u32(round(params.boundary.z));var fraction=0.0;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}fraction=max(fraction,bodySolidFraction(rigidBodies[bodyIndex],p));}
  return fraction;
}
fn worldInsideTerrain(world:vec3f)->bool{
  if(!hasTerrain()){return false;}
  let h=params.cellGravity.xyz;
  let x=clamp(i32(floor((world.x+0.5*params.container.x)/h.x)),0,dims().x-1);
  let z=clamp(i32(floor((world.z+0.5*params.container.z)/h.z)),0,dims().z-1);
  return world.y<terrainHeightCells(x,z)*h.y;
}
fn solidVelocityAtWorld(world:vec3f)->vec4f{
  if(staticSolidVoxelAtWorld(world)){return vec4f(0.0,0.0,0.0,1.0);}
  if(worldInsideTerrain(world)){return vec4f(0.0,0.0,0.0,1.0);}
  let body=rigidBodyIndexAt(world);
  if(body>=0){return vec4f(rigidVelocityAt(body,world),1.0);}
  return vec4f(0.0);
}
// Four transverse quadrature points approximate the non-solid area V^f of a
// MAC face.  Sampling the oriented primitives in world space makes this the
// same fractional variational boundary for static, translating, rotating,
// and non-axis-aligned solids.
fn faceSolidData(id:vec3i,axis:u32)->vec4f{
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;
  var tangentA=(axis+1u)%3u;var tangentB=(axis+2u)%3u;
  var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
    var sampleWorld=world;
    sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
    sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
    let sample=solidVelocityAtWorld(sampleWorld);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }
  return vec4f(select(vec3f(0.0),solidVelocity/max(solid,1e-6),solid>0.0),solid*0.25);
}
fn faceOpenFraction(id:vec3i,axis:u32)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){return 0.0;}
  if(!valid(id)||!valid(neighbor)){
    return select(1.0,0.0,staticSolidVoxelOccupied(id)
      ||staticSolidVoxelOccupied(neighbor));
  }
  return 1.0-faceSolidData(id,axis).w;
}
fn extrapolatedRigidVelocityAtFace(world:vec3f)->vec3f{
  let bodyCount=u32(round(params.boundary.z));let radius=max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  var nearest=radius;var result=vec3f(0.0);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}let distance=abs(rigidSignedDistance(rigidBodies[bodyIndex],world));
    if(distance<=nearest){nearest=distance;result=rigidVelocityAt(i32(bodyIndex),world);}
  }
  return result;
}
// CM11a's V_{i+1/2} is not the Sec. 3.6 face aperture above. It is the
// non-solid fraction of the face-centred overlapping (dual) cell. Eight
// volume samples construct that geometric quantity for embedded solids.
// At a grid-aligned closed domain wall, half of the dual cell lies in the
// authored solid exterior, so its inferred geometric fraction is 1/2.
fn pressureFaceData(id:vec3i,axis:u32)->vec4f{
  var neighbor=id;neighbor[axis]+=1;
  ${geometric ? `if(valid(id)!=valid(neighbor)){
    if(axis==2u&&depthSymmetry()){return vec4f(0.0);}
    let ambient=axis==1u&&max(id.y,neighbor.y)==dims().y&&params.boundary.w>0.5;
    // Intersect the domain wall with interior voxel/terrain/body occupancy.
    // Half a dual cell is fluid only when its interior half is open.
    let interior=select(neighbor,id,valid(id));
    let open=cellOpenFraction(interior);
    return vec4f(0.0,0.0,0.0,select(0.5*open,0.5*(1.0+open),ambient));
  }
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){
    // The pressure weight is dual volume, not the blocked transport aperture.
    let open=0.5*(cellOpenFraction(id)+cellOpenFraction(neighbor));
    return vec4f(0.0,0.0,0.0,open);
  }` : ""}
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){
    return vec4f(0.0,0.0,0.0,0.5);
  }
  if(!valid(id)||!valid(neighbor)){
    // A published 2D case has no z pressure derivative. Its storage-depth
    // faces are symmetry planes, not CM11a separating solid boundaries.
    if(axis==2u&&depthSymmetry()&&valid(id)!=valid(neighbor)){return vec4f(0.0);}
    if(valid(id)==valid(neighbor)){return vec4f(0.0);}

    return vec4f(0.0,0.0,0.0,select(1.0,0.5,
      staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)));
  }
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
    let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
    let sample=solidVelocityAtWorld(sampleWorld);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }
  // CM11a explicitly requires us on nearby liquid faces. For analytic rigid
  // bodies, the nearest body within one cell supplies its rigid velocity at
  // the face centre when the dual-cell samples themselves contain no solid.
  // Terrain and tank walls are static and therefore extrapolate zero.
  let extrapolated=extrapolatedRigidVelocityAtFace(world);
  return vec4f(select(extrapolated,solidVelocity/max(solid,1e-6),solid>0.0),1.0-solid/8.0);
}
fn pressureFaceVolumeFraction(id:vec3i,axis:u32)->f32{return pressureFaceData(id,axis).w;}
// Secs. 3.3 and 3.7 share one interface authority. The extrapolator consumes
// rho'=rho/V (including Eq. 20's adjacent-solid continuation) and the exact
// positive-MAC face fractions used by projection; it never reclassifies raw
// surface density or approximates a second solid boundary.
fn storeExtrapolationAuthority(id:vec3i){if(!valid(id)){return;}
  textureStore(volumeOut,id,vec4f(${geometric ? "0.5-pressurePhi(id)/min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z))" : "pressureDensity(id)"}));
  textureStore(velocityOut,id,vec4f(
    ${geometric ? "pressureFaceVolumeFraction(id,0u),pressureFaceVolumeFraction(id,1u),pressureFaceVolumeFraction(id,2u)" : "faceOpenFraction(id,0u),faceOpenFraction(id,1u),faceOpenFraction(id,2u)"},0.0));
}
@compute @workgroup_size(4,4,4)
fn buildExtrapolationAuthority(@builtin(global_invocation_id) gid:vec3u){storeExtrapolationAuthority(activeId(gid));}
@compute @workgroup_size(4,4,4)
fn buildDenseExtrapolationAuthority(@builtin(global_invocation_id) gid:vec3u){storeExtrapolationAuthority(vec3i(gid));}
// Projection enforces body velocity on interior faces, so that value cannot be
// used as the undisturbed fluid velocity for form drag. Sample six wet, open
// points just beyond the body's bounding sphere instead.
fn ambientFluidVelocity(body:RigidBody,p:vec3i,fallback:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let radius=max(body.dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
  let offsets=array<vec3i,6>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,-reach.y,0),vec3i(0,reach.y,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
  var terms:array<vec3f,6>;var weights:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=p+offsets[n];terms[n]=vec3f(0.0);weights[n]=0.0;if(!valid(q)||staticSolidVoxelOccupied(q)||cellRigidBody(q)>=0||cellInsideTerrain(q)){continue;}let wet=surfaceOccupancy(q);terms[n]=wet*velocity(q);weights[n]=wet;}
  let total=d4Sum6Vec3(terms);let weight=d4Sum6(weights);
  return select(fallback,total/max(weight,1e-6),weight>0.0);
}
fn columnHeight(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return 0.0;}return textureLoad(heightIn,vec2i(x,z),0).x;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn normalSurfaceOccupancy(id:vec3i)->f32{
  if(valid(id)){return surfaceOccupancy(id);}
  return select(0.0,surfaceOccupancy(clampCell(id)),staticSolidVoxelOccupied(id));
}
fn surfaceGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalSurfaceOccupancy(id+vec3i(1,0,0))-normalSurfaceOccupancy(id-vec3i(1,0,0)),normalSurfaceOccupancy(id+vec3i(0,1,0))-normalSurfaceOccupancy(id-vec3i(0,1,0)),normalSurfaceOccupancy(id+vec3i(0,0,1))-normalSurfaceOccupancy(id-vec3i(0,0,1)))/(2.0*h);
}
fn interfaceNormal(id:vec3i)->vec3f{
  let gradient=surfaceGradient(id);
  return gradient/max(length(gradient),1e-6);
}
// The diagnostic/emergency VOF still sharpens along its own density gradient;
// this field does not classify the adaptive pressure or velocity solve.
fn normalVolume(id:vec3i)->f32{
  // Sec. 3.6 guarantees rho=0 inside solid. Closed-domain exterior, embedded
  // solids, and authored open-top air therefore share the same zero-density
  // extension for sharpening gradients; none copies a boundary-cell value.
  if(!valid(id)||cellInsideSolid(id)){return 0.0;}
  return volume(id);
}
fn volumeGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalVolume(id+vec3i(1,0,0))-normalVolume(id-vec3i(1,0,0)),normalVolume(id+vec3i(0,1,0))-normalVolume(id-vec3i(0,1,0)),normalVolume(id+vec3i(0,0,1))-normalVolume(id-vec3i(0,0,1)))/(2.0*h);
}
// --- Conservative surface-density transport (paper Sec. 3.4, modified
// three-scatter scheme). beta, rho deficits, and gamma deficits occupy three
// consecutive fixed-point arrays in conditioningScratch/sharpenDeposits.
fn linearIndex(id:vec3i)->u32{let d=dims();return u32(id.x+d.x*(id.y+d.y*id.z));}
fn cellCount()->u32{let d=dims();return u32(d.x*d.y*d.z);}
fn betaValue(id:vec3i)->f32{
  if(!valid(id)){return 1.0;}
  return f32(atomicLoad(&sharpenDeposits[linearIndex(id)]))/CM12_TRANSPORT_FIXED;
}
// Characteristics at contacting closed-wall faces satisfy u.n=0, so a true
// trace stays in the domain and slides tangentially along the boundary. (A
// CM11a-released face is handled separately below.) A single straight RK2
// step over the paper time step spans
// u*dt/h cells (10+ at the 64-cubed dam front), punches through the wall,
// and any fold-back (mirror or clamp) then maps distinct departure cells
// onto the same near-wall band: the advection operator turns locally
// compressive at exactly the stagnation cells where mass conservation
// deposits everything (a three-wall corner folds 2^3 = 8x -- the measured
// rho pile). Sub-stepping the integration, clamping each sub-step to the
// domain so the wall's zero normal velocity is re-sampled, follows the
// sliding characteristic instead. The paper sub-steps its sharpening trace
// for the same reason (Sec 3.5: "multiple forward Euler sub-steps",
// stopping at solids).
fn clampTraceToDomain(p:vec3f)->vec3f{
  let d=vec3f(dims());var q=p;
  q.x=clamp(q.x,0.5,d.x-0.5);
  q.z=clamp(q.z,0.5,d.z-0.5);
  q.y=max(q.y,0.5);
  // Only the authored open +Y boundary has an exterior-air continuation.
  if(params.boundary.w<=0.5){q.y=min(q.y,d.y-0.5);}
  return q;
}
// CM11a can release a closed solid face: at a released face the forward
// velocity points into the domain, so a backward characteristic legitimately
// enters the solid. Clamping that departure back onto the last cell centre
// makes the conservative sampler repeatedly read the wall film itself.
fn backwardTraceExitsReleasedFace(p:vec3f)->bool{
  let d=dims();let q=clamp(vec3i(floor(p)),vec3i(0),d-vec3i(1));
  if(p.x<0.5){return boundaryVelocity(vec3i(0,q.y,q.z)).x>1e-6;}
  if(p.x>f32(d.x)-0.5){return faceVelocity(vec3i(d.x-1,q.y,q.z)).x< -1e-6;}
  if(p.y<0.5){return boundaryVelocity(vec3i(q.x,0,q.z)).y>1e-6;}
  if(p.y>f32(d.y)-0.5){return faceVelocity(vec3i(q.x,d.y-1,q.z)).y< -1e-6;}
  if(p.z<0.5){return boundaryVelocity(vec3i(q.x,q.y,0)).z>1e-6;}
  if(p.z>f32(d.z)-0.5){return faceVelocity(vec3i(q.x,q.y,d.z-1)).z< -1e-6;}
  return false;
}
fn integrateTraceOffset(id:vec3i,dt:f32,h:vec3f,direction:f32)->vec3f{
  let position=vec3f(id)+vec3f(0.5);
  let hMin=min(h.x,min(h.y,h.z));
  let substeps=clamp(i32(ceil(length(sampleVelocity(position))*dt/hMin)),1,16);
  let sdt=dt/f32(substeps);
  var p=position;
  for(var s=0;s<substeps;s+=1){
    let midpoint=clampTraceToDomain(p+direction*0.5*sampleVelocity(p)*sdt/h);
    let candidate=p+direction*sampleVelocity(midpoint)*sdt/h;
    if(direction<0.0&&backwardTraceExitsReleasedFace(candidate)){p=candidate;break;}
    let next=clampTraceToDomain(candidate);
    // Stop at embedded solids: the paper's trace "stops if it crosses a
    // solid boundary" rather than passing mass through the obstacle.
    if(cellInsideSolid(vec3i(floor(next)))){break;}
    p=next;
  }
  return p-position;
}
fn backwardTraceOffset(id:vec3i,dt:f32,h:vec3f)->vec3f{
  return integrateTraceOffset(id,dt,h,-1.0);
}
fn forwardTraceOffset(id:vec3i,dt:f32,h:vec3f)->vec3f{
  return integrateTraceOffset(id,dt,h,1.0);
}
// CM12 Secs. 3.6-3.7 classify density storage with the cut-cell volume V, not
// the centre-point solid predicate. A cell whose centre is inside a body can
// still have V>0 and retain rho<=V after excess ejection. Masking that partial
// donor deletes its valid open-subcell density from the conservative matrix.
// The host reconciles rho with current V before this predicate is evaluated,
// so only geometrically full cells are excluded.
fn densityTransportDestination(p:vec3i)->bool{return valid(p)&&cellOpenFraction(p)>1e-5;}
fn transportStencilWeight(base:vec3i,f:vec3f,corner:u32)->f32{
  let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  if(!densityTransportDestination(base+offset)){return 0.0;}
  return select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
}
fn sampleGammaStencil(base:vec3i,f:vec3f)->f32{
  var terms:array<f32,8>;
  // Invalid or solid corners are the zero Dirichlet extension of gamma. Do
  // not renormalize this backward gather: the paper permits deficient gamma.
  for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;let weight=transportStencilWeight(base,f,corner);terms[corner]=0.0;if(weight>0.0){terms[corner]=weight*textureLoad(gammaIn,donor,0).x;}}
  return d4Sum8(terms);
}
// Steps 1-3: backward-advect persistent gamma, initialize beta on the host,
// and scatter gamma_i w^-_li to each donor l.
@compute @workgroup_size(4,4,4)
fn traceGammaAndBeta(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  if(!densityTransportDestination(id)){textureStore(gammaOut,id,vec4f(0.0));return;}
  let traced=backwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  // CM12 step 1 is an ordinary backward semi-Lagrangian sample of the
  // cumulative row sum. In particular, a characteristic that leaves a
  // released solid face sees the zero exterior extension. Giving that case a
  // positive floor invents a backward coefficient at the wall and suppresses
  // the forward remainder that is responsible for detaching the wall cell.
  // The interior clamp is the existing large-CFL conditioning policy; it must
  // never turn the exterior zero into a synthetic wall coefficient.
  let sampledGamma=sampleGammaStencil(base,f);
  // Preserve a partially exterior gamma sample instead of applying the
  // interior floor. Beta and density gathering still normalize the visible
  // interpolation stencil below; only cumulative gamma sees the zero exterior.
  let advectedGamma=cm12ConditionedGamma(sampledGamma,total);
  textureStore(gammaOut,id,vec4f(advectedGamma));
  // No visible donor means there is no backward coefficient. Do not credit a
  // synthetic self coefficient: the gather has no corresponding density
  // term, and doing so prevents the donor's missing column weight from being
  // returned by steps 6-7.
  if(total<=1e-9){return;}
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;let weight=transportStencilWeight(base,f,corner)/total;
    if(weight>0.0){
      let betaContribution=cm12VolumeWeightedBetaContribution(1.0,1.0,advectedGamma*weight);
      atomicAdd(&sharpenDeposits[linearIndex(donor)],i32(round(betaContribution*CM12_TRANSPORT_FIXED)));
    }
  }
}

// Steps 6-7: sources whose beta is below one forward-scatter the missing
// column weight. gammaIn is the pre-advection gamma^n prescribed by step 7;
// the rho and gamma corrections are accumulated separately.
@compute @workgroup_size(4,4,4)
fn scatterDensityDeficit(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!densityTransportDestination(id)){return;}
  let deficit=max(0.0,1.0-betaValue(id));if(deficit<=1.0/CM12_TRANSPORT_FIXED){return;}
  let traced=forwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);let count=cellCount();
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  if(total<=1e-9){
    let index=linearIndex(id);
    let densityDeposit=cm12VolumeScaledDeficitTransfer(volume(id),1.0,1.0,deficit,1.0);
    let gammaDeposit=cm12VolumeScaledDeficitTransfer(textureLoad(gammaIn,id,0).x,1.0,1.0,deficit,1.0);
    atomicAdd(&sharpenDeposits[count+index],i32(round(densityDeposit*CM12_TRANSPORT_FIXED)));
    atomicAdd(&sharpenDeposits[2u*count+index],i32(round(gammaDeposit*CM12_TRANSPORT_FIXED)));
    return;
  }
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let receiver=base+offset;let weight=transportStencilWeight(base,f,corner)/total;
    if(weight<=0.0){continue;}let index=linearIndex(receiver);
    let densityDeposit=cm12VolumeScaledDeficitTransfer(volume(id),1.0,1.0,deficit,weight);
    let gammaDeposit=cm12VolumeScaledDeficitTransfer(textureLoad(gammaIn,id,0).x,1.0,1.0,deficit,weight);
    atomicAdd(&sharpenDeposits[count+index],i32(round(densityDeposit*CM12_TRANSPORT_FIXED)));
    atomicAdd(&sharpenDeposits[2u*count+index],i32(round(gammaDeposit*CM12_TRANSPORT_FIXED)));
  }
}

// Steps 4-5 plus resolve of 6-7. gammaIn is the backward-advected gamma from
// step 1. Scaling each donor by max(1,beta_l) performs the paper's clamp
// without materializing A.
@compute @workgroup_size(4,4,4)
fn gatherConservativeDensity(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  if(!densityTransportDestination(id)){textureStore(volumeOut,id,vec4f(0.0));textureStore(gammaOut,id,vec4f(0.0));return;}
  let traced=backwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);
  let advectedGamma=textureLoad(gammaIn,id,0).x;var rhoNext=0.0;var gammaNext=0.0;
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;
    if(total<=1e-9){break;}
    let weight=transportStencilWeight(base,f,corner)/total;if(weight<=0.0){continue;}
    let scaled=cm12ConditionedRowCoefficient(advectedGamma,weight,betaValue(donor));rhoNext+=scaled*volume(donor);
    // Step 5 publishes gamma-prime, the row sum of the conditioned operator.
    gammaNext+=scaled;
  }
  let count=cellCount();let index=linearIndex(id);
  rhoNext+=f32(atomicLoad(&sharpenDeposits[count+index]))/CM12_TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&sharpenDeposits[2u*count+index]))/CM12_TRANSPORT_FIXED;
  // The prescribed inflow is a mass source external to the conservative
  // operator.  Rasterize the entire timestep-swept plug: a one-layer receiver
  // source would cap the paper's CFL-25 jet at 1/25 of its authored flux. The
  // matching swept velocity support moves this new plug clear before the next
  // step; retain the density-capacity guard for recirculating liquid crossing
  // the prescribed nozzle volume.
  let inflowSource=min(inflowSweptPlugSource(id,params.dimsDt.w),max(0.0,1.0-rhoNext));
  rhoNext+=inflowSource;
  // A dropped ball is the same kind of thing as the nozzle — mass appearing
  // from outside the conservative operator — so it lands in the same place and
  // takes the same capacity guard. It is applied on one step and then the host
  // clears the params lane, which is what makes it a drop rather than a source.
  let dropped=min(dropSource(id),max(0.0,1.0-rhoNext));
  rhoNext+=dropped;
  // Gamma is initialized to one throughout the domain at startup. A cell
  // first wetted by the external reservoir needs the same operator state.
  if(inflowSource>0.0||dropped>0.0){gammaNext=max(gammaNext,1.0);}
  if(rhoNext<1e-5){gammaNext=1.0;}
  textureStore(volumeOut,id,vec4f(max(rhoNext,0.0)));
  textureStore(gammaOut,id,vec4f(max(gammaNext,0.0)));
}

// LAF11 Sec. 3.2 uses Gauss-Jacobi iterations within one dimension and
// Gauss-Seidel between dimensional sweeps. The value returned here is the
// signed (rho, gamma) flux into own across one face, calculated entirely
// from the immutable input textures for the current axis.
fn gammaDiffusionFaceOpen(lower:vec3i,axis:u32)->f32{
  var upper=lower;upper[axis]=upper[axis]+1;
  if(!valid(lower)||!valid(upper)){return 0.0;}
  return clamp(faceOpenFraction(lower,axis),0.0,1.0);
}

fn diffuseGammaAxis(id:vec3i,axis:u32){
  if(!valid(id)){return;}
  let ownRho=volume(id);
  let ownGamma=textureLoad(gammaIn,id,0).x;
  var delta=vec2f(0.0);

  var lower=id;lower[axis]=lower[axis]-1;
  let lowerOpen=gammaDiffusionFaceOpen(lower,axis);
  if(lowerOpen>0.0){
    delta+=cm12GammaDiffusionFluxInto(
      ownRho,ownGamma,volume(lower),textureLoad(gammaIn,lower,0).x,lowerOpen,
    );
  }

  var upper=id;upper[axis]=upper[axis]+1;
  let upperOpen=gammaDiffusionFaceOpen(id,axis);
  if(upperOpen>0.0){
    delta+=cm12GammaDiffusionFluxInto(
      ownRho,ownGamma,volume(upper),textureLoad(gammaIn,upper,0).x,upperOpen,
    );
  }

  // The two endpoints evaluate equal and opposite face fluxes from the same
  // snapshot. Avoid per-cell clamps here: the analytic update is nonnegative,
  // while clamping only one endpoint would break exact pair conservation.
  textureStore(volumeOut,id,vec4f(ownRho+delta.x));
  textureStore(gammaOut,id,vec4f(ownGamma+delta.y));
}
@compute @workgroup_size(4,4,4) fn diffuseGammaX(@builtin(global_invocation_id) gid:vec3u){diffuseGammaAxis(activeId(gid),0u);}
@compute @workgroup_size(4,4,4) fn diffuseGammaY(@builtin(global_invocation_id) gid:vec3u){diffuseGammaAxis(activeId(gid),1u);}
@compute @workgroup_size(4,4,4) fn diffuseGammaZ(@builtin(global_invocation_id) gid:vec3u){diffuseGammaAxis(activeId(gid),2u);}

fn diffusionVelocity(p:vec3i)->vec3f{let v=textureLoad(velocityIn,clampCell(p),0).xyz;if(params.boundary.y>0.5&&!valid(p)){return -v;}return v;}
fn strainMagnitude(id:vec3i)->f32{
  let h=params.cellGravity.xyz;let dx=(diffusionVelocity(id+vec3i(1,0,0))-diffusionVelocity(id-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(id+vec3i(0,1,0))-diffusionVelocity(id-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(id+vec3i(0,0,1))-diffusionVelocity(id-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);
  return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));
}
fn velocityLaplacian(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;let centre=diffusionVelocity(id);
  return (diffusionVelocity(id+vec3i(1,0,0))-2.0*centre+diffusionVelocity(id-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(id+vec3i(0,1,0))-2.0*centre+diffusionVelocity(id-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(id+vec3i(0,0,1))-2.0*centre+diffusionVelocity(id-vec3i(0,0,1)))/(h.z*h.z);
}

fn applyVelocityForces(id:vec3i,inputVelocity:vec3f,dt:f32,h:vec3f)->vec3f{
  var v=inputVelocity;let occupancy=surfaceOccupancy(id);if(occupancy>0.0){let molecular=params.physical.y/params.physical.x;v+=dt*molecular*velocityLaplacian(id);}
  // Body force lives on faces. A face participates whenever liquid exists on
  // either side; this is the same rule during impact and at equilibrium.
  let qy=id+vec3i(0,1,0);let yOccupancy=surfaceOccupancy(qy);
  // Sub-isovalue density is still physical liquid. Excluding it from gravity
  // leaves a thin sheet with no way to separate from a ceiling.
  let centerLiquid=occupancy>1e-5;
  let yLiquid=yOccupancy>1e-5;
  if(centerLiquid||yLiquid){v.y+=params.cellGravity.w*dt;}
  let qx=id+vec3i(1,0,0);let qz=id+vec3i(0,0,1);
  let xOccupancy=surfaceOccupancy(qx);let zOccupancy=surfaceOccupancy(qz);
  // Balanced-force CSF: pressure and capillary acceleration use the same
  // positive-face locations and alpha differences. Curvature is a deep
  // stencil, so evaluate the centre once and only on faces whose occupancy
  // difference can produce a non-zero force. The previous formulation
  // evaluated centre curvature three times and paid six curvature stencils in
  // every bulk cell even though the final multiplication was exactly zero.
  let sigmaOverRho=params.boundary.x/params.physical.x;
  if(sigmaOverRho>0.0){
    let dx=select(0.0,xOccupancy-occupancy,valid(qx));
    let dy=select(0.0,yOccupancy-occupancy,valid(qy));
    let dz=select(0.0,zOccupancy-occupancy,valid(qz));
    if(dx!=0.0||dy!=0.0||dz!=0.0){
      let centreCurvature=curvatureAt(id);
      if(dx!=0.0){v.x+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qx))*dx/h.x;}
      if(dy!=0.0){v.y+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qy))*dy/h.y;}
      if(dz!=0.0){v.z+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qz))*dz/h.z;}
    }
  }
  // Give the newly rasterized high-CFL plug its reservoir velocity before
  // projection. The projection may then redirect it at impacts; only the
  // actual nozzle face is re-imposed after projection.
  return applyInflowSweptVelocity(id,v);
}

@compute @workgroup_size(4,4,4)
fn semiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}carryBoundaryVelocity(id);${geometric ? `
  // Experiment E2b. Outside the fine tiles no cell within eight cells carries
  // liquid, a solid or a source, so the projection below rewrites every
  // component of this cell: a face keeps its advected value only when it or its +axis
  // neighbour owns a pressure row. The three backward traces and the force term
  // are therefore dead work. V and the pressure seed are still carried, exactly
  // as the dense path does.
  if(params.twoLevel.z>0.5&&!uvTwoLevelFineAt(vec3f(id)+vec3f(0.5))){
    textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));return;
  }` : ""}let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  // A closed-face sample uses the solid-side zero extension. Preserve an old
  // velocity directed away from a positive wall before adding this step's
  // forces; projection below will clamp only the into-wall sign.
  let d=dims();
  if(id.x==d.x-1){v.x=min(v.x,faceVelocity(id).x);}
  if(id.y==d.y-1&&params.boundary.w<=0.5){v.y=min(v.y,faceVelocity(id).y);}
  if(id.z==d.z-1){v.z=min(v.z,faceVelocity(id).z);}
  v=applyVelocityForces(id,v,dt,h);
  // Surface density is advanced by the dedicated Sec. 3.4 gamma/beta passes.
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let id=activeId(gid); if (!valid(id)) { return; }
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let cell=vec3f(id);var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  let advected=volume(id);let d=dims();
  if (id.x==d.x-1) { v.x=faceVelocity(id).x; }
  if (id.y==d.y-1&&params.boundary.w<=0.5) { v.y=faceVelocity(id).y; }
  if (id.z==d.z-1) { v.z=faceVelocity(id).z; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn reverseAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,-dt,h));let d=dims();
  if(id.x==d.x-1){v.x=faceVelocity(id).x;}if(id.y==d.y-1&&params.boundary.w<=0.5){v.y=faceVelocity(id).y;}if(id.z==d.z-1){v.z=faceVelocity(id).z;}textureStore(velocityOut,id,vec4f(v,0.0));
}

fn boundedMacCormack(id:vec3i,position:vec3f,component:u32,dt:f32,h:vec3f,predicted:f32,original:f32,reversed:f32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lowerCoordinate=vec3f(0.0);lowerCoordinate[component]=-1.0;
  // Same clipped chord the predictor used: bracketing the limiter against
  // donors the predictor never sampled would let the correction reintroduce
  // the through-body velocity the clip just removed.
  let q=clamp(clippedDeparturePoint(position,dt,h)-offset,lowerCoordinate,vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let fraction=fract(q);
  var donorWeights=array<f32,8>();var donorValues=array<f32,8>();
  donorValues[0]=sampledFaceVelocity(b,component);var lower=1e30;var upper=-1e30;
  donorWeights[0]=(1.0-fraction.x)*(1.0-fraction.y)*(1.0-fraction.z);
  if(donorWeights[0]>0.0){lower=donorValues[0];upper=donorValues[0];}
  for(var corner:u32=1u;corner<8u;corner+=1u){
    let cornerOffset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=sampledFaceVelocity(b+cornerOffset,component);
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(cornerOffset)>vec3f(0.5));
    donorWeights[corner]=weights.x*weights.y*weights.z;donorValues[corner]=value;
    if(donorWeights[corner]>0.0){lower=min(lower,value);upper=max(upper,value);}
  }
  let corrected=predicted+0.5*(original-reversed);
  let revert=corrected<lower||corrected>upper;
  if(MACCORMACK_AUDIT_ENABLED){
    let record=(linearIndex(id)*3u+component)*8u;
    macCormackAudit[record]=vec4f(q,f32(b.x));
    macCormackAudit[record+1u]=vec4f(f32(b.y),f32(b.z),fraction.x,fraction.y);
    macCormackAudit[record+2u]=vec4f(fraction.z,predicted,original,reversed);
    macCormackAudit[record+3u]=vec4f(corrected,lower,upper,select(0.0,1.0,revert));
    macCormackAudit[record+4u]=vec4f(donorWeights[0],donorWeights[1],donorWeights[2],donorWeights[3]);
    macCormackAudit[record+5u]=vec4f(donorWeights[4],donorWeights[5],donorWeights[6],donorWeights[7]);
    macCormackAudit[record+6u]=vec4f(donorValues[0],donorValues[1],donorValues[2],donorValues[3]);
    macCormackAudit[record+7u]=vec4f(donorValues[4],donorValues[5],donorValues[6],donorValues[7]);
  }
  return select(corrected,predicted,revert);
}

@compute @workgroup_size(4,4,4)
fn correctAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;
  // Apply body forces to closed-wall faces as well as interior faces. CM11a's
  // separating boundary is an inequality enforced by the following pressure
  // projection, not a permanent zero-normal velocity. Restoring the original
  // here discarded gravity at a released ceiling face, so a detached film
  // lost its downward acceleration and numerically stuck to the lid.
  var v=vec3f(boundedMacCormack(id,cell+vec3f(1.0,0.5,0.5),0u,dt,h,predicted.x,original.x,reversed.x),boundedMacCormack(id,cell+vec3f(0.5,1.0,0.5),1u,dt,h,predicted.y,original.y,reversed.y),boundedMacCormack(id,cell+vec3f(0.5,0.5,1.0),2u,dt,h,predicted.z,original.z,reversed.z));v=applyVelocityForces(id,v,dt,h);
  textureStore(velocityOut,id,vec4f(v,0.0));
}

@compute @workgroup_size(8,8,1)
fn buildHeight(@builtin(global_invocation_id) gid:vec3u){let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(i32(gid.x),y,i32(gid.y)))*params.cellGravity.y;}textureStore(heightOut,vec2i(gid.xy),vec4f(total));}

fn faceWorld(id:vec3i,axis:u32)->vec3f{
  var world=worldCell(id);world[axis]+=0.5*params.cellGravity.xyz[axis];return world;
}
fn domainFaceFluidVelocity(id:vec3i,axis:u32)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(valid(id)){return faceVelocity(id)[axis];}
  if(valid(neighbor)&&id[axis]==-1){return boundaryVelocity(neighbor)[axis];}
  return 0.0;
}
fn domainFaceSolidVelocity(id:vec3i,axis:u32,checkSolid:bool)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(!valid(id)||!valid(neighbor)||(!checkSolid&&!hasTerrain())){return 0.0;}
  return pressureFaceData(id,axis)[axis];
}
fn divergenceAt(id: vec3i, checkSolid: bool) -> f32 {
  // CM11a Eqs. 8-10. Vi is the non-solid cell fraction; V+/- are the
  // corresponding face fractions. This is not the common blended-flux
  // shortcut V u + (1-V) us, whose solid terms are algebraically different.
  let h=params.cellGravity.xyz;let vi=cellOpenFraction(id);var terms:array<f32,6>;
  for(var axis=0u;axis<3u;axis+=1u){
    var minus=id;minus[axis]-=1;
    let vp=pressureFaceVolumeFraction(id,axis);let vm=pressureFaceVolumeFraction(minus,axis);
    let up=domainFaceFluidVelocity(id,axis);let um=domainFaceFluidVelocity(minus,axis);
    let usp=domainFaceSolidVelocity(id,axis,checkSolid);let usm=domainFaceSolidVelocity(minus,axis,checkSolid);
    terms[2u*axis]=(vp*up)/h[axis]+(vp-vi)*usp;
    terms[2u*axis+1u]=-(vm*um)/h[axis]-(vm-vi)*usm;
  }
  return d4Sum6(terms);
}
// Mass-Conserving Eulerian Liquid Simulation Sec 3.7: cells holding more
// density than they represent add min(lambda (rho'-1), eta) artificial
// divergence (lambda = 0.5, eta = 1 per the paper), divided by dx, so the
// pressure solve pushes the excess out.
fn volumeCorrectionDivergence(id: vec3i) -> f32 {
  ${geometric ? `let positive=min(0.5*max(0.0,volume(id)-cellOpenFraction(id)),cellOpenFraction(id));
  ${referenceDimension === 3 ? "let rate=bitcast<f32>(atomicLoad(&sharpenDeposits[uvBalanceBase()]));return (positive-rate*uvSurfaceDeficit(id))/max(params.dimsDt.w,1e-12);" : "return positive/max(params.dimsDt.w,1e-12);"}` : `
  // Preserve CM12's calibrated small-excess slope exactly:
  // min(lambda * (rho' - 1), eta) / dx with lambda=0.5 and eta=1.
  // Replacing this by excess/dt makes the correction three times stronger at
  // the paper's dx=0.05, dt=1/30 settings. In free fall that turns local
  // transport compression into a large sideways pressure impulse and visibly
  // flattens a drop before impact.
  // Keep the existing fine-grid safety policy as a cap only. It prevents the
  // correction from expanding by more than one cell per time step without
  // changing the published equation at the Figure 2/3 resolution.
  return cm12VolumeCorrectionDivergence(
    pressureDensity(id),params.cellGravity.x,params.dimsDt.w,
  ); `}
}

fn curvatureAt(id:vec3i)->f32{
  let h=params.cellGravity.xyz;
  let x=(interfaceNormal(id+vec3i(1,0,0)).x-interfaceNormal(id-vec3i(1,0,0)).x)/(2.0*h.x);
  let y=(interfaceNormal(id+vec3i(0,1,0)).y-interfaceNormal(id-vec3i(0,1,0)).y)/(2.0*h.y);
  let z=(interfaceNormal(id+vec3i(0,0,1)).z-interfaceNormal(id-vec3i(0,0,1)).z)/(2.0*h.z);
  return -((x+z)+y);
}

${geometric ? `
// Match the finest CM11a matrix for interior and domain faces alike. Halo
// pressure texels can hold transfer scratch even when they have no row;
// canonical air pressure is zero, never the stored texel in that case.
fn geometricPressureValue(p:vec3i)->f32{
  return select(0.0,projectPressureValue(p),pressurePhi(p)<0.0);
}
fn geometricProjectedFace(id:vec3i,axis:u32,predicted:f32)->f32{
  var q=id;q[axis]+=1;
  let face=pressureFaceData(id,axis);
  if(face.w<=1e-6){return face[axis];}
  let a=pressurePhi(id)<0.0;let b=pressurePhi(q)<0.0;
  if(!a&&!b){return 0.0;}
  var theta=1.0;
  if(a&&!b){theta=ghostFluidFraction(id,q);}
  if(!a&&b){theta=ghostFluidFraction(q,id);}
  return predicted-params.dimsDt.w/params.physical.x*
    (geometricPressureValue(q)-geometricPressureValue(id))/(params.cellGravity[axis]*theta);
}
` : ""}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=activeId(gid); if (!valid(id)) { return; }${geometric ? `
  // Experiment E2b, the same tile set. A cell outside the fine tiles has no
  // pressure row and no liquid or solid neighbour, so every branch below lands
  // on the far-air arm: each open face is set to zero, and a boundary face of a
  // cell with no row is set to zero too. Write that result directly and skip
  // the face data, the pressure taps and the ghost-fluid fractions.
  if(params.twoLevel.z>0.5&&!uvTwoLevelFineAt(vec3f(id)+vec3f(0.5))){
    textureStore(velocityOut,id,vec4f(0.0));storeBoundaryVelocity(id,vec3f(0.0));
    textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));return;
  }` : ""}
  let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=velocity(id);var boundaryV=boundaryVelocity(id);let d=dims();
  let ex=id+vec3i(1,0,0);let ey=id+vec3i(0,1,0);let ez=id+vec3i(0,0,1);
  let p0=select(0.0,projectPressureValue(id),pressureLiquid(id));
  let neighbors=array<vec3i,3>(ex,ey,ez);
  for(var axis=0u;axis<3u;axis+=1u){
    let neighbor=neighbors[axis];
    ${geometric ? `
    if(id[axis]==0){var halo=id;halo[axis]-=1;
      boundaryV[axis]=geometricProjectedFace(halo,axis,boundaryV[axis]);}
    v[axis]=geometricProjectedFace(id,axis,v[axis]);
    ` : `
    if(id[axis]==0){
      var halo=id;halo[axis]-=1;
      let boundaryOpen=pressureFaceVolumeFraction(halo,axis);
      if(boundaryOpen>1e-5&&pressureLiquid(id)){boundaryV[axis]-=scale*(p0-projectPressureValue(halo))/h[axis];}
      else{boundaryV[axis]=0.0;}
    }
    if(id[axis]==d[axis]-1){
      if(axis==1u&&params.boundary.w>0.5){
        if(pressureLiquid(id)){
          let theta=ghostFluidFraction(id,neighbor);
          v.y-=scale*(0.0-p0)/(h.y*theta);
        }else{v.y=0.0;}
      }else if(pressureLiquid(id)){
        let boundaryOpen=pressureFaceVolumeFraction(id,axis);
        if(boundaryOpen>1e-5){
          // A partially open solid face couples to the CM11a p_min=0 halo.
          v[axis]-=scale*(projectPressureValue(neighbor)-p0)/h[axis];
        }else{v[axis]=domainFaceSolidVelocity(id,axis,true);}
      }else{
        // Thin density has no pressure row, but its boundary face still owns
        // the separating inequality. On a positive wall, retain v<=u_s.
        let solidVelocity=domainFaceSolidVelocity(id,axis,true);
        v[axis]=select(solidVelocity,min(v[axis],solidVelocity),volume(id)>1e-5);
      }
      continue;
    }
    let pressureFace=pressureFaceData(id,axis);let open=pressureFace.w;
    if(open<=1e-5){v[axis]=pressureFace[axis];continue;}
    let centreLiquid=pressureLiquid(id);let neighborLiquid=pressureLiquid(neighbor);
    if(centreLiquid||neighborLiquid){
      let p1=select(0.0,projectPressureValue(neighbor),neighborLiquid);
      var theta=1.0;
      if(centreLiquid&&!neighborLiquid){theta=ghostFluidFraction(id,neighbor);}
      if(!centreLiquid&&neighborLiquid){theta=ghostFluidFraction(neighbor,id);}
      v[axis]-=scale*(p1-p0)/(h[axis]*theta);
    }else{v[axis]=0.0;}
    `}
  }
  var released=0u;
  ${geometric ? `
  // Publish the solved contact active set with the MAC field. Phi must not
  // infer release from a tiny velocity residual at pressure-supported walls.
  for(var axis=0u;axis<3u;axis++){
    var q=id;q[axis]+=1;let ownOpen=cellOpenFraction(id);let otherOpen=cellOpenFraction(q);
    if((ownOpen>1e-5)!=(otherOpen>1e-5)){
      let solid=select(id,q,ownOpen>1e-5);let inward=select(1.0,-1.0,ownOpen>1e-5);
      let wall=pressureFaceData(id,axis);
      if(wall.w>1e-6&&geometricPressureValue(solid)<=0.0&&inward*(v[axis]-wall[axis])*params.dimsDt.w>1e-4*h[axis]){released|=1u<<axis;}
    }
    if(id[axis]==0){var halo=id;halo[axis]-=1;
      if(cellOpenFraction(id)>1e-5&&pressureFaceVolumeFraction(halo,axis)>1e-6&&geometricPressureValue(halo)<=0.0&&boundaryV[axis]*params.dimsDt.w>1e-4*h[axis]){released|=1u<<(axis+3u);}}
  }` : ""}
  v=applyInflowVelocity(id,v);textureStore(velocityOut,id,vec4f(v,f32(released)));storeBoundaryVelocity(id,boundaryV); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

// Moving-solid bookkeeping after the variational projection.  The old
// Brinkman velocity blend changed face velocities after incompressibility and
// reintroduced divergence; solid motion is now imposed inside the projection,
// so this pass only records the diagnostic/reaction load.  Sec. 3.6 already
// performs the conservative covered-density redistribution before projection.
@compute @workgroup_size(4,4,4)
fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}carryBoundaryVelocity(id);let phi=volume(id);let wetFraction=surfaceOccupancy(id);var v=velocity(id);let h=params.cellGravity.xyz;
  let world=vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);
  let bodyCount=u32(round(params.boundary.z));let cellMass=params.physical.x*h.x*h.y*h.z*wetFraction;let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);var coupledBody=12u;var solidFraction=0.0;
  // Match the adaptive voxelizer's overlap rule: the body with the greatest
  // sub-cell coverage owns this cell, so displaced volume is never counted
  // twice and does not depend on body-array order.
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let candidate=bodySolidFraction(rigidBodies[bodyIndex],id);if(candidate>solidFraction){solidFraction=candidate;coupledBody=bodyIndex;}}
  if(coupledBody<12u){
    let bodyIndex=coupledBody;let body=rigidBodies[bodyIndex];
    let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let fluidVelocity=v;let ambientVelocity=ambientFluidVelocity(body,id,fluidVelocity);let fluidImpulse=cellMass*solidFraction*(solidVelocity-fluidVelocity)*blend;
    let reaction=-fluidImpulse;let torque=cross(arm,reaction);let base=bodyIndex*12u;
    atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
    atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
    let displacedWeight=wetFraction*solidFraction;
    atomicAdd(&rigidExchange[base+6u],i32(round(displacedWeight*65536.0)));
    atomicAdd(&rigidExchange[base+7u],i32(round(displacedWeight*ambientVelocity.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displacedWeight*ambientVelocity.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displacedWeight*ambientVelocity.z*10000.0)));
  }
  // The nozzle mouth is an open boundary. Coupling the visual nozzle body
  // must not replace the prescribed reservoir velocity at that opening.
  v=applyInflowVelocity(id,v);
  textureStore(velocityOut,id,vec4f(v,textureLoad(velocityIn,id,0).w));textureStore(volumeOut,id,vec4f(phi));
}

// Paper Sec 3.9.1 phi-s for the resident adaptive level set. While an adaptive
// projection owns the pressure solve the uniform pressure textures are idle,
// so this pass aliases them: pressureIn is a copy of the signed-distance field
// and pressureOut is the resident level-set texture itself.
@compute @workgroup_size(4,4,4)
fn relaxSolidPhi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let phi=textureLoad(pressureIn,id,0).x;
  var result=phi;
  if(staticSolidVoxelOccupied(id)||nearAnyBody(worldCell(id))){
    let s=cellSolidFraction(id);
    if(s>0.0){
      var open=0.0;var openSum=0.0;var total=0.0;var exteriorOpen=0.0;var exteriorSum=0.0;
      let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
      for(var index=0;index<6;index+=1){
        let np=clampCell(id+offsets[index]);
        let neighborPhi=textureLoad(pressureIn,np,0).x;let neighborOpen=cellOpenFraction(np);total+=neighborPhi;
        open+=neighborOpen;openSum+=neighborOpen*neighborPhi;
        // Extend phi from the first genuinely open sample on each coordinate
        // ray. A one-cell relaxation takes many frames to cross a large solid
        // and leaves a newly submerged body falsely dry, under-reporting its
        // displaced volume. Six exterior samples establish the correct phase
        // throughout the solid in this pass while retaining a local fallback
        // for bodies wider than the bounded search.
        for(var step=1;step<=64;step+=1){
          let exterior=id+step*offsets[index];if(!valid(exterior)){break;}
          let exteriorWeight=cellOpenFraction(exterior);
          if(exteriorWeight>0.5){exteriorOpen+=exteriorWeight;exteriorSum+=exteriorWeight*textureLoad(pressureIn,exterior,0).x;break;}
        }
      }
      let localTarget=select(total/6.0,openSum/max(open,1.0),open>0.0);
      let relaxTarget=select(localTarget,exteriorSum/max(exteriorOpen,1.0),exteriorOpen>0.0);
      result=mix(phi,relaxTarget,s);
    }
  }
  textureStore(pressureOut,id,vec4f(result));
}

// --- Density sharpening (Mass-Conserving Eulerian Liquid Simulation Sec 3.5,
// Eq 4-17 and Algorithm 2; docs/TALL_CELLS_PAPER.md Appendix B.3). Pass 1
// applies the local correction (Eq 17 keeps it non-positive: mass only moves
// from the air side to the liquid side); pass 2 returns the removed mass by
// tracing along the density gradient to the 0.5 iso-contour and depositing
// fixed-point trilinear weights; pass 3 folds the deposits back in.
fn cellInsideSolid(p:vec3i)->bool{
  if(staticSolidVoxelOccupied(p)){return true;}
  if(!valid(p)){return false;}
  if(cellInsideTerrain(p)){return true;}
  let bodyCount=u32(round(params.boundary.z));if(bodyCount==0u){return false;}
  let world=worldCell(p);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return true;}}
  return false;
}
fn sharpenDeltaRho(q:vec3i)->f32{
  let rho=volume(q);
  if(cellInsideSolid(q)){return 0.0;}
  let h=params.cellGravity.xyz;
  let deltaT=3.0*params.dimsDt.w*params.tuning.x;
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  // Sec. 3.6 Eqs. 18-19 use non-solid face aperture area V^f. This is
  // deliberately distinct from CM11a's face-centred overlapping dual volume.
  let openXp=faceOpenFraction(q,0u);let openXm=faceOpenFraction(q-ex,0u);
  let openYp=faceOpenFraction(q,1u);let openYm=faceOpenFraction(q-ey,1u);
  let openZp=faceOpenFraction(q,2u);let openZm=faceOpenFraction(q-ez,2u);
  let sxp=-(rho*max(openXp,0.0)-volume(q-ex)*max(openXm,0.0))*deltaT/h.x;let sxm=-(volume(q+ex)*max(openXp,0.0)-rho*max(openXm,0.0))*deltaT/h.x;
  let syp=-(rho*max(openYp,0.0)-volume(q-ey)*max(openYm,0.0))*deltaT/h.y;let sym=-(volume(q+ey)*max(openYp,0.0)-rho*max(openYm,0.0))*deltaT/h.y;
  let szp=-(rho*max(openZp,0.0)-volume(q-ez)*max(openZm,0.0))*deltaT/h.z;let szm=-(volume(q+ez)*max(openZp,0.0)-rho*max(openZm,0.0))*deltaT/h.z;
  let plusX=max(max(sxp,0.0)*max(sxp,0.0),min(sxm,0.0)*min(sxm,0.0));let plusY=max(max(syp,0.0)*max(syp,0.0),min(sym,0.0)*min(sym,0.0));let plusZ=max(max(szp,0.0)*max(szp,0.0),min(szm,0.0)*min(szm,0.0));
  let minusX=max(min(sxp,0.0)*min(sxp,0.0),max(sxm,0.0)*max(sxm,0.0));let minusY=max(min(syp,0.0)*min(syp,0.0),max(sym,0.0)*max(sym,0.0));let minusZ=max(min(szp,0.0)*min(szp,0.0),max(szm,0.0)*max(szm,0.0));
  let gradPlus=sqrt((plusX+plusZ)+plusY);let gradMinus=sqrt((minusX+minusZ)+minusY);
  var maximumDifference=0.0;
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var index=0;index<6;index+=1){maximumDifference=max(maximumDifference,abs(rho-volume(q+offsets[index])));}
  let weight=cm12SharpeningWeight(rho,maximumDifference);
  var deltaRho=select(weight*gradMinus,weight*gradPlus,weight>=0.0);
  if(rho+deltaRho<0.0||rho<1e-5){deltaRho=-rho;}else if(rho>0.5){deltaRho=0.0;}
  return deltaRho;
}
@compute @workgroup_size(4,4,4)
fn sharpenCompute(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let stored=textureLoad(volumeIn,id,0).x;
  let deltaRho=sharpenDeltaRho(id);
  textureStore(volumeOut,id,vec4f(stored+deltaRho));
  textureStore(pressureOut,id,vec4f(deltaRho));
}
@compute @workgroup_size(4,4,4)
fn sharpenScatter(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let deltaRho=textureLoad(pressureIn,id,0).x;if(deltaRho>=0.0){return;}
  var p=vec3f(id)+vec3f(0.5);let maximumDistance=params.tuning.y;var travelled=0.0;let stepLength=0.5;
  for(var stepIndex=0;stepIndex<7;stepIndex+=1){
    if(sampleVolume(p)>=0.5||travelled>=maximumDistance){break;}
    let g=volumeGradient(vec3i(floor(p)));let magnitude=length(g);
    if(magnitude<1e-6){break;}
    let distance=min(stepLength,maximumDistance-travelled);
    let candidate=p+g/magnitude*distance;
    if(cellInsideSolid(vec3i(floor(candidate)))){break;}
    p=candidate;travelled+=distance;
    if(sampleVolume(p)>=0.5){break;}
  }
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<i32,8>();var total=0.0;
  let d=dims();
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let destination=cell+offset;
    var w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    var index=-1;
    if(valid(destination)&&!cellInsideSolid(destination)){index=destination.x+d.x*(destination.y+d.y*destination.z);}else{w=0.0;}
    weights[corner]=w;indices[corner]=index;total+=w;
  }
  if(total<=1e-8){
    let ownIndex=id.x+d.x*(id.y+d.y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*CM12_TRANSPORT_FIXED)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){
    if(weights[corner]<=0.0){continue;}
    atomicAdd(&sharpenDeposits[u32(indices[corner])],i32(round(-deltaRho*weights[corner]/total*CM12_TRANSPORT_FIXED)));
  }
}
@compute @workgroup_size(4,4,4)
fn sharpenResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let d=dims();let index=u32(id.x+d.x*(id.y+d.y*id.z));
  let deposit=f32(atomicLoad(&sharpenDeposits[index]))/CM12_TRANSPORT_FIXED;
  textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x+deposit));
}

// Sec. 3.6: when rho exceeds V, trace the excess for S*dx along the gradient
// of the solid signed-distance field (positive away from the union of solids).
fn solidSignedDistance(world:vec3f)->f32{
  var distance=1e20;
  if(hasTerrain()){
    let h=params.cellGravity.xyz;
    let x=i32(floor((world.x+0.5*params.container.x)/h.x));
    let z=i32(floor((world.z+0.5*params.container.z)/h.z));
    distance=min(distance,world.y-terrainHeightCells(x,z)*h.y);
  }
  let bodyCount=u32(round(params.boundary.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    distance=min(distance,rigidSignedDistance(rigidBodies[bodyIndex],world));
  }
  return distance;
}
fn solidSignedDistanceGradient(world:vec3f)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(
    (solidSignedDistance(world+vec3f(h.x,0.0,0.0))-solidSignedDistance(world-vec3f(h.x,0.0,0.0)))/(2.0*h.x),
    (solidSignedDistance(world+vec3f(0.0,h.y,0.0))-solidSignedDistance(world-vec3f(0.0,h.y,0.0)))/(2.0*h.y),
    (solidSignedDistance(world+vec3f(0.0,0.0,h.z))-solidSignedDistance(world-vec3f(0.0,0.0,h.z)))/(2.0*h.z));
}
@compute @workgroup_size(4,4,4)
fn scatterSolidExcess(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let rho=volume(id);let open=cellOpenFraction(id);
  if(open>=1.0-1e-6){textureStore(volumeOut,id,vec4f(rho));return;}
  let excess=max(0.0,rho-open);
  textureStore(volumeOut,id,vec4f(rho-excess));if(excess<=0.0){return;}
  let h=params.cellGravity.xyz;let dx=min(h.x,min(h.y,h.z));
  let gradient=solidSignedDistanceGradient(worldCell(id));var p=vec3f(id)+vec3f(0.5);
  if(length(gradient)>1e-6){p+=normalize(gradient)*dx/h;}
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<u32,8>();var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let destination=cell+offset;
    var weight=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    if(!valid(destination)||cellInsideSolid(destination)){weight=0.0;}else{indices[corner]=linearIndex(destination);}
    weights[corner]=weight;total+=weight;
  }
  // The donor was already reduced to V. Returning an unplaceable deposit to
  // a V=0 donor would violate the paper's rho=0-inside-solid guarantee.
  if(total<=1e-9){
    // The gradient landed entirely in solid.  Before conceding the mass, sweep
    // the 26-neighbourhood for any open cell: a body sweeping through liquid
    // evicts whole cells at once, and the single-gradient-step trace fails
    // often enough there that conceding on the first miss is visible volume
    // loss exactly while the user is playing with the body.
    var fallbackTotal=0.0;
    for(var neighbor=0u;neighbor<27u;neighbor+=1u){
      if(neighbor==13u){continue;}
      let destination=id+vec3i(i32(neighbor%3u)-1,i32((neighbor/3u)%3u)-1,i32(neighbor/9u)-1);
      if(valid(destination)&&!cellInsideSolid(destination)){fallbackTotal+=cellOpenFraction(destination);}
    }
    if(fallbackTotal<=1e-9){
      // Genuinely enclosed by solid.  Keep rho=0 inside the body and expose
      // the unplaceable conservative mass instead of silently losing it.
      atomicAdd(&reductions[4],u32(round(excess*2048.0)));
      return;
    }
    for(var neighbor=0u;neighbor<27u;neighbor+=1u){
      if(neighbor==13u){continue;}
      let destination=id+vec3i(i32(neighbor%3u)-1,i32((neighbor/3u)%3u)-1,i32(neighbor/9u)-1);
      if(!valid(destination)||cellInsideSolid(destination)){continue;}
      let share=cellOpenFraction(destination)/fallbackTotal;
      if(share<=0.0){continue;}
      atomicAdd(&sharpenDeposits[linearIndex(destination)],i32(round(excess*share*CM12_TRANSPORT_FIXED)));
    }
    return;
  }
  for(var corner=0u;corner<8u;corner+=1u){if(weights[corner]>0.0){atomicAdd(&sharpenDeposits[indices[corner]],i32(round(excess*weights[corner]/total*CM12_TRANSPORT_FIXED)));}}
}
@compute @workgroup_size(4,4,4)
fn resolveSolidExcess(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let deposit=f32(atomicLoad(&sharpenDeposits[linearIndex(id)]))/CM12_TRANSPORT_FIXED;
  textureStore(volumeOut,id,vec4f(volume(id)+deposit));
}

// Render-only wall-film reconstruction. For a solid signed distance s and a
// supported cell density rho<.5, R=.5-s/h+rho has its .5 contour at s=rho*h.
// It therefore displays the mass as a proportionally thin sheet rather than
// promoting it to a half-cell liquid region. R is written only to the render
// texture and never becomes transport or pressure authority.
fn domainWallFilmCell(id:vec3i,rho:f32)->bool{
  if(rho<=1e-5||rho>=0.5){return false;}
  let d=dims();
  return id.x==0||id.z==0||id.x==d.x-1||id.z==d.z-1||id.y==0
    ||(id.y==d.y-1&&params.boundary.w<=0.5);
}
fn embeddedWallFilm(id:vec3i)->vec2f{
  let world=worldCell(id);let distance=solidSignedDistance(world);
  if(distance>=1e19){return vec2f(0.0);}
  let gradient=solidSignedDistanceGradient(world);let gradientLength=length(gradient);
  if(gradientLength<=1e-6){return vec2f(0.0);}
  let normal=gradient/gradientLength;let h=params.cellGravity.xyz;
  let normalCellWidth=1.0/max(length(normal/h),1e-6);
  if(distance>1.25*normalCellWidth){return vec2f(0.0);}
  let sourceWorld=world+normal*(0.5*normalCellWidth-distance);
  let source=vec3i(floor(vec3f(
    (sourceWorld.x+0.5*params.container.x)/h.x,
    sourceWorld.y/h.y,
    (sourceWorld.z+0.5*params.container.z)/h.z)));
  if(!valid(source)||cellInsideSolid(source)){return vec2f(0.0);}
  let rho=textureLoad(volumeIn,source,0).x;
  if(rho<=1e-5||rho>=0.5){return vec2f(0.0);}
  return vec2f(max(0.0,0.5-distance/normalCellWidth+rho),1.0);
}
fn wallFilmResolvedDensity(id:vec3i,base:f32)->f32{
  let film=embeddedWallFilm(id);
  return max(base,film.x);
}
@compute @workgroup_size(4,4,4)
fn wallFilmResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let rho=textureLoad(volumeIn,id,0).x;
  textureStore(volumeOut,id,vec4f(wallFilmResolvedDensity(id,rho)));
}

// Optional Sec. 3.8 reconstruction. Blur g=2 min(rho,.5) with a separable
// Gaussian (sigma=2 cells), then expose sub-grid mass through
// rho''=rho/min(max(g,theta),1), theta=.01. None of these outputs are rebound
// into transport or projection.
fn blurPostprocessAxis(id:vec3i,axis:u32,seedDensity:bool)->f32{
  var weighted=0.0;var total=0.0;
  for(var offset=-6;offset<=6;offset+=1){
    var q=id;q[axis]+=offset;if(!valid(q)){continue;}
    let weight=exp(-f32(offset*offset)/8.0);let sample=textureLoad(volumeIn,q,0).x;
    weighted+=weight*select(sample,2.0*min(sample,0.5),seedDensity);total+=weight;
  }
  return weighted/max(total,1e-9);
}
fn storePostprocessBlur(id:vec3i,axis:u32,seedDensity:bool){textureStore(volumeOut,id,vec4f(blurPostprocessAxis(id,axis,seedDensity)));}
@compute @workgroup_size(4,4,4) fn postprocessBlurX(@builtin(global_invocation_id) gid:vec3u){let id=activeId(gid);if(valid(id)){storePostprocessBlur(id,0u,true);}}
@compute @workgroup_size(4,4,4) fn postprocessBlurY(@builtin(global_invocation_id) gid:vec3u){let id=activeId(gid);if(valid(id)){storePostprocessBlur(id,1u,false);}}
@compute @workgroup_size(4,4,4) fn postprocessBlurZ(@builtin(global_invocation_id) gid:vec3u){let id=activeId(gid);if(valid(id)){storePostprocessBlur(id,2u,false);}}
@compute @workgroup_size(4,4,4)
fn postprocessResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let rho=textureLoad(volumeIn,id,0).x;let blurredGamma=textureLoad(surfaceIn,id,0).x;
  var reconstructed=rho/min(max(blurredGamma,0.01),1.0);
  // Preserve calibrated wall-cell rho: the surface extractor supplies its
  // matching boundary ghost. Sec. 3.8 remains active everywhere unsupported.
  if(domainWallFilmCell(id,rho)){reconstructed=rho;}
  let embedded=embeddedWallFilm(id);
  if(embedded.y>0.5){reconstructed=max(rho,embedded.x);}
  textureStore(volumeOut,id,vec4f(reconstructed));
}

// The census deliberately contains no atomics. Each 4^3 workgroup reduces its
// cells in shared memory and writes one uncontended 32-byte summary. A single
// 256-lane pass then merges those compact records. This avoids serializing all
// wet cells on seven device-global words, which overwhelmed sparse scenes.
var<workgroup> activeMinimumLanes:array<vec3u,256>;
var<workgroup> activeMaximumLanes:array<vec3u,256>;
var<workgroup> activeSpeedLanes:array<u32,256>;
var<workgroup> activeTravelPlusLanes:array<u32,256>;
var<workgroup> activeTravelMinusLanes:array<u32,256>;
// Per-axis, per-DIRECTION travel in cells, reduced over wet cells only. The
// window's padding is built from these rather than from one global |v| so that
// liquid sloshing sideways does not pad the window upward, and falling liquid
// pads only downward. The one-step gravity and inflow terms are added by the
// finalize, which is where their directions are known.
fn activeTravelCells(id:vec3i,positive:bool)->vec3u{
  let v=faceVelocity(id);
  let directed=max(select(-v,v,positive),vec3f(0.0));
  return vec3u(ceil(directed*params.dimsDt.w/params.cellGravity.xyz));
}
fn writeActiveWorkgroupSummary(
  id:vec3i, wet:bool, localIndex:u32, workgroupId:vec3u, groupCount:vec3u,
){
  let d=vec3u(dims());
  var minimum=d;
  var maximum=vec3u(0u);
  var speedBits=0u;
  var travelPlus=0u;
  var travelMinus=0u;
  if(wet){
    minimum=vec3u(id);maximum=minimum+vec3u(1u);
    speedBits=bitcast<u32>(length(faceVelocity(id)));
    travelPlus=activePackTravel(activeTravelCells(id,true));
    travelMinus=activePackTravel(activeTravelCells(id,false));
  }
  activeMinimumLanes[localIndex]=minimum;
  activeMaximumLanes[localIndex]=maximum;
  activeSpeedLanes[localIndex]=speedBits;
  activeTravelPlusLanes[localIndex]=travelPlus;
  activeTravelMinusLanes[localIndex]=travelMinus;
  workgroupBarrier();
  var stride=32u;
  loop{
    if(localIndex<stride){
      activeMinimumLanes[localIndex]=min(activeMinimumLanes[localIndex],activeMinimumLanes[localIndex+stride]);
      activeMaximumLanes[localIndex]=max(activeMaximumLanes[localIndex],activeMaximumLanes[localIndex+stride]);
      activeSpeedLanes[localIndex]=max(activeSpeedLanes[localIndex],activeSpeedLanes[localIndex+stride]);
      activeTravelPlusLanes[localIndex]=activeMaxPacked(activeTravelPlusLanes[localIndex],activeTravelPlusLanes[localIndex+stride]);
      activeTravelMinusLanes[localIndex]=activeMaxPacked(activeTravelMinusLanes[localIndex],activeTravelMinusLanes[localIndex+stride]);
    }
    workgroupBarrier();
    if(stride==1u){break;}stride/=2u;
  }
  if(localIndex==0u){
    if(all(workgroupId==vec3u(0u))){
      activeScratch[ACTIVE_SCAN_GROUPS_WORD]=groupCount.x;
      activeScratch[ACTIVE_SCAN_GROUPS_WORD+1u]=groupCount.y;
      activeScratch[ACTIVE_SCAN_GROUPS_WORD+2u]=groupCount.z;
    }
    let summaryIndex=workgroupId.x+groupCount.x*(workgroupId.y+groupCount.y*workgroupId.z);
    let base=ACTIVE_SUMMARY_BASE+12u*summaryIndex;
    activeScratch[base]=activeMinimumLanes[0].x;activeScratch[base+1u]=activeMinimumLanes[0].y;
    activeScratch[base+2u]=activeMinimumLanes[0].z;activeScratch[base+3u]=activeSpeedLanes[0];
    activeScratch[base+4u]=activeMaximumLanes[0].x;activeScratch[base+5u]=activeMaximumLanes[0].y;
    activeScratch[base+6u]=activeMaximumLanes[0].z;activeScratch[base+7u]=0u;
    activeScratch[base+8u]=activeTravelPlusLanes[0];activeScratch[base+9u]=activeTravelMinusLanes[0];
    activeScratch[base+10u]=0u;activeScratch[base+11u]=0u;
  }
}
// Conservative full-strength swept inlet footprint. Retain this seed while
// the inlet runs even if its current ramp injects no measurable liquid yet.
// The swept disk and its antialias/cell-overlap support fit inside this AABB.
fn uniformInflowWindowSeed(id:vec3i)->bool{
  if(!valid(id)||inflowStrength()<=0.0){return false;}
  let h=params.cellGravity.xyz;
  let start=params.inflowPositionRadius.xyz;
  let end=start+params.inflowVelocityLength.xyz*params.dimsDt.w;
  let pad=vec3f(params.inflowPositionRadius.w+length(h));
  let world=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z)+(vec3f(id)+vec3f(0.5))*h;
  return all(world>=min(start,end)-pad)&&all(world<=max(start,end)+pad);
}
@compute @workgroup_size(4,4,4)
fn scanActiveRegion(
  @builtin(global_invocation_id) gid:vec3u,
  @builtin(local_invocation_index) localIndex:u32,
  @builtin(workgroup_id) workgroupId:vec3u,
  @builtin(num_workgroups) groupCount:vec3u,
){
  let id=activeId(gid);
  let source=uniformInflowWindowSeed(id);
  let wet=${geometric ? "source||geometricActiveSeed(id)" : "source||(valid(id)&&volume(id)>1e-5)"};
  writeActiveWorkgroupSummary(id,wet,localIndex,workgroupId,groupCount);
}
@compute @workgroup_size(4,4,4)
fn scanExternalActiveSources(
  @builtin(global_invocation_id) gid:vec3u,
  @builtin(local_invocation_index) localIndex:u32,
  @builtin(workgroup_id) workgroupId:vec3u,
  @builtin(num_workgroups) groupCount:vec3u,
){
  let id=vec3i(gid);let inDomain=valid(id);
  let source=uniformInflowWindowSeed(id)||(inDomain&&dropSource(id)>0.0);
  let wet=${geometric ? "source||geometricActiveSeed(id)" : "inDomain&&(volume(id)>1e-5||source)"};
  writeActiveWorkgroupSummary(id,wet,localIndex,workgroupId,groupCount);
}
fn reduceActiveSummaryRange(groupCount:vec3u,lane:u32){
  let d=vec3u(dims());let summaryCount=groupCount.x*groupCount.y*groupCount.z;
  var minimum=d;var maximum=vec3u(0u);var speedBits=0u;
  var travelPlus=0u;var travelMinus=0u;
  for(var summaryIndex=lane;summaryIndex<summaryCount;summaryIndex+=256u){
    let base=ACTIVE_SUMMARY_BASE+12u*summaryIndex;
    minimum=min(minimum,vec3u(activeScratch[base],activeScratch[base+1u],activeScratch[base+2u]));
    maximum=max(maximum,vec3u(activeScratch[base+4u],activeScratch[base+5u],activeScratch[base+6u]));
    speedBits=max(speedBits,activeScratch[base+3u]);
    travelPlus=activeMaxPacked(travelPlus,activeScratch[base+8u]);
    travelMinus=activeMaxPacked(travelMinus,activeScratch[base+9u]);
  }
  activeMinimumLanes[lane]=minimum;
  activeMaximumLanes[lane]=maximum;
  activeSpeedLanes[lane]=speedBits;
  activeTravelPlusLanes[lane]=travelPlus;
  activeTravelMinusLanes[lane]=travelMinus;
  workgroupBarrier();
  var stride=128u;
  loop{
    if(lane<stride){
      activeMinimumLanes[lane]=min(activeMinimumLanes[lane],activeMinimumLanes[lane+stride]);
      activeMaximumLanes[lane]=max(activeMaximumLanes[lane],activeMaximumLanes[lane+stride]);
      activeSpeedLanes[lane]=max(activeSpeedLanes[lane],activeSpeedLanes[lane+stride]);
      activeTravelPlusLanes[lane]=activeMaxPacked(activeTravelPlusLanes[lane],activeTravelPlusLanes[lane+stride]);
      activeTravelMinusLanes[lane]=activeMaxPacked(activeTravelMinusLanes[lane],activeTravelMinusLanes[lane+stride]);
    }
    workgroupBarrier();
    if(stride==1u){break;}stride/=2u;
  }
  if(lane==0u){
    activeScratch[0]=activeMinimumLanes[0].x;activeScratch[1]=activeMinimumLanes[0].y;activeScratch[2]=activeMinimumLanes[0].z;
    activeScratch[3]=activeMaximumLanes[0].x;activeScratch[4]=activeMaximumLanes[0].y;activeScratch[5]=activeMaximumLanes[0].z;
    activeScratch[6]=activeSpeedLanes[0];
    activeScratch[ACTIVE_TRAVEL_PLUS_WORD]=activeTravelPlusLanes[0];
    activeScratch[ACTIVE_TRAVEL_MINUS_WORD]=activeTravelMinusLanes[0];
  }
}
@compute @workgroup_size(256)
fn reduceActiveRegionSummaries(@builtin(local_invocation_index) lane:u32){
  // Direct launches deliberately over-cover the exact GPU window. Indexing
  // their summaries with the smaller exact dimensions aliases empty overrun
  // groups onto valid groups, making the census depend on GPU scheduling.
  reduceActiveSummaryRange(vec3u(activeScratch[ACTIVE_SCAN_GROUPS_WORD],activeScratch[ACTIVE_SCAN_GROUPS_WORD+1u],activeScratch[ACTIVE_SCAN_GROUPS_WORD+2u]),lane);
}
@compute @workgroup_size(256)
fn reduceExternalActiveRegionSummaries(@builtin(local_invocation_index) lane:u32){
  reduceActiveSummaryRange((vec3u(dims())+vec3u(3u))/4u,lane);
}
fn activeCeilDiv(value:u32,divisor:u32)->u32{return (value+divisor-1u)/divisor;}
@compute @workgroup_size(1)
fn finalizeActiveRegion(){
  let d=vec3u(dims());
  let observedMin=vec3u(activeScratch[0],activeScratch[1],activeScratch[2]);
  let observedMax=vec3u(activeScratch[3],activeScratch[4],activeScratch[5]);
  let speed=max(bitcast<f32>(activeScratch[6]),length(params.inflowVelocityLength.xyz));
  let travel=vec3u(ceil(vec3f(speed*params.dimsDt.w)/params.cellGravity.xyz));
  // The two one-step terms the scan cannot see, each charged only to the
  // direction it actually pushes: this step's gravity kick (the scan reads the
  // velocity BEFORE the kick, so liquid falls g*dt*dt/h further than it
  // measured) and any authored inflow, whose liquid does not exist yet.
  let stepDt=params.dimsDt.w;
  let gravityCells=abs(params.cellGravity.w)*stepDt*stepDt/params.cellGravity.y;
  var accelerationPlus=vec3f(0.0);
  var accelerationMinus=vec3f(0.0);
  accelerationPlus.y=select(0.0,gravityCells,params.cellGravity.w>0.0);
  accelerationMinus.y=select(0.0,gravityCells,params.cellGravity.w<0.0);
  let inflowCells=abs(params.inflowVelocityLength.xyz)*stepDt/params.cellGravity.xyz;
  accelerationPlus+=select(vec3f(0.0),inflowCells,params.inflowVelocityLength.xyz>vec3f(0.0));
  accelerationMinus+=select(vec3f(0.0),inflowCells,params.inflowVelocityLength.xyz<vec3f(0.0));
  let travelPlus=activeUnpackTravel(activeScratch[ACTIVE_TRAVEL_PLUS_WORD])
    +vec3u(ceil(accelerationPlus));
  let travelMinus=activeUnpackTravel(activeScratch[ACTIVE_TRAVEL_MINUS_WORD])
    +vec3u(ceil(accelerationMinus));
  // What the host pads its lagged box with: how much further each SIDE can
  // move in one step, per axis, and their sum for the dispatch extent.
  activeScratch[ACTIVE_TRAVEL_PLUS_WORD]=activePackTravel(travelPlus);
  activeScratch[ACTIVE_TRAVEL_MINUS_WORD]=activePackTravel(travelMinus);
  activeScratch[ACTIVE_TRAVEL_TOTAL_WORD]=activePackTravel(travelPlus+travelMinus);
${geometric ? `
  // SOLVE-WINDOW PADDING, in cells, for Uniform Geometric. Per axis and per
  // SIDE, because liquid sloshing sideways must not pad the window upward and
  // falling liquid must not pad it upward either.
  //
  // The window must contain every cell that can hold liquid this step plus
  // every stencil such a cell reads. Two groups of reaches, and they do NOT
  // stack: a stencil is read AFTER the motion, so it is measured from where
  // the liquid ends up, while the two-level classes are measured from where it
  // is now. The padding is the larger of the two.
  //
  //   MOTION then stencil (these two add):
  //     travel_side: ceil(v_side dt/h) from the scan, plus this step's gravity
  //       kick and any inflow, per axis and direction ......... travelPlus/Minus
  //     largest post-motion stencil reach:
  //       uvRedistancePhi closest point: q clamped to p +- 4, uvGradient
  //         probes +- 0.25 and uvPhi taps one cell past that ........... 6
  //       uvAgreementShift tent gather: base + [-4,4) plus a tap ....... 5
  //       uvSeedPhi: base + [-2,2) plus a tap .......................... 3
  //       backward characteristics: one trilinear tap .................. 1
  //       Sec. 3.5 sharpening: eight sweeps of one-cell face exchange, each
  //         confined to |phi| < tuning.y*h ............... ceil(tuning.y) + 1
  //
  //   STANDING reach, from the liquid as it is now (no travel term):
  //     two-level SHELL tiles, the set the extension's finest passes run on,
  //       and the set the sampler is allowed to read finely: (k + s) * 4,
  //       floored at the two-coarse-cell (8 cell) trilinear tap into the 4h
  //       face table. At the shipped defaults (k=2, s=1) that is 12 cells.
  //       Exactly 12, with nothing on top: the window is rounded out to the
  //       4h lattice below, and roundDown4(m - 12) is the first cell of the
  //       tile twelve cells under the liquid's own tile, so the alignment
  //       supplies the part-tile the liquid cell sits in. Adding slack here
  //       would buy a whole extra tile of nothing.
  //
  // The E3 live transport set is deliberately NOT a term. The window is the
  // hard clip: a live tile outside it is simply not dispatched, its cells keep
  // the V they already had (zero, by the dust-floor predicate this schedule
  // requires), and no volume is created or destroyed. Its only job is to reach
  // wherever liquid can actually arrive, which is exactly travel_side.
  // Alignment to the 4h tile lattice happens below, after the union with the
  // previous box.
  let fineTiles=u32(max(params.physical.z,0.0));
  let shellTiles=fineTiles+u32(max(params.twoLevel.x,1.0));
  let standingReach=vec3u(max(4u*shellTiles,8u));
  let stencilReach=vec3u(max(6u,u32(ceil(params.tuning.y))+1u));
  // REDIRECTION, the third travel term, and the reason a purely per-direction
  // padding is not safe. Gravity is not the only acceleration a step applies:
  // where a stream meets a wall, the floor or another stream, the pressure
  // impulse turns its momentum into some other direction WITHIN the step, so
  // the velocity the scan measured on the way in says nothing about the jet
  // that leaves. What bounds that jet is the momentum arriving: a collision
  // redistributes speed, it does not manufacture it. The largest travel on any
  // axis and either side is therefore the one-step acceleration bound for
  // every direction, exactly as g*dt*dt/h is for the downward one.
  //
  // It is a FLOOR on each side's travel, not a term added to it: liquid that
  // is only sloshing sideways has a small maximum and still pads the window
  // upward by the standing reach alone, which is the case this padding exists
  // to keep small. It bites only where something is genuinely moving fast, and
  // there it is the difference between a far-wall impact jet that has valid
  // far-air velocity above it and one that does not. Measured on the tall-air
  // dam at 1x, 4x and 8x: without it the impact step reports 4.39 m/s against
  // the whole-domain control's 3.11.
  // It is the SPEED that is redirected, not one component of it, so the floor
  // is the isotropic travel the paper arm pads with -- which makes the
  // per-direction travel a term that can only ever lose to it. That is the
  // honest result: directional padding is safe for the host's LAG allowance,
  // where what grows is the box's span, and not for the window itself.
  //
  // Half again on top, because this step's measured travel is a lower bound on
  // next step's: the impulse that redirects the momentum also concentrates it.
  // Measured on the tall-air dam's far-wall impact, 8.6 cells of displacement
  // became 11.7 in the following step (x1.36). This is what the old isotropic
  // padding's unexplained +4 and the host pad's 1.5 were both standing in for,
  // and it is zero at rest, so the floor above a calm surface is unaffected.
  let redirect=travel+(travel+vec3u(1u))/vec3u(2u);
  let paddingLow=max(standingReach,max(travelMinus,redirect)+stencilReach);
  let paddingHigh=max(standingReach,max(travelPlus,redirect)+stencilReach);
` : `
  let paddingLow=travel+vec3u(u32(ceil(params.tuning.y))+4u);
  let paddingHigh=paddingLow;
`}
  let previousMinimum=vec3u(activeRegion[0],activeRegion[1],activeRegion[2]);
  let previousMaximum=vec3u(activeRegion[3],activeRegion[4],activeRegion[5]);
  var currentMinimum=previousMinimum;
  var currentMaximum=previousMaximum;
  if(all(observedMax>observedMin)){
    currentMinimum=observedMin-min(observedMin,paddingLow);
    currentMaximum=min(d,observedMax+paddingHigh);
  }
${geometric ? `
  // A 4x4x4 workgroup must still coincide with one 4h tile: the sharpening
  // work map, the two-level classes and the E3 live set all index their tile
  // by cell/4 and exit whole-workgroup on that test, which is only uniform
  // across the workgroup while the dispatch origin is a multiple of four.
  currentMinimum=currentMinimum-(currentMinimum%vec3u(4u));
  currentMaximum=min(d,currentMaximum+((vec3u(4u)-(currentMaximum%vec3u(4u)))%vec3u(4u)));
  // Snap to a domain wall the window has come within one padding of.
  // uvReleasedWalls reads the wall-plane velocity from every vertex it
  // processes; the term only reaches vertices within dt*|v|/h of the plane,
  // so with the wall inside the window whenever the liquid is within padding
  // + travel of it, a windowed vertex never reads a stale plane.
  currentMinimum=select(currentMinimum,vec3u(0u),currentMinimum<=paddingLow);
  currentMaximum=select(currentMaximum,d,currentMaximum+paddingHigh>=d);
` : ""}
  let minimum=min(previousMinimum,currentMinimum);
  let maximum=max(previousMaximum,currentMaximum);
  activeScratch[0]=currentMinimum.x;activeScratch[1]=currentMinimum.y;activeScratch[2]=currentMinimum.z;
  activeScratch[3]=currentMaximum.x;activeScratch[4]=currentMaximum.y;activeScratch[5]=currentMaximum.z;
  activeScratch[7]=minimum.x;activeScratch[8]=minimum.y;activeScratch[9]=minimum.z;
  activeScratch[10]=maximum.x;activeScratch[11]=maximum.y;activeScratch[12]=maximum.z;
  let groups=(maximum-minimum+vec3u(3u))/4u;
  activeScratch[13]=groups.x;activeScratch[14]=groups.y;activeScratch[15]=groups.z;
  // The (n+1)^3 vertex lattice the geometric phi passes run on. Vertices
  // minimum..maximum inclusive belong to the window's cells, so the dispatch
  // is one vertex wider than the cell box on every axis and shares its origin.
  let vertexGroups=(maximum-minimum+vec3u(1u)+vec3u(3u))/4u;
  activeScratch[ACTIVE_VERTEX_DISPATCH_WORD]=vertexGroups.x;
  activeScratch[ACTIVE_VERTEX_DISPATCH_WORD+1u]=vertexGroups.y;
  activeScratch[ACTIVE_VERTEX_DISPATCH_WORD+2u]=vertexGroups.z;
  // Containment check for host-sized direct dispatches. The origin every kernel
  // reads is the exact one computed above; only the group COUNT comes from the
  // host, which chose it from a box two or three steps old. A step whose exact
  // extent outgrew that choice is clipped -- the far edge of the window simply
  // is not dispatched -- so each such step is counted here and the host answers
  // by going back to whole-domain counts for a while.
  var violationAxes=0u;
  if(activeScratch[ACTIVE_CPU_MODE_WORD]==1u){
    if(groups.x>activeScratch[ACTIVE_CPU_MAIN_WORD]){violationAxes|=1u;}
    if(groups.y>activeScratch[ACTIVE_CPU_MAIN_WORD+1u]){violationAxes|=2u;}
    if(groups.z>activeScratch[ACTIVE_CPU_MAIN_WORD+2u]){violationAxes|=4u;}
    if(vertexGroups.x>activeScratch[ACTIVE_CPU_VERTEX_WORD]){violationAxes|=8u;}
    if(vertexGroups.y>activeScratch[ACTIVE_CPU_VERTEX_WORD+1u]){violationAxes|=16u;}
    if(vertexGroups.z>activeScratch[ACTIVE_CPU_VERTEX_WORD+2u]){violationAxes|=32u;}
  }
  for(var level=0u;level<16u;level+=1u){
    let base=16u+10u*level;
    let physical=vec3u(activeScratch[base+6u],activeScratch[base+7u],activeScratch[base+8u]);
    if(any(physical==vec3u(0u))){break;}
    let scaledMin=(minimum*physical)/d;
    let scaledMax=vec3u(activeCeilDiv(maximum.x*physical.x,d.x),activeCeilDiv(maximum.y*physical.y,d.y),activeCeilDiv(maximum.z*physical.z,d.z));
    let origin=scaledMin-min(scaledMin,vec3u(2u));
    let end=min(physical+vec3u(2u),scaledMax+vec3u(3u));
    let levelGroups=(end-origin+vec3u(3u))/4u;
    activeScratch[base]=origin.x;activeScratch[base+1u]=origin.y;activeScratch[base+2u]=origin.z;
    activeScratch[base+3u]=levelGroups.x;activeScratch[base+4u]=levelGroups.y;activeScratch[base+5u]=levelGroups.z;
    // The exact extent this level owns, for the clip-mask arms.
    activeScratch[base+9u]=activePackExtent(end-origin);
    if(activeScratch[ACTIVE_CPU_MODE_WORD]==1u){
      let cpuBase=ACTIVE_CPU_LEVEL_BASE_WORD+3u*level;
      if(levelGroups.x>activeScratch[cpuBase]||levelGroups.y>activeScratch[cpuBase+1u]
        ||levelGroups.z>activeScratch[cpuBase+2u]){violationAxes|=64u;}
    }
  }
  // The window-local CM11a lattice the host chose has to contain this step's
  // exact box, for the same reason and with the same answer as the counts
  // above: it was chosen from a lagged box, and a step it does not cover is
  // counted here and answered with whole-domain steps.
  if(activeScratch[ACTIVE_PRESSURE_MODE_WORD]==1u){
    let latticeOrigin=vec3u(activeScratch[ACTIVE_PRESSURE_ORIGIN_WORD],
      activeScratch[ACTIVE_PRESSURE_ORIGIN_WORD+1u],activeScratch[ACTIVE_PRESSURE_ORIGIN_WORD+2u]);
    let latticeCapacity=vec3u(activeScratch[ACTIVE_PRESSURE_CAPACITY_WORD],
      activeScratch[ACTIVE_PRESSURE_CAPACITY_WORD+1u],activeScratch[ACTIVE_PRESSURE_CAPACITY_WORD+2u]);
    if(any(minimum<latticeOrigin)||any(maximum>latticeOrigin+latticeCapacity)){violationAxes|=128u;}
  }
  activeScratch[ACTIVE_VIOLATION_WORD]=activeRegion[ACTIVE_VIOLATION_WORD]
    +select(0u,1u,violationAxes!=0u);
  activeScratch[ACTIVE_VIOLATION_AXES_WORD]=violationAxes;
}
@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=activeId(gid);if(!valid(id)){return;}let represented=surfaceOccupancy(id);let conservative=volume(id);atomicAdd(&reductions[0],u32(represented*2048.0+0.5));if(surfaceLiquid(id)){atomicMax(&reductions[1],u32(id.x+1));}let speed=length(faceVelocity(id));atomicMax(&reductions[2],bitcast<u32>(speed));atomicAdd(&reductions[3],u32(${geometric ? "max(conservative,0.0)" : "clamp(conservative,0.0,8.0)"}*2048.0+0.5));}
${uniformPageDomainWGSL(domain)}
${geometric ? `fn uvDonorId(g:vec3u)->vec3i{return ${domain ? "pageDomainCell(g)" : "vec3i(g)"};}\n` + uniformVolumePagesWGSL(pages) + uniformVolumeWGSL : ""}
`; }

export const uniformReferenceComputeShader = createUniformReferenceComputeShader();
