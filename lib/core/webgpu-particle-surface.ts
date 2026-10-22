import type { GPUFluidParticleSource } from "./webgpu-particle-overlay";

/**
 * Meshes a fine, purely cosmetic liquid skin from a particle method's own
 * particles, every frame, for the Fine particle display.
 *
 * The level set is smoothed at h and 4h, so it drops the sheets, crowns and
 * ripples the particles still carry. This builds the skin from the particles
 * instead. Fine samples sit at h/2, the seed spacing, in 8³-cell bricks that a
 * GPU hash of brick coordinates maps into a fixed pool. The pool is rebuilt
 * from the particles each time, so the skin is as unbounded as they are and
 * keeps nothing from one frame to the next.
 *
 * The field is Zhu & Bridson's (2005): φ = |x − x̄| − r, x̄ the kernel-weighted
 * mean of the particles around x. A flat lattice gives a flat skin, where summed
 * blobs would show the seed lattice and strobe as it is carried across the
 * samples. Under that field a lone particle would be a ball, so the radius is
 * gated by the summed kernel weight: below the Greed threshold a cluster is too
 * thin to be liquid, meshes as nothing, and is drawn as spray by the overlay.
 *
 * One workgroup meshes one brick. It splats its 27 neighbours' particles into a
 * 10³ sample window with workgroup atomics in fixed point, so the sum does not
 * depend on the order the particles came in and the samples two bricks share
 * agree exactly. It then places one surface-nets vertex per cell and emits the
 * quads of the edges it owns as two non-indexed triangles each, in the water's
 * own SurfaceVertex layout. The water draws them in its interface passes beside
 * the level-set mesh, so Shade, Simple and Wire all apply to the skin.
 *
 * Particles exist only in a narrow band, and only where the grid keeps them (an
 * h tile, not a 4h one), so no particles is not the same as air. Each record
 * carries its depth under the level set. Their weighted mean gives each sample
 * an estimate of the level set's distance, and φ is the union of the particles'
 * field with it. The band's floor, voids between sparse particles, and the wall
 * where particle tiles end are liquid in that estimate, so none of them becomes
 * a back face for the water's interface passes. The estimate reaches only as far
 * as the particles' support. An edge leaving the support from a sample that is
 * liquid by the estimate alone is the edge of what the particles cover, not of
 * the liquid, so it is not emitted.
 *
 * Every pass is a fixed direct dispatch that grid-strides over its GPU list.
 */

/** Settings the Fine display's sliders drive; uniforms only, never a rebuild of a pipeline. */
export interface ParticleSurfaceSettings {
  /** 0 meshes only bulk liquid, 1 meshes nearly every pair of particles. */
  readonly greed: number;
  /** Kernel radius in cells (h). */
  readonly smoothing: number;
  /** Surface radius about the weighted mean, in cells (h). */
  readonly thickness: number;
  /** Spray footprint radius as a fraction of a Motion sample's. */
  readonly spraySize: number;
  /** Scales the spray's Motion opacity. */
  readonly sprayOpacity: number;
  /** In cells (h): the skin shows only where it stands this far in front of the level set. */
  readonly merge: number;
}

export const DEFAULT_PARTICLE_SURFACE_SETTINGS: ParticleSurfaceSettings = Object.freeze({
  greed: 0.5, smoothing: 1, thickness: 0.5, spraySize: 2 / 3, sprayOpacity: 1, merge: 0.5,
});
export const PARTICLE_SURFACE_SMOOTHING_RANGE = [0.5, 1.5] as const;
export const PARTICLE_SURFACE_THICKNESS_RANGE = [0.2, 1] as const;
export const PARTICLE_SURFACE_MERGE_RANGE = [0, 2] as const;

/** Fine samples per cell. NB-FLIP seeds 2³ per cell, so this is its spacing. */
const FINE_PER_CELL = 2;
/** Particles per cell in the bulk: the weight a full kernel sums to is scaled from it. */
const SEEDS_PER_CELL = 8;
export const PARTICLE_SURFACE_BRICK_CAPACITY = 32_768;
const HASH_SIZE = 2 * PARTICLE_SURFACE_BRICK_CAPACITY;
/** Two triangles per quad: a multiple of six, so the pool never ends mid-quad. */
export const PARTICLE_SURFACE_VERTEX_CAPACITY = 6 * 349_525;
const SURFACE_VERTEX_BYTES = 32;
const PARTICLE_WORKGROUPS = 512;
const BRICK_WORKGROUPS = 1024;
/** acc 16000 + depth 4000 + phi 4000 + cell vertices and normals 5832 + neighbour lists 324 + scalars. */
export const PARTICLE_SURFACE_WORKGROUP_BYTES = 30_188;
/** Mesh workgroup width; 512 beat 256 and 1024 on M1 Max (1.86 vs 2.29 and 2.42 ms). */
export const PARTICLE_SURFACE_MESH_LANES = 512;
const UNIFORM_BYTES = 96;
/** Counter words: live, bricks, vertices, overflow; then the indirect draw at word 8. */
const COUNTER_BYTES = 48;
export const PARTICLE_SURFACE_INDIRECT_OFFSET = 32;

/** Overflow bits in counter word 3. */
export const PARTICLE_SURFACE_OVERFLOW = Object.freeze({ bricks: 1, hash: 2, vertices: 4, rank: 8 });

export const particleSurfaceShader = /* wgsl */ `
struct Params {
  // xyz metres per position unit, w fine samples per position unit
  scale:vec4f,
  // xyz world position of the tank's minimum corner, w one when the frame's origin is live
  origin:vec4f,
  // x kernel radius, y surface radius, z the weight under which the radius is gated off (fine units),
  // w unused
  kernel:vec4f,
  // x record stride in floats, y record capacity, z ballistic float offset (top bit: the flag
  // is the sign bit), w depth float offset or 0xffffffff where records keep none
  records:vec4u,
  // x brick capacity, y hash size, z vertex capacity, w particle capacity
  pool:vec4u,
  // x the merge distance in metres, carried by every skin vertex as -position.w
  skin:vec4f,
}
struct ParticleFrame{dims:vec4f,h:vec4f,origin:vec4f}
struct SurfaceVertex { position:vec4f, normal:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<uniform> particleFrame:ParticleFrame;
@group(0) @binding(2) var<storage,read> particles:array<f32>;
// live count, brick count, vertex count, overflow bits; the indirect draw at word 8.
@group(0) @binding(3) var<storage,read_write> counters:array<atomic<u32>,12>;
// Keys in [0, hash size), the slot each key compacted to in [hash size, 2 hash size).
@group(0) @binding(4) var<storage,read_write> hash:array<atomic<u32>>;
// Brick keys, particle counts and particle offsets, each over the brick capacity.
@group(0) @binding(5) var<storage,read_write> bricks:array<atomic<u32>>;
// Per record: home slot | rank << 16; then per sorted particle: its record.
@group(0) @binding(6) var<storage,read_write> work:array<u32>;
// Brick-ordered particles: fine position, and the depth under the level set in fine units.
@group(0) @binding(7) var<storage,read_write> sorted:array<vec4f>;
// φ at each record in fine units: zero or more outside the skin.
@group(0) @binding(8) var<storage,read_write> field:array<f32>;
@group(0) @binding(9) var<storage,read_write> vertices:array<SurfaceVertex>;

const EMPTY:u32=0u;
const NONE:u32=0xffffffffu;
const OUTSIDE:f32=1e4;
const BRICK:i32=8;
// The window a brick splats into: its own samples 0..7 plus one either side.
const SPAN:i32=10;
const CELLS:i32=9;
const FIXED:f32=65536.0;
// The depth a record lends the level-set estimate, in fine units: a band's
// worth under the surface at most, and a ballistic record is in the air.
const DEPTH_FLOOR:f32=-8.0;
const DEPTH_AIR:f32=4.0;

fn finePosition(at:u32)->vec3f { return vec3f(particles[at],particles[at+1u],particles[at+2u])*params.scale.w; }
fn liveRecords()->u32 { return min(atomicLoad(&counters[0]),params.records.y); }
fn brickCount()->u32 { return min(atomicLoad(&counters[1]),params.pool.x); }
fn hashMask()->u32 { return params.pool.y-1u; }

// Ten bits an axis about 512 bricks (2048 cells either side), plus one so that
// zero is the empty key. Out of range is EMPTY: the brick is not claimed.
fn brickKey(b:vec3i)->u32 {
  let o=b+vec3i(512);
  if(any(o<vec3i(0))||any(o>vec3i(1023))){return EMPTY;}
  return (u32(o.x)|(u32(o.y)<<10u)|(u32(o.z)<<20u))+1u;
}
fn keyBrick(key:u32)->vec3i {
  let k=key-1u;
  return vec3i(i32(k&1023u),i32((k>>10u)&1023u),i32(k>>20u))-vec3i(512);
}
fn hashOf(key:u32)->u32 {
  var x=key*0x9e3779b1u;
  x^=x>>15u;
  return x&hashMask();
}
fn insertKey(key:u32) {
  var at=hashOf(key);
  var probes=0u;
  loop {
    if(probes>=128u){atomicOr(&counters[3],${PARTICLE_SURFACE_OVERFLOW.hash}u);return;}
    // Most claims find their key already there: read before contending for it.
    let seen=atomicLoad(&hash[at]);
    if(seen==key){return;}
    if(seen==EMPTY){
      let swap=atomicCompareExchangeWeak(&hash[at],EMPTY,key);
      if(swap.exchanged||swap.old_value==key){return;}
      // A weak exchange may fail with the slot still empty: try it again.
      if(swap.old_value==EMPTY){continue;}
    }
    at=(at+1u)&hashMask();
    probes+=1u;
  }
}
fn lookupSlot(key:u32)->u32 {
  if(key==EMPTY){return NONE;}
  var at=hashOf(key);
  for(var probes=0u;probes<128u;probes+=1u){
    let seen=atomicLoad(&hash[at]);
    if(seen==key){return atomicLoad(&hash[params.pool.y+at]);}
    if(seen==EMPTY){return NONE;}
    at=(at+1u)&hashMask();
  }
  return NONE;
}
fn brickOf(fine:vec3f)->vec3i { return vec3i(floor(fine/f32(BRICK))); }

// Every brick whose samples the particle's kernel reaches, and one sample
// further toward -xyz: a sign change between a sample it reaches and the one
// before is an edge that sample's brick owns.
@compute @workgroup_size(256) fn claimMain(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u) {
  let live=liveRecords();
  let reach=params.kernel.x;
  for(var i=id.x;i<live;i+=groups.x*256u){
    let at=i*params.records.x;
    if(!(particles[at+3u]>0.0)){continue;}
    let p=finePosition(at);
    let lo=brickOf(p-vec3f(reach+1.0));
    let hi=brickOf(p+vec3f(reach));
    for(var z=lo.z;z<=hi.z;z+=1){for(var y=lo.y;y<=hi.y;y+=1){for(var x=lo.x;x<=hi.x;x+=1){
      let key=brickKey(vec3i(x,y,z));
      if(key==EMPTY){atomicOr(&counters[3],${PARTICLE_SURFACE_OVERFLOW.hash}u);continue;}
      insertKey(key);
    }}}
  }
}

// Claimed keys take pool slots. Slots are handed out here rather than at
// insertion, so no lookup ever finds a key whose slot is still being written.
@compute @workgroup_size(256) fn compactMain(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u) {
  for(var at=id.x;at<params.pool.y;at+=groups.x*256u){
    let key=atomicLoad(&hash[at]);
    if(key==EMPTY){continue;}
    let slot=atomicAdd(&counters[1],1u);
    if(slot<params.pool.x){
      atomicStore(&hash[params.pool.y+at],slot);
      atomicStore(&bricks[slot],key);
    } else {
      atomicStore(&hash[params.pool.y+at],NONE);
      atomicOr(&counters[3],${PARTICLE_SURFACE_OVERFLOW.bricks}u);
    }
  }
}

// Each particle's rank in its home brick, and the outside default for the
// overlay's φ, which the brick that meshes it overwrites.
@compute @workgroup_size(256) fn countMain(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u) {
  let live=liveRecords();
  for(var i=id.x;i<live;i+=groups.x*256u){
    field[i]=OUTSIDE;
    work[i]=NONE;
    let at=i*params.records.x;
    if(!(particles[at+3u]>0.0)){continue;}
    let slot=lookupSlot(brickKey(brickOf(finePosition(at))));
    if(slot==NONE){continue;}
    let rank=atomicAdd(&bricks[params.pool.x+slot],1u);
    if(rank>=65536u){atomicOr(&counters[3],${PARTICLE_SURFACE_OVERFLOW.rank}u);continue;}
    work[i]=slot|(rank<<16u);
  }
}

var<workgroup> partial:array<u32,256>;
// The exclusive prefix of the brick counts, in one workgroup.
@compute @workgroup_size(256) fn scanMain(@builtin(local_invocation_index) lane:u32) {
  let count=brickCount();
  let chunk=(count+255u)/256u;
  let begin=min(lane*chunk,count);
  let end=min(begin+chunk,count);
  var sum=0u;
  for(var b=begin;b<end;b+=1u){sum+=atomicLoad(&bricks[params.pool.x+b]);}
  partial[lane]=sum;
  workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){
    var before=0u;
    if(lane>=offset){before=partial[lane-offset];}
    workgroupBarrier();
    partial[lane]+=before;
    workgroupBarrier();
  }
  var running=partial[lane]-sum;
  for(var b=begin;b<end;b+=1u){
    atomicStore(&bricks[2u*params.pool.x+b],running);
    running+=atomicLoad(&bricks[params.pool.x+b]);
  }
}

// The record's signed distance to the level set in fine units, negative under
// it. Records that keep no depth, and ballistic ones, are in the air.
fn levelDepth(at:u32)->f32 {
  if(params.records.w==NONE){return DEPTH_AIR;}
  let flagAt=params.records.z&0x7fffffffu;
  let flag=particles[at+flagAt];
  let ballistic=select(flag==1.0,(bitcast<u32>(flag)>>31u)!=0u,(params.records.z&0x80000000u)!=0u);
  if(ballistic){return DEPTH_AIR;}
  return clamp(particles[at+params.records.w]*params.scale.w,DEPTH_FLOOR,DEPTH_AIR);
}

@compute @workgroup_size(256) fn scatterMain(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u) {
  let live=liveRecords();
  for(var i=id.x;i<live;i+=groups.x*256u){
    let home=work[i];
    if(home==NONE){continue;}
    let slot=home&0xffffu;
    let destination=atomicLoad(&bricks[2u*params.pool.x+slot])+(home>>16u);
    let at=i*params.records.x;
    sorted[destination]=vec4f(finePosition(at),levelDepth(at));
    work[params.pool.w+destination]=i;
  }
}

// Wider than the other passes: a brick's splat is latency-bound on workgroup
// atomics and the 30 KB window allows one workgroup per core.
const MESH_LANES=${PARTICLE_SURFACE_MESH_LANES}u;
var<workgroup> acc:array<atomic<i32>,4000>;
// Σ w·depth: with acc's weight, the level set's distance near each sample.
var<workgroup> depthSum:array<atomic<i32>,1000>;
var<workgroup> phi:array<f32,1000>;
var<workgroup> cellVertex:array<u32,729>;
var<workgroup> cellNormalPacked:array<u32,729>;
var<workgroup> neighbours:array<vec2u,27>;
var<workgroup> listEnd:array<u32,27>;
var<workgroup> bricksToMesh:u32;
var<workgroup> quadTotal:atomic<u32>;
var<workgroup> quadBase:u32;

fn sampleIndex(s:vec3i)->i32 { return s.x+SPAN*(s.y+SPAN*s.z); }
fn cellIndex(c:vec3i)->i32 { return c.x+CELLS*(c.y+CELLS*c.z); }

// φ at a window position, trilinear.
fn supported(s:i32)->bool { return atomicLoad(&acc[4*s])>0; }
fn meanOffset(s:i32)->vec3f {
  return vec3f(f32(atomicLoad(&acc[4*s+1])),f32(atomicLoad(&acc[4*s+2])),f32(atomicLoad(&acc[4*s+3])))/f32(atomicLoad(&acc[4*s]));
}
fn meanDepth(s:i32)->f32 { return f32(atomicLoad(&depthSum[s]))/f32(atomicLoad(&acc[4*s])); }

// Zhu–Bridson: the distance to the particles' weighted mean, less a radius
// that is gated off where too little weight gathers. Supported samples only.
fn particlePhi(s:i32)->f32 {
  let gate=params.kernel.z;
  let radius=params.kernel.y*smoothstep(0.5*gate,gate,f32(atomicLoad(&acc[4*s]))/FIXED);
  return length(meanOffset(s))-radius;
}

// The level set's distance at a supported sample. The mean depth is its value at
// the particles' weighted mean, so it is carried back along the normal, which is
// the depth's gradient, over the offset to that mean.
fn levelPhi(c:vec3i)->f32 {
  let s=sampleIndex(c);
  let depth=meanDepth(s);
  var gradient=vec3f(0.0);
  for(var axis=0;axis<3;axis+=1){
    let e=axisUnit(axis);
    let lo=c-e;
    let hi=c+e;
    let hasLo=all(lo>=vec3i(0))&&supported(sampleIndex(lo));
    let hasHi=all(hi<vec3i(SPAN))&&supported(sampleIndex(hi));
    if(hasLo&&hasHi){gradient[axis]=0.5*(meanDepth(sampleIndex(hi))-meanDepth(sampleIndex(lo)));}
    else if(hasHi){gradient[axis]=meanDepth(sampleIndex(hi))-depth;}
    else if(hasLo){gradient[axis]=depth-meanDepth(sampleIndex(lo));}
  }
  if(dot(gradient,gradient)<1e-8){return depth;}
  return depth-dot(meanOffset(s),normalize(gradient));
}

fn phiAt(local:vec3f)->f32 {
  let x=clamp(local,vec3f(0.0),vec3f(f32(SPAN-1)-1e-3));
  let c=vec3i(floor(x));
  let t=x-vec3f(c);
  var value=0.0;
  for(var k=0;k<8;k+=1){
    let o=vec3i(k&1,(k>>1)&1,(k>>2)&1);
    let w=mix(1.0-t,t,vec3f(o));
    value+=w.x*w.y*w.z*phi[sampleIndex(c+o)];
  }
  return value;
}

// The outward normal of φ inside a cell, at a fraction across it.
fn cellNormal(c:vec3i,t:vec3f)->vec3f {
  var gradient=vec3f(0.0);
  for(var k=0;k<8;k+=1){
    let o=vec3i(k&1,(k>>1)&1,(k>>2)&1);
    let value=phi[sampleIndex(c+o)];
    let w=mix(1.0-t,t,vec3f(o));
    let side=vec3f(o)*2.0-1.0;
    gradient+=value*side*vec3f(w.y*w.z,w.x*w.z,w.x*w.y);
  }
  // Positions are anisotropic only through the per-axis cell size.
  gradient/=params.scale.xyz;
  // A crossed cell has a gradient; up is only a guard against a degenerate one.
  if(dot(gradient,gradient)<1e-20){return vec3f(0.0,1.0,0.0);}
  return normalize(gradient);
}

// Ten bits an axis: a fraction across the cell, or a unit normal's component.
fn pack10(v:vec3f)->u32 {
  let q=vec3u(round(clamp(v,vec3f(0.0),vec3f(1.0))*1023.0));
  return q.x|(q.y<<10u)|(q.z<<20u);
}
fn unpack10(packed:u32)->vec3f {
  return vec3f(f32(packed&1023u),f32((packed>>10u)&1023u),f32((packed>>20u)&1023u))/1023.0;
}

fn axisUnit(axis:i32)->vec3i { return vec3i(select(0,1,axis==0),select(0,1,axis==1),select(0,1,axis==2)); }

fn worldOf(fine:vec3f)->vec3f {
  let origin=select(params.origin.xyz,particleFrame.origin.xyz,params.origin.w>0.5);
  return fine/params.scale.w*params.scale.xyz+origin;
}

fn crossesAndKept(q:vec3i,axis:i32)->bool {
  let a=sampleIndex(q);
  let b=sampleIndex(q+axisUnit(axis));
  let aInside=phi[a]<0.0;
  if(aInside==(phi[b]<0.0)){return false;}
  let outside=select(a,b,aInside);
  if(supported(outside)){return true;}
  return particlePhi(select(b,a,aInside))<0.0;
}

// The q-th edge a lane owns: a sample of the brick's own 8³ and an axis.
fn ownedEdge(index:u32)->vec4i {
  let owned=i32(index/3u);
  let axis=i32(index%3u);
  return vec4i(1+owned%BRICK,1+(owned/BRICK)%BRICK,1+owned/(BRICK*BRICK),axis);
}

// One of the four cells around an owned edge, counter-clockwise seen from +axis.
fn quadCell(q:vec3i,axis:i32,k:i32)->vec3i {
  return q-axisUnit((axis+1)%3)*select(1,0,k==1||k==2)-axisUnit((axis+2)%3)*select(1,0,k>=2);
}

fn quadVertex(window:vec3i,cell:vec3i)->SurfaceVertex {
  let index=cellIndex(cell);
  let fine=vec3f(window+cell)+unpack10(cellVertex[index]);
  // position.w is 1 on the level set's vertices. A skin vertex carries minus the
  // merge distance, which the water's raster pushes it back along its view ray.
  return SurfaceVertex(vec4f(worldOf(fine),-params.skin.x),vec4f(unpack10(cellNormalPacked[index])*2.0-1.0,0.0));
}

// The quad's cells and normals were placed once per cell; this only orders
// them. Flipped when the liquid lies on the + side, so every triangle's front
// faces the air.
fn emitQuad(window:vec3i,q:vec3i,axis:i32,at:u32) {
  let v0=quadVertex(window,quadCell(q,axis,0));
  let v1=quadVertex(window,quadCell(q,axis,1));
  let v2=quadVertex(window,quadCell(q,axis,2));
  let v3=quadVertex(window,quadCell(q,axis,3));
  vertices[at]=v0;
  vertices[at+3u]=v0;
  if(phi[sampleIndex(q)]<0.0){
    vertices[at+1u]=v1;vertices[at+2u]=v2;vertices[at+4u]=v2;vertices[at+5u]=v3;
  } else {
    vertices[at+1u]=v2;vertices[at+2u]=v1;vertices[at+4u]=v3;vertices[at+5u]=v2;
  }
}

fn meshBrick(slot:u32,lane:u32) {
  let brick=keyBrick(atomicLoad(&bricks[slot]));
  // Window sample 0 is brick sample -1.
  let window=brick*BRICK-vec3i(1);
  for(var s=lane;s<4000u;s+=MESH_LANES){atomicStore(&acc[s],0);}
  for(var s=lane;s<1000u;s+=MESH_LANES){atomicStore(&depthSum[s],0);}
  if(lane==0u){atomicStore(&quadTotal,0u);}
  if(lane<27u){
    let d=vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-vec3i(1);
    let neighbour=lookupSlot(brickKey(brick+d));
    if(neighbour==NONE){neighbours[lane]=vec2u(0u);}
    else{neighbours[lane]=vec2u(atomicLoad(&bricks[2u*params.pool.x+neighbour]),atomicLoad(&bricks[params.pool.x+neighbour]));}
  }
  workgroupBarrier();
  if(lane==0u){
    var end=0u;
    for(var n=0u;n<27u;n+=1u){end+=neighbours[n].y;listEnd[n]=end;}
  }
  workgroupBarrier();

  let reach=params.kernel.x;
  let reach2=reach*reach;
  // The 27 lists are one flat range, so every lane has a particle; a brick's
  // own list alone fills an eighth of the workgroup.
  let total=listEnd[26];
  for(var k=lane;k<total;k+=MESH_LANES){
    {
      var n=0u;
      for(var step=16u;step>0u;step>>=1u){
        if(n+step<=26u&&listEnd[n+step-1u]<=k){n+=step;}
      }
      let list=neighbours[n];
      let particle=sorted[list.x+k-(listEnd[n]-list.y)];
      let local=particle.xyz-vec3f(window);
      let lo=max(vec3i(ceil(local-vec3f(reach))),vec3i(0));
      let hi=min(vec3i(floor(local+vec3f(reach))),vec3i(SPAN-1));
      if(any(lo>hi)){continue;}
      // Only the samples inside the kernel's sphere: each row's span is solved
      // for, so lanes do not idle through the corners of the box.
      for(var z=lo.z;z<=hi.z;z+=1){
        let dz=local.z-f32(z);
        for(var y=lo.y;y<=hi.y;y+=1){
        let dy=local.y-f32(y);
        let row=reach2-dz*dz-dy*dy;
        if(row<=0.0){continue;}
        let span=sqrt(row);
        let x0=max(i32(ceil(local.x-span)),lo.x);
        let x1=min(i32(floor(local.x+span)),hi.x);
        for(var x=x0;x<=x1;x+=1){
        let offset=vec3f(local.x-f32(x),dy,dz);
        let r2=dot(offset,offset);
        if(r2>=reach2){continue;}
        let t=1.0-r2/reach2;
        let w=t*t*t;
        let s=sampleIndex(vec3i(x,y,z));
        atomicAdd(&acc[4*s],i32(round(w*FIXED)));
        atomicAdd(&acc[4*s+1],i32(round(w*offset.x*FIXED)));
        atomicAdd(&acc[4*s+2],i32(round(w*offset.y*FIXED)));
        atomicAdd(&acc[4*s+3],i32(round(w*offset.z*FIXED)));
        atomicAdd(&depthSum[s],i32(round(w*particle.w*FIXED)));
      }}}
    }
  }
  workgroupBarrier();

  for(var s=lane;s<1000u;s+=MESH_LANES){
    let c=vec3i(i32(s%10u),i32((s/10u)%10u),i32(s/100u));
    let at=sampleIndex(c);
    // Outside the particles' support nothing is known: air, but see crossesAndKept.
    if(!supported(at)){phi[s]=reach;continue;}
    phi[s]=min(particlePhi(at),levelPhi(c));
  }
  workgroupBarrier();

  // Surface nets: one vertex per crossed cell, the mean of its edge crossings.
  for(var cell=lane;cell<729u;cell+=MESH_LANES){
    let c=vec3i(i32(cell%9u),i32((cell/9u)%9u),i32(cell/81u));
    var corner:array<f32,8>;
    var inside=0u;
    for(var k=0;k<8;k+=1){
      corner[k]=phi[sampleIndex(c+vec3i(k&1,(k>>1)&1,(k>>2)&1))];
      if(corner[k]<0.0){inside+=1u;}
    }
    if(inside==0u||inside==8u){cellVertex[cell]=NONE;continue;}
    var sum=vec3f(0.0);
    var crossings=0.0;
    for(var k=0;k<8;k+=1){
      for(var axis=0;axis<3;axis+=1){
        if(((k>>u32(axis))&1)!=0){continue;}
        let other=k|(1<<u32(axis));
        let a=corner[k];
        let b=corner[other];
        if((a<0.0)==(b<0.0)){continue;}
        var p=vec3f(f32(k&1),f32((k>>1)&1),f32((k>>2)&1));
        p[axis]=a/(a-b);
        sum+=p;
        crossings+=1.0;
      }
    }
    let t=clamp(sum/crossings,vec3f(0.0),vec3f(1.0));
    cellVertex[cell]=pack10(t);
    // Stored as placed, so the vertex every quad shares carries one normal.
    cellNormalPacked[cell]=pack10(0.5+0.5*cellNormal(c,unpack10(cellVertex[cell])));
  }
  workgroupBarrier();

  var mine=0u;
  for(var index=lane;index<1536u;index+=MESH_LANES){
    let edge=ownedEdge(index);
    if(crossesAndKept(edge.xyz,edge.w)){mine+=1u;}
  }
  let first=atomicAdd(&quadTotal,mine);
  workgroupBarrier();
  if(lane==0u){
    let quads=atomicLoad(&quadTotal);
    quadBase=select(0u,atomicAdd(&counters[2],6u*quads),quads>0u);
  }
  let base=workgroupUniformLoad(&quadBase);
  var written=0u;
  for(var index=lane;index<1536u;index+=MESH_LANES){
    let edge=ownedEdge(index);
    if(!crossesAndKept(edge.xyz,edge.w)){continue;}
    let at=base+6u*(first+written);
    written+=1u;
    if(at+6u>params.pool.z){atomicOr(&counters[3],${PARTICLE_SURFACE_OVERFLOW.vertices}u);continue;}
    emitQuad(window,edge.xyz,edge.w,at);
  }

  // φ at the brick's own particles: zero or more is spray for the overlay.
  let home=neighbours[13];
  for(var k=lane;k<home.y;k+=MESH_LANES){
    let particle=sorted[home.x+k];
    field[work[params.pool.w+home.x+k]]=phiAt(particle.xyz-vec3f(window));
  }
  workgroupBarrier();
}

@compute @workgroup_size(MESH_LANES) fn meshMain(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u) {
  if(lane==0u){bricksToMesh=brickCount();}
  let count=workgroupUniformLoad(&bricksToMesh);
  for(var slot=group.x;slot<count;slot+=groups.x){meshBrick(slot,lane);}
}

@compute @workgroup_size(1) fn finishMain() {
  let emitted=atomicLoad(&counters[2]);
  atomicStore(&counters[8],min(emitted,params.pool.z));
  atomicStore(&counters[9],1u);
  atomicStore(&counters[10],0u);
  atomicStore(&counters[11],0u);
}
`;

export interface ParticleSurfaceGeometry {
  readonly vertices: GPUBuffer;
  readonly indirect: GPUBuffer;
  readonly indirectOffset: number;
}

export interface ParticleSurfaceFrame {
  readonly container_m: readonly [number, number, number];
  readonly settings: ParticleSurfaceSettings;
  /** Changes whenever the particles may have moved; equal keys skip the rebuild. */
  readonly revision: number;
}

export interface ParticleSurfaceStats {
  readonly bricks: number;
  readonly vertices: number;
  /** PARTICLE_SURFACE_OVERFLOW bits. */
  readonly overflow: number;
}

/** The per-frame chain and the pool it fills. */
export class ParticleSurface {
  /** The chain in order, each pass with its fixed workgroup count. */
  private chain?: ReadonlyArray<readonly [GPUComputePipeline, number]>;
  private layout?: GPUBindGroupLayout;
  private uniforms?: GPUBuffer;
  private counters?: GPUBuffer;
  private hash?: GPUBuffer;
  private bricks?: GPUBuffer;
  private vertexPool?: GPUBuffer;
  private work?: GPUBuffer;
  private sorted?: GPUBuffer;
  private fieldBuffer?: GPUBuffer;
  private particleCapacity = 0;
  private fallbackFrame?: GPUBuffer;
  private group?: GPUBindGroup;
  private groupKey?: { particles: GPUBuffer; frame: GPUBuffer | undefined };
  /** What the pool was last built from: revision, greed, smoothing, thickness, container x, z. */
  private readonly built = new Float64Array(7).fill(NaN);
  private readonly liveWord = new Uint32Array(1);
  private statsReadback?: GPUBuffer;
  private statsPending = false;
  private reportedOverflow = 0;
  private lastStats?: ParticleSurfaceStats;
  private destroyed = false;
  private readonly uniformData = new ArrayBuffer(UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(private readonly device: GPUDevice) {}

  async initialize(): Promise<void> {
    const granted = this.device.limits.maxComputeWorkgroupStorageSize;
    if (granted < PARTICLE_SURFACE_WORKGROUP_BYTES) {
      throw new Error(`Fine particle surface needs ${PARTICLE_SURFACE_WORKGROUP_BYTES} B of workgroup storage per brick; the device grants ${granted} B.`);
    }
    const lanes = Math.min(this.device.limits.maxComputeInvocationsPerWorkgroup, this.device.limits.maxComputeWorkgroupSizeX);
    if (lanes < PARTICLE_SURFACE_MESH_LANES) {
      throw new Error(`Fine particle surface needs ${PARTICLE_SURFACE_MESH_LANES}-lane workgroups per brick; the device grants ${lanes}.`);
    }
    const module = this.device.createShaderModule({ label: "Fine particle surface", code: particleSurfaceShader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Fine particle surface:\n${errors.map(error => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
    }
    const storage = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
    this.layout = this.device.createBindGroupLayout({
      label: "Fine particle surface bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        storage(3), storage(4), storage(5), storage(6), storage(7), storage(8), storage(9),
      ],
    });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const compute = (entryPoint: string) => this.device.createComputePipelineAsync({
      label: `Fine particle surface ${entryPoint}`, layout, compute: { module, entryPoint },
    });
    const [claim, compact, count, scan, scatter, mesh, finish] = await Promise.all(
      ["claimMain", "compactMain", "countMain", "scanMain", "scatterMain", "meshMain", "finishMain"].map(compute));
    this.chain = [
      [claim!, PARTICLE_WORKGROUPS], [compact!, HASH_SIZE / 256], [count!, PARTICLE_WORKGROUPS],
      [scan!, 1], [scatter!, PARTICLE_WORKGROUPS], [mesh!, BRICK_WORKGROUPS], [finish!, 1],
    ];
    this.uniforms = this.device.createBuffer({ label: "Fine particle surface uniforms", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.counters = this.device.createBuffer({
      label: "Fine particle surface counters and draw arguments", size: COUNTER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.hash = this.device.createBuffer({ label: "Fine particle surface brick hash", size: 8 * HASH_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bricks = this.device.createBuffer({ label: "Fine particle surface bricks", size: 12 * PARTICLE_SURFACE_BRICK_CAPACITY, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vertexPool = this.device.createBuffer({ label: "Fine particle surface vertices", size: SURFACE_VERTEX_BYTES * PARTICLE_SURFACE_VERTEX_CAPACITY, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // Bound while a source has no frame: the origin flag is zero then and no stage reads it.
    this.fallbackFrame = this.device.createBuffer({ label: "Fine particle surface frame fallback", size: 48, usage: GPUBufferUsage.UNIFORM });
    this.statsReadback = this.device.createBuffer({ label: "Fine particle surface stats", size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  get ready(): boolean { return Boolean(this.chain && this.layout && this.uniforms); }

  /** The pool the water draws beside its own mesh. */
  get geometry(): ParticleSurfaceGeometry | undefined {
    return this.vertexPool && this.counters
      ? { vertices: this.vertexPool, indirect: this.counters, indirectOffset: PARTICLE_SURFACE_INDIRECT_OFFSET } : undefined;
  }

  /** φ at every record (fine units, zero or more outside the skin), for the overlay's spray. */
  get particleField(): GPUBuffer | undefined { return this.fieldBuffer; }

  /** The last counts read back, a frame or more late. */
  get stats(): ParticleSurfaceStats | undefined { return this.lastStats; }

  /** Per-record buffers follow the solver's capacity; they grow only when it does. */
  private ensureParticleCapacity(capacity: number): void {
    if (capacity <= this.particleCapacity) return;
    this.work?.destroy(); this.sorted?.destroy(); this.fieldBuffer?.destroy();
    this.particleCapacity = capacity;
    this.work = this.device.createBuffer({ label: "Fine particle surface particle order", size: 8 * capacity, usage: GPUBufferUsage.STORAGE });
    this.sorted = this.device.createBuffer({ label: "Fine particle surface sorted particles", size: 16 * capacity, usage: GPUBufferUsage.STORAGE });
    this.fieldBuffer = this.device.createBuffer({ label: "Fine particle surface particle φ", size: 4 * capacity, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.group = undefined;
  }

  private bindGroupFor(source: GPUFluidParticleSource): GPUBindGroup {
    const frame = source.frame?.buffer;
    if (!this.group || this.groupKey?.particles !== source.buffer || this.groupKey.frame !== frame) {
      this.group = this.device.createBindGroup({
        label: "Fine particle surface bind group",
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms! } },
          { binding: 1, resource: source.frame ?? { buffer: this.fallbackFrame! } },
          { binding: 2, resource: { buffer: source.buffer } },
          { binding: 3, resource: { buffer: this.counters! } },
          { binding: 4, resource: { buffer: this.hash! } },
          { binding: 5, resource: { buffer: this.bricks! } },
          { binding: 6, resource: { buffer: this.work! } },
          { binding: 7, resource: { buffer: this.sorted! } },
          { binding: 8, resource: { buffer: this.fieldBuffer! } },
          { binding: 9, resource: { buffer: this.vertexPool! } },
        ],
      });
      this.groupKey = { particles: source.buffer, frame };
    }
    return this.group;
  }

  /** Rebuilds the skin from the source's live particles, unless nothing it reads has changed. */
  encode(encoder: GPUCommandEncoder, source: GPUFluidParticleSource, frame: ParticleSurfaceFrame): boolean {
    if (!this.ready || source.capacity === 0) return false;
    const { settings } = frame;
    const built = this.built;
    if (built[0] === frame.revision && built[1] === settings.greed && built[2] === settings.smoothing
      && built[3] === settings.thickness && built[4] === frame.container_m[0] && built[5] === frame.container_m[2]
      && built[6] === settings.merge && this.groupKey?.particles === source.buffer) return true;
    built[0] = frame.revision; built[1] = settings.greed; built[2] = settings.smoothing;
    built[3] = settings.thickness; built[4] = frame.container_m[0]; built[5] = frame.container_m[2];
    built[6] = settings.merge;
    this.ensureParticleCapacity(source.capacity);
    const f = this.uniformF32, u = this.uniformU32;
    const reach = clamp(settings.smoothing, ...PARTICLE_SURFACE_SMOOTHING_RANGE) * FINE_PER_CELL;
    // The weight a kernel buried in the bulk sums to: the seed density times
    // the integral of (1 - r²/R²)³ over the ball, 64π/315 R³.
    const bulk = SEEDS_PER_CELL / FINE_PER_CELL ** 3 * (64 * Math.PI / 315) * reach ** 3;
    const greed = clamp(settings.greed, 0, 1);
    const scale = source.positionScale_m, grid = source.grid;
    f[0] = scale[0]; f[1] = scale[1]; f[2] = scale[2]; f[3] = FINE_PER_CELL;
    f[4] = -0.5 * frame.container_m[0]; f[5] = 0; f[6] = -0.5 * frame.container_m[2]; f[7] = source.frame ? 1 : 0;
    f[8] = reach;
    f[9] = clamp(settings.thickness, ...PARTICLE_SURFACE_THICKNESS_RANGE) * FINE_PER_CELL;
    f[10] = bulk * (0.8 + (0.05 - 0.8) * greed);
    f[11] = 0;
    u[12] = source.strideFloats; u[13] = source.capacity;
    u[14] = grid ? grid.ballisticFloat + (grid.ballisticSignBit ? 0x80000000 : 0) : 0;
    u[15] = grid ? grid.depthFloat : 0xffffffff;
    u[16] = PARTICLE_SURFACE_BRICK_CAPACITY; u[17] = HASH_SIZE; u[18] = PARTICLE_SURFACE_VERTEX_CAPACITY; u[19] = this.particleCapacity;
    f[20] = clamp(settings.merge, ...PARTICLE_SURFACE_MERGE_RANGE) * Math.min(scale[0], scale[1], scale[2]);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    // Word 0 is the live count: copied on the queue, or written ahead of this
    // submit where the source has no GPU count, so the clear starts past it.
    encoder.clearBuffer(this.counters!, 4, COUNTER_BYTES - 4);
    encoder.clearBuffer(this.hash!, 0, 4 * HASH_SIZE);
    encoder.clearBuffer(this.bricks!, 4 * PARTICLE_SURFACE_BRICK_CAPACITY, 4 * PARTICLE_SURFACE_BRICK_CAPACITY);
    if (source.liveCount) encoder.copyBufferToBuffer(source.liveCount.buffer, source.liveCount.byteOffset, this.counters!, 0, 4);
    else { this.liveWord[0] = source.capacity; this.device.queue.writeBuffer(this.counters!, 0, this.liveWord); }
    const pass = encoder.beginComputePass({ label: "Fine particle surface" });
    pass.setBindGroup(0, this.bindGroupFor(source));
    for (const [pipeline, groups] of this.chain!) {
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(groups);
    }
    pass.end();
    if (!this.statsPending) {
      encoder.copyBufferToBuffer(this.counters!, 4, this.statsReadback!, 0, 12);
      this.statsPending = true;
    }
    return true;
  }

  /** Maps the counts copied by the last rebuild once that submit completes; never blocks. */
  pumpStats(): void {
    const readback = this.statsReadback;
    if (!this.statsPending || !readback || readback.mapState !== "unmapped") return;
    readback.mapAsync(GPUMapMode.READ).then(() => {
      const [bricks, vertices, overflow] = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      this.statsPending = false;
      this.lastStats = { bricks: bricks!, vertices: vertices!, overflow: overflow! };
      const fresh = overflow! & ~this.reportedOverflow;
      if (fresh) {
        this.reportedOverflow |= fresh;
        const names = Object.entries(PARTICLE_SURFACE_OVERFLOW).filter(([, bit]) => fresh & bit).map(([name]) => name);
        console.error(`Fine particle surface overflowed its ${names.join(", ")} pool: parts of the skin are missing (${bricks} bricks, ${vertices} vertices).`);
      }
    }).catch(() => { this.statsPending = false; });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.uniforms, this.counters, this.hash, this.bricks, this.vertexPool, this.work, this.sorted, this.fieldBuffer, this.fallbackFrame, this.statsReadback]) buffer?.destroy();
    this.chain = undefined;
    this.group = undefined;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
