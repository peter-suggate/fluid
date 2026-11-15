/** Parity of lane F's particle formulas (particles-formulas.wgsl.ts) with the old engine's text (parity/uniform-dam.wgsl,
 * captured from the uniform engine at a22d512d for the dam).
 *
 * Each case compiles the old functions, whole where they take values and as checked excerpts where they address the
 * old storage, with stubs for the world they read (an analytic field, depth, heat and solid, and a cell-sorted
 * particle table), and a kernel built from the new pure functions over the same stubs, and compares the outputs'
 * bits. The records are the dam's at step 120 (a 10-cell block of 2,150 records about a shallow record:
 * parity/dam-120-block.bin) and a synthetic crowded set (12 to 20 records a cell, every mass, ballistic, seed-first,
 * dead and FLIP-less records) that reaches the merge, the mass bound and the split. */
import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { nbfParticleFormulasWGSL, nbfParticleRecordWGSL, nbfParticleSupported, nbfSeedFormulasWGSL } from "./particles-formulas.wgsl";
import { NBF_SUPPORT_TRAVEL_WGSL } from "./particles-kernels.wgsl";
import { particleBrickDepositWGSL } from "../uniform/uniform-particle-brick.wgsl";
import { narrowBandSeparationReach } from "../uniform/uniform-narrow-band-flip";
import { oldExcerpt, oldFunction, parityCode, parityGPU, prng, runParity, ulpCompare, withParityDevice, type ParityKernel } from "./parity/harness";

const old = (...names: string[]) => names.map(oldFunction).join("\n");
const NEW = `${nbfParticleRecordWGSL}\n${nbfParticleFormulasWGSL}\n${nbfSeedFormulasWGSL}`;
/** The old engine's constants, each line also in the new module (both kernels compile the same values). */
const OLD_CONSTS = [
  "const NB_BAND_DEPTH:f32=3.00;", "const NB_GRADED:bool=true;", "const NB_MASS_MAX:f32=8.0;", "const NB_CELL_MASS:f32=16.0;",
  "const NB_SEED_FIRST:u32=0x7f7ffff8u;", "const NB_SAMPLE_RADIUS=array<f32,9>(0.0,0.25,0.31498,0.36056,0.39685,0.42750,0.45428,0.47823,0.5);",
];
for (const line of OLD_CONSTS) assert.ok(NEW.includes(line), line);
const OLD_RECORD = `struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
${old("nbParticleMode", "nbParticleAge", "nbParticleState", "nbParticleMass", "nbParticleRestate", "nbWithMass", "nbLive")}
${OLD_CONSTS.join("\n")}
fn nbSeedFirst()->f32{return bitcast<f32>(NB_SEED_FIRST);}`;

// ---- Records: the frozen dam block and a synthetic crowded set, cell-sorted in a 16^3 domain of 4^3 tiles.
const N = 16;
type Records = { particles: Float32Array; links: Uint32Array; counts: Uint32Array; count: number };
const cellId = (x: number, y: number, z: number) => 64 * ((x >> 2) + 4 * (y >> 2) + 16 * (z >> 2)) + ((x & 3) | ((y & 3) << 2) | ((z & 3) << 4));
function sorted(records: Float32Array): Records {
  const count = records.length / 12;
  const keyed = Array.from({ length: count }, (_, i) => {
    const c = [0, 1, 2].map(a => Math.floor(records[12 * i + a]!));
    assert.ok(c.every(v => v >= 0 && v < N), `record ${i} outside the domain`);
    return { i, key: cellId(c[0]!, c[1]!, c[2]!) };
  }).sort((a, b) => a.key - b.key || a.i - b.i);
  const particles = new Float32Array(records.length);
  const links = new Uint32Array(N * N * N), counts = new Uint32Array(N * N * N);
  keyed.forEach(({ i, key }, j) => { particles.set(records.subarray(12 * i, 12 * i + 12), 12 * j); if (counts[key]++ === 0) links[key] = j; });
  for (let key = 0, start = 0; key < links.length; key++) { if (counts[key] === 0) links[key] = start; else start = links[key]! + counts[key]!; }
  return { particles, links, counts, count };
}
const PARITY_DIR = resolve(import.meta.dirname, "parity");
function frozen(): Records {
  const meta = JSON.parse(readFileSync(resolve(PARITY_DIR, "dam-120-block.json"), "utf8")) as { lo: number[]; records: number };
  const bytes = readFileSync(resolve(PARITY_DIR, "dam-120-block.bin"));
  const r = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  assert.equal(r.length, 12 * meta.records);
  // Into cells 3..13 of the domain: an integer shift within a factor of two of each coordinate is exact (Sterbenz).
  for (let i = 0; i < meta.records; i++) for (let a = 0; a < 3; a++) r[12 * i + a] = Math.fround(r[12 * i + a]! - (meta.lo[a]! - 3));
  return sorted(r);
}
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const asFloat = (w: number) => { u32[0] = w >>> 0; return f32[0]!; };
function synthetic(seed: number): Records {
  const rnd = prng(seed), out: number[] = [];
  for (let cell = 0; cell < 160; cell++) {
    const c = [2 + Math.floor(rnd() * 12), 2 + Math.floor(rnd() * 12), 2 + Math.floor(rnd() * 12)];
    const n = cell % 3 === 0 ? 12 + Math.floor(rnd() * 9) : Math.floor(rnd() * 6);
    for (let k = 0; k < n; k++) {
      const mass = 1 + Math.floor(rnd() * 8), ballistic = rnd() < 0.15 ? 1 : 0;
      const ageWord = rnd() < 0.1 ? 0x7f7ffff8 : (u32[0] = 0, f32[0] = rnd() < 0.5 ? 0 : rnd() * 0.6, u32[0]! & 0x7ffffff8);
      const heat = rnd() < 0.06 ? -1 : [0.3, 1, 1.6][Math.floor(rnd() * 3)]!;
      const before = [0, 1, 2].map(() => (rnd() < 0.1 ? 3e38 : (rnd() - 0.5) * 0.4));
      out.push(c[0]! + rnd(), c[1]! + rnd(), c[2]! + rnd(), heat,
        (rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 4, (rnd() - 0.5) * 6,
        before[0]!, before[1]!, before[2]!, asFloat(((ballistic << 31) | ageWord | (mass - 1)) >>> 0));
    }
  }
  return sorted(new Float32Array(out));
}

// ---- The world both kernels read: params from inp, the cell table, an analytic field and analytic depth, heat and solids.
const PARAMS = new Float32Array(20);
PARAMS.set([0.05, 0.05, 0.05, 1 / 60, 0.95, 0, 0, 0, 2.0, 0.5, 0, 8]);
new Uint32Array(PARAMS.buffer).set([0, 2, 0x5eed, 0], 12);
PARAMS.set([0, -9.81, 0, 0], 16);
const WORLD = /* wgsl */`
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f,live:vec4u,gravity:vec4f}
var<private> params:Params;
fn loadParams(){params=Params(vec4f(f(0u),f(1u),f(2u),f(3u)),vec4f(f(4u),f(5u),f(6u),f(7u)),vec4f(f(8u),f(9u),f(10u),f(11u)),vec4u(inp[12],inp[13],inp[14],inp[15]),vec4f(f(16u),f(17u),f(18u),f(19u)));}
@group(0) @binding(3) var<storage,read> particles:array<Particle>;
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> links:array<u32>;
@group(0) @binding(6) var<storage,read_write> state:array<atomic<u32>>;
const NB_NO_BIN:u32=0xffffffffu;
const UM_RESIDENT:u32=0u;const UM_SOLID:u32=2u;
struct umBounds {lo:vec3f,hi:vec3f}
struct NBRigidContact{position:vec3f,velocity:vec3f}
fn nbHeldShift()->vec3f{return vec3f(0.0);}
fn umCellAt(p:vec3u)->u32{if(any(p>=vec3u(16u))){return NB_NO_BIN;}let t=p>>vec3u(2u);return 64u*(t.x+4u*t.y+16u*t.z)+((p.x&3u)|((p.y&3u)<<2u)|((p.z&3u)<<4u));}
fn nbBinCell(cell:u32)->u32{return cell;}
fn nbRun(a:vec3i,b:vec3i)->vec2u{
 if(any(a<vec3i(0))||any(b>=vec3i(16))){return vec2u(0u);}
 let ca=umCellAt(vec3u(a));let cb=umCellAt(vec3u(b));return vec2u(links[ca],links[cb]+atomicLoad(&bins[cb]));
}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
fn nbMoveSample(x:vec3f)->vec3f{return vec3f(0.8*sin(0.9*x.y+0.3*x.z),-1.2+0.5*cos(0.6*x.x),0.6*sin(0.4*x.x+0.7*x.y));}
fn nbMoveDepth(x:vec3f)->f32{return (x.y-8.0)*0.9+0.5*sin(1.1*x.x+0.4*x.z);}
fn nbTile(q:vec3f)->vec3i{return vec3i(floor(q))>>vec3u(2u);}
fn nbMoveCold(q:vec3f)->bool{let t=nbTile(q);return ((t.x+t.y+t.z)&3)==1;}
fn nbMoveHeat(q:vec3f)->f32{let t=nbTile(q);return array<f32,4>(0.0,0.6,1.0,1.7)[(t.x*3+t.y+t.z*2)&3];}
fn nbTileThin(q:vec3f)->bool{let t=nbTile(q);return ((t.x^t.z)&1)==1;}
fn gridSupported(p:vec3f)->bool{let c=vec3i(floor(p));return ((c.x*7+c.y*3+c.z*5)%11)!=0;}
fn nbParticleBlocked(q:vec3f)->bool{return q.y<1.3||q.x>14.6;}
fn umPointKind(p:vec3f)->u32{return select(UM_RESIDENT,UM_SOLID,any(p<vec3f(1.0))||any(p>vec3f(15.0)));}
fn umPointPresent(p:vec3f)->bool{return all(p>=vec3f(0.0))&&all(p<vec3f(16.0));}
fn umBoxWalls()->umBounds{return umBounds(vec3f(1.0),vec3f(15.0));}
fn nbRigidContact(point:vec3f,velocity:vec3f,inset:f32)->NBRigidContact{return NBRigidContact(point,velocity);}
var<private> appended:Particle;var<private> appendedCount:u32;
fn nbMoveAppend(p:Particle){appended=p;appendedCount+=1u;}
fn putParticle(i:u32,p:Particle){put3(i,p.position.xyz);put(i+3u,p.position.w);put3(i+4u,p.velocity.xyz);put(i+7u,p.velocity.w);put3(i+8u,p.before.xyz);put(i+11u,p.before.w);}
`;
/** What the old move calls that the brick engine does not have here: the dam has no solid walk (the box's
 * endpoint clamp holds it), no bodies and an uncut hull. */
const OLD_MOVE_STUBS = /* wgsl */`
struct NbMoved{p:Particle,status:u32}
var<private> nbHullLo:vec3f;
var<private> nbHullHi:vec3f;
fn nbHullInclude(x:vec3f){nbHullLo=min(nbHullLo,x);nbHullHi=max(nbHullHi,x);}
fn nbHullUncut()->bool{return true;}
fn nbFastSolid()->bool{return true;}
fn umBodyCount()->u32{return 0u;}
fn umSolidEnabled()->bool{return false;}
fn umFixedBox()->bool{return true;}
fn nbPointNearIn(home:vec3i,p:vec3f,inset:f32)->vec3f{return p;}
fn nbCooling()->f32{return params.activity.x;}
struct NBTerrainContact{position:vec3f,velocity:vec3f}
const nbAdvectScalarCollision:bool=false;
fn nbClearWalk()->bool{return true;}
fn umTileKey(p:vec3i)->u32{return 0u;}
fn umBodyPoseNearTile(t:u32,margin:u32)->bool{return false;}
fn umCellOpen(p:vec3i)->f32{return 1.0;}
fn nbVoxelBlocked(q:vec3f)->bool{return false;}
fn umSolidHasTerrain()->bool{return false;}
fn umBoundaryRecord(p:vec3i)->u32{return 0u;}
fn nbMoveBodyAt(q:vec3f)->i32{return -1;}
fn nbProceduralContact(start:vec3f,end:vec3f,velocity:vec3f,inset:f32)->NBTerrainContact{return NBTerrainContact(end,velocity);}
fn nbTerrainContact(startPoint:vec3f,endPoint:vec3f,velocity:vec3f,inset:f32)->NBTerrainContact{return NBTerrainContact(endPoint,velocity);}
fn umBedSlope(p:vec3i)->vec2f{return vec2f(0.0);}
fn nbArrived(q:vec3f){}
`;
const OLD_MOVE: ParityKernel = {
  prelude: `${OLD_RECORD}\n${WORLD}\n${OLD_MOVE_STUBS}\n${old("nbSampleRadius", "nbCellCap", "nbSparseExpired", "nbSparseAge", "nbMergeRun",
    "nbMergeAbsorbed", "nbSeparation", "nbSurfaceShift", "nbSeedTarget", "nbTraceEndpointIn", "nbFinish", "nbAdvect")}`,
  body: "loadParams();appendedCount=0u;let m=nbAdvect(particles[t],t);putParticle(26u*t,m.p);outp[26u*t+12u]=m.status;putParticle(26u*t+13u,appended);outp[26u*t+25u]=appendedCount;",
};
/** The move composed of the new formulas in nbAdvect's order (the cell-cap merge first, as the old engine does;
 * the brick engine moves it to the rebin), with the old engine's addressing done here through the same stubs. */
const NEW_MOVE_PARTS = /* wgsl */`
fn sparseCounts(q:vec3f)->vec2u{
 let cell=vec3i(floor(q));let own=nbCellRun(cell);var block=0u;
 for(var k=0u;k<9u;k++){
  let r=(k+4u)%9u;let c=vec3i(cell.x-1,cell.y+i32(r%3u)-1,cell.z+i32(r/3u)-1);
  let stop=min(cell.x+1,c.x|3);
  var run=nbRun(c,vec3i(stop,c.y,c.z));block+=run.y-run.x;
  if(stop<cell.x+1){run=nbRun(vec3i(stop+1,c.y,c.z),vec3i(cell.x+1,c.y,c.z));block+=run.y-run.x;}
 }
 return vec2u(own.y-own.x-1u,block);
}
fn separation(i:u32,q:vec3f,m:f32)->vec3f{
 let base=nbfSeparationBase(q);let far=nbfSeparationFar(m);var acc=vec4f(0.0);
 for(var k=0u;k<4u;k++){
  let c=vec3i(base.x,base.y+i32(k&1u),base.z+i32(k>>1u));
  let stop=min(base.x+1,c.x|3);
  var runs=array<vec2u,2>(nbRun(c,vec3i(stop,c.y,c.z)),vec2u(0u));
  if(stop<base.x+1){runs[1]=nbRun(vec3i(stop+1,c.y,c.z),vec3i(base.x+1,c.y,c.z));}
  for(var r=0u;r<2u;r++){for(var j=runs[r].x;j<runs[r].y;j++){
   if(j==i){continue;}let o=particles[j];if(o.position.w<=0.0){continue;}
   acc=nbfSeparationAdd(acc,q,m,far,o.position.xyz-nbHeldShift(),nbParticleMass(o));
  }}
 }
 return nbfSeparationFinish(acc);
}
fn surfaceShift(q:vec3f,d:vec3f)->vec3f{
 var g=vec3f(0.0);
 for(var k=0u;k<7u;k++){
  let depth=nbMoveDepth(nbfSurfaceShiftPoint(q,k));
  if(k==0u){if(!nbfSurfaceShiftNeeded(depth)){return d;}}else{g=nbfSurfaceShiftGradient(g,k,depth);}
 }
 return nbfSurfaceShiftProject(d,g);
}
`;
const NEW_MOVE: ParityKernel = {
  prelude: `${NEW}\n${WORLD}\n${NEW_MOVE_PARTS}
struct Moved{p:Particle,status:u32}
fn refMove(record:Particle,i:u32)->Moved{
 var p=record;var q=p.position.xyz-nbHeldShift();
 let first=bitcast<u32>(nbParticleAge(record))==NB_SEED_FIRST;
 if(first){p.before.w=nbParticleRestate(record,0.0,nbParticleMode(record));}
 if(params.activity.w>0.0){
  let cell=nbBinCell(umCellAt(vec3u(q)));
  if(cell!=NB_NO_BIN){
   let start=links[cell];let count=atomicLoad(&bins[cell]);
   if(i>=start&&i-start<count){
    let deep=nbMoveDepth(floor(q)+0.5);let cap=nbfCellCap(deep,params.activity.w);
    var end=start+count;
    if(deep< -1.0){
     var held=0.0;
     for(var j=start;j<start+count;j++){let after=nbfCellMassAdd(held,particles[j]);if(after<0.0){end=j;break;}held=after;}
     if(i>=end){return Moved(p,1u);}
    }
    if(deep< -1.0&&end-start>cap){
     if(i-start<cap){var s=nbfMergeStart(p);for(var j=i+cap;j<end;j+=cap){s=nbfMergeAdd(s,particles[j]);}p=nbfMergeFinish(p,s);q=p.position.xyz-nbHeldShift();}
     else{
      let k=start+(i-start)%cap;var m=nbParticleMass(particles[k]);
      for(var j=k+cap;j<i;j+=cap){m=nbfAbsorbAdd(m,particles[j]);}
      if(nbfAbsorbed(m,nbParticleMass(particles[i]))){return Moved(p,1u);}
     }
    }
   }
  }
 }
 let mode=nbParticleMode(p);let h=params.hDt.xyz;let dt=params.hDt.w;
 let lifetime=params.activity.y;let threshold=params.live.y;
 // nbAdvect's stage loop: the field sampled at one call site (stage 0 at q, then Heun's two stages).
 var x=q;var k1=vec3f(0.0);var pic=vec3f(0.0);var end=q;var ballistic=true;
 var stage=select(0u,3u,mode==1.0);
 loop{
  if(stage<3u){
   let k=nbMoveSample(x);
   if(stage==1u){x=nbfHeunStage2(q,k,dt,h);stage=2u;continue;}
   if(stage==2u){end=nbfHeunEnd(q,k1,k,dt,h);break;}
   pic=k;
  }
  var v=p.velocity.xyz;
  if(mode!=1.0){v=nbfG2P(v,p.before.xyz,pic,nbMoveCold(q),first,params.settings.x);}
  else{
   v=nbfBallisticKick(v,params.gravity.xyz,dt);
   for(var axis=0u;axis<3u;axis++){if(nbfBallisticStop(v,axis,umPointKind(nbfBallisticAhead(q,v,nbParticleMass(p),axis))>=UM_SOLID)){v[axis]=0.0;}}
  }
  if(!nbfFinite(v,1e10)){atomicStore(&state[3],2u);return Moved(p,1u);}
  let counts=sparseCounts(q);
  let age=nbfSparseAge(nbParticleAge(p),lifetime,threshold,counts.x,counts.y,dt);let depth=nbMoveDepth(q);
  p.before.w=nbParticleRestate(p,age,mode);
  p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,nbMoveHeat(q));
  p.before=vec4f(pic,p.before.w);
  if(nbfRetireDeep(depth)||nbfSparseExpired(age,lifetime,threshold)){return Moved(p,1u);}
  let rigid=nbRigidContact(q,p.velocity.xyz,nbSampleRadius(nbParticleMass(p)));q=rigid.position;p.velocity=vec4f(rigid.velocity,p.velocity.w);
  if(nbParticleBlocked(q)){return Moved(p,1u);}
  ballistic=mode==1.0||!gridSupported(q);p.before.w=nbParticleRestate(p,nbParticleAge(p),select(0.0,1.0,ballistic));
  p.position.w=nbfHeatAtG2P(p.position.w,nbMoveHeat(q),params.activity.x,dt,ballistic,depth,nbTileThin(q));
  if(p.position.w<=0.0){return Moved(p,1u);}
  end=nbfDrift(q,p.velocity.xyz,pic,mode!=1.0,dt,h);
  if(ballistic){break;}
  k1=nbfHeunK1(pic,h);x=nbfHeunStage1(q,k1,dt);stage=1u;
 }
 if(!ballistic){end+=surfaceShift(q,separation(i,q,nbParticleMass(p)));}
 if(!nbfFinite(end,1e30)){atomicStore(&state[3],2u);return Moved(p,1u);}
 let walls=umBoxWalls();let endpoint=nbfBoxEndpoint(end,walls.lo,walls.hi,nbSampleRadius(nbParticleMass(p)));
 let reach=abs(endpoint-q);
 if(!(max(reach.x,max(reach.y,reach.z))<2048.0)){atomicStore(&state[3],1u);return Moved(p,1u);}
 q=endpoint;p.velocity=vec4f(nbfClippedVelocity(p.velocity.xyz,end,endpoint),p.velocity.w);
 if(!umPointPresent(q)){return Moved(p,1u);}
 var heat=p.position.w;if(!ballistic){heat=nbfArrivalHeat(heat,nbMoveHeat(q));}
 p.position=vec4f(q,heat);
 let mass=nbParticleMass(p);
 let cell=nbBinCell(umCellAt(vec3u(q)));var held=0u;if(cell!=NB_NO_BIN){held=atomicLoad(&bins[cell]);}
 if(nbfSplitWanted(mass,ballistic,params.activity.w,held,nbSeedTarget(nbMoveDepth(floor(q)+0.5)))){
  let along=nbfSplitAlong(q);let a=nbfSplitA(q,mass,along);let b=nbfSplitB(q,mass,along);
  if(umPointPresent(a)&&umPointPresent(b)&&!nbParticleBlocked(a)&&!nbParticleBlocked(b)){
   var other=p;other.position=vec4f(b,heat);nbMoveAppend(nbWithMass(other,mass-nbfSplitKept(mass)));
   p.position=vec4f(a,heat);return Moved(nbWithMass(p,nbfSplitKept(mass)),0u);
  }
 }
 return Moved(p,0u);
}`,
  body: "loadParams();appendedCount=0u;let m=refMove(particles[t],t);putParticle(26u*t,m.p);outp[26u*t+12u]=m.status;putParticle(26u*t+13u,appended);outp[26u*t+25u]=appendedCount;",
};

/** The move's parts on each frozen record: the separation, its surface projection, Heun's end and the split. Each
 * compiles alone, so its output is the formula's bits apart from how the compiler composes it with its neighbours. */
const PARTS_OUT = 18;
const PARTS_BODY = (sep: string, shift: string, heun: string, split: string) => `_=atomicLoad(&state[0]);loadParams();let p=particles[t];let q=p.position.xyz-nbHeldShift();let m=nbParticleMass(p);let o=${PARTS_OUT}u*t;
 let s=${sep}(t,q,m);put3(o,s);let shifted=${shift}(q,s);put3(o+3u,shifted);
 let end=${heun}(q);put3(o+6u,end);put3(o+9u,end+shifted);
 let mass=f32(2u+t%7u);let ab=${split}(end,mass);put3(o+12u,ab[0]);put3(o+15u,ab[1]);`;
const OLD_PARTS: ParityKernel = {
  prelude: `${OLD_RECORD}\n${WORLD}\n${OLD_MOVE_STUBS}\n${old("nbSampleRadius", "nbSeparation", "nbSurfaceShift")}
fn heun(q:vec3f)->vec3f{
 let h=params.hDt.xyz;let dt=params.hDt.w;let pic=nbMoveSample(q);
 let k1=pic/h;let x1=q+dt*k1/3.0;let k2=nbMoveSample(x1);
 let x2=${oldExcerpt("nbAdvect", "q+2.0*dt*k/(3.0*h)").replace("k/", "k2/")};let k=nbMoveSample(x2);
 return ${oldExcerpt("nbAdvect", "q+0.25*dt*(k1+3.0*k/h)")};
}
fn split(q:vec3f,mass:f32)->array<vec3f,2>{
 ${oldExcerpt("nbFinish", `let kept=ceil(0.5*mass);let moved=mass-kept;
    let bits=bitcast<vec3u>(q);let hash=(bits.x*73856093u)^(bits.y*19349663u)^(bits.z*83492791u);
    let along=normalize(vec3f(f32(hash&255u),f32((hash>>8u)&255u),f32((hash>>16u)&255u))-127.5);
    let a=q+0.25*(moved/mass)*along;let b=q-0.25*(kept/mass)*along;`)}
 return array<vec3f,2>(a,b);
}`,
  body: PARTS_BODY("nbSeparation", "nbSurfaceShift", "heun", "split"),
};
const NEW_PARTS: ParityKernel = {
  prelude: `${NEW}\n${WORLD}\n${NEW_MOVE_PARTS}
fn heun(q:vec3f)->vec3f{
 let h=params.hDt.xyz;let dt=params.hDt.w;let pic=nbMoveSample(q);
 let k1=nbfHeunK1(pic,h);let k2=nbMoveSample(nbfHeunStage1(q,k1,dt));let k3=nbMoveSample(nbfHeunStage2(q,k2,dt,h));
 return nbfHeunEnd(q,k1,k3,dt,h);
}
fn split(q:vec3f,mass:f32)->array<vec3f,2>{let along=nbfSplitAlong(q);return array<vec3f,2>(nbfSplitA(q,mass,along),nbfSplitB(q,mass,along));}`,
  body: PARTS_BODY("separation", "surfaceShift", "heun", "split"),
};

// ---- The record's bit packing.
const RECORD_BODY = "let p=particles[t];let o=7u*t;put(o,nbParticleMode(p));put(o+1u,nbParticleAge(p));put(o+2u,nbParticleState(nbParticleAge(particles[(t+1u)%arrayLength(&particles)]),1.0-nbParticleMode(p)));put(o+3u,nbParticleMass(p));put(o+4u,nbParticleRestate(p,f(20u),nbParticleMode(p)));put(o+5u,nbWithMass(p,f32(1u+t%8u)).before.w);outp[o+6u]=select(0u,1u,nbLive(p));";
const RECORD_WORLD = "@group(0) @binding(3) var<storage,read> particles:array<Particle>;\nfn loadParams(){}";
const OLD_RECORD_KERNEL: ParityKernel = { prelude: `${OLD_RECORD}\n${RECORD_WORLD}`, body: RECORD_BODY };
const NEW_RECORD_KERNEL: ParityKernel = { prelude: `${nbfParticleRecordWGSL}\n${RECORD_WORLD}`, body: RECORD_BODY };

// ---- Seeding: the site, its gates, the cell gate, targets and masses, over every site of every cell.
const SEED_WORLD = /* wgsl */`
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
fn particleDepth(x:vec3f)->f32{return (x.y-8.0)*0.9+0.5*sin(1.1*x.x+0.4*x.z);}
fn nbParticleBlocked(q:vec3f)->bool{return q.y<1.3||q.x>14.6;}
fn nbSourcePhi(p:vec3f,value:f32)->f32{return length(p-vec3f(8.0,6.0,8.0))-2.5;}
fn nbUncovered(c:vec3u)->bool{return ((c.x*5u+c.y*3u+c.z)%7u)==0u;}
fn umOwnerAt(p:vec3i)->UMOwner{return UMOwner(0u,0u,select(4u,1u,((p.x+p.z)&3)!=3),0u);}
`;
const SEED_BODY = (site: string) => `let c=vec3u(t%14u+1u,(t/14u)%14u+1u,t/196u%14u+1u);let salt=inp[14];let initial=(t&1u)!=0u;let o=8u*8u*t;
 for(var k=0u;k<8u;k++){let s=${site};put(o+8u*k,s.x);put(o+8u*k+1u,s.y);put(o+8u*k+2u,s.z);put(o+8u*k+3u,s.w);}
 let d=particleDepth(vec3f(c)+0.5);`;
const OLD_SEED_KERNEL: ParityKernel = {
  prelude: `${OLD_RECORD}\n${SEED_WORLD}\nvar<private> nbSeedSalt:u32=0u;\n${old("umCorner", "nbSeedJitter", "nbJitter", "nbSeedTarget", "nbSeedSiteIndex", "nbExpectedMass", "seedSite")}
fn cellOk(d:f32)->bool{if(${oldExcerpt("seedCell", "d>0.9||d< -NB_BAND_DEPTH-0.5")}){return false;}return true;}
fn seedMass(d:f32)->f32{return ${oldExcerpt("seedCell", "f32(8u/nbSeedTarget(d))")};}`,
  body: `nbSeedSalt=inp[14];${SEED_BODY("seedSite(c,k,initial,false)")}
 for(var j=0u;j<8u;j++){outp[o+8u*j+4u]=nbSeedSiteIndex(j,d+0.25*f32(j)-1.0);put(o+8u*j+5u,nbExpectedMass(d+0.2*f32(j)-0.8));}
 outp[o+6u]=select(0u,1u,cellOk(d));put(o+7u,seedMass(d));outp[o+14u]=nbSeedTarget(d);`,
};
const NEW_SEED_KERNEL: ParityKernel = {
  prelude: `${nbfParticleRecordWGSL}\n${nbfParticleFormulasWGSL}\n${nbfSeedFormulasWGSL}\n${SEED_WORLD}
fn site(c:vec3u,k:u32,salt:u32,initial:bool)->vec4f{
 let q=nbfSeedPoint(c,k,salt);let distance=particleDepth(q);
 if(!nbfSeedSiteOk(distance,nbParticleBlocked(q),initial,particleDepth(vec3f(c)+0.5),nbSourcePhi(q,1.0),nbUncovered(c),umOwnerAt(vec3i(c)).width==1u)){return vec4f(-1,0,0,0);}
 return vec4f(q,distance);
}`,
  body: `${SEED_BODY("site(c,k,salt,initial)")}
 for(var j=0u;j<8u;j++){outp[o+8u*j+4u]=nbSeedSiteIndex(j,d+0.25*f32(j)-1.0);put(o+8u*j+5u,nbExpectedMass(d+0.2*f32(j)-0.8));}
 outp[o+6u]=select(0u,1u,nbfSeedCellDepthOk(d));put(o+7u,nbfSeedMass(d));outp[o+14u]=nbSeedTarget(d);`,
};

// ---- Sampling: the vertex trilinear and far rule, the face lerp, the composite blend and the fine weight.
const SAMPLE_WORLD = /* wgsl */`
fn eight(i:u32)->array<f32,8>{return array<f32,8>(f(i),f(i+1u),f(i+2u),f(i+3u),f(i+4u),f(i+5u),f(i+6u),f(i+7u));}
fn frac3(i:u32)->vec3f{return fract(abs(v3(i)));}
`;
const SAMPLE_IN = 24; // words a thread: 8 values, 3 fraction, far, fine, h sample, 4h sample, mask, q (3), tile (3), 2 spare
const OLD_SAMPLE_KERNEL: ParityKernel = {
  prelude: `${SAMPLE_WORLD}
${old("umCorner", "umVelocityLerp8")}
override umRegularFine:bool=false;
struct UMCoarseHome {local:vec3f,origin:vec3i}
struct UMSampleTile {q:vec3f,tile:vec3i,key:u32}
var<private> fineSample:f32;var<private> coarseSample:f32;var<private> blendMask:u32;
fn umSampleVelocity1(p:vec3f,axis:u32)->f32{return fineSample;}
fn umSampleVelocity4In(home:UMCoarseHome,axis:u32)->f32{return coarseSample;}
fn umTileSupport(t:u32)->u32{return 1u;}
fn umTileWidth(t:u32)->u32{return 4u;}
fn umVelocityBlendMask(t:u32)->u32{return blendMask;}
${old("umSampleVelocityWeightedIn", "umVelocitySamplingWeightsAt")}
fn depthAt(v:array<f32,8>,f:vec3f,reached:bool,far:f32)->f32{
 if(!reached){${oldExcerpt("particleDepthIn", "return select(max(far,6.0),min(far,-6.0),far<0.0);")}}
 var value=0.0;
 for(var k=0u;k<8u;k++){
  ${oldExcerpt("particleDepthIn", "let corner=umCorner(k,2u);let w=select(1.0-f,f,corner!=vec3u(0));")}
  ${oldExcerpt("particleDepthIn", "value+=w.x*w.y*w.z*")}v[k];
 }
 return value;
}`,
  body: `let i=${SAMPLE_IN}u*t;let v=eight(i);let fr=frac3(i+8u);let o=6u*t;
 fineSample=f(i+13u);coarseSample=f(i+14u);blendMask=inp[i+15u]&0x7ffffffu;
 put(o,depthAt(v,fr,true,0.0));put(o+1u,depthAt(v,fr,false,f(i+11u)));put(o+2u,umVelocityLerp8(v,fr));
 put(o+3u,umSampleVelocityWeightedIn(fr,UMCoarseHome(vec3f(0),vec3i(0)),0u,fract(abs(f(i+12u)))*1.5-0.25));
 let s=UMSampleTile(v3(i+16u),vec3i(floor(v3(i+19u))),0u);put(o+4u,umVelocitySamplingWeightsAt(s));put(o+5u,0.0);`,
};
const NEW_SAMPLE_KERNEL: ParityKernel = {
  prelude: `${nbfParticleRecordWGSL}\n${nbfParticleFormulasWGSL}\n${SAMPLE_WORLD}`,
  body: `let i=${SAMPLE_IN}u*t;let v=eight(i);let fr=frac3(i+8u);let o=6u*t;
 let fineSample=f(i+13u);let coarseSample=f(i+14u);let blendMask=inp[i+15u]&0x7ffffffu;
 put(o,nbfVertexTrilinear(v,fr));put(o+1u,nbfFarDepth(f(i+11u)));put(o+2u,nbfFaceLerp8(v,fr));
 put(o+3u,nbfCompositeBlend(fract(abs(f(i+12u)))*1.5-0.25,fineSample,coarseSample));
 let q=v3(i+16u);let tile=vec3i(floor(v3(i+19u)));var fine=0.0;var remaining=blendMask;
 while(remaining!=0u){let bit=firstTrailingBit(remaining);remaining&=remaining-1u;
  fine=nbfFineWeight(fine,q,vec3f(tile+vec3i(i32(bit%3u),i32((bit/3u)%3u),i32(bit/9u))-vec3i(1))*4.0);}
 put(o+4u,fine);put(o+5u,0.0);`,
};

// NBF_PARITY_DUMP=DIR writes every kernel's module there, for offline validation (naga DIR/*.wgsl).
if (process.env.NBF_PARITY_DUMP) {
  const dir = process.env.NBF_PARITY_DUMP; mkdirSync(dir, { recursive: true });
  for (const [name, kernel] of Object.entries({ OLD_MOVE, NEW_MOVE, OLD_PARTS, NEW_PARTS, OLD_RECORD_KERNEL, NEW_RECORD_KERNEL, OLD_SEED_KERNEL, NEW_SEED_KERNEL, OLD_SAMPLE_KERNEL, NEW_SAMPLE_KERNEL })) writeFileSync(`${dir}/${name}.wgsl`, parityCode(kernel, 64));
}

test("the brick engine's particle kernels reject settings they do not implement", () => {
  nbfParticleSupported({ particleTransfer: "flip", crowdedCells: "merge", surfaceSamples: 8 });
  assert.throws(() => nbfParticleSupported({ crowdedCells: "retire" }), /crowdedCells "retire" is not implemented/);
  assert.throws(() => nbfParticleSupported({ surfaceSamples: 64 }), /surfaceSamples "64"/);
});

test("the parity fixture holds every old function the formulas port", () => {
  for (const name of ["nbAdvect", "nbFinish", "nbSeparation", "nbSurfaceShift", "nbMergeRun", "seedSite", "particleDepthIn"]) assert.match(oldFunction(name), new RegExp(`^fn ${name}\\(`));
});

/** Every case: [label, old kernel, new kernel, inputs]. */
async function parity(device: GPUDevice, label: string, oldKernel: ParityKernel, newKernel: ParityKernel, io: Parameters<typeof runParity>[2]): Promise<ReturnType<typeof ulpCompare>> {
  const a = await runParity(device, oldKernel, io), b = await runParity(device, newKernel, io);
  const result = ulpCompare(a, b);
  console.log(`parity ${label}: ${io.outWords} words, ${result.differ} differ, max ${result.maxUlp} ulp`);
  return result;
}
function moveIO(records: Records) {
  const inp = new Uint32Array(PARAMS.buffer.slice(0));
  return {
    inp, threads: records.count, outWords: 26 * records.count,
    extra: { 3: records.particles, 4: records.counts, 5: records.links, 6: new Uint32Array(4) },
  };
}

test("the move's parts (separation, surface shift, Heun's end, split) match the old functions bit for bit", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const [label, records] of [["dam block", frozen()], ["crowded", synthetic(7)]] as const) {
      const io = { ...moveIO(records), outWords: PARTS_OUT * records.count };
      const a = await runParity(device, OLD_PARTS, io), b = await runParity(device, NEW_PARTS, io);
      const r = ulpCompare(a, b);
      const hist = new Map<number, number>();
      for (let w = 0; w < a.length; w++) if (a[w] !== b[w]) hist.set(w % PARTS_OUT, (hist.get(w % PARTS_OUT) ?? 0) + 1);
      console.log(`parity move parts ${label}: ${records.count} records; ${r.differ} words differ (by word ${JSON.stringify([...hist])}), max ${r.maxUlp} ulp`);
      assert.equal(r.differ, 0);
    }
  });
});

test("the move (G2P, Heun RK3, sparse age, heat, separation, surface shift, box clamp, merge, mass bound, split) matches the old nbAdvect bit for bit", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const [label, records] of [["dam block", frozen()], ["crowded", synthetic(7)]] as const) {
      const io = moveIO(records);
      const a = await runParity(device, OLD_MOVE, io), b = await runParity(device, NEW_MOVE, io);
      const r = ulpCompare(a, b);
      const status = (w: Uint32Array) => [0, 1].map(s => Array.from({ length: records.count }, (_, i) => w[26 * i + 12]).filter(x => x === s).length);
      const split = Array.from({ length: records.count }, (_, i) => a[26 * i + 25]).filter(x => x !== 0).length;
      const hist = new Map<number, number>();
      for (let w = 0; w < a.length; w++) if (a[w] !== b[w]) hist.set(w % 26, (hist.get(w % 26) ?? 0) + 1);
      console.log(`parity move ${label}: ${records.count} records, live/retired ${status(a).join("/")}, ${split} splits; ${r.differ} words differ (by word ${JSON.stringify([...hist])}), max ${r.maxUlp} ulp at ${r.worst} (${new Float32Array(a.buffer)[r.worst ?? 0]} vs ${new Float32Array(b.buffer)[r.worst ?? 0]}), max abs ${r.maxAbs}`);
      if (r.differ) {
        const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer), shown = new Set<number>();
        for (let w = 0; w < a.length && shown.size < 4; w++) {
          const rec = Math.floor(w / 26);
          if (a[w] === b[w] || shown.has(rec)) continue;
          shown.add(rec);
          const words = (f: Float32Array) => Array.from(f.subarray(26 * rec, 26 * rec + 26), x => x.toPrecision(8)).join(" ");
          console.log(`record ${rec} in : ${Array.from(records.particles.subarray(12 * rec, 12 * rec + 12), x => x.toPrecision(8)).join(" ")}\n old: ${words(fa)}\n new: ${words(fb)}`);
        }
      }
      assert.deepEqual(status(a), status(b));
      assert.equal(r.differ, 0, `first differing word ${r.first} (record ${Math.floor(r.first! / 26)}, word ${r.first! % 26})`);
    }
  });
});

test("the record's bit packing matches the old functions", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const records of [frozen(), synthetic(11)]) {
      const inp = new Float32Array(24); inp[20] = 0.37;
      const r = await parity(device, "record", OLD_RECORD_KERNEL, NEW_RECORD_KERNEL, { inp, threads: records.count, outWords: 7 * records.count, extra: { 3: records.particles } });
      assert.equal(r.differ, 0);
    }
  });
});

test("seeding (jitter, site gates, cell gate, targets, masses) matches the old seedSite and seedCell", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const inp = new Uint32Array(PARAMS.buffer.slice(0)), threads = 14 * 14 * 14;
    const r = await parity(device, "seed", OLD_SEED_KERNEL, NEW_SEED_KERNEL, { inp, threads, outWords: 64 * threads });
    assert.equal(r.differ, 0);
  });
});

test("sampling (vertex trilinear, far depth, face lerp, composite blend, fine weight) matches the old samplers", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const threads = 4096, rnd = prng(23), inp = new Float32Array(SAMPLE_IN * threads);
    for (let t = 0; t < threads; t++) {
      const o = SAMPLE_IN * t;
      for (let k = 0; k < 8; k++) inp[o + k] = (rnd() - 0.5) * 12;
      for (let k = 8; k < 15; k++) inp[o + k] = (rnd() - 0.5) * 20;
      new Uint32Array(inp.buffer)[o + 15] = Math.floor(rnd() * 2 ** 27);
      for (let k = 16; k < 22; k++) inp[o + k] = 4 + rnd() * 8;
    }
    const r = await parity(device, "sampling", OLD_SAMPLE_KERNEL, NEW_SAMPLE_KERNEL, { inp, threads, outWords: 6 * threads });
    assert.equal(r.differ, 0);
  });
});

// ---- DEPB's particle support (pJoin's pSupport): the travel is the old deposit's text, with the gate scenes' reach.
test("particle support: pJoin's travel is particleBrickDepositWGSL's lrParticleTravelConditional, NB_SEPARATION_REACH the old value", () => {
  const reach = narrowBandSeparationReach("pairs", true, 8);
  assert.equal(reach, 1);
  const oldText = particleBrickDepositWGSL(false, { joins: 0, allocation: 0 }, "bins", true, reach, 3);
  const squash = (t: string) => t.replace(/\s+/g, " ").trim();
  assert.ok(squash(oldText).includes(squash(NBF_SUPPORT_TRAVEL_WGSL)), "NBF_SUPPORT_TRAVEL_WGSL is not the old deposit's travel text");
});
