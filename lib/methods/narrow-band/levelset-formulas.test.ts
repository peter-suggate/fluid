/** Parity of lane F's level-set formulas (levelset-formulas.wgsl.ts) with the old engine's text (parity/uniform-dam.wgsl).
 *
 * As particles-formulas.test.ts: the old functions compile whole with stubs for the storage they read (an analytic φ
 * and velocity, a cell table of bins and hashed tile heats in a 16^3 domain of 4^3 tiles), against kernels of the
 * new pure functions over the same stubs, on random inputs; the outputs' bits are compared. */
import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { nbfLevelSetConstantsWGSL, nbfPhiAdvectWGSL, nbfPhiUnionWGSL, nbfRedistanceWGSL } from "./levelset-formulas.wgsl";
import { oldExcerpt, oldFunction, parityCode, parityGPU, prng, runParity, ulpCompare, withParityDevice, type ParityKernel } from "./parity/harness";

const old = (...names: string[]) => names.map(oldFunction).join("\n");

// ---- The world: params, an analytic φ (cells) and velocity (m/s), a bins buffer laid out as below.
// bins: [0, 4096) a cell's nearest crossing cell (NB_DEPTH_B), [4096, 4160) a tile's reach, [4160, 4224) its
// candidate word, [4224, 4288) its heat, [4288, 4352) its surface heat, [4352, 4352 + 17^3) a vertex's nearest
// hot sample key (NB_SURFACE_HOT_NEAREST).
const BINS = 4352 + 17 ** 3;
const WORLD = /* wgsl */`
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f,live:vec4u,gravity:vec4f}
var<private> params:Params;
fn loadParams(){params.hDt=vec4f(0.05,0.05,0.05,1.0/60.0);}
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
struct umBounds {lo:vec3f,hi:vec3f}
const NB_NO_SURFACE:u32=0xffffffffu;
const NB_PREVIOUS_CROSSINGS:bool=true;
const NB_SURFACE_HOT_NEAREST=4352u;
fn NB_DEPTH_B()->u32{return 0u;}
fn NB_BAND_REACH()->u32{return 4096u;}
fn NB_BAND_CANDIDATES()->u32{return 4160u;}
fn NB_ACTIVITY_THETA()->u32{return 4224u;}
fn NB_ACTIVITY_SURFACE_HEAT()->u32{return 4288u;}
fn umTileAt(t:vec3u)->u32{let c=min(t,vec3u(3u));return c.x+4u*c.y+16u*c.z;}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%4u,(t/4u)%4u,t/16u);}
fn umCellAt(p:vec3u)->u32{let c=min(p,vec3u(15u));return 64u*umTileAt(c>>vec3u(2u))+((c.x&3u)|((c.y&3u)<<2u)|((c.z&3u)<<4u));}
fn umCellCoord(k:u32)->vec3u{return (umTileCoord(k>>6u)<<vec3u(2u))|vec3u(k&3u,(k>>2u)&3u,(k>>4u)&3u);}
fn umVertexAt(p:vec3u)->u32{let c=min(p,vec3u(16u));return c.x+17u*(c.y+17u*c.z);}
fn nbBandTile(c:vec3u)->u32{return umTileAt(c/4u);}
fn nbBandReach(c:vec3u)->u32{return atomicLoad(&bins[NB_BAND_REACH()+nbBandTile(c)]);}
fn phiField(p:vec3f)->f32{return (p.y-9.0)*0.7+1.5*sin(0.37*p.x)*cos(0.29*p.z)+0.2*sin(1.7*p.x+0.9*p.y);}
fn bandPhi(p:vec3f)->f32{return phiField(p);}
fn nbSourcePhi(p:vec3f,value:f32)->f32{return min(value,length(p-vec3f(8.0,12.0,8.0))-2.0);}
fn umCellPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(16));}
fn nbParticleCellOpen(c:vec3i)->bool{return ((c.x*5+c.y*3+c.z)%9)!=4;}
fn nbBinCount(cell:u32)->u32{return ((cell*2654435761u)>>27u)%13u;}
fn umTileOpen(t:u32)->f32{return select(1.0,0.0,t%7u==3u);}
fn nbHomeBlock(page:vec3i,lo:vec3i,hi:vec3i)->u32{return 0u;}
fn umTileNearFromPage(home:u32,page:vec3i,p:vec3i)->vec3i{return clamp(p,vec3i(0),vec3i(3));}
var<private> nbCoupleHeat:array<f32,27>;
var<private> nbCoupleHot:array<f32,27>;
fn loadStage(i:u32){for(var k=0u;k<27u;k++){nbCoupleHeat[k]=f(i+k);nbCoupleHot[k]=f(i+27u+k);}}
fn cellVertices(i:u32)->array<f32,8>{return array<f32,8>(f(i),f(i+1u),f(i+2u),f(i+3u),f(i+4u),f(i+5u),f(i+6u),f(i+7u));}
`;
const NEW_WORLD = /* wgsl */`
fn nbxCoupleHeat(tile:u32,surface:bool)->f32{return select(nbCoupleHeat[tile],nbCoupleHot[tile],surface);}
fn nbxHeatCorner(base:vec3i,k:u32,surface:bool)->vec2f{
 let c=umTileNearFromPage(0u,vec3i(0),base+vec3i(vec3u(k%2u,(k/2u)%2u,k/4u)));let t=umTileAt(vec3u(c));
 if(umTileOpen(t)<=1e-5){return vec2f(0.0);}
 let heat=bitcast<f32>(atomicLoad(&bins[select(NB_ACTIVITY_THETA(),NB_ACTIVITY_SURFACE_HEAT(),surface)+t]));
 if(surface&&heat<=0.0&&bandPhi(4.0*vec3f(c)+2.0)>=0.0){return vec2f(heat,0.0);}
 return vec2f(heat,1.0);
}
`;

// ---- Reconstruction, the crossing guard, the heat blends, the cell distance and the far sentinel: one thread a
// vertex. inp a thread: q (3, cells), bulk, nearest2, heat, tiled, surface, 8 cell values, cell-local point (3),
// far value, 54 stage heats; 80 words.
const IN = 80, OUT = 8;
const UNION_BODY = (finish: string, bulk: string, heat: string, couple: string, distance: string, far: string) => `
 loadParams();let i=${IN}u*t;let o=${OUT}u*t;loadStage(i+26u);
 let q=vec3i(floor(v3(i)));let home=q>>vec3u(2u);let p=v3(i);
 let tiled=f(i+6u)>0.5;let surface=f(i+7u)>0.5;
 put(o,${finish});
 put(o+1u,${bulk});
 put(o+2u,${heat});
 put(o+3u,${couple});
 put(o+4u,${distance});
 put(o+5u,${far});
 put(o+6u,0.0);put(o+7u,0.0);`;
const OLD_UNION: ParityKernel = {
  prelude: `${WORLD}\n${old("umCorner", "nbCachedSurfaceErasure", "nbCoupleTheta", "nbInterpolatedHeat", "nbSurfaceThetaFrom", "particleSurfaceFinish", "nbCrossingTravel", "bulkDepth", "nbTrilinear", "nbCellDistance")}
const NB_SURFACE_RADIUS=0.875;
const NB_SEED_DEPTH=0.8003905296791061;
fn farOld(value:f32,h:f32)->f32{var out=value;if(${oldExcerpt("nbRedistanceVertex", "abs(value)>1e-8&&abs(value)<6.0*h")}){out=${oldExcerpt("nbRedistanceVertex", "sign(value)*6.0*h")};}return out;}`,
  body: UNION_BODY("particleSurfaceFinish(home,q,f(i+3u),f(i+4u),f(i+5u),tiled)", "bulkDepth(vec3u(q),p)", "nbInterpolatedHeat(home,p,surface)",
    "nbCoupleTheta(q-4*home,surface)", "nbCellDistance(v3(i+16u),cellVertices(i+8u))", "farOld(f(i+19u),0.05)"),
};
const NEW_UNION: ParityKernel = {
  prelude: `${WORLD}\n${nbfLevelSetConstantsWGSL}\n${nbfPhiUnionWGSL}\n${nbfRedistanceWGSL}\n${NEW_WORLD}
fn finish(home:vec3i,q:vec3i,bulk:f32,nearest2:f32,heat:f32,tiled:bool)->f32{
 if(nbfUnionEmpty(nearest2)){return nbSourcePhi(vec3f(q),bulk);}
 var particle=nbfUnionParticle(bulk,nearest2);
 if(particle<bulk){
  var held=0u;var open=0u;var heldLow=vec3u(0u);var heldHigh=vec3u(0u);var openLow=vec3u(0u);var openHigh=vec3u(0u);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k%2u,(k/2u)%2u,k/4u);let c=q+vec3i(bit)-1;
   if(!umCellPresent(c)||!nbParticleCellOpen(c)){continue;}
   let n=min(nbBinCount(umCellAt(vec3u(c))),8u);open++;held+=n;
   let high=bit!=vec3u(0u);
   heldHigh+=select(vec3u(0u),vec3u(n),high);openHigh+=select(vec3u(0u),vec3u(1u),high);
   heldLow+=select(vec3u(n),vec3u(0u),high);openLow+=select(vec3u(1u),vec3u(0u),high);
  }
  particle=nbfUnionBound(particle,bulk,held,open,heldLow,heldHigh,openLow,openHigh);
 }
 var theta=heat;
 if(particle>bulk){
  if(!nbfErasure(atomicLoad(&bins[NB_SURFACE_HOT_NEAREST+umVertexAt(vec3u(q))]),nearest2)){return nbSourcePhi(vec3f(q),bulk);}
  if(tiled){theta=nbfCoupleTheta(q-4*home,true);}else{theta=nbfInterpolatedHeat(vec3f(q),true);}
 }
 return nbSourcePhi(vec3f(q),nbfUnionBlend(bulk,particle,theta));
}
fn guard(c:vec3u,p:vec3f)->f32{
 let nearest=atomicLoad(&bins[NB_DEPTH_B()+umCellAt(c)]);
 return nbfBulkDepth(bandPhi(p),p,(nbBandReach(c)&1u)!=0u,nearest!=NB_NO_SURFACE,vec3f(umCellCoord(nearest)),nbfCrossingTravel(atomicLoad(&bins[NB_BAND_CANDIDATES()+nbBandTile(c)])));
}`,
  body: UNION_BODY("finish(home,q,f(i+3u),f(i+4u),f(i+5u),tiled)", "guard(vec3u(q),p)", "nbfInterpolatedHeat(p,surface)",
    "nbfCoupleTheta(q-4*home,surface)", "nbfCellDistance(v3(i+16u),cellVertices(i+8u))", "nbfFarSentinel(f(i+19u),0.05)"),
};

// ---- phiAdvect: the trace, the trilinear and the limited tricubic, and the trace helpers. inp a thread: p (3),
// q (3), width flag, held mask, oneStep; 12 words.
const ADVECT_IN = 12, ADVECT_OUT = 12;
// The field the advection reads is data, not arithmetic: φ a table over the vertices (PHI_SIDE^3, from phiTable()),
// the velocity piecewise constant a cell, from a hash, exactly (an analytic stub inlined at each call site composes
// with its caller and rounds differently there).
const PHI_SIDE = 66;
const TRICUBIC_BOUND = 2 ** -17;
const ADVECT_WORLD = /* wgsl */`
@group(0) @binding(7) var<storage,read> phiVertices:array<f32>;
fn phiAt(p:vec3u)->f32{let c=min(p,vec3u(${PHI_SIDE - 1}u));return phiVertices[c.x+${PHI_SIDE}u*(c.y+${PHI_SIDE}u*c.z)];}
fn traceVelocity(x:vec3f)->vec3f{
 let c=vec3u(clamp(vec3i(floor(x)),vec3i(0),vec3i(80)));let h=(c.x*73856093u)^(c.y*19349663u)^(c.z*83492791u);
 return (vec3f(f32(h&255u),f32((h>>8u)&255u),f32((h>>16u)&255u))-127.5)/64.0;
}
fn pointFrom(q:vec3f,x:vec3f)->vec3f{return clamp(x,vec3f(2.0),vec3f(60.0));}
`;
function phiTable(): Float32Array {
  const n = PHI_SIDE, out = new Float32Array(n * n * n);
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
    out[x + n * (y + n * z)] = (y - 30) * 0.7 + 4 * Math.sin(0.37 * x) * Math.cos(0.29 * z) + 0.6 * Math.sin(1.7 * x + 0.9 * y);
  return out;
}
const ADVECT_BODY = (trace: string, cell: string, cubic: string, helpers: string) => `
 loadParams();_=atomicLoad(&bins[0]);_=atomicLoad(&nbState[0]);_=phiVertices[0];let i=${ADVECT_IN}u*t;let o=${ADVECT_OUT}u*t;
 let p=v3(i);let q=v3(i+3u);cellWidth=select(1u,4u,f(i+6u)>0.5);heldMask=inp[i+7u]&7u;let oneStep=f(i+8u)>0.8;
 put3(o,${trace});
 put(o+3u,${cell});
 put(o+4u,${cubic});
 ${helpers}`;
const HELPERS = `let k1=traceVelocity(p)*f(i+9u);let k2=traceVelocity(q)*f(i+9u);let base=-f(i+10u);
 outp[o+5u]=nbTraceSteps(k1,params.hDt.w);outp[o+6u]=select(0u,1u,nbTraceTwo(base,k1,k2,2u+(inp[i+7u]>>3u)%3u));
 put3(o+7u,nbTraceThird(p,base,k1,k2,f(i+11u)>0.5));let e=nbTraceEnd(p,base,k1,k2,traceVelocity(p+q),f(i+11u)>0.5);put(o+10u,e.x+e.y+e.z+e.w);
 outp[o+11u]=nbTraceGradientSteps(k1,k2,params.hDt.w,1u+(inp[i+7u]>>3u)%40u);`;
const CUBIC_STUBS = /* wgsl */`
override umExactSolidSupport:bool=true;
override umRegularFine:bool=false;
var<private> cellWidth:u32;var<private> heldMask:u32;
fn umLoadVertex(p:vec3u)->f32{return phiAt(p);}
fn umLoadCorner(p:vec3u)->f32{return phiAt(p);}
`;
const OLD_ADVECT: ParityKernel = {
  prelude: `${WORLD}\n${ADVECT_WORLD}\n${CUBIC_STUBS}
struct UMVertexCell {origin:vec3u,width:u32,corners:array<f32,8>,value:f32}
fn umSolidEnabled()->bool{return false;}
fn umSolidClear(p:vec3i)->bool{return true;}
fn umCanonicalSupportAt(p:vec3f,reach:f32)->bool{return true;}
fn umBlockOf(page:vec3i)->u32{return 0u;}
fn umBlockHeld(home:u32,page:vec3i,d:vec3i)->vec3<bool>{return vec3<bool>((heldMask&1u)!=0u,(heldMask&2u)!=0u,(heldMask&4u)!=0u);}
fn umTileKey(p:vec3i)->u32{return 0u;}
fn umKeyStored(k:u32)->bool{return true;}
fn umCellOpen(p:vec3i)->f32{return 1.0;}
fn umTileWidth(t:u32)->u32{return cellWidth;}
fn umSolidOpenCells(t:u32)->vec2u{return vec2u(0xffffffffu);}
fn umBuried(p:vec3f)->bool{return false;}
fn umVertexNearFromPage(home:u32,page:vec3i,p:vec3i)->vec3i{return p;}
fn udRun_phi(q:vec3i,n:u32)->vec4i{return vec4i(-1);}
fn udTexel_phi(a:vec3i)->vec4f{return vec4f(0.0);}
fn umOpenPointCell(q:vec3f)->vec3i{return vec3i(floor(q));}
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{var r:array<f32,8>;for(var k=0u;k<8u;k++){r[k]=umLoadVertex(p+s*umCorner(k,2u));}return r;}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return umLoadVertex8(p,s);}
override umSurfaceTraceSeed:bool=true;
override umFastSurface:bool=false;
override umSurfaceStationaryTrace:bool=true;
const UM_SURFACE_ONE_STEP:bool=true;
const NB_TRACE_GRADIENT:bool=true;
const NB_TRACE_TURN:f32=0.25;
var<private> umTraceSeedValid:bool=false;var<private> umTraceSeedPoint:vec3f;var<private> umTraceSeedVelocity:vec3f;
var<private> umHold:umBounds=umBounds(vec3f(-1e9),vec3f(1e9));
var<private> umSlow:bool=false;var<private> umOneStep:bool=false;
@group(0) @binding(5) var<storage,read_write> nbState:array<atomic<u32>>;
fn umPointBounds(p:vec3f,margin:f32)->umBounds{return umBounds(p-margin,p+margin);}
fn umFastVelocity(p:vec3f)->vec3f{return traceVelocity(p);}
fn umSampleVelocity(p:vec3f)->vec3f{return traceVelocity(p);}
fn umPointFrom(home:vec3f,p:vec3f)->vec3f{return pointFrom(home,p);}
fn umReachInclude(x:vec3f){}
${old("umCorner", "umCatmull", "umHeldWeights", "umVertexSum8", "umVertexCell", "umCubicPhi", "nbTraceSteps", "nbTraceTwo", "nbTraceThird", "nbTraceEnd", "nbTraceGradientSteps", "umSurfaceTrace")}`,
  body: ADVECT_BODY("umSurfaceTrace(p)", "umVertexCell(q).value", "umCubicPhi(q,umVertexCell(q))", HELPERS).replace("let oneStep=", "umOneStep=f(i+8u)>0.8;let oneStep="),
};
const NEW_ADVECT: ParityKernel = {
  prelude: `${WORLD}\n${ADVECT_WORLD}\n${CUBIC_STUBS}\n${nbfLevelSetConstantsWGSL}\n${nbfPhiAdvectWGSL}
@group(0) @binding(5) var<storage,read_write> nbState:array<atomic<u32>>;
fn nbxTraceVelocity(x:vec3f)->vec3f{return traceVelocity(x);}
fn nbxPointFrom(q:vec3f,x:vec3f)->vec3f{return pointFrom(q,x);}
fn nbxTraceFatal(){atomicStore(&nbState[3],1u);}
var<private> tapBase:vec3i;var<private> tapWidth:i32;var<private> tapCorners:array<f32,8>;
// As umCubicPhi: the inner taps are the cell's corners, the others direct loads.
fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{
 let offset=vec3i(vec3u(x,y,z))-vec3i(1);let inner=all(offset>=vec3i(0))&&all(offset<=vec3i(1));
 var s=0.0;if(inner){s=tapCorners[x-1u+2u*(y-1u)+4u*(z-1u)];}else{s=umLoadVertex(vec3u(tapBase+offset*tapWidth));}
 return s;
}
fn cellAt(q:vec3f)->vec4f{
 let cell=vec3u(vec3i(floor(q)));let width=cellWidth;let origin=(cell/width)*width;
 var corners:array<f32,8>;for(var k=0u;k<8u;k++){corners[k]=umLoadVertex(origin+width*vec3u(k%2u,(k/2u)%2u,k/4u));}
 return vec4f(vec3f(origin),nbfCellTrilinear(corners,(q-vec3f(origin))/f32(width)));
}
fn cubic(q:vec3f)->f32{
 let width=i32(cellWidth);let base=vec3i(cellAt(q).xyz);let fraction=(q-vec3f(base))/f32(width);
 let at=base&vec3i(31);let past=vec3i(at+vec3i(2*width)>vec3i(32))-vec3i(at<vec3i(width));
 let held=vec3<bool>((heldMask&1u)!=0u,(heldMask&2u)!=0u,(heldMask&4u)!=0u);
 tapBase=base;tapWidth=width;for(var k=0u;k<8u;k++){tapCorners[k]=umLoadVertex(vec3u(base)+u32(width)*vec3u(k%2u,(k/2u)%2u,k/4u));}
 // As umCubicPhi: the width branch outside the 64 taps (one inlined copy a width).
 if(width==4){return nbfCubicLimited(fraction,held&(past<vec3i(0)),held&(past>vec3i(0)));}
 return nbfCubicLimited(fraction,held&(past<vec3i(0)),held&(past>vec3i(0)));
}`,
  body: ADVECT_BODY("nbfSurfaceTrace(p,params.hDt.w,params.hDt.xyz,oneStep)", "cellAt(q).w", "cubic(q)", HELPERS),
};

if (process.env.NBF_PARITY_DUMP) {
  const dir = process.env.NBF_PARITY_DUMP; mkdirSync(dir, { recursive: true });
  for (const [name, kernel] of Object.entries({ OLD_UNION, NEW_UNION, OLD_ADVECT, NEW_ADVECT })) writeFileSync(`${dir}/${name}.wgsl`, parityCode(kernel, 64));
}

function binsWords(rnd: () => number): Uint32Array {
  const b = new Uint32Array(BINS), f = new Float32Array(b.buffer);
  for (let c = 0; c < 4096; c++) b[c] = rnd() < 0.3 ? 0xffffffff : Math.floor(rnd() * 4096);
  for (let t = 0; t < 64; t++) {
    b[4096 + t] = Math.floor(rnd() * 4);
    let w = 0; for (let k = 0; k < 6; k++) w |= (rnd() < 0.04 ? 31 : Math.floor(rnd() * 6)) << (5 * k); b[4160 + t] = w >>> 0;
    f[4224 + t] = rnd() < 0.2 ? 0 : rnd() * 1.4; f[4288 + t] = rnd() < 0.25 ? -0.2 : rnd() * 2.2;
  }
  for (let v = 0; v < 17 ** 3; v++) { b[4352 + v] = rnd() < 0.4 ? 0 : ~new Uint32Array(new Float32Array([rnd() * 4]).buffer)[0]! >>> 0; }
  return b;
}

test("level-set formulas: the old text is in the fixture", () => {
  for (const name of ["particleSurfaceFinish", "bulkDepth", "umCubicPhi", "umSurfaceTrace", "nbCellDistance"]) assert.match(oldFunction(name), new RegExp(`^fn ${name}\\(`));
});

test("reconstruction (Ferstl union and its bounds), the crossing guard, heat blends, cell distance and the far sentinel match the old functions", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const threads = 8192, rnd = prng(31), inp = new Float32Array(IN * threads);
    for (let t = 0; t < threads; t++) {
      const o = IN * t;
      for (let a = 0; a < 3; a++) inp[o + a] = 1.2 + rnd() * 13.5;
      inp[o + 3] = (rnd() - 0.6) * 7; inp[o + 4] = rnd() < 0.15 ? 4 + rnd() : rnd() * rnd() * 4.2; inp[o + 5] = rnd() * 1.2;
      inp[o + 6] = rnd(); inp[o + 7] = rnd();
      for (let k = 0; k < 8; k++) inp[o + 8 + k] = (rnd() - 0.45) * 2;
      for (let a = 0; a < 3; a++) inp[o + 16 + a] = rnd() * 2 - 0.5;
      inp[o + 19] = rnd() < 0.1 ? 0 : (rnd() - 0.5) * 0.8;
      for (let k = 0; k < 54; k++) inp[o + 26 + k] = rnd() < 0.2 ? -1 : rnd();
    }
    const io = { inp, threads, outWords: OUT * threads, extra: { 4: binsWords(prng(5)) } };
    const a = await runParity(device, OLD_UNION, io), b = await runParity(device, NEW_UNION, io);
    const r = ulpCompare(a, b);
    console.log(`parity union/guard/heat/distance/far: ${threads} vertices, ${r.differ} words differ, max ${r.maxUlp} ulp`);
    assert.equal(r.differ, 0, `first differing word ${r.first} (thread ${Math.floor(r.first! / OUT)}, output ${r.first! % OUT})`);
  });
});

test("phi advection (the surface trace, trilinear and trace helpers bit for bit; the limited tricubic with held taps within two ulps of its taps) matches the old functions", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const threads = 8192, rnd = prng(37), inp = new Float32Array(ADVECT_IN * threads), words = new Uint32Array(inp.buffer);
    for (let t = 0; t < threads; t++) {
      const o = ADVECT_IN * t;
      for (let a = 0; a < 6; a++) inp[o + a] = 8 + rnd() * 48;
      if (t % 5 === 0) inp[o + 4] = Math.floor(inp[o + 4]!); // a lattice plane: the limiter's one-sided case
      inp[o + 6] = rnd(); words[o + 7] = Math.floor(rnd() * 4096); inp[o + 8] = rnd();
      inp[o + 9] = t % 17 === 0 ? 0 : rnd() * 40; inp[o + 10] = rnd() * 0.05; inp[o + 11] = rnd();
    }
    const io = { inp, threads, outWords: ADVECT_OUT * threads, extra: { 4: new Uint32Array(BINS), 5: new Uint32Array(4), 7: phiTable() } };
    const a = await runParity(device, OLD_ADVECT, io), b = await runParity(device, NEW_ADVECT, io);
    const r = ulpCompare(a, b);
    const hist = new Map<number, number>();
    for (let w = 0; w < a.length; w++) if (a[w] !== b[w]) hist.set(w % ADVECT_OUT, (hist.get(w % ADVECT_OUT) ?? 0) + 1);
    console.log(`parity phi advection: ${threads} vertices, ${r.differ} words differ (by output ${JSON.stringify([...hist])}), max ${r.maxUlp} ulp, max abs ${r.maxAbs}`);
    if (r.differ) {
      const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer);
      const t0 = Math.floor(r.first! / ADVECT_OUT), tw = Math.floor(r.worst! / ADVECT_OUT);
      const byWidth = [0, 0];
      for (let t = 0; t < threads; t++) if (a[ADVECT_OUT * t + 4] !== b[ADVECT_OUT * t + 4]) byWidth[inp[ADVECT_IN * t + 6]! > 0.5 ? 1 : 0]++;
      console.log(`cubic differs at width 1 / 4: ${byWidth.join(" / ")}`);
      for (const t of [t0, tw]) console.log(`thread ${t} in ${Array.from(inp.subarray(ADVECT_IN * t, ADVECT_IN * t + ADVECT_IN), x => x.toPrecision(8)).join(" ")} mask ${words[ADVECT_IN * t + 7]}\n old ${Array.from(fa.subarray(ADVECT_OUT * t, ADVECT_OUT * t + ADVECT_OUT), x => x.toPrecision(9)).join(" ")}\n new ${Array.from(fb.subarray(ADVECT_OUT * t, ADVECT_OUT * t + ADVECT_OUT), x => x.toPrecision(9)).join(" ")}`);
    }
    // Bitwise but for the tricubic, whose 64-term sum Metal's compiler reassociates by the code around it (the old
    // kernel's tap loop sits in a width branch beside the solid and buried paths): it is held to TRICUBIC_BOUND, two
    // ulps of the taps' range (|φ| < 32 cells here; 2^-18 is one); one was seen.
    const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer);
    let cubicAbs = 0;
    for (let w = 0; w < a.length; w++) {
      if (a[w] === b[w]) continue;
      assert.equal(w % ADVECT_OUT, 4, `first differing word ${w} (thread ${Math.floor(w / ADVECT_OUT)}, output ${w % ADVECT_OUT})`);
      cubicAbs = Math.max(cubicAbs, Math.abs(fa[w]! - fb[w]!));
    }
    assert.ok(cubicAbs <= TRICUBIC_BOUND, `tricubic differs by ${cubicAbs}, over ${TRICUBIC_BOUND}`);
  });
});
