/** Parity of lane F's velocity formulas (velocity-formulas.wgsl.ts) with the old engine's text (parity/uniform-dam.wgsl).
 *
 * As particles-formulas.test.ts: the old functions compile whole (or as checked excerpts where they address the old
 * storage) with stubs for what they read, against kernels of the new pure functions over the same stubs, on random
 * inputs; the outputs' bits are compared. Every field a formula reads is data (inputs, or a velocity piecewise
 * constant a cell from an exact hash), so no stub's arithmetic composes with its caller. */
import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { nbfDeltaWGSL, nbfExtensionWGSL, nbfMomentumWGSL, nbfP2GWGSL, nbfRemapWGSL, nbfVelocitySupported } from "./velocity-formulas.wgsl";
import { oldExcerpt, oldFunction, parityCode, parityGPU, prng, runParity, ulpCompare, withParityDevice, type ParityKernel } from "./parity/harness";

const old = (...names: string[]) => names.map(oldFunction).join("\n");
const words = (f: Float32Array) => new Uint32Array(f.buffer, f.byteOffset, f.length);

// ---- p2g's deposit: one sample a thread onto its 81 face nodes; the words land in slots in the old visiting order.
const DEPOSIT_IN = 8, DEPOSIT_NODES = 81;
const DEPOSIT_WORLD = /* wgsl */`
@group(0) @binding(3) var<storage,read_write> pbMomentum:array<atomic<i32>>;
@group(0) @binding(4) var<storage,read_write> pbWeight:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> nodes:array<u32>;
var<private> slotNext:u32;
fn slotOf(node:vec3i)->u32{let s=slotNext;slotNext++;nodes[s]=bitcast<u32>(node.x)*65536u+bitcast<u32>(node.y)*256u+bitcast<u32>(node.z);return s;}
`;
const DEPOSIT_BODY = (scatter: string) => `let i=${DEPOSIT_IN}u*t;slotNext=${DEPOSIT_NODES}u*t;_=atomicLoad(&pbMomentum[0]);_=atomicLoad(&pbWeight[0]);
 ${scatter}(v3(i),v3(i+3u),f(i+6u));
 for(var k=0u;k<${DEPOSIT_NODES}u;k++){let s=${DEPOSIT_NODES}u*t+k;outp[3u*s]=bitcast<u32>(atomicLoad(&pbMomentum[s]));outp[3u*s+1u]=atomicLoad(&pbWeight[s]);outp[3u*s+2u]=nodes[s];}`;
const OLD_DEPOSIT: ParityKernel = {
  prelude: `${DEPOSIT_WORLD}
const PB_MOMENTUM:f32=8192.0;const PB_WEIGHT:f32=65536.0;
fn pbSlot(axis:u32,node:vec3i)->u32{return slotOf(node);}
${old("pbQuadratic", "pbAdd", "pbScatter")}`,
  body: DEPOSIT_BODY("pbScatter"),
};
const NEW_DEPOSIT: ParityKernel = {
  prelude: `${DEPOSIT_WORLD}
fn nbxDeposit(axis:u32,node:vec3i,momentum:i32,weight:u32){let s=slotOf(node);atomicAdd(&pbMomentum[s],momentum);atomicAdd(&pbWeight[s],weight);}
fn nbxTransferHeat(tile:u32)->vec2f{return vec2f(0.0);}
fn nbxTransferDepth(vertex:vec3u)->f32{return 0.0;}
${nbfP2GWGSL}`,
  body: DEPOSIT_BODY("nbfP2GScatter"),
};

// ---- p2g's resolve: a cell's three +faces; their depth, window, expected mass and heats, then the new velocities.
// inp a thread: 27 heats, 27 surface heats, 125 window depths, the cell (3, 0..3), original (4), momenta (3),
// weights (3), thaw (3); 196 words.
const RESOLVE_IN = 196, RESOLVE_OUT = 18;
const RESOLVE_WORLD = /* wgsl */`
var<private> nbTransferHeat:array<f32,27>;var<private> nbTransferHot:array<f32,27>;var<private> nbTransferDepth:array<f32,125>;
fn loadWindow(i:u32){for(var k=0u;k<27u;k++){nbTransferHeat[k]=f(i+k);nbTransferHot[k]=f(i+27u+k);}for(var k=0u;k<125u;k++){nbTransferDepth[k]=f(i+54u+k);}}
`;
const RESOLVE_HEAD = `let i=${RESOLVE_IN}u*t;let o=${RESOLVE_OUT}u*t;loadWindow(i);
 let local=vec3u(inp[i+179u],inp[i+180u],inp[i+181u]);let original=vec4f(f(i+182u),f(i+183u),f(i+184u),f(i+185u));
 let momenta=vec3u(inp[i+186u],inp[i+187u],inp[i+188u]);let weights=vec3u(inp[i+189u],inp[i+190u],inp[i+191u]);let thaw=v3(i+192u);
 var properties:array<vec3f,3>;`;
const OLD_RESOLVE: ParityKernel = {
  prelude: `${RESOLVE_WORLD}
const PB_MOMENTUM:f32=8192.0;const PB_WEIGHT:f32=65536.0;
${old("umCorner", "nbTransferTheta")}
fn face(local:vec3u,axis:u32,o:u32)->vec3f{
 var near=vec2f(0.0);var expected=1.0;var low=local;low[axis]+=1u;
 var depth=0.0;
 ${oldExcerpt("transferPropertiesJob", "for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);if(bit[axis]!=0u){continue;}let q=low+bit;depth+=0.25*nbTransferDepth[q.x+5u*(q.y+5u*q.z)];}")}
 let window=${oldExcerpt("transferPropertiesJob", "depth<=1.5&&depth>=-2.0")};
 if(window){var q=vec3f(local)+0.5;q[axis]+=0.5;near=nbTransferTheta(q);expected=${oldExcerpt("transferPropertiesJob", "max(1.0,8.0*clamp(0.5-depth,0.0,1.0))")};}
 put(o,depth);outp[o+1u]=select(0u,1u,window);put(o+2u,expected);put(o+3u,near.x);put(o+4u,near.y);
 return vec3f(near,expected);
}
fn resolve(original:vec4f,momenta:vec3u,weights:vec3u,properties:array<vec3f,3>,thaw:vec3f)->vec3f{
 var value=original.xyz;
 for(var axis=0u;axis<3u;axis++){
  ${oldExcerpt("transferPropertiesJob", `let mass=f32(weights[axis])/PB_WEIGHT;
   if(mass<1e-5){continue;}
   let momentum=f32(bitcast<i32>(momenta[axis]))/PB_MOMENTUM;
   let near=properties[axis];let transition=near.x*mix(min(1.0,mass/near.z),1.0,near.y);
   value[axis]=mix(original[axis],momentum/mass+thaw[axis],transition);`)}
 }
 return value;
}`,
  body: `${RESOLVE_HEAD}
 for(var axis=0u;axis<3u;axis++){properties[axis]=face(local,axis,o+5u*axis);}
 put3(o+15u,resolve(original,momenta,weights,properties,thaw));`,
};
const NEW_RESOLVE: ParityKernel = {
  prelude: `${RESOLVE_WORLD}
fn nbxDeposit(axis:u32,node:vec3i,momentum:i32,weight:u32){}
fn nbxTransferHeat(tile:u32)->vec2f{return vec2f(nbTransferHeat[tile],nbTransferHot[tile]);}
fn nbxTransferDepth(v:vec3u)->f32{return nbTransferDepth[v.x+5u*(v.y+5u*v.z)];}
${nbfP2GWGSL}
fn face(local:vec3u,axis:u32,o:u32)->vec3f{
 var near=vec2f(0.0);var expected=1.0;var low=local;low[axis]+=1u;
 let depth=nbfFaceDepth(low,axis);let window=nbfP2GWindow(depth);
 if(window){var q=vec3f(local)+0.5;q[axis]+=0.5;near=nbfTransferTheta(q);expected=nbfP2GExpected(depth);}
 put(o,depth);outp[o+1u]=select(0u,1u,window);put(o+2u,expected);put(o+3u,near.x);put(o+4u,near.y);
 return vec3f(near,expected);
}`,
  body: `${RESOLVE_HEAD}
 for(var axis=0u;axis<3u;axis++){properties[axis]=face(local,axis,o+5u*axis);}
 var value=original.xyz;
 for(var axis=0u;axis<3u;axis++){value[axis]=nbfP2GResolve(original[axis],momenta[axis],weights[axis],properties[axis],thaw[axis]);}
 put3(o+15u,value);`,
};

// ---- uAdvect's momentum trace and the forces. inp a thread: position (3), axis, dt, phi, neighbour phi, widths
// (2), coarse, value, g; 12 words. The field: piecewise constant a cell (an exact hash, m/s), the hold the box.
const MOMENTUM_IN = 12, MOMENTUM_OUT = 4;
const MOMENTUM_WORLD = /* wgsl */`
@group(0) @binding(3) var<storage,read_write> claims:array<atomic<u32>>;
fn field(p:vec3f)->vec3f{
 let c=vec3u(clamp(vec3i(floor(p)),vec3i(0),vec3i(31)));let h=(c.x*73856093u)^(c.y*19349663u)^(c.z*83492791u);
 return (vec3f(f32(h&255u),f32((h>>8u)&255u),f32((h>>16u)&255u))-127.5)/64.0;
}
fn sampled(p:vec3f,lo:u32,hi:u32)->vec3f{let full=field(p);var v=vec3f(0.0);for(var a=lo;a<hi;a++){v[a]=full[a];}return v;}
fn held(p:vec3f)->vec3f{return clamp(p,vec3f(1.0),vec3f(31.0));}
struct UMOwner {tile:u32,lane:u32,width:u32,index:u32}
struct UMForceParams {hDt:vec4f,physical:vec4f,controls:vec4f,gravity:vec4f}
var<private> force:UMForceParams;var<private> phis:vec2f;
`;
const MOMENTUM_HEAD = `let i=${MOMENTUM_IN}u*t;let o=${MOMENTUM_OUT}u*t;_=atomicLoad(&claims[0]);
 let h=vec3f(0.05,0.04,0.05);let axis=inp[i+3u]%3u;let dt=f(i+4u);
 force=UMForceParams(vec4f(h,dt),vec4f(0.0),vec4f(0.0),vec4f(0.0,f(i+11u),0.0,0.0));phis=vec2f(f(i+5u),f(i+6u));
 let own=UMOwner(0u,0u,inp[i+7u],0u);let neighbor=UMOwner(0u,0u,inp[i+8u],1u);let coarse=inp[i+9u]!=0u;var value=f(i+10u);`;
const OLD_MOMENTUM: ParityKernel = {
  prelude: `${MOMENTUM_WORLD}
const UM_TRACE_SEGMENTS:u32=128u;
fn umUnitSample(p:vec3f,lo:u32,hi:u32)->vec3f{return sampled(p,lo,hi);}
fn umClampMomentum(p:vec3f)->vec3f{return held(p);}
fn umTraceCapped(){atomicAdd(&claims[1],1u);}
fn umCellPhi(owner:UMOwner)->f32{return phis[owner.index];}
fn umCoarseMass(owner:UMOwner)->bool{return coarseFlag;}
var<private> coarseFlag:bool;
${old("umUnitTrace", "umOccupancy")}`,
  body: `${MOMENTUM_HEAD}coarseFlag=inp[i+9u]!=0u;
 put(o,umUnitTrace(v3(i),axis,dt,h));
 let occupancy=umOccupancy(own);put(o+1u,occupancy);
 let gravity=${oldExcerpt("forcesFineJob", "occupancy>1e-5||umOccupancy(neighbor)>1e-5||umCoarseMass(neighbor)")};
 outp[o+2u]=select(0u,1u,gravity);
 if(gravity){${oldExcerpt("forcesFineJob", "value[axis]+=force.hDt.w*force.gravity[axis];").replace("value[axis]", "value").replace("gravity[axis]", "gravity[1]")}}
 put(o+3u,value);`,
};
const NEW_MOMENTUM: ParityKernel = {
  prelude: `${MOMENTUM_WORLD}
fn nbxMomentumSample(p:vec3f,lo:u32,hi:u32)->vec3f{return sampled(p,lo,hi);}
fn nbxMomentumHeld(p:vec3f)->vec3f{return held(p);}
fn nbxTraceCapped(){atomicAdd(&claims[1],1u);}
${nbfMomentumWGSL}`,
  body: `${MOMENTUM_HEAD}
 put(o,nbfMomentumTrace(v3(i),axis,dt,h));
 let occupancy=nbfOccupancy(phis.x,force.hDt.y,own.width);put(o+1u,occupancy);
 let gravity=nbfGravityFace(occupancy,nbfOccupancy(phis.y,force.hDt.y,neighbor.width),coarse);
 outp[o+2u]=select(0u,1u,gravity);
 if(gravity){value=nbfGravityKick(value,force.hDt.w,force.gravity[1]);}
 put(o+3u,value);`,
};

// ---- uExtend: the fast-sweeping update over random states, the seed codes, and the 4h prolongation.
// inp a thread: 7 states (14), h (3), the face (3), axis, seed code; 22 words.
const EXTEND_IN = 22, EXTEND_OUT = 8;
const EXTEND_HEAD = `let i=${EXTEND_IN}u*t;let o=${EXTEND_OUT}u*t;
 var s:array<Ufx,7>;for(var k=0u;k<7u;k++){s[k]=Ufx(f(i+2u*k),f(i+2u*k+1u));}
 let h=v3(i+14u);let p=vec3i(vec3u(inp[i+17u],inp[i+18u],inp[i+19u]));let a=inp[i+20u]%3u;
 let r=ufxCombine(s[0],s[1],s[2],s[3],s[4],s[5],s[6],h);put(o,r.v);put(o+1u,r.d);
 put(o+2u,ufxSeedCode(inp[i+21u]%4u));outp[o+3u]=select(0u,1u,ufxReached(s[1].d));`;
const EXTEND_WORLD = /* wgsl */`
var<private> prolongLo:Ufx;var<private> prolongHi:Ufx;var<private> prolongC:vec3i;
`;
const OLD_EXTEND: ParityKernel = {
  prelude: `alias UfxV=f32;\nstruct Ufx{v:UfxV,d:f32}\n${EXTEND_WORLD}
fn ufxAxis(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn ufxAt4(C:vec3i,base:vec3i,a:u32)->Ufx{if(all(C==prolongC)){return prolongHi;}return prolongLo;}
${old("ufxReached", "ufxSeedCode", "ufxTerm", "ufxCombine", "ufxProlong")}`,
  body: `${EXTEND_HEAD}
 prolongC=p>>vec3u(2u);prolongLo=s[3];prolongHi=s[4];
 let e=ufxProlong(p,a,vec3i(0));put(o+4u,e.v);put(o+5u,e.d);put(o+6u,0.0);put(o+7u,0.0);`,
};
const NEW_EXTEND: ParityKernel = {
  prelude: `${nbfExtensionWGSL}\n${EXTEND_WORLD}`,
  body: `${EXTEND_HEAD}
 let e=nbfExtProlong(s[3],s[4],(p[a]&3)+1);put(o+4u,e.v);put(o+5u,e.d);put(o+6u,0.0);put(o+7u,0.0);`,
};

// ---- uDelta: the FLIP delta at a sample from its tile's 3 x 180 window faces. inp a thread: p (3, 0..4 a tile),
// 540 faces; 544 words.
const DELTA_IN = 544, DELTA_OUT = 3;
const DELTA_WORLD = /* wgsl */`
var<private> ndDelta:array<f32,540>;
fn loadDelta(i:u32){for(var k=0u;k<540u;k++){ndDelta[k]=f(i+3u+k);}}
`;
const OLD_DELTA: ParityKernel = {
  prelude: `${DELTA_WORLD}\nconst NB_DELTA_NONE:f32=3e38;\n${old("umCorner", "ndSample")}`,
  body: `let i=${DELTA_IN}u*t;loadDelta(i);put3(${DELTA_OUT}u*t,ndSample(v3(i)));`,
};
const NEW_DELTA: ParityKernel = {
  prelude: `${DELTA_WORLD}
fn nbxDelta(axis:u32,r:vec3u)->f32{return ndDelta[axis*180u+r[axis]+5u*r[(axis+1u)%3u]+30u*r[(axis+2u)%3u]];}
${nbfDeltaWGSL}`,
  body: `let i=${DELTA_IN}u*t;loadDelta(i);put3(${DELTA_OUT}u*t,nbfDeltaSample(v3(i)));`,
};

// ---- The remap: φ's refine, a 4h face's slope, an h face's sample and a 4h face's restriction. inp a thread:
// corners (8), vertex, value, below (2), above (2), plane, origin (3), p (3), width, axis, low and high faces,
// slopes (12, the tile's), 16 weights and 16 samples; 72 words.
const REMAP_IN = 72, REMAP_OUT = 4;
const REMAP_WORLD = /* wgsl */`
var<private> tileCorners:array<f32,8>;var<private> tileSlopes:array<f32,12>;
fn loadRemap(i:u32){for(var k=0u;k<8u;k++){tileCorners[k]=f(i+k);}for(var k=0u;k<12u;k++){tileSlopes[k]=f(i+28u+k);}}
`;
const REMAP_HEAD = `let i=${REMAP_IN}u*t;let o=${REMAP_OUT}u*t;loadRemap(i);
 let v=inp[i+8u]%125u;let value=f(i+9u);let below=vec2f(f(i+10u),f(i+11u));let above=vec2f(f(i+12u),f(i+13u));
 let plane=bitcast<i32>(inp[i+14u]);let origin=vec3u(inp[i+15u],inp[i+16u],inp[i+17u]);let p=vec3u(inp[i+18u],inp[i+19u],inp[i+20u]);
 let width=inp[i+21u];let axis=inp[i+22u]%3u;let lowFace=f(i+23u);let highFace=f(i+24u);`;
const OLD_REMAP: ParityKernel = {
  prelude: `${REMAP_WORLD}
const REMAP_SLOPE_LIMIT:f32=4.0/3.0;
${old("umCorner", "d4Sum8")}
var<private> tileVertices:array<f32,125>;
fn refine(v:u32)->f32{
 ${oldExcerpt("remapTileCells", `let t=vec3f(umCorner(v,5u))*0.25;var values:array<f32,8>;
   for(var k=0u;k<8u;k++){let w=select(vec3f(1)-t,t,umCorner(k,2u)!=vec3u(0));values[k]=tileCorners[k]*w.x*w.y*w.z;}
   tileVertices[v]=d4Sum8(values);`)}
 return tileVertices[v];
}
fn slope(value:f32,below:vec2f,above:vec2f)->f32{
 var slope=0.0;
 ${oldExcerpt("remapSlope", `let lower=select(0.0,value-below.x/16.0,below.y==16.0);
   let upper=select(0.0,above.x/16.0-value,above.y==16.0);
   if(lower*upper>0.0){slope=sign(lower)*min(abs(lower+upper)/8.0,REMAP_SLOPE_LIMIT*min(abs(lower),abs(upper))/4.0);}`)}
 return slope;
}
struct RemapOwner{width:u32}
fn sample(p:vec3u,origin:vec3u,plane:i32,axis:u32,width:u32,lowFace:f32,highFace:f32)->f32{
 let o=RemapOwner(width);
 let t=${oldExcerpt("remapSample", "f32(plane+1-i32(origin[axis]))/f32(o.width)")};
 var low=lowFace;var high=highFace;
 if(o.width==4u){
  ${oldExcerpt("remapSample", `let across=vec2f(f32(p[(axis+1u)%3u]-origin[(axis+1u)%3u]),f32(p[(axis+2u)%3u]-origin[(axis+2u)%3u]))-vec2f(1.5);
  low+=dot(vec2f(tileSlopes[4u*axis],tileSlopes[4u*axis+1u]),across);
  high+=dot(vec2f(tileSlopes[4u*axis+2u],tileSlopes[4u*axis+3u]),across);`)}
 }
 ${oldExcerpt("remapSample", "return mix(low,high,t);")}
}
fn restriction(i:u32)->f32{
 var value=0.0;var weight=0.0;
 for(var k=0u;k<16u;k++){let w=f(i+k);let s=f(i+16u+k);${oldExcerpt("remapFace", "value+=w*remapSample(p,f.anchor[axis],axis);weight+=w;").replace("remapSample(p,f.anchor[axis],axis)", "s")}}
 ${oldExcerpt("remapFace", "if(weight>0.0){value/=weight;}")}
 return value;
}`,
  body: `${REMAP_HEAD}
 put(o,refine(v));put(o+1u,slope(value,below,above));put(o+2u,sample(p,origin,plane,axis,width,lowFace,highFace));put(o+3u,restriction(i+40u));`,
};
const NEW_REMAP: ParityKernel = {
  prelude: `${REMAP_WORLD}\n${nbfRemapWGSL}
fn restriction(i:u32)->f32{var acc=vec2f(0.0);for(var k=0u;k<16u;k++){acc=nbfRemapFaceAdd(acc,f(i+k),f(i+16u+k));}return nbfRemapFaceMean(acc);}`,
  body: `${REMAP_HEAD}
 let u=(axis+1u)%3u;let w=(axis+2u)%3u;
 let along=nbfRemapAlong(plane,origin[axis],width);
 let across=nbfRemapAcross(vec2u(p[u],p[w]),vec2u(origin[u],origin[w]));
 let slopes=vec4f(tileSlopes[4u*axis],tileSlopes[4u*axis+1u],tileSlopes[4u*axis+2u],tileSlopes[4u*axis+3u]);
 put(o,nbfRemapRefine(tileCorners,v));put(o+1u,nbfRemapSlope(value,below,above));put(o+2u,nbfRemapSample(lowFace,highFace,along,width==4u,slopes,across));put(o+3u,restriction(i+40u));`,
};

const KERNELS = { OLD_DEPOSIT, NEW_DEPOSIT, OLD_RESOLVE, NEW_RESOLVE, OLD_MOMENTUM, NEW_MOMENTUM, OLD_EXTEND, NEW_EXTEND, OLD_DELTA, NEW_DELTA, OLD_REMAP, NEW_REMAP };
// NBF_PARITY_DUMP=DIR writes every kernel's module there, for offline validation (naga DIR/*.wgsl).
if (process.env.NBF_PARITY_DUMP) {
  const dir = process.env.NBF_PARITY_DUMP; mkdirSync(dir, { recursive: true });
  for (const [name, kernel] of Object.entries(KERNELS)) writeFileSync(`${dir}/${name}.wgsl`, parityCode(kernel, 64));
}

test("the brick engine's uAdvect rejects viscosity and surface tension", () => {
  nbfVelocitySupported({ viscosity: 0, surfaceTension: 0 });
  assert.throws(() => nbfVelocitySupported({ viscosity: 1e-6 }), /viscosity/);
  assert.throws(() => nbfVelocitySupported({ surfaceTension: 0.07 }), /surface tension/);
});

async function parity(device: GPUDevice, label: string, oldKernel: ParityKernel, newKernel: ParityKernel, io: Parameters<typeof runParity>[2], out: number): Promise<void> {
  const a = await runParity(device, oldKernel, io), b = await runParity(device, newKernel, io);
  const r = ulpCompare(a, b);
  const hist = new Map<number, number>();
  for (let w = 0; w < a.length; w++) if (a[w] !== b[w]) hist.set(w % out, (hist.get(w % out) ?? 0) + 1);
  console.log(`parity ${label}: ${io.threads} threads, ${io.outWords} words, ${r.differ} differ (by output ${JSON.stringify([...hist])}), max ${r.maxUlp} ulp, max abs ${r.maxAbs}`);
  assert.equal(r.differ, 0, `${label}: first differing word ${r.first} (thread ${Math.floor(r.first! / out)}, output ${r.first! % out})`);
}

test("p2g (the quadratic deposit's nodes and fixed-point words; the face depth, window, expected mass, heats and resolve) matches the old pbScatter and transferPropertiesJob", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const rnd = prng(41);
    {
      const threads = 2048, inp = new Float32Array(DEPOSIT_IN * threads);
      for (let t = 0; t < threads; t++) {
        const o = DEPOSIT_IN * t;
        for (let a = 0; a < 3; a++) inp[o + a] = 4 + rnd() * 8;
        if (t % 7 === 0) inp[o + t % 3] = Math.floor(inp[o + t % 3]!) + 0.5; // on a node plane
        for (let a = 3; a < 6; a++) inp[o + a] = (rnd() - 0.5) * 8;
        inp[o + 6] = 1 + Math.floor(rnd() * 8);
      }
      const n = DEPOSIT_NODES * threads;
      await parity(device, "p2g deposit", OLD_DEPOSIT, NEW_DEPOSIT, { inp, threads, outWords: 3 * n, extra: { 3: new Uint32Array(n), 4: new Uint32Array(n), 5: new Uint32Array(n) } }, 3);
    }
    {
      const threads = 4096, inp = new Float32Array(RESOLVE_IN * threads), w = words(inp);
      for (let t = 0; t < threads; t++) {
        const o = RESOLVE_IN * t;
        for (let k = 0; k < 54; k++) inp[o + k] = rnd() < 0.2 ? -1 : rnd() < 0.2 ? Math.floor(rnd() * 2) : rnd();
        for (let k = 0; k < 125; k++) inp[o + 54 + k] = -3 + rnd() * 5;
        for (let a = 0; a < 3; a++) w[o + 179 + a] = Math.floor(rnd() * 4);
        for (let a = 0; a < 4; a++) inp[o + 182 + a] = (rnd() - 0.5) * 6;
        for (let a = 0; a < 3; a++) { w[o + 186 + a] = (Math.floor((rnd() - 0.5) * 2 ** 21) | 0) >>> 0; w[o + 189 + a] = rnd() < 0.15 ? Math.floor(rnd() * 0.6) : Math.floor(rnd() * 2 ** 20); }
        for (let a = 0; a < 3; a++) inp[o + 192 + a] = rnd() < 0.7 ? 0 : (rnd() - 0.5) * 0.2;
      }
      await parity(device, "p2g resolve", OLD_RESOLVE, NEW_RESOLVE, { inp, threads, outWords: RESOLVE_OUT * threads }, RESOLVE_OUT);
    }
  });
});

test("uAdvect (the momentum trace with its segment cap; occupancy, the gravity face and kick) matches the old umUnitTrace and forcesFineJob", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const rnd = prng(43), threads = 8192, inp = new Float32Array(MOMENTUM_IN * threads), w = words(inp);
    for (let t = 0; t < threads; t++) {
      const o = MOMENTUM_IN * t;
      for (let a = 0; a < 3; a++) inp[o + a] = 2 + rnd() * 28;
      w[o + 3] = t % 3;
      inp[o + 4] = t % 11 === 0 ? 3 + rnd() * 6 : t % 13 === 0 ? 0 : (rnd() < 0.2 ? -1 : 1) * rnd() / 30;
      inp[o + 5] = (rnd() - 0.5) * 0.4; inp[o + 6] = (rnd() - 0.5) * 0.4;
      w[o + 7] = [0, 1, 4][Math.floor(rnd() * 3)]!; w[o + 8] = [0, 1, 4][Math.floor(rnd() * 3)]!; w[o + 9] = rnd() < 0.1 ? 1 : 0;
      inp[o + 10] = (rnd() - 0.5) * 4; inp[o + 11] = -9.81;
    }
    await parity(device, "uAdvect", OLD_MOMENTUM, NEW_MOMENTUM, { inp, threads, outWords: MOMENTUM_OUT * threads, extra: { 3: new Uint32Array(4) } }, MOMENTUM_OUT);
  });
});

test("uExtend (the fast-sweeping update, seed codes, the 4h prolongation) matches the old ufxCombine and ufxProlong", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const rnd = prng(47), threads = 16384, inp = new Float32Array(EXTEND_IN * threads), w = words(inp);
    const distance = () => { const r = rnd(); return r < 0.25 ? 1e20 : r < 0.35 ? -1 : r < 0.45 ? 0 : r < 0.55 ? Math.floor(rnd() * 3) * 0.05 : rnd() * 0.15; };
    for (let t = 0; t < threads; t++) {
      const o = EXTEND_IN * t;
      for (let k = 0; k < 7; k++) { inp[o + 2 * k] = (rnd() - 0.5) * 4; inp[o + 2 * k + 1] = k === 0 && rnd() < 0.5 ? 1e20 : distance(); }
      const h = t % 2 === 0 ? [0.05, 0.05, 0.05] : [0.05, 0.04, 0.2];
      for (let a = 0; a < 3; a++) { inp[o + 14 + a] = h[a]!; w[o + 17 + a] = Math.floor(rnd() * 64); }
      w[o + 20] = t % 3; w[o + 21] = Math.floor(rnd() * 4);
    }
    await parity(device, "uExtend", OLD_EXTEND, NEW_EXTEND, { inp, threads, outWords: EXTEND_OUT * threads }, EXTEND_OUT);
  });
});

test("uDelta (the FLIP delta at a sample, renormalised over the faces that hold one) matches the old ndSample", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const rnd = prng(53), threads = 2048, inp = new Float32Array(DELTA_IN * threads);
    for (let t = 0; t < threads; t++) {
      const o = DELTA_IN * t;
      for (let a = 0; a < 3; a++) inp[o + a] = t % 9 === 0 ? Math.floor(rnd() * 4) : rnd() * 3.999;
      const none = rnd() * 0.6;
      for (let k = 0; k < 540; k++) inp[o + 3 + k] = rnd() < none ? 3e38 : (rnd() - 0.5) * 2;
    }
    await parity(device, "uDelta", OLD_DELTA, NEW_DELTA, { inp, threads, outWords: DELTA_OUT * threads }, DELTA_OUT);
  });
});

test("the remap (φ's refine, the 4h face slope, the h face sample, the 4h face restriction) matches the old remapTileCells, remapSlope, remapSample and remapFace", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const rnd = prng(59), threads = 8192, inp = new Float32Array(REMAP_IN * threads), w = words(inp);
    for (let t = 0; t < threads; t++) {
      const o = REMAP_IN * t;
      for (let k = 0; k < 8; k++) inp[o + k] = (rnd() - 0.5) * 6;
      w[o + 8] = t % 125; inp[o + 9] = (rnd() - 0.5) * 2;
      for (const k of [10, 12]) { inp[o + k] = (rnd() - 0.5) * 32; inp[o + k + 1] = rnd() < 0.7 ? 16 : Math.floor(rnd() * 16); }
      if (t % 5 === 0) inp[o + 12] = 2 * 16 * inp[o + 9]! - inp[o + 10]!; // a linear field
      const width = rnd() < 0.5 ? 4 : 1, axis = t % 3;
      for (let a = 0; a < 3; a++) { w[o + 15 + a] = 4 * Math.floor(rnd() * 16); w[o + 18 + a] = w[o + 15 + a]! + Math.floor(rnd() * width); }
      w[o + 14] = (w[o + 15 + axis]! - 1 + Math.floor(rnd() * (width + 1))) >>> 0;
      w[o + 21] = width; w[o + 22] = axis; inp[o + 23] = (rnd() - 0.5) * 4; inp[o + 24] = (rnd() - 0.5) * 4;
      for (let k = 0; k < 12; k++) inp[o + 28 + k] = (rnd() - 0.5) * 0.5;
      for (let k = 0; k < 16; k++) { inp[o + 40 + k] = t % 7 === 0 ? 0 : rnd() < 0.3 ? 0 : rnd(); inp[o + 56 + k] = (rnd() - 0.5) * 4; }
    }
    await parity(device, "remap", OLD_REMAP, NEW_REMAP, { inp, threads, outWords: REMAP_OUT * threads }, REMAP_OUT);
  });
});
