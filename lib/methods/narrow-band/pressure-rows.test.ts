/** Lane P's row formulas (pressure-rows.wgsl.ts) against the old arena's resolved text
 * (parity/uniform-dam-pressure.wgsl, tools/nbflip-pressure-parity-extract.ts), bit for bit, on seeded states that
 * reach every branch. Whole old functions where they are pure; elsewhere verbatim excerpts over stubs that read the
 * thread's inputs (the method of lane F's formula parity, parity/harness.ts). GPU tests skip without
 * WEBGPU_NODE_MODULE. */
import { BRICK_CELLS, BRICK_STRIDE } from "./brick-abi";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { parityGPU, prng, runParity, ulpCompare, withParityDevice } from "./parity/harness";
import { NP_ACROSS, NP_K, PRESSURE_ROWS_WGSL } from "./pressure-rows.wgsl";

const FIXTURE = resolve(import.meta.dirname, "parity/uniform-dam-pressure.wgsl");
let fixture: Map<string, string> | undefined;
function oldFunction(name: string): string {
  if (!fixture) {
    fixture = new Map();
    for (const part of readFileSync(FIXTURE, "utf8").split(/^\/\/ ==== /m).slice(1)) fixture.set(/^(\w+)/.exec(part)![1]!, part.slice(part.indexOf("\n") + 1).trimEnd());
  }
  return fixture.get(name) ?? assert.fail(`pressure parity fixture has no ${name}`);
}
function oldExcerpt(name: string, excerpt: string): string {
  // Line comments are dropped on both sides (the excerpts leave out the old text's comments), then whitespace runs.
  const squash = (s: string) => s.replace(/\/\/[^\n]*/g, "").replace(/\s+/g, " ").trim();
  assert.ok(squash(oldFunction(name)).includes(squash(excerpt)), `excerpt not verbatim in old ${name}:\n${excerpt}`);
  return excerpt;
}
const f32 = (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0]!;
const OLD_CONSTS = /* wgsl */`
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;const K_SEAM:u32=6u;
const N_NONE:u32=0u;const N_NATIVE:u32=1u;const N_SEAM:u32=2u;const N_GHOST:u32=3u;const N_WALL:u32=4u;const N_HELD:u32=5u;const N_CLOSED:u32=${1 << 21}u;
const UM_RESIDENT:u32=0u;const UM_AIR:u32=1u;const UM_SOLID:u32=2u;
struct BandParams {hDt:vec4f,policy:vec4f}
var<private> params:BandParams;
var<private> T:u32;
fn fw(i:u32)->f32{return bitcast<f32>(inp[i]);}
`;
function same(a: Uint32Array, b: Uint32Array, what: string) {
  let differ = 0, first = -1;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { differ++; if (first < 0) first = i; }
  assert.equal(differ, 0, `${what}: ${differ} of ${a.length} words differ (first ${first}: old ${a[first]} new ${b[first]})`);
}
/** Every row of `rows` in a thread: inputs as words. */
function words(rows: number[][]): Uint32Array { return new Uint32Array(rows.flat()); }

test("pressure rows: the band face (pRows) bit for bit, every kind and solid branch", { skip: !parityGPU }, async () => {
  const N = 8192, W = 10, rnd = prng(11);
  const rows: number[][] = [], kinds = new Set<number>();
  const pick = <T,>(xs: readonly T[]) => xs[Math.floor(rnd() * xs.length)]!;
  for (let t = 0; t < N; t++) {
    const across = rnd() < 0.15 ? pick([NP_ACROSS.AIR, NP_ACROSS.WALL]) : NP_ACROSS.RESIDENT;
    const own = (rnd() - 0.5) * 3, nPhi = (rnd() - 0.5) * 3, facePhi = (rnd() - 0.5) * 3;
    const volume = pick([1, 1, 0.5, 1e-7, rnd()]), open = pick([1, 0, 0.05, rnd()]), qOpen = pick([1, 0, 0.05, rnd()]);
    let bits = 0;
    for (let k = 0; k < 10; k++) if (rnd() < (k === 0 ? 0.4 : k === 1 ? 0.3 : k === 4 ? 0.5 : 0.35)) bits |= 1 << (k + 1);
    rows.push([across, f32(own), f32(pick([1, 0.5, 2.25])), f32(pick([1, 1.5, 0.75])), bits, f32(nPhi), f32(facePhi), f32(volume), f32(open), f32(qOpen)]);
  }
  // Flags (bits from 2): band, native, nWet, nNeumann, solidClear, shut, inside, qIsOpen, qInside, liquid.
  const decode = /* wgsl */`T=${W}u*t;let b=inp[T+4u];let own=f(T+1u);let h=f(T+2u);let liquid=(b&1024u)!=0u;let solidClear=(b&32u)!=0u;let shut=(b&64u)!=0u;let inside=(b&128u)!=0u;let open=f(T+8u);`;
  const old = {
    prelude: OLD_CONSTS + /* wgsl */`
struct BNeighbour {phi:f32,wet:bool,neumann:bool}
fn umCellKind(q:vec3i)->u32{return inp[T];}
fn bNeighbour(q:vec3i)->BNeighbour{let b=inp[T+4u];return BNeighbour(bitcast<f32>(inp[T+5u]),(b&8u)!=0u,(b&16u)!=0u);}
fn umCellIsOpen(q:vec3i)->bool{return (inp[T+4u]&256u)!=0u;}
fn umCellInsideSolid(q:vec3i)->bool{return (inp[T+4u]&512u)!=0u;}
fn umCellOpen(q:vec3i)->f32{return bitcast<f32>(inp[T+9u]);}
fn bFacePhi(p:vec3i,axis:u32,sign:i32)->f32{return bitcast<f32>(inp[T+6u]);}
fn pBandCell(q:vec3i)->bool{return (inp[T+4u]&2u)!=0u;}
fn bNative(q:vec3i)->bool{return (inp[T+4u]&4u)!=0u;}
${oldFunction("bTheta")}
${oldFunction("bPairTheta")}
${oldFunction("bFaceTheta")}`,
    body: decode + /* wgsl */`params.policy=vec4f(0.0,0.0,0.0,f(T+3u));
 let axis=0u;let sign=1;let l=vec3u(3u);let p=vec3i(0);var q=p;q[axis]+=sign;
 var volume=f(T+7u);var sealed=false;var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
 ${oldExcerpt("pRows", /* wgsl */`var across=UM_RESIDENT;if(l[axis]==select(0u,3u,sign>0)){across=umCellKind(q);}
   if(across!=UM_RESIDENT){
    if(across==UM_AIR){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }`)}
 ${oldExcerpt("pRows", /* wgsl */`else{
    let n=bNeighbour(q);
    var theta=1.0;
    var acrossShut=false;if(!solidClear&&liquid!=n.wet){acrossShut=!umCellIsOpen(q);}
    if(liquid!=n.wet){
     if(!solidClear&&(shut||acrossShut)){theta=bPairTheta(own,liquid,n.phi,n.wet);}
     else if(inside&&umCellInsideSolid(q)&&open<0.1&&umCellOpen(q)<0.1){theta=1.0;}
     else{theta=bFaceTheta(bFacePhi(p,axis,sign),own,liquid,n.phi,n.wet);}
    }
    if(pBandCell(q)){kind=K_BAND;coefficient=1.0/(h*h*theta);}
    else if(bNative(q)){kind=K_SEAM;coefficient=1.0/(2.5*h*h*select(theta,1.0,liquid));}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*theta);}
    if(!solidClear&&kind!=K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&acrossShut);}
   }
   if(volume<=1e-6||sealed){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else if(kind==K_NEUMANN){fraction=volume;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}`)}
 outp[3u*t]=kind;put(3u*t+1u,coefficient);put(3u*t+2u,fraction);`,
  };
  const now = {
    prelude: OLD_CONSTS + PRESSURE_ROWS_WGSL,
    body: decode + /* wgsl */`
 let r=npBandFace(inp[T],own,liquid,h,f(T+3u),(b&2u)!=0u,(b&4u)!=0u,NpCell(f(T+5u),(b&8u)!=0u,(b&16u)!=0u),f(T+6u),f(T+7u),
   solidClear,shut,inside,open,(b&256u)!=0u,(b&512u)!=0u,f(T+9u));
 outp[3u*t]=r.kind;put(3u*t+1u,r.coefficient);put(3u*t+2u,r.fraction);`,
  };
  await withParityDevice(async device => {
    const io = { inp: words(rows), threads: N, outWords: 3 * N };
    const a = await runParity(device, old, io), b = await runParity(device, now, io);
    for (let t = 0; t < N; t++) kinds.add(a[3 * t]!);
    for (const k of Object.values(NP_K)) assert.ok(kinds.has(k), `kind ${k} not reached`);
    same(a, b, "band face");
  });
});

test("pressure rows: the seam head (bSeamHead) to 4 ulp of its terms", { skip: !parityGPU }, async () => {
  const N = 2048, S = BRICK_STRIDE, rnd = prng(23);
  // pool: the kinds plane (slot t's BRICK_STRIDE words: the row at its local cell, its 6 neighbours at BRICK_CELLS + g), then the base plane.
  const pool = new Uint32Array(2 * S * N), inp: number[] = [];
  for (let t = 0; t < N; t++) {
    const local = Math.floor(rnd() * BRICK_CELLS), f = Math.floor(rnd() * 6), at = S * t + local;
    let kinds = rnd() < 0.85 ? 0x80000000 : 0;
    for (let g = 0; g < 6; g++) kinds |= (rnd() < 0.7 ? NP_K.BAND : Math.floor(rnd() * 7)) << (3 * g);
    pool[at] = kinds >>> 0; pool[S * N + at] = f32((rnd() - 0.5) * 2000);
    for (let g = 0; g < 6; g++) { const m = S * t + BRICK_CELLS + g; pool[m] = rnd() < 0.8 ? 0x80000000 : 0; pool[S * N + m] = f32((rnd() - 0.5) * 2000); }
    inp.push(at, f);
  }
  const stubs = OLD_CONSTS + /* wgsl */`
@group(0) @binding(3) var<storage,read> pool:array<u32>;
const UBA_STRIDE:u32=${S}u;const P_KINDS:u32=0u;const P_BASE:u32=${S * N}u;
fn pF(base:u32,at:u32)->f32{return bitcast<f32>(pool[base+at]);}
fn pNear0(at:u32,slot:u32,lb:vec3u,f:u32)->u32{return UBA_STRIDE*slot+512u+f;}
${oldFunction("bSign")}
${oldFunction("bLiquid")}
${oldFunction("bFaceKind")}`;
  const old = { prelude: stubs + oldFunction("bSeamHead"), body: `put(t,bSeamHead(inp[2u*t],inp[2u*t+1u]));` };
  const now = {
    prelude: stubs + PRESSURE_ROWS_WGSL, body: /* wgsl */`
 let at=inp[2u*t];let f=inp[2u*t+1u];let kinds=pool[P_KINDS+at];let local=at%UBA_STRIDE;let lb=vec3u(local&7u,(local>>3u)&7u,local>>6u);
 var usable=0u;var base:array<f32,6>;
 for(var g=0u;g<6u;g++){let m=pNear0(at,at/UBA_STRIDE,lb,g);base[g]=pF(P_BASE,m);if(bFaceKind(kinds,g)==K_BAND&&bLiquid(pool[P_KINDS+m])){usable|=1u<<g;}}
 put(t,npSeamHead(bLiquid(kinds),pF(P_BASE,at),f,lb,usable,base));`,
  };
  await withParityDevice(async device => {
    const io = { inp: new Uint32Array(inp), extra: { 3: pool }, threads: N, outWords: N };
    const a = await runParity(device, old, io), b = await runParity(device, now, io);
    assert.ok(new Float32Array(a.buffer).filter(x => x !== 0).length > N / 2, "most heads nonzero");
    // The one difference is contraction: head += rise/count * offset is fused or not as Metal's compiler chooses
    // for the code around it (lane F's finding). Each fused or unfused step errs by at most one rounding of its
    // own term (Higham 2002, §2.6), and the two tangential terms can cancel to a head far below either, so each
    // head is held to 4 ulp of the sum of its terms' magnitudes, not of the head.
    const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer), pf = new Float32Array(pool.buffer);
    let worst = 0;
    for (let t = 0; t < N; t++) {
      const at = inp[2 * t]!, f = inp[2 * t + 1]!, kinds = pool[at]!, local = at % S, lb = [local & 7, (local >> 3) & 7, local >> 6];
      if ((kinds & 0x80000000) === 0) { assert.equal(fb[t], 0); continue; }
      let scale = 0;
      for (let axis = 0; axis < 3; axis++) {
        if (axis === (f >> 1)) continue;
        let rise = 0, count = 0;
        for (let side = 0; side < 2; side++) {
          const g = 2 * axis + side, m = S * t + BRICK_CELLS + g;
          if (((kinds >>> (3 * g)) & 7) === NP_K.BAND && (pool[m]! & 0x80000000) !== 0) { rise += Math.abs(pf[S * N + m]! - pf[S * N + at]!); count++; }
        }
        if (count > 0) scale += rise / count * Math.abs((lb[axis]! & 3) - 1.5);
      }
      const err = Math.abs(fa[t]! - fb[t]!) / (scale * 2 ** -23 || 1);
      worst = Math.max(worst, err);
      assert.ok(err <= 4, `seam head ${t}: ${fa[t]} vs ${fb[t]}, ${err.toFixed(2)} ulp of its terms ${scale}`);
    }
    const d = ulpCompare(a, b);
    console.log(`# seam head: ${d.differ} of ${N} differ, worst ${worst.toFixed(3)} ulp of the terms' scale`);
  });
});

test("pressure rows: the native row's wall, ghost, link, seam and first residual (pNative), the lift and the active set", { skip: !parityGPU }, async () => {
  const N = 8192, W = 12, rnd = prng(37), rows: number[][] = [];
  for (let t = 0; t < N; t++) {
    const pick = <T,>(xs: readonly T[]) => xs[Math.floor(rnd() * xs.length)]!;
    rows.push([f32(pick([1, 0.5, 1e-7, rnd()])), f32(pick([4, 2, 9])), f32((rnd() - 0.5) * 2e4), f32(pick([0, -3.4e38, -1])),
      f32(pick([0, -5, 5, (rnd() - 0.5) * 2e3])), f32(pick([1, 0.25, 1000, 0.001])), f32((rnd() - 0.5) * 2e3), f32(rnd() * 3),
      f32((rnd() - 0.5) * 2e3), f32((rnd() - 0.5) * 1e5), rnd() < 0.5 ? 1 << 21 : 0, f32((rnd() - 0.5) * 1e3)]);
  }
  // Inputs: fraction/volume, h, slot rhs b, minimum, start, theta, pn, ch, base, rhs, closed m, head.
  const decode = /* wgsl */`T=${W}u*t;let fraction=f(T);let volume=f(T);let h4=4.0*f(T+1u);let start=f(T+4u);let m=inp[T+10u];
 var kind=N_NONE;var c=0.0;var rhs=f(T+9u);var diagonal=0.0;var walls=0.0;var lifted=0.0;var link=0u;var level2=0.0;var bulkDelta=0.0;
 rootArena[0]=f(T+2u);rootArena[1]=f(T+3u);params.hDt=vec4f(f(T+1u),0.5,2.0,1.0/60.0);params.policy=vec4f(f(T+5u),0.0,0.0,1.0);`;
  const outs = /* wgsl */`let o=12u*t;outp[o]=kind;put(o+1u,c);put(o+2u,rhs);put(o+3u,diagonal);put(o+4u,walls);put(o+5u,lifted);outp[o+6u]=link;put(o+7u,level2);put(o+8u,bulkDelta);`;
  const tail = (r: string, lift: string, free: string) => `put(12u*t+9u,${r});put(12u*t+10u,${lift});outp[12u*t+11u]=select(0u,1u,${free});`;
  const old = {
    prelude: OLD_CONSTS + /* wgsl */`
var<private> rootArena:array<f32,2>;const R_RHS:u32=0u;const R_MINIMUM:u32=1u;
fn nTheta(k:u32,n:u32,axis:u32,side:u32)->f32{return bitcast<f32>(inp[T+5u]);}
fn nBulkLink(k:u32,n:u32,f:u32,member:bool)->vec2f{return vec2f(64.0,0.0);}
fn nBulkStart(n:u32)->f32{return bitcast<f32>(inp[T+6u]);}
fn pF(base:u32,at:u32)->f32{return bitcast<f32>(inp[T+base]);}
const P_C:u32=7u;const PLANE:u32=0u;const P_BASE:u32=8u;
fn bSeamHead(at:u32,f:u32)->f32{return bitcast<f32>(inp[T+11u]);}
fn pForced(at:u32,f:u32)->f32{return bitcast<f32>(inp[T+8u]);}
${oldFunction("bSign")}
${oldFunction("bResidualMagnitude")}
${oldFunction("pLift")}
${oldFunction("pCorrectionFree")}`,
    body: decode + /* wgsl */`let s=0u;let k=0u;let n=1u;let axis=0u;let side=1u;let member=false;let f=1u;let g=0u;let at=0u;
 var w0=0.0;
 if(inp[T+7u]<${f32(1)}u){
  ${oldExcerpt("pNative", /* wgsl */`if(fraction>1e-6){
      let w=fraction/(h4*h4);let lift=rootArena[R_RHS+s]/w;c=64.0*w;
      if(rootArena[R_MINIMUM+s]<-1e38){kind=N_HELD;rhs+=c*lift;}
      else{kind=N_WALL;diagonal+=c;walls+=c;lifted+=c*max(lift,-start);link=bitcast<u32>(lift);}
     }`)}
 }else if(inp[T+7u]<${f32(2)}u){
  ${oldExcerpt("pNative", /* wgsl */`else{kind=N_GHOST;c=64.0*volume/(h4*h4*nTheta(k,n,axis,side));diagonal+=c;rhs-=c*start;}`).slice(5, -1)}
 }else{
  if(inp[T+10u]!=0u){${oldExcerpt("pNative", "kind=N_NATIVE;level2=64.0*volume/(h4*h4);let b=nBulkLink(k,n,f,member);c=b.x*volume/(h4*h4);bulkDelta+=level2-c;diagonal+=c;rhs+=c*(nBulkStart(n)-select(start,nBulkStart(k),member)+b.y);")}}
  else{${oldExcerpt("pNative", "let ch=pF(P_C+g*PLANE,at);c+=ch;rhs+=ch*(pF(P_BASE,at)-bSeamHead(at,g)-start);")}}
 }
 ${outs}
 ${oldExcerpt("pNative", "var r=rhs+lifted;if((m&N_CLOSED)!=0u&&start<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}")}
 ${tail("r", "pLift(0u,3u)", "pCorrectionFree((m&N_CLOSED)!=0u,start+fw(T+6u),fw(T+9u))")}`,
  };
  const now = {
    prelude: OLD_CONSTS + PRESSURE_ROWS_WGSL + "var<private> rootArena:array<f32,2>;\n",
    body: decode + /* wgsl */`
 if(inp[T+7u]<${f32(1)}u){
  if(fraction>1e-6){let wall=npNativeWall(fraction,h4,rootArena[0],rootArena[1]);c=wall.c;kind=wall.kind;
   if(kind==NP_N_HELD){rhs+=c*wall.lift;}else{diagonal+=c;walls+=c;lifted+=c*max(wall.lift,-start);link=bitcast<u32>(wall.lift);}}
 }else if(inp[T+7u]<${f32(2)}u){
  kind=NP_N_GHOST;c=npNativeGhost(volume,h4,f(T+5u));diagonal+=c;rhs-=c*start;
 }else{
  if(inp[T+10u]!=0u){kind=NP_N_NATIVE;level2=npNativeLink(volume,h4);c=npNativeLink(volume,h4);bulkDelta+=level2-c;diagonal+=c;rhs+=c*(f(T+6u)-start);}
  else{let ch=f(T+7u);c+=ch;rhs+=npSeamRhs(ch,f(T+8u),f(T+11u),start);}
 }
 ${outs}
 let r=npNativeResidual0(rhs,lifted,(m&N_CLOSED)!=0u,start);
 ${tail("r", "npLift(3u,params.policy.x,f(T+8u),params.hDt[1u],params.hDt.w)", "npCorrectionFree((m&N_CLOSED)!=0u,start+f(T+6u),f(T+9u))")}`,
  };
  await withParityDevice(async device => {
    const io = { inp: words(rows), threads: N, outWords: 12 * N };
    const a = await runParity(device, old, io), b = await runParity(device, now, io);
    const kinds = new Set<number>(); let clamped = 0, frozen = 0;
    for (let t = 0; t < N; t++) { kinds.add(a[12 * t]!); if (new Float32Array(a.buffer)[12 * t + 9] === 0) clamped++; if (a[12 * t + 11] === 0) frozen++; }
    for (const k of [0, 1, 3, 4, 5]) assert.ok(kinds.has(k), `native kind ${k} not reached`);
    assert.ok(clamped > 0 && frozen > 0, `the clamp (${clamped}) and the frozen set (${frozen}) are reached`);
    same(a, b, "native row");
  });
});
