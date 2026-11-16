/** M5 regressions R6 (stamp parity) and R7 (couple parity) of the design note
 * (docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md §10), for lane P's shared body module (bodies-shapes.wgsl.ts).
 *
 * The oracle is the old uniform engine's own WGSL, generated from its sources in this tree (uniformMixedSolidWGSL,
 * uniformMixedCoupleWGSL, narrowBandRigidContactSource) and compiled whole, function by function, over stubs of the
 * world it reads: a 16^3 lattice with a voxel set, no terrain and no procedural boundary (E's statics bring those; their
 * parity is E's). Both kernels read the same body records and cut atlas, and the outputs are compared bit for bit:
 * - R6: per cell umCellBodyFraction / umCellOpenGeometry, per face umPressureFaceV / umSolidFaceVelocity, against
 *   pbCellBodyFraction / pbStampCell / pbStampFace / pbStampFaceVelocity, with every body as candidates and again with
 *   each brick's BMASK (pbBodyBoxCandidates);
 * - R6 contact: nbRigidContact (mask form) against pbRigidContact;
 * - R7: couple's per-cell, per-body reaction, torque, wet weight and near bit, and the exchange words they round to;
 *   and the hydrostatic impulse of a submerged sphere is rho g V dt within 2%.
 * GPU tests skip without WEBGPU_NODE_MODULE. */
import test from "node:test";
import assert from "node:assert/strict";
import { uniformMixedSolidWGSL } from "../uniform/uniform-mixed-solid.wgsl";
import { uniformMixedCoupleWGSL } from "../uniform/uniform-mixed-bodies";
import { narrowBandRigidContactSource } from "../uniform/uniform-particle-contact.wgsl";
import { sceneShapeWgsl, SCENE_SHAPE_TABLE } from "../../core/scene-shape";
import { fluidRigidBoundaryWGSL } from "../../core/fluid-rigid-boundary.wgsl";
import { applySolidBoxCuts, packRigidBodyCutAtlas, solidBoxCutWGSL, type SolidBoxCut } from "../../core/solid-cuts";
import { quatRotate } from "../uniform/uniform-body-tile-bound";
import { wgslFunction } from "../../../tools/nbflip-parity-extract";
import { pbBodyCoupleWGSL, pbBodyShapesWGSL, pbBodyStampWGSL, pbCheckCutAtlas, PB_MAX_CUTS, PB_SUPPORTED_SHAPE_CODES } from "./bodies-shapes.wgsl";
import { RIGID_BODY_CUT_ATLAS_WIDTH } from "../../core/solid-cuts";
import { pbCoupleBrickWGSL, pbStampBrickWGSL, PB_EXCHANGE_WORDS } from "./bodies-kernels.wgsl";
import { parityGPU, prng, runParity, ulpCompare, withParityDevice } from "./parity/harness";

const N = 16;
const CELLS = N * N * N;
const H = 0.05;
const FRAME = [0.1, -0.2, 0.3] as const;
const DT = 1 / 30;

// ---- Bodies: one of each stamped shape, an interior (inverted) box, a cut box, and overlapping pairs.
type Shape = keyof typeof SCENE_SHAPE_TABLE;
interface Body { shape: Shape; dims: [number, number, number]; at: [number, number, number]; axis: [number, number, number]; angle: number; v: [number, number, number]; w: [number, number, number]; interior?: boolean; cuts?: SolidBoxCut[] }
const BODIES: Body[] = [
  { shape: "sphere", dims: [0.13, 0, 0], at: [0.33, 0.12, 0.55], axis: [0, 1, 0], angle: 0, v: [0.3, -0.2, 0.1], w: [1, 2, -0.5] },
  { shape: "box", dims: [0.25, 0.1, 0.18], at: [0.52, 0.07, 0.62], axis: [1, 1, 0], angle: 0.5, v: [-0.1, 0.25, 0], w: [0, -1.5, 0.7] },
  { shape: "capsule", dims: [0.06, 0.2, 0], at: [0.7, 0.3, 0.5], axis: [0, 0, 1], angle: 0.9, v: [0, 0, 0.4], w: [3, 0, 0] },
  { shape: "cylinder", dims: [0.08, 0.16, 0], at: [0.3, 0.36, 0.86], axis: [1, 0, 0.3], angle: 0.4, v: [0.2, 0.1, -0.3], w: [0.2, 0.4, 0.6] },
  { shape: "cup", dims: [0.12, 0.18, 0.025], at: [0.6813, 0.1437, 0.9271], axis: [0, 1, 0], angle: 0.3, v: [0, -0.5, 0], w: [0, 1, 0] },
  { shape: "box", dims: [0.66, 0.68, 0.7], at: [0.5, 0.15, 0.7], axis: [0, 1, 0], angle: 0.2, v: [0, 0, 0], w: [0, 0, 0.1], interior: true },
  { shape: "box", dims: [0.2, 0.2, 0.2], at: [0.45, 0.4, 0.45], axis: [1, 2, 3], angle: 0.7, v: [0.1, 0, 0.1], w: [-1, 0, 1],
    cuts: [{ center_m: { x: 0.05, y: 0.06, z: 0 }, halfExtents_m: { x: 0.07, y: 0.05, z: 0.2 }, orientation: { w: Math.cos(0.3), x: 0, y: Math.sin(0.3), z: 0 } }] },
];
function quat(axis: readonly number[], angle: number): [number, number, number, number] {
  const l = Math.hypot(...axis), s = Math.sin(angle / 2);
  return [Math.cos(angle / 2), axis[0]! / l * s, axis[1]! / l * s, axis[2]! / l * s];
}
/** The 32-float rigid state records (webgpu-rigid-body.ts rigidStateRecord's layout), 24 slots. */
function records(bodies: readonly Body[]): Float32Array {
  const out = new Float32Array(24 * 32);
  bodies.forEach((b, i) => {
    const s = SCENE_SHAPE_TABLE[b.shape], d = { x: b.dims[0], y: b.dims[1], z: b.dims[2] }, q = quat(b.axis, b.angle);
    out.set([...b.at, b.interior ? -s.code - 1 : s.code, ...b.dims, s.boundingRadius_m(d), ...q, ...b.v, 0, ...b.w, 1000], 32 * i);
  });
  return out;
}
/** The voxel set: a floor slab one cell thick, a pillar through the sphere and the box, and scattered cells. */
function voxels(): Uint32Array {
  const v = new Uint32Array(CELLS), rnd = prng(17);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const pillar = x === 8 && z >= 5 && z <= 7 && y <= 7, scattered = rnd() < 0.03;
    if (y === 0 || pillar || scattered) v[x + N * (y + N * z)] = 1;
  }
  return v;
}

// ---- The old engine's WGSL, from its generators, and the stubs of the world its functions read.
const OLD_SOLID = uniformMixedSolidWGSL(2, undefined, true, false, { plates: true }, true);
const oldFn = (src: string, name: string) => wgslFunction(src, name) ?? assert.fail(`old ${name} not found`);
/** The old cut atlas is a texture; the parity kernel reads the same texels from a storage array. */
function cutsFromStorage(text: string): string {
  let n = 0;
  const out = text.replace(/textureLoad\(umSolidBodyCuts,vec2i\(([^;]*?),0\),0\)/g, (_m, at: string) => { n++; return `oldCuts[u32(${at})]`; });
  assert.ok(n > 0, "no cut atlas read rewritten");
  return out;
}
const OLD_NAMES = ["umQuatRotate", "umBodyLocal", "umBodyNear", "umInsideBody", "umBodyDistance", "umApplyBodyCuts", "umBodyVelocity",
  "umBodyAtCandidates", "umBodyCandidates", "umBodyFraction", "umCellBodyFraction", "umCellOpenGeometry", "umFaceOpen", "umPressureFaceV",
  "umSolidFaceVelocity", "umSolidFaceWorld", "umSolidWorldCell", "umSolidTraceWorld", "umSolidFaceClear", "umSolidVoxelAtWorld",
  "umSolidTerrainAtWorld", "umUnitBoxOpen", "umBodyCount", "umBodyCellMask", "umSolidStaticAt"];
const WORLD = /* wgsl */`
@group(0) @binding(3) var<storage,read> BODIES_NAME:array<UMRigidBody,24>;
@group(0) @binding(4) var<storage,read> CUTS_NAME:array<vec4f>;
fn tVoxel(p:vec3i)->bool{if(any(p<vec3i(0))||any(p>=vec3i(${N}))){return false;}return aux[u32(p.x+${N}*(p.y+${N}*p.z))]!=0u;}
fn tCell(t:u32)->vec3i{return vec3i(i32(t%${N}u),i32((t/${N}u)%${N}u),i32(t/${N * N}u));}
`;
const OLD_STUBS = /* wgsl */`
struct UMSolidParams{cellGravity:vec4f,frame:vec4f,dimsDt:vec4f,boundary:vec4f}
var<private> umSolidParams:UMSolidParams;
struct UMRigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
${WORLD.replace("BODIES_NAME", "umSolidBodies").replace("CUTS_NAME", "oldCuts")}
const UM_STATIC_NO_VOXEL:u32=1u;const UM_STATIC_NO_TERRAIN:u32=2u;
var<private> umRigidCoverageFailure:vec2u;
fn umSolidEnabled()->bool{return true;}
fn umSolidValid(p:vec3i)->bool{return true;}
fn umSolidHasTerrain()->bool{return false;}
fn umSolidDimensions()->vec3u{return vec3u(${N}u);}
fn umSolidTileClear(p:vec3i)->bool{return false;}
fn umSolidVoxel(p:vec3i)->bool{return tVoxel(p);}
fn umBoundaryRecord(p:vec3i)->u32{return 0u;}
fn umBoundarySample(q:vec3f)->vec4f{return vec4f(0,0,0,1e6);}
fn umBoundaryTouchesBox(q:vec3f,w:vec3f)->bool{return false;}
fn umTerrainBoundary(q:vec3f)->vec4f{return vec4f(0,1,0,1e6);}
fn umTerrainPlaneBoxFraction(n:vec3f,o:f32,part:vec3f)->f32{return 1.0;}
fn umCellOpen(p:vec3i)->f32{return umCellOpenGeometry(p);}
fn umRigidHasPlate(m:u32)->bool{return false;}
fn umRigidIntegrationAxis(m:u32,a:u32)->u32{return 0u;}
fn umRigidIntegratedFraction(m:u32,c:vec3f,h:vec3f,a:u32,t:vec2u,k:u32)->f32{return 0.0;}
${sceneShapeWgsl()}
${fluidRigidBoundaryWGSL}
${solidBoxCutWGSL}
${OLD_NAMES.map(n => cutsFromStorage2(oldFn(OLD_SOLID, n))).join("\n")}
fn tInit(){umSolidParams=UMSolidParams(vec4f(${H},${H},${H},-9.81),vec4f(${FRAME.join(",")},0),vec4f(${N},${N},${N},${DT}),vec4f(0,0,f32(${BODIES.length}),0));}
`;
function cutsFromStorage2(text: string): string { return text.includes("umSolidBodyCuts") ? cutsFromStorage(text) : text; }

const NEW_STUBS = /* wgsl */`
${WORLD.replace("BODIES_NAME", "pbBodies").replace("CUTS_NAME", "pbCuts").replace("array<UMRigidBody,24>", "array<PBRigidBody,24>")}
${pbBodyShapesWGSL({ bodies: "pbBodies", count: `${BODIES.length}u`, cuts: "pbCuts", frame: `vec3f(${FRAME.join(",")})`, h: `vec3f(${H})`, cutFail: "tCutFail" })}
// The fixture's cut headers are valid (pbCheckCutAtlas in io()); the cut-header test below checks the latch.
fn tCutFail(i:u32){}
${pbBodyStampWGSL({ solidAt: "tStaticSolidAt", cellOpen: "tStaticCellOpen", cellKind: "tStaticCellKind", faceOpen: "tStaticFaceOpen", faceBoxed: "tStaticFaceBoxed" })}
${pbBodyCoupleWGSL}
// The static world as E's hooks give it: voxels only, no terrain, no boundary record.
fn tStaticSolidAt(world:vec3f)->bool{let h=pbH();let f=vec3f(${FRAME.join(",")});return tVoxel(vec3i(floor(vec3f((world.x-f.x)/h.x,(world.y-f.y)/h.y,(world.z-f.z)/h.z))));}
fn tStaticCellOpen(c:vec3i)->f32{return 1.0;}
fn tStaticCellKind(c:vec3i)->u32{return select(PB_STATIC_PLAIN,PB_STATIC_VOXEL,tVoxel(c));}
fn tStaticFaceOpen(c:vec3i,axis:u32)->f32{return 1.0;}
fn tStaticFaceBoxed(c:vec3i,axis:u32)->bool{return false;}
// A brick's BMASK: the 8^3 brick holding c, dilated by length(2h).
fn tBrickMask(c:vec3i)->u32{let o=8*(c/8);let f=vec3f(${FRAME.join(",")});let h=pbH();
 return pbBodyBoxCandidates(f+vec3f(o)*h,f+vec3f(o+8)*h,length(2.0*h));}
`;

const STAMP_WORDS = 8;
const OLD_STAMP = { prelude: OLD_STUBS, body: /* wgsl */`tInit();_=oldCuts[0];_=umSolidBodies[0];let c=tCell(t);let o=${STAMP_WORDS}u*t;
 put(o,umCellBodyFraction(c));put(o+1u,umCellOpenGeometry(c));
 for(var a=0u;a<3u;a++){put(o+2u+a,umPressureFaceV(c,a));put(o+5u+a,umSolidFaceVelocity(c,a));}` };
const newStamp = (mask: string) => ({ prelude: NEW_STUBS, body: /* wgsl */`_=pbCuts[0];_=pbBodies[0];let c=tCell(t);let o=${STAMP_WORDS}u*t;let m=${mask};
 let frac=pbStampCell(c,m);put(o,pbCellBodyFraction(c,m));put(o+1u,frac);
 for(var a=0u;a<3u;a++){let n=c+pbEi(a);let mn=${mask.replace("(c)", "(n)")};
  // The face's two cells' FRAC, each from its own brick's BMASK (bodyStamp reads the neighbour's stamped FRAC).
  put(o+2u+a,pbStampFace(c,a,m,frac,pbStampCell(n,mn)));put(o+5u+a,pbStampFaceVelocity(c,a,m));}` });

function io() {
  const atlas = packRigidBodyCutAtlas(BODIES.map(b => ({ cuts: b.cuts })));
  pbCheckCutAtlas(atlas, BODIES.length);
  return { inp: new Uint32Array(4), aux: voxels(), extra: { 3: records(BODIES), 4: atlas } };
}

/** Four cut headers: valid (one cut), none, a count over PB_MAX_CUTS, and cuts running off the atlas end. */
function badAtlas(): Float32Array {
  const a = packRigidBodyCutAtlas([{ cuts: BODIES.find(b => b.cuts)!.cuts! }, {}, {}, {}]);
  a.set([24, 1e6], 8); a.set([RIGID_BODY_CUT_ATLAS_WIDTH - 1, 3], 12);
  return a;
}

test("bodies-shapes: the host refuses a cut header the GPU loop could not bound", () => {
  assert.equal(PB_MAX_CUTS, Math.floor((RIGID_BODY_CUT_ATLAS_WIDTH - 24) / 3));
  const good = packRigidBodyCutAtlas([{ cuts: BODIES.find(b => b.cuts)!.cuts! }, {}]);
  pbCheckCutAtlas(good, 2);
  const bad = badAtlas();
  pbCheckCutAtlas(bad, 2);
  assert.throws(() => pbCheckCutAtlas(bad, 3), /body 2 has 1000000 cuts/);
  const off = Float32Array.from(bad); off.set([0, 0], 8);
  assert.throws(() => pbCheckCutAtlas(off, 4), /body 3's 3 cuts at texel 2047 leave the atlas/);
  assert.throws(() => pbCheckCutAtlas(good, 25), /exceed the rigid capacity/);
});

test("bodies-shapes: an out-of-range cut header latches the kernel's FAIL and applies no cut", { skip: !parityGPU }, async () => {
  const prelude = /* wgsl */`
${WORLD.replace("BODIES_NAME", "pbBodies").replace("CUTS_NAME", "pbCuts").replace("array<UMRigidBody,24>", "array<PBRigidBody,24>")}
${pbBodyShapesWGSL({ bodies: "pbBodies", count: "4u", cuts: "pbCuts", frame: "vec3f(0)", h: "vec3f(1)", cutFail: "tFail" })}
fn tFail(i:u32){outp[4u+i]=1u;}`;
  await withParityDevice(async device => {
    const out = await runParity(device, { prelude, body: "_=pbBodies[0];put(t,pbApplyBodyCuts(t,vec3f(0.05,0.06,0.0),-1.0));" },
      { inp: new Uint32Array(4), aux: new Uint32Array(4), extra: { 3: records(BODIES), 4: badAtlas() }, threads: 4, outWords: 8 });
    const d = new Float32Array(out.buffer);
    assert.deepEqual([...out.subarray(4)], [0, 0, 1, 1], "FAIL latched for the over-count and off-atlas headers only");
    // The point is the cut box's centre, inside the body (distance -1): the valid cut carves it out to +0.05.
    assert.ok(Math.abs(d[0]! - 0.05) < 1e-6, `the valid header's cut applies (${d[0]})`);
    assert.deepEqual([d[1], d[2], d[3]], [-1, -1, -1], "no cut for an empty or refused header");
  });
});

test("bodies-shapes: the stamped shape codes are the table's but the rounded plate", () => {
  assert.deepEqual([...PB_SUPPORTED_SHAPE_CODES].sort(), [0, 1, 2, 3, 4]);
});

test("bodies-shapes R6: FRAC, AP and W match the old umCellOpenGeometry / umPressureFaceV / umSolidFaceVelocity bitwise", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const base = { ...io(), threads: CELLS, outWords: STAMP_WORDS * CELLS };
    const old = await runParity(device, OLD_STAMP, base);
    const all = await runParity(device, newStamp("pbBodyMaskAll()"), base);
    const brick = await runParity(device, newStamp("tBrickMask(c)"), base);
    // The world is not trivial: every form of every word occurs.
    const f = new Float32Array(old.buffer);
    const count = (k: number, p: (x: number) => boolean) => { let n = 0; for (let t = 0; t < CELLS; t++) if (p(f[STAMP_WORDS * t + k]!)) n++; return n; };
    const partial = (x: number) => x > 0 && x < 1;
    assert.ok(count(0, partial) > 300, `partial body cells ${count(0, partial)}`);
    assert.ok(count(1, x => x === 0) > 300, "closed cells");
    assert.ok(count(2, partial) + count(3, partial) + count(4, partial) > 800, "partial apertures");
    assert.ok(count(5, x => x !== 0) + count(6, x => x !== 0) + count(7, x => x !== 0) > 800, "moving-wall faces");
    for (const [name, got] of [["all bodies", all], ["brick BMASK", brick]] as const) {
      const r = ulpCompare(old, got);
      assert.equal(r.differ, 0, `${name}: ${r.differ} words differ (first word ${r.first}, cell ${Math.floor((r.first ?? 0) / STAMP_WORDS)}, word ${(r.first ?? 0) % STAMP_WORDS}; max ${r.maxUlp} ulp)`);
    }
  });
});

test("bodies-shapes R6: pbRigidContact matches the old nbRigidContact (mask form) bitwise", { skip: !parityGPU }, async () => {
  const samples = 4096, rnd = prng(5), inp = new Float32Array(8 * samples);
  for (let i = 0; i < samples; i++) inp.set([1 + 14 * rnd(), 1 + 14 * rnd(), 1 + 14 * rnd(), 0.3 + 0.4 * rnd(), (rnd() - 0.5) * 3, (rnd() - 0.5) * 3, (rnd() - 0.5) * 3, 0], 8 * i);
  const oldContact = narrowBandRigidContactSource("((1u<<umBodyCount())-1u)");
  assert.ok(oldContact.includes("candidates=((1u<<umBodyCount())-1u)&~((1u<<(i+1u))-1u)"), "old mask form");
  await withParityDevice(async device => {
    const base = { inp, aux: voxels(), extra: io().extra, threads: samples, outWords: 6 * samples };
    const read = "let p=v3(8u*t);let inset=f(8u*t+3u);let v=v3(8u*t+4u);";
    const old = await runParity(device, { prelude: `${OLD_STUBS}\nstruct NBParams{hDt:vec4f}\nvar<private> params:NBParams;\n${oldContact}`,
      body: `tInit();params=NBParams(vec4f(${H},${H},${H},${DT}));_=oldCuts[0];_=umSolidBodies[0];${read}let r=nbRigidContact(p,v,inset);put3(6u*t,r.position);put3(6u*t+3u,r.velocity);` }, base);
    const got = await runParity(device, { prelude: NEW_STUBS,
      body: `_=pbCuts[0];_=pbBodies[0];${read}let r=pbRigidContact(p,v,inset,pbBodyMaskAll());put3(6u*t,r.position);put3(6u*t+3u,r.velocity);` }, base);
    const f = new Float32Array(old.buffer); let moved = 0;
    for (let i = 0; i < samples; i++) if (f[6 * i] !== inp[8 * i] || f[6 * i + 1] !== inp[8 * i + 1] || f[6 * i + 2] !== inp[8 * i + 2]) moved++;
    assert.ok(moved > 200, `contacts ${moved}`);
    const r = ulpCompare(old, got);
    assert.equal(r.differ, 0, `${r.differ} words differ (first ${r.first}, max ${r.maxUlp} ulp)`);
  });
});

// ---- R7: couple. The old per-owner body of couple's loop, each line checked verbatim in the old kernel.
const COUPLE_LINES = [
  "if(!umBodyNear(owner,world,length(h))){continue;}",
  "let distance=umBodyDistance(owner,world);let clear=abs(distance)>0.505*length(h);",
  "var area=vec3f(0.0);",
  "if(!clear){ for(var axis=0u;axis<3u;axis++){ var low=id;low[axis]-=1; area[axis]=(umBodyFaceFraction(owner,id,axis)-umBodyFaceFraction(owner,low,axis))*h[(axis+1u)%3u]*h[(axis+2u)%3u]; } }",
  "if(any(area!=vec3f(0.0))){ let reaction=umBodyPressure(id)*area*umSolidParams.dimsDt.w; let torque=cross(world-umSolidBodies[owner].positionShape.xyz,reaction);",
  "var fraction=select(0.0,1.0,distance<0.0);if(!clear){fraction=umBodyFraction(owner,id);}",
  "coupleAdd(base+6u,i32(round(umBodyOwnerWet(o)*fraction*65536.0)));",
];
const squash = (s: string) => s.replace(/\s+/g, "");
for (const line of COUPLE_LINES) assert.ok(squash(uniformMixedCoupleWGSL).includes(squash(line)), `not verbatim in the old couple: ${line}`);
const COUPLE_WORDS = 9;
/** Hydrostatic pressure (Pa) under a free surface at world y = SURFACE, and the wet fraction stub: aux2 is the
 * pressure per cell, the owner wet a fixed function of the cell (both kernels read the same words). */
const SURFACE = 0.5;
const RHO = 1000, G = 9.81;
function pressures(): Float32Array {
  const p = new Float32Array(CELLS);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) p[x + N * (y + N * z)] = Math.max(0, RHO * G * (SURFACE - (FRAME[1] + (y + 0.5) * H)));
  return p;
}
const COUPLE_WORLD = /* wgsl */`
@group(0) @binding(5) var<storage,read> tPressure:array<f32>;
fn tCellPressure(p:vec3i)->f32{if(any(p<vec3i(0))||any(p>=vec3i(${N}))){return 0.0;}return tPressure[u32(p.x+${N}*(p.y+${N}*p.z))];}
fn tWet(p:vec3i)->f32{return fract(0.37*f32(p.x)+0.61*f32(p.y)+0.23*f32(p.z));}
`;
function coupleKernels(bodies: number) {
  const oldPrelude = `${OLD_STUBS}${COUPLE_WORLD}
${oldFn(uniformMixedCoupleWGSL, "umBodyFaceFraction")}
fn umBodyPressure(p:vec3i)->f32{if(umCellOpen(p)<=1e-5){return 0.0;}return tCellPressure(p);}
struct TCouple{reaction:vec3f,torque:vec3f,wet:i32,near:bool}
fn oldCouple(owner:u32,id:vec3i)->TCouple{
 let h=umSolidParams.cellGravity.xyz;let world=umSolidWorldCell(id);var out=TCouple(vec3f(0),vec3f(0),0,false);
 loop{
 ${COUPLE_LINES[0]} out.near=true;
 ${COUPLE_LINES[1]}
 ${COUPLE_LINES[2]}
 ${COUPLE_LINES[3]}
 ${COUPLE_LINES[4]} out.reaction=reaction;out.torque=torque;}
 ${COUPLE_LINES[5]}
 out.wet=i32(round(tWet(id)*fraction*65536.0));break;
 }
 return out;
}`;
  const write = "put3(o,r.reaction);put3(o+3u,r.torque);outp[o+6u]=bitcast<u32>(r.wet);outp[o+7u]=select(0u,1u,r.near);outp[o+8u]=bitcast<u32>(pbFixedOr(r.reaction.y));";
  const head = `let c=tCell(t/${bodies}u);let owner=t%${bodies}u;let o=${COUPLE_WORDS}u*t;`;
  return {
    old: { prelude: oldPrelude.replace("fn tInit()", "fn pbFixedOr(x:f32)->i32{return i32(round(x*1000000.0));}\nfn tInit()"),
      body: `tInit();_=oldCuts[0];_=umSolidBodies[0];_=tPressure[0];${head}let r=oldCouple(owner,c);${write}` },
    new: { prelude: `${NEW_STUBS}${COUPLE_WORLD}\nfn pbFixedOr(x:f32)->i32{return pbFixed(x);}`,
      body: `_=pbCuts[0];_=pbBodies[0];_=tPressure[0];${head}let m=tBrickMask(c);
 if(((m>>owner)&1u)==0u){put3(o,vec3f(0));put3(o+3u,vec3f(0));outp[o+6u]=0u;outp[o+7u]=0u;outp[o+8u]=0u;return;}
 // couple's p: X rho/dt in the engine, 0 where FRAC <= 1e-5 (the stamped channel).
 let p=select(tCellPressure(c),0.0,pbStampCell(c,m)<=1e-5);
 let q=pbCoupleBody(owner,c,p,tWet(c),${DT});let r=TCoupleOut(q.reaction,q.torque,pbFixedWet(q.wet),q.near);${write}` },
  };
}

test("bodies-shapes R7: couple's per-cell reaction, torque, wet weight and near bit match the old couple bitwise", { skip: !parityGPU }, async () => {
  const n = BODIES.length, k = coupleKernels(n);
  k.new.prelude += "\nstruct TCoupleOut{reaction:vec3f,torque:vec3f,wet:i32,near:bool}";
  await withParityDevice(async device => {
    const base = { ...io(), extra: { ...io().extra, 5: pressures() }, threads: CELLS * n, outWords: COUPLE_WORDS * CELLS * n };
    const old = await runParity(device, k.old, base);
    const got = await runParity(device, k.new, base);
    // The interior box's near bit is set everywhere (an inverted body is near every point); the brick mask holds it.
    let reacting = 0; const f = new Float32Array(old.buffer);
    for (let i = 0; i < CELLS * n; i++) if (f[COUPLE_WORDS * i + 1] !== 0) reacting++;
    assert.ok(reacting > 500, `reacting pairs ${reacting}`);
    const r = ulpCompare(old, got);
    assert.equal(r.differ, 0, `${r.differ} words differ (first ${r.first}: pair ${Math.floor((r.first ?? 0) / COUPLE_WORDS)}, word ${(r.first ?? 0) % COUPLE_WORDS}; max ${r.maxUlp} ulp)`);
  });
});

test("bodies-shapes R7: a submerged sphere's pressure impulse is rho g V dt within 2%", { skip: !parityGPU }, async () => {
  // One sphere, r = 4h, well below the surface; no voxels. The discrete sum is -1.06% (Float64, the same samples).
  const r = 0.2, sphere: Body = { shape: "sphere", dims: [r, 0, 0], at: [0.51, 0.13, 0.69], axis: [0, 1, 0], angle: 0, v: [0, 0, 0], w: [0, 0, 0] };
  const k = coupleKernels(1);
  const prelude = `${k.new.prelude}\nstruct TCoupleOut{reaction:vec3f,torque:vec3f,wet:i32,near:bool}`
    .replace(`fn pbBodyCount()->u32{return min(${BODIES.length}u,PB_BODY_CAPACITY);}`, "fn pbBodyCount()->u32{return min(1u,PB_BODY_CAPACITY);}");
  assert.ok(prelude.includes("return min(1u,PB_BODY_CAPACITY)"), "body count rewritten");
  await withParityDevice(async device => {
    const out = await runParity(device, { prelude, body: k.new.body }, {
      inp: new Uint32Array(4), aux: new Uint32Array(CELLS), threads: CELLS, outWords: COUPLE_WORDS * CELLS,
      extra: { 3: records([sphere]), 4: packRigidBodyCutAtlas([{}]), 5: pressures() },
    });
    const words = new Int32Array(out.buffer);
    let fixed = 0; for (let i = 0; i < CELLS; i++) fixed += words[COUPLE_WORDS * i + 8]!;
    const impulse = fixed / 1e6, expected = RHO * G * (4 / 3) * Math.PI * r ** 3 * DT;
    assert.ok(Math.abs(impulse / expected - 1) < 0.02, `impulse ${impulse} N s, rho g V dt ${expected} (${((impulse / expected - 1) * 100).toFixed(2)}%)`);
  });
});

// ---- A Float64 mirror of the old formulas over the shape table's host maths: an oracle no WGSL compiler shares. The
// old-WGSL parity above cannot see a miscompile both texts hit; this one can (the verbatim couple loop, a dynamic vector
// write per axis, miscompiled on Metal in one kernel: bodies-shapes.wgsl.ts, pbE).
type V3 = [number, number, number];
const conj = (q: readonly number[]) => [q[0]!, -q[1]!, -q[2]!, -q[3]!] as const;
function mirror(bodies: readonly Body[], vox: Uint32Array) {
  const R = bodies.map(b => { const s = SCENE_SHAPE_TABLE[b.shape]; return { b, s, q: quat(b.axis, b.angle), R: s.boundingRadius_m({ x: b.dims[0], y: b.dims[1], z: b.dims[2] }) }; });
  const all = (1 << bodies.length) - 1, h = H, len = (x: number) => Math.hypot(x, x, x);
  const near = (i: number, w: V3, m: number) => !!R[i]!.b.interior || Math.hypot(w[0] - R[i]!.b.at[0], w[1] - R[i]!.b.at[1], w[2] - R[i]!.b.at[2]) <= R[i]!.R + m;
  const local = (i: number, w: V3) => { const l = quatRotate(conj(R[i]!.q), [w[0] - R[i]!.b.at[0], w[1] - R[i]!.b.at[1], w[2] - R[i]!.b.at[2]]); return { x: l[0], y: l[1], z: l[2] }; };
  const dims = (i: number) => ({ x: R[i]!.b.dims[0], y: R[i]!.b.dims[1], z: R[i]!.b.dims[2] });
  const distance = (i: number, w: V3) => { const l = local(i, w); return applySolidBoxCuts((R[i]!.b.interior ? -1 : 1) * R[i]!.s.distance_m(dims(i), l), R[i]!.b.cuts, l); };
  const inside = (i: number, w: V3) => { if (!near(i, w, 0)) return false; if (R[i]!.b.cuts?.length) return distance(i, w) <= 0; const v = R[i]!.s.inside(dims(i), local(i, w)); return R[i]!.b.interior ? !v : v; };
  const at = (w: V3, m: number) => { for (let i = 0; i < bodies.length; i++) if (m >> i & 1 && inside(i, w)) return i; return -1; };
  const velocity = (i: number, w: V3): V3 => { const b = R[i]!.b, r = [w[0] - b.at[0], w[1] - b.at[1], w[2] - b.at[2]]; return [b.v[0] + b.w[1] * r[2]! - b.w[2] * r[1]!, b.v[1] + b.w[2] * r[0]! - b.w[0] * r[2]!, b.v[2] + b.w[0] * r[1]! - b.w[1] * r[0]!]; };
  const voxel = (c: V3) => c.every(v => v >= 0 && v < N) && vox[c[0] + N * (c[1] + N * c[2])] !== 0;
  const staticAt = (w: V3) => voxel([0, 1, 2].map(a => Math.floor((w[a]! - FRAME[a]!) / h)) as V3);
  const cell = (c: V3): V3 => [0, 1, 2].map(a => FRAME[a]! + (c[a]! + 0.5) * h) as V3;
  const face = (c: V3, a: number): V3 => { const w = cell(c); w[a] += 0.5 * h; return w; };
  const step = (c: V3, a: number, d: number): V3 => { const n = [...c] as V3; n[a] += d; return n; };
  const corners = (w: V3, k: number): V3 => [0, 1, 2].map(a => w[a]! + ((k >> a & 1) ? 0.4 : -0.4) * h) as V3;
  const transverse = (w: V3, a: number, k: number): V3 => { const s = [...w] as V3; s[(a + 1) % 3] += (k & 1 ? 0.35 : -0.35) * h; s[(a + 2) % 3] += (k & 2 ? 0.35 : -0.35) * h; return s; };
  const bodyCover = (c: V3) => { let n = 0; for (let k = 0; k < 8; k++) if (at(corners(cell(c), k), all) >= 0) n++; return n / 8; };
  const frac = (c: V3) => voxel(c) ? 0 : 1 - bodyCover(c);
  const wall = (c: V3, a: number) => voxel(c) || voxel(step(c, a, 1));
  const candidates = (w: V3, m: number) => { let r = 0; for (let i = 0; i < bodies.length; i++) if (near(i, w, m)) r |= 1 << i; return r; };
  const aperture = (c: V3, a: number) => {
    const w = face(c, a), cands = candidates(w, len(0.5 * h));
    if (wall(c, a)) return 0.5 * (frac(c) + frac(step(c, a, 1)));
    if (!cands) return 1;
    let solid = 0; for (let k = 0; k < 4; k++) { const s = transverse(w, a, k); if (staticAt(s) || at(s, cands) >= 0) solid++; }
    return 1 - 0.25 * solid;
  };
  const wallVelocity = (c: V3, a: number) => {
    const w = face(c, a), cands = candidates(w, 2 * h);
    if (!cands || wall(c, a)) return 0;
    let nearest = h, result = 0;
    for (let i = 0; i < bodies.length; i++) if (cands >> i & 1) { const d = Math.abs(distance(i, w)); if (d <= nearest) { nearest = d; result = velocity(i, w)[a]!; } }
    let solid = 0, sum = 0;
    for (let k = 0; k < 8; k++) { const s = corners(w, k); if (staticAt(s)) { solid++; continue; } const i = at(s, cands); if (i >= 0) { solid++; sum += velocity(i, w)[a]!; } }
    return solid > 0 ? sum / solid : result;
  };
  const faceFraction = (i: number, c: V3, a: number) => { let n = 0; for (let k = 0; k < 4; k++) if (inside(i, transverse(face(c, a), a, k))) n++; return n / 4; };
  const couple = (i: number, c: V3, p: number) => {
    const w = cell(c);
    if (!near(i, w, len(h))) return undefined;
    const area = [0, 1, 2].map(a => (faceFraction(i, c, a) - faceFraction(i, step(c, a, -1), a)) * h * h);
    const reaction = area.map(x => p * x * DT) as V3, b = R[i]!.b.at, r = [w[0] - b[0], w[1] - b[1], w[2] - b[2]];
    return { reaction, torque: [r[1]! * reaction[2] - r[2]! * reaction[1], r[2]! * reaction[0] - r[0]! * reaction[2], r[0]! * reaction[1] - r[1]! * reaction[0]] as V3 };
  };
  return { bodyCover, frac, aperture, wallVelocity, couple };
}
/** Words of a against the mirror's values: those that differ by more than tol (|a - b| > tol (1 + |b|)). */
function mismatches(got: Float32Array, want: Float64Array, tol: number) {
  let n = 0, first = -1;
  for (let i = 0; i < want.length; i++) if (Math.abs(got[i]! - want[i]!) > tol * (1 + Math.abs(want[i]!))) { n++; if (first < 0) first = i; }
  return { n, first };
}

test("bodies-shapes R6/R7: the stamp and couple agree with a Float64 mirror of the old formulas", { skip: !parityGPU }, async () => {
  const vox = voxels(), m = mirror(BODIES, vox), n = BODIES.length, k = coupleKernels(n);
  k.new.prelude += "\nstruct TCoupleOut{reaction:vec3f,torque:vec3f,wet:i32,near:bool}";
  const stampWant = new Float64Array(STAMP_WORDS * CELLS), p = pressures();
  const coupleWant = new Float64Array(6 * CELLS * n);
  for (let t = 0; t < CELLS; t++) {
    const c: V3 = [t % N, Math.floor(t / N) % N, Math.floor(t / (N * N))], o = STAMP_WORDS * t;
    stampWant[o] = m.bodyCover(c); stampWant[o + 1] = m.frac(c);
    for (let a = 0; a < 3; a++) { stampWant[o + 2 + a] = m.aperture(c, a); stampWant[o + 5 + a] = m.wallVelocity(c, a); }
    const pressure = stampWant[o + 1]! <= 1e-5 ? 0 : p[t]!;
    for (let i = 0; i < n; i++) { const r = m.couple(i, c, pressure); if (r) coupleWant.set([...r.reaction, ...r.torque], 6 * (n * t + i)); }
  }
  await withParityDevice(async device => {
    const stamp = new Float32Array((await runParity(device, newStamp("tBrickMask(c)"), { ...io(), threads: CELLS, outWords: STAMP_WORDS * CELLS })).buffer);
    const couple = new Float32Array((await runParity(device, k.new, { ...io(), extra: { ...io().extra, 5: p }, threads: CELLS * n, outWords: COUPLE_WORDS * CELLS * n })).buffer);
    const coupleGot = new Float32Array(6 * CELLS * n);
    for (let j = 0; j < CELLS * n; j++) coupleGot.set(couple.subarray(COUPLE_WORDS * j, COUPLE_WORDS * j + 6), 6 * j);
    // A sample within float rounding of a surface may fall either way; the fixture keeps the bodies off lattice planes (none today).
    const s = mismatches(stamp, stampWant, 1e-4), c = mismatches(coupleGot, coupleWant, 1e-4);
    if (process.env.PB_DEBUG) for (let i = 0; i < stampWant.length; i++) if (Math.abs(stamp[i]! - stampWant[i]!) > 1e-4 * (1 + Math.abs(stampWant[i]!))) { const t = Math.floor(i / STAMP_WORDS); console.log(`cell ${t % N},${Math.floor(t / N) % N},${Math.floor(t / (N * N))} word ${i % STAMP_WORDS} got ${stamp[i]} want ${stampWant[i]}`); }
    assert.ok(s.n <= 4, `stamp: ${s.n} of ${stampWant.length} words differ (first ${s.first}: cell ${Math.floor(s.first / STAMP_WORDS)} word ${s.first % STAMP_WORDS}, got ${stamp[s.first]}, want ${stampWant[s.first]})`);
    assert.ok(c.n <= 4, `couple: ${c.n} of ${coupleWant.length} words differ (first ${c.first}, got ${coupleGot[c.first]}, want ${coupleWant[c.first]})`);
  });
});

// ---- bodies-kernels: the per-brick passes over the fixture's 2^3 bricks.
const BRICKS = (N / 8) ** 3;
const brickOrigin = (b: string) => `8*vec3i(i32(${b}%${N / 8}u),i32((${b}/${N / 8}u)%${N / 8}u),i32(${b}/${(N / 8) ** 2}u))`;
const BRICK_STUBS = /* wgsl */`
var<private> tO:vec3i;
fn tCellIndex(c:vec3i)->u32{return u32(c.x+${N}*(c.y+${N}*c.z));}
fn tPutCut(l:u32,ch:u32,v:f32){outp[8u*tCellIndex(tO+pbBrickCell(l))+ch]=bitcast<u32>(v);}
fn tPutOpen(l:u32,open:bool){outp[8u*tCellIndex(tO+pbBrickCell(l))+7u]=select(0u,1u,open);}
${pbStampBrickWGSL({ putCut: "tPutCut", putOpen: "tPutOpen" })}`;

test("bodies-kernels: pbStampBrick's FRAC, lower-face AP and W are the old upper-face values of the cell below, bitwise", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    const old = new Float32Array((await runParity(device, OLD_STAMP, { ...io(), threads: CELLS, outWords: STAMP_WORDS * CELLS })).buffer);
    const run = async (mask: string) => new Float32Array((await runParity(device, { prelude: NEW_STUBS + BRICK_STUBS,
      body: `_=pbCuts[0];_=pbBodies[0];let b=t/128u;tO=${brickOrigin("b")};pbStampBrick(tO,${mask},t%128u,128u);` },
    { ...io(), threads: 128 * BRICKS, outWords: 8 * CELLS })).buffer);
    const got = await run("tBrickMask(tO)"), u = new Uint32Array(got.buffer);
    let differ = 0, faces = 0, first = "";
    const at = (x: number, y: number, z: number) => x + N * (y + N * z);
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const t = at(x, y, z), want: [number, number][] = [[got[8 * t]!, old[STAMP_WORDS * t + 1]!]];
      if (u[8 * t + 7] !== (old[STAMP_WORDS * t + 1]! > 1e-5 ? 1 : 0)) { differ++; first ||= `OPEN ${x},${y},${z}`; }
      for (let a = 0; a < 3; a++) {
        const q = [x, y, z]; q[a]!--;
        if (q[a]! < 0) continue;
        const tq = at(q[0]!, q[1]!, q[2]!); faces++;
        want.push([got[8 * t + 1 + a]!, old[STAMP_WORDS * tq + 2 + a]!], [got[8 * t + 4 + a]!, old[STAMP_WORDS * tq + 5 + a]!]);
      }
      for (const [g, w] of want) if (Math.fround(g) !== Math.fround(w) && !(Number.isNaN(g) && Number.isNaN(w))) { differ++; first ||= `cell ${x},${y},${z}: ${g} vs ${w}`; }
    }
    assert.ok(faces > 3 * CELLS * 0.9, `${faces} faces compared`);
    assert.equal(differ, 0, `${differ} words differ (first ${first})`);
    // The BODY_WAS clean-up pass (mask 0): the static values, as the hooks give them, and W = 0.
    const plain = await run("0u"), vox = voxels();
    let bad = 0;
    for (let t = 0; t < CELLS; t++) {
      if (plain[8 * t] !== (vox[t] ? 0 : 1)) bad++;
      for (let a = 0; a < 3; a++) if (plain[8 * t + 4 + a] !== 0) bad++;
    }
    assert.equal(bad, 0, `${bad} static FRAC / W words off the hooks' values`);
  });
});

test("bodies-kernels: pbCoupleBrick's exchange is the per-cell couple words' integer sum, bitwise", { skip: !parityGPU }, async () => {
  const n = BODIES.length, k = coupleKernels(n);
  k.new.prelude += "\nstruct TCoupleOut{reaction:vec3f,torque:vec3f,wet:i32,near:bool}";
  await withParityDevice(async device => {
    const p = pressures();
    const cells = new Int32Array((await runParity(device, k.new, { ...io(), extra: { ...io().extra, 5: p }, threads: CELLS * n, outWords: COUPLE_WORDS * CELLS * n })).buffer);
    const f = new Float32Array(cells.buffer);
    const want = new Int32Array(PB_EXCHANGE_WORDS * 24);
    const fixed = (x: number) => { const v = Math.fround(Math.fround(x) * 1e6), r = Math.round(v); return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; };
    for (let j = 0; j < CELLS * n; j++) {
      const o = COUPLE_WORDS * j, i = j % n;
      if (cells[o + 7] !== 1) continue;
      want[PB_EXCHANGE_WORDS * i + 10] = 1;
      for (let w = 0; w < 6; w++) want[PB_EXCHANGE_WORDS * i + w]! += fixed(f[o + w]!);
      want[PB_EXCHANGE_WORDS * i + 6]! += cells[o + 6]!;
    }
    const prelude = `${NEW_STUBS}${COUPLE_WORLD}
@group(0) @binding(6) var<storage,read_write> tX:array<atomic<i32>>;
var<private> tO:vec3i;
fn tFrac(l:u32)->f32{return pbStampCell(tO+pbBrickCell(l),tBrickMask(tO));}
fn tP(c:vec3i,l:u32)->f32{return tCellPressure(c);}
fn tW(c:vec3i,l:u32)->f32{return tWet(c);}
${pbStampBrickWGSL({ putCut: "tNoCut", putOpen: "tNoOpen" })}
fn tNoCut(l:u32,ch:u32,v:f32){}fn tNoOpen(l:u32,o:bool){}
${pbCoupleBrickWGSL({ frac: "tFrac", pressure: "tP", wet: "tW" }, "tX")}`;
    // One thread: every brick into the workgroup words, then one flush (the kernel's barriers are between these calls).
    const body = `_=pbCuts[0];_=pbBodies[0];_=tPressure[0];if(t!=0u){return;}
 pbCoupleClear(0u,1u);
 for(var b=0u;b<${BRICKS}u;b++){tO=${brickOrigin("b")};pbCoupleBrick(tO,tBrickMask(tO),0u,1u,${DT});}
 pbCoupleFlush(pbBodyMaskAll(),0u,1u);
 for(var w=0u;w<${PB_EXCHANGE_WORDS * 24}u;w++){outp[w]=bitcast<u32>(atomicLoad(&tX[w]));}`;
    const got = new Int32Array((await runParity(device, { prelude, body }, { ...io(), extra: { ...io().extra, 5: p, 6: new Int32Array(PB_EXCHANGE_WORDS * 24) as never }, threads: 1, outWords: PB_EXCHANGE_WORDS * 24 })).buffer);
    assert.ok(want.some((v, i) => i % PB_EXCHANGE_WORDS === 1 && v > 0), "a body takes an upward reaction");
    assert.deepEqual([...got], [...want]);
  });
});
