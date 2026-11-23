// M1 (docs/plans/gpu-shape/m1-distinct-once.md): "look up each distinct brick once, then branch-free taps" as WGSL text
// variants of the kernels' compiled code (no lib/ edits). Each variant is exact by construction: the fast path runs
// only where every tap would take lPhi0's slot-hit branch, and it reads the same word with the same arithmetic.
//
// The stencil helper: a stencil spanning cells lo..hi (hi - lo <= 7 on each axis) touches at most two bricks an axis,
// lo >> 3 and hi >> 3. lSlots(lo, hi) looks up those (up to) 8 bricks' slots up front (independent 2-link walks that
// issue together), keeps them in private scalars, and returns whether all hit and lo >= 0 (then every tap's fgSlot is
// one of them and not NB_NONE, and lPhi0 returns its FC_PHIU word). A tap then selects its slot with ALU only.
export type Variant = { kernel: string; name: string; apply: (code: string) => string };

/** The text of `fn name(...){...}` in code. */
export function fnText(code: string, name: string): string {
  const i = code.indexOf(`fn ${name}(`); if (i < 0) throw new Error(`no fn ${name}`);
  let d = 0; for (let k = code.indexOf("{", i); k < code.length; k++) { if (code[k] === "{") d++; else if (code[k] === "}" && --d === 0) return code.slice(i, k + 1); }
  throw new Error(`unterminated ${name}`);
}
export function sub(code: string, from: string, to: string, count = 1): string {
  const n = code.split(from).length - 1; if (n !== count) throw new Error(`sub "${from.slice(0, 60)}" hits ${n}, want ${count}`);
  return code.split(from).join(to);
}

const SLOTS = `
var<private> lmB0:vec3i;var<private> lmS0:u32;var<private> lmS1:u32;var<private> lmS2:u32;var<private> lmS3:u32;
var<private> lmS4:u32;var<private> lmS5:u32;var<private> lmS6:u32;var<private> lmS7:u32;
// M1: the slots of the (up to) 8 bricks a stencil over cells lo..hi touches; true when all hit and lo >= 0.
fn lmSlots(lo:vec3i,hi:vec3i)->bool{
 let b0=lo>>vec3u(3u);let b1=hi>>vec3u(3u);lmB0=b0;
 lmS0=fgBrickSlot(vec3i(b0.x,b0.y,b0.z));lmS1=fgBrickSlot(vec3i(b1.x,b0.y,b0.z));
 lmS2=fgBrickSlot(vec3i(b0.x,b1.y,b0.z));lmS3=fgBrickSlot(vec3i(b1.x,b1.y,b0.z));
 lmS4=fgBrickSlot(vec3i(b0.x,b0.y,b1.z));lmS5=fgBrickSlot(vec3i(b1.x,b0.y,b1.z));
 lmS6=fgBrickSlot(vec3i(b0.x,b1.y,b1.z));lmS7=fgBrickSlot(vec3i(b1.x,b1.y,b1.z));
 let any0=(lmS0==NB_NONE)||(lmS1==NB_NONE)||(lmS2==NB_NONE)||(lmS3==NB_NONE);
 let any1=(lmS4==NB_NONE)||(lmS5==NB_NONE)||(lmS6==NB_NONE)||(lmS7==NB_NONE);
 return !(any0||any1)&&all(lo>=vec3i(0));
}
// M1: lPhi0 at v inside the last lmSlots stencil (all hit): the FC_PHIU word, the slot chosen by ALU.
fn lmSlot(v:vec3i)->u32{
 let d=(v>>vec3u(3u))!=lmB0;
 return select(select(select(lmS0,lmS1,d.x),select(lmS2,lmS3,d.x),d.y),select(select(lmS4,lmS5,d.x),select(lmS6,lmS7,d.x),d.y),d.z);
}
fn lmTap(v:vec3i)->f32{return fcLoadF(FC_PHIU+512u*lmSlot(v)+nbLocal(v));}
`;
// The lPhi0 FC_PHIU fast word is lPhi0's tiered branch (the lAdvect call sites test FG_TIERED); check the kernel still has it.
const LPHI0 = "fn lPhi0(v:vec3i)->f32{\n if(FG_TIERED){let s=fgSlot(v);if(s!=NB_NONE&&all(v>=vec3i(0))){return fcLoadF(FC_PHIU+512u*s+nbLocal(v));}}";

/** The departure cubic (64 taps over lBase-1..lBase+2): a fast copy of nbfCubicLimited behind one lmSlots test. */
function cubic(code: string): string {
  if (!code.includes(LPHI0)) throw new Error("lPhi0 text changed");
  const f = fnText(code, "nbfCubicLimited");
  const fast = sub(sub(f, "fn nbfCubicLimited(", "fn lmCubicLimited("), "let s=nbxCubicTap(x,y,z);", "let s=lmTap(lBase+vec3i(i32(x),i32(y),i32(z))-vec3i(1));");
  code = sub(code, f, f + "\n" + fast);
  const fastCall = "if(FG_TIERED&&lmSlots(cell-vec3i(1),cell+vec3i(2))){value=lmCubicLimited(f,lowHeld,highHeld);}else{value=nbfCubicLimited(f,lowHeld,highHeld);}";
  if (code.includes("  value=nbfCubicLimited(f,lowHeld,highHeld);")) code = sub(code, "  value=nbfCubicLimited(f,lowHeld,highHeld);", "  " + fastCall);
  else code = sub(code, "{value=nbfCubicLimited(f,lowHeld,highHeld);}", "{" + fastCall + "}"); // phiSolid: behind psCubicBuried
  return code;
}
/** lCorners0 (8 corners c..c+1). */
function corners(code: string): string {
  const f = fnText(code, "lCorners0");
  const fast = `fn lCorners0(c:vec3i)->array<f32,8>{
 if(!(FG_TIERED&&lmSlots(c,c+vec3i(1)))){return lCorners0Slow(c);}
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=lmTap(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return v;
}`;
  return sub(code, f, sub(f, "fn lCorners0(", "fn lCorners0Slow(") + "\n" + fast);
}
/** The foot's 10-tap gradient stencil (p ± e, p ± 2e on x and z; inside v-2..v+2). */
function gradient(code: string): string {
  return sub(code, "  for(var k=0u;k<10u;k++){t[k]=lPhi0(vec3i(at[k]));}",
    "  if(FG_TIERED&&lmSlots(v-vec3i(2),v+vec3i(2))){for(var k=0u;k<10u;k++){t[k]=lmTap(vec3i(at[k]));}}else{for(var k=0u;k<10u;k++){t[k]=lPhi0(vec3i(at[k]));}}");
}
const withSlots = (code: string) => sub(code, "fn lPhi0(v:vec3i)->f32{", SLOTS + "\nfn lPhi0(v:vec3i)->f32{");

/** phiSolid's buried test: its 8-slot private array (stack, per the memory notes) as lmSlots' scalars. */
function buried(code: string): string {
  code = sub(code, " var slots:array<u32,8>;\n for(var k=0u;k<8u;k++){let b=b0+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));slots[k]=select(NB_NONE,fgBrickSlot(b),all(b>=vec3i(0)));}",
    " _=lmSlots(lo,lo+vec3i(4));");
  return sub(code, "let r=vec3u((c>>vec3u(3u))-b0);let s=slots[r.x+2u*r.y+4u*r.z];", "let s=lmSlot(c);");
}
const lAdv = (c: string) => gradient(corners(cubic(withSlots(c))));
// lmSlots looking up only the distinct bricks (a stencil inside one brick on an axis copies that axis's slots).
const DISTINCT = `fn lmSlots(lo:vec3i,hi:vec3i)->bool{
 let b0=lo>>vec3u(3u);let b1=hi>>vec3u(3u);lmB0=b0;let d=b1!=b0;
 lmS0=fgBrickSlot(b0);
 if(d.x){lmS1=fgBrickSlot(vec3i(b1.x,b0.y,b0.z));}else{lmS1=lmS0;}
 if(d.y){lmS2=fgBrickSlot(vec3i(b0.x,b1.y,b0.z));if(d.x){lmS3=fgBrickSlot(vec3i(b1.x,b1.y,b0.z));}else{lmS3=lmS2;}}else{lmS2=lmS0;lmS3=lmS1;}
 if(d.z){
  lmS4=fgBrickSlot(vec3i(b0.x,b0.y,b1.z));if(d.x){lmS5=fgBrickSlot(vec3i(b1.x,b0.y,b1.z));}else{lmS5=lmS4;}
  if(d.y){lmS6=fgBrickSlot(vec3i(b0.x,b1.y,b1.z));if(d.x){lmS7=fgBrickSlot(b1);}else{lmS7=lmS6;}}else{lmS6=lmS4;lmS7=lmS5;}
 }else{lmS4=lmS0;lmS5=lmS1;lmS6=lmS2;lmS7=lmS3;}`;
const distinct = (code: string) => sub(code, fnText(SLOTS, "lmSlots").split("\n let any0")[0]!, DISTINCT);
// pMove: the 8-corner depth (fgDepth, run 1-3 times a record) and the 8-corner φ (fgCornersOf of NB_PHI_F_PHI: fgPhiAt,
// fgLiquidCell's centre φ). Both read slot-hit words only where every corner hits (FG_PHI_RESOLVED, no ring).
const PM_DEPTH = "  value+=w.x*w.y*w.z*fgDepthVertex(c+vec3i(corner));";
function pmDepth(code: string): string {
  const f = fnText(code, "fgDepth");
  const loop = f.slice(f.indexOf(" for(var k=0u;k<8u;k++){"), f.indexOf(" return value;"));
  const fast = loop.replace(PM_DEPTH, "  let vv=c+vec3i(corner);value+=w.x*w.y*w.z*fcLoadF(FC_DEPTH+(512u*lmSlot(vv)+nbLocal(vv)));");
  if (fast === loop) throw new Error("fgDepth loop text changed");
  return sub(code, f, f.replace(loop, ` if(lmSlots(c,c+vec3i(1))){\n${fast} }else{\n${loop} }\n`));
}
function pmCorners(code: string): string {
  return sub(code, "fn fgCornersOf(base:u32,c:vec3i)->array<f32,8>{\n var v:array<f32,8>;\n",
    "fn fgCornersOf(base:u32,c:vec3i)->array<f32,8>{\n var v:array<f32,8>;\n if(FG_TIERED&&FG_PHI_RESOLVED&&!FG_RING_STRICT&&base==NB_PHI_F_PHI&&all(c+vec3i(1)<=NB_LATTICE)&&lmSlots(c,c+vec3i(1))){for(var k=0u;k<8u;k++){let w=c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));v[k]=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*lmSlot(w)+nbLocal(w)]);}return v;}\n");
}
const pmSlots = (code: string) => distinct(sub(code, "fn fgDepth(p:vec3f)->f32{", SLOTS.replace(/\nfn lmTap[^\n]*\n/, "\n") + "\nfn fgDepth(p:vec3f)->f32{"));
/** Diagnostic (not a candidate): the fast corners plus the slow ones; any corner word that differs poisons the corner
 *  with a NaN, so a NaN count of zero with phi differences means the corner words agree and the difference is codegen. */
function cornersVerify(code: string): string {
  code = corners(code);
  return sub(code, " for(var k=0u;k<8u;k++){v[k]=lmTap(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;",
    " for(var k=0u;k<8u;k++){v[k]=lmTap(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n let w=lCorners0Slow(c);for(var k=0u;k<8u;k++){if(bitcast<u32>(v[k])!=bitcast<u32>(w[k])){v[k]=bitcast<f32>(0x7fc00000u|u32(c.x<0));}}\n return v;");
}
/** lCubicAt's surface heat: the 8 cells v-1..v, each one fgTileAt (a slot lookup). All hit (so all in the lattice):
 *  the same tile words, max'd in the same order. */
function heat(code: string): string {
  // The slow loop skips a cell outside the box (nbInLattice or nbInBox, per layout): the fast one needs all 8 inside.
  const inBox = /fn lCubicAt[^]*?if\(!(nb\w+)\(c\)\)\{continue;\}/.exec(code)?.[1]; if (!inBox) throw new Error("lCubicAt text changed");
  return sub(code, "fn lCubicAt(v:vec3i)->bool{\n var heat=0.0;\n",
    `fn lCubicAt(v:vec3i)->bool{\n var heat=0.0;\n if(${inBox}(v-vec3i(1))&&${inBox}(v)&&lmSlots(v-vec3i(1),v)){for(var k=0u;k<8u;k++){let c=v-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));heat=max(heat,fcLoadF(fgTileWord(lmSlot(c),fgTileOf(c),FT_SURF)));}return heat<=1.5;}\n`);
}
/** The cell's trilinear (8 corners at cell..cell+1): the whole trilinear in two copies, one over the stencil's slots. */
const TRI = "var value=nbfCellTrilinear(lCorners0(cell),f);";
function tri2(code: string): string {
  return sub(code, TRI, "var value=0.0;if(FG_TIERED&&lmSlots(cell,cell+vec3i(1))){var v:array<f32,8>;for(var k=0u;k<8u;k++){v[k]=lmTap(cell+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}value=nbfCellTrilinear(v,f);}else{value=nbfCellTrilinear(lCorners0(cell),f);}");
}
/** lCorners0 with one loop: each corner from the stencil's slots or lPhi0 by a predicate known before the loop. */
function cornSel(code: string): string {
  return sub(code, " for(var k=0u;k<8u;k++){v[k]=lPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;",
    " let fast=FG_TIERED&&lmSlots(c,c+vec3i(1));\n for(var k=0u;k<8u;k++){let w=c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(fast){v[k]=lmTap(w);}else{v[k]=lPhi0(w);}}\n return v;");
}
// "cached": one copy of each stencil, lPhi0's own per-tap rule (slot hit and v >= 0: the PHIU word, else fgPhi) with
// fgSlot's directory walk replaced by the stencil's slot picked by ALU (lmSlots' walks, each distinct brick once). A
// miss still takes lPhi0's other arm, so the rule holds whatever lmSlots finds; the branch a tap is on a value known
// before the taps, not on the tap's own load.
const LMPHI0 = `
fn lmPhi0(v:vec3i)->f32{
 if(FG_TIERED){let s=lmSlot(v);if(s!=NB_NONE&&all(v>=vec3i(0))){return fcLoadF(FC_PHIU+512u*s+nbLocal(v));}}
 return fgPhi(v);
}`;
const withCached = (code: string) => distinct(sub(withSlots(code), "fn lPhi0(v:vec3i)->f32{", LMPHI0 + "\nfn lPhi0(v:vec3i)->f32{"));
function cubicCached(code: string): string {
  code = sub(code, "fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{return lPhi0(lBase+", "fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{return lmPhi0(lBase+");
  return sub(code, "  lBase=cell;\n", "  lBase=cell;_=lmSlots(cell-vec3i(1),cell+vec3i(2));\n");
}
const gradCached = (code: string) => sub(code, "  for(var k=0u;k<10u;k++){t[k]=lPhi0(vec3i(at[k]));}",
  "  _=lmSlots(v-vec3i(2),v+vec3i(2));for(var k=0u;k<10u;k++){t[k]=lmPhi0(vec3i(at[k]));}");
const cornersCached = (code: string) => sub(code, " for(var k=0u;k<8u;k++){v[k]=lPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;",
  " _=lmSlots(c,c+vec3i(1));for(var k=0u;k<8u;k++){v[k]=lmPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;");
/** pMove's depth, cached: fgDepth's one loop, each corner fgDepthVertex's rule with the stencil's slot. */
function pmDepthCached(code: string): string {
  const f = fnText(code, "fgDepth");
  const g = sub(sub(f, " let f=p-vec3f(c);var value=0.0;\n", " let f=p-vec3f(c);var value=0.0;\n _=lmSlots(c,c+vec3i(1));\n"), "fgDepthVertex(c+vec3i(corner))", "lmDepthVertex(c+vec3i(corner))");
  return sub(code, f, "fn lmDepthVertex(v:vec3i)->f32{let s=lmSlot(v);if(s==NB_NONE){return NB_PHI_AIR;}return fcLoadF(FC_DEPTH+(512u*s+nbLocal(v)));}\n" + g);
}
// A landed fgDepth (FG_DEPTH_WGSL, fdBricks + fdVertex) back to HEAD 734dd9e2's text: the reference for a candidate
// tree's kernels in one process (REF without a dump run).
const FD_HEAD = `fn fgDepth(p:vec3f)->f32{
 let c=vec3i(floor(p));
 if(!fgReach1(c)){let far=fgPhiAt(p);return select(max(far,6.0),min(far,-6.0),far<0.0);}
 let f=p-vec3f(c);var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*fgDepthVertex(c+vec3i(corner));
 }
 return value;
}`;
export function depthHead(code: string): string {
  const at = code.indexOf("var<private> fdBrick:vec3i;"), f = fnText(code, "fgDepth");
  if (at < 0 || code.indexOf(f) < at) throw new Error("no landed fgDepth");
  return code.slice(0, at) + FD_HEAD + code.slice(code.indexOf(f) + f.length);
}
export const variants: Variant[] = [
  ...["pMove", "pSolid", "uTransfer"].map(kernel => ({ kernel, name: "depth-head", apply: depthHead })),
  { kernel: "pMove", name: "depthc", apply: c => pmDepthCached(pmSlots(c)) },
  { kernel: "pMove", name: "depthc+corners", apply: c => pmCorners(pmDepthCached(pmSlots(c))) },
  ...["lAdvect", "phiSolid"].flatMap(kernel => [
    { kernel, name: "cubic-cached", apply: (c: string) => cubicCached(withCached(c)) },
    { kernel, name: "cubic+grad-cached", apply: (c: string) => gradCached(cubicCached(withCached(c))) },
    { kernel, name: "all-cached", apply: (c: string) => cornersCached(gradCached(cubicCached(withCached(c)))) },
  ]),
  { kernel: "lAdvect", name: "cg+tri2", apply: c => distinct(tri2(gradient(cubic(withSlots(c))))) },
  { kernel: "lAdvect", name: "cg+cornsel", apply: c => distinct(cornSel(gradient(cubic(withSlots(c))))) },
  { kernel: "lAdvect", name: "cubic+grad+heat-distinct", apply: c => distinct(heat(gradient(cubic(withSlots(c))))) },
  { kernel: "phiSolid", name: "cubic-distinct", apply: c => distinct(cubic(withSlots(c))) },
  { kernel: "phiSolid", name: "cubic+grad-distinct", apply: c => distinct(gradient(cubic(withSlots(c)))) },
  { kernel: "phiSolid", name: "cubic+grad+heat-distinct", apply: c => distinct(heat(gradient(cubic(withSlots(c))))) },
  { kernel: "phiSolid", name: "cubic+grad+heat+buried-distinct", apply: c => distinct(buried(heat(gradient(cubic(withSlots(c)))))) },
  { kernel: "lAdvect", name: "cubic-distinct", apply: c => distinct(cubic(withSlots(c))) },
  { kernel: "lAdvect", name: "cubic+grad-distinct", apply: c => distinct(gradient(cubic(withSlots(c)))) },
  { kernel: "lAdvect", name: "corners-verify", apply: c => cornersVerify(withSlots(c)) },
  // Diagnostic: lPhi0 itself behind the lmSlots test (the same words by construction; only the control flow differs).
  { kernel: "lAdvect", name: "corners-same", apply: c => sub(corners(withSlots(c)), "v[k]=lmTap(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;", "v[k]=lPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;") },
  { kernel: "pMove", name: "depth", apply: c => pmDepth(pmSlots(c)) },
  { kernel: "pMove", name: "corners", apply: c => pmCorners(pmSlots(c)) },
  { kernel: "pMove", name: "depth+corners", apply: c => pmCorners(pmDepth(pmSlots(c))) },
  { kernel: "lAdvect", name: "all-distinct", apply: c => distinct(lAdv(c)) },
  { kernel: "phiSolid", name: "all+buried-distinct", apply: c => distinct(buried(lAdv(c))) },
  { kernel: "phiSolid", name: "cubic+corners+grad", apply: lAdv },
  { kernel: "phiSolid", name: "all+buried", apply: c => buried(lAdv(c)) },
  { kernel: "lAdvect", name: "cubic", apply: c => cubic(withSlots(c)) },
  { kernel: "lAdvect", name: "cubic+corners", apply: c => corners(cubic(withSlots(c))) },
  { kernel: "lAdvect", name: "cubic+corners+grad", apply: c => gradient(corners(cubic(withSlots(c)))) },
];
