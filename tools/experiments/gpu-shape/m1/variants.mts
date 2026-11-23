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
// nbOperator (rowsF): the tile roots staged for every brick, not only a cold one. A warm brick's seam faces (a face
// into a 4h-owned tile) walked poRootPhiG a face (page, tile bit, child, then PPHI: a chain of dependent loads); the
// 96 tiles of the window are staged once by 96 lanes in parallel. The starts stay cold-only (a warm brick reads none:
// poStartRoot / poStartTile run only on a cold row's prolongation, and poTileStart is unused).
const OP_TILES_OLD = `fn poTiles(lid:u32,cold:bool){
  poRC=false;
  if(cold){
    if(lid<96u){
      let T=(poSO>>vec3u(2u))-vec3i(1,2,1)+vec3i(i32(lid%4u),i32((lid/4u)%6u),i32(lid/24u));
      let root=poRootPhiG(T);poSR[lid]=root;poSS[lid]=poTileStartR(T,root);
    }
    workgroupBarrier();
  }
  poRC=cold;
}`;
const OP_TILES_ROOTS = `var<private> poRS:bool;
fn poTiles(lid:u32,cold:bool){
  poRC=false;
  if(lid<96u){
    let T=(poSO>>vec3u(2u))-vec3i(1,2,1)+vec3i(i32(lid%4u),i32((lid/4u)%6u),i32(lid/24u));
    let root=poRootPhiG(T);poSR[lid]=root;if(cold){poSS[lid]=poTileStartR(T,root);}
  }
  workgroupBarrier();
  poRC=true;poRS=cold;
}`;
const opRoots = (c: string) => sub(sub(c, OP_TILES_OLD, OP_TILES_ROOTS),
  "fn poWindowAt(T:vec3i)->u32{let i=poTileWindow(T);if(i<0||!poRC){", "fn poWindowAt(T:vec3i)->u32{let i=poTileWindow(T);if(i<0||!poRS){");
// nbOperator's face loop (the faces are 46% of paddleboats' operator, 21% of dam's: the abl-nofaces attribution).
// frac: the row's own open fraction once a lane (poBandFace asked it a face, the capacity and the RHS again).
const LOOP_HEAD = "      for(var f=0u;f<6u;f++){\n        let a=f/2u;let sign=npSign(f);let pf=";
const opFrac = (c: string) => {
  if (!c.includes("fn poFrac(")) return c;
  c = sub(c, "struct PoFace {face:NpFace,qWet:bool,native:bool}", "var<private> poFracC:f32;\nstruct PoFace {face:NpFace,qWet:bool,native:bool}");
  c = sub(c, "let open=poFrac(c);", "let open=poFracC;");
  c = sub(c, "      let capacity=poFrac(c);", "      let capacity=poFracC;");
  c = sub(c, "rate*poFrac(c)+extra", "rate*poFracC+extra");
  return sub(c, "  let liquid=owned&&poRowWet(c,own);", "  let liquid=owned&&poRowWet(c,own);poFracC=poFrac(c);");
};
// pre: the loop's own loads issued together before it, a face each (the face plane's velocity, and with cuts the
// moving wall's W): the loop's branches no longer hold them back.
const opPre = (c: string) => {
  const cuts = c.includes("fn poFaceW(");
  c = sub(c, LOOP_HEAD, `      var preUS:array<f32,6>;${cuts ? "var preW:array<f32,6>;" : ""}
      for(var f=0u;f<6u;f++){let a=f/2u;var fc=c;if(npSign(f)>0){fc[a]+=1;}preUS[f]=poFacePlane(fc,a);${cuts ? "preW[f]=poFaceW(fc,a);" : ""}}
${LOOP_HEAD}`);
  c = sub(c, "        let us=poFacePlane(fc,a);", "        let us=preUS[f];");
  return cuts ? sub(c, "*(capacity-face.fraction)*poFaceW(fc,a);}", "*(capacity-face.fraction)*preW[f];}") : c;
};
// page: the row's own page word once a lane; a face whose cell across lies in the same page takes it (poAcross read it
// a face: the dir word, then its branches, ahead of the face's other reads).
const opPage = (c: string) => {
  c = sub(c, "fn poAcross(q:vec3i)->u32{", "var<private> poPP0:vec3i;var<private> poPW0:u32;\nfn poPageOf(p:vec3i)->u32{if(all(p==poPP0)){return poPW0;}return nbPageWord(p);}\nfn poAcross(q:vec3i)->u32{");
  c = sub(c, "  let w=nbPageWord(q>>vec3u(5u));\n  if(w==NB_NONE){return NP_ACROSS_AIR;}", "  let w=poPageOf(q>>vec3u(5u));\n  if(w==NB_NONE){return NP_ACROSS_AIR;}");
  return sub(c, "  let liquid=owned&&", "  poPP0=c>>vec3u(5u);poPW0=nbPageWord(poPP0);\n  let liquid=owned&&");
};
// pSeed's depth: a cell's centre and its sites share one floor cell (a site lies in its cell, but for a rounding to
// c + 1), so the cell's reach bit and 8 staged depth vertices are read once a cell, not once a site; the sum as psDepth's.
const PS_DEPTH = `fn psDepth(p:vec3f)->f32{
 let c=vec3i(floor(p));
 if(!psReach1(c)){let far=fgPhiAt(p);return select(max(far,6.0),min(far,-6.0),far<0.0);}`;
const psDepthOnce = (c: string) => {
  c = sub(c, PS_DEPTH, `var<private> psCC:vec3i=vec3i(-2147483647);var<private> psCR:bool;var<private> psCV:array<f32,8>;
fn psDepthCell(p:vec3f)->f32{
 let c=vec3i(floor(p));
 if(any(c!=psCC)){return psDepth(p);}
 if(!psCR){let far=fgPhiAt(p);return select(max(far,6.0),min(far,-6.0),far<0.0);}
 let f=p-vec3f(c);var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*psCV[k];
 }
 return value;
}
${PS_DEPTH}`);
  c = sub(c, "fn psSiteDistance(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepth(site);}", "fn psSiteDistance(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepthCell(site);}");
  return sub(c, " else{d=psDepth(centre);}", " else{psCC=c;psCR=psReach1(c);if(psCR){for(var k=0u;k<8u;k++){psCV[k]=psDepthVertex(c+vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)));}}d=psDepthCell(centre);}");
};
// On the landed lAdvect (e8a33481: lStencil/lStencilPhi): the cell's 8 corners (lCorners0) through the stencil too,
// for the ulp tier (2026-11-24). "ls-corners": lCorners0 itself takes the stencil; "ls-tri2": the trilinear in two
// copies, one over the stencil's words.
const LS_CORNERS = "fn lCorners0(c:vec3i)->array<f32,8>{\n var v:array<f32,8>;\n for(var k=0u;k<8u;k++){v[k]=lPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}\n return v;\n}";
const lsCorners = (c: string) => sub(c, LS_CORNERS, LS_CORNERS.replace("fn lCorners0(", "fn lCorners0Slow(") + `
fn lCorners0(c:vec3i)->array<f32,8>{
 if(!lStencil(c,c+vec3i(1))){return lCorners0Slow(c);}
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=lStencilPhi(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return v;
}`);
const lsTri2 = (c: string) => sub(c, "var value=nbfCellTrilinear(lCorners0(cell),f);", "var value=0.0;if(lStencil(cell,cell+vec3i(1))){var v:array<f32,8>;for(var k=0u;k<8u;k++){v[k]=lStencilPhi(cell+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}value=nbfCellTrilinear(v,f);}else{value=nbfCellTrilinear(lCorners0(cell),f);}");
// uExtend in 256-lane groups over the brick's two z halves in turn (GPU-shape, lane A2, 2026-11-24). The per-half
// sweep window (12 x 12 x 8, wV/wD 9.2 KB, not 13.8), wS4 as three f32 a state (4.5 KB, not 6) and wLiqW in wD's
// words (wD a u32 array, the distances bitcast; dead before the first seed): ~15.4 KB a group, under 16 KB, so two groups fit a core where one did.
const uextHalf = (code: string) => {
  code = sub(code, "var<workgroup> wV:array<f32,1728>;\nvar<workgroup> wD:array<f32,1728>;", "var<workgroup> wV:array<f32,1152>;\nvar<workgroup> wD:array<u32,1152>;\nvar<private> xZ0:i32;");
  code = sub(code, "var<workgroup> wS4:array<vec4f,375>;", "var<workgroup> wS4:array<f32,1125>;");
  code = sub(code, "var<workgroup> wLiqW:array<u32,432>;\n", "");
  code = sub(code, "return ((wLiqW[16u*(n.x+3u*n.y+9u*n.z)+(l>>5u)]>>(l&31u))&1u)!=0u;", "return ((wD[16u*(n.x+3u*n.y+9u*n.z)+(l>>5u)]>>(l&31u))&1u)!=0u;");
  code = sub(code, "fn xCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2);}", "fn xCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2,2,2-xZ0);}");
  code = sub(code, "fn xInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1);}", "fn xInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1,1,1-xZ0);}");
  code = sub(code, "fn xAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2));return q.x+12u*(q.y+12u*q.z);}", "fn xAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2,2,2-xZ0));return q.x+12u*(q.y+12u*q.z);}");
  code = sub(code, "let s=wS4[125u*a+d.x+5u*d.y+25u*d.z];return Uf4(s.xy,s.z);}", "let i=3u*(125u*a+d.x+5u*d.y+25u*d.z);return Uf4(vec2f(wS4[i],wS4[i+1u]),wS4[i+2u]);}");
  code = sub(code, "fn xUpwind(i:u32)->Ufx{let d=wD[i];", "fn xUpwind(i:u32)->Ufx{let d=bitcast<f32>(wD[i]);");
  code = sub(code, "let old=Ufx(wV[i],wD[i]);", "let old=Ufx(wV[i],bitcast<f32>(wD[i]));");
  const body = fnText(code, "uExtend");
  const half = `fn uExtend(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);var live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var o=vec3i(0);
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);}
 vRingMarkDir(o,lid,live);workgroupBarrier();
 if(lid==0u){sRing=select(0u,1u,!live||vRingOf(live));}
 if(workgroupUniformLoad(&sRing)!=0u){return;}
 let h=vH();
 xLow=o-vec3i(3);xBase4=o>>vec3u(2u);xBrick=o>>vec3u(3u);
 fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();
 if(lid<69u){atomicStore(&wLiq[lid],0u);}
 if(lid<27u){atomicStore(&wSwept[lid],0u);}
 if(lid==0u){atomicStore(&wReach,0u);}
 if(live){
  for(var j=lid;j<432u;j+=256u){
   let n=j>>4u;let b=xBrick+vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);
   let s=fgBrickSlot(b);var w=0u;if(s!=NB_NONE){w=fcLoad(FC_LIQ+16u*s+(j&15u));}
   wD[j]=w;
  }
  if(lid>=192u){
   let j=lid-192u;let T=xBase4+vec3i(i32(j&3u),i32((j>>2u)&3u),i32(j>>4u))-vec3i(1);
   var bits=0u;
   if(FG_TIERED&&nbInWallBox(4*T)&&!foTileOwnerH(T)){bits=1u;if(fgTileCentre4(T)< -0.001){bits=3u;}}
   if(fgTileSupported(4*T)){bits|=4u;}
   if(FG_TIERED&&foTileOwnerH(T)){bits|=8u;}
   wTile4[j]=bits;
  }
  if(FG_TIERED){
   for(var j=lid;j<375u;j+=256u){
    let d=j%125u;let a=j/125u;let C=xBase4+vec3i(vec3u(d%5u,(d/5u)%5u,d/25u))-vec3i(2);let T=C+fgUnit(a);
    let cs=x4Slot(T);var st=vec3f(0.0,0.0,-1.0);if(cs!=NB_NONE){st=x4Load(cs,1u,foTileIndex(T),a);}
    wS4[3u*j]=st.x;wS4[3u*j+1u]=st.y;wS4[3u*j+2u]=st.z;
   }
  }
 }
 workgroupBarrier();
 if(live){
  var wet=false;
  for(var j=lid;j<2197u;j+=256u){
   if(xLiquidStaged(xLow+vec3i(i32(j%13u),i32((j/13u)%13u),i32(j/169u)))){atomicOr(&wLiq[j>>5u],1u<<(j&31u));wet=true;}
  }
  var reach=select(0u,8u,wet);
  if(FG_TIERED){for(var j=lid;j<375u;j+=256u){if(ufxReached(wS4[3u*j+2u])){reach|=1u<<(j/125u);}}}
  if(reach!=0u){atomicOr(&wReach,reach);}
  if(FG_TIERED&&lid<216u){
   let n=lid>>3u;let b=xBrick+vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);let tc=2*b+fsBit4(lid&7u);
   if(foTileOwnerH(tc)&&fgTileSupported(4*tc)){atomicOr(&wSwept[n],1u);}
  }
 }
 workgroupBarrier();
 let reach=atomicLoad(&wReach);
 for(var a=0u;a<3u;a++){
  let sweep=(reach&(8u|(1u<<a)))!=0u;
  for(var hz=0u;hz<2u;hz++){
   xZ0=4*i32(hz);
   let l=lid+256u*hz;let c=o+nbLocalCoord(l);
   if(live&&sweep){
    for(var r=0u;r<5u;r++){
     let k=lid+256u*r;if(k>=1152u){continue;}
     let s=xSeed(o+xCell(k),a);
     wV[k]=s.v;wD[k]=bitcast<u32>(s.d);
    }
   }
   workgroupBarrier();
   var held:array<Ufx,3>;
   if(live&&sweep){for(var r=0u;r<3u;r++){let k=lid+256u*r;if(k<600u){held[r]=xSweepAt(xInset(k),h);}}}
   workgroupBarrier();
   if(live&&sweep){for(var r=0u;r<3u;r++){let k=lid+256u*r;if(k<600u){let i=xAt(xInset(k));wV[i]=held[r].v;wD[i]=bitcast<u32>(held[r].d);}}}
   workgroupBarrier();
   if(live&&(!FG_TIERED||xOwnerH(c)||xOwnerH(c-fgUnit(a)))&&!xCoarse(c-fgUnit(a))){
    var s=Ufx(0.0,-1.0);
    if(sweep){s=xSweepAt(nbLocalCoord(l),h);}else{s=xSeed(c,a);}
    let iu=vAt(NB_VEL_F_UP0,slot,a,l);
    var out=xFinish(s,c,a,xCode(c,a),vel[iu],xSwept(c-fgUnit(a)));
    if(!fgFaceOpen(c,a)){out.y=0.0;}
    vel[vAt(NB_VEL_F_US0,slot,a,l)]=out.x;vel[iu]=out.y;
   }
   workgroupBarrier();
  }
 }
}`;
  // The box test as the scene's text resolved it (pressureResolveBox-style: nbInLattice on an unpadded box).
  const box = /&&(nb\w+)\(4\*T\)&&!foTileOwnerH/.exec(body)?.[1]; if (!box) throw new Error("uExtend: the tile box test moved");
  const halfText = half.split("nbInWallBox(").join(`${box}(`);
  return sub(sub(code, body, halfText), "@compute @workgroup_size(512) fn uExtend(", "@compute @workgroup_size(256) fn uExtend(");
};

// uExt4 (GPU-shape lane A2, fifth task): uExtend's pattern on the 4h sweep. "half": 256 lanes sweep the page's two z
// halves in turn, wV/wD a half's 12 x 12 x 8 window (13.8 KB, not 20.7). "pub": one axis loop, each axis's state
// published (yPublish: the old loop body, continue as return) as soon as it is swept. "seam": the seam's
// vLiquidOwner(c) on the h-owned tile T, whose slot fs is held: fgCellCoarse(c) false (foCellOwnerH(c) =
// foTileOwnerH(T)) and fgSlot(c) = fs, so the FC_LIQ bit of fs, vLiquidOwner's own path.
const UX4_OLD = ` var state:array<vec3f,3>;
 if(flagged){
  for(var a=0u;a<3u;a++){
   for(var r=0u;r<4u;r++){let k=lid+512u*r;if(k<1728u){let s=ySeed(P+yCell(k),a);wV[k]=s.xy;wD[k]=s.z;}}
   workgroupBarrier();
   var held:array<Ufx,2>;
   for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){held[r]=ySweepAt(yInset(k),h4);}}
   workgroupBarrier();
   for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){let i=yAt(yInset(k));wV[i]=held[r].v;wD[i]=held[r].d;}}
   workgroupBarrier();
   let s=ySweepAt(nbLocalCoord(j),h4);state[a]=vec3f(s.v,s.d);
   workgroupBarrier();
  }
 }else{for(var a=0u;a<3u;a++){state[a]=x4Load(slot,0u,j,a);}}
 for(var a=0u;a<3u;a++){
`;
const UX4_END = `   vel[vAt(NB_VEL_F_US0,fs,a,nbLocal(c))]=value;vel[iu]=value-select(kept,s.y,take);
  }
 }
}`;
const ux4Half = (code: string) => {
  code = sub(code, "var<workgroup> wV:array<vec2f,1728>;\nvar<workgroup> wD:array<f32,1728>;", "var<workgroup> wV:array<vec2f,1152>;\nvar<workgroup> wD:array<f32,1152>;");
  code = sub(code, "var<private> yPage:vec3i;\nfn yCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2);}\nfn yInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1);}\nfn yAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2));return q.x+12u*(q.y+12u*q.z);}",
    "var<private> yPage:vec3i;\nvar<private> yZ0:i32;\nfn yCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2,2,2-yZ0);}\nfn yInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1,1,1-yZ0);}\nfn yAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2,2,2-yZ0));return q.x+12u*(q.y+12u*q.z);}");
  code = sub(code, "@compute @workgroup_size(512) fn uExt4(", "@compute @workgroup_size(256) fn uExt4(");
  code = sub(code, " let j=lid;let T=P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));let h4=4.0*vH();\n", " let h4=4.0*vH();\n");
  code = sub(code, " if(x4Support(T)){atomicOr(&wFlag,1u);}\n", " for(var hz=0u;hz<2u;hz++){let j=lid+256u*hz;if(x4Support(P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u)))){atomicOr(&wFlag,1u);}}\n");
  code = sub(code, UX4_OLD, " for(var hz=0u;hz<2u;hz++){\n yZ0=4*i32(hz);let j=lid+256u*hz;let T=P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));\n" +
    UX4_OLD.replace("r<4u;r++){let k=lid+512u*r;if(k<1728u)", "r<5u;r++){let k=lid+256u*r;if(k<1152u)").replace("array<Ufx,2>", "array<Ufx,3>")
      .split("for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u)").join("for(var r=0u;r<3u;r++){let k=lid+256u*r;if(k<600u)"));
  return sub(code, UX4_END, UX4_END.slice(0, -1) + " }\n}");
};
const ux4Pub = (code: string) => {
  code = sub(code, " var state:array<vec3f,3>;\n if(flagged){\n  for(var a=0u;a<3u;a++){\n   for(var r=0u;r<5u;r++){let k=lid+256u*r;if(k<1152u){let s=ySeed(P+yCell(k),a);wV[k]=s.xy;wD[k]=s.z;}}",
    " for(var a=0u;a<3u;a++){\n  var s=vec3f(0.0);\n  if(flagged){\n   for(var r=0u;r<5u;r++){let k=lid+256u*r;if(k<1152u){let e=ySeed(P+yCell(k),a);wV[k]=e.xy;wD[k]=e.z;}}");
  code = sub(code, `   let s=ySweepAt(nbLocalCoord(j),h4);state[a]=vec3f(s.v,s.d);
   workgroupBarrier();
  }
 }else{for(var a=0u;a<3u;a++){state[a]=x4Load(slot,0u,j,a);}}
 for(var a=0u;a<3u;a++){
  let s=state[a];x4Store(slot,1u,j,a,s);
  if(vcFace(T,a)){vel[vcAt(NB_VEL_C_US0,slot,a,j)]=s.x;vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=s.x-s.y;continue;}
  let O=T-fgUnit(a);
  if(!foTileOwnerH(T)||!foTileInLattice(O)||foTileOwnerH(O)||fgTileCSlot(O)==NB_NONE){continue;}
`, `   let t=ySweepAt(nbLocalCoord(j),h4);s=vec3f(t.v,t.d);
   workgroupBarrier();
  }else{s=x4Load(slot,0u,j,a);}
  yPublish(slot,j,T,a,s);
`);
  const i = code.indexOf("  // ufxPublish4's seam"), e = code.indexOf("  }\n", code.indexOf("vel[iu]=value-select(kept,s.y,take);", i)) + 4;
  const seam = code.slice(i, e); code = code.slice(0, i) + code.slice(e);
  const fn = "// uExt4's publish of tile T's axis-a state s: the C face, or ufxPublish4's seam to an h tile.\nfn yPublish(slot:u32,j:u32,T:vec3i,a:u32,s:vec3f){\n x4Store(slot,1u,j,a,s);\n" +
    " if(vcFace(T,a)){vel[vcAt(NB_VEL_C_US0,slot,a,j)]=s.x;vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=s.x-s.y;return;}\n" +
    " let O=T-fgUnit(a);\n if(!foTileOwnerH(T)||!foTileInLattice(O)||foTileOwnerH(O)||fgTileCSlot(O)==NB_NONE){return;}\n" +
    seam.replace(/\n$/, "").split("\n").map(l => l.slice(1)).join("\n") + "\n}\n";
  return sub(code, "@compute @workgroup_size(256) fn uExt4(", fn + "@compute @workgroup_size(256) fn uExt4(");
};
const ux4Seam = (code: string) => sub(sub(code, "let source=!fgFaceOpen(c,a)||owner||vLiquidOwner(c);", "let source=!fgFaceOpen(c,a)||owner||yLiquidH(c,fs);"),
  "// uExt4's publish of tile T's axis-a state s:", "// vLiquidOwner(c) for a cell c of an h-owned tile held by slot fs (fgCellCoarse false, fgSlot(c) = fs).\nfn yLiquidH(c:vec3i,fs:u32)->bool{if(!nbInLattice(c)){return false;}let l=nbLocal(c);return ((fcLoad(FC_LIQ+16u*fs+(l>>5u))>>(l&31u))&1u)!=0u;}\n// uExt4's publish of tile T's axis-a state s:");

export const variants: Variant[] = [
  { kernel: "uExt4", name: "half", apply: ux4Half },
  // "axis": the page over C_ACT4_G1's four workgroups (g/4 the page, as uExt4Seed), workgroup g%4 sweeps axis g%4 and
  // the fourth leaves: the three axes' sweeps run side by side, not one after another (the dispatch moves to C_ACT4_G1).
  { kernel: "uExt4", name: "axis", apply: c => sub(sub(sub(sub(c, "sCount=nbListCount(NB_LIST_C_ACT);", "sCount=4u*nbListCount(NB_LIST_C_ACT);"),
    "if(g>=workgroupUniformLoad(&sCount)){return;}", "if(g>=workgroupUniformLoad(&sCount)||g%4u==3u){return;}"),
    "let slot=nbListSlot(NB_LIST_C_ACT,g);", "let slot=nbListSlot(NB_LIST_C_ACT,g/4u);"), "for(var a=0u;a<3u;a++){", "for(var a=g%4u;a<g%4u+1u;a++){", 3) },
  { kernel: "uExt4", name: "pub", apply: c => ux4Pub(ux4Half(c)) },
  { kernel: "uExt4", name: "seam", apply: c => ux4Seam(ux4Pub(ux4Half(c))) },
  // "seam2": the seam's face tests before its loads (pure reads either way; Metal caps it at 768, three groups a core).
  { kernel: "uExt4", name: "seam2", apply: c => sub(ux4Seam(ux4Pub(ux4Half(c))), "  let iu=vAt(NB_VEL_F_UP0,fs,a,nbLocal(c));\n  let own=fgFace(NB_VEL_F_U0,c,a);let kept=vel[iu];\n  let source=!fgFaceOpen(c,a)||owner||yLiquidH(c,fs);\n  let take=!source&&reached;",
    "  let source=!fgFaceOpen(c,a)||owner||yLiquidH(c,fs);\n  let take=!source&&reached;\n  let iu=vAt(NB_VEL_F_UP0,fs,a,nbLocal(c));\n  let own=fgFace(NB_VEL_F_U0,c,a);let kept=vel[iu];") },
  { kernel: "uExtend", name: "half", apply: uextHalf },
  { kernel: "lAdvect", name: "ls-corners", apply: lsCorners },
  { kernel: "lAdvect", name: "ls-tri2", apply: lsTri2 },
  { kernel: "pSeed", name: "depth", apply: psDepthOnce },
  // depth-w: the write pass takes the uncached depth (a probe of where depth's difference lies).
  { kernel: "pSeed", name: "depth-w", apply: c => sub(psDepthOnce(c), "let distance=psSiteDistance(site,reentry);\n  let u=fgSampleVel", "let distance=select(psDepth(site),fgPhiAt(site),reentry);\n  let u=fgSampleVel") },
  // depth-x: the count pass's sites cached (psSiteDistanceCell); the write pass keeps psSiteDistance (uncached).
  { kernel: "pSeed", name: "depth-x", apply: c => sub(sub(psDepthOnce(c), "fn psSiteDistance(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepthCell(site);}", "fn psSiteDistance(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepth(site);}\nfn psSiteDistanceCell(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepthCell(site);}"), "let distance=psSiteDistance(site,reentry);\n  if(!known", "let distance=psSiteDistanceCell(site,reentry);\n  if(!known") },
  // depth-c: only the count pass's sites cached; the centre's depth stays psDepth's.
  { kernel: "pSeed", name: "depth-c", apply: c => sub(psDepthOnce(c), "d=psDepthCell(centre);}", "d=psDepth(centre);}") },
  { kernel: "operator", name: "page", apply: opPage },
  { kernel: "operator", name: "frac", apply: opFrac },
  { kernel: "operator", name: "pre", apply: opPre },
  { kernel: "operator", name: "frac+pre", apply: c => opPre(opFrac(c)) },
  { kernel: "operator", name: "roots", apply: opRoots },
  // Attribution only (not exact, not landings): the stage run twice; the rows skipped after the stage; the faces
  // skipped (the starts kept); the starts skipped (0) with the faces kept.
  { kernel: "operator", name: "abl-stage2x", apply: c => sub(c, "  poStage(nbFOrigin(slot),lid,512u);", "  poStage(nbFOrigin(slot),lid,512u);poStage(nbFOrigin(slot),lid,512u);") },
  { kernel: "operator", name: "abl-norows", apply: c => sub(c, "  poTiles(lid,workgroupUniformLoad(&poGo)!=0u);", "  poTiles(lid,workgroupUniformLoad(&poGo)!=0u);if(cfg.w!=12345u){return;}") },
  { kernel: "operator", name: "abl-nofaces", apply: c => sub(c, "      for(var f=0u;f<6u;f++){\n        let a=f/2u;let sign=npSign(f);let pf=", "      for(var f=0u;f<6u*u32(cfg.w==12345u);f++){\n        let a=f/2u;let sign=npSign(f);let pf=") },
  { kernel: "operator", name: "abl-nostart", apply: c => sub(c, "var start=0.0;if(liquid){start=poStart(c,own);}", "var start=0.0;if(liquid&&cfg.w==12345u){start=poStart(c,own);}") },
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
