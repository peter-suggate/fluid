/** Ownership-only recipes in the second stencil word. Compact owner and
 * hanging-slot indices are deliberately absent: a remote edit renumbers them.
 * Bit 0 remains the detail ring; bits 27..31 remain the minimum width.
 * 1..8: coarse incident tiles T+c-1, c in {0,1}³.
 * 9..15: coarse tiles T+c, c=1..7 (surface h vertex ownership).
 * 16..23: canonical corners owned by this coarse tile.
 * Builders supply a clipped 27-bit valid mask so domain walls need no fake
 * neighbors. Both builders use the same masks and bit assignments below. */
export const UNIFORM_COMPILED_TOPOLOGY = { incident: 1, positive: 8, corners: 16 } as const;
/** Six ordered 3-bit side IDs: fine sides first, then coarse/boundary sides.
 * A seam lane selects its ordinal directly instead of scanning six sides
 * again in every proposal, limiter and commit sweep. */
export const uniformSeamSideOrders = Array.from({ length: 64 }, (_, sides) => {
  const order = [0, 1, 2, 3, 4, 5].filter(s => sides & (1 << s))
    .concat([0, 1, 2, 3, 4, 5].filter(s => !(sides & (1 << s))));
  return order.reduce((word, side, ordinal) => word | (side << (3 * ordinal)), 0);
});
const corner = (k: number) => [k & 1, (k >> 1) & 1, k >> 2];
const stencilCoord = (k: number) => [k % 3 - 1, Math.floor(k / 3) % 3 - 1, Math.floor(k / 9) - 1];
// Within the center tile, a neighbor's box distance is the maximum of its
// nonzero-axis distances. A matching strict subset of axes always dominates
// it. Keep the minimal fine neighbors under that partial order.
const blendDominators = Array.from({ length: 27 }, (_, i) => {
  const a = stencilCoord(i); let mask = 0;
  for (let j = 0; j < 27; j++) if (j !== i && stencilCoord(j).every((v, axis) => v === 0 || v === a[axis])) mask |= 1 << j;
  return mask;
});
export function compileUniformBlendMask(fine: number): number {
  let result = fine & 0x07ffffff;
  for (let i = 0; i < 27; i++) if (fine & blendDominators[i]!) result &= ~(1 << i);
  return result;
}
const spatial = (x: number, y: number, z: number) => 1 << (x + 1 + 3 * (y + 1 + 3 * (z + 1)));
const negative = Array.from({ length: 8 }, (_, k) => { const c = corner(k); return spatial(c[0]! - 1, c[1]! - 1, c[2]! - 1); });
const positive = Array.from({ length: 8 }, (_, k) => { const c = corner(k); return spatial(c[0]!, c[1]!, c[2]!); });
// At corner k the current tile is incident candidate 7^k. All preceding
// candidates have smaller tile keys, hence smaller coarse owner indices.
const earlier = Array.from({ length: 8 }, (_, k) => {
  const c = corner(k); let mask = 0;
  for (let j = 0; j < (7 ^ k); j++) { const b = corner(j); mask |= spatial(c[0]! + b[0]! - 1, c[1]! + b[1]! - 1, c[2]! + b[2]! - 1); }
  return mask;
});
export const uniformVertexIncidentMasks = Array.from({ length: 8 }, (_, required) => {
  let mask = 0; for (let k = 0; k < 8; k++) if ((k & required) === required) mask |= 1 << k;
  return mask;
});

/** One 2³ octant of a 3³ fine stencil. Eight such masks cover the
 * redistance window's 4³ tiles, using centers T+octant and clipping walls. */
export function uniformStencilOctant(fine: number, octant: number): number {
  const c = corner(octant), shifted = fine >>> (c[0]! + 3 * c[1]! + 9 * c[2]!);
  let mask = 0;
  for (let k = 0; k < 8; k++) if (shifted & negative[k]!) mask |= 1 << k;
  return mask;
}

export function compileUniformStencil(fine: number, valid: number): number {
  const coarse = valid & ~fine; let recipe = 0;
  for (let k = 0; k < 8; k++) {
    if (coarse & negative[k]!) recipe |= 1 << (UNIFORM_COMPILED_TOPOLOGY.incident + k);
    if (k && (coarse & positive[k]!)) recipe |= 1 << (UNIFORM_COMPILED_TOPOLOGY.positive + k);
    if ((coarse & (1 << 13)) && !(coarse & earlier[k]!)) recipe |= 1 << (UNIFORM_COMPILED_TOPOLOGY.corners + k);
  }
  return recipe;
}

/** Same finite compiler as compileUniformStencil, run by GPU classification
 * before owner numbering. No extra dispatch or metadata allocation. */
export const uniformCompileStencilWGSL = /* wgsl */ `
fn umCompileBlendMask(fine:u32)->u32{
 if((fine&8192u)!=0u){return 8192u;}
 var result=fine&0x07ffffffu;
 ${blendDominators.map((mask,i)=>mask?`if((fine&${mask}u)!=0u){result&=~${1<<i}u;}`:"").join("\n ")}
 return result;
}
fn umCompileStencil(fine:u32,valid:u32)->u32{
 let coarse=valid&~fine;var recipe=0u;
 ${negative.map((mask, k) => `recipe|=select(0u,${1 << (UNIFORM_COMPILED_TOPOLOGY.incident + k)}u,(coarse&${mask}u)!=0u);`).join("\n ")}
 ${positive.slice(1).map((mask, j) => `recipe|=select(0u,${1 << (UNIFORM_COMPILED_TOPOLOGY.positive + j + 1)}u,(coarse&${mask}u)!=0u);`).join("\n ")}
 if((coarse&8192u)!=0u){
 ${earlier.map((mask, k) => `recipe|=select(0u,${1 << (UNIFORM_COMPILED_TOPOLOGY.corners + k)}u,(coarse&${mask}u)==0u);`).join("\n ")}
 }return recipe;
}
`;

/** Shared consumers. The selector names a relative coarse tile, not a
 * generation's compact index. Its eligibility is the local coordinate's
 * nonzero axes. firstTrailingBit preserves the old minimum-owner ordering. */
export const uniformCompiledTopologyWGSL = /* wgsl */ `
const UM_VERTEX_INCIDENT:array<u32,8> = array<u32,8>(${uniformVertexIncidentMasks.map(n => `${n}u`).join(",")});
fn umVertexRecipe(t:u32)->u32{return umTopology[2u*UM_TILES+2u*t+1u];}
fn umVelocityBlendMask(t:u32)->u32{return umTopology[4u*UM_TILES+t];}
fn umFineWideMask(t:u32)->u32{return (umVertexRecipe(t)>>${UNIFORM_COMPILED_TOPOLOGY.positive}u)&254u;}
fn umCoarseCornerMask(t:u32)->u32{return (umVertexRecipe(t)>>${UNIFORM_COMPILED_TOPOLOGY.corners}u)&255u;}
fn umCoarseIncident(t:u32,local:vec3u)->u32{
 let required=u32(local.x!=0u)|(u32(local.y!=0u)<<1u)|(u32(local.z!=0u)<<2u);
 return ((umVertexRecipe(t)>>${UNIFORM_COMPILED_TOPOLOGY.incident}u)&255u)&UM_VERTEX_INCIDENT[required];
}
// Window tile offsets -1..+2 are eight 2³ octants. Their widths already
// exist in eight neighboring 3³ stencils: no per-tile word reads or atomics.
fn umWindowFineOctant(tile:vec3u,octant:u32)->u32{
 let c=umCorner(octant,2u);let center=tile+c;if(any(center>=UM_T)){return 0u;}
 let m=umTileStencil(umTileAt(center)).x>>(c.x+3u*c.y+9u*c.z);
 return ${negative.map((mask,k)=>`(((m>>${Math.log2(mask)}u)&1u)<<${k}u)`).join("|")};
}
// Six face-neighbor h bits, +x,-x,+y,-y,+z,-z. The original 27-bit
// stencil already compiled them, including clipped domain boundaries.
fn umFineFaceSides(t:u32)->u32{
 let m=umTileStencil(t).x;
 return ((m>>14u)&1u)|(((m>>12u)&1u)<<1u)|(((m>>16u)&1u)<<2u)|(((m>>10u)&1u)<<3u)|(((m>>22u)&1u)<<4u)|(((m>>4u)&1u)<<5u);
}
`;

/** The shared field reconstruction body. Callers stage their 27 coarse
 * lattice values; selection, weights, lane ownership and D4 arithmetic are
 * identical for phi and surface-volume scales. */
export function uniformCompiledVertexResolveWGSL(lattice: string, weight: string, sum: string): string {
  return /* wgsl */ `
 let local=umCorner(lane,5u);let p=vec3u(base)*4u+local;
 if(any((local==vec3u(4u))&(p!=UM_D))||all(p%4u==vec3u(0))){return;}
 let incident=umCoarseIncident(tile,local);if(incident==0u){return;}
 let at=umCorner(firstTrailingBit(incident),2u);
 let t=vec3f(local+(vec3u(1)-at)*4u)/4.0;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=${weight}(t,corner);let m=at+corner;
  if(w>0.0){values[k]=w*${lattice}[m.x+3u*(m.y+3u*m.z)];}
 }
 let value=${sum}(values);
`;
}
