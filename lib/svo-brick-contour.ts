/**
 * A conservative slab — a "contour", after Laine & Karras, *Efficient Sparse
 * Voxel Octrees* (I3D 2010) — for one 8^3 render-facing leaf brick.
 *
 * A leaf brick exists because it holds *at least one* solid cell, so the
 * in-brick DDA routinely enters a legitimately occupied brick, walks the ray's
 * whole chord, finds nothing, and moves on. Grazing terrain, the pond surface
 * and the sparse oak canopy all produce that. The only pre-test today is the
 * ray/brick interval, which every such chord passes.
 *
 * The contour is a plane normal `n` and two offsets `d0 <= d1` along it that
 * provably contain every solid cell of the brick. At brick entry the ray is
 * intersected with the slab and `[entry, brickExit]` is clamped; if the interval
 * collapses the brick is rejected with **zero DDA steps and zero payload loads**.
 *
 * ## Why it is straight-line ALU outside the loop
 *
 * The measured lesson on this exact code path is that *where* the test lives
 * decides whether it wins. `bounds` — one span clamp at brick entry, outside the
 * DDA — measured -5.9%. `macro` — a 2x2x2 mask consulted per DDA step, inside
 * the loop — measured +33%, because the skip path costs three vector reciprocals
 * plus a full DDA re-seed and burns one of the 32 iteration slots without
 * advancing the search. The slab test is therefore a fixed ~20 scalar operations
 * evaluated once, in exactly the place `bounds` already clamps, and never a
 * per-step predicate.
 *
 * ## Why it is conservative by construction
 *
 * The producer widens each solid cell's projected centre by that cell's own
 * projection radius `h * (|n.x| + |n.y| + |n.z|)`, which is the exact half-width
 * of an axis-aligned cell measured along `n`. No solid cell can then lie outside
 * `[d0, d1]`, so the test can only ever reject genuinely empty chords and the
 * settled frame must be byte-identical. The offsets are quantised *outwards*
 * (floor for `d0`, ceil for `d1`) and padded by one further quantisation step on
 * each side, so decoding can only widen the slab, never narrow it.
 *
 * ## Where it is stored
 *
 * In the 24 spare bits of `address.w` — node record word 3 — of the brick's own
 * terminal node. That is the same 32-byte record `traceLeafPayload` already
 * loads for `links.w` (the occupancy word), so the test adds no storage binding,
 * no resident bytes and no memory load. The low eight bits of that word are the
 * child mask, which is zero on a terminal node and which **every** reader in the
 * tree masks with `& 0xff` before use.
 *
 * `links.y` (child count) is 32 contiguous free bits on a leaf and looks like
 * the better home, but `packSvoCompactHierarchy` (`svo-compact-hierarchy.ts:82`)
 * reads word 5 on *every* node, terminals included, and throws unless it equals
 * `popcount(childMask)`. 24 bits it is.
 *
 * ## The invalid encoding is zero
 *
 * A brick whose fit is degenerate — a near-isotropic solid set, an unfittable
 * normal, or a slab that ends up spanning the whole brick — encodes zero, and
 * zero decodes to "no contour" so the test becomes a no-op. That is also what
 * every writer of word 3 that knows nothing about contours leaves behind:
 * `initializeNode` and `insertChild` in the topology mutation shader, and the
 * CPU packer in `sparse-brick-octree.ts`. A brick whose payload changes without
 * its contour being refitted is therefore the one real hazard, and the fluid
 * residency summariser clears the field for exactly that reason.
 *
 * A valid contour always has `q1 >= 8`: the cell-radius widening alone makes the
 * slab at least a quarter of the brick's own extent along `n`, which is 1/8 of
 * the encoded range. `q1 == 0` is therefore unambiguously "invalid" and costs no
 * separate validity bit.
 */

export const SVO_BRICK_CONTOUR = Object.freeze({
  /** Node record word 3. Bits 0..7 are the child mask; the contour rides above. */
  nodeWord: 3,
  shift: 8,
  bits: 24,
  /** Hemi-octahedral direction code: 6 bits per axis of the folded square. */
  normalBits: 12,
  normalMask: 0xfff,
  /** Quantised slab offsets, in 1/63 of the brick's own extent along the normal. */
  offsetBits: 6,
  offsetLevels: 63,
  /** Defined only for 8^3 bricks; the half-extent is in cell units. */
  brickSize: 8,
  halfExtentCells: 4,
  /** One existing node word is reused, so the incremental allocation is zero. */
  incrementalStorageBytesPerBrick: 0,
});

/** Pack a normal code and two quantised offsets into the 24-bit field. */
export function encodeSvoBrickContour(normalCode: number, low: number, high: number): number {
  return (((normalCode & SVO_BRICK_CONTOUR.normalMask)
    | ((low & 0x3f) << SVO_BRICK_CONTOUR.normalBits)
    | ((high & 0x3f) << (SVO_BRICK_CONTOUR.normalBits + SVO_BRICK_CONTOUR.offsetBits))) >>> 0);
}

/** The contour field of a node record word 3, or zero when the brick carries none. */
export function svoBrickContourField(nodeWord3: number): number {
  return ((nodeWord3 >>> SVO_BRICK_CONTOUR.shift) & 0xff_ffff) >>> 0;
}

export interface SvoBrickContourSummary {
  valid: boolean;
  /** Unit normal, in brick-local cell units (which are world units up to cell aspect). */
  normal: readonly [number, number, number];
  /** Slab offsets from the brick centre along the normal, in cell units. */
  low: number;
  high: number;
  /** Slab thickness as a fraction of the brick's own extent along the normal. */
  thickness: number;
}

/** The CPU mirror of `svoBrickContourDecode`, for oracles and censuses. */
export function decodeSvoBrickContour(nodeWord3: number): SvoBrickContourSummary {
  const packed = svoBrickContourField(nodeWord3);
  const high = (packed >>> (SVO_BRICK_CONTOUR.normalBits + SVO_BRICK_CONTOUR.offsetBits)) & 0x3f;
  if (high === 0) return { valid: false, normal: [0, 0, 1], low: 0, high: 0, thickness: 1 };
  const low = (packed >>> SVO_BRICK_CONTOUR.normalBits) & 0x3f;
  const normal = decodeSvoBrickContourNormal(packed & SVO_BRICK_CONTOUR.normalMask);
  const radius = SVO_BRICK_CONTOUR.halfExtentCells
    * (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]));
  const scale = (2 * radius) / SVO_BRICK_CONTOUR.offsetLevels;
  return {
    valid: true,
    normal,
    low: low * scale - radius,
    high: high * scale - radius,
    thickness: (high - low) / SVO_BRICK_CONTOUR.offsetLevels,
  };
}

/** The CPU mirror of `svoBrickContourNormal`. */
export function decodeSvoBrickContourNormal(code: number): readonly [number, number, number] {
  const levels = SVO_BRICK_CONTOUR.offsetLevels;
  const ex = ((code & 0x3f) * 2) / levels - 1;
  const ey = (((code >>> 6) & 0x3f) * 2) / levels - 1;
  const px = (ex + ey) * 0.5;
  const py = (ex - ey) * 0.5;
  const pz = 1 - Math.abs(px) - Math.abs(py);
  const length = Math.hypot(px, py, pz) || 1;
  return [px / length, py / length, pz / length];
}

/**
 * The CPU mirror of `svoBrickContourFit`, as the executable spec the GPU pass is
 * checked against. `solid` is one brick's 512 cells in the producer's own index
 * order (`x | y<<3 | z<<6`).
 */
export function fitSvoBrickContour(solid: ArrayLike<number>): number {
  const half = SVO_BRICK_CONTOUR.halfExtentCells;
  const levels = SVO_BRICK_CONTOUR.offsetLevels;
  let count = 0;
  const sum = [0, 0, 0];
  const square = [0, 0, 0];
  const mixed = [0, 0, 0];
  const minimum = [7, 7, 7];
  const maximum = [0, 0, 0];
  for (let index = 0; index < 512; index += 1) {
    if (!solid[index]) continue;
    const local = [index & 7, (index >>> 3) & 7, index >>> 6];
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], local[axis]);
      maximum[axis] = Math.max(maximum[axis], local[axis]);
    }
    const centre = local.map((value) => value + 0.5);
    count += 1;
    for (let axis = 0; axis < 3; axis += 1) { sum[axis] += centre[axis]; square[axis] += centre[axis] ** 2; }
    mixed[0] += centre[0] * centre[1];
    mixed[1] += centre[0] * centre[2];
    mixed[2] += centre[1] * centre[2];
  }
  if (count === 0) return 0;
  const mean = sum.map((value) => value / count);
  const diagonal = square.map((value, axis) => value / count - mean[axis] ** 2);
  const off = [
    mixed[0] / count - mean[0] * mean[1],
    mixed[1] / count - mean[0] * mean[2],
    mixed[2] / count - mean[1] * mean[2],
  ];
  let direction = leastSpread(diagonal, off);
  if (!(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2 > 0.5)) {
    const extent = maximum.map((value, axis) => value - minimum[axis]);
    let axis = 0;
    if (extent[1] < extent[0]) axis = 1;
    if (extent[2] < extent[axis]) axis = 2;
    direction = [0, 0, 0];
    direction[axis] = 1;
  }
  const code = encodeSvoBrickContourNormal(direction);
  if (code > SVO_BRICK_CONTOUR.normalMask) return 0;
  const normal = decodeSvoBrickContourNormal(code);
  const reach = Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]);
  const radius = half * reach;
  const cellRadius = 0.5 * reach;
  let lowest = 1e30;
  let highest = -1e30;
  for (let index = 0; index < 512; index += 1) {
    if (!solid[index]) continue;
    const projection = ((index & 7) + 0.5 - half) * normal[0]
      + (((index >>> 3) & 7) + 0.5 - half) * normal[1]
      + ((index >>> 6) + 0.5 - half) * normal[2];
    lowest = Math.min(lowest, projection - cellRadius);
    highest = Math.max(highest, projection + cellRadius);
  }
  if (!(highest >= lowest)) return 0;
  const scale = levels / (2 * radius);
  const low = Math.min(63, Math.max(0, Math.floor((lowest + radius) * scale) - 1));
  const high = Math.min(63, Math.max(0, Math.ceil((highest + radius) * scale) + 1));
  if (low <= 0 && high >= 63) return 0;
  if (high <= 0) return 0;
  return encodeSvoBrickContour(code, low, high);
}

/** The CPU mirror of `svoBrickContourEncodeNormal`. */
export function encodeSvoBrickContourNormal(direction: readonly number[]): number {
  const levels = SVO_BRICK_CONTOUR.offsetLevels;
  const normal = direction[2] < 0 ? direction.map((value) => -value) : [...direction];
  const norm = Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]);
  if (!(norm > 1e-12)) return 0xffff_ffff;
  const projected = [normal[0] / norm, normal[1] / norm];
  const folded = [projected[0] + projected[1], projected[0] - projected[1]];
  const quantized = folded.map((value) => Math.min(levels, Math.max(0, Math.round((value * 0.5 + 0.5) * levels))));
  return (quantized[0] | (quantized[1] << 6)) >>> 0;
}

/** Eigenvector of the smallest eigenvalue of a symmetric 3x3, mirroring the WGSL. */
function leastSpread(diagonal: readonly number[], off: readonly number[]): number[] {
  const cross2 = off[0] ** 2 + off[1] ** 2 + off[2] ** 2;
  const mean = (diagonal[0] + diagonal[1] + diagonal[2]) / 3;
  const centred = diagonal.map((value) => value - mean);
  const spread = centred[0] ** 2 + centred[1] ** 2 + centred[2] ** 2 + 2 * cross2;
  if (!(spread > 1e-9)) return [0, 0, 0];
  const radius = Math.sqrt(spread / 6);
  const matrix = [
    [diagonal[0], off[0], off[1]],
    [off[0], diagonal[1], off[2]],
    [off[1], off[2], diagonal[2]],
  ];
  const shifted = (value: number) => matrix.map((row, r) => row.map((entry, c) => entry - (r === c ? value : 0)));
  const determinant = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const normalized = shifted(mean).map((row) => row.map((value) => value / radius));
  const angle = Math.acos(Math.min(1, Math.max(-1, 0.5 * determinant(normalized)))) / 3;
  const largest = mean + 2 * radius * Math.cos(angle);
  const smallest = mean + 2 * radius * Math.cos(angle + 2.0943951023931953);
  const middle = 3 * mean - largest - smallest;
  const left = shifted(largest);
  const right = shifted(middle);
  const product = [0, 1, 2].map((r) => [0, 1, 2].map(
    (c) => left[r][0] * right[0][c] + left[r][1] * right[1][c] + left[r][2] * right[2][c]));
  let best = [product[0][0], product[1][0], product[2][0]];
  let bestLength = best[0] ** 2 + best[1] ** 2 + best[2] ** 2;
  for (const column of [1, 2]) {
    const candidate = [product[0][column], product[1][column], product[2][column]];
    const length = candidate[0] ** 2 + candidate[1] ** 2 + candidate[2] ** 2;
    if (length > bestLength) { best = candidate; bestLength = length; }
  }
  if (!(bestLength > 1e-20)) return [0, 0, 0];
  const inverse = 1 / Math.sqrt(bestLength);
  return best.map((value) => value * inverse);
}

/**
 * Decode and clamp. Shared verbatim by the producer and every consumer, because
 * the fit is measured against the *decoded* normal: the producer quantises the
 * direction first, decodes it, and only then projects the solid cells onto it.
 * Any direction is then legal — the normal's accuracy decides how thin the slab
 * is, never whether it contains the geometry.
 */
export const svoBrickContourWGSL = /* wgsl */ `
const SVO_BRICK_CONTOUR_SHIFT:u32=${SVO_BRICK_CONTOUR.shift}u;
const SVO_BRICK_CONTOUR_LEVELS:f32=${SVO_BRICK_CONTOUR.offsetLevels}.0;
const SVO_BRICK_CONTOUR_HALF:f32=${SVO_BRICK_CONTOUR.halfExtentCells}.0;
struct SvoBrickContour{valid:u32,normal:vec3f,low:f32,high:f32}
/**
 * Hemi-octahedral decode. The slab is symmetric under n -> -n, so only a
 * hemisphere needs codes and the folded square wastes none: 4 096 directions
 * over 2*pi steradians is about 1.8 degrees of angular resolution.
 */
fn svoBrickContourNormal(code:u32)->vec3f{
  let e=vec2f(f32(code&63u),f32((code>>6u)&63u))*(2.0/SVO_BRICK_CONTOUR_LEVELS)-vec2f(1.0);
  let p=vec2f(e.x+e.y,e.x-e.y)*0.5;
  return normalize(vec3f(p.x,p.y,1.0-abs(p.x)-abs(p.y)));
}
fn svoBrickContourDecode(nodeWord:u32)->SvoBrickContour{
  let packed=nodeWord>>SVO_BRICK_CONTOUR_SHIFT;
  let high=(packed>>18u)&63u;
  if(high==0u){return SvoBrickContour(0u,vec3f(0.0,0.0,1.0),0.0,0.0);}
  let normal=svoBrickContourNormal(packed&4095u);
  let radius=SVO_BRICK_CONTOUR_HALF*(abs(normal.x)+abs(normal.y)+abs(normal.z));
  let scale=2.0*radius/SVO_BRICK_CONTOUR_LEVELS;
  return SvoBrickContour(1u,normal,f32((packed>>12u)&63u)*scale-radius,f32(high)*scale-radius);
}
/**
 * \`vec3f(accepted, entry, exit)\` for a ray already expressed in brick-local cell
 * units — \`(origin - bounds[0]) / extent\` and \`direction / extent\`, which the
 * caller already has in hand. Working in cell units rather than world metres is
 * what lets the producer measure integer cell centres and the consumer measure a
 * ray without the two ever having to agree bit-for-bit about world position.
 */
fn svoBrickContourClamp(contour:SvoBrickContour,localOrigin:vec3f,localDirection:vec3f,tEnter:f32,tExit:f32)->vec3f{
  let offset=dot(localOrigin-vec3f(SVO_BRICK_CONTOUR_HALF),contour.normal);
  let slope=dot(localDirection,contour.normal);
  if(abs(slope)<1e-12){
    return select(vec3f(0.0),vec3f(1.0,tEnter,tExit),offset>=contour.low&&offset<=contour.high);
  }
  let first=(contour.low-offset)/slope;
  let second=(contour.high-offset)/slope;
  let spanEnter=max(tEnter,min(first,second));
  let spanExit=min(tExit,max(first,second));
  return select(vec3f(0.0),vec3f(1.0,spanEnter,spanExit),spanEnter<=spanExit);
}
`;

/**
 * The producer's half: a least-squares plane fit over the brick's own solid
 * cells, and the outward quantisation that makes the result conservative.
 *
 * The normal minimises the slab's thickness directly — the eigenvector of the
 * smallest eigenvalue of the solid set's covariance is, by definition, the
 * direction of least spread. That needs no primitive, no owner resolution, no
 * SDF evaluation and no local-to-world transform, and it uses the producer's own
 * solidity predicate over the cells it is already visiting, so predicate
 * divergence between fit and payload is impossible by construction.
 *
 * Requires \`svoBrickContourWGSL\` to be in scope: the fit is measured against
 * \`svoBrickContourNormal\` of the code it is about to store, never against the
 * unquantised direction.
 */
export const svoBrickContourFitWGSL = /* wgsl */ `
fn svoBrickContourEncodeNormal(direction:vec3f)->u32{
  var normal=direction;
  if(normal.z<0.0){normal=-normal;}
  let norm=abs(normal.x)+abs(normal.y)+abs(normal.z);
  if(!(norm>1e-12)){return 0xffffffffu;}
  let projected=normal.xy/norm;
  let folded=vec2f(projected.x+projected.y,projected.x-projected.y);
  let quantized=clamp(round((folded*0.5+vec2f(0.5))*SVO_BRICK_CONTOUR_LEVELS),
    vec2f(0.0),vec2f(SVO_BRICK_CONTOUR_LEVELS));
  return u32(quantized.x)|(u32(quantized.y)<<6u);
}
/**
 * The eigenvector of the smallest eigenvalue of a symmetric 3x3, by Smith's
 * closed form for the eigenvalues followed by \`(C - e1 I)(C - e2 I)\`, whose
 * columns are all proportional to the remaining eigenvector. Returns the zero
 * vector when the matrix is too near isotropic for a direction to mean anything;
 * the caller then falls back on the occupied box's thinnest axis.
 */
fn svoBrickContourLeastSpread(diagonal:vec3f,offDiagonal:vec3f)->vec3f{
  let cross2=dot(offDiagonal,offDiagonal);
  let mean=(diagonal.x+diagonal.y+diagonal.z)/3.0;
  let centred=diagonal-vec3f(mean);
  let spread=dot(centred,centred)+2.0*cross2;
  if(!(spread>1e-9)){return vec3f(0.0);}
  let radius=sqrt(spread/6.0);
  let matrix=mat3x3f(
    vec3f(diagonal.x,offDiagonal.x,offDiagonal.y),
    vec3f(offDiagonal.x,diagonal.y,offDiagonal.z),
    vec3f(offDiagonal.y,offDiagonal.z,diagonal.z));
  let unit=mat3x3f(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),vec3f(0.0,0.0,1.0));
  let normalized=(1.0/radius)*(matrix-mean*unit);
  let angle=acos(clamp(0.5*determinant(normalized),-1.0,1.0))/3.0;
  let largest=mean+2.0*radius*cos(angle);
  let smallest=mean+2.0*radius*cos(angle+2.0943951023931953);
  let middle=3.0*mean-largest-smallest;
  let product=(matrix-largest*unit)*(matrix-middle*unit);
  var best=product[0];var bestLength=dot(product[0],product[0]);
  let secondLength=dot(product[1],product[1]);
  if(secondLength>bestLength){best=product[1];bestLength=secondLength;}
  let thirdLength=dot(product[2],product[2]);
  if(thirdLength>bestLength){best=product[2];bestLength=thirdLength;}
  if(!(bestLength>1e-20)){return vec3f(0.0);}
  return best*inverseSqrt(bestLength);
}
/**
 * \`solid\` is the brick's own 512-bit occupancy, in the producer's own predicate;
 * \`sum\`, \`square\` and \`mixed\` are first and second moments of the solid cell
 * centres in cell units; \`extent\` is the occupied box the caller already has.
 */
fn svoBrickContourFit(solid:ptr<function,array<u32,16>>,count:f32,sum:vec3f,
    square:vec3f,mixed:vec3f,extent:vec3u)->u32{
  let mean=sum/count;
  let diagonal=square/count-mean*mean;
  let offDiagonal=mixed/count-vec3f(mean.x*mean.y,mean.x*mean.z,mean.y*mean.z);
  var direction=svoBrickContourLeastSpread(diagonal,offDiagonal);
  if(!(dot(direction,direction)>0.5)){
    var axis=0u;
    if(extent.y<extent.x){axis=1u;}
    if(extent.z<extent[axis]){axis=2u;}
    direction=vec3f(0.0);direction[axis]=1.0;
  }
  let code=svoBrickContourEncodeNormal(direction);
  if(code>4095u){return 0u;}
  // Measured against the *decoded* normal, which is what makes the quantisation
  // of the direction irrelevant to soundness.
  let normal=svoBrickContourNormal(code);
  let reach=abs(normal.x)+abs(normal.y)+abs(normal.z);
  let radius=SVO_BRICK_CONTOUR_HALF*reach;
  let cellRadius=0.5*reach;
  var lowest=1e30;var highest=-1e30;
  for(var localIndex=0u;localIndex<512u;localIndex+=1u){
    if((((*solid)[localIndex>>5u]>>(localIndex&31u))&1u)==0u){continue;}
    let centre=vec3f(f32(localIndex&7u),f32((localIndex>>3u)&7u),f32(localIndex>>6u))
      +vec3f(0.5-SVO_BRICK_CONTOUR_HALF);
    let projection=dot(centre,normal);
    lowest=min(lowest,projection-cellRadius);
    highest=max(highest,projection+cellRadius);
  }
  if(!(highest>=lowest)){return 0u;}
  // Outward quantisation, plus one further step of slack on each side. The step
  // is 1/63 of the brick's extent along the normal — about an eighth of a cell —
  // which is four orders of magnitude above any float disagreement between this
  // fit and the consumer's ray, and is an integer rather than a tuned epsilon.
  let scale=SVO_BRICK_CONTOUR_LEVELS/(2.0*radius);
  let low=clamp(i32(floor((lowest+radius)*scale))-1,0,63);
  let high=clamp(i32(ceil((highest+radius)*scale))+1,0,63);
  // A slab that spans the brick rejects nothing; spend no ALU on it at render time.
  if(low<=0&&high>=63){return 0u;}
  if(high<=0){return 0u;}
  return (code&4095u)|(u32(low)<<12u)|(u32(high)<<18u);
}
`;
