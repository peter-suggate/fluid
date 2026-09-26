/**
 * The backdrop's inner tile levels as stored octree voxels: the detail rings.
 *
 * ## What is stored
 *
 * Ring `l` (`l < rings`) is walk level `l` of `backdrop-terrain-tiles.ts`: the
 * square `c0 +- R0 2^l` minus ring `l - 1` (ring 0 minus the set's footprint).
 * Its ground is the walk's own voxelisation, one column per cell of
 * `h0 2^l`, the top `anchor + round((h - anchor) / cell) cell` from the same
 * f32 field evaluator, so a stored ring is exactly what the walk drew at that
 * level and every seam (set slab, ring to ring, last ring to the walk) is
 * flush by construction. Unlike the walk, it is octree content: it casts and
 * receives voxel shadows, AO and cone GI like the set.
 *
 * On the ground stands seeded, set-style scatter: gravel, stones and lobed
 * shrub puffs on jittered grids. Every item lies inside its own grid cell, so
 * a voxel reads one hash per class. Items smaller than 0.6 of their ring's
 * cell are dropped and density fades over the outer half of the detail
 * square, so detail thins and coarsens outward rather than stopping at an
 * edge. The scatter is evaluated in the voxelizer only: it is never an
 * authored primitive, so the solver, the scenery graph and the primitive arena
 * never see it.
 *
 * ## Planning
 *
 * `createBackdropDetailClassifier` answers the planner's supplemental descent
 * (`classifySupplementalNode`): empty, leaf (at the ring's leaf level,
 * `solverLevel - l`) or split. Coarse nodes use conservative height bounds;
 * leaf-level nodes use the exact column tops over the brick and a one-column
 * margin, plus the scatter items that reach it, with relaxed acceptance so the
 * CPU claims a superset of what the GPU voxelizes.
 *
 * No DOM or GPU imports.
 */
import {
  backdropFieldHeight,
  backdropHash,
  backdropHeightBounds,
  type BackdropField,
} from "./backdrop-field";
import {
  BACKDROP_TERRAIN_TABLE,
  backdropTerrainWGSL,
  type BackdropTilePlan,
} from "./backdrop-terrain-tiles";

/**
 * Scatter classes. `shape` 0 is a rotated, half-seated ellipsoid; 1 a
 * three-lobe puff. `minimumRing` keeps a class out of the inner rings: a bush
 * voxelized at the set's own cell is most of ring 0's bricks (its surface
 * area in 50 mm bricks), and beside the pond the set brings its own planting.
 */
export const BACKDROP_SCATTER_CLASSES = Object.freeze([
  { name: "gravel", grid_m: 0.1, density: 0.5, radius_m: [0.01, 0.028], shape: 0, minimumRing: 0 },
  { name: "stones", grid_m: 0.3, density: 0.6, radius_m: [0.03, 0.11], shape: 0, minimumRing: 0 },
  { name: "shrubs", grid_m: 0.8, density: 0.45, radius_m: [0.1, 0.25], shape: 1, minimumRing: 0 },
  { name: "bushes", grid_m: 2.4, density: 0.5, radius_m: [0.22, 0.45], shape: 1, minimumRing: 1 },
] as const);

/** Horizontal reach of an item in its radius: the ellipsoid's 1.3 R, the puff's lobes. */
const SCATTER_REACH = [1.3, 1.3] as const;
/** Height of an item's top above its seat, in its radius. */
const SCATTER_RISE = [0.85, 1.6] as const;
/** Items below this fraction of their ring's cell are dropped. */
export const BACKDROP_SCATTER_MINIMUM_CELLS = 0.6;
/** Density fades from this fraction of the detail half-width to `FADE_END`. */
const FADE_BEGIN = 0.45;
const FADE_END = 0.95;
/** The largest height any item reaches above the ground. */
export const BACKDROP_SCATTER_MAXIMUM_RISE_M = Math.max(...BACKDROP_SCATTER_CLASSES.map(
  (entry) => entry.radius_m[1] * SCATTER_RISE[entry.shape]));

export interface BackdropDetail {
  rings: number;
  /** Clipmap centre c0 (x, z). */
  centre: readonly [number, number];
  halfWidth0_m: number;
  cellSize0_m: number;
  anchor_m: number;
  /** Half-width of the outermost stored ring's square. */
  outerHalf_m: number;
  footprint_m: readonly [number, number, number, number];
  seed: number;
}

export function backdropDetailFromPlan(field: BackdropField, plan: BackdropTilePlan, rings: number): BackdropDetail {
  if (!Number.isInteger(rings) || rings < 1) throw new RangeError("Backdrop detail needs at least one ring");
  return {
    rings,
    centre: plan.centre,
    halfWidth0_m: plan.halfWidth0_m,
    cellSize0_m: plan.cellSize0_m,
    anchor_m: plan.anchor_m,
    outerHalf_m: plan.halfWidth0_m * 2 ** (rings - 1),
    footprint_m: field.seam.footprint_m as unknown as readonly [number, number, number, number],
    seed: field.description.seed >>> 0,
  };
}

/** The outermost ring's brick edge: the lattice the centre snaps to. */
export function backdropDetailCentreLattice(cellSize0_m: number, rings: number, brickSize = 8): number {
  return brickSize * cellSize0_m * 2 ** (rings - 1);
}

/** Ring of a Chebyshev distance from the centre (the walk's level lookup). */
export function backdropDetailRing(detail: BackdropDetail, m: number): number {
  return m <= detail.halfWidth0_m ? 0 : Math.min(detail.rings - 1, Math.ceil(Math.log2(m / detail.halfWidth0_m)));
}

function insideFootprint(detail: BackdropDetail, x: number, z: number): boolean {
  const [minX, minZ, maxX, maxZ] = detail.footprint_m;
  return x > minX && x < maxX && z > minZ && z < maxZ;
}

function footprintDistance(detail: BackdropDetail, x: number, z: number): number {
  const [minX, minZ, maxX, maxZ] = detail.footprint_m;
  return Math.hypot(Math.max(minX - x, 0, x - maxX), Math.max(minZ - z, 0, z - maxZ));
}

function ringCellAt(detail: BackdropDetail, x: number, z: number): number {
  const m = Math.max(Math.abs(x - detail.centre[0]), Math.abs(z - detail.centre[1]));
  return detail.cellSize0_m * 2 ** backdropDetailRing(detail, m);
}

/** The column top holding (x, z) at `cell`: the walk's rule. */
export function backdropDetailColumnTop(field: BackdropField, detail: BackdropDetail, cell: number, x: number, z: number): number {
  const cx = detail.centre[0] + (Math.floor((x - detail.centre[0]) / cell) + 0.5) * cell;
  const cz = detail.centre[1] + (Math.floor((z - detail.centre[1]) / cell) + 0.5) * cell;
  return detail.anchor_m + Math.floor((backdropFieldHeight(field, cx, cz) - detail.anchor_m) / cell + 0.5) * cell;
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export interface BackdropScatterItem {
  classIndex: number;
  shape: 0 | 1;
  x: number;
  z: number;
  radius: number;
  /** Ground top the item sits on. */
  seat: number;
  /** Highest point. */
  top: number;
  /** Horizontal reach from (x, z). */
  reach: number;
}

/**
 * The item in grid cell (qx, qz) of a class, or undefined. `slack` loosens the
 * acceptance and size tests (the CPU planner's superset); zero is the GPU rule.
 */
export function backdropScatterItem(
  field: BackdropField, detail: BackdropDetail, classIndex: number, qx: number, qz: number, slack = 0,
): BackdropScatterItem | undefined {
  const entry = BACKDROP_SCATTER_CLASSES[classIndex]!;
  const salt = classIndex + 1;
  const accept = backdropHash(detail.seed, salt, qx, qz, 0) / 0x1_0000_0000;
  const u1 = backdropHash(detail.seed, salt, qx, qz, 1) / 0x1_0000_0000;
  const u2 = backdropHash(detail.seed, salt, qx, qz, 2) / 0x1_0000_0000;
  const u3 = backdropHash(detail.seed, salt, qx, qz, 3) / 0x1_0000_0000;
  const radius = entry.radius_m[0] + (entry.radius_m[1] - entry.radius_m[0]) * u1 * u1;
  const reach = SCATTER_REACH[entry.shape] * radius;
  const g = entry.grid_m;
  const x = (qx + (reach + (g - 2 * reach) * u2) / g) * g;
  const z = (qz + (reach + (g - 2 * reach) * u3) / g) * g;
  const m = Math.max(Math.abs(x - detail.centre[0]), Math.abs(z - detail.centre[1]));
  const fade = 1 - smoothstep(FADE_BEGIN * detail.outerHalf_m, FADE_END * detail.outerHalf_m, m);
  if (!(accept < entry.density * fade + slack)) return undefined;
  if (m + reach >= detail.outerHalf_m + slack * detail.outerHalf_m) return undefined;
  if (footprintDistance(detail, x, z) <= reach - slack) return undefined;
  const cell = ringCellAt(detail, x, z);
  // Ring cells are exact powers of two of cell 0, so 0.75 separates rings robustly.
  if (cell < 0.75 * detail.cellSize0_m * 2 ** entry.minimumRing) return undefined;
  if (radius < BACKDROP_SCATTER_MINIMUM_CELLS * cell * (1 - slack)) return undefined;
  const seat = backdropDetailColumnTop(field, detail, cell, x, z);
  return { classIndex, shape: entry.shape, x, z, radius, seat, top: seat + SCATTER_RISE[entry.shape] * radius, reach };
}

/** Every scatter item (GPU rule) whose reach overlaps the xz rectangle. */
export function backdropScatterItemsIn(
  field: BackdropField, detail: BackdropDetail, minX: number, minZ: number, maxX: number, maxZ: number, slack = 0,
): BackdropScatterItem[] {
  const items: BackdropScatterItem[] = [];
  BACKDROP_SCATTER_CLASSES.forEach((entry, classIndex) => {
    const margin = SCATTER_REACH[entry.shape] * entry.radius_m[1];
    const g = entry.grid_m;
    for (let qz = Math.floor((minZ - margin) / g); qz <= Math.floor((maxZ + margin) / g); qz += 1) {
      for (let qx = Math.floor((minX - margin) / g); qx <= Math.floor((maxX + margin) / g); qx += 1) {
        const item = backdropScatterItem(field, detail, classIndex, qx, qz, slack);
        if (!item) continue;
        if (item.x + item.reach < minX || item.x - item.reach > maxX
          || item.z + item.reach < minZ || item.z - item.reach > maxZ) continue;
        items.push(item);
      }
    }
  });
  return items;
}

/** The world box the stored rings (ground and scatter) can occupy, widened by `pad_m` horizontally. */
export function backdropDetailWorldBounds(field: BackdropField, detail: BackdropDetail, pad_m = 0) {
  const [cx, cz] = detail.centre;
  const half = detail.outerHalf_m + pad_m;
  const [low, high] = backdropHeightBounds(field, cx - half, cz - half, cx + half, cz + half);
  const coarse = detail.cellSize0_m * 2 ** (detail.rings - 1);
  return {
    min: { x: cx - half, y: Math.min(low, detail.anchor_m) - 2 * coarse, z: cz - half },
    max: { x: cx + half, y: Math.max(high, detail.anchor_m) + BACKDROP_SCATTER_MAXIMUM_RISE_M + 2 * coarse, z: cz + half },
  };
}

export const BACKDROP_DETAIL_EMPTY = 0;
export const BACKDROP_DETAIL_LEAF = 1;
export const BACKDROP_DETAIL_SPLIT = 2;

export interface BackdropDetailClassifierOptions {
  field: BackdropField;
  detail: BackdropDetail;
  worldOrigin_m: readonly [number, number, number];
  /** Node edge per level, as the builder tabulates it. */
  nodeEdge_m: readonly (readonly number[])[];
  /** Level whose voxel is the scene cell h0: ring l's leaves sit at `solverLevel - l`. */
  solverLevel: number;
}

export interface BackdropDetailCensus {
  leaves: number;
  leavesPerRing: number[];
  classified: number;
}

/** The planner's supplemental classifier, plus a running census of what it admitted. */
export function createBackdropDetailClassifier(options: BackdropDetailClassifierOptions) {
  const { field, detail, worldOrigin_m, nodeEdge_m, solverLevel } = options;
  const [cx, cz] = detail.centre;
  const W = detail.outerHalf_m;
  const [fMinX, fMinZ, fMaxX, fMaxZ] = detail.footprint_m;
  const census: BackdropDetailCensus = { leaves: 0, leavesPerRing: Array(detail.rings).fill(0), classified: 0 };
  const edge0 = nodeEdge_m[solverLevel]![0]!;
  if (Math.abs(edge0 - 8 * detail.cellSize0_m) > 1e-9 * edge0) {
    throw new RangeError(`Backdrop detail ring 0 needs ${8 * detail.cellSize0_m} m bricks at the solver level; the tree has ${edge0} m`);
  }
  if (solverLevel - (detail.rings - 1) < 0) throw new RangeError("Backdrop detail rings reach above the octree root");
  const spans = new Map<string, readonly [number, number]>();
  const columnSpan = (minX: number, maxX: number, minZ: number, maxZ: number, cell: number): readonly [number, number] => {
    let lowest = Number.POSITIVE_INFINITY, highest = Number.NEGATIVE_INFINITY;
    const i0 = Math.floor((minX - cx) / cell + 1e-9) - 1, i1 = Math.ceil((maxX - cx) / cell - 1e-9);
    const j0 = Math.floor((minZ - cz) / cell + 1e-9) - 1, j1 = Math.ceil((maxZ - cz) / cell - 1e-9);
    for (let j = j0; j <= j1; j += 1) for (let i = i0; i <= i1; i += 1) {
      const x = cx + (i + 0.5) * cell, z = cz + (j + 0.5) * cell;
      if (insideFootprint(detail, x, z)) continue;
      if (Math.max(Math.abs(x - cx), Math.abs(z - cz)) >= W) continue;
      const top = detail.anchor_m + Math.floor((backdropFieldHeight(field, x, z) - detail.anchor_m) / cell + 0.5) * cell;
      lowest = Math.min(lowest, top); highest = Math.max(highest, top);
    }
    for (const item of backdropScatterItemsIn(field, detail, minX - cell, minZ - cell, maxX + cell, maxZ + cell, 1e-3)) {
      highest = Math.max(highest, item.top);
      lowest = Math.min(lowest, item.seat);
    }
    return [lowest, highest];
  };
  const classify = (level: number, coordinate: { x: number; y: number; z: number }): 0 | 1 | 2 => {
    census.classified += 1;
    const e = nodeEdge_m[level]![0]!;
    const minX = worldOrigin_m[0] + coordinate.x * e, maxX = minX + e;
    const minY = worldOrigin_m[1] + coordinate.y * e, maxY = minY + e;
    const minZ = worldOrigin_m[2] + coordinate.z * e, maxZ = minZ + e;
    if (maxX <= cx - W || minX >= cx + W || maxZ <= cz - W || minZ >= cz + W) return BACKDROP_DETAIL_EMPTY;
    if (minX >= fMinX && maxX <= fMaxX && minZ >= fMinZ && maxZ <= fMaxZ) return BACKDROP_DETAIL_EMPTY;
    const m = Math.max(Math.max(minX - cx, 0, cx - maxX), Math.max(minZ - cz, 0, cz - maxZ)) + 1e-7 * e;
    const ring = backdropDetailRing(detail, m);
    const target = solverLevel - ring;
    if (level < target) {
      const pad = e / 8;
      const [low, high] = backdropHeightBounds(field, minX - pad, minZ - pad, maxX + pad, maxZ + pad);
      const top = Math.max(high, detail.anchor_m) + pad + BACKDROP_SCATTER_MAXIMUM_RISE_M;
      const bottom = Math.min(low, detail.anchor_m) - pad;
      return top > minY && bottom < maxY ? BACKDROP_DETAIL_SPLIT : BACKDROP_DETAIL_EMPTY;
    }
    // A leaf-level node: exact column tops over the brick and a one-column
    // margin (a solid brick beside a lower column holds a riser face). The
    // span depends only on the node's xz footprint, so a column of nodes
    // shares one evaluation.
    const cell = e / 8;
    const spanKey = `${level}:${coordinate.x}:${coordinate.z}`;
    let span = spans.get(spanKey);
    if (!span) { span = columnSpan(minX, maxX, minZ, maxZ, cell); spans.set(spanKey, span); }
    const [lowest, highest] = span;
    if (!(lowest <= highest)) return BACKDROP_DETAIL_EMPTY;
    // Tops sit on the cell lattice, as do the brick's faces: a brick holds a
    // solid voxel when some top is above its floor, and one a ray can reach
    // when some top (a margin column's included) is at or below its ceiling —
    // so ground flush with a brick's ceiling is stored in that brick alone.
    if (highest > minY + 0.5 * cell && lowest < maxY + 0.5 * cell) {
      census.leaves += 1; census.leavesPerRing[ring]! += 1;
      return BACKDROP_DETAIL_LEAF;
    }
    return BACKDROP_DETAIL_EMPTY;
  };
  return { classify, census };
}

// ---------------------------------------------------------------------------
// WGSL: the voxelizer's side
// ---------------------------------------------------------------------------

/**
 * WGSL defining `sampleBackdropDetail(world, cellExtent) -> SolidWorldSample`
 * (the struct is the voxelizer's): ground and scatter at a voxel centre, full
 * or empty. The table header rides along as a constant array, so the field
 * evaluator is the renderer's own `backdropTerrainSurfaceAt`.
 */
export function backdropDetailVoxelizerWGSL(table: Uint32Array, detail: BackdropDetail): string {
  const header = Array.from(table.subarray(0, BACKDROP_TERRAIN_TABLE.tilesWord));
  const last = header.length - 1;
  const f = (value: number) => {
    const text = Math.fround(value).toPrecision(9);
    return text.includes(".") || text.includes("e") ? text : `${text}.0`;
  };
  const classes = BACKDROP_SCATTER_CLASSES.map((entry, index) => /* wgsl */ `
  if(!solid){solid=backdropScatterInside(p,${index + 1}u,${f(entry.grid_m)},${f(entry.density)},${f(entry.radius_m[0])},${f(entry.radius_m[1])},${entry.shape}u,${f(0.75 * 2 ** entry.minimumRing)});}`).join("");
  return /* wgsl */ `
const BACKDROP_DETAIL_TABLE=array<u32,${header.length}>(${header.map((word) => `${word}u`).join(",")});
${backdropTerrainWGSL({ load: (index) => `BACKDROP_DETAIL_TABLE[min(${index},${last}u)]`, tableBase: "0u" })}
const BACKDROP_DETAIL_OUTER:f32=${f(detail.outerHalf_m)};
const BACKDROP_DETAIL_RINGS:u32=${detail.rings}u;
const BACKDROP_DETAIL_SEED:u32=${detail.seed >>> 0}u;
fn backdropDetailHash(a:u32,b:u32,c:u32,d:u32,e:u32)->u32{
  var h=0x9e3779b9u^(a*0x85ebca6bu);
  var words=array<u32,4>(b,c,d,e);
  for(var index=0u;index<4u;index+=1u){
    h=(h^(h>>16u))*0x85ebca6bu;h=(h^(h>>13u))*0xc2b2ae35u;h^=(h>>16u)^(words[index]*0x27d4eb2fu);
  }
  h=(h^(h>>16u))*0x85ebca6bu;h=(h^(h>>13u))*0xc2b2ae35u;return h^(h>>16u);
}
fn backdropDetailRandom(salt:u32,q:vec2i,n:u32)->f32{return f32(backdropDetailHash(BACKDROP_DETAIL_SEED,salt,bitcast<u32>(q.x),bitcast<u32>(q.y),n))/4294967296.0;}
fn backdropDetailCell(p:vec2f)->f32{
  let q=p-backdropTerrainCentre();let m=max(abs(q.x),abs(q.y));
  let half0=backdropTableFloat(${BACKDROP_TERRAIN_TABLE.shapeWord + 2}u);
  return backdropTableFloat(${BACKDROP_TERRAIN_TABLE.shapeWord + 3}u)*exp2(f32(select(0u,min(u32(max(ceil(log2(m/half0)),0.0)),BACKDROP_DETAIL_RINGS-1u),m>half0)));
}
fn backdropDetailTop(p:vec2f,cell:f32)->f32{
  let centre=backdropTerrainCentre();let anchor=backdropTableFloat(${BACKDROP_TERRAIN_TABLE.latticeWord}u);
  let column=centre+(floor((p-centre)/cell)+vec2f(0.5))*cell;
  return anchor+floor((backdropTerrainSurfaceAt(column).height-anchor)/cell+0.5)*cell;
}
// Whether the voxel centre p lies in this class's item for its grid cell.
fn backdropScatterInside(p:vec3f,salt:u32,grid:f32,density:f32,rMin:f32,rMax:f32,shape:u32,minimumCells:f32)->bool{
  let q=vec2i(floor(p.xz/grid));
  let u1=backdropDetailRandom(salt,q,1u);
  let radius=rMin+(rMax-rMin)*u1*u1;
  let reach=${f(SCATTER_REACH[0])}*radius;
  let item=(vec2f(q)+(vec2f(reach)+(grid-2.0*reach)*vec2f(backdropDetailRandom(salt,q,2u),backdropDetailRandom(salt,q,3u)))/grid)*grid;
  let offset=p.xz-item;
  if(max(abs(offset.x),abs(offset.y))>reach){return false;}
  let rel=item-backdropTerrainCentre();let m=max(abs(rel.x),abs(rel.y));
  let fade=1.0-smoothstep(${f(FADE_BEGIN)}*BACKDROP_DETAIL_OUTER,${f(FADE_END)}*BACKDROP_DETAIL_OUTER,m);
  if(!(backdropDetailRandom(salt,q,0u)<density*fade)){return false;}
  if(m+reach>=BACKDROP_DETAIL_OUTER){return false;}
  if(backdropFootprintDistance(item)<=reach){return false;}
  let cell=backdropDetailCell(item);
  if(cell<minimumCells*backdropTableFloat(${BACKDROP_TERRAIN_TABLE.shapeWord + 3}u)){return false;}
  if(radius<${f(BACKDROP_SCATTER_MINIMUM_CELLS)}*cell){return false;}
  let seat=backdropDetailTop(item,cell);
  let angle=6.2831853*backdropDetailRandom(salt,q,4u);
  let axis=vec2f(cos(angle),sin(angle));
  if(shape==0u){
    let local=vec2f(dot(offset,axis),dot(offset,vec2f(-axis.y,axis.x)));
    let squash=backdropDetailRandom(salt,q,5u);
    let ry=radius*(0.45+0.25*squash);let rz=radius*(0.7+0.6*squash);
    let d=vec3f(local.x/radius,(p.y-(seat+0.2*ry))/ry,local.y/rz);
    return dot(d,d)<1.0;
  }
  let core=vec3f(offset.x,p.y-(seat+0.55*radius),offset.y);
  if(dot(core,core)<radius*radius){return true;}
  let lobe=0.65*radius;
  let second=vec2f(cos(angle+2.3),sin(angle+2.3));
  let a=vec3f(offset.x-0.6*radius*axis.x,p.y-(seat+0.4*radius),offset.y-0.6*radius*axis.y);
  let b=vec3f(offset.x-0.6*radius*second.x,p.y-(seat+0.4*radius),offset.y-0.6*radius*second.y);
  return dot(a,a)<lobe*lobe||dot(b,b)<lobe*lobe;
}
fn sampleBackdropDetail(world:vec3f,cellExtent:vec3f)->SolidWorldSample{
  let empty=SolidWorldSample(0.0,1e20,0u,vec3f(0.0));
  let rel=world.xz-backdropTerrainCentre();
  if(max(abs(rel.x),abs(rel.y))>=BACKDROP_DETAIL_OUTER||backdropInsideFootprint(world.xz)){return empty;}
  let cell=backdropDetailCell(world.xz);
  let column=backdropTerrainCentre()+(floor(rel/cell)+vec2f(0.5))*cell;
  if(backdropInsideFootprint(column)){return empty;}
  let top=backdropDetailTop(world.xz,cell);
  var solid=world.y<top;
  let p=world;${classes}
  if(!solid){return empty;}
  return SolidWorldSample(1.0,world.y-top,backdropTerrainMaterial(),vec3f(0.0,1.0,0.0));
}`;
}
