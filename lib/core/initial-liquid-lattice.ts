import { sourceInitialBounds } from '../cloth/source-initial';
import { proceduralSolidField } from "./procedural-solid";
import { initialHeightFieldHeight } from "./initial-height-field";
import { damBreakSignedDistanceAtNode, initialFluidBrickSignedDistanceAtNode,
  initialLiquidVolumesSignedDistance, baseInitialLiquidFractionAtCell, damBreakBoxContains, initialLiquidFractionAtCell, sceneDamBreakBox,
  sceneInitialLiquidVolumes, INITIAL_FLUID_BRICK_SIZE } from "./initial-fluid";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { sampleSolidWorld, solidWorldForScene } from "./solid-world";
import { quaternionInverseRotate } from "./rigid-body";
import { sceneShape } from "./scene-shape";
import type { InitialLiquidBox, SceneDescription } from "./model";
import { sceneFluidBoxScene, sceneFluidLatticeDimensions, sceneLatticeDimensions } from "./scene-lattice-dimensions";

/** Construction-only analytic samples, in metres, on shared lattice vertices.
 * coarseSolids: static solids run on cut 4h owners (uniformDetailCoarseSolids),
 * so a 4-aligned vertex is buried by its tiles' capacity, not its h cells'. */
export function initialLiquidVertexPhi(scene: SceneDescription, dimensions: readonly [number, number, number], sliceZ?: number, coarseSolids = false): Float32Array {
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const phi = new Float32Array((nx+1)*(ny+1)*(sliceZ === undefined ? nz+1 : 1));
  const closed = closedCells(scene, dimensions);
  const buried = buriedVertices(closed, dimensions, sliceZ, coarseSolids);
  const sample = vertexPhiSampler(scene, dimensions, boxUnionVertexPhi(scene, dimensions, (x, y, z) => closed[x+nx*(y+ny*z)] === 1, sliceZ));
  const empty = Math.max(c.width_m,c.height_m,c.depth_m);
  for (let outputZ=0; outputZ<(sliceZ === undefined ? nz+1 : 1); outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    // phi inside a solid is not state (uvBuried in uniform-volume.wgsl.ts,
    // umVertexBuried on the mixed lattice): a vertex with no open incident
    // owner is air by construction, whatever seeded body the analytic samples
    // would put there.
    const i = x+(nx+1)*(y+(ny+1)*outputZ);
    phi[i] = buried[i] ? empty : sample(x, y, sliceZ ?? outputZ, outputZ);
  }
  return phi;
}

/** initialLiquidVertexPhi at a vertex no solid buries: x, y, z its lattice
 * coordinates (z a slice's plane), outputZ its layer of the authored union's
 * (boxes, from boxUnionVertexPhi). */
function vertexPhiSampler(scene: SceneDescription, dimensions: readonly [number, number, number], boxes: ((x: number, y: number, z: number) => number) | undefined): (x: number, y: number, z: number, outputZ: number) => number {
  const [nx, ny, nz] = dimensions;
  const c = scene.container, empty = Math.max(c.width_m,c.height_m,c.depth_m), height = scene.fluid.initialHeightField;
  // Free space has no walls: the box bounds the initial liquid, so its faces
  // are free surfaces (the liquid is the fill intersected with the box), not
  // walls the fill continues past.
  const free = c.boundary === "free-space", half = [c.width_m/2, c.height_m/2, c.depth_m/2];
  const outsideBox = (p: {x: number; y: number; z: number}) => {
    const q = [Math.abs(p.x)-half[0]!, Math.abs(p.y-half[1]!)-half[1]!, Math.abs(p.z)-half[2]!];
    return Math.hypot(Math.max(q[0]!,0), Math.max(q[1]!,0), Math.max(q[2]!,0)) + Math.min(Math.max(q[0]!,q[1]!,q[2]!),0);
  };
  return (x, y, z, outputZ) => {
    const point = { x: (x/nx-0.5)*c.width_m, y: y/ny*c.height_m, z: (z/nz-0.5)*c.depth_m };
    let base = height ? point.y-initialHeightFieldHeight(height, point.x, point.z)
      : c.fillFraction > 0 ? damBreakSignedDistanceAtNode(scene,x,y,z,dimensions)
        ?? (point.y-c.fillFraction*c.height_m) : empty;
    const brick = initialFluidBrickSignedDistanceAtNode(scene,x,y,z,dimensions);
    if (brick !== undefined) base = scene.fluid.initialBrickSeedsAdditive ? Math.min(base,brick) : brick;
    const inside = boxes ? boxes(x, y, outputZ) : Infinity;
    const liquid = scene.systems?.fluid === false ? empty
      : Math.min(base,inside <= 0 ? inside : initialLiquidVolumesSignedDistance(scene,point) ?? empty);
    const value = free ? Math.max(liquid, outsideBox(point)) : liquid;
    return Number.isFinite(value) ? value : empty;
  };
}

/**
 * Vertices whose every in-range incident cell is closed by the solid world,
 * under the occupancy mask's rule (any solid fraction closes the cell). A
 * z slice reads the one cell layer the slice plane lies in.
 * coarseSolids: a 4-aligned vertex is a corner of 4h owners, which read their
 * plane at it wherever they have any open cell. It is buried only when every
 * in-range incident tile is closed throughout (umVertexBuried); an h tile
 * beside it (a Fine region, a body) never reads the value its own rule hides.
 */
function buriedVertices(closed: Uint8Array, dimensions: readonly [number, number, number], sliceZ?: number, coarseSolids = false): Uint8Array {
  const [nx, ny, nz] = dimensions;
  const layers = sliceZ === undefined ? nz+1 : 1;
  const buried = new Uint8Array((nx+1)*(ny+1)*layers);
  for (let outputZ=0; outputZ<layers; outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    // The incident cells: z in [outputZ-1, outputZ] (or the slice's one layer), y-1..y, x-1..x.
    const zLow = sliceZ === undefined ? outputZ-1 : Math.floor(sliceZ), zHigh = sliceZ === undefined ? outputZ : zLow;
    let allClosed = true, any = false;
    for (let z=zLow; z<=zHigh; z++) for (let cy=y-1; cy<=y; cy++) for (let cx=x-1; cx<=x; cx++) {
      if (cx<0 || cy<0 || z<0 || cx>=nx || cy>=ny || z>=nz) continue;
      any = true;
      if (!closed[cx+nx*(cy+ny*z)]) { allClosed = false; }
    }
    buried[x+(nx+1)*(y+(ny+1)*outputZ)] = any && allClosed ? 1 : 0;
  }
  if (coarseSolids) {
    if (sliceZ !== undefined || nx%4 || ny%4 || nz%4) throw new Error("Cut 4h solid owners need the whole 4h-tiled lattice");
    const tx = nx/4, ty = ny/4, tz = nz/4, open = new Uint8Array(tx*ty*tz);
    for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++)
      if (!closed[x+nx*(y+ny*z)]) open[(x>>2)+tx*((y>>2)+ty*(z>>2))] = 1;
    for (let z=0; z<=tz; z++) for (let y=0; y<=ty; y++) for (let x=0; x<=tx; x++) {
      let any = false;
      for (let cz=z-1; cz<=z; cz++) for (let cy=y-1; cy<=y; cy++) for (let cx=x-1; cx<=x; cx++)
        if (cx>=0 && cy>=0 && cz>=0 && cx<tx && cy<ty && cz<tz && open[cx+tx*(cy+ty*cz)]) any = true;
      if (any) buried[4*x+(nx+1)*(4*y+(ny+1)*4*z)] = 0;
    }
  }
  return buried;
}

/** The rigid bodies' solid in a cell at their authored pose, sampled as the
 * fluid's body solid samples it (umCellBodyFraction): eight points at 0.4 of
 * the cell either side of its centre, a closed interior boundary solid outside
 * its box. 0 where no point is solid, 2 where every one is, 1 between;
 * undefined without bodies. */
function bodyCellSolid(scene: SceneDescription, dimensions: readonly [number, number, number]): ((x: number, y: number, z: number) => number) | undefined {
  const bodies = scene.rigidBodies ?? [];
  if (bodies.length === 0) return undefined;
  const [nx, ny, nz] = dimensions, c = scene.container;
  const h = [c.width_m/nx, c.height_m/ny, c.depth_m/nz], reach = 0.4*Math.hypot(h[0]!, h[1]!, h[2]!);
  const shapes = bodies.map(body => { const shape = sceneShape(body.shape);
    return { body, shape, interior: body.fluidBoundary === "interior", radius: shape.boundingRadius_m(body.dimensions_m) }; });
  return (x, y, z) => {
    const centre = { x: ((x+0.5)/nx-0.5)*c.width_m, y: (y+0.5)/ny*c.height_m, z: ((z+0.5)/nz-0.5)*c.depth_m };
    let mask = 0;
    for (const entry of shapes) {
      const { body, shape, interior } = entry, p = body.position_m;
      if (!interior && Math.hypot(centre.x-p.x, centre.y-p.y, centre.z-p.z) > entry.radius+reach) continue;
      for (let corner=0; corner<8; corner++) {
        const bit = 1 << corner;
        if (mask & bit) continue;
        const local = quaternionInverseRotate(body.orientation, {
          x: centre.x+(corner&1 ? 0.4 : -0.4)*h[0]!-p.x, y: centre.y+(corner&2 ? 0.4 : -0.4)*h[1]!-p.y, z: centre.z+(corner&4 ? 0.4 : -0.4)*h[2]!-p.z });
        if (shape.inside(body.dimensions_m, local) !== interior) mask |= bit;
      }
      if (mask === 255) return 2;
    }
    return mask ? 1 : 0;
  };
}

/** Authored voxels close whole cells; native terrain closes only its full cells. */
function closedCells(scene: SceneDescription, dimensions: readonly [number, number, number]): Uint8Array {
  const [nx, ny, nz] = dimensions, at = closedCell(scene, dimensions);
  const closed = new Uint8Array(nx*ny*nz);
  for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++) closed[x+nx*(y+ny*z)] = at(x, y, z) ? 1 : 0;
  return closed;
}
function closedCell(scene: SceneDescription, dimensions: readonly [number, number, number]): (x: number, y: number, z: number) => boolean {
  const [nx, ny, nz] = dimensions;
  const world = solidWorldForScene(scene);
  const boundary = proceduralSolidField(scene);
  // A scene with no terrain has every column at zero, and holds no column array.
  const terrain = sceneHasTerrain(scene) ? terrainColumnHeights(scene, nx, nz) : undefined, hy = scene.container.height_m / ny;
  const cell: [number, number, number] = [0, 0, 0];
  return (x, y, z) => (cell[0] = x, cell[1] = y, cell[2] = z,
    boundary?.open(cell) ?? Number(sampleSolidWorld(world,cell,true).solidFraction === 0)) <= 1e-5
    || (terrain !== undefined && Math.fround(terrain[x+nx*z]! / hy) >= y + 1);
}

/**
 * Authored boxes as one body: at each vertex inside their union, minus the
 * distance in metres to the union's nearest free face; +Infinity outside,
 * where the minimum over the boxes' own distances is already exact. Inside it
 * is not: that minimum reads every seam between two boxes and every wall
 * contact as a surface, so a body tiled from boxes is a few cells deep at
 * most, whatever its depth. Only h vertices are ever redistanced; a 4h corner
 * keeps the magnitude it starts with and its chord puts the surface wherever
 * the nearest seam left it.
 * A face is free where the cell beyond it holds no solid: closed, or a rigid
 * body's at its authored pose (bodyCellSolid). One against a solid or past
 * the domain is contact, and the liquid continues through it as in
 * damBreakSignedDistanceAtNode. Each axis takes the distance along its lattice
 * lines to their free ends and spreads it across the other two; a z slice has
 * no z faces. The value at a vertex (x, y, z: a slice's layer), or undefined
 * where no box is authored. windowed: the transform runs over the vertices of
 * the boxes' lattice ranges alone, in lattice coordinates, which is the whole
 * lattice's at every vertex of the union. Every site and every free end lies
 * there: a line outside it holds no span, so each pass's lower envelope at a
 * vertex inside is made from the same sites, in the same order, by the same
 * arithmetic.
 */
function boxUnionVertexPhi(scene: SceneDescription, dimensions: readonly [number, number, number], closed: (x: number, y: number, z: number) => boolean, sliceZ?: number, windowed = false): ((x: number, y: number, z: number) => number) | undefined {
  const volumes = sceneInitialLiquidVolumes(scene);
  const authored = volumes.filter((volume): volume is InitialLiquidBox => volume.shape === "box");
  if (authored.length === 0 || authored.length !== volumes.length) return undefined;
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const cells = [nx, ny, nz];
  const spacing = [c.width_m/nx, c.height_m/ny, c.depth_m/nz];
  // Lattice coordinates. A face authored on a cell face must land on it, and
  // one past the domain only needs to stay past it.
  const lattice = (value: number, axis: number) => {
    const rounded = Math.round(value);
    return Math.max(-1, Math.min(cells[axis]!+1, Math.abs(value-rounded) < 1e-6 ? rounded : value));
  };
  const boxes = authored.map(box => ({
    lo: [lattice((box.min_m.x/c.width_m+0.5)*nx, 0), lattice(box.min_m.y/c.height_m*ny, 1), lattice((box.min_m.z/c.depth_m+0.5)*nz, 2)],
    hi: [lattice((box.max_m.x/c.width_m+0.5)*nx, 0), lattice(box.max_m.y/c.height_m*ny, 1), lattice((box.max_m.z/c.depth_m+0.5)*nz, 2)],
  }));
  const sliced = (axis: number) => axis === 2 && sliceZ !== undefined;
  // Contact is any solid in the cell: the closed cells, and the rigid bodies
  // at their authored pose as the fluid's body solid samples them.
  const bodies = bodyCellSolid(scene, dimensions);
  const contact = bodies ? (x: number, y: number, z: number) => closed(x, y, z) || bodies(x, y, z) > 0 : closed;
  const solid = bodies ? (x: number, y: number, z: number) => closed(x, y, z) || bodies(x, y, z) === 2 : closed;
  // A face inside a cut cell, with solid throughout the cells beyond it, moves
  // out to the cut cell's far vertex plane. The liquid then continues through
  // the contact into the solid (Batty, Bertails and Bridson 2007; Ng, Min and
  // Gibou 2009): the cut cell holds its open part filled, as the liquid
  // authored up to the wall does, where a level set stopping at the face
  // fills that open part only in proportion. Only vertices of cells holding
  // solid change.
  const cut = [0, 0, 0], beyond = [0, 0, 0];
  // A box's faces are its own, whatever slice of it is asked for.
  for (const box of boxes) for (let a=0; a<3; a++) {
    const b = (a+1)%3, d = (a+2)%3;
    const span = (axis: number, lo: number, hi: number) => [Math.max(0, Math.floor(lo)), Math.min(cells[axis]!, Math.ceil(hi))-1];
    const [b0, b1] = span(b, box.lo[b]!, box.hi[b]!), [d0, d1] = span(d, box.lo[d]!, box.hi[d]!);
    for (const side of [0, 1]) {
      const face = side ? box.hi[a]! : box.lo[a]!, inner = Math.floor(face), outer = side ? inner+1 : inner-1;
      if (Number.isInteger(face) || inner < 0 || inner >= cells[a]! || outer < 0 || outer >= cells[a]! || b0 > b1 || d0 > d1) continue;
      let held = true;
      for (let id=d0; id<=d1 && held; id++) for (let ib=b0; ib<=b1 && held; ib++) {
        cut[a] = inner; cut[b] = ib; cut[d] = id; beyond[a] = outer; beyond[b] = ib; beyond[d] = id;
        held = contact(cut[0]!, cut[1]!, cut[2]!) && solid(beyond[0]!, beyond[1]!, beyond[2]!);
      }
      if (held) { if (side) box.hi[a] = inner+1; else box.lo[a] = inner; }
    }
  }
  const first = (axis: number, lo: number) => sliced(axis) ? Number(lo > sliceZ!) : Math.max(0, Math.ceil(lo));
  const last = (axis: number, hi: number) => sliced(axis) ? Number(hi >= sliceZ!)-1 : Math.min(cells[axis]!, Math.floor(hi));
  // The vertices the transform runs over, low and size an axis.
  const low = [0, 1, 2].map(a => sliced(a) || !windowed ? 0 : Math.min(...boxes.map(box => first(a, box.lo[a]!))));
  const size = [0, 1, 2].map(a => sliced(a) ? 1 : !windowed ? cells[a]!+1 : Math.max(...boxes.map(box => last(a, box.hi[a]!)))-low[a]!+1);
  if (size.some(n => n <= 0)) return undefined;
  const stride = [1, size[0]!, size[0]!*size[1]!];
  const cell = [0, 0, 0];
  const free = (a: number, beyond: number, b: number, ib: number, d: number, id: number) => {
    if (beyond < 0 || beyond >= cells[a]!) return false;
    for (const cd of sliced(d) ? [Math.floor(sliceZ!)] : [id-1, id]) for (const cb of sliced(b) ? [Math.floor(sliceZ!)] : [ib-1, ib]) {
      if (cb<0 || cd<0 || cb>=cells[b]! || cd>=cells[d]!) continue;
      cell[a] = beyond; cell[b] = cb; cell[d] = cd;
      if (!contact(cell[0]!, cell[1]!, cell[2]!)) return true;
    }
    return false;
  };
  const total = size[0]!*size[1]!*size[2]!;
  const nearest = new Float64Array(total).fill(Infinity), along = new Float64Array(total);
  const inside = new Uint8Array(total);
  for (const a of sliceZ === undefined ? [0, 1, 2] : [0, 1]) {
    const b = (a+1)%3, d = (a+2)%3;
    const lines: [number, number][][] = Array.from({ length: size[b]!*size[d]! }, () => []);
    for (const box of boxes) for (let id=first(d,box.lo[d]!); id<=last(d,box.hi[d]!); id++) for (let ib=first(b,box.lo[b]!); ib<=last(b,box.hi[b]!); ib++)
      lines[ib-low[b]!+size[b]!*(id-low[d]!)]!.push([box.lo[a]!, box.hi[a]!]);
    along.fill(Infinity);
    for (let jd=0; jd<size[d]!; jd++) for (let jb=0; jb<size[b]!; jb++) {
      const id = jd+low[d]!, ib = jb+low[b]!, spans = lines[jb+size[b]!*jd]!.sort((p, q) => p[0]-q[0]);
      for (let s=0; s<spans.length;) {
        const lo = spans[s]![0];
        let hi = spans[s]![1];
        for (s++; s<spans.length && spans[s]![0] <= hi; s++) hi = Math.max(hi, spans[s]![1]);
        const below = free(a, Math.ceil(lo)-1, b, ib, d, id), above = free(a, Math.floor(hi), b, ib, d, id);
        for (let k=Math.max(0, Math.ceil(lo)); k<=Math.min(cells[a]!, Math.floor(hi)); k++) {
          const reach = Math.min(below ? k-lo : Infinity, above ? hi-k : Infinity)*spacing[a]!;
          const i = (k-low[a]!)*stride[a]!+jb*stride[b]!+jd*stride[d]!;
          inside[i] = 1; along[i] = reach*reach;
        }
      }
    }
    spreadSquaredDistance(along, size, stride, low, b, spacing[b]!);
    spreadSquaredDistance(along, size, stride, low, d, spacing[d]!);
    for (let i=0; i<total; i++) nearest[i] = Math.min(nearest[i]!, along[i]!);
  }
  const empty = Math.max(c.width_m, c.height_m, c.depth_m);
  const phi = new Float32Array(total).fill(Infinity);
  for (let i=0; i<total; i++) if (inside[i]) phi[i] = Number.isFinite(nearest[i]!) ? -Math.sqrt(nearest[i]!) : -empty;
  const [lx, ly, lz] = low as [number, number, number], [sx, sy, sz] = size as [number, number, number];
  return (x, y, z) => x < lx || y < ly || z < lz || x >= lx+sx || y >= ly+sy || z >= lz+sz ? Infinity : phi[x-lx+sx*(y-ly+sy*(z-lz))]!;
}

/**
 * One axis of Felzenszwalb and Huttenlocher's squared distance transform, in
 * place: field(q) <- min over p on q's line of field(p) + (spacing (q-p))^2,
 * q and p in lattice coordinates (the field's own plus low).
 */
function spreadSquaredDistance(field: Float64Array, size: readonly number[], stride: readonly number[], low: readonly number[], axis: number, spacing: number): void {
  const n = size[axis]!, o = low[axis]!, b = (axis+1)%3, d = (axis+2)%3, step = stride[axis]!, scale = spacing*spacing;
  const line = new Float64Array(n), site = new Int32Array(n), from = new Float64Array(n);
  for (let id=0; id<size[d]!; id++) for (let ib=0; ib<size[b]!; ib++) {
    const start = ib*stride[b]!+id*stride[d]!;
    let top = -1;
    for (let q=0; q<n; q++) {
      const value = line[q] = field[start+q*step]!;
      if (value === Infinity) continue;
      let begin = -Infinity;
      for (const u = q+o; top >= 0; top--) {
        const p = site[top]!, v = p+o;
        begin = (value-line[p]!+scale*(u*u-v*v))/(2*scale*(u-v));
        if (begin > from[top]!) break;
      }
      if (top < 0) begin = -Infinity;
      site[++top] = q; from[top] = begin;
    }
    if (top < 0) continue;
    for (let q=0, k=0; q<n; q++) {
      while (k < top && from[k+1]! < q+o) k++;
      const p = site[k]!;
      field[start+q*step] = line[p]!+scale*(q-p)*(q-p);
    }
  }
}

/** Shared initial conservative field; keeps the density comparison lane's seed rule. */
export function initialLiquidCellVolumes(scene: SceneDescription, dimensions: readonly [number, number, number], geometric = true, sliceZ?: number) {
  const [nx,ny,nz]=dimensions;
  const volume=new Float32Array(nx*ny*(sliceZ === undefined ? nz : 1));
  const {density:at,terrain=new Float32Array(nx*nz),dam}=cellVolumeSampler(scene,dimensions,geometric);
  let initial=0;
  const wetMinimum=[nx,ny,nz];
  const wetMaximum=[0,0,0];
  for(let outputZ=0;outputZ<(sliceZ === undefined ? nz : 1);outputZ++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++) {
    const z=sliceZ ?? outputZ;
    const density=at(x,y,z);
    volume[x+nx*(y+ny*outputZ)]=density;
    initial+=density;
    if(density>1e-5){
      wetMinimum[0]=Math.min(wetMinimum[0]!,x);wetMinimum[1]=Math.min(wetMinimum[1]!,y);wetMinimum[2]=Math.min(wetMinimum[2]!,z);
      wetMaximum[0]=Math.max(wetMaximum[0]!,x+1);wetMaximum[1]=Math.max(wetMaximum[1]!,y+1);wetMaximum[2]=Math.max(wetMaximum[2]!,z+1);
    }
  }
  return {volume,terrain,initial,wetMinimum,wetMaximum,dam};
}

/** initialLiquidCellVolumes at one cell. */
function cellVolumeSampler(scene: SceneDescription, dimensions: readonly [number, number, number], geometric: boolean) {
  const [nx,ny,nz]=dimensions;
  const c=scene.container;
  // The column heights, or undefined with no terrain (every column at zero): no array a box column.
  const terrain=sceneHasTerrain(scene)?terrainColumnHeights(scene,nx,nz):undefined;
  const cellHeight=c.height_m/ny;
  const solidWorld=solidWorldForScene(scene);
  const boundary=geometric?proceduralSolidField(scene):undefined;
  const dam=sceneDamBreakBox(scene);
  const density=(x: number, y: number, z: number): number => {
    const ground=terrain?terrain[x+nx*z]!:0,aboveGround=(y+0.5)*cellHeight>ground;
    const solidFraction=sampleSolidWorld(solidWorld,[x,y,z],geometric).solidFraction;
    // Authored voxels retain whole-cell occupancy. Terrain is represented
    // separately by the same fractional column capacity the GPU reads.
    const solidOpen=boundary?.open([x,y,z]) ?? (geometric ? Number(solidFraction===0) : 1-solidFraction);
    const base=geometric?baseInitialLiquidFractionAtCell(scene,x,y,z,dimensions):scene.fluid.initialCondition==="dam-break"
      ?damBreakBoxContains(dam,(x+0.5)/nx,(y+0.5)/ny,(z+0.5)/nz):(y+0.5)/ny<=c.fillFraction;
    const liquidFraction=geometric||aboveGround?initialLiquidFractionAtCell(scene,x,y,z,dimensions,base):0;
    // Match the uploaded f32 terrain height in cell units. A sliver can round
    // to zero in SolidWorld's u8 mask while still reducing GPU cell capacity.
    // This matters more as a finely resolved pond exposes more shore cells.
    const terrainOpen=geometric ? Math.fround(1-Math.max(0,Math.min(1,
      Math.fround(ground/cellHeight)-y))) : 1;
    return Math.fround(Math.min(solidOpen,terrainOpen,liquidFraction));
  };
  return {density,terrain,dam};
}

/**
 * Boxes of cells, [low, high), outside which no cell holds initial liquid
 * (initialLiquidFractionAtCell is zero), or undefined: anywhere. Each
 * source's own: the base fill's slab or reservoir, a height field's columns
 * under its highest point, the seeded bricks, and each analytic volume's
 * bounding box, which its eight samples a cell lie in. A cell of margin
 * covers the rounding of each face.
 */
export function initialLiquidCellBounds(scene: SceneDescription, dimensions: readonly [number, number, number]): [number[], number[]][] | undefined {
  const extents = initialLiquidCellExtents(scene, dimensions);
  if (!extents) return undefined;
  const out: [number[], number[]][] = [];
  for (const [low, high] of extents) {
    const lo = low.map(v => Math.max(0, Math.floor(v)-1)), hi = high.map((v, a) => Math.min(dimensions[a]!, Math.floor(v)+2));
    if (lo.every((v, a) => v < hi[a]!)) out.push([lo, hi]);
  }
  return out.every(([lo, hi]) => lo.concat(hi).every(Number.isFinite)) ? out : undefined;
}
/** Far air at least this many finest cells from liquid: the AIR value a page
 * the census claims takes (UniformPageInitialise's near, 16 hmax). */
export const FREE_SPACE_SEED_MARGIN_CELLS = 16;
/**
 * A free-space scene's seed stores the box's pages alone (LatticePages, the
 * uploads and the host censuses address box pages), and every page beyond
 * the box is far air, at least FREE_SPACE_SEED_MARGIN_CELLS h from liquid,
 * until the first census claims it. Initial liquid nearer a box face than
 * that breaks the certificate: the claimed pages hold far air where the
 * liquid's distance is smaller, the band measure's open-axis slopes read
 * them, the liquid measures about a tenth empty and volume control drives
 * a divergence. Such a scene is refused here, with the face and distance.
 * dimensions: the container's cells (initialLiquidCellBounds' frame);
 * lattice: the window's, which adds cells past the container's high faces.
 */
export function assertFreeSpaceSeedMargin(scene: SceneDescription, dimensions: readonly [number, number, number], lattice: readonly [number, number, number] = dimensions): void {
  if (scene.container.boundary !== "free-space") return;
  const c = scene.container, size = [c.width_m, c.height_m, c.depth_m], h = size.map((v, a) => v/dimensions[a]!);
  const margin_m = FREE_SPACE_SEED_MARGIN_CELLS*Math.max(...h), extents = initialLiquidCellExtents(scene, dimensions);
  if (!extents) throw new Error(`Free-space scene ${scene.sceneId ?? "(unnamed)"}: the initial liquid's extent is not finite, so its margin to the seed window cannot be certified`);
  for (const [low, high] of extents) {
    if (!low.every((v, a) => v < high[a]!)) continue;
    for (let a = 0; a < 3; a++) for (const [gap, face] of [[low[a]!*h[a]!, "low"], [(lattice[a]!-high[a]!)*h[a]!, "high"]] as const) {
      // Compared in micro-cells: metres-to-cells rounding must not refuse a
      // window authored at exactly the margin.
      if (Math.round(gap/margin_m*FREE_SPACE_SEED_MARGIN_CELLS*1e6) >= FREE_SPACE_SEED_MARGIN_CELLS*1e6) continue;
      throw new Error(`Free-space scene ${scene.sceneId ?? "(unnamed)"}: initial liquid lies ${gap.toFixed(4)} m from the seed window's ${face} ${"xyz"[a]} face, inside the ${margin_m.toFixed(4)} m (${FREE_SPACE_SEED_MARGIN_CELLS} h) far-air margin. `
        + `The seed stores the window's pages alone, and the pages beyond it are far air (at least ${FREE_SPACE_SEED_MARGIN_CELLS} h) until the first census claims them, so liquid nearer the window measures part-empty and loses volume. Enlarge the container (the seed window) or move the liquid inward.`);
    }
  }
}
/** initialLiquidCellBounds' sources before rounding: [low, high) in lattice
 * cells, unclamped (a box past the lattice keeps its extent). */
function initialLiquidCellExtents(scene: SceneDescription, dimensions: readonly [number, number, number]): [number[], number[]][] | undefined {
  const [nx, ny, nz] = dimensions, c = scene.container, out: [number[], number[]][] = [];
  const cell = (axis: number, metres: number) => [(metres/c.width_m+0.5)*nx, metres/c.height_m*ny, (metres/c.depth_m+0.5)*nz][axis]!;
  const add = (low: number[], high: number[]) => { out.push([low, high]); };
  const metres = (min: { x: number; y: number; z: number }, max: { x: number; y: number; z: number }) =>
    add([cell(0, min.x), cell(1, min.y), cell(2, min.z)], [cell(0, max.x), cell(1, max.y), cell(2, max.z)]);
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds || scene.fluid.initialBrickSeedsAdditive) {
    const field = scene.fluid.initialHeightField;
    if (field) {
      if (![field.baseHeight_m, ...(field.kind === "cosine" ? [field.amplitude_m] : [field.curvatureX_mInv, field.curvatureZ_mInv, field.center_m.x, field.center_m.z])].every(Number.isFinite)) return undefined;
      const reach = (centre: number, half: number) => Math.max(Math.abs(-half-centre), Math.abs(half-centre))**2;
      const top = field.kind === "cosine" ? field.baseHeight_m+Math.abs(field.amplitude_m)
        : field.baseHeight_m+Math.max(0, field.curvatureX_mInv)*reach(field.center_m.x, c.width_m/2)+Math.max(0, field.curvatureZ_mInv)*reach(field.center_m.z, c.depth_m/2);
      add([0, 0, 0], [nx, cell(1, top), nz]);
    } else if (scene.systems?.fluid !== false) {
      if (scene.fluid.initialCondition === "tank-fill") add([0, 0, 0], [nx, c.fillFraction*ny, nz]);
      else { const dam = sceneDamBreakBox(scene); add([dam.min.x*nx, dam.min.y*ny, dam.min.z*nz], [dam.max.x*nx, dam.max.y*ny, dam.max.z*nz]); }
    }
  }
  for (const seed of seeds ?? []) {
    const at = [cell(0, seed.x), cell(1, seed.y), cell(2, seed.z)].map((v, a) => Math.floor(Math.min(dimensions[a]!-1, Math.max(0, Math.floor(v)))/INITIAL_FLUID_BRICK_SIZE)*INITIAL_FLUID_BRICK_SIZE);
    add(at, at.map(v => v+INITIAL_FLUID_BRICK_SIZE));
  }
  for (const v of sceneInitialLiquidVolumes(scene)) {
    if (v.shape === "box") { metres(v.min_m, v.max_m); continue; }
    const r = v.shape === "torus" ? v.radius_m+v.tubeRadius_m : v.radius_m, o = v.center_m;
    const half = v.shape === "torus" ? { x: r, y: v.tubeRadius_m, z: r } : v.shape === "cylinder" ? { x: r, y: r, z: v.halfHeight_m } : { x: r, y: r, z: r };
    metres({ x: o.x-half.x, y: o.y-half.y, z: o.z-half.z }, { x: o.x+half.x, y: o.y+half.y, z: o.z+half.z });
  }
  for (const b of sourceInitialBounds(scene)) metres({x:b.min[0],y:b.min[1],z:b.min[2]}, {x:b.max[0],y:b.max[1],z:b.max[2]});
  return out;
}

/**
 * A lattice field page by page, sparse: the page at (x, y, z) of `pages` is
 * at.get(x + pages[0] (y + pages[1] z)), edge^3 cells x fastest (a vertex
 * field's (edge + 1)^3 vertices from the page's low corner), absent where it
 * holds zero throughout. Its keys ascend, so a walk of `at` is the lattice's
 * page order, and its size follows the pages held, never the box.
 */
export interface LatticePages {
  readonly edge: number;
  readonly cells: readonly [number, number, number];
  readonly pages: readonly [number, number, number];
  readonly at: ReadonlyMap<number, Float32Array>;
}
/** The page keys (x + pages[0] (y + pages[1] z)) a set of cell boxes [lo, hi) of `edge`-cell pages meets, ascending, each once. */
export function latticeBoxPageKeys(boxes: readonly (readonly [readonly number[], readonly number[]])[], edge: number, pages: readonly [number, number, number]): number[] {
  const [px, py, pz] = pages, keys = new Set<number>();
  for (const [lo, hi] of boxes) {
    const a = lo.map(v => Math.max(0, Math.floor(v/edge))), b = hi.map((v, k) => Math.min(pages[k]!-1, Math.ceil(v/edge)-1));
    for (let z=a[2]!; z<=b[2]! && z<pz; z++) for (let y=a[1]!; y<=b[1]! && y<py; y++) for (let x=a[0]!; x<=b[0]! && x<px; x++) keys.add(x+px*(y+py*z));
  }
  return [...keys].sort((p, q) => p-q);
}

/**
 * Vertex phi far from liquid on the scene's fluid lattice: its largest extent.
 * For a lattice that is its container this is the value initialLiquidVertexPhi
 * gives a vertex with no liquid near.
 */
export function fluidLatticeFarPhi(scene: SceneDescription): number {
  if (scene.container.walls === "none") scene = sceneFluidBoxScene(scene);
  const c = scene.container, [cx, cy, cz] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  return Math.max(c.width_m*(nx/cx), c.height_m*(ny/cy), c.depth_m*(nz/cz));
}

/**
 * initialLiquidVertexPhi on the scene's fluid lattice
 * (sceneFluidLatticeDimensions): the container's own samples where it stands.
 * Past a closed wall every incident cell is solid, so a vertex there is far
 * air (fluidLatticeFarPhi) by the buried rule; above an open top the top
 * plane's distance grows by the height above it.
 */
export function fluidLatticeVertexPhi(scene: SceneDescription, coarseSolids = false): Float32Array {
  // An open world is built over its box alone (sceneFluidBoxScene), which is whole pages.
  if (scene.container.walls === "none") scene = sceneFluidBoxScene(scene);
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER), [cx, cy, cz] = cells;
  const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const inner = initialLiquidVertexPhi(scene, cells, undefined, coarseSolids);
  if (nx === cx && ny === cy && nz === cz) return inner;
  const c = scene.container, cellHeight = c.height_m/cy, rows = c.top === "open" ? ny : cy;
  const phi = new Float32Array((nx+1)*(ny+1)*(nz+1)).fill(fluidLatticeFarPhi(scene));
  for (let z=0; z<=cz; z++) for (let y=0; y<=rows; y++) for (let x=0; x<=cx; x++)
    phi[x+(nx+1)*(y+(ny+1)*z)] = inner[x+(cx+1)*(Math.min(y,cy)+(cy+1)*z)]! + Math.max(0,y-cy)*cellHeight;
  return phi;
}

/**
 * initialLiquidCellVolumes on the scene's fluid lattice: the container's
 * liquid where it stands and none in the padding. A terrain column past the
 * container repeats the container's nearest, which is what a read clamped to
 * the container's columns returns.
 */
export function fluidLatticeCellVolumes(scene: SceneDescription): ReturnType<typeof initialLiquidCellVolumes> {
  if (scene.container.walls === "none") scene = sceneFluidBoxScene(scene);
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER), [cx, cy, cz] = cells;
  const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const inner = initialLiquidCellVolumes(scene, cells, true);
  if (nx === cx && ny === cy && nz === cz) return inner;
  const volume = new Float32Array(nx*ny*nz), terrain = new Float32Array(nx*nz);
  for (let z=0; z<cz; z++) for (let y=0; y<cy; y++)
    volume.set(inner.volume.subarray(cx*(y+cy*z), cx*(y+cy*z)+cx), nx*(y+ny*z));
  for (let z=0; z<nz; z++) for (let x=0; x<nx; x++) terrain[x+nx*z] = inner.terrain[Math.min(x,cx-1)+cx*Math.min(z,cz-1)]!;
  return {...inner, volume, terrain};
}

/**
 * fluidLatticeCellVolumes page by page (LatticePages, `edge` cells a page):
 * the same values, and the same sum, added in the lattice's order. Only the
 * pages initialLiquidCellBounds reaches are sampled; every other cell is
 * zero there, and adding zero leaves the sum as it is.
 */
export function fluidLatticePagedCellVolumes(scene: SceneDescription, edge: number) {
  if (scene.container.walls === "none") scene = sceneFluidBoxScene(scene);
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER), [cx, cy, cz] = cells;
  const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const {density, terrain: inner, dam} = cellVolumeSampler(scene, cells, true);
  const pages: [number, number, number] = [Math.ceil(nx/edge), Math.ceil(ny/edge), Math.ceil(nz/edge)], [px, py] = pages;
  const bounds = initialLiquidCellBounds(scene, cells) ?? [[[0, 0, 0], [cx, cy, cz]]];
  const at = new Map<number, Float32Array>();
  const wetMinimum = [cx, cy, cz], wetMaximum = [0, 0, 0];
  // The pages the bounds meet, in the lattice's order: work follows the liquid, not the box.
  for (const key of latticeBoxPageKeys(bounds, edge, pages)) {
    const x = key%px, y = Math.floor(key/px)%py, z = Math.floor(key/(px*py));
    const o = [x*edge, y*edge, z*edge], e = [Math.min(cx, o[0]!+edge), Math.min(cy, o[1]!+edge), Math.min(cz, o[2]!+edge)];
    if (!bounds.some(([lo, hi]) => [0, 1, 2].every(a => lo[a]! < e[a]! && o[a]! < hi[a]!))) continue;
    const page = new Float32Array(edge**3);
    let any = false;
    for (let k=o[2]!; k<e[2]!; k++) for (let j=o[1]!; j<e[1]!; j++) for (let i=o[0]!; i<e[0]!; i++) {
      const v = density(i, j, k);
      if (Object.is(v, 0)) continue;
      page[i-o[0]!+edge*(j-o[1]!+edge*(k-o[2]!))] = v; any = true;
      if (v > 1e-5) {
        wetMinimum[0] = Math.min(wetMinimum[0]!, i); wetMinimum[1] = Math.min(wetMinimum[1]!, j); wetMinimum[2] = Math.min(wetMinimum[2]!, k);
        wetMaximum[0] = Math.max(wetMaximum[0]!, i+1); wetMaximum[1] = Math.max(wetMaximum[1]!, j+1); wetMaximum[2] = Math.max(wetMaximum[2]!, k+1);
      }
    }
    if (any) at.set(key, page);
  }
  // The sum in the lattice's order (z, then y, then x), over the rows of the pages held:
  // the pages of a slab (qy, qz) in x order, the slabs in (qz, qy) order.
  const slabs = new Map<number, number[]>();
  for (const key of at.keys()) { const qx = key%px, slab = (key-qx)/px; let row = slabs.get(slab); if (!row) slabs.set(slab, row = []); row.push(qx); }
  const layers = new Map<number, [number, number[]][]>();
  for (const [slab, row] of slabs) { const qz = Math.floor(slab/py); let held = layers.get(qz); if (!held) layers.set(qz, held = []); held.push([slab%py, row]); }
  let initial = 0;
  for (const [qz, held] of layers) for (let z=qz*edge; z<Math.min(cz, (qz+1)*edge); z++) for (const [qy, row] of held) for (let y=qy*edge; y<Math.min(cy, (qy+1)*edge); y++) for (const x of row) {
    const page = at.get(x+px*(qy+py*qz))!, line = edge*(y%edge+edge*(z%edge));
    for (let i=0; i<Math.min(edge, cx-x*edge); i++) initial += page[line+i]!;
  }
  // The lattice's column heights where the scene has terrain; undefined, every column at zero, where it has none.
  let terrain: Float32Array | undefined;
  if (inner) {
    terrain = new Float32Array(nx*nz);
    for (let z=0; z<nz; z++) for (let x=0; x<nx; x++) terrain[x+nx*z] = inner[Math.min(x,cx-1)+cx*Math.min(z,cz-1)]!;
  }
  const volume: LatticePages = { edge, cells: [nx, ny, nz], pages, at };
  return {volume, terrain, initial, wetMinimum, wetMaximum, dam};
}

/**
 * fluidLatticeVertexPhi at the vertices of the pages `stored` names
 * (LatticePages, `edge` cells a page, a page's (edge + 1)^3 vertices from
 * its low corner; every other page undefined), sampled vertex by vertex: the
 * closed cells as each page of them is first read, the authored union's
 * transform over its own lattice ranges (boxUnionVertexPhi, windowed). The
 * same value at every vertex it holds.
 */
export function fluidLatticePagedVertexPhi(scene: SceneDescription, edge: number, stored: (x: number, y: number, z: number) => boolean, coarseSolids = false): LatticePages {
  if (scene.container.walls === "none") scene = sceneFluidBoxScene(scene);
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER), [cx, cy, cz] = cells;
  const [nx, ny, nz] = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (coarseSolids && (cx%4 || cy%4 || cz%4)) throw new Error("Cut 4h solid owners need the whole 4h-tiled lattice");
  const c = scene.container, empty = Math.max(c.width_m,c.height_m,c.depth_m), far = fluidLatticeFarPhi(scene);
  // The container's closed cells, a page of them at a time.
  const closedAt = closedCell(scene, cells), held = new Map<number, Uint8Array>(), qx = Math.ceil(cx/edge), qy = Math.ceil(cy/edge);
  let lastKey = -1, last: Uint8Array = new Uint8Array(0);
  const closed = (x: number, y: number, z: number): boolean => {
    const key = (x/edge|0)+qx*((y/edge|0)+qy*(z/edge|0));
    if (key !== lastKey) {
      let page = held.get(key);
      if (!page) {
        page = new Uint8Array(edge**3); held.set(key, page);
        const o = [x-x%edge, y-y%edge, z-z%edge];
        for (let k=0; k<Math.min(edge, cz-o[2]!); k++) for (let j=0; j<Math.min(edge, cy-o[1]!); j++) for (let i=0; i<Math.min(edge, cx-o[0]!); i++)
          page[i+edge*(j+edge*k)] = closedAt(o[0]!+i, o[1]!+j, o[2]!+k) ? 1 : 0;
      }
      lastKey = key; last = page;
    }
    return last[x%edge+edge*(y%edge+edge*(z%edge))] === 1;
  };
  // buriedVertices at one vertex.
  const openTile = (x: number, y: number, z: number) => {
    for (let k=0; k<64; k++) if (!closed(4*x+(k&3), 4*y+((k>>2)&3), 4*z+(k>>4))) return true;
    return false;
  };
  const buried = (x: number, y: number, z: number): boolean => {
    let any = false;
    for (let k=z-1; k<=z; k++) for (let j=y-1; j<=y; j++) for (let i=x-1; i<=x; i++) {
      if (i<0 || j<0 || k<0 || i>=cx || j>=cy || k>=cz) continue;
      any = true;
      if (!closed(i, j, k)) return false;
    }
    if (!any) return false;
    if (coarseSolids && !(x&3) && !(y&3) && !(z&3)) {
      const tx = x>>2, ty = y>>2, tz = z>>2;
      for (let k=tz-1; k<=tz; k++) for (let j=ty-1; j<=ty; j++) for (let i=tx-1; i<=tx; i++)
        if (i>=0 && j>=0 && k>=0 && i<cx/4 && j<cy/4 && k<cz/4 && openTile(i, j, k)) return false;
    }
    return true;
  };
  const sample = vertexPhiSampler(scene, cells, boxUnionVertexPhi(scene, cells, closed, undefined, true));
  const inner = (x: number, y: number, z: number) => Math.fround(buried(x, y, z) ? empty : sample(x, y, z, z));
  const same = nx === cx && ny === cy && nz === cz, cellHeight = c.height_m/cy, rows = c.top === "open" ? ny : cy;
  const pages: [number, number, number] = [Math.ceil(nx/edge), Math.ceil(ny/edge), Math.ceil(nz/edge)], [px, py, pz] = pages, n = edge+1;
  const at = new Map<number, Float32Array>();
  for (let z=0; z<pz; z++) for (let y=0; y<py; y++) for (let x=0; x<px; x++) {
    if (!stored(x, y, z)) continue;
    const page = new Float32Array(n**3); at.set(x+px*(y+py*z), page);
    for (let k=0; k<n && z*edge+k<=nz; k++) for (let j=0; j<n && y*edge+j<=ny; j++) for (let i=0; i<n && x*edge+i<=nx; i++) {
      const X = x*edge+i, Y = y*edge+j, Z = z*edge+k;
      page[i+n*(j+n*k)] = same ? inner(X, Y, Z) : X<=cx && Z<=cz && Y<=rows ? inner(X, Math.min(Y,cy), Z) + Math.max(0,Y-cy)*cellHeight : far;
    }
  }
  return { edge, cells: [nx, ny, nz], pages, at };
}
