import { proceduralSolidField } from "./procedural-solid";
import { initialHeightFieldHeight } from "./initial-height-field";
import { damBreakSignedDistanceAtNode, initialFluidBrickSignedDistanceAtNode,
  initialLiquidVolumesSignedDistance, baseInitialLiquidFractionAtCell, damBreakBoxContains, initialLiquidFractionAtCell, sceneDamBreakBox,
  sceneInitialLiquidVolumes, INITIAL_FLUID_BRICK_SIZE } from "./initial-fluid";
import { terrainColumnHeights } from "./terrain";
import { sampleSolidWorld, solidWorldForScene } from "./solid-world";
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
  return (x, y, z, outputZ) => {
    const point = { x: (x/nx-0.5)*c.width_m, y: y/ny*c.height_m, z: (z/nz-0.5)*c.depth_m };
    let base = height ? point.y-initialHeightFieldHeight(height, point.x, point.z)
      : c.fillFraction > 0 ? damBreakSignedDistanceAtNode(scene,x,y,z,dimensions)
        ?? (point.y-c.fillFraction*c.height_m) : empty;
    const brick = initialFluidBrickSignedDistanceAtNode(scene,x,y,z,dimensions);
    if (brick !== undefined) base = scene.fluid.initialBrickSeedsAdditive ? Math.min(base,brick) : brick;
    const inside = boxes ? boxes(x, y, outputZ) : Infinity;
    const value = scene.systems?.fluid === false ? empty
      : Math.min(base,inside <= 0 ? inside : initialLiquidVolumesSignedDistance(scene,point) ?? empty);
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
    const zs = sliceZ === undefined ? [outputZ-1, outputZ] : [Math.floor(sliceZ)];
    let allClosed = true, any = false;
    for (const z of zs) for (const cy of [y-1, y]) for (const cx of [x-1, x]) {
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
      for (const cz of [z-1, z]) for (const cy of [y-1, y]) for (const cx of [x-1, x])
        if (cx>=0 && cy>=0 && cz>=0 && cx<tx && cy<ty && cz<tz && open[cx+tx*(cy+ty*cz)]) any = true;
      if (any) buried[4*x+(nx+1)*(4*y+(ny+1)*4*z)] = 0;
    }
  }
  return buried;
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
  const terrain = terrainColumnHeights(scene, nx, nz), hy = scene.container.height_m / ny;
  return (x, y, z) => (boundary?.open([x,y,z]) ?? Number(sampleSolidWorld(world,[x,y,z],true).solidFraction === 0)) <= 1e-5
    || Math.fround(terrain[x+nx*z]! / hy) >= y + 1;
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
 * A face is free where the cell beyond it is open. One inside a solid or past
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
      if (!closed(cell[0]!, cell[1]!, cell[2]!)) return true;
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
  const {density:at,terrain,dam}=cellVolumeSampler(scene,dimensions,geometric);
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
  const terrain=terrainColumnHeights(scene,nx,nz);
  const cellHeight=c.height_m/ny;
  const solidWorld=solidWorldForScene(scene);
  const boundary=geometric?proceduralSolidField(scene):undefined;
  const dam=sceneDamBreakBox(scene);
  const density=(x: number, y: number, z: number): number => {
    const aboveGround=(y+0.5)*cellHeight>terrain[x+nx*z];
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
      Math.fround(terrain[x+nx*z]!/cellHeight)-y))) : 1;
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
  const [nx, ny, nz] = dimensions, c = scene.container, out: [number[], number[]][] = [];
  const cell = (axis: number, metres: number) => [(metres/c.width_m+0.5)*nx, metres/c.height_m*ny, (metres/c.depth_m+0.5)*nz][axis]!;
  const add = (low: number[], high: number[]) => {
    const lo = low.map(v => Math.max(0, Math.floor(v)-1)), hi = high.map((v, a) => Math.min(dimensions[a]!, Math.floor(v)+2));
    if (lo.every((v, a) => v < hi[a]!)) out.push([lo, hi]);
  };
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
  return out.every(([lo, hi]) => lo.concat(hi).every(Number.isFinite)) ? out : undefined;
}

/**
 * A lattice field page by page: the page at (x, y, z) of `pages` is at[x +
 * pages[0] (y + pages[1] z)], edge^3 cells x fastest (a vertex field's
 * (edge + 1)^3 vertices from the page's low corner), or undefined where it
 * holds zero throughout.
 */
export interface LatticePages {
  readonly edge: number;
  readonly cells: readonly [number, number, number];
  readonly pages: readonly [number, number, number];
  readonly at: readonly (Float32Array | undefined)[];
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
  const pages: [number, number, number] = [Math.ceil(nx/edge), Math.ceil(ny/edge), Math.ceil(nz/edge)], [px, py, pz] = pages;
  const bounds = initialLiquidCellBounds(scene, cells) ?? [[[0, 0, 0], [cx, cy, cz]]];
  const at: (Float32Array | undefined)[] = new Array(px*py*pz).fill(undefined);
  const wetMinimum = [cx, cy, cz], wetMaximum = [0, 0, 0];
  for (let z=0; z<pz; z++) for (let y=0; y<py; y++) for (let x=0; x<px; x++) {
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
    if (any) at[x+px*(y+py*z)] = page;
  }
  let initial = 0;
  for (let z=0; z<cz; z++) for (let y=0; y<cy; y++) for (let x=0; x<px; x++) {
    const page = at[x+px*((y/edge|0)+py*(z/edge|0))];
    if (!page) continue;
    const row = edge*(y%edge+edge*(z%edge));
    for (let i=0; i<Math.min(edge, cx-x*edge); i++) initial += page[row+i]!;
  }
  const terrain = new Float32Array(nx*nz);
  for (let z=0; z<nz; z++) for (let x=0; x<nx; x++) terrain[x+nx*z] = inner[Math.min(x,cx-1)+cx*Math.min(z,cz-1)]!;
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
  const at: (Float32Array | undefined)[] = new Array(px*py*pz).fill(undefined);
  for (let z=0; z<pz; z++) for (let y=0; y<py; y++) for (let x=0; x<px; x++) {
    if (!stored(x, y, z)) continue;
    const page = at[x+px*(y+py*z)] = new Float32Array(n**3);
    for (let k=0; k<n && z*edge+k<=nz; k++) for (let j=0; j<n && y*edge+j<=ny; j++) for (let i=0; i<n && x*edge+i<=nx; i++) {
      const X = x*edge+i, Y = y*edge+j, Z = z*edge+k;
      page[i+n*(j+n*k)] = same ? inner(X, Y, Z) : X<=cx && Z<=cz && Y<=rows ? inner(X, Math.min(Y,cy), Z) + Math.max(0,Y-cy)*cellHeight : far;
    }
  }
  return { edge, cells: [nx, ny, nz], pages, at };
}
