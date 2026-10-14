import { initialHeightFieldHeight } from "./initial-height-field";
import { damBreakSignedDistanceAtNode, initialFluidBrickSignedDistanceAtNode,
  initialLiquidVolumesSignedDistance, baseInitialLiquidFractionAtCell, damBreakBoxContains, initialLiquidFractionAtCell, sceneDamBreakBox,
  sceneInitialLiquidVolumes } from "./initial-fluid";
import { terrainColumnHeights } from "./terrain";
import { sampleSolidWorld, solidWorldForScene } from "./solid-world";
import type { InitialLiquidBox, SceneDescription } from "./model";
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from "./scene-lattice-dimensions";

/** Construction-only analytic samples, in metres, on shared lattice vertices.
 * coarseSolids: static solids run on cut 4h owners (uniformDetailCoarseSolids),
 * so a 4-aligned vertex is buried by its tiles' capacity, not its h cells'. */
export function initialLiquidVertexPhi(scene: SceneDescription, dimensions: readonly [number, number, number], sliceZ?: number, coarseSolids = false): Float32Array {
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const phi = new Float32Array((nx+1)*(ny+1)*(sliceZ === undefined ? nz+1 : 1));
  const closed = closedCells(scene, dimensions);
  const buried = buriedVertices(closed, dimensions, sliceZ, coarseSolids);
  const boxes = boxUnionVertexPhi(scene, dimensions, closed, sliceZ);
  for (let outputZ=0; outputZ<(sliceZ === undefined ? nz+1 : 1); outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    const z = sliceZ ?? outputZ;
    const point = { x: (x/nx-0.5)*c.width_m, y: y/ny*c.height_m, z: (z/nz-0.5)*c.depth_m };
    const empty = Math.max(c.width_m,c.height_m,c.depth_m);
    // phi inside a solid is not state (uvBuried in uniform-volume.wgsl.ts,
    // umVertexBuried on the mixed lattice): a vertex with no open incident
    // owner is air by construction, whatever seeded body the analytic samples
    // would put there.
    if (buried[x+(nx+1)*(y+(ny+1)*outputZ)]) { phi[x+(nx+1)*(y+(ny+1)*outputZ)] = empty; continue; }
    const height = scene.fluid.initialHeightField;
    let base = height ? point.y-initialHeightFieldHeight(height, point.x, point.z)
      : c.fillFraction > 0 ? damBreakSignedDistanceAtNode(scene,x,y,z,dimensions)
        ?? (point.y-c.fillFraction*c.height_m) : empty;
    const brick = initialFluidBrickSignedDistanceAtNode(scene,x,y,z,dimensions);
    if (brick !== undefined) base = scene.fluid.initialBrickSeedsAdditive ? Math.min(base,brick) : brick;
    const inside = boxes ? boxes[x+(nx+1)*(y+(ny+1)*outputZ)]! : Infinity;
    const value = scene.systems?.fluid === false ? empty
      : Math.min(base,inside <= 0 ? inside : initialLiquidVolumesSignedDistance(scene,point) ?? empty);
    phi[x+(nx+1)*(y+(ny+1)*outputZ)] = Number.isFinite(value) ? value : Math.max(c.width_m,c.height_m,c.depth_m);
  }
  return phi;
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

/** Cells the solid world closes, under the occupancy mask's rule (any solid fraction). */
function closedCells(scene: SceneDescription, dimensions: readonly [number, number, number]): Uint8Array {
  const [nx, ny, nz] = dimensions;
  const world = solidWorldForScene(scene);
  const closed = new Uint8Array(nx*ny*nz);
  for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++)
    closed[x+nx*(y+ny*z)] = sampleSolidWorld(world,[x,y,z]).solidFraction > 0 ? 1 : 0;
  return closed;
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
 * no z faces.
 */
function boxUnionVertexPhi(scene: SceneDescription, dimensions: readonly [number, number, number], closed: Uint8Array, sliceZ?: number): Float32Array | undefined {
  const volumes = sceneInitialLiquidVolumes(scene);
  const authored = volumes.filter((volume): volume is InitialLiquidBox => volume.shape === "box");
  if (authored.length === 0 || authored.length !== volumes.length) return undefined;
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const cells = [nx, ny, nz], size = [nx+1, ny+1, sliceZ === undefined ? nz+1 : 1], stride = [1, nx+1, (nx+1)*(ny+1)];
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
  const cell = [0, 0, 0];
  const free = (a: number, beyond: number, b: number, ib: number, d: number, id: number) => {
    if (beyond < 0 || beyond >= cells[a]!) return false;
    for (const cd of sliced(d) ? [Math.floor(sliceZ!)] : [id-1, id]) for (const cb of sliced(b) ? [Math.floor(sliceZ!)] : [ib-1, ib]) {
      if (cb<0 || cd<0 || cb>=cells[b]! || cd>=cells[d]!) continue;
      cell[a] = beyond; cell[b] = cb; cell[d] = cd;
      if (!closed[cell[0]!+nx*(cell[1]!+ny*cell[2]!)]) return true;
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
      lines[ib+size[b]!*id]!.push([box.lo[a]!, box.hi[a]!]);
    along.fill(Infinity);
    for (let id=0; id<size[d]!; id++) for (let ib=0; ib<size[b]!; ib++) {
      const spans = lines[ib+size[b]!*id]!.sort((p, q) => p[0]-q[0]);
      for (let s=0; s<spans.length;) {
        const lo = spans[s]![0];
        let hi = spans[s]![1];
        for (s++; s<spans.length && spans[s]![0] <= hi; s++) hi = Math.max(hi, spans[s]![1]);
        const below = free(a, Math.ceil(lo)-1, b, ib, d, id), above = free(a, Math.floor(hi), b, ib, d, id);
        for (let k=Math.max(0, Math.ceil(lo)); k<=Math.min(cells[a]!, Math.floor(hi)); k++) {
          const reach = Math.min(below ? k-lo : Infinity, above ? hi-k : Infinity)*spacing[a]!;
          inside[k*stride[a]!+ib*stride[b]!+id*stride[d]!] = 1;
          along[k*stride[a]!+ib*stride[b]!+id*stride[d]!] = reach*reach;
        }
      }
    }
    spreadSquaredDistance(along, size, stride, b, spacing[b]!);
    spreadSquaredDistance(along, size, stride, d, spacing[d]!);
    for (let i=0; i<total; i++) nearest[i] = Math.min(nearest[i]!, along[i]!);
  }
  const empty = Math.max(c.width_m, c.height_m, c.depth_m);
  const phi = new Float32Array(total).fill(Infinity);
  for (let i=0; i<total; i++) if (inside[i]) phi[i] = Number.isFinite(nearest[i]!) ? -Math.sqrt(nearest[i]!) : -empty;
  return phi;
}

/**
 * One axis of Felzenszwalb and Huttenlocher's squared distance transform, in
 * place: field(q) <- min over p on q's line of field(p) + (spacing (q-p))^2.
 */
function spreadSquaredDistance(field: Float64Array, size: readonly number[], stride: readonly number[], axis: number, spacing: number): void {
  const n = size[axis]!, b = (axis+1)%3, d = (axis+2)%3, step = stride[axis]!, scale = spacing*spacing;
  const line = new Float64Array(n), site = new Int32Array(n), from = new Float64Array(n);
  for (let id=0; id<size[d]!; id++) for (let ib=0; ib<size[b]!; ib++) {
    const start = ib*stride[b]!+id*stride[d]!;
    let top = -1;
    for (let q=0; q<n; q++) {
      const value = line[q] = field[start+q*step]!;
      if (value === Infinity) continue;
      let begin = -Infinity;
      for (; top >= 0; top--) {
        const p = site[top]!;
        begin = (value-line[p]!+scale*(q*q-p*p))/(2*scale*(q-p));
        if (begin > from[top]!) break;
      }
      if (top < 0) begin = -Infinity;
      site[++top] = q; from[top] = begin;
    }
    if (top < 0) continue;
    for (let q=0, k=0; q<n; q++) {
      while (k < top && from[k+1]! < q) k++;
      const p = site[k]!;
      field[start+q*step] = line[p]!+scale*(q-p)*(q-p);
    }
  }
}

/** Shared initial conservative field; keeps the density comparison lane's seed rule. */
export function initialLiquidCellVolumes(scene: SceneDescription, dimensions: readonly [number, number, number], geometric = true, sliceZ?: number) {
  const [nx,ny,nz]=dimensions;
  const c=scene.container;
  const terrain=terrainColumnHeights(scene,nx,nz);
  const cellHeight=c.height_m/ny;
  const volume=new Float32Array(nx*ny*(sliceZ === undefined ? nz : 1));
  const solidWorld=solidWorldForScene(scene);
  const dam=sceneDamBreakBox(scene);
  let initial=0;
  const wetMinimum=[nx,ny,nz];
  const wetMaximum=[0,0,0];
  for(let outputZ=0;outputZ<(sliceZ === undefined ? nz : 1);outputZ++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++) {
    const z=sliceZ ?? outputZ;
    const aboveGround=(y+0.5)*cellHeight>terrain[x+nx*z];
    const solidFraction=sampleSolidWorld(solidWorld,[x,y,z]).solidFraction;
    // The geometric solver's SolidOccupancyMask closes any cell touched by
    // the static solid world, including fractional terrain voxels. Seeding
    // their fractional remainder creates immobile liquid inside closed cells;
    // the global surface-volume constraint then raises the free surface to
    // represent that inaccessible mass. Use the same capacity as the solver.
    const solidOpen=geometric ? Number(solidFraction===0) : 1-solidFraction;
    const base=geometric?baseInitialLiquidFractionAtCell(scene,x,y,z,dimensions):scene.fluid.initialCondition==="dam-break"
      ?damBreakBoxContains(dam,(x+0.5)/nx,(y+0.5)/ny,(z+0.5)/nz):(y+0.5)/ny<=c.fillFraction;
    const liquidFraction=aboveGround?initialLiquidFractionAtCell(scene,x,y,z,dimensions,base):0;
    // Match the uploaded f32 terrain height in cell units. A sliver can round
    // to zero in SolidWorld's u8 mask while still reducing GPU cell capacity.
    // This matters more as a finely resolved pond exposes more shore cells.
    const terrainOpen=geometric ? Math.fround(1-Math.max(0,Math.min(1,
      Math.fround(terrain[x+nx*z]!/cellHeight)-y))) : 1;
    const density=Math.fround(Math.min(solidOpen,terrainOpen,liquidFraction));
    volume[x+nx*(y+ny*outputZ)]=density;
    initial+=density;
    if(density>1e-5){
      wetMinimum[0]=Math.min(wetMinimum[0]!,x);wetMinimum[1]=Math.min(wetMinimum[1]!,y);wetMinimum[2]=Math.min(wetMinimum[2]!,z);
      wetMaximum[0]=Math.max(wetMaximum[0]!,x+1);wetMaximum[1]=Math.max(wetMaximum[1]!,y+1);wetMaximum[2]=Math.max(wetMaximum[2]!,z+1);
    }
  }
  return {volume,terrain,initial,wetMinimum,wetMaximum,dam};
}

/**
 * Vertex phi far from liquid on the scene's fluid lattice: its largest extent.
 * For a lattice that is its container this is the value initialLiquidVertexPhi
 * gives a vertex with no liquid near.
 */
export function fluidLatticeFarPhi(scene: SceneDescription): number {
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
