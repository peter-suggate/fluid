import type { SceneDescription } from "./model";
import { sceneCellSizes_m, sceneLatticeDimensions } from "./scene-lattice-dimensions";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { bathVoxelBoxes } from "./voxel-bath";

/** How far, in cells, the cavity is felt: the solid within this of it is given an outward direction. */
export const VESSEL_CUTAWAY_REACH_CELLS = 4;
/** The thickest wall, in cells, that is cut. A hillside around a pond is not a vessel. */
export const VESSEL_CUTAWAY_WALL_CELLS = 32;

/** Cells of open margin kept beside and above the lattice, so a wall standing just outside it is still seen. None below: that is the ground. */
const MARGIN_CELLS = 2;
const INTERIOR = 0, SOLID = 1, DRAIN = 2;

/**
 * The drawn solids' near walls, as clear patches for the render-only solid
 * world: what a vessel has to lose for `scene.cutaway.eye_m` to see inside it.
 * Empty when the scene asks for no cutaway.
 *
 * Nothing about the vessel is authored. From the voxels alone:
 *
 *  - **Cavity.** An empty cell is inside a vessel when water put there could
 *    not leave: no path sideways or down reaches the open sides of the
 *    lattice. The ground under the lattice is closed.
 *  - **Wall.** Solid within reach of the cavity points outward, the way the
 *    cavity thins out fastest, and the solid behind it points as it does. A
 *    voxel is wall when the solid run through it, along that direction, has
 *    the cavity on one side and the open on the other. An obstacle standing
 *    in the liquid has cavity on both sides and is not.
 *  - **Cut.** Wall whose outward side faces the eye: the wall the eye would
 *    see from outside. Taken a whole vertical run of solid at a time.
 *
 * Within reach of a corner the direction turns with it, so a wall that meets
 * a cut one may lose that many cells of its own end.
 *
 * `patches` are the solids as drawn, in document order and lattice cells.
 */
export function vesselCutawayPatches(
  scene: Pick<SceneDescription, "container" | "voxelDomain" | "cutaway">,
  patches: readonly SolidWorldVoxelPatch[],
): SolidWorldVoxelPatch[] {
  if (!scene.cutaway) return [];
  const lattice = sceneLatticeDimensions(scene), cell = sceneCellSizes_m(scene), M = MARGIN_CELLS;
  const nx = lattice[0] + 2 * M, ny = lattice[1] + M, nz = lattice[2] + 2 * M, n = nx * ny * nz;
  const margin = [M, 0, M] as const;
  const strideY = nx, strideZ = nx * ny;
  const kind = new Uint8Array(n);
  for (const patch of patches) {
    const value = patch.operation === "fill" ? SOLID : INTERIOR;
    const from = patch.minimum.map((v, axis) => Math.max(0, v + margin[axis]!));
    const to = patch.maximumExclusive.map((v, axis) => Math.min([nx, ny, nz][axis]!, v + margin[axis]!));
    for (let z = from[2]!; z < to[2]!; z++) for (let y = from[1]!; y < to[1]!; y++) {
      kind.fill(value, from[0]! + strideY * y + strideZ * z, Math.max(from[0]!, to[0]!) + strideY * y + strideZ * z);
    }
  }

  // Where water drains: flood back from the lattice's open sides, sideways and up.
  const queue = new Int32Array(n);
  let head = 0, tail = 0;
  const drain = (index: number) => { if (kind[index] === INTERIOR) { kind[index] = DRAIN; queue[tail++] = index; } };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (x === 0 || x === nx - 1 || z === 0 || z === nz - 1) drain(x + strideY * y + strideZ * z);
  }
  while (head < tail) {
    const index = queue[head++]!, x = index % nx, y = Math.floor(index / strideY) % ny, z = Math.floor(index / strideZ);
    if (x > 0) drain(index - 1);
    if (x < nx - 1) drain(index + 1);
    if (z > 0) drain(index - strideZ);
    if (z < nz - 1) drain(index + strideZ);
    if (y < ny - 1) drain(index + strideY);
  }

  // Cavity cells within reach of each cell, as a box sum one axis at a time.
  const R = VESSEL_CUTAWAY_REACH_CELLS;
  let cavity = new Uint16Array(n), scratch = new Uint16Array(n), any = false;
  for (let index = 0; index < n; index++) if (kind[index] === INTERIOR) { cavity[index] = 1; any = true; }
  if (!any) return [];
  for (const [stride, length] of [[1, nx], [strideY, ny], [strideZ, nz]] as const) {
    // Every line along this axis: `stride` lines to a block, a block every `stride * length` cells.
    for (let block = 0; block < n; block += stride * length) for (let line = block; line < block + stride; line++) {
      let sum = 0;
      for (let along = 0; along < R && along < length; along++) sum += cavity[line + along * stride]!;
      for (let along = 0, index = line; along < length; along++, index += stride) {
        // Slide the window one cell on.
        if (along + R < length) sum += cavity[index + R * stride]!;
        if (along > R) sum -= cavity[index - (R + 1) * stride]!;
        scratch[index] = sum;
      }
    }
    [cavity, scratch] = [scratch, cavity];
  }

  const eye = [
    (scene.cutaway.eye_m.x + 0.5 * scene.container.width_m) / cell[0] + M,
    scene.cutaway.eye_m.y / cell[1],
    (scene.cutaway.eye_m.z + 0.5 * scene.container.depth_m) / cell[2] + M,
  ] as const;
  // Outward, in 127ths: down the gradient of the cavity count.
  const outwardX = new Int8Array(n), outwardY = new Int8Array(n), outwardZ = new Int8Array(n);
  const directed = (index: number) => (outwardX[index]! | outwardY[index]! | outwardZ[index]!) !== 0;
  head = tail = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const index = x + strideY * y + strideZ * z;
    if (kind[index] !== SOLID) continue;
    const ox = (x > 0 ? cavity[index - 1]! : 0) - (x < nx - 1 ? cavity[index + 1]! : 0);
    const oy = (y > 0 ? cavity[index - strideY]! : 0) - (y < ny - 1 ? cavity[index + strideY]! : 0);
    const oz = (z > 0 ? cavity[index - strideZ]! : 0) - (z < nz - 1 ? cavity[index + strideZ]! : 0);
    const steepness = Math.hypot(ox, oy, oz);
    if (steepness === 0) continue;
    outwardX[index] = Math.round(127 * ox / steepness);
    outwardY[index] = Math.round(127 * oy / steepness);
    outwardZ[index] = Math.round(127 * oz / steepness);
    queue[tail++] = index;
  }
  // Solid deeper in a thick wall takes the direction of the wall in front of it.
  const inherit = (from: number, to: number) => {
    if (kind[to] !== SOLID || directed(to)) return;
    outwardX[to] = outwardX[from]!; outwardY[to] = outwardY[from]!; outwardZ[to] = outwardZ[from]!;
    queue[tail++] = to;
  };
  for (let layer = 0; layer < VESSEL_CUTAWAY_WALL_CELLS && head < tail; layer++) {
    for (const end = tail; head < end; head++) {
      const index = queue[head]!, x = index % nx, y = Math.floor(index / strideY) % ny, z = Math.floor(index / strideZ);
      if (x > 0) inherit(index, index - 1);
      if (x < nx - 1) inherit(index, index + 1);
      if (y > 0) inherit(index, index - strideY);
      if (y < ny - 1) inherit(index, index + strideY);
      if (z > 0) inherit(index, index - strideZ);
      if (z < nz - 1) inherit(index, index + strideZ);
    }
  }

  const cut = new Uint8Array(n);
  let cutCells = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const index = x + strideY * y + strideZ * z;
    if (!directed(index)) continue;
    const ex = eye[0] - x - 0.5, ey = eye[1] - y - 0.5, ez = eye[2] - z - 0.5;
    if (outwardX[index]! * ex + outwardY[index]! * ey + outwardZ[index]! * ez <= 0) continue;
    // Wall only if the solid run through this voxel has the cavity before it and the open behind.
    const length = Math.hypot(outwardX[index]!, outwardY[index]!, outwardZ[index]!);
    const ox = outwardX[index]! / length, oy = outwardY[index]! / length, oz = outwardZ[index]! / length;
    const past = (sign: number) => {
      for (let t = sign; Math.abs(t) <= VESSEL_CUTAWAY_WALL_CELLS; t += sign) {
        const px = Math.round(x + ox * t), py = Math.round(y + oy * t), pz = Math.round(z + oz * t);
        if (px < 0 || py < 0 || pz < 0 || px >= nx || py >= ny || pz >= nz) return sign > 0 ? DRAIN : SOLID;
        const found = kind[px + strideY * py + strideZ * pz]!;
        if (found !== SOLID) return found;
      }
      return SOLID;
    };
    if (past(1) === DRAIN && past(-1) === INTERIOR) { cut[index] = 1; cutCells++; }
  }
  if (cutCells === 0) return [];
  // Cut whole vertical runs of solid or none of them, by majority, so that the
  // cut's edge on a wall stands upright instead of stepping with the corners.
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny;) {
      const base = x + strideZ * z;
      if (kind[base + strideY * y] !== SOLID) { y++; continue; }
      let top = y, votes = 0;
      for (; top < ny && kind[base + strideY * top] === SOLID; top++) votes += cut[base + strideY * top]!;
      if (votes > 0) for (let run = y; run < top; run++) cut[base + strideY * run] = 2 * votes >= top - y ? 1 : 0;
      y = top;
    }
  }
  return bathVoxelBoxes([nx, ny, nz], (x, y, z) => cut[x + strideY * y + strideZ * z] === 1)
    .map(({ minimum, maximumExclusive }) => ({
      operation: "clear",
      minimum: [minimum[0] - M, minimum[1], minimum[2] - M],
      maximumExclusive: [maximumExclusive[0] - M, maximumExclusive[1], maximumExclusive[2] - M],
    }));
}
