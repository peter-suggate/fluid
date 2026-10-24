import type { LatticePages } from "../../core/initial-liquid-lattice";
export { initialLiquidVertexPhi as uniformVolumeInitialPhi, initialLiquidCellVolumes as uniformInitialVolume,
  fluidLatticeVertexPhi as uniformLatticeInitialPhi, fluidLatticeCellVolumes as uniformLatticeInitialVolume,
  fluidLatticePagedVertexPhi as uniformLatticePagedPhi, fluidLatticePagedCellVolumes as uniformLatticePagedVolume,
  fluidLatticeFarPhi as uniformLatticeFarPhi } from "../../core/initial-liquid-lattice";

/** The t=0 MAC velocity texels (x, y, z, 0 per cell, as the velocity field
 * stores each cell's +faces) of a scene whose liquid starts moving at the
 * uniform `velocity` (scene.fluid.initialVelocity_m_s): every face of each 4h
 * tile that holds liquid (volume > 0) and of the tiles around it, zero
 * beyond. A 4h owner's faces sit at its anchors, a cell's -face in its
 * neighbour, so every face of a liquid owner is set whatever the layout; the
 * extension carries the field on into the air from there. Undefined: the
 * liquid starts at rest (no upload). */
export function uniformInitialLiquidVelocity(velocity: { x: number; y: number; z: number } | undefined, volume: Float32Array, dimensions: readonly [number, number, number]): Float32Array | undefined {
  if (!velocity || (velocity.x === 0 && velocity.y === 0 && velocity.z === 0)) return undefined;
  if (![velocity.x, velocity.y, velocity.z].every(Number.isFinite)) throw new Error(`Initial liquid velocity must be finite: ${JSON.stringify(velocity)}`);
  const [nx, ny, nz] = dimensions, tx = Math.ceil(nx / 4), ty = Math.ceil(ny / 4), tz = Math.ceil(nz / 4);
  if (volume.length !== nx * ny * nz) throw new Error(`Initial liquid velocity: ${volume.length} volume cells for a ${nx}x${ny}x${nz} lattice`);
  const wet = new Uint8Array(tx * ty * tz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
    if (volume[x + nx * (y + ny * z)]! > 0) wet[(x >> 2) + tx * ((y >> 2) + ty * (z >> 2))] = 1;
  const near = new Uint8Array(wet.length);
  for (let k = 0; k < tz; k++) for (let j = 0; j < ty; j++) for (let i = 0; i < tx; i++) {
    if (!wet[i + tx * (j + ty * k)]) continue;
    for (let c = Math.max(0, k - 1); c <= Math.min(tz - 1, k + 1); c++) for (let b = Math.max(0, j - 1); b <= Math.min(ty - 1, j + 1); b++)
      for (let a = Math.max(0, i - 1); a <= Math.min(tx - 1, i + 1); a++) near[a + tx * (b + ty * c)] = 1;
  }
  const field = new Float32Array(4 * nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (!near[(x >> 2) + tx * ((y >> 2) + ty * (z >> 2))]) continue;
    const i = 4 * (x + nx * (y + ny * z));
    field[i] = velocity.x; field[i + 1] = velocity.y; field[i + 2] = velocity.z;
  }
  return field;
}

/** uniformInitialLiquidVelocity page by page (LatticePages, its texels four
 * values a cell), from the volume's pages: a page is held where a tile of it
 * is wet or beside one. */
export function uniformInitialLiquidVelocityPages(velocity: { x: number; y: number; z: number } | undefined, volume: LatticePages): LatticePages | undefined {
  if (!velocity || (velocity.x === 0 && velocity.y === 0 && velocity.z === 0)) return undefined;
  if (![velocity.x, velocity.y, velocity.z].every(Number.isFinite)) throw new Error(`Initial liquid velocity must be finite: ${JSON.stringify(velocity)}`);
  const { edge: E, cells: [nx, ny, nz], pages: [px, py, pz] } = volume, T = E / 4, tx = Math.ceil(nx / 4), ty = Math.ceil(ny / 4), tz = Math.ceil(nz / 4);
  if (E % 4) throw new Error(`Initial liquid velocity: pages of ${E} cells are not whole 4h tiles`);
  const wet = volume.at.map(page => {
    if (!page) return undefined;
    const tiles = new Uint8Array(T ** 3);
    for (let z = 0, i = 0; z < E; z++) for (let y = 0; y < E; y++) for (let x = 0; x < E; x++, i++) if (page[i]! > 0) tiles[(x >> 2) + T * ((y >> 2) + T * (z >> 2))] = 1;
    return tiles;
  });
  const wetTile = (i: number, j: number, k: number) => wet[((i / T) | 0) + px * (((j / T) | 0) + py * ((k / T) | 0))]?.[i % T + T * (j % T + T * (k % T))] === 1;
  const at: (Float32Array | undefined)[] = new Array(px * py * pz).fill(undefined);
  for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = 0; x < px; x++) {
    let field: Float32Array | undefined, close = false;
    for (let c = Math.max(0, z - 1); c <= Math.min(pz - 1, z + 1); c++) for (let b = Math.max(0, y - 1); b <= Math.min(py - 1, y + 1); b++)
      for (let a = Math.max(0, x - 1); a <= Math.min(px - 1, x + 1); a++) close ||= !!wet[a + px * (b + py * c)];
    if (!close) continue;
    for (let k = z * T; k < Math.min(tz, (z + 1) * T); k++) for (let j = y * T; j < Math.min(ty, (y + 1) * T); j++) for (let i = x * T; i < Math.min(tx, (x + 1) * T); i++) {
      let near = false;
      for (let c = Math.max(0, k - 1); c <= Math.min(tz - 1, k + 1) && !near; c++) for (let b = Math.max(0, j - 1); b <= Math.min(ty - 1, j + 1) && !near; b++)
        for (let a = Math.max(0, i - 1); a <= Math.min(tx - 1, i + 1) && !near; a++) near = wetTile(a, b, c);
      if (!near) continue;
      field ??= at[x + px * (y + py * z)] = new Float32Array(4 * E ** 3);
      for (let w = 4 * k; w < Math.min(nz, 4 * k + 4); w++) for (let v = 4 * j; v < Math.min(ny, 4 * j + 4); v++) for (let u = 4 * i; u < Math.min(nx, 4 * i + 4); u++) {
        const o = 4 * (u - E * x + E * (v - E * y + E * (w - E * z)));
        field[o] = velocity.x; field[o + 1] = velocity.y; field[o + 2] = velocity.z;
      }
    }
  }
  return { edge: E, cells: volume.cells, pages: volume.pages, at };
}
