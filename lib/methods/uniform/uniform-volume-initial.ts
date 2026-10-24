export { initialLiquidVertexPhi as uniformVolumeInitialPhi, initialLiquidCellVolumes as uniformInitialVolume,
  fluidLatticeVertexPhi as uniformLatticeInitialPhi, fluidLatticeCellVolumes as uniformLatticeInitialVolume,
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
