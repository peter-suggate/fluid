/** CPU diagnostics on canonical mixed owners. No solver decisions use these.
 * Volume is a fraction per owner, phi is stored at canonical owner corners.
 * Positions are in finest-cell coordinates; projected volume is depth-mean.
 */
export function uniformQualityCensus(
  dimensions: readonly [number, number, number], tiles: Uint32Array,
  volume: Float32Array, phi: Float32Array,
) {
  const [nx, ny, nz] = dimensions, tx = nx / 4, ty = ny / 4;
  if (dimensions.some(n => !Number.isInteger(n / 4)) || tiles.length !== nx * ny * nz / 64
    || volume.length !== nx * ny * nz || phi.length !== (nx + 1) * (ny + 1) * (nz + 1))
    throw new Error("Quality census requires complete canonical mixed fields");
  const column = new Float64Array(nx), projection = new Float64Array(nx * ny);
  const phiSlice = new Float64Array(nx * ny);
  let mass = 0, excess = 0, negative = 0, mixedMass = 0, massInPhiAir = 0;
  let maxVolume = 0, nonfinite = 0, owners = 0, fineTiles = 0;
  const moment = [0, 0, 0];
  for (let t = 0; t < tiles.length; t++) {
    const w = tiles[t]! & 0x80000000 ? 1 : 4;
    if (w === 1) fineTiles++;
    const base = [4 * (t % tx), 4 * (Math.floor(t / tx) % ty), 4 * Math.floor(t / (tx * ty))];
    for (let z = base[2]!; z < base[2]! + 4; z += w)
      for (let y = base[1]!; y < base[1]! + 4; y += w)
        for (let x = base[0]!; x < base[0]! + 4; x += w) {
          owners++;
          const v = volume[x + nx * (y + ny * z)]!, capacity = w ** 3;
          const corners = Array.from({length: 8}, (_, k) => phi[x + (k & 1) * w
            + (nx + 1) * (y + ((k >> 1) & 1) * w + (ny + 1) * (z + ((k >> 2) & 1) * w))]!);
          if (!Number.isFinite(v) || !corners.every(Number.isFinite)) { nonfinite++; continue; }
          mass += v * capacity;
          excess += Math.max(v - 1, 0) * capacity;
          negative += Math.max(-v, 0) * capacity;
          maxVolume = Math.max(maxVolume, v);
          if (v > 0 && v < 1) mixedMass += v * capacity;
          if (corners.reduce((a, b) => a + b, 0) > 0) massInPhiAir += Math.max(v, 0) * capacity;
          [x, y, z].forEach((p, a) => { moment[a]! += v * capacity * (p + w / 2); });
          for (let dx = 0; dx < w; dx++) {
            column[x + dx]! += Math.max(v, 0) * w * w;
            for (let dy = 0; dy < w; dy++) {
              projection[x + dx + nx * (y + dy)]! += v * w / nz;
              if (z <= nz / 2 && z + w > nz / 2) {
                const f = [(dx + .5) / w, (dy + .5) / w, (nz / 2 - z) / w];
                let p = 0;
                for (let k = 0; k < 8; k++) p += corners[k]! * f.reduce((a, q, axis) => a * ((k >> axis) & 1 ? q : 1 - q), 1);
                phiSlice[x + dx + nx * (y + dy)] = p;
              }
            }
          }
        }
  }
  const positiveMass = column.reduce((a, b) => a + b, 0);
  const percentile = (q: number) => {
    if (positiveMass === 0) return null;
    let sum = 0;
    for (let x = 0; x < nx; x++) {
      const v = column[x]!;
      if (v > 0 && sum + v >= q * positiveMass) return x + (q * positiveMass - sum) / v;
      sum += v;
    }
    return nx;
  };
  return {
    mass, excess, negative, mixedMass, massInPhiAir, maxVolume, nonfinite, owners, fineTiles,
    centroid_cells: mass > 0 ? moment.map(m => m / mass) : null,
    massFront_cells: {p95: percentile(.95), p99: percentile(.99), p999: percentile(.999)},
    // These diagnostic images are not a reconstructed production rendering.
    projection: Array.from(projection), phiSlice: Array.from(phiSlice),
  };
}
