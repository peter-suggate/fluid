/** Offline representation experiment. Never changes solver decisions.
 * Tests h -> 2h/4h interpolation against all 125 vertices of each fine tile.
 * This bounds sampled phi error, NOT surface displacement or dynamic error.
 * Solids, pressure, velocity, remap and cross-tile conformity are not certified.
 */
export function uniformDetailCensus(
  dimensions: readonly [number, number, number], tiles: Uint32Array,
  volume: Float32Array, phi: Float32Array, cellSize: number,
) {
  const [nx, ny, nz] = dimensions, tx = nx / 4, ty = ny / 4, tz = nz / 4;
  if (dimensions.some(n => n % 4 !== 0) || tiles.length !== tx * ty * tz
    || volume.length !== nx * ny * nz || phi.length !== (nx + 1) * (ny + 1) * (nz + 1)
    || !(cellSize > 0)) throw new Error("Detail census needs complete canonical fields");
  const thresholds = [0.05, 0.125, 0.25, 0.5];
  const candidates = [2, 4].map(width => ({width, errors: [] as number[], signSafe: 0,
    eligible: thresholds.map(() => new Uint8Array(tiles.length))}));
  let fineTiles = 0, crossingTiles = 0, wetFineTiles = 0, crossingCells = 0;
  let fineCells = 0, nearSurfaceCells = 0, deepWetCells = 0, farDryCells = 0;
  const crossings = new Uint8Array(tiles.length);
  const vertex = (x: number, y: number, z: number) => phi[x + (nx + 1) * (y + (ny + 1) * z)]! / cellSize;
  for (let t = 0; t < tiles.length; t++) {
    if (!(tiles[t]! & 0x80000000)) continue;
    fineTiles++;
    const bx = 4 * (t % tx), by = 4 * (Math.floor(t / tx) % ty), bz = 4 * Math.floor(t / (tx * ty));
    const patch = new Float64Array(125);
    let wet = false, crossing = false;
    for (let z = 0; z <= 4; z++) for (let y = 0; y <= 4; y++) for (let x = 0; x <= 4; x++) {
      const v = vertex(bx + x, by + y, bz + z);
      if (!Number.isFinite(v)) throw new Error("Nonfinite phi in detail census");
      patch[x + 5 * (y + 5 * z)] = v;
    }
    for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
      const v = volume[bx + x + nx * (by + y + ny * (bz + z))]!;
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < 8; k++) {
        const p = patch[x + (k & 1) + 5 * (y + ((k >> 1) & 1) + 5 * (z + ((k >> 2) & 1)))]!;
        lo = Math.min(lo, p); hi = Math.max(hi, p);
      }
      fineCells++;
      if (lo <= 0 && hi >= 0) {crossingCells++; crossing = true;}
      if (lo <= 2 && hi >= -2) nearSurfaceCells++;
      if (hi < -2 && v >= .75) deepWetCells++;
      if (lo > 2 && v <= .001) farDryCells++;
      wet ||= v > .001;
    }
    wetFineTiles += +wet; crossingTiles += +crossing; crossings[t] = +crossing;
    for (const candidate of candidates) {
      const w = candidate.width;
      let error = 0, signSafe = true;
      for (let z = 0; z <= 4; z++) for (let y = 0; y <= 4; y++) for (let x = 0; x <= 4; x++) {
        const ox = Math.min(4 - w, Math.floor(x / w) * w), oy = Math.min(4 - w, Math.floor(y / w) * w), oz = Math.min(4 - w, Math.floor(z / w) * w);
        const fx = (x - ox) / w, fy = (y - oy) / w, fz = (z - oz) / w;
        let estimate = 0;
        for (let k = 0; k < 8; k++) estimate += patch[ox + (k & 1) * w + 5 * (oy + ((k >> 1) & 1) * w + 5 * (oz + ((k >> 2) & 1) * w))]!
          * (k & 1 ? fx : 1 - fx) * (k & 2 ? fy : 1 - fy) * (k & 4 ? fz : 1 - fz);
        const exact = patch[x + 5 * (y + 5 * z)]!;
        // All vertices, including those away from the surface. A zero is
        // treated as a separate sign to avoid losing sampled thin features.
        error = Math.max(error, Math.abs(estimate - exact));
        if (Math.sign(estimate) !== Math.sign(exact) && Math.abs(estimate - exact) > 1e-6) signSafe = false;
      }
      if (crossing) {candidate.errors.push(error); candidate.signSafe += +signSafe;}
      thresholds.forEach((limit, i) => {candidate.eligible[i]![t] = +(signSafe && error <= limit);});
    }
  }
  const count = (a: Uint8Array) => a.reduce((n, x) => n + x, 0);
  const currentOwners = fineTiles * 64 + tiles.length - fineTiles;
  return {
    fineTiles, crossingTiles, wetFineTiles, fineCells, crossingCells, nearSurfaceCells, deepWetCells, farDryCells, currentOwners,
    scope: "Frozen geometry only; owner reductions are ceilings, not measured speedups. No velocity, pressure, solid, seam or temporal-error certificate.",
    candidates: candidates.map(c => {
      c.errors.sort((a, b) => a - b);
      const q = (p: number) => c.errors.length ? c.errors[Math.min(c.errors.length - 1, Math.floor(p * c.errors.length))]! : null;
      return {width: c.width, crossingSignSafe: c.signSafe, crossingError_h: {p50: q(.5), p90: q(.9), p99: q(.99)},
        thresholds: thresholds.map((limit, i) => {
          const eligible = c.eligible[i]!, n = count(eligible);
          const surface = eligible.reduce((n, e, t) => n + (e && crossings[t] ? 1 : 0), 0);
          // Illustrative one-tile protection around rejected fine tiles:
          // not a complete numerical stencil closure or balance rule.
          let guarded = 0;
          for (let t = 0; t < tiles.length; t++) if (eligible[t]) {
            const x = t % tx, y = Math.floor(t / tx) % ty, z = Math.floor(t / (tx * ty));
            let reject = false;
            for (let dz = -1; dz <= 1 && !reject; dz++) for (let dy = -1; dy <= 1 && !reject; dy++) for (let dx = -1; dx <= 1; dx++) {
              const a = x + dx, b = y + dy, d = z + dz;
              if (a < 0 || b < 0 || d < 0 || a >= tx || b >= ty || d >= tz) continue;
              const j = a + tx * (b + ty * d);
              if ((tiles[j]! & 0x80000000) && !eligible[j]) {reject = true; break;}
            }
            guarded += +!reject;
          }
          const savedPerTile = 64 - (4 / c.width) ** 3;
          return {maxError_h: limit, eligibleFineTiles: n, eligibleCrossingTiles: surface, guardedFineTiles: guarded,
            hypotheticalOwnerReduction: n * savedPerTile / currentOwners,
            guardedOwnerReduction: guarded * savedPerTile / currentOwners};
        })};
    }),
  };
}
