/** Deterministic subcell quadrature. Positions are in tank-local metres. */
export function seedLiquidParticles(phi: Float32Array, solid: Uint32Array,
  dimensions: readonly [number, number, number], h: readonly [number, number, number], capacity: number, stride = 8, label = "Particle method", samplesPerAxis = 2) {
  const [nx, ny, nz] = dimensions;
  const inside = (x: number, y: number, z: number, fx: number, fy: number, fz: number) => {
    let value = 0;
    for (let k = 0; k < 8; k++) {
      const dx = k & 1, dy = (k >> 1) & 1, dz = k >> 2;
      value += phi[x + dx + (nx + 1) * (y + dy + (ny + 1) * (z + dz))] *
        (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
    }
    return value < 0;
  };
  const visit = (write?: Float32Array) => {
    let count = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      if (solid[x + nx * (y + ny * z)]) continue;
      for (let k = 0; k < samplesPerAxis ** 3; k++) {
        const fx = (k % samplesPerAxis + 0.5) / samplesPerAxis, fy = (Math.floor(k / samplesPerAxis) % samplesPerAxis + 0.5) / samplesPerAxis, fz = (Math.floor(k / samplesPerAxis ** 2) + 0.5) / samplesPerAxis;
        if (!inside(x, y, z, fx, fy, fz)) continue;
        if (++count > capacity) throw new Error(`${label} needs more than ${capacity.toLocaleString()} particles; increase the scene cell size.`);
        if (write) write.set([(x + fx) * h[0], (y + fy) * h[1], (z + fz) * h[2], h[0] * h[1] * h[2] / samplesPerAxis ** 3], (count - 1) * stride);
      }
    }
    return count;
  };
  const count = visit(), data = new Float32Array(Math.max(1, count) * stride);
  visit(data);
  return { count, data, volume_m3: count * h[0] * h[1] * h[2] / samplesPerAxis ** 3 };
}
