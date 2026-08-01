import type { Vec3 } from "./model";

const index3 = (x: number, y: number, z: number, nx: number, ny: number) => x + nx * (y + ny * z);
const clamp = (value: number, lower: number, upper: number) => Math.max(lower, Math.min(upper, value));

function squaredDistanceTransform1D(input: Float64Array, spacing: number) {
  const n = input.length, output = new Float64Array(n), sites = new Int32Array(n), boundaries = new Float64Array(n + 1);
  let first = -1;
  for (let index = 0; index < n; index += 1) if (Number.isFinite(input[index])) { first = index; break; }
  if (first < 0) { output.fill(Infinity); return output; }
  const weight = spacing * spacing; let count = 0; sites[0] = first; boundaries[0] = -Infinity; boundaries[1] = Infinity;
  for (let q = first + 1; q < n; q += 1) {
    if (!Number.isFinite(input[q])) continue;
    let intersection = 0;
    for (;;) {
      const p = sites[count]; intersection = ((input[q] + weight * q * q) - (input[p] + weight * p * p)) / (2 * weight * (q - p));
      if (intersection > boundaries[count] || count === 0) break;
      count -= 1;
    }
    count += 1; sites[count] = q; boundaries[count] = intersection; boundaries[count + 1] = Infinity;
  }
  let site = 0;
  for (let q = 0; q < n; q += 1) {
    while (boundaries[site + 1] < q) site += 1;
    const delta = q - sites[site]; output[q] = input[sites[site]] + weight * delta * delta;
  }
  return output;
}

export function signedDistanceFromVolume(volume: ArrayLike<number>, nx: number, ny: number, nz: number, h: Vec3) {
  const count = nx * ny * nz;
  if (volume.length !== count) throw new Error("Invalid VOF field for signed-distance reconstruction");
  let squared = new Float64Array(count); squared.fill(Infinity);
  const interfaceOffsets = new Float32Array(count); interfaceOffsets.fill(NaN);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny), alpha = clamp(volume[index], 0, 1);
    if (alpha > 1e-6 && alpha < 1 - 1e-6) { squared[index] = 0; interfaceOffsets[index] = Math.abs(0.5 - alpha) * 2 * Math.min(h.x, h.y, h.z); }
    for (const [dx, dy, dz, spacing] of [[1, 0, 0, h.x], [0, 1, 0, h.y], [0, 0, 1, h.z]] as const) {
      const qx = x + dx, qy = y + dy, qz = z + dz;
      if (qx >= nx || qy >= ny || qz >= nz) continue;
      const neighbor = index3(qx, qy, qz, nx, ny);
      if ((alpha >= 0.5) !== (volume[neighbor] >= 0.5)) {
        squared[index] = 0; squared[neighbor] = 0;
        interfaceOffsets[index] = Math.min(Number.isFinite(interfaceOffsets[index]) ? interfaceOffsets[index] : Infinity, 0.5 * spacing);
        interfaceOffsets[neighbor] = Math.min(Number.isFinite(interfaceOffsets[neighbor]) ? interfaceOffsets[neighbor] : Infinity, 0.5 * spacing);
      }
    }
  }
  const transformAxis = (axis: 0 | 1 | 2, length: number, spacing: number) => {
    const next = new Float64Array(count), line = new Float64Array(length);
    const outerA = axis === 0 ? nz : nx, outerB = axis === 1 ? nz : ny;
    for (let a = 0; a < outerA; a += 1) for (let b = 0; b < outerB; b += 1) {
      for (let q = 0; q < length; q += 1) {
        const x = axis === 0 ? q : a, y = axis === 1 ? q : b, z = axis === 2 ? q : axis === 0 ? a : b;
        line[q] = squared[index3(x, y, z, nx, ny)];
      }
      const transformed = squaredDistanceTransform1D(line, spacing);
      for (let q = 0; q < length; q += 1) {
        const x = axis === 0 ? q : a, y = axis === 1 ? q : b, z = axis === 2 ? q : axis === 0 ? a : b;
        next[index3(x, y, z, nx, ny)] = transformed[q];
      }
    }
    squared = next;
  };
  transformAxis(0, nx, h.x); transformAxis(1, ny, h.y); transformAxis(2, nz, h.z);
  const phi = new Float32Array(count);
  const halfFinest = 0.5 * Math.min(h.x, h.y, h.z);
  for (let index = 0; index < count; index += 1) {
    const distance = Number.isFinite(interfaceOffsets[index]) ? interfaceOffsets[index] : Math.sqrt(squared[index]) + halfFinest;
    phi[index] = (volume[index] >= 0.5 ? -1 : 1) * distance;
  }
  return phi;
}
