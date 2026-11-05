/** Conservative pore-water operations. These do not modify NB-FLIP by
 * themselves: the grid adapter must commit the paired free-water changes. */
export interface PoreWaterExchangeRequest { readonly cell: number; readonly vertex: number; readonly volume_m3: number }
export interface PoreWaterExchangePlan {
  readonly transfers: readonly PoreWaterExchangeRequest[];
  readonly freeVolumeDelta_m3: Float64Array;
  readonly boundVolumeDelta_m3: Float64Array;
  readonly transferredVolume_m3: number;
}

function volumes(values: ArrayLike<number>, name: string): void {
  for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i]) || values[i]! < 0) throw new Error(`Invalid ${name} at ${i}`);
}

/** Resolve all competing capture requests before committing either reservoir.
 * Donor and receiver scaling never create water or overdraw a shared cell. */
export function planPoreAbsorption(freeVolume_m3: ArrayLike<number>, boundVolume_m3: ArrayLike<number>,
  capacity_m3: ArrayLike<number>, requests: readonly PoreWaterExchangeRequest[]): PoreWaterExchangePlan {
  volumes(freeVolume_m3, 'free volume'); volumes(boundVolume_m3, 'bound volume'); volumes(capacity_m3, 'pore capacity');
  if (boundVolume_m3.length !== capacity_m3.length) throw new Error('Pore reservoir lengths differ');
  const donorDemand = new Float64Array(freeVolume_m3.length), receiverDemand = new Float64Array(boundVolume_m3.length);
  for (const request of requests) {
    if (!Number.isInteger(request.cell) || request.cell < 0 || request.cell >= donorDemand.length
      || !Number.isInteger(request.vertex) || request.vertex < 0 || request.vertex >= receiverDemand.length
      || !Number.isFinite(request.volume_m3) || request.volume_m3 < 0) throw new Error('Invalid pore capture request');
    donorDemand[request.cell]! += request.volume_m3;
  }
  const donorLimited = requests.map(request => ({ ...request, volume_m3: request.volume_m3
    * Math.min(1, freeVolume_m3[request.cell]! / Math.max(donorDemand[request.cell]!, Number.MIN_VALUE)) }));
  for (const request of donorLimited) receiverDemand[request.vertex]! += request.volume_m3;
  const freeVolumeDelta_m3 = new Float64Array(freeVolume_m3.length), boundVolumeDelta_m3 = new Float64Array(boundVolume_m3.length);
  const transfers = donorLimited.map(request => {
    const room = Math.max(0, capacity_m3[request.vertex]! - boundVolume_m3[request.vertex]!);
    const volume_m3 = request.volume_m3 * Math.min(1, room / Math.max(receiverDemand[request.vertex]!, Number.MIN_VALUE));
    freeVolumeDelta_m3[request.cell]! -= volume_m3; boundVolumeDelta_m3[request.vertex]! += volume_m3;
    return { ...request, volume_m3 };
  });
  return { transfers, freeVolumeDelta_m3, boundVolumeDelta_m3,
    transferredVolume_m3: transfers.reduce((sum, transfer) => sum + transfer.volume_m3, 0) };
}

export interface PoreDiffusionEdge { readonly a: number; readonly b: number; readonly conductance_m3_s: number }
export interface PoreDiffusionResult { readonly volume_m3: Float64Array; readonly iterations: number; readonly residual_m3: number }

/** Backward Euler on a graph finite-volume manifold. Positive conductances
 * give an M-matrix, so wet fronts spread without negative saturation.
 * The caller constructs conductance from material permeability and geometry. */
export function diffusePoreWater(water_m3: ArrayLike<number>, capacity_m3: ArrayLike<number>, edges: readonly PoreDiffusionEdge[],
  dt_s: number, relativeTolerance = 1e-10, maximumIterations = 2000): PoreDiffusionResult {
  volumes(water_m3, 'water volume'); volumes(capacity_m3, 'capacity');
  const n = water_m3.length;
  if (capacity_m3.length !== n || !Number.isFinite(dt_s) || dt_s < 0 || !(relativeTolerance > 0) || maximumIterations < 1) throw new Error('Invalid pore diffusion inputs');
  const diagonal = Float64Array.from(capacity_m3), x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (!(capacity_m3[i]! > 0)) throw new Error('Diffusion requires positive pore capacities; emit compressed overflow first');
    x[i] = water_m3[i]! / capacity_m3[i]!;
  }
  for (const e of edges) {
    if (!Number.isInteger(e.a) || !Number.isInteger(e.b) || e.a < 0 || e.b < 0 || e.a >= n || e.b >= n || e.a === e.b
      || !Number.isFinite(e.conductance_m3_s) || e.conductance_m3_s < 0) throw new Error('Invalid pore diffusion edge');
    diagonal[e.a]! += dt_s * e.conductance_m3_s; diagonal[e.b]! += dt_s * e.conductance_m3_s;
  }
  const apply = (input: Float64Array, output: Float64Array) => {
    for (let i = 0; i < n; i++) output[i] = capacity_m3[i]! * input[i]!;
    for (const e of edges) {
      const flux = dt_s * e.conductance_m3_s * (input[e.a]! - input[e.b]!);
      output[e.a]! += flux; output[e.b]! -= flux;
    }
  };
  const r = new Float64Array(n), z = new Float64Array(n), direction = new Float64Array(n), product = new Float64Array(n);
  apply(x, product);
  let rz = 0, total = 0, residual_m3 = 0;
  for (let i = 0; i < n; i++) {
    r[i] = water_m3[i]! - product[i]!; z[i] = r[i]! / diagonal[i]!; direction[i] = z[i]!;
    rz += r[i]! * z[i]!; total += water_m3[i]!; residual_m3 = Math.max(residual_m3, Math.abs(r[i]!));
  }
  const tolerance = relativeTolerance * Math.max(total, 1e-30);
  let iterations = 0;
  while (residual_m3 > tolerance && iterations < maximumIterations) {
    apply(direction, product); let denominator = 0;
    for (let i = 0; i < n; i++) denominator += direction[i]! * product[i]!;
    if (!(denominator > 0)) throw new Error('Pore diffusion lost positive definiteness');
    const alpha = rz / denominator; let nextRz = 0; residual_m3 = 0;
    for (let i = 0; i < n; i++) {
      x[i]! += alpha * direction[i]!; r[i]! -= alpha * product[i]!;
      z[i] = r[i]! / diagonal[i]!; nextRz += r[i]! * z[i]!; residual_m3 = Math.max(residual_m3, Math.abs(r[i]!));
    }
    const beta = nextRz / rz;
    for (let i = 0; i < n; i++) direction[i] = z[i]! + beta * direction[i]!;
    rz = nextRz; iterations++;
  }
  if (residual_m3 > tolerance) throw new Error(`Pore diffusion did not converge: ${residual_m3} m3`);
  const volume_m3 = Float64Array.from(x, (saturation, i) => saturation * capacity_m3[i]!);
  for (const value of volume_m3) if (value < -tolerance) throw new Error('Pore diffusion produced negative water');
  return { volume_m3, iterations, residual_m3 };
}

/** Keep fractional overflow stored until the fluid adapter can represent it;
 * returning it here is a plan only. Commit after emission is accepted. */
export function poreOverflow(water_m3: ArrayLike<number>, capacity_m3: ArrayLike<number>): Float64Array {
  volumes(water_m3, 'water volume'); volumes(capacity_m3, 'capacity');
  if (water_m3.length !== capacity_m3.length) throw new Error('Pore reservoir lengths differ');
  return Float64Array.from(water_m3, (water, i) => Math.max(0, water - capacity_m3[i]!));
}
