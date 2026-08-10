export interface AdaptiveSurfaceDiagnosticSnapshot {
  readonly graphControl: ArrayLike<number>;
  readonly phiControl: ArrayLike<number>;
  readonly leaves: ArrayLike<number>;
  readonly nodalPhi: ArrayLike<number>;
  readonly nodes: ArrayLike<number>;
  readonly constraints: ArrayLike<number>;
  readonly renderer: ArrayLike<number>;
  readonly dimensions: readonly [number, number, number];
}

export interface AdaptiveSurfacePublicationAnalysis {
  readonly leafCount: number;
  readonly nodeCount: number;
  readonly interfaceLeafCountsBySize: Readonly<Record<string, number>>;
  readonly coarseInterfaceLeafCount: number;
  readonly constrainedNodeCount: number;
  readonly maximumStoredConstraintError: number;
  readonly rendererCornerCount: number;
  readonly maximumRendererCornerError: number;
  readonly analyticNodeErrorsByLeafSize?: Readonly<Record<string, Readonly<{
    count: number; maximumAbsoluteError: number; rootMeanSquareError: number;
  }>>>;
}

export interface AdaptiveSurfaceFeatureProbeGroup {
  readonly name: string;
  readonly points: readonly (readonly [x: number, z: number])[];
}

export interface AdaptiveSurfaceFeatureGeometry {
  readonly coveredNodalSamples: number;
  readonly missingNodalSamples: number;
  readonly maximumSharedNodeMismatch: number;
  readonly activeCubeCount: number;
  readonly zeroSetExtentsCells?: Readonly<{
    minimum: readonly [number, number, number];
    maximum: readonly [number, number, number];
  }>;
  readonly topFeatures: Readonly<Record<string, Readonly<{
    sampleCount: number;
    phiAtReferenceTop: Readonly<{ minimum: number; maximum: number; mean: number }>;
    surfaceHeightCells: Readonly<{ minimum: number; maximum: number; mean: number }>;
    meanRetreatFromReferenceTopCells: number;
  }>>>;
}

const floatBits = new Uint32Array(1);
const floatValue = new Float32Array(floatBits.buffer);
function fromBits(bits: number): number {
  floatBits[0] = bits >>> 0;
  return floatValue[0]!;
}

function constrainedValue(snapshot: AdaptiveSurfaceDiagnosticSnapshot,
  slot: number, bank: number): number {
  const count = snapshot.constraints[12 * slot + 1] ?? 0;
  if (count === 0) return fromBits(snapshot.nodalPhi[2 * slot + bank] ?? 0);
  const denominator = snapshot.constraints[12 * slot + 2] ?? 0;
  if ((count !== 2 && count !== 4) || denominator === 0) return Number.NaN;
  let value = 0;
  for (let term = 0; term < count; term += 1) {
    const master = snapshot.constraints[12 * slot + 4 + term] ?? 0xffff_ffff;
    const numerator = snapshot.constraints[12 * slot + 8 + term] ?? 0;
    value += numerator * fromBits(snapshot.nodalPhi[2 * master + bank] ?? 0);
  }
  return value / denominator;
}

/** Reconstruct the continuous adaptive field at finest-lattice nodes and
 * measure visible feature motion without invoking the renderer. Shared leaf
 * boundaries are written by both sides; their mismatch is retained as a
 * direct continuity diagnostic instead of silently choosing one side. */
export function analyzeAdaptiveSurfaceFeatureGeometry(
  snapshot: AdaptiveSurfaceDiagnosticSnapshot,
  referenceTopY: number,
  groups: readonly AdaptiveSurfaceFeatureProbeGroup[],
): AdaptiveSurfaceFeatureGeometry {
  const [nx, ny, nz] = snapshot.dimensions;
  const dx = nx + 1, dy = ny + 1, dz = nz + 1;
  const lattice = new Float32Array(dx * dy * dz);
  lattice.fill(Number.NaN);
  const index = (x: number, y: number, z: number) => x + dx * (y + dy * z);
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0,
    Math.floor(snapshot.leaves.length / 16));
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodes.length / 4), Math.floor(snapshot.nodalPhi.length / 2),
    Math.floor(snapshot.constraints.length / 12));
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  let maximumSharedNodeMismatch = 0;
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const ox = snapshot.leaves[16 * leaf] ?? 0;
    const oy = snapshot.leaves[16 * leaf + 1] ?? 0;
    const oz = snapshot.leaves[16 * leaf + 2] ?? 0;
    const span = snapshot.leaves[16 * leaf + 3] ?? 0;
    if (span === 0 || ox + span > nx || oy + span > ny || oz + span > nz) continue;
    const corners = new Array<number>(8);
    let valid = true;
    for (let corner = 0; corner < 8; corner += 1) {
      const slot = snapshot.leaves[16 * leaf + 8 + corner] ?? 0xffff_ffff;
      const value = slot < nodeCount ? constrainedValue(snapshot, slot, bank) : Number.NaN;
      corners[corner] = value;
      valid &&= Number.isFinite(value);
    }
    if (!valid) continue;
    for (let z = 0; z <= span; z += 1) for (let y = 0; y <= span; y += 1)
      for (let x = 0; x <= span; x += 1) {
        const tx = x / span, ty = y / span, tz = z / span;
        let value = 0;
        for (let corner = 0; corner < 8; corner += 1) {
          const weight = (corner & 1 ? tx : 1 - tx)
            * (corner & 2 ? ty : 1 - ty) * (corner & 4 ? tz : 1 - tz);
          value += weight * corners[corner]!;
        }
        const at = index(ox + x, oy + y, oz + z);
        if (Number.isFinite(lattice[at])) {
          maximumSharedNodeMismatch = Math.max(maximumSharedNodeMismatch,
            Math.abs(lattice[at]! - value));
        } else lattice[at] = value;
      }
  }
  let coveredNodalSamples = 0;
  for (const value of lattice) coveredNodalSamples += Number(Number.isFinite(value));

  let activeCubeCount = 0;
  const minimum = [nx, ny, nz] as [number, number, number];
  const maximum = [0, 0, 0] as [number, number, number];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) {
      let lo = Number.POSITIVE_INFINITY, hi = Number.NEGATIVE_INFINITY;
      for (let corner = 0; corner < 8; corner += 1) {
        const value = lattice[index(x + (corner & 1), y + ((corner >> 1) & 1),
          z + ((corner >> 2) & 1))]!;
        lo = Math.min(lo, value); hi = Math.max(hi, value);
      }
      if (!(lo <= 0 && hi >= 0)) continue;
      activeCubeCount += 1;
      minimum[0] = Math.min(minimum[0], x); minimum[1] = Math.min(minimum[1], y);
      minimum[2] = Math.min(minimum[2], z);
      maximum[0] = Math.max(maximum[0], x + 1); maximum[1] = Math.max(maximum[1], y + 1);
      maximum[2] = Math.max(maximum[2], z + 1);
    }

  const summarize = (values: readonly number[]) => Object.freeze({
    minimum: Math.min(...values), maximum: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
  });
  const topFeatures: Record<string, {
    sampleCount: number;
    phiAtReferenceTop: Readonly<{ minimum: number; maximum: number; mean: number }>;
    surfaceHeightCells: Readonly<{ minimum: number; maximum: number; mean: number }>;
    meanRetreatFromReferenceTopCells: number;
  }> = {};
  for (const group of groups) {
    const phi: number[] = [], heights: number[] = [];
    for (const [x, z] of group.points) {
      if (!Number.isInteger(x) || !Number.isInteger(z) || x < 0 || x > nx
          || z < 0 || z > nz || referenceTopY < 0 || referenceTopY > ny) continue;
      const top = lattice[index(x, referenceTopY, z)]!;
      if (Number.isFinite(top)) phi.push(top);
      let height = Number.NaN;
      for (let y = 0; y < ny; y += 1) {
        const a = lattice[index(x, y, z)]!, b = lattice[index(x, y + 1, z)]!;
        if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
        if (a === 0) height = y;
        if (b === 0) height = y + 1;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) height = y - a / (b - a);
      }
      if (Number.isFinite(height)) heights.push(height);
    }
    if (phi.length === 0 || heights.length === 0) continue;
    const heightSummary = summarize(heights);
    topFeatures[group.name] = { sampleCount: Math.min(phi.length, heights.length),
      phiAtReferenceTop: summarize(phi), surfaceHeightCells: heightSummary,
      meanRetreatFromReferenceTopCells: referenceTopY - heightSummary.mean };
  }
  return Object.freeze({ coveredNodalSamples,
    missingNodalSamples: lattice.length - coveredNodalSamples,
    maximumSharedNodeMismatch, activeCubeCount,
    ...(activeCubeCount > 0 ? { zeroSetExtentsCells: Object.freeze({
      minimum: Object.freeze(minimum), maximum: Object.freeze(maximum),
    }) } : {}), topFeatures: Object.freeze(topFeatures) });
}

/** CPU audit of the accepted factor-one graph and its one-way renderer view.
 * It reads only explicit diagnostic snapshots; recurring simulation and
 * presentation code do not call it. */
export function analyzeAdaptiveSurfacePublication(
  snapshot: AdaptiveSurfaceDiagnosticSnapshot,
  analyticPhiAtNode?: (x: number, y: number, z: number) => number,
): AdaptiveSurfacePublicationAnalysis {
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0,
    Math.floor(snapshot.leaves.length / 16));
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodes.length / 4), Math.floor(snapshot.nodalPhi.length / 2),
    Math.floor(snapshot.constraints.length / 12));
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  let constrainedNodeCount = 0;
  let maximumStoredConstraintError = 0;
  for (let node = 0; node < nodeCount; node += 1) {
    if ((snapshot.constraints[12 * node + 1] ?? 0) === 0) continue;
    constrainedNodeCount += 1;
    maximumStoredConstraintError = Math.max(maximumStoredConstraintError,
      Math.abs(fromBits(snapshot.nodalPhi[2 * node + bank] ?? 0)
        - constrainedValue(snapshot, node, bank)));
  }

  const interfaceCounts = new Map<number, number>();
  const analyticBySize = new Map<number, { count: number; maximum: number; squares: number }>();
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const span = snapshot.leaves[16 * leaf + 3] ?? 0;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    for (let corner = 0; corner < 8; corner += 1) {
      const slot = snapshot.leaves[16 * leaf + 8 + corner] ?? 0xffff_ffff;
      const value = slot < nodeCount ? constrainedValue(snapshot, slot, bank) : Number.NaN;
      minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
      if (!analyticPhiAtNode || slot >= nodeCount) continue;
      const item = snapshot.nodes[4 * slot] ?? 0;
      const dx = snapshot.dimensions[0] + 1;
      const dy = snapshot.dimensions[1] + 1;
      const z = Math.floor(item / (dx * dy));
      const remainder = item - z * dx * dy;
      const y = Math.floor(remainder / dx);
      const x = remainder - y * dx;
      const error = Math.abs(value - analyticPhiAtNode(x, y, z));
      const group = analyticBySize.get(span) ?? { count: 0, maximum: 0, squares: 0 };
      group.count += 1; group.maximum = Math.max(group.maximum, error);
      group.squares += error * error; analyticBySize.set(span, group);
    }
    if (minimum <= 0 && maximum >= 0) {
      interfaceCounts.set(span, (interfaceCounts.get(span) ?? 0) + 1);
    }
  }

  const rendererRows = Math.min(snapshot.renderer[2] ?? 0, leafCount);
  let rendererCornerCount = 0;
  let maximumRendererCornerError = 0;
  const rendererAux = 8 + 8 * rendererRows;
  for (let row = 0; row < rendererRows; row += 1) {
    const leaf = snapshot.renderer[8 + 8 * row + 6] ?? 0xffff_ffff;
    if (leaf >= leafCount) continue;
    for (let corner = 0; corner < 8; corner += 1) {
      const slot = snapshot.leaves[16 * leaf + 8 + corner] ?? 0xffff_ffff;
      if (slot >= nodeCount) continue;
      rendererCornerCount += 1;
      maximumRendererCornerError = Math.max(maximumRendererCornerError,
        Math.abs(fromBits(snapshot.renderer[rendererAux + 8 * row + corner] ?? 0)
          - constrainedValue(snapshot, slot, bank)));
    }
  }
  const interfaceLeafCountsBySize = Object.fromEntries([...interfaceCounts]
    .sort(([a], [b]) => a - b).map(([size, count]) => [String(size), count]));
  const analyticNodeErrorsByLeafSize = analyticPhiAtNode
    ? Object.fromEntries([...analyticBySize].sort(([a], [b]) => a - b)
      .map(([size, group]) => [String(size), Object.freeze({ count: group.count,
        maximumAbsoluteError: group.maximum,
        rootMeanSquareError: Math.sqrt(group.squares / Math.max(1, group.count)) })]))
    : undefined;
  return Object.freeze({ leafCount, nodeCount, interfaceLeafCountsBySize,
    coarseInterfaceLeafCount: [...interfaceCounts].reduce((sum, [size, count]) =>
      sum + (size > 1 ? count : 0), 0),
    constrainedNodeCount, maximumStoredConstraintError, rendererCornerCount,
    maximumRendererCornerError,
    ...(analyticNodeErrorsByLeafSize ? { analyticNodeErrorsByLeafSize } : {}) });
}
