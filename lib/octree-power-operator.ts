/** CPU contract for generalized power faces and compact Chebyshev rows. */

import {
  constructOctreePowerCell,
  type OctreePowerSite,
  type PowerBoundaryPlane,
  type PowerFaceGeometry,
  type PowerVec3,
} from "./octree-power-geometry";

export const OCTREE_POWER_INVALID_ROW = 0xffff_ffff;
export const OCTREE_POWER_ROW_METRIC_BYTES = 16;
export const OCTREE_POWER_FACE_RECORD_BYTES = 32;

export interface OctreePowerFaceRecord {
  readonly id: number;
  readonly negativeRow: number;
  readonly positiveRow: number;
  readonly key: string;
  readonly centroid: PowerVec3;
  readonly normal: PowerVec3;
  readonly area: number;
  readonly inverseDistance: number;
  readonly openFraction: number;
  readonly normalVelocity: number;
  readonly boundaryKey?: string;
}

export interface OctreePowerIncidence {
  readonly face: number;
  /** +1 means the stored normal points out of this row. */
  readonly sign: 1 | -1;
}

export interface OctreePowerCompactEntry {
  readonly row: number;
  readonly coefficient: number;
}

export interface OctreePowerCompactRow {
  readonly row: number;
  readonly siteKey: string;
  readonly diagonal: number;
  readonly rhs: number;
  readonly volume: number;
  readonly entries: readonly OctreePowerCompactEntry[];
}

export interface OctreePowerOperator {
  readonly sites: readonly OctreePowerSite[];
  readonly faces: readonly OctreePowerFaceRecord[];
  readonly incidence: readonly (readonly OctreePowerIncidence[])[];
  readonly rows: readonly OctreePowerCompactRow[];
  readonly maximumIncidence: number;
  readonly maximumNeighborRows: number;
}

export interface OctreePowerOperatorOptions {
  readonly openFraction?: (negative: OctreePowerSite, positive: OctreePowerSite | undefined, face: PowerFaceGeometry) => number;
  readonly normalVelocity?: (centroid: PowerVec3, normal: PowerVec3, negative: OctreePowerSite, positive?: OctreePowerSite) => number;
}

const distance = (a: PowerVec3, b: PowerVec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const vectorError = (a: PowerVec3, b: PowerVec3) => distance(a, b);
const oppositeError = (a: PowerVec3, b: PowerVec3) => Math.hypot(a[0] + b[0], a[1] + b[1], a[2] + b[2]);

/**
 * Builds one deterministic physical face sequence, equal/opposite incidence,
 * and the exact positive neighbor coefficients consumed by iterateChebyshev.
 */
export function buildOctreePowerOperator(
  inputSites: readonly OctreePowerSite[],
  boundaries: readonly PowerBoundaryPlane[] = [],
  options: OctreePowerOperatorOptions = {},
): OctreePowerOperator {
  if (inputSites.length === 0) throw new RangeError("Power operator requires at least one site");
  const sites = [...inputSites].sort((a, b) => a.key.localeCompare(b.key));
  const rowByKey = new Map(sites.map((site, row) => [site.key, row]));
  if (rowByKey.size !== sites.length) throw new RangeError("Power operator site keys must be unique");
  const cells = sites.map((site) => constructOctreePowerCell(site, sites, boundaries));
  const faces: OctreePowerFaceRecord[] = [];
  const incidence = sites.map(() => [] as OctreePowerIncidence[]);
  for (let negativeRow = 0; negativeRow < sites.length; negativeRow += 1) {
    for (const geometry of cells[negativeRow].faces) {
      const candidateRow = geometry.incidentSiteKey === undefined ? undefined : rowByKey.get(geometry.incidentSiteKey);
      if (geometry.kind === "site" && candidateRow === undefined) {
        throw new Error(`Power face ${geometry.key} references an unknown incident site`);
      }
      if (candidateRow !== undefined && candidateRow < negativeRow) continue;
      const positiveRow = candidateRow ?? OCTREE_POWER_INVALID_ROW;
      if (candidateRow !== undefined) {
        const reciprocal = cells[candidateRow].faces.find((face) => face.incidentSiteKey === sites[negativeRow].key);
        if (!reciprocal) throw new Error(`Power face ${sites[negativeRow].key}/${sites[candidateRow].key} has no reciprocal`);
        const metricScale = Math.max(1, sites[negativeRow].size, sites[candidateRow].size);
        if (Math.abs(geometry.area - reciprocal.area) > 2e-8 * metricScale * metricScale
          || vectorError(geometry.centroid, reciprocal.centroid) > 2e-8 * metricScale
          || oppositeError(geometry.normal, reciprocal.normal) > 2e-8) {
          throw new Error(`Power face ${sites[negativeRow].key}/${sites[candidateRow].key} is asymmetric`);
        }
      }
      const positive = candidateRow === undefined ? undefined : sites[candidateRow];
      const openFraction = options.openFraction?.(sites[negativeRow], positive, geometry) ?? 1;
      if (!Number.isFinite(openFraction) || openFraction < 0 || openFraction > 1) {
        throw new RangeError(`Power face ${geometry.key} open fraction must be in [0, 1]`);
      }
      const normalVelocity = options.normalVelocity?.(geometry.centroid, geometry.normal, sites[negativeRow], positive) ?? 0;
      if (!Number.isFinite(normalVelocity)) throw new RangeError(`Power face ${geometry.key} velocity must be finite`);
      const inverseDistance = positive ? 1 / distance(sites[negativeRow].center, positive.center) : 0;
      const id = faces.length;
      faces.push({
        id,
        negativeRow,
        positiveRow,
        key: positive ? `${sites[negativeRow].key}|${positive.key}` : `${sites[negativeRow].key}|boundary:${geometry.boundaryKey}`,
        centroid: geometry.centroid,
        normal: geometry.normal,
        area: geometry.area,
        inverseDistance,
        openFraction,
        normalVelocity,
        boundaryKey: geometry.boundaryKey,
      });
      incidence[negativeRow].push({ face: id, sign: 1 });
      if (positive) incidence[candidateRow!].push({ face: id, sign: -1 });
    }
  }
  for (const row of incidence) row.sort((a, b) => a.face - b.face);
  const rows = sites.map((site, row): OctreePowerCompactRow => {
    const coefficients = new Map<number, number>();
    let diagonal = 0, rhs = 0;
    for (const item of incidence[row]) {
      const face = faces[item.face];
      rhs += item.sign * face.area * face.normalVelocity;
      const neighbor = face.negativeRow === row ? face.positiveRow : face.negativeRow;
      if (neighbor === OCTREE_POWER_INVALID_ROW) continue;
      const coefficient = face.openFraction * face.area * face.inverseDistance;
      if (!(coefficient >= 0) || !Number.isFinite(coefficient)) throw new Error(`Power face ${face.key} has an invalid coefficient`);
      diagonal += coefficient;
      coefficients.set(neighbor, (coefficients.get(neighbor) ?? 0) + coefficient);
    }
    return {
      row,
      siteKey: site.key,
      diagonal,
      rhs,
      volume: cells[row].volume,
      entries: [...coefficients].sort((a, b) => a[0] - b[0]).map(([neighbor, coefficient]) => ({ row: neighbor, coefficient })),
    };
  });
  return {
    sites,
    faces,
    incidence,
    rows,
    maximumIncidence: Math.max(...incidence.map((row) => row.length)),
    maximumNeighborRows: Math.max(...rows.map((row) => row.entries.length)),
  };
}

/** Applies A = diagonal - positive neighbor coefficients. */
export function applyOctreePowerMatrix(operator: OctreePowerOperator, vector: readonly number[]): number[] {
  if (vector.length !== operator.rows.length || vector.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Power matrix vector must be finite and match the row count");
  }
  return operator.rows.map((row) => row.diagonal * vector[row.row]
    - row.entries.reduce((sum, entry) => sum + entry.coefficient * vector[entry.row], 0));
}

export function octreePowerMatrixEnergy(operator: OctreePowerOperator, vector: readonly number[]): number {
  const applied = applyOctreePowerMatrix(operator, vector);
  return vector.reduce((sum, value, row) => sum + value * applied[row], 0);
}

/** Ghost-fluid zero crossing measured along the power dual edge. */
export function octreePowerBoundaryDistance(
  liquidPhi: number,
  airPhi: number,
  dualDistance: number,
  minimumTheta = 0.01,
): number {
  if (![liquidPhi, airPhi, dualDistance, minimumTheta].every(Number.isFinite) || liquidPhi > 0 || airPhi < 0
    || dualDistance <= 0 || minimumTheta <= 0 || minimumTheta > 1) {
    throw new RangeError("Power free-surface crossing needs finite liquid/air phi, positive distance, and bounded theta");
  }
  const denominator = liquidPhi - airPhi;
  const theta = Math.max(minimumTheta, Math.min(1, Math.abs(denominator) > 1e-12 ? liquidPhi / denominator : minimumTheta));
  return theta * dualDistance;
}

export function projectOctreePowerFaceVelocities(
  operator: OctreePowerOperator,
  pressure: readonly number[],
  pressureScale = 1,
): number[] {
  if (pressure.length !== operator.rows.length || pressure.some((value) => !Number.isFinite(value)) || !Number.isFinite(pressureScale)) {
    throw new RangeError("Power projection pressure must be finite and match the row count");
  }
  return operator.faces.map((face) => face.positiveRow === OCTREE_POWER_INVALID_ROW ? face.normalVelocity
    : face.normalVelocity - pressureScale * (pressure[face.positiveRow] - pressure[face.negativeRow])
      * face.inverseDistance * face.openFraction);
}

export function octreePowerDivergence(
  operator: OctreePowerOperator,
  faceVelocities: readonly number[] = operator.faces.map((face) => face.normalVelocity),
  volumeNormalized = true,
): number[] {
  if (faceVelocities.length !== operator.faces.length || faceVelocities.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Power divergence velocity count must match the face store");
  }
  return operator.rows.map((row) => {
    const integrated = operator.incidence[row.row].reduce((sum, item) => sum
      + item.sign * operator.faces[item.face].area * faceVelocities[item.face], 0);
    return volumeNormalized ? integrated / row.volume : integrated;
  });
}

export interface PowerNormalSample {
  readonly normal: PowerVec3;
  readonly normalVelocity: number;
  readonly weight: number;
}

export interface PowerVelocityReconstruction {
  readonly velocity: PowerVec3;
  readonly determinant: number;
  readonly usedFallback: boolean;
}

/** Area-weighted least-squares fit of a Cartesian vector to general normals. */
export function reconstructPowerVelocity(
  samples: readonly PowerNormalSample[],
  fallback: PowerVec3 = [0, 0, 0],
  determinantTolerance = 1e-10,
): PowerVelocityReconstruction {
  const m = [0, 0, 0, 0, 0, 0]; // xx, xy, xz, yy, yz, zz
  const b = [0, 0, 0];
  for (const sample of samples) {
    const [x, y, z] = sample.normal;
    if (![x, y, z, sample.normalVelocity, sample.weight].every(Number.isFinite) || sample.weight < 0) {
      throw new RangeError("Power velocity samples must be finite with non-negative weights");
    }
    const w = sample.weight;
    m[0] += w * x * x; m[1] += w * x * y; m[2] += w * x * z;
    m[3] += w * y * y; m[4] += w * y * z; m[5] += w * z * z;
    b[0] += w * x * sample.normalVelocity;
    b[1] += w * y * sample.normalVelocity;
    b[2] += w * z * sample.normalVelocity;
  }
  const [xx, xy, xz, yy, yz, zz] = m;
  const c00 = yy * zz - yz * yz;
  const c01 = xz * yz - xy * zz;
  const c02 = xy * yz - xz * yy;
  const c11 = xx * zz - xz * xz;
  const c12 = xy * xz - xx * yz;
  const c22 = xx * yy - xy * xy;
  const determinant = xx * c00 + xy * c01 + xz * c02;
  const scale = Math.max(1, xx + yy + zz);
  if (!Number.isFinite(determinant) || determinant <= determinantTolerance * scale ** 3) {
    return { velocity: [...fallback], determinant, usedFallback: true };
  }
  return {
    velocity: [
      (c00 * b[0] + c01 * b[1] + c02 * b[2]) / determinant,
      (c01 * b[0] + c11 * b[1] + c12 * b[2]) / determinant,
      (c02 * b[0] + c12 * b[1] + c22 * b[2]) / determinant,
    ],
    determinant,
    usedFallback: false,
  };
}

export interface OctreePowerStoragePlan {
  readonly rowMetricBytes: number;
  readonly faceBytes: number;
  readonly incidenceOffsetBytes: number;
  readonly incidenceEntryBytes: number;
  readonly incidenceBytes: number;
  readonly maximumIncidencePerRow: number;
  readonly allocatedBytes: number;
}

/**
 * Persistent allocation is compact-capacity-scaled; no domain volume enters.
 * Incidence uses deterministic CSR offsets plus at most two entries per
 * physical face. The proven per-row bound remains a fail-closed validation
 * constant, not a wasteful fixed slab multiplied by every row.
 */
export function planOctreePowerStorage(rowCapacity: number, faceCapacity: number, provenIncidencePerRow: number): OctreePowerStoragePlan {
  for (const [label, value] of [["row", rowCapacity], ["face", faceCapacity], ["incidence", provenIncidencePerRow]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Power ${label} capacity must be a positive integer`);
  }
  const rowMetricBytes = rowCapacity * OCTREE_POWER_ROW_METRIC_BYTES;
  const faceBytes = faceCapacity * OCTREE_POWER_FACE_RECORD_BYTES;
  const incidenceOffsetBytes = (rowCapacity + 1) * 4;
  const incidenceEntryBytes = faceCapacity * 2 * 4;
  const incidenceBytes = incidenceOffsetBytes + incidenceEntryBytes;
  return {
    rowMetricBytes, faceBytes, incidenceOffsetBytes, incidenceEntryBytes, incidenceBytes,
    maximumIncidencePerRow: provenIncidencePerRow,
    allocatedBytes: rowMetricBytes + faceBytes + incidenceBytes,
  };
}
