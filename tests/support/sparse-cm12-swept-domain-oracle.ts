/** Shared 2-D/3-D analytic fixture for sparse material-page membership. */
export const SWEPT_DOMAIN_PAGE_FINE_SIZE = 8;
export const SWEPT_DOMAIN_FRAME_DT = 1 / 60;
export const SWEPT_DOMAIN_PLANE_PAGES = 6;

export type SweptDomainVec2 = readonly [number, number];
export type SweptDomainPage2 = readonly [number, number];
export type SweptDomainVec3 = readonly [number, number, number];
export type SweptDomainPage3 = readonly [number, number, number];

export interface SweptDomainDonor2 {
  /** Half-open material rectangle in finest-cell coordinates. */
  readonly minimumFine: SweptDomainVec2;
  readonly maximumExclusiveFine: SweptDomainVec2;
  /** Whole-frame translation in finest cells. */
  readonly displacementFine: SweptDomainVec2;
}

export interface SweptDomainDonor3 {
  /** Half-open material box in finest-cell coordinates. */
  readonly minimumFine: SweptDomainVec3;
  readonly maximumExclusiveFine: SweptDomainVec3;
  /** Whole-frame translation in finest cells. */
  readonly displacementFine: SweptDomainVec3;
}

export interface SweptDomainOracleCase {
  readonly id: string;
  readonly donors: readonly SweptDomainDonor2[];
  /** Independently reviewed positive-measure Minkowski-sweep page set. */
  readonly expectedPages2D: readonly SweptDomainPage2[];
}

export interface SweptDomainOracleCase3D {
  readonly id: string;
  readonly donors: readonly SweptDomainDonor3[];
  /** Independently reviewed positive-measure Minkowski-sweep page set. */
  readonly expectedPages3D: readonly SweptDomainPage3[];
}

const pageBlock = (pageX: number, pageY: number,
  displacementFine: SweptDomainVec2): SweptDomainDonor2 => ({
  minimumFine: [SWEPT_DOMAIN_PAGE_FINE_SIZE * pageX,
    SWEPT_DOMAIN_PAGE_FINE_SIZE * pageY],
  maximumExclusiveFine: [SWEPT_DOMAIN_PAGE_FINE_SIZE * (pageX + 1),
    SWEPT_DOMAIN_PAGE_FINE_SIZE * (pageY + 1)],
  displacementFine,
});

const pageBlock3D = (pageX: number, pageY: number, pageZ: number,
  displacementFine: SweptDomainVec3): SweptDomainDonor3 => ({
  minimumFine: [SWEPT_DOMAIN_PAGE_FINE_SIZE * pageX,
    SWEPT_DOMAIN_PAGE_FINE_SIZE * pageY,
    SWEPT_DOMAIN_PAGE_FINE_SIZE * pageZ],
  maximumExclusiveFine: [SWEPT_DOMAIN_PAGE_FINE_SIZE * (pageX + 1),
    SWEPT_DOMAIN_PAGE_FINE_SIZE * (pageY + 1),
    SWEPT_DOMAIN_PAGE_FINE_SIZE * (pageZ + 1)],
  displacementFine,
});

export const SWEPT_DOMAIN_ORACLE_CASES: readonly SweptDomainOracleCase[] = [
  { id: "static", donors: [pageBlock(1, 1, [0, 0])], expectedPages2D: [[1, 1]] },
  { id: "axis-2.5", donors: [pageBlock(1, 1, [2.5, 0])],
    expectedPages2D: [[1, 1], [2, 1]] },
  { id: "axis-10", donors: [pageBlock(1, 1, [10, 0])],
    expectedPages2D: [[1, 1], [2, 1], [3, 1]] },
  { id: "axis-18", donors: [pageBlock(1, 1, [18, 0])],
    expectedPages2D: [[1, 1], [2, 1], [3, 1], [4, 1]] },
  { id: "diagonal-2.5", donors: [pageBlock(1, 1, [2.5, 2.5])],
    expectedPages2D: [[1, 1], [2, 1], [1, 2], [2, 2]] },
  { id: "diagonal-10", donors: [pageBlock(1, 1, [10, 10])], expectedPages2D: [
    [1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [2, 3], [3, 3],
  ] },
  { id: "diagonal-18", donors: [pageBlock(1, 1, [18, 18])], expectedPages2D: [
    [1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [2, 3], [3, 3],
    [4, 3], [3, 4], [4, 4],
  ] },
  { id: "disconnected", donors: [pageBlock(1, 1, [10, 0]), pageBlock(4, 4, [-10, 0])],
    expectedPages2D: [[1, 1], [2, 1], [3, 1], [2, 4], [3, 4], [4, 4]] },
] as const;

export const SWEPT_DOMAIN_ORACLE_CASES_3D: readonly SweptDomainOracleCase3D[] = [
  { id: "all-axis-2.5", donors: [pageBlock3D(1, 1, 1, [2.5, 2.5, 2.5])],
    expectedPages3D: [
      [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
      [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2],
    ] },
  { id: "all-axis-10", donors: [pageBlock3D(1, 1, 1, [10, 10, 10])],
    expectedPages3D: [
      [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
      [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2],
      [3, 2, 2], [2, 3, 2], [3, 3, 2],
      [2, 2, 3], [3, 2, 3], [2, 3, 3], [3, 3, 3],
    ] },
  { id: "all-axis-18", donors: [pageBlock3D(1, 1, 1, [18, 18, 18])],
    expectedPages3D: [
      [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
      [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2],
      [3, 2, 2], [2, 3, 2], [3, 3, 2],
      [2, 2, 3], [3, 2, 3], [2, 3, 3], [3, 3, 3],
      [4, 3, 3], [3, 4, 3], [4, 4, 3],
      [3, 3, 4], [4, 3, 4], [3, 4, 4], [4, 4, 4],
    ] },
] as const;

function overlapTimeInterval(donorMinimum: number, donorMaximum: number,
  displacement: number, pageMinimum: number, pageMaximum: number): readonly [number, number] | undefined {
  if (displacement === 0) {
    return donorMinimum < pageMaximum && donorMaximum > pageMinimum
      ? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY] : undefined;
  }
  return displacement > 0
    ? [(pageMinimum - donorMaximum) / displacement,
      (pageMaximum - donorMinimum) / displacement]
    : [(pageMaximum - donorMinimum) / displacement,
      (pageMinimum - donorMaximum) / displacement];
}

function donorPageRange(donorMinimum: number, donorMaximum: number,
  displacement: number): readonly [number, number] {
  const sweptMinimum = Math.min(donorMinimum, donorMinimum + displacement);
  const sweptMaximum = Math.max(donorMaximum, donorMaximum + displacement);
  return [Math.floor(sweptMinimum / SWEPT_DOMAIN_PAGE_FINE_SIZE),
    Math.ceil(sweptMaximum / SWEPT_DOMAIN_PAGE_FINE_SIZE) - 1];
}

function validateDonor(minimum: readonly number[], maximum: readonly number[],
  displacement: readonly number[]): void {
  if (minimum.length !== maximum.length || minimum.length !== displacement.length
    || minimum.some((value, axis) => !Number.isFinite(value)
      || !Number.isFinite(maximum[axis]) || !Number.isFinite(displacement[axis])
      || !(maximum[axis]! > value))) {
    throw new RangeError("swept-domain donor must be a finite positive-volume box");
  }
}

function donorReachesPage(minimum: readonly number[], maximum: readonly number[],
  displacement: readonly number[], page: readonly number[]): boolean {
  let lower = 0, upper = 1;
  for (let axis = 0; axis < minimum.length; axis += 1) {
    const pageMinimum = SWEPT_DOMAIN_PAGE_FINE_SIZE * page[axis]!;
    const interval = overlapTimeInterval(minimum[axis]!, maximum[axis]!,
      displacement[axis]!, pageMinimum, pageMinimum + SWEPT_DOMAIN_PAGE_FINE_SIZE);
    if (!interval) return false;
    lower = Math.max(lower, interval[0]);
    upper = Math.min(upper, interval[1]);
  }
  return lower < upper;
}

/** Pages whose interiors meet M + t*d for a common t in [0,1]. */
export function exactSweptDomainPages2D(testCase: Pick<SweptDomainOracleCase, "donors">):
readonly SweptDomainPage2[] {
  const keys = new Set<string>();
  for (const donor of testCase.donors) {
    validateDonor(donor.minimumFine, donor.maximumExclusiveFine, donor.displacementFine);
    const x = donorPageRange(donor.minimumFine[0], donor.maximumExclusiveFine[0],
      donor.displacementFine[0]);
    const y = donorPageRange(donor.minimumFine[1], donor.maximumExclusiveFine[1],
      donor.displacementFine[1]);
    for (let pageY = y[0]; pageY <= y[1]; pageY += 1) {
      for (let pageX = x[0]; pageX <= x[1]; pageX += 1) {
        const page = [pageX, pageY] as const;
        if (donorReachesPage(donor.minimumFine, donor.maximumExclusiveFine,
          donor.displacementFine, page)) keys.add(`${pageX},${pageY}`);
      }
    }
  }
  return [...keys].map(key => key.split(",").map(Number) as [number, number])
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}

/** Three-dimensional positive-measure sweep, evaluated without 2-D expected data. */
export function exactSweptDomainPages3D(donors: readonly SweptDomainDonor3[]):
readonly SweptDomainPage3[] {
  const keys = new Set<string>();
  for (const donor of donors) {
    validateDonor(donor.minimumFine, donor.maximumExclusiveFine, donor.displacementFine);
    const x = donorPageRange(donor.minimumFine[0], donor.maximumExclusiveFine[0],
      donor.displacementFine[0]);
    const y = donorPageRange(donor.minimumFine[1], donor.maximumExclusiveFine[1],
      donor.displacementFine[1]);
    const z = donorPageRange(donor.minimumFine[2], donor.maximumExclusiveFine[2],
      donor.displacementFine[2]);
    for (let pageZ = z[0]; pageZ <= z[1]; pageZ += 1) {
      for (let pageY = y[0]; pageY <= y[1]; pageY += 1) {
        for (let pageX = x[0]; pageX <= x[1]; pageX += 1) {
          const page = [pageX, pageY, pageZ] as const;
          if (donorReachesPage(donor.minimumFine, donor.maximumExclusiveFine,
            donor.displacementFine, page)) keys.add(`${pageX},${pageY},${pageZ}`);
        }
      }
    }
  }
  return [...keys].map(key => key.split(",").map(Number) as [number, number, number])
    .sort((a, b) => a[2] - b[2] || a[1] - b[1] || a[0] - b[0]);
}

/** Strict parity extrusion: [x0,x1)×[y0,y1)×[0,8), with dz=0. */
export function extrudedDonors3D(testCase: Pick<SweptDomainOracleCase, "donors">):
readonly SweptDomainDonor3[] {
  return testCase.donors.map(donor => ({
    minimumFine: [...donor.minimumFine, 0],
    maximumExclusiveFine: [...donor.maximumExclusiveFine, SWEPT_DOMAIN_PAGE_FINE_SIZE],
    displacementFine: [...donor.displacementFine, 0],
  }));
}

export function extrudedExpectedPages3D(testCase: Pick<SweptDomainOracleCase, "donors">):
readonly SweptDomainPage3[] {
  return exactSweptDomainPages3D(extrudedDonors3D(testCase));
}
