/** Pure ABI and catalog-oracle helpers for structured positive-air velocity support. */

import type { GeneratedOctreePowerCatalogViews } from "./generated/octree-power-catalog";

export const OCTREE_AIR_SUPPORT_LAYOUT_VERSION = 2;
export const OCTREE_AIR_SUPPORT_VALID = 0x4156_5350;
export const OCTREE_AIR_SUPPORT_INVALID = 0xffff_ffff;
export const OCTREE_AIR_SUPPORT_TAG = 0x8000_0000;
export const OCTREE_AIR_SUPPORT_SELECTOR_STRIDE = 256;
export const OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE = 27;
export const OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS = 36;
export const OCTREE_AIR_SUPPORT_CONTROL_WORDS = 16;

const SUPPORT_INDEX_MASK = 0x7fff_ffff;

export interface OctreeAirVelocitySupportLayout {
  readonly rowCapacity: number;
  readonly slotCapacity: number;
  readonly supportCapacity: number;
  readonly selectorStride: number;
  readonly transportMetricOffsetBytes: 0;
  readonly transportMetricBytes: number;
  readonly selectorTagOffsetBytes: number;
  readonly selectorTagOffsetWords: number;
  readonly selectorTagBytes: number;
  readonly regularTagOffsetBytes: number;
  readonly regularTagOffsetWords: number;
  readonly regularTagBytes: number;
  readonly controlOffsetBytes: number;
  readonly controlOffsetWords: number;
  readonly controlBytes: number;
  readonly supportVectorOffsetBytes: number;
  readonly supportVectorOffsetWords: number;
  readonly supportVectorBytes: number;
  /** Dense finest-cell owner records `{tag, originCell, size, case|transform}`.
   * This is coalesced into the already-bound arena so fine transport does not
   * need an eleventh storage binding for the identity record arena. */
  readonly ownerDirectoryOffsetBytes: number;
  readonly ownerDirectoryOffsetWords: number;
  readonly ownerDirectoryBytes: number;
  readonly ownerDirectoryCellCapacity: number;
  readonly totalBytes: number;
}

export interface OctreeAirSupportPublication {
  readonly structuredEpoch: number;
  readonly structuredBank: 0 | 1;
  readonly boundaryEpoch: number;
  readonly pressureRowCount: number;
  readonly supportCount: number;
  readonly selectorReferenceCount: number;
  readonly regularReferenceCount: number;
  readonly faceCount: number;
  readonly seedFaceCount: number;
  readonly marchDepth: number;
  readonly fineGeneration: number;
}

export interface OctreeAirSupportDecode {
  readonly publication: OctreeAirSupportPublication | null;
  readonly blocker: string | null;
}

export interface OctreeCellIdentity {
  readonly cell: number;
  readonly size: number;
}

export interface OctreePublishedCellIdentity extends OctreeCellIdentity {
  readonly row: number;
}

export type OctreeSelectorDisposition = "anchor" | "published-row" | "support" | "exterior";

export interface OctreeSelectorCandidate {
  readonly selector: number;
  readonly canonicalOffset: readonly [number, number, number];
  readonly worldOffset: readonly [number, number, number];
  readonly sizeRatio: number;
  readonly disposition: OctreeSelectorDisposition;
  readonly identity: OctreeCellIdentity | null;
  readonly publishedRow: number | null;
}

export interface OctreeAirSupportCandidateInput {
  readonly catalog: Pick<GeneratedOctreePowerCatalogViews,
    "tetrahedronHeaders" | "tetrahedronData" | "tetrahedronVertexData">;
  readonly caseId: number;
  readonly transformCode: number;
  readonly anchor: OctreeCellIdentity;
  readonly dimensions: readonly [number, number, number];
  readonly published?: readonly OctreePublishedCellIdentity[];
}

export interface OctreeCatalogSelectorBound {
  readonly maximum: number;
  readonly caseId: number;
}

export type OctreeVelocityTag = Readonly<
  { readonly kind: "published-row"; readonly row: number }
  | { readonly kind: "support"; readonly support: number }
  | { readonly kind: "invalid" }
>;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be an unsigned u32`);
  }
  return value >>> 0;
}

function checkedProduct(label: string, ...values: number[]): number {
  const result = values.reduce((product, value) => product * value, 1);
  if (!Number.isSafeInteger(result) || result < 0) throw new RangeError(`${label} exceeds exact host addressing`);
  return result;
}

function align(value: number, alignment: number): number {
  const result = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(result)) throw new RangeError("Air-support arena exceeds exact host addressing");
  return result;
}

/**
 * The selector map retains its current offset immediately after transport
 * metrics. New support state is suffix-only, so the coarse-phi selector writer
 * cannot overwrite it and existing transport handle indices remain unchanged.
 */
export function planOctreeAirVelocitySupport(
  rowCapacityValue: number,
  slotCapacityValue: number,
  alignmentValue = 256,
  ownerDirectoryCellCapacityValue = 0,
): OctreeAirVelocitySupportLayout {
  const rowCapacity = positiveInteger(rowCapacityValue, "Air-support row capacity");
  const slotCapacity = positiveInteger(slotCapacityValue, "Air-support slot capacity");
  const alignment = positiveInteger(alignmentValue, "Air-support alignment");
  const ownerDirectoryCellCapacity = u32(ownerDirectoryCellCapacityValue,
    "Air-support owner-directory capacity");
  if ((alignment & (alignment - 1)) !== 0 || alignment % 16 !== 0) {
    throw new RangeError("Air-support alignment must be a vec4-aligned power of two");
  }
  const supportCapacity = checkedProduct("Air-support capacity",
    rowCapacity, OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS);
  if (supportCapacity > SUPPORT_INDEX_MASK) throw new RangeError("Air-support tag payload exceeds 31 bits");
  const transportMetricBytes = checkedProduct("Air-support transport metrics", slotCapacity, 16);
  const selectorTagOffsetBytes = transportMetricBytes;
  const selectorTagBytes = checkedProduct("Air-support selector tags",
    rowCapacity, OCTREE_AIR_SUPPORT_SELECTOR_STRIDE, 4);
  const regularTagOffsetBytes = align(selectorTagOffsetBytes + selectorTagBytes, alignment);
  const regularTagBytes = checkedProduct("Air-support regular tags",
    rowCapacity, OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE, 4);
  const controlOffsetBytes = align(regularTagOffsetBytes + regularTagBytes, alignment);
  const controlBytes = OCTREE_AIR_SUPPORT_CONTROL_WORDS * 4;
  const supportVectorOffsetBytes = align(controlOffsetBytes + controlBytes, alignment);
  const supportVectorBytes = checkedProduct("Air-support vectors", supportCapacity, 16);
  const ownerDirectoryOffsetBytes = align(supportVectorOffsetBytes + supportVectorBytes, alignment);
  const ownerDirectoryBytes = checkedProduct("Air-support owner directory",
    ownerDirectoryCellCapacity, 16);
  const totalBytes = align(ownerDirectoryOffsetBytes + ownerDirectoryBytes, alignment);
  return Object.freeze({
    rowCapacity, slotCapacity, supportCapacity,
    selectorStride: OCTREE_AIR_SUPPORT_SELECTOR_STRIDE,
    transportMetricOffsetBytes: 0 as const,
    transportMetricBytes,
    selectorTagOffsetBytes, selectorTagOffsetWords: selectorTagOffsetBytes / 4,
    selectorTagBytes,
    regularTagOffsetBytes, regularTagOffsetWords: regularTagOffsetBytes / 4,
    regularTagBytes,
    controlOffsetBytes, controlOffsetWords: controlOffsetBytes / 4,
    controlBytes,
    supportVectorOffsetBytes, supportVectorOffsetWords: supportVectorOffsetBytes / 4,
    supportVectorBytes,
    ownerDirectoryOffsetBytes, ownerDirectoryOffsetWords: ownerDirectoryOffsetBytes / 4,
    ownerDirectoryBytes, ownerDirectoryCellCapacity,
    totalBytes,
  });
}

export function encodeOctreeAirSupportTag(supportIndexValue: number): number {
  const supportIndex = u32(supportIndexValue, "Air-support index");
  if (supportIndex >= SUPPORT_INDEX_MASK) throw new RangeError("Air-support index collides with INVALID");
  return (OCTREE_AIR_SUPPORT_TAG | supportIndex) >>> 0;
}

/** A tag outside either published count is invalid rather than a fallback. */
export function decodeOctreeVelocityTag(
  tagValue: number,
  pressureRowCountValue: number,
  supportCountValue: number,
): OctreeVelocityTag {
  const tag = u32(tagValue, "Structured velocity tag");
  const pressureRowCount = u32(pressureRowCountValue, "Pressure-row count");
  const supportCount = u32(supportCountValue, "Air-support count");
  if (tag === OCTREE_AIR_SUPPORT_INVALID) return Object.freeze({ kind: "invalid" });
  if ((tag & OCTREE_AIR_SUPPORT_TAG) !== 0) {
    const support = tag & SUPPORT_INDEX_MASK;
    return support < supportCount
      ? Object.freeze({ kind: "support", support })
      : Object.freeze({ kind: "invalid" });
  }
  return tag < pressureRowCount
    ? Object.freeze({ kind: "published-row", row: tag })
    : Object.freeze({ kind: "invalid" });
}

/** Decode only an all-or-nothing, generation-coherent support publication. */
export function decodeOctreeAirSupportControl(
  words: ArrayLike<number>,
  layout: OctreeAirVelocitySupportLayout,
  expected?: Readonly<{
    structuredEpoch?: number;
    structuredBank?: 0 | 1;
    boundaryEpoch?: number;
    pressureRowCount?: number;
    fineGeneration?: number;
  }>,
): OctreeAirSupportDecode {
  const blocked = (blocker: string): OctreeAirSupportDecode => Object.freeze({ publication: null, blocker });
  if (words.length < OCTREE_AIR_SUPPORT_CONTROL_WORDS) return blocked("air-support control is truncated");
  const values = Array.from({ length: OCTREE_AIR_SUPPORT_CONTROL_WORDS },
    (_, index) => Number(words[index]) >>> 0);
  const [flags, firstError, structuredEpoch, structuredBank, boundaryEpoch, pressureRowCount,
    supportCount, supportCapacity, selectorReferenceCount, regularReferenceCount,
    faceCount, seedFaceCount, marchDepth, published, layoutVersion, fineGeneration] = values;
  if (flags !== 0) return blocked(`air-support publication failed with flags ${flags}`);
  if (firstError !== OCTREE_AIR_SUPPORT_INVALID) return blocked("air-support success retains an error index");
  if (published !== OCTREE_AIR_SUPPORT_VALID) return blocked("air-support publication is not committed");
  if (layoutVersion !== OCTREE_AIR_SUPPORT_LAYOUT_VERSION) {
    return blocked("air-support control ABI is incompatible");
  }
  if (structuredEpoch === 0 || structuredBank > 1 || boundaryEpoch === 0) {
    return blocked("air-support source generation is invalid");
  }
  if (pressureRowCount === 0 || pressureRowCount > layout.rowCapacity
    || supportCapacity !== layout.supportCapacity || supportCount > supportCapacity) {
    return blocked("air-support publication exceeds its arena");
  }
  if (supportCount > 0 && (selectorReferenceCount === 0 || faceCount === 0 || seedFaceCount === 0)) {
    return blocked("air-support publication is incomplete");
  }
  if (seedFaceCount > faceCount) return blocked("air-support seed count exceeds face count");
  if (expected?.structuredEpoch !== undefined && structuredEpoch !== expected.structuredEpoch
    || expected?.structuredBank !== undefined && structuredBank !== expected.structuredBank
    || expected?.boundaryEpoch !== undefined && boundaryEpoch !== expected.boundaryEpoch
    || expected?.pressureRowCount !== undefined && pressureRowCount !== expected.pressureRowCount
    || expected?.fineGeneration !== undefined && fineGeneration !== expected.fineGeneration) {
    return blocked("air-support publication does not match the accepted authority");
  }
  return Object.freeze({
    publication: Object.freeze({
      structuredEpoch, structuredBank: structuredBank as 0 | 1, boundaryEpoch,
      pressureRowCount, supportCount, selectorReferenceCount, regularReferenceCount,
      faceCount, seedFaceCount, marchDepth, fineGeneration,
    }),
    blocker: null,
  });
}

function dimensionsOf(dimensions: readonly [number, number, number]): readonly [number, number, number] {
  return dimensions.map((value) => positiveInteger(value, "Air-support dimension")) as
    unknown as readonly [number, number, number];
}

function originOf(cell: number, dimensions: readonly [number, number, number]): readonly [number, number, number] {
  const volume = checkedProduct("Air-support domain", ...dimensions);
  if (!Number.isSafeInteger(cell) || cell < 0 || cell >= volume) {
    throw new RangeError("Air-support anchor cell is outside the domain");
  }
  return [cell % dimensions[0], Math.floor(cell / dimensions[0]) % dimensions[1],
    Math.floor(cell / (dimensions[0] * dimensions[1]))];
}

/** CPU mirror of the production inverse structured transform. */
export function inverseOctreePowerTransform(
  value: readonly [number, number, number], transformCodeValue: number,
): readonly [number, number, number] {
  const transformCode = u32(transformCodeValue, "Power transform code");
  if (transformCode >= 48) throw new RangeError("Power transform code must name one of 48 cube symmetries");
  const signed: [number, number, number] = [
    value[0] * ((transformCode & 1) === 0 ? 1 : -1),
    value[1] * ((transformCode & 2) === 0 ? 1 : -1),
    value[2] * ((transformCode & 4) === 0 ? 1 : -1),
  ];
  for (let axis = 0; axis < 3; axis += 1) {
    if (Object.is(signed[axis], -0)) signed[axis] = 0;
  }
  switch (Math.floor(transformCode / 8) % 6) {
    case 0: return signed;
    case 1: return [signed[0], signed[2], signed[1]];
    case 2: return [signed[1], signed[0], signed[2]];
    case 3: return [signed[2], signed[0], signed[1]];
    case 4: return [signed[1], signed[2], signed[0]];
    default: return [signed[2], signed[1], signed[0]];
  }
}

function selectorsForCase(
  catalog: Pick<GeneratedOctreePowerCatalogViews, "tetrahedronHeaders" | "tetrahedronData">,
  caseId: number,
): readonly number[] {
  if (!Number.isSafeInteger(caseId) || caseId < 0
    || 3 * caseId + 2 >= catalog.tetrahedronHeaders.length) {
    throw new RangeError("Air-support case ID is outside the tetrahedron catalog");
  }
  const first = catalog.tetrahedronHeaders[3 * caseId]!;
  const count = catalog.tetrahedronHeaders[3 * caseId + 1]!;
  if (first > catalog.tetrahedronData.length || count > catalog.tetrahedronData.length - first) {
    throw new RangeError("Air-support tetrahedron range is invalid");
  }
  const seen = new Uint8Array(OCTREE_AIR_SUPPORT_SELECTOR_STRIDE);
  const selectors: number[] = [];
  for (let local = 0; local < count; local += 1) {
    const packed = catalog.tetrahedronData[first + local]!;
    for (const selector of [packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff]) {
      if (seen[selector] === 0) { seen[selector] = 1; selectors.push(selector); }
    }
  }
  return Object.freeze(selectors);
}

export function catalogMaximumOctreeAirSupportSelectors(
  catalog: Pick<GeneratedOctreePowerCatalogViews, "tetrahedronHeaders" | "tetrahedronData">,
): OctreeCatalogSelectorBound {
  if (catalog.tetrahedronHeaders.length % 3 !== 0) throw new RangeError("Tetrahedron headers are truncated");
  let maximum = 0; let maximumCase = 0;
  for (let caseId = 0; caseId < catalog.tetrahedronHeaders.length / 3; caseId += 1) {
    const count = selectorsForCase(catalog, caseId).length;
    if (count > maximum) { maximum = count; maximumCase = caseId; }
  }
  return Object.freeze({ maximum, caseId: maximumCase });
}

/**
 * Deterministically enumerates the local Delaunay selectors in first-tetra
 * order and distinguishes a real row, a required in-domain support identity,
 * and physical exterior. It never substitutes the anchor for a support cell.
 */
export function enumerateOctreeAirSupportCandidates(
  input: OctreeAirSupportCandidateInput,
): readonly OctreeSelectorCandidate[] {
  const dimensions = dimensionsOf(input.dimensions);
  const anchorSize = positiveInteger(input.anchor.size, "Air-support anchor size");
  const anchorOrigin = originOf(input.anchor.cell, dimensions);
  if (anchorOrigin.some((value, axis) => value + anchorSize > dimensions[axis]!)) {
    throw new RangeError("Air-support anchor extent is outside the domain");
  }
  const transformCode = u32(input.transformCode, "Power transform code");
  const selectors = selectorsForCase(input.catalog, input.caseId);
  if (selectors.length > OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS) {
    throw new RangeError("Air-support case exceeds the proven selector bound");
  }
  const published = new Map<string, number>();
  for (const entry of input.published ?? []) {
    const row = u32(entry.row, "Published support row");
    const key = `${u32(entry.cell, "Published support cell")}:${positiveInteger(entry.size, "Published support size")}`;
    const old = published.get(key);
    if (old !== undefined && old !== row) throw new RangeError("Published support identity is ambiguous");
    published.set(key, row);
  }
  const vertexCount = input.catalog.tetrahedronVertexData.length / 4;
  if (!Number.isInteger(vertexCount)) throw new RangeError("Tetrahedron vertex table is truncated");
  const center = anchorOrigin.map((value) => value + 0.5 * anchorSize);
  return Object.freeze(selectors.map((selector): OctreeSelectorCandidate => {
    if (selector >= vertexCount) throw new RangeError("Air-support selector is outside the vertex table");
    const at = 4 * selector;
    const canonicalOffset: [number, number, number] = [
      input.catalog.tetrahedronVertexData[at]!, input.catalog.tetrahedronVertexData[at + 1]!,
      input.catalog.tetrahedronVertexData[at + 2]!,
    ];
    const sizeRatio = input.catalog.tetrahedronVertexData[at + 3]!;
    const worldOffset = inverseOctreePowerTransform(canonicalOffset, transformCode);
    const neighborSizeFloat = anchorSize * sizeRatio;
    const neighborSize = Math.round(neighborSizeFloat);
    const neighborOriginFloat = center.map((value, axis) =>
      value + anchorSize * worldOffset[axis]! - 0.5 * neighborSizeFloat);
    const neighborOrigin = neighborOriginFloat.map(Math.round);
    if (!Number.isFinite(sizeRatio) || sizeRatio <= 0 || neighborSize < 1
      || Math.abs(neighborSizeFloat - neighborSize) > 2e-4
      || neighborOrigin.some((value, axis) => Math.abs(neighborOriginFloat[axis]! - value) > 2e-4)) {
      throw new RangeError("Air-support selector does not resolve to an octree identity");
    }
    const exterior = neighborOrigin.some((value, axis) =>
      value < 0 || value + neighborSize > dimensions[axis]!);
    if (exterior) return Object.freeze({ selector, canonicalOffset, worldOffset, sizeRatio,
      disposition: "exterior", identity: null, publishedRow: null });
    const cell = neighborOrigin[0]! + dimensions[0]
      * (neighborOrigin[1]! + dimensions[1] * neighborOrigin[2]!);
    const identity = Object.freeze({ cell, size: neighborSize });
    if (cell === input.anchor.cell && neighborSize === anchorSize) {
      return Object.freeze({ selector, canonicalOffset, worldOffset, sizeRatio,
        disposition: "anchor", identity, publishedRow: null });
    }
    const row = published.get(`${cell}:${neighborSize}`);
    return Object.freeze({ selector, canonicalOffset, worldOffset, sizeRatio,
      disposition: row === undefined ? "support" : "published-row", identity,
      publishedRow: row ?? null });
  }));
}

function dot(a: readonly number[], b: readonly number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

function cross(a: readonly number[], b: readonly number[]): readonly [number, number, number] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!];
}

/** CPU mirror of the transition sampler's anchor-plus-three tetra weights. */
export function octreeAirSupportTetraWeights(
  point: readonly [number, number, number],
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
): readonly [number, number, number, number] {
  const determinant = dot(a, cross(b, c));
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-10) return [-2, -2, -2, -2];
  const wa = dot(point, cross(b, c)) / determinant;
  const wb = dot(a, cross(point, c)) / determinant;
  const wc = dot(a, cross(b, point)) / determinant;
  return [1 - wa - wb - wc, wa, wb, wc];
}

/** Selectors with exactly zero clamped barycentric weight require no support fetch. */
export function positiveOctreeAirSupportSelectors(
  packedTetrahedronValue: number,
  weights: readonly [number, number, number, number],
): readonly number[] {
  const packed = u32(packedTetrahedronValue, "Packed tetrahedron");
  const selectors = [packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff];
  return Object.freeze(selectors.filter((_, index) => Math.max(0, weights[index + 1]!) > 0));
}
