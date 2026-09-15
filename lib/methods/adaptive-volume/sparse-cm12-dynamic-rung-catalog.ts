/** Relocatable topology prepared once for an eight-spacing dynamic page.
 * Cells from different rungs never alias: transfer can read the accepted rung
 * while writing a candidate. Boundary rows reserve the CM12 maximum of five
 * terms (one coarse cell and four fine cells), including when initially dry.
 */
export const DYNAMIC_PAGE_RUNGS = [8, 4, 2, 1] as const;
export type DynamicPageRung = typeof DYNAMIC_PAGE_RUNGS[number];
export type DynamicPageSide = 0 | 1 | 2 | 3 | 4 | 5;
export interface DynamicRungLayout {
  readonly resolution: DynamicPageRung;
  readonly width: number;
  readonly cellOffset: number;
  readonly cellCount: number;
  readonly rowOffset: number;
  readonly rowCount: number;
  readonly termOffset: number;
  readonly termCount: number;
  readonly boundaryOffset: number;
}
let cells = 0, rows = 0, terms = 0, boundaries = 0;
export const DYNAMIC_RUNG_LAYOUTS: readonly DynamicRungLayout[] = DYNAMIC_PAGE_RUNGS.map(resolution => {
  const cellCount = resolution ** 3;
  const rowCount = 3 * (resolution + 1) * resolution ** 2;
  const boundaryCount = 6 * resolution ** 2;
  const termCount = 2 * rowCount + 3 * boundaryCount;
  const layout = Object.freeze({ resolution, width: 8 / resolution,
    cellOffset: cells, cellCount, rowOffset: rows, rowCount,
    termOffset: terms, termCount, boundaryOffset: boundaries });
  cells += cellCount; rows += rowCount; terms += termCount; boundaries += boundaryCount;
  return layout;
});
export const DYNAMIC_PAGE_CELL_COUNT = cells;
export const DYNAMIC_PAGE_ROW_COUNT = rows;
export const DYNAMIC_PAGE_TERM_COUNT = terms;
export const DYNAMIC_PAGE_BOUNDARY_COUNT = boundaries;
export function dynamicRungLayout(resolution: number): DynamicRungLayout {
  const layout = DYNAMIC_RUNG_LAYOUTS.find(layout => layout.resolution === resolution);
  if (!layout) throw new RangeError(`unprepared dynamic rung B${resolution}`);
  return layout;
}
/** Global-in-page term address for a row in one rung. */
export function dynamicRowTermOffset(resolution: DynamicPageRung, row: number): number {
  const layout = dynamicRungLayout(resolution);
  if (!Number.isInteger(row) || row < 0 || row >= layout.rowCount) throw new RangeError("dynamic row outside rung");
  const perAxis = (resolution + 1) * resolution ** 2;
  const axis = Math.floor(row / perAxis), within = row % perAxis;
  const uv = Math.floor(within / (resolution + 1));
  const face = within % (resolution + 1);
  const boundariesBefore = 2 * axis * resolution ** 2 + 2 * uv + Number(face > 0);
  return layout.termOffset + 2 * row + 3 * boundariesBefore;
}
export interface DynamicSeamTerm {
  readonly neighbor: boolean;
  /** Rung-local cell index, resolved against the prepared cell range. */
  readonly local: number;
  readonly coefficient: number;
}
export interface DynamicSeamRow {
  /** Rung-local row slot; fine-side rows use the lower corner as their anchor. */
  readonly row: number;
  readonly center: readonly [number, number, number];
  readonly area: number;
  readonly distance: number;
  readonly terms: readonly DynamicSeamTerm[];
}
export interface DynamicSeamVariant {
  readonly own: DynamicPageRung;
  readonly neighbor: DynamicPageRung | 0;
  readonly side: DynamicPageSide;
  readonly rows: readonly DynamicSeamRow[];
  /** One (row slot, term ordinal) per own boundary cell, u-major. */
  readonly incidence: readonly (readonly [number, number])[];
}
const localIndex = (q: readonly number[], resolution: number) => q[0]! + resolution * (q[1]! + resolution * q[2]!);

/** Compile complete face variants; runtime binds identities, never fits weights. */
export function compileDynamicSeamVariant(own: DynamicPageRung,
  neighbor: DynamicPageRung | 0, side: DynamicPageSide): DynamicSeamVariant {
  const ownLayout = dynamicRungLayout(own);
  if (!Number.isInteger(side) || side < 0 || side > 5) throw new RangeError("invalid page side");
  if (neighbor !== 0) {
    dynamicRungLayout(neighbor);
    if (Math.max(own, neighbor) > 2 * Math.min(own, neighbor)) throw new RangeError("ungraded dynamic seam");
  }
  const axis = Math.floor(side / 2), positive = side % 2 === 1;
  const uAxis = (axis + 1) % 3, vAxis = (axis + 2) % 3;
  const patchResolution = neighbor === 0 ? own : Math.min(own, neighbor);
  const ownRatio = own / patchResolution;
  const neighborRatio = neighbor === 0 ? 0 : neighbor / patchResolution;
  const patchWidth = 8 / patchResolution;
  const distance = neighbor === 0 ? ownLayout.width : (ownLayout.width + 8 / neighbor) / 2;
  const result: DynamicSeamRow[] = [];
  const incidence: [number, number][] = Array.from({ length: own * own }, () => [-1, -1]);
  for (let v = 0; v < patchResolution; v++) for (let u = 0; u < patchResolution; u++) {
    const row = axis * (own + 1) * own ** 2 + (positive ? own : 0)
      + (own + 1) * (u * ownRatio + own * v * ownRatio);
    const center: [number, number, number] = [0, 0, 0];
    center[axis] = positive ? 8 : 0;
    center[uAxis] = (u + 0.5) * patchWidth;
    center[vAxis] = (v + 0.5) * patchWidth;
    const rowTerms: DynamicSeamTerm[] = [];
    // Negative-side terms first, matching the authoritative composite builder.
    for (const other of positive ? [false, true] : [true, false]) {
      if (other && neighbor === 0) continue;
      const resolution = other ? neighbor : own;
      const ratio = other ? neighborRatio : ownRatio;
      for (let dv = 0; dv < ratio; dv++) for (let du = 0; du < ratio; du++) {
        const q = [0, 0, 0];
        q[axis] = other ? (positive ? 0 : resolution - 1) : (positive ? resolution - 1 : 0);
        q[uAxis] = u * ratio + du; q[vAxis] = v * ratio + dv;
        const coefficient = (other === positive ? 1 : -1) / (ratio ** 2 * distance);
        if (!other) incidence[q[uAxis]! + own * q[vAxis]!] = [row, rowTerms.length];
        rowTerms.push(Object.freeze({ neighbor: other, local: localIndex(q, resolution), coefficient }));
      }
    }
    result.push(Object.freeze({ row, center: Object.freeze(center), area: patchWidth ** 2,
      distance, terms: Object.freeze(rowTerms) }));
  }
  return Object.freeze({ own, neighbor, side, rows: Object.freeze(result),
    incidence: Object.freeze(incidence.map(pair => Object.freeze(pair))) });
}
let catalogue: ReadonlyMap<string, DynamicSeamVariant> | undefined;
export function preparedDynamicSeamCatalogue(): ReadonlyMap<string, DynamicSeamVariant> {
  if (catalogue) return catalogue;
  const result = new Map<string, DynamicSeamVariant>();
  for (const own of DYNAMIC_PAGE_RUNGS) for (const neighbor of [0, ...DYNAMIC_PAGE_RUNGS] as const) {
    if (neighbor !== 0 && Math.max(own, neighbor) > 2 * Math.min(own, neighbor)) continue;
    for (let side = 0; side < 6; side++) result.set(`${own}/${neighbor}/${side}`,
      compileDynamicSeamVariant(own, neighbor, side as DynamicPageSide));
  }
  catalogue = result;
  return result;
}

/** The physical 2:1 floor for a newly admitted ordinary eight-spacing page.
 * Missing/dormant neighbours impose no constraint. Joint admission resolves
 * new/new constraints with the same refine-only closure as existing leaves.
 */
export function coarseDynamicAdmissionRung(neighborWidths: readonly number[],
  minimumResolution: DynamicPageRung = 1, maximumResolution: DynamicPageRung = 8): DynamicPageRung {
  dynamicRungLayout(minimumResolution); dynamicRungLayout(maximumResolution);
  let rung: number = minimumResolution;
  for (const width of neighborWidths) {
    if (!Number.isFinite(width) || width <= 0) throw new RangeError("invalid neighbour cell width");
    while (8 / rung > 2 * width && rung < 8) rung *= 2;
    if (8 / rung > 2 * width) throw new RangeError("no prepared rung satisfies 2:1 grading");
  }
  if (rung > maximumResolution) throw new RangeError("authored cap conflicts with 2:1 admission");
  return rung as DynamicPageRung;
}

export function dynamicRungLayoutWGSL(): string {
  return `
const CM12_DYNAMIC_CELLS:u32=${DYNAMIC_PAGE_CELL_COUNT}u;
const CM12_DYNAMIC_ROWS:u32=${DYNAMIC_PAGE_ROW_COUNT}u;
const CM12_DYNAMIC_TERMS:u32=${DYNAMIC_PAGE_TERM_COUNT}u;
const CM12_DYNAMIC_BOUNDARIES:u32=${DYNAMIC_PAGE_BOUNDARY_COUNT}u;
fn cm12DynamicRungIndex(resolution:u32)->u32{
  return 3u-firstLeadingBit(resolution);
}
fn cm12DynamicCellOffset(resolution:u32)->u32{
  return array<u32,4>(${DYNAMIC_RUNG_LAYOUTS.map(l => `${l.cellOffset}u`).join(",")})[cm12DynamicRungIndex(resolution)];
}
fn cm12DynamicRowOffset(resolution:u32)->u32{
  return array<u32,4>(${DYNAMIC_RUNG_LAYOUTS.map(l => `${l.rowOffset}u`).join(",")})[cm12DynamicRungIndex(resolution)];
}
fn cm12DynamicTermOffset(resolution:u32)->u32{
  return array<u32,4>(${DYNAMIC_RUNG_LAYOUTS.map(l => `${l.termOffset}u`).join(",")})[cm12DynamicRungIndex(resolution)];
}
fn cm12DynamicBoundaryOffset(resolution:u32)->u32{
  return array<u32,4>(${DYNAMIC_RUNG_LAYOUTS.map(l => `${l.boundaryOffset}u`).join(",")})[cm12DynamicRungIndex(resolution)];
}
fn cm12DynamicCellRung(within:u32)->u32{
  if(within<512u){return 8u;}if(within<576u){return 4u;}
  return select(1u,2u,within<584u);
}
fn cm12DynamicRowRung(within:u32)->u32{
  if(within<1728u){return 8u;}if(within<1968u){return 4u;}
  return select(1u,2u,within<2004u);
}
fn cm12DynamicRowFirstTerm(resolution:u32,row:u32)->u32{
  let perAxis=(resolution+1u)*resolution*resolution;
  let axis=row/perAxis;let within=row%perAxis;
  let uv=within/(resolution+1u);let face=within%(resolution+1u);
  let preceding=2u*axis*resolution*resolution+2u*uv+select(0u,1u,face>0u);
  return cm12DynamicTermOffset(resolution)+2u*row+3u*preceding;
}
`;
}

export const DYNAMIC_PAGE_HEADER_WORDS = 16;
export const DYNAMIC_PAGE_ROW_BASE = DYNAMIC_PAGE_HEADER_WORDS;
export const DYNAMIC_PAGE_TERM_BASE = DYNAMIC_PAGE_ROW_BASE + 8 * DYNAMIC_PAGE_ROW_COUNT;
export const DYNAMIC_PAGE_INCIDENCE_BASE = DYNAMIC_PAGE_TERM_BASE + 2 * DYNAMIC_PAGE_TERM_COUNT;
export const DYNAMIC_PAGE_WORDS = DYNAMIC_PAGE_INCIDENCE_BASE + 2 * DYNAMIC_PAGE_BOUNDARY_COUNT;

/** Construction-time image for one reusable physical slot. Coordinates are
 * relative to the leaf origin. No fluid, directory membership or neighbor
 * identity is published by this image. Boundary term slots are reserved even
 * for absent neighbors, so activation never allocates connectivity storage.
 */
export function prepareDynamicPageImage(cellBase: number, rowBase: number, termBase: number): Uint32Array {
  for (const base of [cellBase, rowBase, termBase]) {
    if (!Number.isSafeInteger(base) || base < 0) throw new RangeError("invalid stable page base");
  }
  if (termBase + DYNAMIC_PAGE_TERM_COUNT > 0x800000) throw new RangeError("dynamic terms exceed packed row address space");
  if (cellBase + DYNAMIC_PAGE_CELL_COUNT >= 0xffffffff || rowBase + DYNAMIC_PAGE_ROW_COUNT >= 0xffffffff)
    throw new RangeError("dynamic identities exceed u32 address space");
  const words = new Uint32Array(DYNAMIC_PAGE_WORDS);
  const floats = new Float32Array(words.buffer);
  words[0] = 0xffffffff;
  words[4] = DYNAMIC_PAGE_ROW_COUNT;
  words[5] = 6 * DYNAMIC_PAGE_CELL_COUNT;
  words[7] = DYNAMIC_PAGE_ROW_BASE;
  words[8] = DYNAMIC_PAGE_TERM_BASE;
  words[10] = DYNAMIC_PAGE_INCIDENCE_BASE;
  for (const layout of DYNAMIC_RUNG_LAYOUTS) {
    const r = layout.resolution, width = layout.width, perAxis = (r + 1) * r * r;
    for (let row = 0; row < layout.rowCount; row++) {
      const axis = Math.floor(row / perAxis), within = row % perAxis;
      const face = within % (r + 1), uv = Math.floor(within / (r + 1));
      const u = uv % r, v = Math.floor(uv / r);
      const boundary = face === 0 || face === r;
      const stableRow = layout.rowOffset + row;
      const firstTerm = dynamicRowTermOffset(r, row);
      words[DYNAMIC_PAGE_ROW_BASE + stableRow] = (termBase + firstTerm) | ((boundary ? 1 : 2) << 23);
      words[DYNAMIC_PAGE_ROW_BASE + DYNAMIC_PAGE_ROW_COUNT + stableRow] = ((axis << 30) | ((boundary ? 3 : 0) << 28)) >>> 0;
      floats[DYNAMIC_PAGE_ROW_BASE + 2 * DYNAMIC_PAGE_ROW_COUNT + stableRow] = width;
      floats[DYNAMIC_PAGE_ROW_BASE + 3 * DYNAMIC_PAGE_ROW_COUNT + stableRow] = width ** 3;
      const center = [0, 0, 0]; center[axis] = face * width;
      center[(axis + 1) % 3] = (u + 0.5) * width; center[(axis + 2) % 3] = (v + 0.5) * width;
      for (let a = 0; a < 3; a++) floats[DYNAMIC_PAGE_ROW_BASE + (4 + a) * DYNAMIC_PAGE_ROW_COUNT + stableRow] = center[a]!;
      floats[DYNAMIC_PAGE_ROW_BASE + 7 * DYNAMIC_PAGE_ROW_COUNT + stableRow] = width ** 2;
      const q = [0, 0, 0]; q[axis] = Math.max(0, face - 1);
      q[(axis + 1) % 3] = u; q[(axis + 2) % 3] = v;
      words[DYNAMIC_PAGE_TERM_BASE + 2 * firstTerm] = cellBase + layout.cellOffset + localIndex(q, r);
      floats[DYNAMIC_PAGE_TERM_BASE + 2 * firstTerm + 1] = (face === 0 ? 1 : -1) / width;
      if (!boundary) {
        q[axis] = face;
        words[DYNAMIC_PAGE_TERM_BASE + 2 * (firstTerm + 1)] = cellBase + layout.cellOffset + localIndex(q, r);
        floats[DYNAMIC_PAGE_TERM_BASE + 2 * (firstTerm + 1) + 1] = 1 / width;
      } else {
        const side = 2 * axis + Number(face === r);
        const incidence = DYNAMIC_PAGE_INCIDENCE_BASE + 2 * (layout.boundaryOffset + side * r * r + u + r * v);
        words[incidence] = rowBase + stableRow;
        words[incidence + 1] = termBase + firstTerm;
      }
    }
  }
  return words;
}

export const DYNAMIC_SEAM_HEADER_WORDS = 4;
export const DYNAMIC_SEAM_ROW_WORDS = 23;
/** Fixed lookup table followed by immutable relative row/term/incidence data. */
export function packDynamicSeamCatalogue(): Uint32Array {
  const variants = preparedDynamicSeamCatalogue();
  const headerCount = 4 * 5 * 6;
  let length = headerCount * DYNAMIC_SEAM_HEADER_WORDS;
  for (const variant of variants.values()) length += variant.rows.length * DYNAMIC_SEAM_ROW_WORDS + 2 * variant.incidence.length;
  const words = new Uint32Array(length), f = new Float32Array(words.buffer);
  let cursor = headerCount * DYNAMIC_SEAM_HEADER_WORDS;
  for (const variant of variants.values()) {
    const own = DYNAMIC_PAGE_RUNGS.indexOf(variant.own);
    const neighbor = variant.neighbor === 0 ? 0 : 1 + DYNAMIC_PAGE_RUNGS.indexOf(variant.neighbor);
    const header = ((own * 5 + neighbor) * 6 + variant.side) * DYNAMIC_SEAM_HEADER_WORDS;
    words[header] = cursor; words[header + 1] = variant.rows.length;
    for (const row of variant.rows) {
      words[cursor] = row.row;
      f.set(row.center, cursor + 1); f[cursor + 4] = row.area;
      f[cursor + 5] = row.distance; words[cursor + 6] = row.terms.length;
      for (let t = 0; t < row.terms.length; t++) {
        const term = row.terms[t]!; const at = cursor + 8 + 3 * t;
        words[at] = Number(term.neighbor); words[at + 1] = term.local; f[at + 2] = term.coefficient;
      }
      cursor += DYNAMIC_SEAM_ROW_WORDS;
    }
    words[header + 2] = cursor; words[header + 3] = 1;
    for (const [row, term] of variant.incidence) {
      // Point straight to the prepared row record, avoiding a runtime search.
      const ordinal = variant.rows.findIndex(candidate => candidate.row === row);
      words[cursor++] = words[header]! + ordinal * DYNAMIC_SEAM_ROW_WORDS;
      words[cursor++] = term;
    }
  }
  return words;
}
