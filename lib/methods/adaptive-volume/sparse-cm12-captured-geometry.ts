import { sparseBrickSpan, sparseBrickMaximumFine, type SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";
import { sparseCM12TransferFaceGeometry, type SparseCM12TransferBox } from "./sparse-cm12-generation-transfer";

export interface CM12CapturedGeometryRecipe {
  atlas: SparseAdaptiveMassAtlas; active: ReadonlySet<number>; sourceFirst: ReadonlyMap<number,number>;
  sourcePageCoordinates: ReadonlyMap<number,readonly [number,number,number]>;
  dynamicKeys: ReadonlySet<number>; rows: Uint32Array; templateWords: Uint32Array;
}
/** CPU preparation only. Never expand these per-cell/per-face objects in the advancing worker. */
export function compileCM12CapturedGeometry(input: CM12CapturedGeometryRecipe) {
  const {atlas, active, sourceFirst, sourcePageCoordinates, dynamicKeys, rows, templateWords} = input;
  const INVALID = 0xffffffff;
    const sourceCells: SparseCM12TransferBox[] = [], physicalCells: number[] = [];
    let maximumSpan = 1;
    for (const brick of atlas.bricks) {
      const span = atlas.brickFineResolution * sparseBrickSpan(brick) / brick.resolution;
      maximumSpan = Math.max(maximumSpan, span);
      let compact = 0;
      for (let z=0;z<brick.resolution;z++) for(let y=0;y<brick.resolution;y++) for(let x=0;x<brick.resolution;x++) {
        const lower = [x,y,z].map((q,axis)=>brick.coordinate[axis]! * atlas.brickFineResolution + q*span);
        const widths = lower.map((q,axis)=>Math.max(0,Math.min(span,sparseBrickMaximumFine(atlas,brick,axis)-q)));
        if (widths.some(w=>w<=0)) continue;
        sourceCells.push({ id:sourceCells.length,lower,widths,span });
        const local = dynamicKeys.has(brick.key) ? x+brick.resolution*(y+brick.resolution*z) : compact;
        physicalCells.push(active.has(brick.key) ? sourceFirst.get(brick.key)!+local : INVALID);
        compact++;
      }
    }
    const cellIds = Uint32Array.from(physicalCells);
    const keyFor = (axis: number, center: readonly number[], area: number) =>
      `${axis}/${center.join("/")}/${area}`;
    const rowIdsByGeometry = new Map<string, number>();
    const f = new Float32Array(templateWords.buffer, templateWords.byteOffset, templateWords.length);
    const hostRows = templateWords[3]!, rowBase = templateWords[7]!;
    const rowPlane = (row: number, plane: number) => rowBase + plane * hostRows + row;
    const hostRowTerms = (row: number) => {
      const packed = templateWords[rowPlane(row, 0)]!;
      const first = packed & 0x007f_ffff, count = packed >>> 23;
      return Array.from({ length: count }, (_, at) => ({
        cell: templateWords[templateWords[8]! + 2 * (first + at)]!,
        coefficient: f[templateWords[8]! + 2 * (first + at) + 1]!,
      }));
    };
    const hostRowRequirements = (row: number) => {
      const at = templateWords[rowPlane(row, 1)]! & 0x0fff_ffff;
      return Array.from({ length: templateWords[at]! }, (_, index) => {
        const metadata = templateWords[at + 1 + index]!;
        return `${metadata >>> 5}@${metadata & 0x1f}`;
      });
    };
    const describeRow = (row: number) => {
      if (row < hostRows) {
        return `host row ${row} (kind ${(templateWords[rowPlane(row, 1)]! >>> 28) & 3}`
          + `, terms ${JSON.stringify(hostRowTerms(row))}`
          + `, requires ${hostRowRequirements(row).join("+")}`
          + `, dual ${f[rowPlane(row, 2)]}, distance ${f[rowPlane(row, 4)]})`;
      }
      const local = row - hostRows;
      return `dynamic row ${row} (page ${Math.floor(local / 1728)}`
        + ` at ${sourcePageCoordinates.get(Math.floor(local / 1728))?.join(",") ?? "?"}`
        + `, normal ${(local % 1728) % 576 % 9})`;
    };
    const sourceFaces: ReturnType<typeof sparseCM12TransferFaceGeometry>[] = [];
    for (const row of rows) {
      let axis: number, area: number, center: number[];
      if (row < hostRows) {
        axis = templateWords[rowBase + hostRows + row]! >>> 30;
        area = f[rowBase + 3 * hostRows + row]!;
        center = [6, 7, 8].map((plane) => f[rowBase + plane * hostRows + row]!);
      } else {
        const local = row - hostRows, page = Math.floor(local / 1728), within = local % 1728;
        const coordinate = sourcePageCoordinates.get(page);
        if (!coordinate) throw new Error(`CM12 accepted row ${row} names absent dynamic page ${page}; captured ${sourcePageCoordinates.size} pages`);
        axis = Math.floor(within / 576); area = 1;
        const index = within % 576, normal = index % 9, uv = Math.floor(index / 9);
        center = coordinate.map((n) => 8 * n);
        center[axis] += normal;
        center[(axis + 1) % 3] += uv % 8 + 0.5;
        center[(axis + 2) % 3] += Math.floor(uv / 8) + 0.5;
      }
      const key = keyFor(axis, center, area);
      if (rowIdsByGeometry.has(key)) {
        const first = rowIdsByGeometry.get(key)!;
        let replacements = "";
        if (first < hostRows) {
          const hostCell = hostRowTerms(first)[0]?.cell ?? 0;
          const begin = templateWords[templateWords[9]! + hostCell]!;
          const end = templateWords[templateWords[9]! + hostCell + 1]!;
          const candidates: string[] = [];
          for (let incidence = begin; incidence < end && incidence - begin < 4096; incidence += 1) {
            const other = templateWords[templateWords[10]! + 2 * incidence]!;
            if (other === first || other >= hostRows) continue;
            if ((templateWords[rowPlane(other, 1)]! >>> 30) !== axis) continue;
            if (f[rowPlane(other, 6 + axis)] !== center[axis]) continue;
            candidates.push(`${describeRow(other)} own ${
              f[templateWords[8]! + 2 * templateWords[templateWords[10]! + 2 * incidence + 1]! + 1]}`);
          }
          replacements = `; cell ${hostCell} coplanar incidences [${begin},${end}): ${
            candidates.join(" | ") || "none"}`;
        }
        throw new Error("CM12 accepted faces have overlapping flux authority: "
          + `${describeRow(first)} and ${describeRow(row)}`
          + ` both claim axis ${axis} center ${center.join(",")} area ${area}${replacements}`);
      }
      rowIdsByGeometry.set(key, row);
      sourceFaces.push(sparseCM12TransferFaceGeometry(sourceFaces.length,axis,center,area,maximumSpan));
    }
    const rowIds = rows;
  return { geometry: {dimensions:atlas.dimensions,cells:sourceCells,faces:sourceFaces}, cellIds, rowIds };
}
