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
    const sourceFaces: ReturnType<typeof sparseCM12TransferFaceGeometry>[] = [];
    const f = new Float32Array(templateWords.buffer, templateWords.byteOffset, templateWords.length);
    const hostRows = templateWords[3]!, rowBase = templateWords[7]!;
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
      if (rowIdsByGeometry.has(key)) throw new Error("CM12 accepted faces have overlapping flux authority");
      rowIdsByGeometry.set(key, row);
      sourceFaces.push(sparseCM12TransferFaceGeometry(sourceFaces.length,axis,center,area,maximumSpan));
    }
    const rowIds = rows;
  return { geometry: {dimensions:atlas.dimensions,cells:sourceCells,faces:sourceFaces}, cellIds, rowIds };
}
