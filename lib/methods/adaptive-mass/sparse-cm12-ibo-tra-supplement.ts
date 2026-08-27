import type { SparseCM12InternedBoundaryCompilation } from
  "./sparse-cm12-interned-boundary-operators";

export const SPARSE_CM12_IBO_TRA_MAGIC = 0x4954_5231; // ITR1
export const SPARSE_CM12_IBO_TRA_HEADER_WORDS = 16;
export const SPARSE_CM12_IBO_TRA_DIRECTORY_WORDS = 5;
export const SPARSE_CM12_IBO_TRA_INVALID = 0xffff_ffff;

export interface SparseCM12IboTRASupplementLayout {
  readonly baseWords: number;
  readonly templateCount: number;
  readonly directoryBaseWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export const SPARSE_CM12_IBO_TRA_HOST_CHRONOLOGY = Object.freeze([
  "copy prior DCA row/cell indirect records to an isolated indirect buffer",
  "clear only prior touched DCA row/cell packets before the IBO selector flip",
  "commit the validated IBO shadow selector",
  "begin DCA generation; gather publishes staged stable packet ids and exact nonzero ballots",
  "seal source indirects; copy TRA/VEX indirect records to the isolated indirect buffer",
  "compile exact IBO TRA spatial row masks and VEX cell masks",
  "seal target indirects; copy row-scatter/VEX-seed records to the isolated indirect buffer",
  "scatter the single configured gamma iteration from persistent spatial row masks; finalize its immutable snapshot",
  "seed packetized VEX frontier; retain touched masks until next-frame sparse clear",
] as const);


const align64 = (value: number) => Math.ceil(value / 64) * 64;
const f32 = (bits: number) => {
  const buffer = new ArrayBuffer(4);new Uint32Array(buffer)[0] = bits;
  return new Float32Array(buffer)[0]!;
};

/** Exact template-local source incidence and positive-owner lookup. */
export function createSparseCM12IboTRASupplement(options: Readonly<{
  ibo: SparseCM12InternedBoundaryCompilation;
  baseWords?: number;
}>): Readonly<{ layout: SparseCM12IboTRASupplementLayout; words: Uint32Array }> {
  const baseWords = align64(options.baseWords ?? 0), templates = options.ibo.templates;
  const directoryBaseWords = baseWords + SPARSE_CM12_IBO_TRA_HEADER_WORDS;
  let at = align64(directoryBaseWords
    + SPARSE_CM12_IBO_TRA_DIRECTORY_WORDS * templates.length);
  const records: Array<{ offsets: number[]; entries: number[]; owners: number[];
    sparseAirOwners: number[]; resolution: number; offsetsBase: number;
    entriesBase: number; ownersBase: number; sparseAirOwnersBase: number }> = [];
  for (const template of templates) {
    const resolution = template.sourceResolution, boundaryCount = resolution ** 2;
    const byBoundary: number[][] = Array.from({ length: boundaryCount }, () => []);
    const owners = new Array<number>(boundaryCount).fill(SPARSE_CM12_IBO_TRA_INVALID);
    const sparseAirOwners = new Array<number>(boundaryCount)
      .fill(SPARSE_CM12_IBO_TRA_INVALID);
    const termBase = 8 + 7 * template.rowCount;
    const dimensions = template.sourceDimensions;
    for (let row = 0; row < template.rowCount; row += 1) {
      const rowAt = 8 + 7 * row, packed = template.words[rowAt + 1]!;
      const first = packed & 0x007f_ffff, count = packed >>> 23;
      let ownerTerm = -1;
      for (let term = 0; term < count; term += 1) {
        const termAt = termBase + 2 * (first + term);
        const normalized = template.words[termAt]!;
        if ((normalized & 0x8000_0000) === 0 && f32(template.words[termAt + 1]!) > 0
          && ownerTerm < 0) ownerTerm = term;
        if ((normalized & 0x8000_0000) !== 0) continue;
        const ordinal = normalized & 0x7fff_ffff;
        const z = Math.floor(ordinal / (dimensions[0] * dimensions[1]));
        const remain = ordinal - z * dimensions[0] * dimensions[1];
        const y = Math.floor(remain / dimensions[0]), x = remain - y * dimensions[0];
        const axis = template.side >>> 1;
        const u = axis === 0 ? y : x, v = axis === 2 ? y : z;
        const boundary = u + resolution * v;
        if (!byBoundary[boundary]!.includes(row)) byBoundary[boundary]!.push(row);
      }
      if (ownerTerm >= 0) {
        const normalized = template.words[termBase + 2 * (first + ownerTerm)]!;
        const ordinal = normalized & 0x7fff_ffff;
        const z = Math.floor(ordinal / (dimensions[0] * dimensions[1]));
        const remain = ordinal - z * dimensions[0] * dimensions[1];
        const y = Math.floor(remain / dimensions[0]), x = remain - y * dimensions[0];
        const axis = template.side >>> 1;
        const boundary = (axis === 0 ? y : x) + resolution * (axis === 2 ? y : z);
        if (owners[boundary] !== SPARSE_CM12_IBO_TRA_INVALID && owners[boundary] !== row) {
          throw new Error(`ITR1 template ${template.id} owner collision at ${boundary}`);
        }
        owners[boundary] = row;
      } else {
        const metadata = template.words[rowAt + 2]!;
        const normalized = template.words[termBase + 2 * first]!;
        const coefficient = count === 1
          ? f32(template.words[termBase + 2 * first + 1]!) : 0;
        if (((metadata >>> 28) & 3) === 3 && count === 1 && coefficient < 0
          && (normalized & 0x8000_0000) === 0) {
          const ordinal = normalized & 0x7fff_ffff;
          const z = Math.floor(ordinal / (dimensions[0] * dimensions[1]));
          const remain = ordinal - z * dimensions[0] * dimensions[1];
          const y = Math.floor(remain / dimensions[0]);
          const x = remain - y * dimensions[0];
          const axis = template.side >>> 1;
          const boundary = (axis === 0 ? y : x)
            + resolution * (axis === 2 ? y : z);
          if (sparseAirOwners[boundary] !== SPARSE_CM12_IBO_TRA_INVALID
            && sparseAirOwners[boundary] !== row) {
            throw new Error(`ITR1 template ${template.id} sparse-air collision at ${boundary}`);
          }
          sparseAirOwners[boundary] = row;
        }
      }
    }
    const offsets = [0], entries: number[] = [];
    for (const rows of byBoundary) {
      for (const row of rows) {
        const rowAt = 8 + 7 * row, packed = template.words[rowAt + 1]!;
        const first = packed & 0x007f_ffff, count = packed >>> 23;
        let ownerTerm = 0xf;
        for (let term = 0; term < count; term += 1) {
          if (f32(template.words[termBase + 2 * (first + term) + 1]!) > 0) {
            ownerTerm = term;break;
          }
        }
        if (row >= 0x1000 || ownerTerm >= 0x10) throw new Error("ITR1 packed entry overflow");
        entries.push(row | (ownerTerm << 12));
      }
      offsets.push(entries.length);
    }
    const offsetsBase = at;at += offsets.length;
    const entriesBase = at;at += entries.length;
    const ownersBase = at;at += owners.length;
    const sparseAirOwnersBase = at;at += sparseAirOwners.length;
    records.push({ offsets, entries, owners, sparseAirOwners, resolution,
      offsetsBase, entriesBase, ownersBase, sparseAirOwnersBase });
  }
  const totalWords = align64(at), words = new Uint32Array(totalWords - baseWords);
  const put = (absolute: number, values: readonly number[]) =>
    words.set(values, absolute - baseWords);
  put(baseWords, [SPARSE_CM12_IBO_TRA_MAGIC, 2, templates.length,
    directoryBaseWords, totalWords, 0, 0, 0]);
  records.forEach((record, template) => {
    put(directoryBaseWords + SPARSE_CM12_IBO_TRA_DIRECTORY_WORDS * template,
      [record.offsetsBase, record.entriesBase, record.ownersBase, record.resolution]);
    put(record.offsetsBase, record.offsets);put(record.entriesBase, record.entries);
    put(record.ownersBase, record.owners);
    put(directoryBaseWords + SPARSE_CM12_IBO_TRA_DIRECTORY_WORDS * template + 4,
      [record.sparseAirOwnersBase]);
    put(record.sparseAirOwnersBase, record.sparseAirOwners);
  });
  return Object.freeze({ layout: Object.freeze({ baseWords, templateCount: templates.length,
    directoryBaseWords, totalWords, totalBytes: 4 * (totalWords - baseWords) }), words });
}
