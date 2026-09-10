import type { SparseAtlasCompositeCell, SparseAtlasGradientRow, SparseAtlasGradientTerm } from "./sparse-atlas-composite-projection";
import { sparseBrickMaximumFine, sparseBrickSpan, type SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";

/** Typed chunks keep construction residency proportional to numeric payload,
 * never to a JavaScript object per native cell, row, term or requirement. */
class NumericChunks {
  private readonly chunks: Float64Array[] = [];
  length = 0;
  push(value: number) {
    const chunk = this.length >>> 14;
    this.chunks[chunk] ??= new Float64Array(16384);
    this.chunks[chunk]![this.length++ & 16383] = value;
  }
  get(index: number) { return this.chunks[index >>> 14]![index & 16383]!; }
}

/** Array-shaped read view used by the remaining CPU certification oracles.
 * Indexing materializes one short-lived value; the backing graph is numeric. */
function arrayView<T>(length: () => number, get: (index: number) => T,
  push?: (value: T) => void): T[] {
  const view = {
    get length() { return length(); },
    push(...values: T[]) { for (const value of values) push!(value); return length(); },
    *[Symbol.iterator]() { for (let i = 0; i < length(); i++) yield get(i); },
    forEach(visit: (value: T, index: number) => void) {
      for (let i = 0; i < length(); i++) visit(get(i), i);
    },
    map<U>(visit: (value: T, index: number) => U) {
      const result: U[] = []; for (let i = 0; i < length(); i++) result.push(visit(get(i), i)); return result;
    },
    flatMap<U>(visit: (value: T, index: number) => readonly U[]) {
      const result: U[] = []; for (let i = 0; i < length(); i++) result.push(...visit(get(i), i)); return result;
    },
  };
  return new Proxy(view, { get(target, property, receiver) {
    if (typeof property === "string" && /^(0|[1-9][0-9]*)$/.test(property)) {
      const index = Number(property); return index < length() ? get(index) : undefined;
    }
    return Reflect.get(target, property, receiver);
  } }) as unknown as T[];
}

interface CellRange {
  readonly first: number;
  count: number;
  readonly leaf: number;
  readonly key: number;
  readonly resolution: number;
  readonly coordinate: readonly [number, number, number];
  readonly lower: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
  readonly dimensions: readonly [number, number, number];
  readonly width: number;
  readonly stableBase: number;
}

interface RowArchetype {
  readonly kind: SparseAtlasGradientRow["kind"];
  readonly axis: SparseAtlasGradientRow["axis"];
  readonly center: readonly [number, number, number];
  readonly area: number;
  readonly distance: number;
  readonly dualWeight: number;
  readonly exteriorPhi: number | undefined;
  readonly terms: readonly { readonly local: number; readonly target: boolean; readonly coefficient: number }[];
}

class CellView implements SparseAtlasCompositeCell {
  constructor(readonly id: number, private readonly range: CellRange,
    readonly density: number, readonly gamma: number) {}
  get brickKey() { return this.range.key; }
  get brickCoordinate() { return this.range.coordinate; }
  get brickResolution() { return this.range.resolution as SparseAtlasCompositeCell["brickResolution"]; }
  private get x() { return (this.id - this.range.first) % this.range.dimensions[0]; }
  private get y() { return Math.floor((this.id - this.range.first) / this.range.dimensions[0]) % this.range.dimensions[1]; }
  private get z() { return Math.floor((this.id - this.range.first) / (this.range.dimensions[0] * this.range.dimensions[1])); }
  get local(): readonly [number, number, number] { return [this.x, this.y, this.z]; }
  get localIndex() { return this.x + this.range.resolution * (this.y + this.range.resolution * this.z); }
  get stableLeafId() { return this.range.stableBase + this.localIndex; }
  get minimumFine(): readonly [number, number, number] {
    const r = this.range;
    return [r.lower[0] + this.x * r.width, r.lower[1] + this.y * r.width, r.lower[2] + this.z * r.width];
  }
  get maximumFine(): readonly [number, number, number] {
    const r = this.range, lower = this.minimumFine;
    return [Math.min(lower[0] + r.width, r.maximum[0]), Math.min(lower[1] + r.width, r.maximum[1]), Math.min(lower[2] + r.width, r.maximum[2])];
  }
  get widthsFine(): readonly [number, number, number] {
    const lower = this.minimumFine, upper = this.maximumFine;
    return [upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]];
  }
  get centerFine(): readonly [number, number, number] {
    const lower = this.minimumFine, upper = this.maximumFine;
    return [.5 * (lower[0] + upper[0]), .5 * (lower[1] + upper[1]), .5 * (lower[2] + upper[2])];
  }
  get volume() { const width = this.widthsFine; return width[0] * width[1] * width[2]; }
  get volumeFineCells() { return this.volume; }
}

export interface SparseCM12TemplateExpansion {
  readonly words: Uint32Array;
  readonly cellRangeCount: number;
  readonly rowCount: number;
  readonly rangeBase: number;
  readonly instanceBase: number;
  readonly archetypeBase: number;
  readonly archetypeCount: number;
  readonly maximumRangeCells: number;
}

/** Intern one exact relative operator, then retain just three numeric addresses
 * per instance. Both CPU certification and GPU placement consume this recipe. */
export class SparseCM12TemplateArchetypes {
  private readonly ranges: CellRange[] = [];
  private readonly rangeByConfiguration = new Map<number, number>();
  private readonly cellRangeIds = new NumericChunks();
  private readonly scalar = new NumericChunks();
  private readonly patterns: RowArchetype[] = [];
  private readonly patternByShape = new Map<string, number>();
  private readonly instances = new NumericChunks();
  private readonly instanceKeys = new Set<number>();
  private readonly leafByKey: Map<number, number>;
  private rowOrder?: Uint32Array;
  private readonly rangeKeyCapacity: number;
  readonly cells: SparseAtlasCompositeCell[];
  readonly rows: SparseAtlasGradientRow[];
  readonly requirements: (readonly number[])[];

  constructor(private readonly atlas: SparseAdaptiveMassAtlas) {
    this.leafByKey = new Map(atlas.bricks.map((brick, leaf) => [brick.key, leaf]));
    this.rangeKeyCapacity = 1 + atlas.bricks.length * (Math.log2(atlas.brickFineResolution) + 1);
    this.cells = arrayView(() => this.cellRangeIds.length, id => this.cell(id), cell => this.appendCell(cell));
    this.rows = arrayView(() => this.instances.length / 3, row => this.row(row));
    this.requirements = arrayView(() => this.instances.length / 3, row => this.rowRequirements(row));
  }

  /** Compatible with the old exact-number key, without a Map entry per cell. */
  readonly cellIds = {
    get: (key: number) => {
      const brickKey = Math.floor(key / 0x100000);
      const resolution = Math.floor((key % 0x100000) / 0x1000);
      const local = key % 0x1000;
      const range = this.ranges[this.rangeByConfiguration.get(brickKey * 32 + resolution)!];
      if (!range) return undefined;
      const x = local % resolution, y = Math.floor(local / resolution) % resolution;
      const z = Math.floor(local / (resolution * resolution));
      if (x >= range.dimensions[0] || y >= range.dimensions[1] || z >= range.dimensions[2]) return undefined;
      const ordinal = x + range.dimensions[0] * (y + range.dimensions[1] * z);
      return ordinal < range.count ? range.first + ordinal : undefined;
    },
    set: (_key: number, _id: number) => {},
    clear: () => {},
  };

  private appendCell(source: SparseAtlasCompositeCell) {
    const configuration = source.brickKey * 32 + source.brickResolution;
    let rangeId = this.rangeByConfiguration.get(configuration);
    if (rangeId === undefined) {
      rangeId = this.ranges.length;
      const brick = this.atlas.directory.get(source.brickKey)!;
      const width = this.atlas.brickFineResolution * sparseBrickSpan(brick) / source.brickResolution;
      const lower = brick.coordinate.map(q => this.atlas.brickFineResolution * q) as [number, number, number];
      const maximum = ([0, 1, 2] as const).map(axis => sparseBrickMaximumFine(this.atlas, brick, axis)) as [number, number, number];
      const dimensions = maximum.map((q, axis) => Math.min(source.brickResolution,
        Math.ceil((q - lower[axis]!) / width))) as [number, number, number];
      this.ranges.push({ first: this.cellRangeIds.length, count: 0,
        leaf: this.leafByKey.get(source.brickKey)!, key: source.brickKey,
        resolution: source.brickResolution, coordinate: brick.coordinate,
        lower, maximum, dimensions, width, stableBase: source.stableLeafId - source.localIndex });
      this.rangeByConfiguration.set(configuration, rangeId);
    }
    const range = this.ranges[rangeId]!;
    if (range.first + range.count !== this.cellRangeIds.length)
      throw new Error("CM12 template cell configuration is not contiguous");
    this.cellRangeIds.push(rangeId);
    this.scalar.push(source.density); this.scalar.push(source.gamma); range.count++;
  }

  private cell(id: number): SparseAtlasCompositeCell {
    const range = this.ranges[this.cellRangeIds.get(id)]!;
    return new CellView(id, range, this.scalar.get(2 * id), this.scalar.get(2 * id + 1));
  }

  /** Returns false for the exact same physical row encountered in a halo. */
  appendRow(source: SparseAtlasGradientRow, terms: readonly SparseAtlasGradientTerm[],
    maximumRows = Number.POSITIVE_INFINITY, remap?: (cell: number) => number): boolean {
    const sourceRange = this.cellRangeIds.get(remap ? remap(terms[0]!.cellId) : terms[0]!.cellId);
    let targetRange = -1;
    const normalized = terms.map(term => {
      const cellId = remap ? remap(term.cellId) : term.cellId;
      const rangeId = this.cellRangeIds.get(cellId);
      if (rangeId !== sourceRange) {
        if (targetRange >= 0 && rangeId !== targetRange)
          throw new Error("CM12 template row spans more than two leaf/rung ranges");
        targetRange = rangeId;
      }
      return { local: cellId - this.ranges[rangeId]!.first,
        target: rangeId !== sourceRange, coefficient: term.coefficient };
    });
    const origin = this.ranges[sourceRange]!.lower;
    const center = source.centerFine.map((q, axis) => q - origin[axis]!) as [number, number, number];
    const shape = [source.kind, source.axis, ...center, source.area, source.distance,
      source.dualWeight, source.exteriorPhi ?? .5,
      ...normalized.flatMap(term => [Number(term.target), term.local, term.coefficient])].join("/");
    let pattern = this.patternByShape.get(shape);
    if (pattern === undefined) {
      pattern = this.patterns.length;
      this.patterns.push({ kind: source.kind, axis: source.axis, center,
        area: source.area, distance: source.distance, dualWeight: source.dualWeight,
        exteriorPhi: source.exteriorPhi, terms: normalized });
      this.patternByShape.set(shape, pattern);
    }
    const key = (pattern * this.rangeKeyCapacity + sourceRange) * this.rangeKeyCapacity + targetRange + 1;
    if (!Number.isSafeInteger(key)) throw new Error("CM12 archetype instance identity exceeds exact integer range");
    if (this.instanceKeys.has(key)) return false;
    if (this.instances.length / 3 >= maximumRows) return true;
    this.instanceKeys.add(key);
    this.instances.push(pattern); this.instances.push(sourceRange); this.instances.push(targetRange);
    return true;
  }

  private instance(row: number) { return 3 * (this.rowOrder?.[row] ?? row); }

  private row(row: number): SparseAtlasGradientRow {
    const at = this.instance(row);
    const pattern = this.patterns[this.instances.get(at)]!;
    const source = this.ranges[this.instances.get(at + 1)]!;
    const target = this.ranges[this.instances.get(at + 2)];
    return { id: row, kind: pattern.kind, axis: pattern.axis,
      centerFine: pattern.center.map((q, axis) => q + source.lower[axis]!) as [number, number, number],
      area: pattern.area, areaFineCells2: pattern.area, distance: pattern.distance,
      centerDistanceFine: pattern.distance, dualWeight: pattern.dualWeight,
      exteriorPhi: pattern.exteriorPhi,
      terms: pattern.terms.map(term => ({ cellId: (term.target ? target! : source).first + term.local,
        coefficient: term.coefficient })) };
  }

  private rowRequirements(row: number): readonly number[] {
    const at = this.instance(row);
    const source = this.ranges[this.instances.get(at + 1)]!;
    const target = this.ranges[this.instances.get(at + 2)];
    const first = source.leaf * 32 + source.resolution;
    return target ? [first, target.leaf * 32 + target.resolution] : [first];
  }

  requirementCount(row: number): number { return this.instances.get(this.instance(row) + 2) < 0 ? 1 : 2; }

  private visitRows(visit: (row: number, pattern: RowArchetype, source: CellRange, target: CellRange | undefined) => void) {
    for (let row = 0; row < this.rows.length; row++) {
      const at = this.instance(row);
      visit(row, this.patterns[this.instances.get(at)]!, this.ranges[this.instances.get(at + 1)]!,
        this.ranges[this.instances.get(at + 2)]);
    }
  }

  incidenceCounts(incidence: Uint32Array, edges: Uint32Array): number {
    let count = 0;
    this.visitRows((_row, pattern, source, target) => {
      count += pattern.terms.length;
      for (const term of pattern.terms) {
        const cell = (term.target ? target! : source).first + term.local;
        incidence[cell]++; edges[cell] += pattern.terms.length - 1;
      }
    });
    return count;
  }

  candidateFaces(levels: readonly number[]) {
    const configurations = new Uint32Array(this.atlas.bricks.length * levels.length * 6);
    let patchCount = 0;
    for (let leaf = 0; leaf < this.atlas.bricks.length; leaf++) for (let level = 0; level < levels.length; level++) {
      for (let side = 0; side < 6; side++) {
        configurations[(leaf * levels.length + level) * 6 + side] = patchCount;
        patchCount += levels[level]! ** 2 + 1;
      }
    }
    const counts = new Uint32Array(patchCount);
    const visitMappings = (visit: (configuration: number, boundary: number, row: number) => void) => this.visitRows((row, pattern, source, target) => {
      const coordinate = pattern.center[pattern.axis] + source.lower[pattern.axis];
      for (let which = 0; which < (target ? 2 : 1); which++) {
        const range = which === 0 ? source : target!;
        let side = -1;
        if (Math.abs(coordinate - range.lower[pattern.axis]) <= 1e-4) side = 2 * pattern.axis;
        else if (Math.abs(coordinate - range.maximum[pattern.axis]) <= 1e-4) side = 2 * pattern.axis + 1;
        if (side < 0) continue;
        let local = Number.POSITIVE_INFINITY;
        for (const term of pattern.terms) if (term.target === (which === 1)) local = Math.min(local, term.local);
        if (!Number.isFinite(local)) continue;
        const [nx, ny] = range.dimensions;
        const x = local % nx, y = Math.floor(local / nx) % ny, z = Math.floor(local / (nx * ny));
        const u = pattern.axis === 0 ? y : x, v = pattern.axis === 2 ? y : z;
        visit((range.leaf * levels.length + levels.indexOf(range.resolution)) * 6 + side, u + range.resolution * v, row);
      }
    });
    visitMappings((configuration, boundary) => { counts[configurations[configuration]! + boundary + 1]++; });
    const patchOffsets = new Uint32Array(patchCount);
    let total = 0;
    for (let configuration = 0; configuration < configurations.length; configuration++) {
      const first = configurations[configuration]!, end = configurations[configuration + 1] ?? patchCount;
      patchOffsets[first] = total;
      for (let at = first + 1; at < end; at++) { total += counts[at]!; patchOffsets[at] = total; }
    }
    const rows = new Uint32Array(total), cursors = patchOffsets.slice();
    visitMappings((configuration, boundary, row) => { rows[cursors[configurations[configuration]! + boundary]++] = row; });
    return { configurations, patchOffsets, rows };
  }

  writeNativeShadow(words: Uint32Array, requirements: ArrayLike<number>) {
    const f = new Float32Array(words.buffer, words.byteOffset, words.length);
    const cellBase = words[6]!, rowBase = words[7]!, termBase = words[8]!, count = this.rows.length;
    for (const range of this.ranges) for (let ordinal = 0; ordinal < range.count; ordinal++) {
      const [nx, ny] = range.dimensions;
      const q = [ordinal % nx, Math.floor(ordinal / nx) % ny, Math.floor(ordinal / (nx * ny))];
      const at = cellBase + 8 * (range.first + ordinal);
      let volume = 1;
      for (let axis = 0; axis < 3; axis++) {
        const lower = range.lower[axis]! + q[axis]! * range.width;
        const upper = Math.min(lower + range.width, range.maximum[axis]!);
        f[at + axis] = .5 * (lower + upper); f[at + 4 + axis] = upper - lower; volume *= upper - lower;
      }
      f[at + 3] = volume; words[at + 7] = range.leaf * 32 + range.resolution;
    }
    let nextTerm = 0;
    const kinds = { "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3 };
    this.visitRows((row, pattern, source, target) => {
      if (nextTerm >= 0x800000 || pattern.terms.length >= 512 || requirements[row]! >= 0x10000000)
        throw new Error("CM12 archetype shadow exceeds packed topology address ABI");
      words[rowBase + row] = nextTerm | (pattern.terms.length << 23);
      words[rowBase + count + row] = requirements[row]! | (kinds[pattern.kind] << 28) | (pattern.axis << 30);
      f[rowBase + 2 * count + row] = pattern.dualWeight; f[rowBase + 3 * count + row] = pattern.area;
      f[rowBase + 4 * count + row] = pattern.distance; f[rowBase + 5 * count + row] = pattern.exteriorPhi ?? .5;
      for (let axis = 0; axis < 3; axis++) f[rowBase + (6 + axis) * count + row] = pattern.center[axis]! + source.lower[axis]!;
      for (const term of pattern.terms) {
        words[termBase + 2 * nextTerm] = (term.target ? target! : source).first + term.local;
        f[termBase + 2 * nextTerm + 1] = term.coefficient; nextTerm++;
      }
    });
  }

  writeIncidenceShadow(words: Uint32Array, at: number, cursors: Uint32Array) {
    let ordinal = 0;
    this.visitRows((row, pattern, source, target) => {
      for (const term of pattern.terms) {
        const cell = (term.target ? target! : source).first + term.local, output = at + 2 * cursors[cell]++;
        words[output] = row; words[output + 1] = ordinal++;
      }
    });
  }

  writeRequirementsShadow(words: Uint32Array, at: number) {
    this.visitRows((_row, _pattern, source, target) => {
      words[at++] = target ? 2 : 1; words[at++] = source.leaf * 32 + source.resolution;
      if (target) words[at++] = target.leaf * 32 + target.resolution;
    });
  }

  writePressureEdgesShadow(words: Uint32Array, at: number, cursors: Uint32Array) {
    const f = new Float32Array(words.buffer, words.byteOffset, words.length);
    this.visitRows((row, pattern, source, target) => {
      for (const own of pattern.terms) {
        const cell = (own.target ? target! : source).first + own.local;
        for (const other of pattern.terms) {
          const neighbor = (other.target ? target! : source).first + other.local;
          if (cell === neighbor) continue;
          const output = at + 3 * cursors[cell]++;
          words[output] = row; words[output + 1] = neighbor;
          f[output + 2] = own.coefficient * pattern.dualWeight * other.coefficient;
        }
      }
    });
  }

  initialScalars(kind: 0 | 1): Float32Array {
    const result = new Float32Array(this.cells.length);
    for (let id = 0; id < result.length; id++) result[id] = this.scalar.get(2 * id + kind);
    return result;
  }

  orderRows(brickCount: number, levels: readonly number[]) {
    this.instanceKeys.clear(); this.patternByShape.clear();
    const count = this.rows.length;
    const counts = new Uint32Array(brickCount * levels.length);
    const bucketFor = (row: number) => {
      const at = this.instance(row), source = this.ranges[this.instances.get(at + 1)]!;
      const target = this.ranges[this.instances.get(at + 2)];
      const owner = !target || source.leaf < target.leaf ? source : target;
      return owner.leaf * levels.length + levels.indexOf(owner.resolution);
    };
    for (let row = 0; row < count; row++) counts[bucketFor(row)]++;
    const offsets = new Uint32Array(counts.length + 1);
    let maximumOwnedRowCount = 0;
    for (let bucket = 0; bucket < counts.length; bucket++) {
      offsets[bucket + 1] = offsets[bucket]! + counts[bucket]!;
      maximumOwnedRowCount = Math.max(maximumOwnedRowCount, counts[bucket]!);
    }
    const cursors = offsets.slice(), oldToNew = new Uint32Array(count), order = new Uint32Array(count);
    for (let row = 0; row < count; row++) {
      const next = cursors[bucketFor(row)]++;
      oldToNew[row] = next; order[next] = row;
    }
    this.rowOrder = order;
    return { rows: this.rows, requirements: this.requirements, offsets, oldToNew, maximumOwnedRowCount };
  }

  expansion(rowRequirementOffsets: ArrayLike<number>): SparseCM12TemplateExpansion {
    const rangeBase = 0, instanceBase = 12 * this.ranges.length;
    const archetypeBase = instanceBase + 5 * this.rows.length;
    const patternOffsets = new Uint32Array(this.patterns.length);
    let size = archetypeBase;
    this.patterns.forEach((pattern, id) => { patternOffsets[id] = size; size += 10 + 2 * pattern.terms.length; });
    const words = new Uint32Array(size), f = new Float32Array(words.buffer);
    let maximumRangeCells = 0;
    this.ranges.forEach((range, id) => {
      const at = 12 * id; f.set(range.lower, at); f[at + 3] = range.width;
      words.set(range.dimensions, at + 4); words[at + 7] = range.first;
      f.set(range.maximum, at + 8); words[at + 11] = range.leaf * 32 + range.resolution;
      maximumRangeCells = Math.max(maximumRangeCells, range.count);
    });
    const kinds = { "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3 };
    this.patterns.forEach((pattern, id) => {
      const at = patternOffsets[id]!;
      words[at] = pattern.terms.length;
      words[at + 1] = (kinds[pattern.kind] << 28) | (pattern.axis << 30);
      f[at + 2] = pattern.dualWeight; f[at + 3] = pattern.area; f[at + 4] = pattern.distance;
      f[at + 5] = pattern.exteriorPhi ?? .5; f.set(pattern.center, at + 6);
      for (let local = 0; local < pattern.terms.length; local++) {
        const term = pattern.terms[local]!;
        words[at + 10 + 2 * local] = term.local | (Number(term.target) << 31);
        f[at + 11 + 2 * local] = term.coefficient;
      }
    });
    let firstTerm = 0;
    for (let row = 0; row < this.rows.length; row++) {
      const at = this.instance(row), pattern = this.instances.get(at), output = instanceBase + 5 * row;
      words[output] = patternOffsets[pattern]!;
      words[output + 1] = this.instances.get(at + 1);
      words[output + 2] = this.instances.get(at + 2) >>> 0;
      words[output + 3] = firstTerm; words[output + 4] = rowRequirementOffsets[row]!;
      firstTerm += this.patterns[pattern]!.terms.length;
    }
    return { words, rangeBase, instanceBase, archetypeBase, cellRangeCount: this.ranges.length,
      rowCount: this.rows.length, archetypeCount: this.patterns.length, maximumRangeCells };
  }
}
