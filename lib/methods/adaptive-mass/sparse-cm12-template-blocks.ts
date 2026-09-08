import {
  createSparseAdaptiveMassAtlas, sparseBrickMaximumFine, sparseBrickSpan,
  type SparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick,
} from "./sparse-brick-atlas";
import {
  buildSparseAtlasCompositeGrid, type SparseAtlasGradientRow,
} from "./sparse-atlas-composite-projection";

type Axis = 0 | 1 | 2;
type Vec3 = readonly [number, number, number];

/** A native operator block has no absolute cell IDs or leaf identities. Its
 * term addresses identify one participating leaf and that leaf's native slot. */
export interface SparseCM12TemplateBlockRow {
  readonly kind: SparseAtlasGradientRow["kind"];
  readonly axis: Axis;
  readonly centerFine: Vec3;
  readonly area: number;
  readonly distance: number;
  readonly dualWeight: number;
  readonly exteriorPhi: number | undefined;
  readonly terms: readonly { readonly slot: number; readonly local: number; readonly coefficient: number }[];
}

export interface SparseCM12TemplateBlockPlacement {
  readonly originFine: Vec3;
  readonly rows: readonly SparseCM12TemplateBlockRow[];
}

/** Visit complete row blocks in exactly the geometric oracle's public order:
 * every brick interior first, then each sorted brick's positive interfaces,
 * positive air and negative air, in axis order. */
export function visitSparseCM12TemplateBlocks(atlas: SparseAdaptiveMassAtlas,
  cache: SparseCM12TemplateBlockCache,
  visit: (placement: SparseCM12TemplateBlockPlacement, bricks: readonly SparseAdaptiveMassBrick[]) => void,
  include: (kind: "interior" | "interface" | "air", bricks: readonly SparseAdaptiveMassBrick[], axis?: Axis) => boolean,
): void {
  const bricks = [...atlas.bricks].sort((a, b) => a.key - b.key);
  const minus = [new Map<number, SparseAdaptiveMassBrick[]>(), new Map<number, SparseAdaptiveMassBrick[]>(), new Map<number, SparseAdaptiveMassBrick[]>()];
  const plus = [new Map<number, SparseAdaptiveMassBrick[]>(), new Map<number, SparseAdaptiveMassBrick[]>(), new Map<number, SparseAdaptiveMassBrick[]>()];
  for (const brick of bricks) {
    if (include("interior", [brick])) visit(cache.interior(atlas, brick), [brick]);
    for (const axis of [0, 1, 2] as const) {
      const lo = brick.coordinate[axis], hi = lo + sparseBrickSpan(brick);
      if (!minus[axis]!.has(lo)) minus[axis]!.set(lo, []);
      if (!plus[axis]!.has(hi)) plus[axis]!.set(hi, []);
      minus[axis]!.get(lo)!.push(brick); plus[axis]!.get(hi)!.push(brick);
    }
  }
  const overlap = (a: SparseAdaptiveMassBrick, b: SparseAdaptiveMassBrick, axis: Axis) =>
    ([0, 1, 2] as const).every(t => t === axis || Math.min(a.coordinate[t] + sparseBrickSpan(a),
      b.coordinate[t] + sparseBrickSpan(b)) > Math.max(a.coordinate[t], b.coordinate[t]));
  for (const brick of bricks) for (const axis of [0, 1, 2] as const) {
    const positive = (minus[axis]!.get(brick.coordinate[axis] + sparseBrickSpan(brick)) ?? [])
      .filter(neighbor => overlap(brick, neighbor, axis));
    for (const neighbor of positive) if (include("interface", [brick, neighbor], axis))
      visit(cache.interface(atlas, brick, neighbor, axis), [brick, neighbor]);
    if (include("air", [brick], axis)) {
      visit(cache.air(atlas, brick, axis, 1, positive), [brick]);
      const negative = (plus[axis]!.get(brick.coordinate[axis]) ?? [])
        .filter(neighbor => overlap(brick, neighbor, axis));
      visit(cache.air(atlas, brick, axis, -1, negative), [brick]);
    }
  }
}

/** The existing geometric
 * oracle compiles each distinct local physical shape once. Placement then
 * reuses the same relative rows for any leaf IDs and translated coordinates.
 * Each preparation owns one cache; no coordinates or generation snapshots
 * survive in a process-global cache. GPU placement emits the native records. */
export class SparseCM12TemplateBlockCache {
  private readonly blocks = new Map<string, readonly SparseCM12TemplateBlockRow[]>();
  private referenceBuilds = 0;
  private referenceCells = 0;
  private referenceRows = 0;
  get statistics() {
    return { blocks: this.blocks.size, referenceBuilds: this.referenceBuilds,
      referenceCells: this.referenceCells, referenceRows: this.referenceRows };
  }

  private shape(atlas: SparseAdaptiveMassAtlas, brick: SparseAdaptiveMassBrick,
    origin: Vec3, includeResolution: boolean): string {
    return [sparseBrickSpan(brick), includeResolution ? brick.resolution : 0,
      ...brick.coordinate.map((q, axis) => q * atlas.brickFineResolution - origin[axis]!),
      ...([0, 1, 2] as const).map(axis => sparseBrickMaximumFine(atlas, brick, axis)
        - brick.coordinate[axis] * atlas.brickFineResolution)].join(",");
  }

  private compile(atlas: SparseAdaptiveMassAtlas, bricks: readonly SparseAdaptiveMassBrick[],
    key: string, accept: (row: SparseAtlasGradientRow) => boolean): SparseCM12TemplateBlockPlacement {
    const originFine = bricks[0]!.coordinate.map(q => q * atlas.brickFineResolution) as [number, number, number];
    let rows = this.blocks.get(key);
    if (!rows) {
      // Keys, alignment and clipping stay valid in this reference build. Only
      // the participating leaves are present; no translated synthetic atlas
      // or changed domain bounds are needed to normalize an operator.
      const local = createSparseAdaptiveMassAtlas(atlas.dimensions, bricks,
        atlas.generation, atlas.brickFineResolution, atlas.signedCoordinates, false);
      const grid = buildSparseAtlasCompositeGrid(local);
      const slotByKey = new Map(bricks.map((brick, slot) => [brick.key, slot]));
      rows = grid.gradientRows.filter(accept).map(row => ({
        kind: row.kind, axis: row.axis,
        centerFine: row.centerFine.map((q, axis) => q - originFine[axis]!) as [number, number, number],
        area: row.area, distance: row.distance, dualWeight: row.dualWeight,
        exteriorPhi: row.exteriorPhi,
        terms: row.terms.map(term => ({ slot: slotByKey.get(grid.cells[term.cellId]!.brickKey)!,
          local: grid.cells[term.cellId]!.localIndex, coefficient: term.coefficient })),
      }));
      this.blocks.set(key, rows);
      this.referenceBuilds++;
      this.referenceCells += grid.cells.length;
      this.referenceRows += grid.gradientRows.length;
    }
    return { originFine, rows };
  }

  interior(atlas: SparseAdaptiveMassAtlas, brick: SparseAdaptiveMassBrick): SparseCM12TemplateBlockPlacement {
    const origin = brick.coordinate.map(q => q * atlas.brickFineResolution) as [number, number, number];
    const key = `i/${atlas.brickFineResolution}/${this.shape(atlas, brick, origin, true)}`;
    return this.compile(atlas, [brick], key, row => row.kind === "intra-brick");
  }

  interface(atlas: SparseAdaptiveMassAtlas, negative: SparseAdaptiveMassBrick,
    positive: SparseAdaptiveMassBrick, axis: Axis): SparseCM12TemplateBlockPlacement {
    const origin = negative.coordinate.map(q => q * atlas.brickFineResolution) as [number, number, number];
    const key = `f/${atlas.brickFineResolution}/${axis}/${this.shape(atlas, negative, origin, true)}`
      + `/${this.shape(atlas, positive, origin, true)}`;
    return this.compile(atlas, [negative, positive], key, row => row.axis === axis
      && (row.kind === "brick-face" || row.kind === "mixed-seam")
      && row.negativeBrickKey === negative.key && row.positiveBrickKey === positive.key);
  }

  air(atlas: SparseAdaptiveMassAtlas, brick: SparseAdaptiveMassBrick, axis: Axis,
    side: -1 | 1, neighbors: readonly SparseAdaptiveMassBrick[]): SparseCM12TemplateBlockPlacement {
    const origin = brick.coordinate.map(q => q * atlas.brickFineResolution) as [number, number, number];
    const key = `a/${atlas.brickFineResolution}/${axis}/${side}/${this.shape(atlas, brick, origin, true)}`
      + `/${neighbors.map(neighbor => this.shape(atlas, neighbor, origin, false)).sort().join("/")}`;
    // Sparse-air coverage depends only on physical neighbor footprints, so
    // the reference compiler needs one geometric cell in each neighbor.
    const geometry = neighbors.map(neighbor => ({ ...neighbor, resolution: 1 as const,
      density: new Float64Array(1), gamma: new Float64Array(1) }));
    return this.compile(atlas, [brick, ...geometry], key, row => row.kind === "sparse-air"
      && row.axis === axis && (side < 0 ? row.positiveBrickKey : row.negativeBrickKey) === brick.key);
  }
}
