import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, sparseBrickSpan,
  type SparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick, type SparseBrickResolution } from "./sparse-brick-atlas";
import { compileSparseCM12StableLeafFaceNeighbors } from "./sparse-cm12-factored-aei-topology";

export interface SparseCM12GenerationIntent {
  readonly resolution: SparseBrickResolution;
  /** Fresh accepted bulk evidence, never an inference from placeholder fields. */
  readonly mergeable: boolean;
  /** Physical demand can outgrow the local rung ladder and split macro coverage. */
  readonly maximumCellWidth?: number;
}

/** Build one bounded dyadic change and close physical 2:1 grading. No dense
 * logical-world coverage is allocated. Field transfer is a separate device operation. */
export function planSparseCM12ResidentGeneration(atlas: SparseAdaptiveMassAtlas,
  active: ReadonlySet<number>, intents: ReadonlyMap<number, SparseCM12GenerationIntent>,
  limits: { maximumLeaves: number; maximumCells: number; maximumSpanBricks: number },
): { status: "ready"; atlas: SparseAdaptiveMassAtlas; active: ReadonlySet<number> }
  | { status: "deferred"; leaves: number; cells: number } | undefined {
  const B = atlas.brickFineResolution;
  const seed = (brick: SparseAdaptiveMassBrick, resolution: SparseBrickResolution) => ({
    ...brick, resolution, density: new Float64Array(resolution ** 3).fill(brick.density[0] ?? 0),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  // Inactive leaves own no accepted field work. Reclaim their static backing;
  // the ordinary signed frontier allocator can admit them again on demand.
  let bricks: SparseAdaptiveMassBrick[] = atlas.bricks.filter(brick => active.has(brick.key)).map(brick => seed(brick, intents.get(brick.key)?.resolution ?? brick.resolution));
  const nextActive = new Set(active);
  const demandedWidths = new Map<number, number>();
  for (const [key, intent] of intents) if (intent.maximumCellWidth !== undefined) {
    if (!Number.isFinite(intent.maximumCellWidth) || intent.maximumCellWidth < 1)
      throw new Error("CM12 maximum cell width must be finite and at least one");
    demandedWidths.set(key, intent.maximumCellWidth);
  }
  const groups = new Map<string, SparseAdaptiveMassBrick[]>();
  for (const brick of bricks) {
    const span = sparseBrickSpan(brick);
    if (demandedWidths.has(brick.key) || brick.unclipped || brick.resolution !== 1 || !active.has(brick.key) || !intents.get(brick.key)?.mergeable
      || span * 2 > limits.maximumSpanBricks) continue;
    const origin = brick.coordinate.map(q => Math.floor(q / (2 * span)) * 2 * span);
    const key = `${span}/${origin.join("/")}`;
    let group = groups.get(key); if (!group) groups.set(key, group = []);
    group.push(brick);
  }
  const removed = new Set<number>(), parents: SparseAdaptiveMassBrick[] = [];
  for (const group of groups.values()) {
    if (group.length !== 8) continue;
    const span = 2 * sparseBrickSpan(group[0]!);
    const origin = group[0]!.coordinate.map(q => Math.floor(q / span) * span) as [number, number, number];
    // Partial-domain siblings retain their explicit boundary cells.
    if (origin.some((q, axis) => (q + span) * B > atlas.dimensions[axis]!)) continue;
    for (const brick of group) { removed.add(brick.key); nextActive.delete(brick.key); }
    const key = sparseAtlasBrickKey(origin, atlas);
    parents.push({ key, coordinate: origin, spanBricks: span, resolution: 1,
      density: new Float64Array([1]), gamma: new Float64Array([1]) });
    nextActive.add(key);
  }
  bricks = [...bricks.filter(b => !removed.has(b.key)), ...parents];
  const split = (brick: SparseAdaptiveMassBrick): SparseAdaptiveMassBrick[] => {
    const half = sparseBrickSpan(brick) / 2;
    const children: SparseAdaptiveMassBrick[] = [];
    const wasActive = nextActive.delete(brick.key);
    const demand = demandedWidths.get(brick.key);
    demandedWidths.delete(brick.key);
    for (let child=0; child<8; child++) {
      const coordinate = brick.coordinate.map((q, axis) => q + ((child >>> axis) & 1) * half) as [number, number, number];
      if (!brick.unclipped && coordinate.some((q, axis) => q * B >= atlas.dimensions[axis]!)) continue;
      const key = sparseAtlasBrickKey(coordinate, atlas);
      children.push(seed({ ...brick, key, coordinate, spanBricks: half }, Math.max(1, brick.resolution / 2) as SparseBrickResolution));
      if (wasActive) nextActive.add(key);
      if (demand !== undefined) demandedWidths.set(key, demand);
    }
    return children;
  };
  for (;;) {
    const cells = bricks.reduce((n,b) => n + b.resolution ** 3, 0);
    if (bricks.length > limits.maximumLeaves || cells > limits.maximumCells) {
      return { status: "deferred", leaves: bricks.length, cells };
    }
    let demanded = false;
    bricks = bricks.flatMap(brick => {
      const width = demandedWidths.get(brick.key);
      if (width === undefined || B * sparseBrickSpan(brick) / brick.resolution <= width) return [brick];
      demanded = true;
      return brick.resolution < B ? [seed(brick, (2 * brick.resolution) as SparseBrickResolution)] : split(brick);
    });
    // Recheck budgets before constructing adjacency for a larger generation.
    if (demanded) continue;
    const neighbors = compileSparseCM12StableLeafFaceNeighbors({
      coordinates: bricks.map(b => b.coordinate), spans: bricks.map(sparseBrickSpan) });
    let changed = false;
    const toSplit = new Set<number>();
    for (let leaf=0; leaf<bricks.length; leaf++) for (const other of neighbors[leaf]!) {
      const a = bricks[leaf]!, b = bricks[other]!;
      const aSpan = sparseBrickSpan(a), bSpan = sparseBrickSpan(b);
      // The boundary image has four dyadic patches per face. Split coverage
      // as well as grading cell widths so that each shared face fits that ABI.
      if (Math.max(aSpan, bSpan) > 2 * Math.min(aSpan, bSpan)) {
        toSplit.add(aSpan > bSpan ? leaf : other); changed = true;
      }
      const aw = B * sparseBrickSpan(a) / a.resolution, bw = B * sparseBrickSpan(b) / b.resolution;
      if (Math.max(aw,bw) <= 2 * Math.min(aw,bw)) continue;
      const coarser = aw > bw ? leaf : other, brick = bricks[coarser]!;
      if (brick.resolution < B) bricks[coarser] = seed(brick, (2 * brick.resolution) as SparseBrickResolution);
      else if (sparseBrickSpan(brick) > 1) toSplit.add(coarser);
      else throw new Error("CM12 physical grading has no dyadic refinement");
      changed = true;
    }
    if (toSplit.size) bricks = bricks.flatMap((brick, id) => toSplit.has(id) ? split(brick) : [brick]);
    if (!changed) break;
  }
  const unchanged = bricks.length === atlas.bricks.length && bricks.every(b => {
    const old = atlas.directory.get(b.key);
    return old && sparseBrickSpan(old) === sparseBrickSpan(b) && old.resolution === b.resolution;
  });
  if (unchanged) return undefined;
  return { status: "ready", atlas: createSparseAdaptiveMassAtlas(atlas.dimensions, bricks, atlas.generation + 1, B, atlas.signedCoordinates), active: nextActive };
}
