import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, sparseBrickSpan,
  type SparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick, type SparseBrickResolution } from "./sparse-brick-atlas";
import { compileSparseCM12StableLeafFaceNeighbors } from "./sparse-cm12-factored-aei-topology";
import type { SparseCM12NewAirCoverage } from "./sparse-cm12-generation-transfer";

export interface SparseCM12GenerationIntent {
  readonly resolution: SparseBrickResolution;
  /** Fresh accepted bulk evidence, never an inference from placeholder fields. */
  readonly mergeable: boolean;
  /** Physical demand can outgrow the local rung ladder and split macro coverage. */
  readonly maximumCellWidth?: number;
  /** Hard physical floor; grading must coarsen neighbors rather than cross it. */
  readonly minimumCellWidth?: number;
  /** Preserve this accepted leaf's coverage and cell size during a rebuild. */
  readonly frozen?: boolean;
  /** Publish a demanded dry receiver with the new connected row graph. */
  readonly activate?: boolean;
}

/** Build one bounded dyadic change and close physical 2:1 grading. No dense
 * logical-world coverage is allocated. Field transfer is a separate device operation. */
export function planSparseCM12ResidentGeneration(atlas: SparseAdaptiveMassAtlas,
  active: ReadonlySet<number>, intents: ReadonlyMap<number, SparseCM12GenerationIntent>,
  limits: { maximumLeaves: number; maximumCells: number; maximumSpanBricks: number },
): { status: "ready"; atlas: SparseAdaptiveMassAtlas; active: ReadonlySet<number>;
    newAirCoverage: readonly SparseCM12NewAirCoverage[] }
  | { status: "deferred"; leaves: number; cells: number } | undefined {
  const B = atlas.brickFineResolution;
  const frozen = new Set([...intents].filter(([, intent]) => intent.frozen).map(([key]) => key));
  const newAirCoverage: SparseCM12NewAirCoverage[] = [];
  const minimumWidths = new Map<number, number>();
  for (const [key, intent] of intents) if (intent.minimumCellWidth !== undefined) {
    if (!Number.isFinite(intent.minimumCellWidth) || intent.minimumCellWidth < 1)
      throw new Error("CM12 minimum cell width must be finite and at least one");
    minimumWidths.set(key, intent.minimumCellWidth);
  }
  for (const brick of atlas.bricks) if (frozen.has(brick.key)) {
    minimumWidths.set(brick.key, B * sparseBrickSpan(brick) / brick.resolution);
  }
  const seed = (brick: SparseAdaptiveMassBrick, resolution: SparseBrickResolution) => {
    if (frozen.has(brick.key)) return brick;
    const floor = minimumWidths.get(brick.key) ?? 1;
    while (resolution > 1 && B * sparseBrickSpan(brick) / resolution < floor)
      resolution = (resolution / 2) as SparseBrickResolution;
    return {
      ...brick, resolution, density: new Float64Array(resolution ** 3).fill(brick.density[0] ?? 0),
      gamma: new Float64Array(resolution ** 3).fill(1),
    };
  };
  // Keep explicitly requested refinement of a dormant receiver so its next
  // activation can satisfy grading. Other inactive backing is reclaimed unless
  // a hard floor prevents recreating it as an ordinary fine frontier page.
  let bricks: SparseAdaptiveMassBrick[] = atlas.bricks.filter(brick => active.has(brick.key)
    || intents.get(brick.key)?.activate || frozen.has(brick.key)
    || (minimumWidths.get(brick.key) ?? 1) > 1
    || (intents.get(brick.key)?.resolution ?? brick.resolution) > brick.resolution)
    .map(brick => seed(brick, intents.get(brick.key)?.resolution ?? brick.resolution));
  const nextActive = new Set(active);
  for (const [key, intent] of intents) if (intent.activate) nextActive.add(key);
  const demandedWidths = new Map<number, number>();
  for (const [key, intent] of intents) if (intent.maximumCellWidth !== undefined && !frozen.has(key)) {
    if (!Number.isFinite(intent.maximumCellWidth) || intent.maximumCellWidth < 1)
      throw new Error("CM12 maximum cell width must be finite and at least one");
    demandedWidths.set(key, intent.maximumCellWidth);
  }
  const mergeCoverage = (forcedOnly: boolean): boolean => {
    const groups = new Map<string, SparseAdaptiveMassBrick[]>();
    for (const brick of bricks) {
      const span = sparseBrickSpan(brick);
      if (frozen.has(brick.key) || brick.unclipped || brick.resolution !== 1 || span * 2 > limits.maximumSpanBricks) continue;
      const origin = brick.coordinate.map(q => Math.floor(q / (2 * span)) * 2 * span);
      const key = `${span}/${origin.join("/")}`;
      let group = groups.get(key); if (!group) groups.set(key, group = []);
      group.push(brick);
    }
    const removed = new Set<number>(), parents: SparseAdaptiveMassBrick[] = [];
    for (const group of groups.values()) {
      const forced = group.some(brick => (minimumWidths.get(brick.key) ?? 1) > B * sparseBrickSpan(brick));
      if (!forced && (forcedOnly || group.some(brick => demandedWidths.has(brick.key)
        || !active.has(brick.key) || !intents.get(brick.key)?.mergeable))) continue;
      const span = 2 * sparseBrickSpan(group[0]!);
      const origin = group[0]!.coordinate.map(q => Math.floor(q / span) * span) as [number, number, number];
      const volume = (coordinate: readonly number[], extent: number) => coordinate.reduce((v,q,axis) =>
        v * Math.max(0, Math.min(extent * B, atlas.dimensions[axis]! - q * B)), 1);
      // Forced region coarsening may include a clipped parent, but only when
      // represented children cover its entire in-domain volume. Never fill holes.
      if (forced ? group.reduce((v,brick) => v + volume(brick.coordinate, sparseBrickSpan(brick)),0) !== volume(origin,span)
        : group.length !== 8 || origin.some((q,axis) => (q+span)*B > atlas.dimensions[axis]!)) continue;
      const wasActive = group.some(brick => nextActive.has(brick.key));
      const floor = Math.max(...group.map(brick => minimumWidths.get(brick.key) ?? 1));
      const ceiling = Math.min(...group.map(brick => demandedWidths.get(brick.key) ?? Infinity));
      for (const brick of group) { removed.add(brick.key); nextActive.delete(brick.key); demandedWidths.delete(brick.key); }
      const key = sparseAtlasBrickKey(origin, atlas);
      minimumWidths.set(key, floor);
      if (Number.isFinite(ceiling)) demandedWidths.set(key, Math.max(floor, ceiling));
      parents.push({ key, coordinate: origin, spanBricks: span, resolution: 1,
        density: new Float64Array([1]), gamma: new Float64Array([1]) });
      if (wasActive) nextActive.add(key);
    }
    bricks = [...bricks.filter(brick => !removed.has(brick.key)), ...parents];
    return parents.length > 0;
  };
  mergeCoverage(false);
  const split = (brick: SparseAdaptiveMassBrick): SparseAdaptiveMassBrick[] => {
    if (frozen.has(brick.key)) throw new Error("CM12 generation cannot split a frozen leaf");
    const half = sparseBrickSpan(brick) / 2;
    const children: SparseAdaptiveMassBrick[] = [];
    const wasActive = nextActive.delete(brick.key);
    const demand = demandedWidths.get(brick.key);
    const minimum = minimumWidths.get(brick.key);
    demandedWidths.delete(brick.key);
    for (let child=0; child<8; child++) {
      const coordinate = brick.coordinate.map((q, axis) => q + ((child >>> axis) & 1) * half) as [number, number, number];
      if (!brick.unclipped && coordinate.some((q, axis) => q * B >= atlas.dimensions[axis]!)) continue;
      const key = sparseAtlasBrickKey(coordinate, atlas);
      if (minimum !== undefined) minimumWidths.set(key, minimum);
      children.push(seed({ ...brick, key, coordinate, spanBricks: half }, Math.max(1, brick.resolution / 2) as SparseBrickResolution));
      if (wasActive) nextActive.add(key);
      if (demand !== undefined) demandedWidths.set(key, demand);
    }
    return children;
  };
  const expandDryReceiver = (receiver: SparseAdaptiveMassBrick,
    host: SparseAdaptiveMassBrick): boolean => {
    const span = sparseBrickSpan(host) / 2;
    if (span > limits.maximumSpanBricks) return false;
    const coordinate = receiver.coordinate.map(q => Math.floor(q / span) * span) as
      [number, number, number];
    const overlaps = bricks.filter(brick => brick.coordinate.every((q, axis) =>
      q < coordinate[axis]! + span && q + sparseBrickSpan(brick) > coordinate[axis]!));
    // Grow new dry coverage to the smallest dyadic face patch that the frozen
    // host can represent. Existing accepted geometry is never swallowed by an
    // allocation, even when it occupies a different corner of the new tile.
    for (const brick of overlaps) {
      if (frozen.has(brick.key) || active.has(brick.key)) {
        throw new Error(`CM12 frozen frontier at ${receiver.coordinate.join(",")} needs span ${span}, `
          + `but that support would overlap accepted brick ${brick.coordinate.join(",")}`);
      }
      if (!brick.coordinate.every((q, axis) => q >= coordinate[axis]!
        && q + sparseBrickSpan(brick) <= coordinate[axis]! + span)) return false;
      if (brick.density.some(rho => rho !== 0)) {
        throw new Error("CM12 cannot expand a nonempty frontier receiver");
      }
    }
    const widthFloor = Math.max(B * sparseBrickSpan(host) / (2 * host.resolution),
      ...overlaps.map(brick => minimumWidths.get(brick.key) ?? 1));
    const ceiling = Math.min(...overlaps.map(brick => demandedWidths.get(brick.key) ?? Infinity));
    const enabled = overlaps.some(brick => nextActive.has(brick.key));
    const removed = new Set(overlaps.map(brick => brick.key));
    for (const key of removed) {
      nextActive.delete(key); minimumWidths.delete(key); demandedWidths.delete(key);
    }
    const key = sparseAtlasBrickKey(coordinate, atlas);
    minimumWidths.set(key, widthFloor);
    if (Number.isFinite(ceiling)) demandedWidths.set(key, Math.max(widthFloor, ceiling));
    const parent = seed({ ...receiver, key, coordinate, spanBricks: span,
      density: new Float64Array(receiver.resolution ** 3),
      gamma: new Float64Array(receiver.resolution ** 3).fill(1) }, B);
    bricks = [...bricks.filter(brick => !removed.has(brick.key)), parent];
    newAirCoverage.push({ minimumFine: coordinate.map(q => q * B),
      maximumExclusiveFine: coordinate.map(q => (q + span) * B) });
    if (enabled) nextActive.add(key);
    return true;
  };
  for (;;) {
    const cells = bricks.reduce((n,b) => n + b.resolution ** 3, 0);
    if (bricks.length > limits.maximumLeaves || cells > limits.maximumCells) {
      return { status: "deferred", leaves: bricks.length, cells };
    }
    if (bricks.some(brick => B * sparseBrickSpan(brick) / brick.resolution
      < (minimumWidths.get(brick.key) ?? 1))) {
      if (mergeCoverage(true)) continue;
      return { status: "deferred", leaves: bricks.length, cells };
    }
    let demanded = false;
    bricks = bricks.flatMap(brick => {
      const requestedWidth = demandedWidths.get(brick.key);
      const width = requestedWidth === undefined ? undefined : Math.max(requestedWidth, minimumWidths.get(brick.key) ?? 1);
      if (width === undefined || B * sparseBrickSpan(brick) / brick.resolution <= width) return [brick];
      demanded = true;
      return brick.resolution < B ? [seed(brick, (2 * brick.resolution) as SparseBrickResolution)] : split(brick);
    });
    // Recheck budgets before constructing adjacency for a larger generation.
    if (demanded) continue;
    const neighbors = compileSparseCM12StableLeafFaceNeighbors({
      coordinates: bricks.map(b => b.coordinate), spans: bricks.map(sparseBrickSpan) });
    let macroReceiver: readonly [SparseAdaptiveMassBrick, SparseAdaptiveMassBrick] | undefined;
    for (let leaf = 0; leaf < bricks.length && !macroReceiver; leaf++) {
      const host = bricks[leaf]!;
      if (!frozen.has(host.key)) continue;
      for (const other of neighbors[leaf]!) {
        const receiver = bricks[other]!;
        if (sparseBrickSpan(host) > 2 * sparseBrickSpan(receiver)) {
          macroReceiver = [receiver, host]; break;
        }
      }
    }
    if (macroReceiver) {
      if (!expandDryReceiver(...macroReceiver)) return { status: "deferred", leaves: bricks.length, cells };
      continue;
    }
    let changed = false;
    const toSplit = new Set<number>();
    for (let leaf=0; leaf<bricks.length; leaf++) for (const other of neighbors[leaf]!) {
      const a = bricks[leaf]!, b = bricks[other]!;
      const aSpan = sparseBrickSpan(a), bSpan = sparseBrickSpan(b);
      // The boundary image has four dyadic patches per face. Split coverage
      // as well as grading cell widths so that each shared face fits that ABI.
      if (Math.max(aSpan, bSpan) > 2 * Math.min(aSpan, bSpan)) {
        const larger = aSpan > bSpan ? a : b;
        if (B * sparseBrickSpan(larger) / 2 < (minimumWidths.get(larger.key) ?? 1))
          return { status: "deferred", leaves: bricks.length, cells };
        toSplit.add(aSpan > bSpan ? leaf : other); changed = true;
      }
      const aw = B * sparseBrickSpan(a) / a.resolution, bw = B * sparseBrickSpan(b) / b.resolution;
      if (Math.max(aw,bw) <= 2 * Math.min(aw,bw)) continue;
      const coarser = aw > bw ? leaf : other, brick = bricks[coarser]!;
      if (Math.max(aw, bw) / 2 < (minimumWidths.get(brick.key) ?? 1)) {
        const finer = coarser === leaf ? other : leaf;
        const fine = bricks[finer]!;
        if (frozen.has(fine.key)) return { status: "deferred", leaves: bricks.length, cells };
        minimumWidths.set(fine.key, Math.max(minimumWidths.get(fine.key) ?? 1, Math.max(aw, bw) / 2));
        bricks[finer] = seed(fine, fine.resolution);
        changed = true;
        continue;
      }
      if (brick.resolution < B) bricks[coarser] = seed(brick, (2 * brick.resolution) as SparseBrickResolution);
      else if (sparseBrickSpan(brick) > 1) toSplit.add(coarser);
      else throw new Error("CM12 physical grading has no dyadic refinement");
      changed = true;
    }
    if (toSplit.size) bricks = bricks.flatMap((brick, id) => toSplit.has(id) ? split(brick) : [brick]);
    if (!changed) break;
  }
  const unchanged = nextActive.size === active.size && [...nextActive].every(key => active.has(key))
    && bricks.length === atlas.bricks.length && bricks.every(b => {
    const old = atlas.directory.get(b.key);
    return old && sparseBrickSpan(old) === sparseBrickSpan(b) && old.resolution === b.resolution;
  });
  if (unchanged) return undefined;
  return { status: "ready", atlas: createSparseAdaptiveMassAtlas(atlas.dimensions, bricks, atlas.generation + 1, B, atlas.signedCoordinates), active: nextActive, newAirCoverage };
}
