/**
 * Conservative prescribed-translation oracle for the generalized sparse 4/8
 * atlas. It builds CPU CM12 rows for one step; production GPU transport must
 * replace the maps/CSR construction with compact brick worklists and scatters.
 */

import {
  cm12ConditionedGamma,
  cm12ConditionedRowCoefficient,
  cm12VolumeScaledDeficitCoefficient,
  cm12VolumeWeightedBetaContribution,
} from "../../core/cm12-numerics";
import {
  createSparseAdaptiveMassAtlas,
  sparseAtlasLeaves,
  sparseBrickKey,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseAtlasLeaf,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";

export interface SparseBrickTranslationOptions {
  /** Translation in finest-cell units. Each component is bounded to one brick. */
  readonly displacementFine: SparseBrickVec3;
  readonly emptyEpsilon?: number;
}

export interface SparseBrickTranslationStats {
  readonly sourceBrickCount: number;
  readonly transientSupportBrickCount: number;
  readonly retainedBrickCount: number;
  readonly sourceLeafCount: number;
  readonly workLeafCount: number;
  readonly finalBetaMaximumAbsoluteError: number;
  readonly massBeforeFineCells: number;
  readonly massAfterFineCells: number;
  readonly massAbsoluteErrorFineCells: number;
  readonly minimumDensity: number;
  readonly maximumDensity: number;
}

export interface SparseBrickTranslationResult {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly stats: SparseBrickTranslationStats;
}

type Row = Map<number, number>;

function add(row: Row, id: number, value: number): void {
  if (value !== 0) row.set(id, (row.get(id) ?? 0) + value);
}

function sum(values: Iterable<number>): number {
  let total = 0, correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = total + adjusted;
    correction = next - total - adjusted;
    total = next;
  }
  return total;
}

function zeroBrick(
  key: number,
  coordinate: SparseBrickVec3,
  resolution: SparseAdaptiveMassAtlas["brickFineResolution"],
): SparseAdaptiveMassBrick {
  const count = resolution ** 3;
  return {
    key,
    coordinate,
    resolution,
    density: new Float64Array(count),
    gamma: new Float64Array(count).fill(1),
  };
}

/** Active bricks plus a one-brick swept/trilinear closure; not retained if dry. */
function transportClosure(atlas: SparseAdaptiveMassAtlas): SparseAdaptiveMassAtlas {
  const bricks = new Map<number, SparseAdaptiveMassBrick>();
  for (const brick of atlas.bricks) bricks.set(brick.key, brick);
  for (const brick of atlas.bricks) {
    for (let dz = -1; dz <= 1; dz += 1)
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dx = -1; dx <= 1; dx += 1) {
          const coordinate = [brick.coordinate[0] + dx, brick.coordinate[1] + dy,
            brick.coordinate[2] + dz] as const;
          if (coordinate.some((value, axis) => value < 0 || value >= atlas.brickDimensions[axis])) continue;
          const key = sparseBrickKey(coordinate, atlas.brickDimensions);
          if (!bricks.has(key)) bricks.set(key, zeroBrick(
            key, coordinate, atlas.brickFineResolution,
          ));
        }
  }
  return createSparseAdaptiveMassAtlas(
    atlas.dimensions, [...bricks.values()], atlas.generation,
    atlas.brickFineResolution,
  );
}

function leafAtFineCell(
  atlas: SparseAdaptiveMassAtlas,
  x: number,
  y: number,
  z: number,
): number | undefined {
  const width = atlas.brickFineResolution;
  const coordinate = [Math.floor(x / width), Math.floor(y / width),
    Math.floor(z / width)] as const;
  const key = sparseBrickKey(coordinate, atlas.brickDimensions);
  const brick = atlas.directory.get(key);
  if (!brick) return undefined;
  const factor = width / brick.resolution;
  const localX = Math.floor((x - coordinate[0] * width) / factor);
  const localY = Math.floor((y - coordinate[1] * width) / factor);
  const localZ = Math.floor((z - coordinate[2] * width) / factor);
  return key * atlas.brickCellCapacity + localX
    + brick.resolution * (localY + brick.resolution * localZ);
}

/** Nonnegative cell-centred trilinear stencil, clamped at closed domain walls. */
function sampleWeights(atlas: SparseAdaptiveMassAtlas, position: SparseBrickVec3): Row {
  const base: [number, number, number] = [0, 0, 0];
  const fraction: [number, number, number] = [0, 0, 0];
  for (const axis of [0, 1, 2] as const) {
    const coordinate = Math.max(0.5, Math.min(atlas.dimensions[axis] - 0.5, position[axis])) - 0.5;
    base[axis] = Math.floor(coordinate);
    fraction[axis] = coordinate - base[axis];
  }
  const row: Row = new Map();
  for (let dz = 0; dz <= 1; dz += 1)
    for (let dy = 0; dy <= 1; dy += 1)
      for (let dx = 0; dx <= 1; dx += 1) {
        const coordinates = [Math.min(atlas.dimensions[0] - 1, base[0] + dx),
          Math.min(atlas.dimensions[1] - 1, base[1] + dy),
          Math.min(atlas.dimensions[2] - 1, base[2] + dz)] as const;
        const id = leafAtFineCell(atlas, ...coordinates);
        if (id === undefined) continue;
        const weight = (dx ? fraction[0] : 1 - fraction[0])
          * (dy ? fraction[1] : 1 - fraction[1])
          * (dz ? fraction[2] : 1 - fraction[2]);
        add(row, id, weight);
      }
  return row;
}

function beta(
  rows: ReadonlyMap<number, Row>, leaves: ReadonlyMap<number, SparseAtlasLeaf>,
): Map<number, number> {
  const result = new Map<number, number>();
  for (const [receiverId, row] of rows) {
    const receiver = leaves.get(receiverId)!;
    for (const [donorId, coefficient] of row) {
      const donor = leaves.get(donorId)!;
      result.set(donorId, (result.get(donorId) ?? 0) + cm12VolumeWeightedBetaContribution(
        receiver.volumeFineCells, donor.volumeFineCells, coefficient,
      ));
    }
  }
  return result;
}

export function advanceSparseBrickAtlasTranslation(
  source: SparseAdaptiveMassAtlas,
  options: SparseBrickTranslationOptions,
): SparseBrickTranslationResult {
  for (const value of options.displacementFine) {
    if (!Number.isFinite(value) || Math.abs(value) > source.brickFineResolution) {
      throw new RangeError("each displacementFine component must be finite and within one brick");
    }
  }
  const epsilon = options.emptyEpsilon ?? 1e-12;
  if (source.bricks.length === 0) {
    return { atlas: createSparseAdaptiveMassAtlas(
      source.dimensions, [], source.generation + 1,
      source.brickFineResolution,
    ),
      stats: { sourceBrickCount: 0, transientSupportBrickCount: 0, retainedBrickCount: 0,
        sourceLeafCount: 0, workLeafCount: 0, finalBetaMaximumAbsoluteError: 0,
        massBeforeFineCells: 0, massAfterFineCells: 0, massAbsoluteErrorFineCells: 0,
        minimumDensity: 0, maximumDensity: 0 } };
  }
  const work = transportClosure(source);
  const leafList = sparseAtlasLeaves(work);
  const leaves = new Map(leafList.map((leaf) => [leaf.id, leaf]));
  const backwardWeights = new Map<number, Row>();
  for (const receiver of leafList) backwardWeights.set(receiver.id, sampleWeights(work, [
    receiver.centerFine[0] - options.displacementFine[0],
    receiver.centerFine[1] - options.displacementFine[1],
    receiver.centerFine[2] - options.displacementFine[2],
  ]));

  const unconditioned = new Map<number, Row>();
  for (const receiver of leafList) {
    const weights = backwardWeights.get(receiver.id)!;
    const visible = sum(weights.values());
    const sampledGamma = sum([...weights].map(([id, weight]) => weight * leaves.get(id)!.gamma));
    const advectedGamma = cm12ConditionedGamma(sampledGamma, visible);
    const row: Row = new Map();
    if (visible > 1e-15) for (const [id, weight] of weights) add(row, id, advectedGamma * weight / visible);
    unconditioned.set(receiver.id, row);
  }
  const backwardBeta = beta(unconditioned, leaves);
  const rows = new Map<number, Row>();
  for (const receiver of leafList) {
    const weights = backwardWeights.get(receiver.id)!;
    const visible = sum(weights.values());
    const sampledGamma = sum([...weights].map(([id, weight]) => weight * leaves.get(id)!.gamma));
    const advectedGamma = cm12ConditionedGamma(sampledGamma, visible);
    const row: Row = new Map();
    if (visible > 1e-15) for (const [id, weight] of weights) add(row, id,
      cm12ConditionedRowCoefficient(advectedGamma, weight / visible, backwardBeta.get(id) ?? 0));
    rows.set(receiver.id, row);
  }
  const conditionedBeta = beta(rows, leaves);
  for (const donor of leafList) {
    const deficit = Math.max(0, 1 - (conditionedBeta.get(donor.id) ?? 0));
    if (deficit <= 0) continue;
    const forward = sampleWeights(work, [donor.centerFine[0] + options.displacementFine[0],
      donor.centerFine[1] + options.displacementFine[1],
      donor.centerFine[2] + options.displacementFine[2]]);
    const total = sum(forward.values());
    if (total <= 1e-15) continue;
    for (const [receiverId, weight] of forward) add(rows.get(receiverId)!, donor.id,
      cm12VolumeScaledDeficitCoefficient(donor.volumeFineCells,
        leaves.get(receiverId)!.volumeFineCells, deficit, weight / total));
  }
  const finalBeta = beta(rows, leaves);
  const nextDensity = new Map<number, number>();
  const nextGamma = new Map<number, number>();
  for (const receiver of leafList) {
    let rho = 0, gamma = 0;
    for (const [donorId, coefficient] of rows.get(receiver.id)!) {
      const donor = leaves.get(donorId)!;
      rho += coefficient * donor.density;
      gamma += coefficient * donor.gamma;
    }
    nextDensity.set(receiver.id, rho);
    nextGamma.set(receiver.id, gamma);
  }
  const retained: SparseAdaptiveMassBrick[] = [];
  for (const brick of work.bricks) {
    const density = new Float64Array(brick.resolution ** 3);
    const gamma = new Float64Array(brick.resolution ** 3);
    let wet = false;
    for (let local = 0; local < density.length; local += 1) {
      const id = brick.key * work.brickCellCapacity + local;
      density[local] = nextDensity.get(id) ?? 0;
      gamma[local] = nextGamma.get(id) ?? 1;
      wet ||= density[local] > epsilon;
    }
    if (wet) retained.push({ ...brick, density, gamma });
  }
  const atlas = createSparseAdaptiveMassAtlas(
    source.dimensions, retained, source.generation + 1,
    source.brickFineResolution,
  );
  const sourceLeaves = sparseAtlasLeaves(source);
  const resultLeaves = sparseAtlasLeaves(atlas);
  const massBeforeFineCells = sourceLeaves.reduce(
    (total, leaf) => total + leaf.volumeFineCells * leaf.density, 0);
  const massAfterFineCells = resultLeaves.reduce(
    (total, leaf) => total + leaf.volumeFineCells * leaf.density, 0);
  const densities = resultLeaves.map((leaf) => leaf.density);
  return {
    atlas,
    stats: {
      sourceBrickCount: source.bricks.length,
      transientSupportBrickCount: work.bricks.length - source.bricks.length,
      retainedBrickCount: atlas.bricks.length,
      sourceLeafCount: sourceLeaves.length,
      workLeafCount: leafList.length,
      finalBetaMaximumAbsoluteError: Math.max(...leafList.map(
        (leaf) => Math.abs((finalBeta.get(leaf.id) ?? 0) - 1),
      )),
      massBeforeFineCells,
      massAfterFineCells,
      massAbsoluteErrorFineCells: Math.abs(massAfterFineCells - massBeforeFineCells),
      minimumDensity: densities.length > 0 ? Math.min(...densities) : 0,
      maximumDensity: densities.length > 0 ? Math.max(...densities) : 0,
    },
  };
}
