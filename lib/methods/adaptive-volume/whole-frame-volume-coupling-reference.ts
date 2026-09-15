export interface VolumeBoxCell {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
  readonly donorCapacity: number;
  readonly receiverCapacity: number;
  readonly amount: number;
}

export interface VolumeCouplingEdge {
  readonly receiver: number;
  readonly donor: number;
  weight: number;
}

const overlapMeasure = (a: VolumeBoxCell, b: VolumeBoxCell,
  displacement: readonly [number, number, number]): number => {
  let measure = 1;
  for (let axis = 0; axis < 3; ++axis) {
    const low = Math.max(a.minimum[axis]! + displacement[axis]!, b.minimum[axis]!);
    const high = Math.min(a.maximum[axis]! + displacement[axis]!, b.maximum[axis]!);
    measure *= Math.max(0, high - low);
  }
  return measure;
};

/** CPU oracle for the GPU translated-box coupling and its three balancing rounds. */
export function referenceWholeFrameVolumeCoupling(
  cells: readonly VolumeBoxCell[],
  receiverDisplacements: readonly (readonly [number, number, number])[],
  reverseEdges = false,
): { readonly edges: readonly VolumeCouplingEdge[]; readonly amounts: readonly number[] } {
  if (receiverDisplacements.length !== cells.length) {
    throw new RangeError("one departure displacement is required per receiver");
  }
  const edges: VolumeCouplingEdge[] = [];
  for (let receiver = 0; receiver < cells.length; ++receiver) {
    if (!(cells[receiver]!.receiverCapacity > 0)) continue;
    const row: VolumeCouplingEdge[] = [];
    for (let donor = 0; donor < cells.length; ++donor) {
      if (!(cells[donor]!.donorCapacity > 0)) continue;
      const weight = overlapMeasure(cells[receiver]!, cells[donor]!, receiverDisplacements[receiver]!);
      if (weight > 0) row.push({ receiver, donor, weight });
    }
    if (row.length === 0 && cells[receiver]!.donorCapacity > 0) {
      row.push({ receiver, donor: receiver, weight: cells[receiver]!.donorCapacity });
    }
    edges.push(...row);
  }
  for (let donor = 0; donor < cells.length; ++donor) {
    if (cells[donor]!.donorCapacity <= 0 || edges.some(edge => edge.donor === donor)) continue;
    if (!(cells[donor]!.receiverCapacity > 0)) {
      throw new Error(`donor ${donor} has no open receiver`);
    }
    edges.push({ receiver: donor, donor, weight: cells[donor]!.donorCapacity });
  }
  if (reverseEdges) edges.reverse();
  for (let round = 0; round < 3; ++round) {
    for (let receiver = 0; receiver < cells.length; ++receiver) {
      const row = edges.filter(edge => edge.receiver === receiver);
      const sum = row.reduce((total, edge) => total + edge.weight, 0);
      const factor = sum > 0 ? cells[receiver]!.receiverCapacity / sum : 0;
      for (const edge of row) edge.weight *= factor;
    }
    for (let donor = 0; donor < cells.length; ++donor) {
      const column = edges.filter(edge => edge.donor === donor);
      const sum = column.reduce((total, edge) => total + edge.weight, 0);
      const factor = sum > 0 ? cells[donor]!.donorCapacity / sum : 0;
      for (const edge of column) edge.weight *= factor;
    }
  }
  const amounts = cells.map((_cell, receiver) => edges
    .filter(edge => edge.receiver === receiver)
    .reduce((sum, edge) => sum + cells[edge.donor]!.amount
      * edge.weight / cells[edge.donor]!.donorCapacity, 0));
  return { edges, amounts };
}

/** Budget-only CPU oracle for supplied face connections; GPU relay/gating is tested separately. */
export function referenceSharpenVolume(amounts: readonly number[], targets: readonly number[],
  capacities: readonly number[], faces: readonly (readonly [number, number])[]): readonly number[] {
  if (targets.length !== amounts.length || capacities.length !== amounts.length) {
    throw new RangeError("sharpening arrays must have equal length");
  }
  const surplus = amounts.map((value, i) => Math.max(0, value - targets[i]!));
  const deficit = amounts.map((value, i) => Math.max(0, Math.min(targets[i]!, capacities[i]!) - value));
  const proposals = faces.map(([a, b]) => Math.min(surplus[a]!, deficit[b]!)
    - Math.min(surplus[b]!, deficit[a]!));
  const outgoing = amounts.map(() => 0), incoming = amounts.map(() => 0);
  faces.forEach(([a, b], face) => {
    const transfer = proposals[face]!;
    outgoing[a]! += Math.max(0, transfer); incoming[b]! += Math.max(0, transfer);
    outgoing[b]! += Math.max(0, -transfer); incoming[a]! += Math.max(0, -transfer);
  });
  const give = outgoing.map((v, i) => v > 0 ? Math.min(1, surplus[i]! / v) : 0);
  const take = incoming.map((v, i) => v > 0 ? Math.min(1, deficit[i]! / v) : 0);
  const delta = amounts.map(() => 0);
  faces.forEach(([a, b], face) => {
    const raw = proposals[face]!;
    const transfer = raw * (raw >= 0 ? Math.min(give[a]!, take[b]!) : Math.min(give[b]!, take[a]!));
    delta[a]! -= transfer; delta[b]! += transfer;
  });
  return amounts.map((value, i) => value + delta[i]!);
}

/** Liquid-connected branch of the GPU gate; monotone air-side relay is a separate branch. */
export function sharpeningFaceHasLiquidConnection(
  phiAtFaceCentre: number,
  metric: boolean,
  faceBand: number,
): boolean {
  const roundoff = 9.5367431640625e-7 * (1 + faceBand);
  return metric && Number.isFinite(phiAtFaceCentre) && phiAtFaceCentre <= roundoff;
}
