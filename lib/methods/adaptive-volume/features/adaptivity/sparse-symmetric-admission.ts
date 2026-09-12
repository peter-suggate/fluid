export interface SparseSymmetricAdmissionCandidate {
  readonly key: number;
  readonly coordinate: readonly [number, number, number];
  readonly spanBricks?: number;
  readonly priority: number;
  /** Current/target rung pair; only equivalent work belongs to one orbit. */
  readonly transition: string;
}

/**
 * Admit topology work without cutting a domain-reflection orbit at the budget
 * boundary. Reflections are formed only from requested resident bricks, so
 * asymmetric edits and incomplete sparse domains do not acquire new work.
 */
export function admitSparseReflectionOrbits(
  candidates: readonly SparseSymmetricAdmissionCandidate[],
  brickDimensions: readonly [number, number, number],
  budget: number,
): ReadonlySet<number> {
  if (budget >= candidates.length) return new Set(candidates.map(({ key }) => key));
  const byCoordinate = new Map(candidates.map((candidate) =>
    [candidate.coordinate.join("/"), candidate] as const));
  const ordered = [...candidates].sort((left, right) =>
    right.priority - left.priority || left.key - right.key);
  const visited = new Set<number>(), admitted = new Set<number>();
  for (const candidate of ordered) {
    if (visited.has(candidate.key)) continue;
    const span = candidate.spanBricks ?? 1;
    const orbit = new Map<number, SparseSymmetricAdmissionCandidate>();
    for (let mask = 0; mask < 8; mask += 1) {
      const coordinate = candidate.coordinate.map((value, axis) => (mask & (1 << axis)) === 0
        ? value : brickDimensions[axis]! - value - span) as [number, number, number];
      const reflected = byCoordinate.get(coordinate.join("/"));
      if (reflected && (reflected.spanBricks ?? 1) === span
        && reflected.priority === candidate.priority
        && reflected.transition === candidate.transition) orbit.set(reflected.key, reflected);
    }
    for (const key of orbit.keys()) visited.add(key);
    if (admitted.size + orbit.size > Math.max(0, budget)) continue;
    for (const key of orbit.keys()) admitted.add(key);
  }
  return admitted;
}

export interface SparseReflectionAdmissionStep {
  readonly admitted: ReadonlySet<number>;
  readonly remainingCredit: number;
}

/** Accrue one frame's selection tokens, admit complete orbits, and spend only
 * the tokens actually selected. At most seven tokens carry beyond the next
 * increment: enough to make progress on one eight-member 3-D orbit. */
export function advanceSparseReflectionAdmission(
  candidates: readonly SparseSymmetricAdmissionCandidate[],
  brickDimensions: readonly [number, number, number],
  currentCredit: number,
  frameIncrement: number,
): SparseReflectionAdmissionStep {
  const increment = Math.max(0, Math.floor(frameIncrement));
  const available = Math.min(increment + 7,
    Math.max(0, Math.floor(currentCredit)) + increment);
  const admitted = admitSparseReflectionOrbits(candidates, brickDimensions, available);
  return { admitted, remainingCredit: available - admitted.size };
}
