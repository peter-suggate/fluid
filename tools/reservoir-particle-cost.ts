/** CPU-only work census, not a GPU timing prediction. Canonical pair counts
 * describe the former all-pairs sort (a concentration metric after its
 * replacement); merge loops mirror uniform-narrow-band-flip.ts. */
export function particleCellCost(counts: Iterable<number>) {
  let particles = 0, occupied = 0, max = 0, rankComparisons = 0;
  const histogram: Record<string, number> = {};
  for (const n of counts) {
    if (!Number.isSafeInteger(n) || n < 0) throw new Error("Invalid cell population");
    if (!n) continue;
    particles += n; occupied++; max = Math.max(max, n);
    rankComparisons += n * (n - 1);
    const bucket = n <= 8 ? "1-8" : n <= 20 ? "9-20" : n <= 64 ? "21-64" : n <= 256 ? "65-256" : ">256";
    histogram[bucket] = (histogram[bucket] ?? 0) + 1;
  }
  return { particles, occupied, max, rankComparisons, comparisonsPerParticle: particles ? rankComparisons / particles : 0, histogram };
}

/** Full bitwise record order, including the identical-record slot tie break.
 * Executes the all-pairs algorithm, independently checked against Array.sort. */
export function canonicalRankCPU(records: readonly Uint32Array[]) {
  const less = (a: Uint32Array, b: Uint32Array) => {
    for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return a[k]! < b[k]!;
    return false;
  };
  let comparisons = 0;
  const result: Uint32Array[] = new Array(records.length);
  for (let i = 0; i < records.length; i++) {
    let rank = 0;
    for (let j = 0; j < records.length; j++) {
      if (j === i) continue;
      comparisons++;
      if (less(records[j]!, records[i]!) || (j < i && !less(records[i]!, records[j]!))) rank++;
    }
    result[rank] = records[i]!;
  }
  return { records: result, comparisons };
}

/** Only the mass/retention decisions, for a live sorted cell. Geometry,
 * G2P, heat and deep-band retirement are deliberately outside this twin.
 * Count actual loop visits, including visits after a gatherer is full. */
export function mergeCellCPU(masses: readonly number[], depth: number, cap = 20, graded = true) {
  if (masses.some(m => !Number.isInteger(m) || m < 1 || m > 8)) throw new Error("Invalid particle mass");
  const limit = graded && depth <= -1.5 ? Math.min(cap, 6) : cap;
  if (!(cap > 0 && depth < -1 && masses.length > limit)) return { masses: [...masses], visits: 0 };
  let visits = 0;
  const kept: number[] = [];
  for (let i = 0; i < masses.length; i++) {
    if (i < limit) {
      let mass = masses[i]!;
      for (let j = i + limit; j < masses.length; j += limit) {
        visits++;
        if (mass + masses[j]! <= 8) mass += masses[j]!;
      }
      kept.push(mass);
    } else {
      const first = i % limit;
      let mass = masses[first]!;
      for (let j = first + limit; j < i; j += limit) {
        visits++;
        if (mass + masses[j]! <= 8) mass += masses[j]!;
      }
      if (mass + masses[i]! > 8) kept.push(masses[i]!);
    }
  }
  return { masses: kept, visits };
}

export function uniformPopulations(total: number, perCell: number) {
  const counts = new Array<number>(Math.floor(total / perCell)).fill(perCell);
  if (total % perCell) counts.push(total % perCell);
  return counts;
}
