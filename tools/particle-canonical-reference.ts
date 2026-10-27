/** CPU twin of the segmented parallel merge in particleCanonicalWGSL.
 * Only indices move; complete records are copied once after the last round. */
export function compareParticleWords(a: Uint32Array, b: Uint32Array): number {
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return a[k]! < b[k]! ? -1 : 1;
  return 0;
}

export function canonicalMergeCPU(records: readonly Uint32Array[], counts: readonly number[], reverseLanes = false) {
  if (counts.some(n => !Number.isSafeInteger(n) || n < 0) || counts.reduce((a, b) => a + b, 0) !== records.length) throw new Error("Invalid cell runs");
  let input = records.map((_, i) => i), comparisons = 0, rounds = 0;
  const maxCount = Math.max(0, ...counts);
  for (let width = 1; width < maxCount; width *= 2) {
    const output = new Array<number>(input.length).fill(-1);
    let start = 0;
    for (const count of counts) {
      const end = start + count;
      for (let lane = 0; lane < count; lane++) {
        const at = start + (reverseLanes ? count - 1 - lane : lane), index = input[at]!;
        const run = Math.floor((at - start) / width);
        const other = Math.min(end, start + (run ^ 1) * width);
        let low = other, high = Math.min(end, other + width);
        while (low < high) {
          const mid = low + Math.floor((high - low) / 2), candidate = input[mid]!;
          comparisons++;
          const order = compareParticleWords(records[candidate]!, records[index]!);
          if (order < 0 || (order === 0 && candidate < index)) low = mid + 1;
          else high = mid;
        }
        const destination = start + (run & ~1) * width + (at - start) % width + low - other;
        if (destination < start || destination >= end || output[destination] !== -1) throw new Error("Merge destination collision or escaped cell");
        output[destination] = index;
      }
      start = end;
    }
    if (output.some(i => i < 0) || new Set(output).size !== records.length) throw new Error("Merge lost or duplicated a record");
    input = output; rounds++;
  }
  return { indices: input, records: input.map(i => records[i]!), comparisons, rounds };
}

/** CPU model of the local/dense dispatch handoff, including slab holes and
 * cells finishing in different index banks. Guards model the compact stream. */
export function canonicalDispatchMergeCPU(records: readonly Uint32Array[], counts: readonly number[], holes = 3) {
  if (counts.reduce((a, b) => a + b, 0) !== records.length) throw new Error("Invalid cell runs");
  const slots = records.length + holes * (counts.length + 1);
  const banks = [new Int32Array(slots).fill(-1), new Int32Array(slots).fill(-1)];
  const output = new Int32Array(slots).fill(-1);
  const runs: { start: number; count: number }[] = [];
  let start = holes, input = 0;
  const less = (a: number, b: number) => compareParticleWords(records[a]!, records[b]!) || a - b;
  for (const count of counts) {
    const indices = Array.from({ length: count }, (_, k) => input + k);
    runs.push({ start, count });
    if (count <= 256) output.set(indices.sort(less), start);
    else banks[0]!.set(indices, start);
    input += count; start += count + holes;
  }
  for (let width = 1, bank = 0; width < records.length; width *= 2, bank ^= 1) {
    const writes = new Set<number>();
    // Reversed scheduling exercises independence between cells and lanes.
    for (const { start, count } of [...runs].reverse()) {
      if (count <= 256 || count <= width) continue;
      const end = start + count;
      for (let at = end - 1; at >= start; at--) {
        const index = banks[bank]![at]!;
        const run = Math.floor((at - start) / width);
        const other = Math.min(end, start + (run ^ 1) * width);
        let low = other, high = Math.min(end, other + width);
        while (low < high) {
          const mid = low + Math.floor((high - low) / 2);
          if (less(banks[bank]![mid]!, index) < 0) low = mid + 1; else high = mid;
        }
        const destination = start + (run & ~1) * width + (at - start) % width + low - other;
        if (index < 0 || destination < start || destination >= end || writes.has(destination)) throw new Error("Invalid dense merge destination");
        writes.add(destination); banks[bank ^ 1]![destination] = index;
      }
    }
  }
  const indices: number[] = [];
  const live = new Set<number>();
  for (const { start, count } of runs) {
    if (count > 256) output.set(banks[Math.ceil(Math.log2(count)) & 1]!.subarray(start, start + count), start);
    for (let at = start; at < start + count; at++) { live.add(at); indices.push(output[at]!); }
  }
  for (let at = 0; at < slots; at++) if (!live.has(at) && (output[at] !== -1 || banks.some(bank => bank[at] !== -1))) throw new Error("Slab hole overwritten");
  if (indices.some(i => i < 0) || new Set(indices).size !== records.length) throw new Error("Particle lost or duplicated");
  return { indices, records: indices.map(i => records[i]!) };
}
