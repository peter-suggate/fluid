import assert from "node:assert/strict";
import test from "node:test";
import { canonicalDispatchMergeCPU, canonicalMergeCPU, compareParticleWords } from "../tools/particle-canonical-reference";
import { canonicalRankCPU } from "../tools/reservoir-particle-cost";

function fixture(count: number, stride: number, mode: "random" | "coincident" | "identical") {
  let state = 0x12345678;
  return Array.from({ length: count }, (_, i) => Uint32Array.from({ length: stride }, (_, k) => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    if (mode === "identical") return k;
    if (mode === "coincident" && k < 3) return 0x3f800000;
    // Include signed zero, infinities, NaN payloads and every mass/flag bit:
    // sorting compares uint words, never floating arithmetic.
    return i % 7 === 0 ? [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0xffffffff][k % 6]! : state >>> 0;
  }));
}

test("parallel canonical merge preserves every word and cell across uneven runs and lane orders", () => {
  const counts = [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 5319];
  for (const stride of [12, 28]) for (const mode of ["random", "coincident", "identical"] as const) {
    const records = fixture(counts.reduce((a, b) => a + b, 0), stride, mode);
    for (const reversed of [false, true]) {
      let start = 0; const input: Uint32Array[] = [], expected: Uint32Array[] = [];
      for (const n of counts) {
        const run = records.slice(start, start + n); start += n;
        expected.push(...[...run].sort(compareParticleWords)); input.push(...(reversed ? run.reverse() : run));
      }
      const merged = canonicalMergeCPU(input, counts, reversed);
      assert.deepEqual(merged.records, expected, `${stride} words, ${mode}, reversed=${reversed}`);
      assert.equal(new Set(merged.indices).size, input.length);
    }
  }
});

test("parallel merge agrees with the previous all-pairs canonical rank", () => {
  for (const stride of [12, 28]) for (const n of [0, 1, 7, 32, 65, 257]) {
    const records = fixture(n, stride, "coincident");
    assert.deepEqual(canonicalMergeCPU(records, [n]).records, canonicalRankCPU(records).records);
  }
});

test("a concentrated cell has bounded logarithmic rounds and subquadratic comparison growth", () => {
  for (const mode of ["random", "coincident", "identical"] as const) {
    const a = canonicalMergeCPU(fixture(5319, 12, mode), [5319]);
    const b = canonicalMergeCPU(fixture(10638, 12, mode), [10638]);
    assert.equal(a.rounds, 13); assert.equal(b.rounds, 14);
    assert.ok(a.comparisons < 5319 * 5318 / 40, `${mode}: ${a.comparisons} comparisons`);
    assert.ok(b.comparisons < 2.6 * a.comparisons, `${mode}: ${a.comparisons} -> ${b.comparisons}`);
  }
});


test("dense dispatch handoff preserves odd/even banks, slab holes, every payload and small cells", () => {
  const counts = [0, 1, 255, 256, 257, 511, 512, 513, 1025, 5319, 65537, 8, 0];
  for (const stride of [12, 28]) for (const mode of ["random", "coincident", "identical"] as const) {
    const records = fixture(counts.reduce((a,b)=>a+b,0), stride, mode);
    let at = 0;
    const expected = counts.flatMap(count => { const cell=records.slice(at,at+count);at+=count;return cell.sort(compareParticleWords); });
    assert.deepEqual(canonicalDispatchMergeCPU(records, counts).records, expected, `${stride}: ${mode}`);
  }
});
