/** The band solve's pressure cap record (uniform-mixed-frame-status words
 * 10-13): its words stay clear of the failure record's, and the host decodes
 * them as the verdict writes them (counts, the worst capped residual's f32
 * bits, the last capped frame). */
import test from "node:test";
import assert from "node:assert/strict";
import { UNIFORM_MIXED_STATUS, UNIFORM_MIXED_STATUS_WORDS, uniformPressureCapRecord } from "../lib/methods/uniform/uniform-mixed-frame-status";

test("pressure cap words are their own and inside the record", () => {
  const S = UNIFORM_MIXED_STATUS, words = Object.values(S);
  assert.equal(new Set(words).size, words.length, "every status word is distinct");
  for (const w of [S.pressureCapped, S.pressureShort, S.pressureCappedWorst, S.pressureCappedFrame]) {
    assert.ok(w > S.residencyAudit && w < UNIFORM_MIXED_STATUS_WORDS, `word ${w} is a reserved word of the record`);
  }
});

test("pressure cap record decodes counts, the worst residual and the frame", () => {
  const words = new Uint32Array(UNIFORM_MIXED_STATUS_WORDS), S = UNIFORM_MIXED_STATUS;
  assert.deepEqual(uniformPressureCapRecord(words), { capped: 0, short: 0, worstCapped: 0, lastCappedFrame: 0 });
  words[S.pressureCapped] = 31; words[S.pressureShort] = 2; words[S.pressureCappedFrame] = 88;
  words[S.pressureCappedWorst] = new Uint32Array(Float32Array.of(0.2084).buffer)[0]!;
  const r = uniformPressureCapRecord(words);
  assert.equal(r.capped, 31); assert.equal(r.short, 2); assert.equal(r.lastCappedFrame, 88);
  assert.equal(r.worstCapped, Math.fround(0.2084));
});
