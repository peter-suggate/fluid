import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12PressureExecutionImageInitialWords,
  createSparseCM12PressureExecutionImageLayout,
  SPARSE_CM12_FULL_PRESSURE_IMAGE_ENTRY_POINTS,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER as H,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION,
} from "../lib/methods/adaptive-volume/sparse-cm12-pressure-execution-image";
import { createSparseCM12PressureExecutionImageWGSL } from
  "../lib/methods/adaptive-volume/sparse-cm12-pressure-execution-image.wgsl";

test("PEI2 full image slots and scan scratch are disjoint and initialized", () => {
  const layout = createSparseCM12PressureExecutionImageLayout({
    baseWords: 7, cellCapacity: 2_083, rowCapacity: 5_071,
    brickCapacity: 0, hierarchyCapacity: 0,
    brickFineResolution: 8, presentationPageResolution: 8,
  });
  assert.equal(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION, 2);
  assert.equal(SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS, 40);
  assert.equal(layout.pressureMembershipWordCount, Math.ceil(2_083 / 32));
  assert.equal(layout.pressureRowMembershipWordCount, Math.ceil(5_071 / 32));
  assert.equal(layout.fullWordScratchCount % 64, 0);
  const ranges = [
    ...layout.pressureCellSlotBaseWords.map(base => [base, base + layout.cellCapacity]),
    ...layout.pressureMembershipSlotBaseWords.map(base =>
      [base, base + layout.pressureMembershipWordCount]),
    ...layout.pressureRowMembershipSlotBaseWords.map(base =>
      [base, base + layout.pressureRowMembershipWordCount]),
    [layout.fullWordCountBaseWords,
      layout.fullWordCountBaseWords + layout.fullWordScratchCount],
    [layout.fullWordPrefixBaseWords,
      layout.fullWordPrefixBaseWords + layout.fullWordScratchCount],
  ] as const;
  for (let left = 0; left < ranges.length; left += 1) {
    for (let right = left + 1; right < ranges.length; right += 1) {
      assert.ok(ranges[left]![1] <= ranges[right]![0]
        || ranges[right]![1] <= ranges[left]![0], `${left}/${right} overlap`);
    }
  }
  const words = createSparseCM12PressureExecutionImageInitialWords(layout);
  assert.equal(words[H.activeSlot], 0);
  for (const base of layout.pressureCellSlotBaseWords) {
    const local = base - layout.baseWords;
    assert.ok(words.subarray(local, local + layout.cellCapacity)
      .every(value => value === 0xffff_ffff));
  }
});

test("PEI2 WGSL owns each word, preserves the active image, and emits stable IDs", () => {
  const layout = createSparseCM12PressureExecutionImageLayout({
    baseWords: 64, cellCapacity: 257, rowCapacity: 769,
    brickCapacity: 8, hierarchyCapacity: 1,
    brickFineResolution: 8, presentationPageResolution: 8,
  });
  const source = createSparseCM12PressureExecutionImageWGSL({
    layout, fullRebuild: true, sourcePrefix: "source",
    publishFailure: (fault, owner) => `recordFailure(${fault},${owner});`,
  });
  for (const entry of SPARSE_CM12_FULL_PRESSURE_IMAGE_ENTRY_POINTS) {
    assert.match(source, new RegExp(`fn ${entry}\\b`));
  }
  const cells = source.slice(source.indexOf("fn classifyFullPressureCellWords"),
    source.indexOf("var<workgroup>peiFullScan"));
  assert.match(cells, /let first=word<<5u;var bits=0u/);
  assert.match(cells, /sourceFullCellOrdinal\(cell\)/);
  assert.match(cells, /peiFullBuildingCellMembershipBase\(\)\+word\]=bits/);
  assert.doesNotMatch(cells, /peiPressureMembershipSlotBase\([^)]*activeSlot/);
  const publish = source.slice(source.indexOf("fn publishFullPressureCellIds"),
    source.indexOf("fn classifyFullPressureRowWords"));
  assert.match(publish, /for\(var bit=0u;bit<32u;bit\+=1u\)/);
  assert.match(publish, /\(word<<5u\)\+bit/);
  assert.match(source, /peiPressureCellSlotBase\(slot\)\+linear/);
  assert.match(source, /peiPressureMembershipSlotBase\(slot\)\+\(cell>>5u\)/);
  assert.match(source, /!=3u\|\|!sourceFullBuildAccepted\(\)/);
  assert.match(source, /!sourceFullBuildAccepted\(\)/);
  assert.match(source, /fn peiFullImageAccepted\(\)->bool/);
  assert.match(source, /recordFailure\(code,id\)/);
  assert.doesNotMatch(source, /peiBeginFromCanonicalPressureRows/);
});

test("stable-word compaction is independent of accepted worklist order", () => {
  const accepted = [96, 2, 191, 64, 33, 32, 190, 7];
  const selected = new Set([191, 2, 64, 32, 7]);
  const capacity = 200;
  const acceptedMap = new Set(accepted);
  const words = Array.from({ length: Math.ceil(capacity / 32) }, (_, word) => {
    let bits = 0;
    for (let bit = 0; bit < 32; bit += 1) {
      const cell = 32 * word + bit;
      if (acceptedMap.has(cell) && selected.has(cell)) bits |= 1 << bit;
    }
    return bits >>> 0;
  });
  const output: number[] = [];
  for (let word = 0; word < words.length; word += 1) {
    for (let bit = 0; bit < 32; bit += 1) {
      if ((words[word]! & (1 << bit)) !== 0) output.push(32 * word + bit);
    }
  }
  assert.deepEqual(output, [2, 7, 32, 64, 191]);
});
