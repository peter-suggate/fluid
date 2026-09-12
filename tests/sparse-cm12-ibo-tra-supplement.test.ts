import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12IboTRASupplementWGSL } from
  "../lib/methods/adaptive-volume/sparse-cm12-ibo-tra-supplement.wgsl";

const layout = { baseWords: 4096, templateCount: 3, directoryBaseWords: 4112,
  totalWords: 8192, totalBytes: 16384 } as const;

test("ITR1 WGSL maps face packets to stable IBO rows", () => {
  const source = createSparseCM12IboTRASupplementWGSL({ layout,
    arenaName: "fixtureArena", hookPrefix: "fixture" });
  assert.match(source, /fn itr1StableRowAndBucketOwner/);
  assert.match(source, /fn itr1NegativeBoundaryRefCount/);
  assert.match(source, /fn itr1NegativeBoundaryOwnerRows/);
  assert.match(source, /fn itr1StablePositiveSparseAirRowAndBucketOwner/);
  assert.match(source, /fixtureIBORef/);
  assert.match(source, /fixtureIBOTemplateRowWord/);
  assert.doesNotMatch(source, /TRA1|tra1|incidenceBegin|incidenceRow|rowTermCount|ownerCellAt/);
  assert.doesNotMatch(source, /for\(var row=0u;row</);
});

test("ITR1 validates composed identifiers", () => {
  assert.throws(() => createSparseCM12IboTRASupplementWGSL({ layout,
    arenaName: "bad-name" }), /identifier/);
});

test("ITR1 relocates its image-relative CSR addresses under the shared arena base", () => {
  const baseWords = 65536;
  const source = createSparseCM12IboTRASupplementWGSL({ layout,
    baseWords, hookPrefix: "fixture" });
  assert.match(source, new RegExp(`const ITR1_BASE:u32=${baseWords}u`));
  assert.match(source, new RegExp(
    `const ITR1_DIRECTORY:u32=${baseWords + layout.directoryBaseWords}u`,
  ));
  assert.match(source, /itr1Load\(ITR1_BASE\+directory\.z\+boundary\)/);
});


test("ITR1 macro incidence preserves positive-owner indices beyond fifteen", async () => {
  const { createSparseCM12IboTRASupplement } = await import(
    "../lib/methods/adaptive-volume/sparse-cm12-ibo-tra-supplement");
  const words = new Uint32Array(15 + 2 * 17);
  words[9] = 17 << 23;
  for (let term = 0; term < 17; term++) {
    // Sixteen remote negative terms precede the local positive owner.
    words[15 + 2 * term] = term < 16 ? 0x80000000 | term : 0;
    words[16 + 2 * term] = term < 16 ? 0xbf800000 : 0x41800000;
  }
  const result = createSparseCM12IboTRASupplement({ ibo: { templates: [{
    id: 0, relation: 0, sourceResolution: 1, targetResolution: 4, side: 0,
    sourceDimensions: [1, 1, 1], targetDimensions: [4, 4, 4],
    rowCount: 1, termCount: 17, words,
  }] } });
  const directory = result.layout.directoryBaseWords;
  const entry = result.words[result.words[directory + 1]!]!;
  assert.equal(entry & 0x7fffff, 0);
  assert.equal(entry >>> 23, 16);
  assert.equal(result.words[result.words[directory + 2]!]!, 0);
});
