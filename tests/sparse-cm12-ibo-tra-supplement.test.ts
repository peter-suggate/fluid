import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12IboTRASupplementWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-supplement.wgsl";

const layout = { baseWords: 4096, templateCount: 3, directoryBaseWords: 4112,
  totalWords: 8192, totalBytes: 16384 } as const;

test("ITR1 WGSL compiles packet ballots without legacy/global topology walks", () => {
  const source = createSparseCM12IboTRASupplementWGSL({ layout,
    arenaName: "fixtureArena", hookPrefix: "fixture" });
  assert.match(source, /fn compileSparseCM12GammaRowMasks/);
  assert.match(source, /fn itr1ApplyCanonical/);
  assert.match(source, /itr1ShiftWithin/);
  assert.match(source, /itr1ShiftCross/);
  assert.match(source, /fixtureIBORef/);
  assert.match(source, /fixtureIBOTemplateRowWord/);
  assert.match(source, /cm12TransportMarkGammaMask/);
  assert.match(source, /if\(ownerTerm==0xfu\)\{continue;\}/);
  assert.match(source, /scatterSparseCM12GammaSnapshotRows/);
  assert.match(source, /scatterSparseCM12GammaRefinementRows/);
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
  assert.match(source, /itr1Load\(ITR1_BASE\+directory\.x\+boundary\)/);
  assert.match(source, /itr1Load\(ITR1_BASE\+directory\.z\+boundary\)/);
});
