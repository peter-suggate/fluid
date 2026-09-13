import assert from "node:assert/strict";
import test from "node:test";

import {
  compileSparseCM12TopologyGenerationManifest,
  createSparseCM12CompiledTopologyInitialWords,
  createSparseCM12CompiledTopologyLayout,
  sparseCM12CompiledTopologyAccepted,
  sparseCM12CompiledTopologyAdditionalWords,
  SPARSE_CM12_COMPILED_TOPOLOGY_FAULT as F,
  SPARSE_CM12_COMPILED_TOPOLOGY_HEADER as H,
  SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS,
  SPARSE_CM12_COMPILED_TOPOLOGY_VIEW as V,
} from "../lib/methods/adaptive-volume/sparse-cm12-compiled-topology";
import {
  createSparseCM12CompiledTopologyTransportAccessWGSL,
  createSparseCM12CompiledTopologyWGSL,
} from "../lib/methods/adaptive-volume/sparse-cm12-compiled-topology.wgsl";

test("compiled topology allocates one deterministic full-generation image", () => {
  const layout = createSparseCM12CompiledTopologyLayout({
    baseWords: 67, cellCapacity: 11, rowCapacity: 17,
    termCapacity: 29, incidenceCapacity: 31, physicalFaceCapacity: 37,
  });
  assert.equal(layout.headerBaseWords, 128);
  assert.equal(layout.acceptedCellIdsBaseWords, 192);
  assert.equal(layout.cellOrdinalByStableBaseWords, 204);
  assert.equal(layout.acceptedCellRangesBaseWords, 216);
  assert.equal(layout.acceptedRowRecordsBaseWords, 240);
  assert.equal(layout.rowOrdinalByStableBaseWords, 308);
  assert.equal(layout.orderedTermsBaseWords, 328);
  assert.equal(layout.cellIncidencesBaseWords, 388);
  assert.equal(layout.totalWords, 452);
  assert.equal(sparseCM12CompiledTopologyAdditionalWords(layout), 385);
  assert.equal(layout.headerBaseWords % 64, 0);
  const initial = createSparseCM12CompiledTopologyInitialWords(layout);
  assert.equal(initial.length, SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS,
    "bootstrap upload must remain header-only regardless of capacity");
  assert.equal(initial[H.totalWords], layout.totalWords);
  assert.equal(initial[H.acceptedCellIdsBase], layout.acceptedCellIdsBaseWords);
});

test("mini32 capacity ledger stays below one WebGPU storage binding", () => {
  const layout = createSparseCM12CompiledTopologyLayout({
    baseWords: 0, cellCapacity: 244_288, rowCapacity: 845_040,
    termCapacity: 2_024_352, incidenceCapacity: 2_024_352,
    physicalFaceCapacity: 2_024_352,
    maximumArenaWords: 256 * 1024 * 1024 / 4,
  });
  assert.equal(layout.totalWords, 13_299_824);
  assert.equal(layout.totalWords * 4, 53_199_296);
  assert.throws(() => createSparseCM12CompiledTopologyLayout({
    baseWords: 0, cellCapacity: 244_288, rowCapacity: 845_040,
    termCapacity: 2_024_352, incidenceCapacity: 2_024_352,
    physicalFaceCapacity: 2_024_352,
    maximumArenaWords: layout.totalWords - 1,
  }), /arena limit/);
});

test("generation manifest accepts atomically and fails closed when source changes", () => {
  const layout = createSparseCM12CompiledTopologyLayout({
    cellCapacity: 8, rowCapacity: 12, termCapacity: 24,
    incidenceCapacity: 20, physicalFaceCapacity: 16,
    requiredViews: V.connectivity | V.transport,
  });
  const manifest = compileSparseCM12TopologyGenerationManifest(layout, {
    sourceTopologyGeneration: 7, sourceAcceptedSlot: 1,
    acceptedCellCount: 6, acceptedRowWorklistCount: 10, acceptedRowCount: 9,
    orderedTermCount: 18, cellIncidenceCount: 17,
    readyViews: V.connectivity | V.transport,
    physicalFaceCount: 12, physicalFaceEntryCount: 19,
    transportGeneration: 7,
  });
  assert.equal(sparseCM12CompiledTopologyAccepted(manifest, layout, 7, 1), true);
  assert.equal(sparseCM12CompiledTopologyAccepted(manifest, layout, 8, 1), false);
  assert.equal(sparseCM12CompiledTopologyAccepted(manifest, layout, 7, 0), false);
  manifest[H.cellIncidenceCount]! += 1;
  assert.equal(sparseCM12CompiledTopologyAccepted(manifest, layout, 7, 1), false,
    "the seal certificate covers all published counts");
});

test("manifest refuses partial views and every bounded overflow", () => {
  const layout = createSparseCM12CompiledTopologyLayout({
    cellCapacity: 2, rowCapacity: 3, termCapacity: 5,
    incidenceCapacity: 7, physicalFaceCapacity: 11,
    requiredViews: V.connectivity | V.transport,
  });
  const base = {
    sourceTopologyGeneration: 1, sourceAcceptedSlot: 0,
    acceptedCellCount: 2, acceptedRowWorklistCount: 3, acceptedRowCount: 3,
    orderedTermCount: 5, cellIncidenceCount: 7,
    readyViews: V.connectivity | V.transport,
    physicalFaceCount: 11, physicalFaceEntryCount: 22,
  };
  assert.throws(() => compileSparseCM12TopologyGenerationManifest(layout,
    { ...base, readyViews: V.connectivity }), /not ready/);
  assert.throws(() => compileSparseCM12TopologyGenerationManifest(layout,
    { ...base, orderedTermCount: 6 }), /term capacity/);
  assert.throws(() => compileSparseCM12TopologyGenerationManifest(layout,
    { ...base, cellIncidenceCount: 8 }), /incidence capacity/);
  assert.throws(() => compileSparseCM12TopologyGenerationManifest(layout,
    { ...base, physicalFaceEntryCount: 23 }), /transport capacity/);
});

test("WGSL compiler is a full rebuild with generation-fenced hot accessors", () => {
  const layout = createSparseCM12CompiledTopologyLayout({
    baseWords: 256, cellCapacity: 32, rowCapacity: 48,
    termCapacity: 96, incidenceCapacity: 88, physicalFaceCapacity: 64,
  });
  const wgsl = createSparseCM12CompiledTopologyWGSL({ layout });
  for (const entry of ["beginCompiledTopologyGeneration", "clearCompiledTopologyGeneration",
    "compileCompiledTopologyCells", "compileCompiledTopologyRows",
    "compileCompiledTopologyCellIncidences", "sealCompiledTopologyGeneration"]) {
    assert.match(wgsl, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(wgsl, /fn cnxAccepted\(\)->bool/);
  assert.match(wgsl, /fn cnxAcceptedCellInvocation\(ordinal:u32\)->u32/);
  assert.match(wgsl, /fn cnxAcceptedRowInvocation\(ordinal:u32\)->u32/);
  assert.match(wgsl, /fn cnxCellIncidenceRangeUnchecked\(stableCell:u32\)->vec2u/);
  assert.match(wgsl, /fn cnxRowTermCoefficientUnchecked\(term:u32\)->f32/);
  assert.doesNotMatch(wgsl, /fn cnxCellIncidenceRange\(stableCell:u32\)->vec2u/,
    "hot consumers fence once; obsolete per-item checked wrappers stay deleted");
  assert.match(wgsl, new RegExp(
    `if\\(cnxCellOrdinalUnchecked\\(stableCell\\)==INVALID\\)\\{\\s*cnxFault\\(${F.malformedRow}u,row\\);return;`),
    "an accepted row cannot publish a term outside the accepted cell image");
  assert.match(wgsl, /fn cnxAllocatePhysicalFaces\(count:u32,owner:u32\)->u32/);
  assert.match(wgsl, /fn cnxAllocatePhysicalFaceEntries\(count:u32,owner:u32\)->u32/);
  assert.match(wgsl, /fn cnxPublishTransportView\(\)/);
  assert.match(wgsl, /gid\.x\+64u\*65535u\*gid\.y/);
  assert.match(wgsl,
    /fn sealCompiledTopologyGeneration\(\)\{\s*if\(!cnxBuildRequired\(\)\)\{return;}[\s\S]*if\(!cnxBuilding\(\)\)\{[\s\S]*cnxPublishSealFailure\(\);/,
    "a stale source during a pending rebuild must publish a terminal failure");
  assert.doesNotMatch(wgsl, /dirty|delta/i);

  const staticOpen = createSparseCM12CompiledTopologyWGSL({
    layout, source: { rowStaticOpen: row => `staticOpen(${row})` },
  });
  assert.match(staticOpen, /if\(staticOpen\(row\)\)\{metadata\|=1u<<8u;\}/);
  assert.match(staticOpen, /fn cnxRowStaticOpenUnchecked\(stableRow:u32\)->bool/);

  const failurePublishing = createSparseCM12CompiledTopologyWGSL({
    layout,
    publishFailure: (fault, owner) => `recordFailure(${fault},${owner});`,
  });
  assert.match(failurePublishing,
    /fn cnxPublishSealFailure\(\)[\s\S]*recordFailure\(cnxLoad\(CNX_H_FAULT\),cnxLoad\(CNX_H_FIRST_FAULT\)\);/);
  assert.equal((failurePublishing.match(/cnxPublishSealFailure\(\);/g) ?? []).length, 4,
    "every seal rejection path must publish the sticky failure exactly once");
});

test("transport views retain deterministic ordering and first/end range semantics", () => {
  type Term = { cell: number; coefficient: number };
  type Face = { negative: number; positive: number; row: number; area: number };
  const rows = new Map<number, Term[]>([
    [4, [{ cell: 2, coefficient: -0.75 }, { cell: 8, coefficient: -0.25 },
      { cell: 5, coefficient: 0.5 }, { cell: 6, coefficient: 0.5 }]],
    [7, [{ cell: 2, coefficient: 1 }]],
  ]);
  // This is a CPU stand-in for geometricSubface: one mixed pair is clipped.
  const subface = (row: number, negative: Term, positive: Term): Face | undefined =>
    negative.cell === 8 && positive.cell === 6 ? undefined : {
      negative: negative.cell, positive: positive.cell, row,
      area: Math.fround(Math.abs(negative.coefficient * positive.coefficient)),
    };
  const boundary = (row: number, term: Term): Face => term.coefficient < 0
    ? { negative: term.cell, positive: 0xffff_ffff, row, area: Math.fround(Math.abs(term.coefficient)) }
    : { negative: 0xffff_ffff, positive: term.cell, row, area: Math.fround(Math.abs(term.coefficient)) };
  const incidences = [4, 7, 4]; // duplicate seam contact is intentional
  const cell = 2;
  const bits = (value: number) => {
    const f = new Float32Array([value]);
    return new Uint32Array(f.buffer)[0]!;
  };
  const tuple = (face: Face) => [face.negative === cell ? face.positive : face.negative,
    face.row, bits(face.area)] as const;

  // Original PLIC path: incidence -> row -> opposite-sign term.
  const legacy: (readonly number[])[] = [];
  for (const row of incidences) {
    const terms = rows.get(row)!;
    const own = terms.find(term => term.cell === cell)!;
    if (terms.length === 1) legacy.push(tuple(boundary(row, own)));
    else for (const other of terms) {
      if (own.coefficient * other.coefficient >= 0) continue;
      const face = own.coefficient < 0 ? subface(row, own, other) : subface(row, other, own);
      if (face) legacy.push(tuple(face));
    }
  }

  // Full CNX/GV build: accepted row order -> negative/positive physical faces,
  // followed by cell CSR in the original incidence -> row -> face order.
  const physical: Face[] = [];
  const rowRanges = new Map<number, readonly [number, number]>();
  for (const [row, terms] of rows) {
    const first = physical.length;
    if (terms.length === 1) physical.push(boundary(row, terms[0]!));
    else for (const negative of terms) {
      if (negative.coefficient >= 0) continue;
      for (const positive of terms) {
        if (positive.coefficient <= 0) continue;
        const face = subface(row, negative, positive); if (face) physical.push(face);
      }
    }
    rowRanges.set(row, [first, physical.length]);
  }
  const compiled: (readonly number[])[] = [];
  for (const row of incidences) {
    const [first, end] = rowRanges.get(row)!;
    for (let face = first; face < end; face++) {
      const record = physical[face]!;
      if (record.negative === cell || record.positive === cell) compiled.push(tuple(record));
    }
  }
  assert.deepEqual(compiled, legacy);
  assert.equal(compiled.length, 5, "duplicate row incidence remains duplicated and ordered");

  const access = createSparseCM12CompiledTopologyTransportAccessWGSL({
    rowSubfaceRanges: 100, cellSubfaceRanges: 200,
    cellSubfaceEntries: 300, subfaceMetadata: 400,
  });
  assert.match(access, /return vec2u\(first,first\+count\)/);
  assert.match(access, /fn cnxPhysicalFaceRangeUnchecked\(stableRow:u32\)->vec2u/);
  assert.match(access, /fn cnxCellFaceEntryUnchecked\(at:u32\)->u32/);
  assert.doesNotMatch(access, /fn cnxPhysicalFaceArea\(face:u32\)->f32/,
    "production transport fences once and carries no per-item checked compatibility API");
});
