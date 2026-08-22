import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_DISPATCH_ORDER,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VEX_CAUSE,
  compileSparseCM12TransportProducerMaskReference,
  createSparseCM12TransportProducerMaskInitialWords,
  createSparseCM12TransportProducerMaskLayout,
  packSparseCM12TransportProducerMask,
  sparseCM12TransportProducerMaskHasLane,
} from "../lib/methods/adaptive-mass/sparse-cm12-transport-producer-masks";
import { createSparseCM12TransportProducerMaskWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-producer-masks.wgsl";

test("TPM1 lays out exact packet masks in an appendable atomic arena", () => {
  const layout = createSparseCM12TransportProducerMaskLayout({
    packetCapacity: 129, baseWords: 65,
  });
  assert.equal(layout.baseWords, 128);
  for (const value of [layout.packetStampBaseWords, layout.surfaceLowBaseWords,
    layout.surfaceHighBaseWords, layout.densityLowBaseWords,
    layout.densityHighBaseWords, layout.totalWords]) assert.equal(value % 64, 0);
  assert.ok(layout.surfaceLowBaseWords >= layout.packetStampBaseWords + 129);
  const words = createSparseCM12TransportProducerMaskInitialWords(layout);
  const h = SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER;
  assert.equal(words.length, layout.totalWords - layout.baseWords);
  assert.equal(words[h.magic], SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC);
  assert.equal(words[h.headerWords], SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS);
  assert.equal(words[h.packetCapacity], 129);
  assert.equal(words[h.packetStampBase], layout.packetStampBaseWords);
  assert.equal(words[h.totalWords], layout.totalWords);
});

test("TPM1 portable ballot keeps all boundary lane bits exact", () => {
  const lanes = Array<boolean>(64).fill(false);
  for (const lane of [0, 1, 30, 31, 32, 33, 62, 63]) lanes[lane] = true;
  const [low, high] = packSparseCM12TransportProducerMask(lanes);
  assert.equal(low, 0xc000_0003);
  assert.equal(high, 0xc000_0003);
  for (let lane = 0; lane < 64; lane += 1) {
    assert.equal(sparseCM12TransportProducerMaskHasLane(low, high, lane), lanes[lane]);
  }
});

test("TPM1 compiler oracle equals legacy TRA and VEX per-cell closure", () => {
  const cells = Array.from({ length: 64 }, (_, lane) => 100 + lane);
  const surface = Array<boolean>(64).fill(false);
  const density = Array<boolean>(64).fill(false);
  surface[0] = true; surface[31] = true; surface[63] = true;
  density[1] = true; density[32] = true; density[63] = true;
  const [surfaceLow, surfaceHigh] = packSparseCM12TransportProducerMask(surface);
  const [densityLow, densityHigh] = packSparseCM12TransportProducerMask(density);
  const incidenceRows = (cell: number): readonly number[] => [cell, cell + 1000];
  const rowAccepted = (row: number): boolean => row !== 163;
  const rowTermCells = (row: number): readonly number[] => [row % 1000, row % 1000 + 1, 999];
  const cellActive = (cell: number): boolean => cell !== 999;
  const compiled = compileSparseCM12TransportProducerMaskReference({
    mask: { surfaceLow, surfaceHigh, densityLow, densityHigh },
    packetCells: cells, incidenceRows, rowAccepted, rowTermCells, cellActive,
  });

  const legacyRows = new Set<number>();
  const legacyRoots = new Set<number>();
  for (let lane = 0; lane < 64; lane += 1) {
    const cell = cells[lane]!;
    if (surface[lane]) for (const row of incidenceRows(cell)) {
      if (rowAccepted(row)) legacyRows.add(row);
    }
    if (density[lane]) {
      legacyRoots.add(cell);
      for (const row of incidenceRows(cell)) if (rowAccepted(row)) {
        for (const neighbor of rowTermCells(row)) {
          if (neighbor !== cell && cellActive(neighbor)) legacyRoots.add(neighbor);
        }
      }
    }
  }
  assert.deepEqual(compiled.traRows, [...legacyRows].sort((a, b) => a - b));
  assert.deepEqual(compiled.vexRoots, [...legacyRoots].sort((a, b) => a - b));
});

test("TPM1 WGSL publishes stamp last and compiles masks outside gather", () => {
  const layout = createSparseCM12TransportProducerMaskLayout({ packetCapacity: 512 });
  const source = createSparseCM12TransportProducerMaskWGSL({
    layout, arenaName: "activity", hookPrefix: "testResident",
  });
  assert.match(source, /var<workgroup>tpm1SurfaceLow:atomic<u32>/);
  assert.match(source, /atomicOr\(&tpm1SurfaceLow,1u<<lane\)/);
  assert.match(source, /atomicOr\(&tpm1DensityHigh,1u<<\(lane-32u\)\)/);
  assert.doesNotMatch(source, /@builtin\(subgroup|subgroup[A-Z]|enable\s+subgroups/i);
  assert.match(source,
    /atomicStore\(&activity\[TPM1_DENSITY_HIGH\+packet\],highD\);[\s\S]*atomicStore\(&activity\[TPM1_STAMP\+packet\],generation\)/);
  assert.match(source, /fn compileSparseCM12TransportRowMasks/);
  assert.match(source, /let packet=testResidentTransportProducerMaskPacket\(rank\)/);
  assert.match(source, /testResidentTransportProducerMaskMarkRow\(row\)/);
  assert.match(source, /fn compileSparseCM12VexRootMasks/);
  assert.match(source, new RegExp(
    `testResidentTransportProducerMaskRecordVexRoot\\(cell,${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VEX_CAUSE}u\\)`,
  ));
  assert.doesNotMatch(source, /tra1MarkScalarCellClosure/);
  assert.doesNotMatch(source, /cm12ResidentRecordExtensionIncidence/);
});

test("TPM1 integration order seals masks before either authority compiler", () => {
  assert.deepEqual(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_DISPATCH_ORDER, [
    "beginSparseCM12TransportProducerMasks + beginTransportRowAuthority",
    "gatherConservativeDensity (calls cm12TransportProducerMaskPublish once per lane)",
    "sealSparseCM12TransportProducerMasks",
    "compileSparseCM12TransportRowMasks + compileSparseCM12VexRootMasks (VEX candidate is already collecting)",
    "finalizeTransportRowAuthority",
    "late frame: beginVelocityExtensionCandidate + sealVelocityExtensionRoots",
  ]);
});

test("TPM1 rejects invalid capacities and mask widths", () => {
  assert.throws(() => createSparseCM12TransportProducerMaskLayout({ packetCapacity: 0 }),
    /positive/);
  assert.throws(() => createSparseCM12TransportProducerMaskLayout({
    packetCapacity: 0xffff_ff00,
  }), /base|totalWords/);
  assert.throws(() => packSparseCM12TransportProducerMask([true]), /64 lanes/);
  assert.throws(() => sparseCM12TransportProducerMaskHasLane(0, 0, 64), /\[0, 63\]/);
});
