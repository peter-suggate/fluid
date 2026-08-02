import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  OCTREE_AIR_SUPPORT_OWNER_HASH,
  octreeAirSupportOwnerHashStartWGSL,
  planOctreeAirVelocitySupport,
} from "../lib/webgpu-octree-air-velocity-support";
import { octreeAirVelocitySupportPublicationWGSL } from "../lib/webgpu-octree-air-velocity-support-gpu";
import { structuredVelocityDynamicsWGSL } from "../lib/webgpu-octree-structured-dynamics";
import { structuredFineLevelSetTransportWGSL } from "../lib/webgpu-octree-fine-levelset-transport";
import { coarseSummaryWGSL } from "../lib/webgpu-octree-coarse-summary";

/**
 * The adaptive owner directory replaced a dense finest-cell map whose lookup
 * was a single indexed load. Three consumers were converted and one --
 * `webgpu-octree-coarse-summary.ts`, the only surface tracker at fine factor
 * one -- kept reading `owners[4 * cell]`, so it resolved almost every coarse
 * cell to an empty hash slot, read the zero key as row zero, and advected the
 * whole dense phi on one row's velocity. The `symmetric-expansion` dam then
 * never reached a wall and broke into nine components.
 *
 * Nothing about that failure was visible to a type or a unit test: producer
 * and consumer agreed on a buffer and disagreed on what its words meant. These
 * tests pin the ABI itself. Any module that reads the directory must probe it,
 * and every probe must be built from the one exported constant set.
 */

const OWNER_DIRECTORY_READERS = [
  ["air-support producer", octreeAirVelocitySupportPublicationWGSL],
  ["structured velocity dynamics", structuredVelocityDynamicsWGSL],
  ["fine level-set transport", structuredFineLevelSetTransportWGSL],
  ["coarse-only summary", coarseSummaryWGSL],
] as const;

test("every owner-directory reader emits the shared probe-start mix", () => {
  const expected = octreeAirSupportOwnerHashStartWGSL("x").slice("fn x(".length);
  for (const [name, source] of OWNER_DIRECTORY_READERS) {
    const starts = [...source.matchAll(/fn\s+(\w*[Oo]wnerHashStart)\s*\(([\s\S]*?)\n/g)];
    assert.equal(starts.length, 1, `${name} must declare exactly one owner hash start`);
    assert.ok(starts[0]![0].includes(expected.split("->")[1]!.trim()),
      `${name} must mix with the exported constants, not its own copy`);
  }
});

test("no owner-directory reader retains a dense finest-cell index", () => {
  // The retired ABI was `ownerDirectoryOffset + 4 * cell`, where `cell` is a
  // linear finest-cell id. The live ABI is `... + 4 * (slot % capacity)`.
  const denseIndex = /[Oo]wnerDirectoryOffset\w*\s*\+\s*4u\s*\*\s*(?:cell|item)\b/;
  for (const [name, source] of OWNER_DIRECTORY_READERS) {
    assert.doesNotMatch(source, denseIndex,
      `${name} must not index the owner directory by finest cell`);
  }
});

test("owner-directory probes are bounded by the shared maximum", () => {
  const bound = new RegExp(`min\\(capacity,${OCTREE_AIR_SUPPORT_OWNER_HASH.maximumProbes}u\\)`);
  for (const [name, source] of OWNER_DIRECTORY_READERS) {
    assert.match(source, bound, `${name} must bound its probe run at the shared maximum`);
  }
});

test("lookups start at or above every authored octree leaf size", () => {
  // `octreeLeafSize` clamps the authored maximum to 32. A probe that starts
  // below the real maximum silently misses every coarser leaf, which is
  // indistinguishable from an empty corridor at the call site.
  assert.equal(OCTREE_AIR_SUPPORT_OWNER_HASH.maximumLeafSize, 32);
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const clamp = octree.match(/function octreeLeafSize\(value: number\)[\s\S]*?\n}/);
  assert.notEqual(clamp, null, "octreeLeafSize must remain the authored-leaf clamp");
  assert.match(clamp![0], /rounded >= 32\) return 32/,
    "the authored leaf ceiling must stay at the owner-hash probe start");
});

test("the accepted control census counts corridor misses without rejecting", () => {
  const census = structuredVelocityDynamicsWGSL.match(/fn countOutOfCorridorRead[\s\S]*?\n}/);
  assert.notEqual(census, null, "the corridor coverage census must exist");
  assert.doesNotMatch(census![0], /atomicOr\(&accepted\[0\]/,
    "a speculative owner probe must not poison the accepted authority header");
  assert.match(census![0], /atomicAdd\(&accepted\[11\],1u\);atomicMin\(&accepted\[12\],cell\)/,
    "the census must still record the count and the first missing finest cell");
  // The bounds fault is a different thing and must stay fail-closed.
  const bounds = structuredVelocityDynamicsWGSL.match(/fn rejectOwnerDirectoryBounds[\s\S]*?\n}/);
  assert.notEqual(bounds, null);
  assert.match(bounds![0], /atomicOr\(&accepted\[0\],ERROR_SAMPLE\)/,
    "a directory that cannot address its arena must still reject the generation");
});

test("the directory's slot capacity is derivable identically on both sides", () => {
  // The producer reads its capacity from `arrayLength(&supportArena)`; the
  // consumers read `ownerDirectorySlotCapacity` from a uniform. They agree
  // only because the terminal allocation is exactly the hash and is already
  // aligned, so the layout must never pad past it.
  for (const [rowCapacity, slotCapacity, alignment] of [
    [4096, 8192, 256], [1024, 2048, 256], [12032, 24064, 256], [64, 128, 16],
  ] as const) {
    const layout = planOctreeAirVelocitySupport(rowCapacity, slotCapacity, alignment, 16_384);
    assert.equal(layout.totalBytes, layout.ownerDirectoryOffsetBytes + layout.ownerDirectoryBytes,
      "the owner hash must be the terminal allocation with no trailing pad");
    assert.equal(
      (layout.totalBytes / 4 - layout.ownerDirectoryOffsetWords)
        / OCTREE_AIR_SUPPORT_OWNER_HASH.recordWords,
      layout.ownerDirectorySlotCapacity,
      "arrayLength-derived capacity must equal the published slot capacity");
    assert.ok(layout.ownerDirectorySlotCapacity >= 2 * (rowCapacity + layout.supportCapacity),
      "the <=0.5 load factor that bounds the probe run must hold");
  }
});
