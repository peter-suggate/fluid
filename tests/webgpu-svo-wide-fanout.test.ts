import assert from "node:assert/strict";
import test from "node:test";

import { SVO_WIDE_MICRO_MIP_WORDS, planSvoWideFanout } from "../lib/svo-wide-fanout";
import {
  SVO_WIDE_GPU_LAYOUT,
  SVO_WIDE_PUBLICATION_STAGES,
  packSvoWideFanout,
  packSvoWideOpacity,
  planWebgpuSvoWideFanoutAllocation,
  unpackSvoWideOpacity,
  validateSvoWideFanoutPublication,
  webgpuSvoWideFanoutHelpersWGSL,
} from "../lib/webgpu-svo-wide-fanout";

function publication() {
  return packSvoWideFanout(planSvoWideFanout({ sourceGeneration: 19, generation: 20, maximumDepth: 4, terminals: [
    { sourceNodeIndex: 4, sourceLeafIndex: 2, level: 4, coordinate: [10, 2, 1], solidOpacity: 0.7, fluidFraction: 0.2 },
    { sourceNodeIndex: 5, level: 3, coordinate: [6, 2, 0], solidOpacity: 1 },
  ] }));
}

test("fixed GPU ABI packs level-major pages, compact descriptors, and 73-word micro-mips", () => {
  const packed = publication();
  assert.deepEqual(SVO_WIDE_GPU_LAYOUT, {
    pageStrideBytes: 32, descriptorStrideBytes: 16, controlStrideBytes: 64,
    microMipStrideBytes: 292, pageWords: 8, descriptorWords: 4, microMipWords: 73,
    descriptorKinds: { terminal: 1, page: 2 },
    controlWords: {
      publishedPages: 0, publishedDescriptors: 1, generation: 2, sourceGeneration: 3,
      overflowFlags: 4, requiredStages: 5, completedStages: 6, payloadWritesComplete: 7,
      maximumDepth: 8, microMipWords: 9,
    },
  });
  assert.equal(packed.pages.length, packed.control[0] * 8);
  assert.equal(packed.descriptors.length, packed.control[1] * 4);
  assert.equal(packed.microMips.length, packed.control[0] * SVO_WIDE_MICRO_MIP_WORDS);
  assert.deepEqual(validateSvoWideFanoutPublication({ ...packed, expectedSourceGeneration: 19 }), {
    status: "ready", generation: 20, sourceGeneration: 19,
    pageCount: packed.control[0], descriptorCount: packed.control[1],
  });
});

test("GPU owner allocation accounts exactly for all four immutable bindings", () => {
  assert.deepEqual(planWebgpuSvoWideFanoutAllocation({ maximumPages: 3, maximumDescriptors: 70 }), {
    maximumPages: 3,
    maximumDescriptors: 70,
    controlBytes: 64,
    pageBytes: 96,
    descriptorBytes: 1_120,
    microMipBytes: 876,
    allocatedBytes: 2_156,
  });
  assert.throws(() => planWebgpuSvoWideFanoutAllocation({ maximumPages: 0, maximumDescriptors: 1 }), /positive safe integers/);
});

test("UNORM opacity packing is round-to-nearest and channel stable", () => {
  const packed = packSvoWideOpacity({ solidMean: 0.1, solidMaximum: 0.7, fluidMean: 0.2, fluidMaximum: 1 });
  const unpacked = unpackSvoWideOpacity(packed);
  assert.ok(Math.abs(unpacked.solidMean - 0.1) <= 1 / 255);
  assert.ok(Math.abs(unpacked.solidMaximum - 0.7) <= 1 / 255);
  assert.ok(Math.abs(unpacked.fluidMean - 0.2) <= 1 / 255);
  assert.equal(unpacked.fluidMaximum, 1);
});

test("generation validation fails closed for stale, partial, overflowed, and malformed publications", () => {
  const stale = publication();
  assert.equal(validateSvoWideFanoutPublication({ ...stale, expectedSourceGeneration: 20 }).status, "source-stale");

  const partial = publication();
  partial.control[SVO_WIDE_GPU_LAYOUT.controlWords.completedStages] &= ~SVO_WIDE_PUBLICATION_STAGES.microMips;
  assert.equal(validateSvoWideFanoutPublication(partial).status, "incomplete");

  const unwritten = publication();
  unwritten.control[SVO_WIDE_GPU_LAYOUT.controlWords.payloadWritesComplete] = 0;
  assert.equal(validateSvoWideFanoutPublication(unwritten).status, "incomplete");

  const overflow = publication();
  overflow.control[SVO_WIDE_GPU_LAYOUT.controlWords.overflowFlags] = 3;
  assert.equal(validateSvoWideFanoutPublication(overflow).status, "overflow");

  const badCount = publication();
  badCount.control[SVO_WIDE_GPU_LAYOUT.controlWords.publishedPages] += 1;
  assert.equal(validateSvoWideFanoutPublication(badCount).status, "invalid");

  const badChild = publication();
  const pageKindDescriptor = Array.from({ length: badChild.control[1] }, (_, index) => index)
    .find((index) => (badChild.descriptors[index * 4] & 3) === SVO_WIDE_GPU_LAYOUT.descriptorKinds.page);
  assert.notEqual(pageKindDescriptor, undefined);
  badChild.descriptors[(pageKindDescriptor as number) * 4 + 1] = 0xffff_ffff;
  assert.equal(validateSvoWideFanoutPublication(badChild).status, "invalid");
});

test("WGSL hierarchy helpers are binding-free and mirror the packed ABI", () => {
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /struct SvoWidePage/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /struct SvoWideDescriptor/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /SVO_WIDE_MICRO_MIP_WORDS: u32 = 73u/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideDescriptorRank/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /countOneBits/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideOpacityChannels/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideMicroMipOffset/);
  assert.doesNotMatch(webgpuSvoWideFanoutHelpersWGSL, /@group|@binding|var<storage/);
});
